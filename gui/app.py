"""
gui/app.py

The Tkinter window.

    left   every validate_adaptive.py flag, grouped; defaults = the latest run (or the last GUI run)
    right  tabs "Setup" (place bodies by hand) and "3D view" (playback), the experiment list,
           the log

Only the adaptive integrator runs: no IAS15 / Leapfrog reference, no correction. The flags that
need a reference are shown but locked off.
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from gui import settings, systems
from gui.controller import RunController
from gui.editor import SetupTab
from gui.settings import FLAGS, GROUPS
from gui.view3d import IAS15, LEAPFROG, RunData, Viewer

POLL_MS = 200
PLAY_MS = 45
SPEEDS = ("1", "2", "5", "10", "25", "50", "100")


class Tooltip:
    def __init__(self, widget, text: str) -> None:
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _e=None) -> None:
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", wraplength=380, background="#ffffe0",
                 relief="solid", borderwidth=1, padx=6, pady=4).pack()

    def _hide(self, _e=None) -> None:
        if self.tip:
            self.tip.destroy()
            self.tip = None


def _text(value) -> str:
    return "" if value is None else repr(value) if isinstance(value, float) else str(value)


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("IRIS adaptive integrator - 3D")
        root.geometry("1560x940")
        root.minsize(1100, 700)

        self.values, source = settings.startup_defaults()
        self.custom_system = self.values.pop("custom_system", None) or systems.default_custom_system()
        self.controller = RunController()
        self.run_dir: str | None = None
        self.meta_paths: dict[int, str] = {}
        self.run: RunData | None = None
        self.current_idx: int | None = None
        self.was_active = False
        self.playing = False
        self._seeking = False
        self.vars: dict[str, tk.Variable] = {}
        self.widgets: dict[str, list] = {}

        self._build(source)
        self.write_form(self.values)
        self.setup_tab.set_system(self.custom_system)
        self.log(f"Settings loaded from {source}.")
        self.log("Defaults marked with a diamond could not be recovered exactly from that run's outputs.")
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.after(POLL_MS, self._poll)

    # ================================================================== layout
    def _build(self, source: str) -> None:
        pw = ttk.PanedWindow(self.root, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw, width=420)
        right = ttk.Frame(pw)
        pw.add(left, weight=0)
        pw.add(right, weight=1)
        self._build_left(left, source)
        self._build_right(right)

    def _build_left(self, left, source: str) -> None:
        bar = ttk.Frame(left)
        bar.pack(fill="x", padx=6, pady=6)
        self.run_btn = ttk.Button(bar, text="\u25B6  Run", command=self.on_run)
        self.run_btn.pack(side="left")
        self.cancel_btn = ttk.Button(bar, text="Cancel", command=self.on_cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=4)
        ttk.Button(bar, text="Open run...", command=self.on_open_run).pack(side="left", padx=(12, 4))
        bar2 = ttk.Frame(left)
        bar2.pack(fill="x", padx=6)
        ttk.Button(bar2, text="Reset to latest run", command=lambda: self.write_form(settings.gui_defaults())
                   ).pack(side="left")
        ttk.Button(bar2, text="Copy command", command=self.on_copy_command).pack(side="left", padx=4)
        ttk.Button(bar2, text="Paste command...", command=self.on_paste_command).pack(side="left")
        ttk.Label(left, text="\u25C6 = could not be recovered exactly. Greyed = needs a reference run.",
                  foreground="#666").pack(anchor="w", padx=8, pady=(4, 2))

        holder = ttk.Frame(left)
        holder.pack(fill="both", expand=True, padx=2, pady=2)
        canvas = tk.Canvas(holder, highlightthickness=0, width=400)
        sb = ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        form = ttk.Frame(canvas)
        win = canvas.create_window((0, 0), window=form, anchor="nw")
        form.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        self._bind_wheel(canvas)
        self._build_form(form)

    def _bind_wheel(self, canvas) -> None:
        def on_wheel(e):
            if e.num == 4:
                canvas.yview_scroll(-2, "units")
            elif e.num == 5:
                canvas.yview_scroll(2, "units")
            else:
                canvas.yview_scroll(int(-e.delta / 120) if abs(e.delta) >= 120 else -e.delta, "units")

        def enter(_e):
            canvas.bind_all("<MouseWheel>", on_wheel)
            canvas.bind_all("<Button-4>", on_wheel)
            canvas.bind_all("<Button-5>", on_wheel)

        def leave(_e):
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                canvas.unbind_all(ev)

        canvas.bind("<Enter>", enter)
        canvas.bind("<Leave>", leave)

    def _build_form(self, form) -> None:
        frames = {}
        for g in GROUPS:
            lf = ttk.LabelFrame(form, text=g)
            lf.pack(fill="x", padx=4, pady=3)
            lf.columnconfigure(1, weight=1)
            frames[g] = lf
        row_of = {g: 0 for g in GROUPS}
        for f in FLAGS:
            lf = frames[f.group]
            r = row_of[f.group]
            row_of[f.group] += 1
            label = f.label + (" \u25C6" if f.inferred else "")
            if f.kind == "bool":
                var = tk.BooleanVar()
                w = ttk.Checkbutton(lf, text=label, variable=var)
                w.grid(row=r, column=0, columnspan=2, sticky="w", padx=6, pady=1)
                parts = [w]
            else:
                var = tk.StringVar()
                lab = ttk.Label(lf, text=label)
                lab.grid(row=r, column=0, sticky="w", padx=6, pady=1)
                if f.kind in ("choice", "intchoice"):
                    w = ttk.Combobox(lf, textvariable=var, values=[str(c) for c in f.choices],
                                     state="readonly", width=12)
                else:
                    w = ttk.Entry(lf, textvariable=var, width=14)
                w.grid(row=r, column=1, sticky="e", padx=6, pady=1)
                parts = [lab, w]
            tip = f.help + (f"\n\nCommand line: {f.cli}" if f.cli else "\n\n(GUI only)")
            if f.reference_only:
                tip += "\n\nLocked: this window runs only the adaptive integrator."
            for p in parts:
                Tooltip(p, tip)
            self.vars[f.dest] = var
            self.widgets[f.dest] = parts
            if f.reference_only:
                for p in parts:
                    p.state(["disabled"])
        for dest in ("index_scale", "rewind", "detect_false_positives", "system_source"):
            self.vars[dest].trace_add("write", lambda *_: self._refresh_enabled())
        self.vars["system_source"].trace_add("write", lambda *_: self._source_changed())

    def _build_right(self, right) -> None:
        top = ttk.Frame(right)
        top.pack(fill="x", padx=6, pady=(6, 0))
        self.progress = ttk.Progressbar(top, mode="determinate", maximum=1.0)
        self.progress.pack(side="left", fill="x", expand=True)
        self.status_lbl = ttk.Label(top, text="Idle", width=34, anchor="e")
        self.status_lbl.pack(side="left", padx=8)

        vpw = ttk.PanedWindow(right, orient="vertical")
        vpw.pack(fill="both", expand=True, padx=4, pady=4)
        self.nb = ttk.Notebook(vpw)
        vpw.add(self.nb, weight=5)

        self.setup_tab = SetupTab(self.nb)
        self.nb.add(self.setup_tab, text="Setup (place bodies)")

        view = ttk.Frame(self.nb)
        self.nb.add(view, text="3D view")
        self.viewer = Viewer(view, on_seek=self._seek)
        self.viewer.widget.pack(fill="both", expand=True)
        self._build_playback(view)

        bottom = ttk.PanedWindow(vpw, orient="horizontal")
        vpw.add(bottom, weight=2)
        tf = ttk.Frame(bottom)
        bottom.add(tf, weight=3)
        cols = ("status", "ias15", "switches", "rewinds", "edrift")
        self.tree = ttk.Treeview(tf, columns=cols, show="tree headings", height=6, selectmode="browse")
        self.tree.heading("#0", text="Experiment")
        self.tree.column("#0", width=210)
        for c, t, w in (("status", "Status", 120), ("ias15", "IAS15 steps", 80), ("switches", "Switches", 70),
                        ("rewinds", "Rewinds", 70), ("edrift", "max |dE/E|", 80)):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="e")
        ts = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ts.set)
        self.tree.pack(side="left", fill="both", expand=True)
        ts.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        lf = ttk.Frame(bottom)
        bottom.add(lf, weight=2)
        self.logbox = tk.Text(lf, height=6, wrap="word", state="disabled", font=("Consolas", 9))
        ls = ttk.Scrollbar(lf, orient="vertical", command=self.logbox.yview)
        self.logbox.configure(yscrollcommand=ls.set)
        self.logbox.pack(side="left", fill="both", expand=True)
        ls.pack(side="right", fill="y")

    def _build_playback(self, view) -> None:
        p1 = ttk.Frame(view)
        p1.pack(fill="x", padx=6, pady=(2, 0))
        self.play_btn = ttk.Button(p1, text="\u25B6", width=4, command=self._toggle_play)
        self.play_btn.pack(side="left")
        ttk.Button(p1, text="\u23EE", width=3, command=lambda: self._seek(0)).pack(side="left", padx=2)
        self.slider = ttk.Scale(p1, from_=0, to=1, command=self._on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=8)
        self.time_lbl = ttk.Label(p1, text="t = -", width=16)
        self.time_lbl.pack(side="left")
        p2 = ttk.Frame(view)
        p2.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(p2, text="Speed (frames/tick)").pack(side="left")
        self.speed_var = tk.StringVar(value="5")
        ttk.Combobox(p2, textvariable=self.speed_var, values=SPEEDS, width=5, state="readonly").pack(side="left", padx=4)
        ttk.Label(p2, text="Trail (frames)").pack(side="left", padx=(12, 0))
        self.trail_var = tk.StringVar(value="150")
        sp = ttk.Spinbox(p2, from_=5, to=5000, increment=25, textvariable=self.trail_var, width=6,
                         command=self._options_changed)
        sp.pack(side="left", padx=4)
        sp.bind("<Return>", lambda _e: self._options_changed())
        self.follow_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(p2, text="Follow bodies", variable=self.follow_var,
                        command=self._options_changed).pack(side="left", padx=10)
        self.labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(p2, text="Labels", variable=self.labels_var, command=self._options_changed).pack(side="left")
        ttk.Button(p2, text="Reset zoom", command=self.viewer.reset_zoom).pack(side="left", padx=10)
        self.details = ttk.Label(p2, text="", foreground="#555")
        self.details.pack(side="right")
        ttk.Label(p2, text="blue = Leapfrog, orange = IAS15", foreground="#555").pack(side="right", padx=12)

    # ================================================================== form <-> values
    def write_form(self, values: dict) -> None:
        for f in FLAGS:
            v = values.get(f.dest, f.latest)
            var = self.vars[f.dest]
            if f.kind == "bool":
                var.set(False if f.reference_only else bool(v))
            else:
                var.set(_text(v))
        self._refresh_enabled()

    def read_form(self) -> tuple[dict, list[str]]:
        vals, errs = {}, []
        for f in FLAGS:
            raw = self.vars[f.dest].get()
            try:
                vals[f.dest] = settings.coerce(f, raw)
            except ValueError as exc:
                errs.append(f"{f.label}: {exc}")
        return vals, errs

    def _peek(self, dest: str):
        try:
            return self.vars[dest].get()
        except tk.TclError:
            return None

    def _refresh_enabled(self) -> None:
        rewind = bool(self._peek("rewind"))
        custom = self._peek("system_source") == "custom"
        minmax = self._peek("index_scale") in ("minmax", "log_minmax")
        rules = {
            "index_lo": minmax, "index_hi": minmax,
            "rewind_back": rewind, "checkpoint_count": rewind, "checkpoint_interval": rewind,
            "rewind_to_anchor": rewind,
            "false_positive_threshold": bool(self._peek("detect_false_positives")),
            "n_experiments": not custom, "seed": not custom, "three_body": not custom,
            "only_experiment": not custom,
        }
        for dest, enabled in rules.items():
            for w in self.widgets[dest]:
                w.state(["!disabled"] if enabled else ["disabled"])

    def _source_changed(self) -> None:
        if self._peek("system_source") == "custom":
            self.nb.select(self.setup_tab)

    # ================================================================== buttons
    def log(self, msg: str) -> None:
        self.logbox.configure(state="normal")
        self.logbox.insert("end", msg + "\n")
        self.logbox.see("end")
        self.logbox.configure(state="disabled")

    def _collect(self) -> dict | None:
        """Validated settings for a run, or None after showing the problems."""
        vals, errs = self.read_form()
        if errs:
            messagebox.showerror("Settings", "\n".join(errs))
            return None
        settings.apply_locks(vals)
        try:
            self.custom_system = self.setup_tab.get_system()
        except ValueError as exc:
            if vals["system_source"] == "custom":
                messagebox.showerror("Custom system", str(exc))
                return None
        vals["custom_system"] = self.custom_system
        errs = settings.validate(vals)
        if vals["system_source"] == "custom":
            errs += systems.validate_custom(self.custom_system)
        if errs:
            messagebox.showerror("Settings", "\n".join(errs))
            return None
        return vals

    def on_run(self) -> None:
        if self.controller.active:
            return
        vals = self._collect()
        if vals is None:
            return
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(settings.RUNS_DIR, stamp + ("_custom" if vals["system_source"] == "custom" else ""))
        os.makedirs(self.run_dir, exist_ok=True)
        settings.save_json(os.path.join(self.run_dir, "settings.json"), vals)
        settings.save_json(settings.LAST_SETTINGS_PATH, vals)
        try:
            paths, names = systems.build_configs(vals, self.run_dir)
        except Exception as exc:  # generator / IO problems
            messagebox.showerror("Could not create the systems", str(exc))
            return
        only = vals["only_experiment"] if vals["system_source"] == "random" else None
        indices = [only] if only is not None else list(range(len(paths)))
        self.meta_paths.clear()
        self.tree.delete(*self.tree.get_children())
        for i in indices:
            self.tree.insert("", "end", iid=str(i), text=f"#{i}  {os.path.splitext(os.path.basename(paths[i]))[0]}",
                             values=("queued", "", "", "", ""))
        self.run, self.current_idx = None, None
        self.controller.start(vals, self.run_dir, paths, names, indices)
        self.was_active = True
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.log(f"Run started: {len(indices)} experiment(s), {vals['workers']} worker(s) -> {self.run_dir}")
        if vals["system_source"] == "random":
            self.log("Equivalent command (adaptive part): " + settings.to_command(vals))
        else:
            self.log(f"Custom system with {len(self.custom_system['bodies'])} bodies "
                     f"(initial conditions: {os.path.join(self.run_dir, 'configs', 'custom.json')}).")

    def on_cancel(self) -> None:
        self.controller.cancel()
        self.log("Cancelled.")
        self._after_run()

    def on_copy_command(self) -> None:
        vals, errs = self.read_form()
        if errs:
            messagebox.showerror("Settings", "\n".join(errs))
            return
        if vals["system_source"] == "custom":
            messagebox.showinfo("Custom system", "A custom system has no command-line equivalent. Its initial "
                                "conditions are saved in <run folder>/configs/custom.json when you run.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(settings.to_command(vals))
        self.log("Command copied to the clipboard (non-default flags only; reference-only flags omitted).")

    def on_paste_command(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Paste a validate_adaptive.py command")
        win.transient(self.root)
        ttk.Label(win, text="Paste the command line; its flags fill the form.\n"
                  "Flags that are not in it return to their command-line defaults.").pack(padx=8, pady=6, anchor="w")
        box = tk.Text(win, width=90, height=6)
        box.pack(padx=8, pady=4)
        try:
            box.insert("1.0", self.root.clipboard_get())
        except tk.TclError:
            pass

        def apply():
            try:
                vals = settings.from_command(box.get("1.0", "end"))
            except ValueError as exc:
                messagebox.showerror("Could not read the command", str(exc), parent=win)
                return
            ignored = settings.apply_locks(vals)
            vals["system_source"] = "random"
            self.write_form(vals)
            self.log("Command applied." + (f" Ignored (need a reference run): {', '.join(ignored)}." if ignored else ""))
            win.destroy()

        ttk.Button(win, text="Apply", command=apply).pack(pady=6)

    def on_open_run(self) -> None:
        start = settings.RUNS_DIR if os.path.isdir(settings.RUNS_DIR) else settings.PROJECT_ROOT
        path = filedialog.askdirectory(initialdir=start, title="Choose a run folder (outputs/gui_runs/...)")
        if not path:
            return
        traj = os.path.join(path, "traj")
        metas = sorted(f for f in os.listdir(traj) if f.endswith(".json")) if os.path.isdir(traj) else []
        if not metas:
            messagebox.showinfo("Open run", "That folder has no finished experiments (no traj/exp_*.json).")
            return
        if self.controller.active:
            messagebox.showinfo("Open run", "Wait for the current run to finish or cancel it first.")
            return
        sp = os.path.join(path, "settings.json")
        if os.path.isfile(sp):
            vals = settings.load_json(sp)
            self.custom_system = vals.pop("custom_system", None) or self.custom_system
            self.write_form(vals)
            self.setup_tab.set_system(self.custom_system)
        self.run_dir = path
        self.meta_paths.clear()
        self.tree.delete(*self.tree.get_children())
        for m in metas:
            mp = os.path.join(traj, m)
            with open(mp, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            self.meta_paths[meta["index"]] = mp
            self.tree.insert("", "end", iid=str(meta["index"]), text="", values=("", "", "", "", ""))
            self._fill_row(meta["index"], meta["simulation_id"], meta["summary"])
        self.log(f"Opened {path} ({len(metas)} experiment(s)); settings restored.")
        first = str(min(self.meta_paths))
        self.tree.selection_set(first)

    # ================================================================== timer
    def _poll(self) -> None:
        try:
            for res in self.controller.poll():
                idx = res["idx"]
                if res.get("ok"):
                    self.meta_paths[idx] = res["meta_path"]
                    self._fill_row(idx, res["simulation_id"], res["summary"])
                    s = res["summary"]
                    self.log(f"#{idx} {res['simulation_id']}: done in {s['wall_s']:.1f}s, IAS15 "
                             f"{100 * s['ias15_fraction']:.1f}% of steps, {s['n_switches']} switches, "
                             f"{s['n_rewinds']} rewinds, max |dE/E| {s['max_energy_drift']:.1e}")
                    if self.run is None:
                        self.tree.selection_set(str(idx))
                else:
                    self.tree.set(str(idx), "status", "failed")
                    self.log(f"#{idx} FAILED: {res.get('error')}")
                    if res.get("trace"):
                        self.log(res["trace"].strip().splitlines()[-1])
            if self.controller.status:
                for idx, st in self.controller.status.items():
                    if st == "running":
                        self.tree.set(str(idx), "status",
                                      f"{self.controller.phase(idx)} {100 * self.controller.progress(idx):.0f}%")
                self.progress["value"] = self.controller.overall_progress()
                n_done = sum(1 for s in self.controller.status.values() if s in ("done", "failed"))
                self.status_lbl.configure(text=f"{n_done}/{len(self.controller.status)} finished")
            if self.was_active and not self.controller.active:
                self._after_run()
                self._finish_run()
        finally:
            self.root.after(POLL_MS, self._poll)

    def _after_run(self) -> None:
        self.was_active = False
        self.run_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")

    def _finish_run(self) -> None:
        done = [r["summary"] for _, r in sorted(self.controller.results.items()) if r.get("ok")]
        failed = sum(1 for r in self.controller.results.values() if not r.get("ok"))
        self.status_lbl.configure(text=f"Finished: {len(done)} ok, {failed} failed")
        if done and self.run_dir:
            path = os.path.join(self.run_dir, "summary.csv")
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(done[0]))
                w.writeheader()
                w.writerows(done)
            share = sum(s["n_ias15_steps"] for s in done) / max(sum(s["n_steps"] for s in done), 1)
            self.log(f"Finished. IAS15 took {100 * share:.1f}% of all steps. Summary: {path}")

    def _fill_row(self, idx: int, sim_id: str, s: dict) -> None:
        edrift = s.get("max_energy_drift")
        self.tree.item(str(idx), text=f"#{idx}  {sim_id}")
        self.tree.item(str(idx), values=("done", f"{100 * s['ias15_fraction']:.1f}%", s["n_switches"],
                                         s["n_rewinds"], "-" if edrift is None else f"{edrift:.1e}"))

    # ================================================================== viewer
    def _on_select(self, _e=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if idx not in self.meta_paths:
            return
        try:
            run = RunData(self.meta_paths[idx])
        except Exception as exc:
            messagebox.showerror("Could not load the experiment", str(exc))
            return
        self.playing = False
        self.play_btn.configure(text="\u25B6")
        self.run, self.current_idx = run, idx
        self.viewer.set_options(trail=int(self.trail_var.get() or 150), follow=self.follow_var.get(),
                                labels=self.labels_var.get())
        self.viewer.set_data(run)
        self.slider.configure(to=max(run.t_count - 1, 1))
        self._seek(0)
        s = run.summary
        mm = s.get("per_body_mismatch")
        self.details.configure(text=(
            f"{run.sim_id}: {s['n_bodies']} bodies, {s['n_steps']} steps, IAS15 {100 * s['ias15_fraction']:.1f}%, "
            f"{s['n_switches']} switches, {s['n_rewinds']} rewinds, {s['n_resyncs']} resyncs, "
            f"max |dE/E| {s['max_energy_drift']:.1e}, max |dL/L| {s['max_momentum_drift']:.1e}"))
        if run.body_score is None:
            self.log("Per-body risk is not available for this index source; the timeline shows the system score.")
        elif mm is not None and mm > 1e-3:
            self.log(f"Note: reconstructed per-body scores differ from the logged score by up to {mm:.1e}.")
        self.nb.select(self.nb.tabs()[1])

    def _seek(self, k: int) -> None:
        if self.run is None:
            return
        k = max(0, min(int(k), self.run.t_count - 1))
        self._seeking = True
        self.slider.set(k)
        self._seeking = False
        self.time_lbl.configure(text=f"t = {self.run.times[k]:.3f}")
        self.viewer.set_frame(k)

    def _on_slider(self, val) -> None:
        if self._seeking or self.run is None:
            return
        k = int(float(val))
        self.time_lbl.configure(text=f"t = {self.run.times[k]:.3f}")
        self.viewer.set_frame(k)

    def _options_changed(self) -> None:
        try:
            trail = int(float(self.trail_var.get()))
        except ValueError:
            trail = 150
        self.viewer.set_options(trail=trail, follow=self.follow_var.get(), labels=self.labels_var.get())

    def _toggle_play(self) -> None:
        if self.run is None:
            return
        self.playing = not self.playing
        self.play_btn.configure(text="\u23F8" if self.playing else "\u25B6")
        if self.playing:
            if self.viewer.frame >= self.run.t_count - 1:
                self._seek(0)
            self._play_tick()

    def _play_tick(self) -> None:
        if not self.playing or self.run is None:
            return
        k = self.viewer.frame + int(self.speed_var.get())
        if k >= self.run.t_count - 1:
            self._seek(self.run.t_count - 1)
            self.playing = False
            self.play_btn.configure(text="\u25B6")
            return
        self._seek(k)
        self.root.after(PLAY_MS, self._play_tick)

    # ================================================================== exit
    def _close(self) -> None:
        if self.controller.active:
            if not messagebox.askyesno("Quit", "A run is in progress. Cancel it and quit?"):
                return
            self.controller.cancel()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()
