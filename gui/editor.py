"""
gui/editor.py

The Setup tab: place bodies by hand and configure them.

    * a table with one row per body: name, mass, position, velocity (type the numbers), plus G and
      the softening length
    * presets (figure-eight, Pythagorean problem, ...) and "random system from a seed", so you can
      start from something and tweak it
    * an interactive preview with a top view (x, y) and a side view (x, z):
          Move / add mode : click empty space to add a body, drag a body to move it
          Velocity mode   : drag from a body to set its velocity (the arrow is the velocity)
    * "Centre of mass" shifts the system to the COM frame with zero total momentum

The simulated system is whatever is in here when you press Run with System source = custom.
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk

import numpy as np

from gui import systems
from gui.view3d import BG, FG, body_color

PICK_PIXELS = 14
FIELDS = ("mass", "x", "y", "z", "vx", "vy", "vz")


def _fmt(x: float) -> str:
    return f"{x:.6g}"


class SetupTab(ttk.Frame):
    def __init__(self, parent, on_change=None) -> None:
        super().__init__(parent)
        self.on_change = on_change
        self.rows: list[dict] = []
        self.g_var = tk.StringVar(value="1.0")
        self.soft_var = tk.StringVar(value="0.0")
        self.new_mass_var = tk.StringVar(value="1.0")
        self.mode_var = tk.StringVar(value="move")
        self.auto_arrow_var = tk.BooleanVar(value=True)
        self.arrow_var = tk.StringVar(value="1.0")
        self.preset_var = tk.StringVar(value=list(systems.PRESETS)[0])
        self.seed_var = tk.StringVar(value="956")
        self.count_var = tk.StringVar(value="3")
        self._drag: dict | None = None
        self.selected: int | None = None
        self._lim = (0.0, 0.0, 0.0, 2.0)  # cx, cy, cz, half
        self._build()
        self.set_system(systems.default_custom_system())

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Label(top, text="Preset").pack(side="left")
        ttk.Combobox(top, textvariable=self.preset_var, values=list(systems.PRESETS), width=28,
                     state="readonly").pack(side="left", padx=4)
        ttk.Button(top, text="Load", command=self._load_preset).pack(side="left")
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(top, text="Random: seed").pack(side="left")
        ttk.Entry(top, textvariable=self.seed_var, width=7).pack(side="left", padx=3)
        ttk.Label(top, text="bodies").pack(side="left")
        ttk.Combobox(top, textvariable=self.count_var, values=("2", "3", "4", "5"), width=3,
                     state="readonly").pack(side="left", padx=3)
        ttk.Button(top, text="Load", command=self._load_random).pack(side="left")

        mid = ttk.Frame(self)
        mid.pack(fill="x", padx=6, pady=2)
        ttk.Label(mid, text="G").pack(side="left")
        g = ttk.Entry(mid, textvariable=self.g_var, width=8)
        g.pack(side="left", padx=(3, 10))
        ttk.Label(mid, text="Softening").pack(side="left")
        s = ttk.Entry(mid, textvariable=self.soft_var, width=8)
        s.pack(side="left", padx=(3, 10))
        for e in (g, s):
            e.bind("<Return>", self._commit)
            e.bind("<FocusOut>", self._commit)
        ttk.Button(mid, text="Add body", command=self.add_body).pack(side="left", padx=3)
        ttk.Label(mid, text="mass").pack(side="left")
        ttk.Entry(mid, textvariable=self.new_mass_var, width=6).pack(side="left", padx=3)
        ttk.Button(mid, text="Centre of mass / zero momentum", command=self._center).pack(side="left", padx=10)

        # table of bodies (scrollable)
        holder = ttk.Frame(self)
        holder.pack(fill="x", padx=6, pady=2)
        self._canvas = tk.Canvas(holder, height=190, highlightthickness=0)
        sb = ttk.Scrollbar(holder, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=sb.set)
        self._canvas.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")
        self.table = ttk.Frame(self._canvas)
        self._canvas.create_window((0, 0), window=self.table, anchor="nw")
        self.table.bind("<Configure>", lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))

        # preview controls
        pc = ttk.Frame(self)
        pc.pack(fill="x", padx=6, pady=(4, 0))
        ttk.Radiobutton(pc, text="Move / add bodies", variable=self.mode_var, value="move").pack(side="left")
        ttk.Radiobutton(pc, text="Set velocity", variable=self.mode_var, value="velocity",
                        command=self._commit).pack(side="left", padx=8)
        ttk.Radiobutton(pc, text="Delete bodies", variable=self.mode_var, value="delete").pack(side="left")
        ttk.Button(pc, text="Delete selected", command=self.delete_selected).pack(side="left", padx=8)
        ttk.Checkbutton(pc, text="auto arrow length", variable=self.auto_arrow_var,
                        command=self._commit).pack(side="left", padx=(16, 4))
        ae = ttk.Entry(pc, textvariable=self.arrow_var, width=7)
        ae.pack(side="left")
        ae.bind("<Return>", self._commit)
        ttk.Label(pc, text="plot units per unit speed").pack(side="left", padx=4)
        self.hint = ttk.Label(pc, text="", foreground="#8b949e")
        self.hint.pack(side="right")

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.fig = Figure(figsize=(7.4, 3.3), dpi=100, facecolor=BG)
        self.ax_xy = self.fig.add_subplot(1, 2, 1)
        self.ax_xz = self.fig.add_subplot(1, 2, 2)
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.12, wspace=0.18)
        self.mpl = FigureCanvasTkAgg(self.fig, master=self)
        self.mpl.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=4)
        self.mpl.mpl_connect("button_press_event", self._on_press)
        self.mpl.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl.mpl_connect("button_release_event", self._on_release)
        self.mpl.mpl_connect("key_press_event", self._on_key)
        self._update_hint()
        self.mode_var.trace_add("write", lambda *_: self._update_hint())

    def _update_hint(self) -> None:
        self.hint.config(text={
            "move": "click empty = add, drag body = move, drag a \u25C6 arrow tip = velocity, right-click = delete",
            "velocity": "drag from a body to set its velocity (right-click = delete)",
            "delete": "click a body to delete it"}[self.mode_var.get()])

    # ------------------------------------------------------------------ table
    def _rebuild_rows(self, bodies: list[dict]) -> None:
        for w in self.table.winfo_children():
            w.destroy()
        self.rows = []
        heads = ("#", "Name", "Mass", "x", "y", "z", "vx", "vy", "vz", "")
        for c, h in enumerate(heads):
            ttk.Label(self.table, text=h).grid(row=0, column=c, padx=2)
        for i, b in enumerate(bodies):
            r = i + 1
            tk.Label(self.table, text=str(i + 1), fg=body_color(i), width=3,
                     font=("TkDefaultFont", 10, "bold")).grid(row=r, column=0)
            vars_ = {"name": tk.StringVar(value=b["name"]), "mass": tk.StringVar(value=_fmt(b["mass"]))}
            for k, key in enumerate(("x", "y", "z")):
                vars_[key] = tk.StringVar(value=_fmt(b["pos"][k]))
            for k, key in enumerate(("vx", "vy", "vz")):
                vars_[key] = tk.StringVar(value=_fmt(b["vel"][k]))
            for c, key in enumerate(("name",) + FIELDS, start=1):
                e = ttk.Entry(self.table, textvariable=vars_[key], width=12 if key == "name" else 10)
                e.grid(row=r, column=c, padx=1, pady=1)
                e.bind("<Return>", self._commit)
                e.bind("<FocusOut>", self._commit)
            ttk.Button(self.table, text="\u2715", width=3,
                       command=lambda idx=i: self.remove_body(idx)).grid(row=r, column=9, padx=2)
            self.rows.append(vars_)

    def add_body(self, pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0)) -> None:
        if len(self.rows) >= systems.MAX_BODIES:
            self.hint.config(text=f"at most {systems.MAX_BODIES} bodies")
            return
        try:
            mass = float(self.new_mass_var.get())
        except ValueError:
            mass = systems.DEFAULT_NEW_MASS
        bodies = self._bodies_loose() + [systems.body(f"B{len(self.rows) + 1}", mass, pos, vel)]
        self._rebuild_rows(bodies)
        self.selected = len(self.rows) - 1
        self.refresh_preview(fit=True)
        self._changed()

    def remove_body(self, idx: int) -> None:
        bodies = self._bodies_loose()
        del bodies[idx]
        self._rebuild_rows(bodies)
        self.selected = None
        self.refresh_preview(fit=True)
        self._changed()

    def delete_selected(self) -> None:
        if self.selected is not None and self.selected < len(self.rows):
            self.remove_body(self.selected)
        else:
            self.hint.config(text="select a body first (click it)")

    def _bodies_loose(self) -> list[dict]:
        """Current rows as body dicts; unparsable numbers become 0 (used when rebuilding)."""

        def num(var, default=0.0):
            try:
                return float(var.get())
            except ValueError:
                return default

        return [systems.body(r["name"].get(), num(r["mass"], 1.0), [num(r[k]) for k in ("x", "y", "z")],
                             [num(r[k]) for k in ("vx", "vy", "vz")]) for r in self.rows]

    # ------------------------------------------------------------------ model
    def get_system(self) -> dict:
        """The system in the table. Raises ValueError (with the row) if a number can't be read."""

        def num(text, what):
            try:
                v = float(text)
            except ValueError:
                raise ValueError(f"{what}: '{text}' is not a number") from None
            return v

        bodies = []
        for i, r in enumerate(self.rows):
            name = r["name"].get().strip() or str(i + 1)
            bodies.append(systems.body(
                name, num(r["mass"].get(), f"Body {i + 1} mass"),
                [num(r[k].get(), f"Body {i + 1} {k}") for k in ("x", "y", "z")],
                [num(r[k].get(), f"Body {i + 1} {k}") for k in ("vx", "vy", "vz")]))
        return systems.system(bodies, num(self.g_var.get(), "G"), num(self.soft_var.get(), "Softening"))

    def set_system(self, cs: dict) -> None:
        self.selected = None
        self.g_var.set(_fmt(cs["g"]))
        self.soft_var.set(_fmt(cs["softening"]))
        self._rebuild_rows(cs["bodies"])
        self.refresh_preview(fit=True)
        self._changed()

    def _changed(self) -> None:
        if self.on_change:
            self.on_change()

    def _commit(self, _event=None) -> None:
        self.refresh_preview(fit=True)
        self._changed()

    def _load_preset(self) -> None:
        self.set_system(systems.PRESETS[self.preset_var.get()]())

    def _load_random(self) -> None:
        try:
            self.set_system(systems.random_system(int(self.seed_var.get()), int(self.count_var.get())))
        except ValueError as exc:
            self.hint.config(text=str(exc))

    def _center(self) -> None:
        try:
            self.set_system(systems.center_system(self.get_system()))
        except ValueError as exc:
            self.hint.config(text=str(exc))

    # ------------------------------------------------------------------ preview
    def _valid(self) -> list[tuple[int, float, np.ndarray, np.ndarray]]:
        out = []
        for i, r in enumerate(self.rows):
            try:
                m = float(r["mass"].get())
                p = np.array([float(r[k].get()) for k in ("x", "y", "z")])
                v = np.array([float(r[k].get()) for k in ("vx", "vy", "vz")])
            except ValueError:
                continue
            if np.all(np.isfinite(p)) and np.all(np.isfinite(v)):
                out.append((i, m, p, v))
        return out

    def _arrow_scale(self, bodies) -> float:
        if not self.auto_arrow_var.get():
            try:
                return max(float(self.arrow_var.get()), 1e-12)
            except ValueError:
                return 1.0
        vmax = max([float(np.linalg.norm(v)) for _, _, _, v in bodies] + [0.0])
        return 0.2 * (2 * self._lim[3]) / (vmax if vmax > 1e-9 else 1.0)

    def refresh_preview(self, fit: bool = True) -> None:
        bodies = self._valid()
        if fit and bodies:
            pos = np.array([p for _, _, p, _ in bodies])
            c = (pos.max(axis=0) + pos.min(axis=0)) / 2
            span = float((pos.max(axis=0) - pos.min(axis=0)).max())
            self._lim = (c[0], c[1], c[2], max(span * 0.65, 1.0))
        cx, cy, cz, half = self._lim
        scale = self._drag["scale"] if self._drag else self._arrow_scale(bodies)
        mmax = max([m for _, m, _, _ in bodies] + [1e-12])
        for ax, (ia, ib), title, centre in (
                (self.ax_xy, (0, 1), "top view  (x, y)", (cx, cy)),
                (self.ax_xz, (0, 2), "side view  (x, z)", (cx, cz))):
            ax.cla()
            ax.set_facecolor(BG)
            ax.set_title(title, color=FG, fontsize=8, pad=3)
            ax.tick_params(colors=FG, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color("#30363d")
            ax.grid(True, color=(1, 1, 1, 0.08), lw=0.6)
            ax.axhline(0, color=(1, 1, 1, 0.15), lw=0.6)
            ax.axvline(0, color=(1, 1, 1, 0.15), lw=0.6)
            for i, m, p, v in bodies:
                col = body_color(i)
                ax.scatter([p[ia]], [p[ib]], s=40 + 90 * (m / mmax) ** (1 / 3), color=col, zorder=3,
                           edgecolors="white", linewidths=0.6)
                if i == self.selected:
                    ax.scatter([p[ia]], [p[ib]], s=(40 + 90 * (m / mmax) ** (1 / 3)) * 2.6, facecolors="none",
                               edgecolors="#ffd43b", linewidths=1.5, zorder=3)
                ax.annotate(str(i + 1), (p[ia], p[ib]), xytext=(6, 6), textcoords="offset points",
                            color=col, fontsize=8, zorder=4)
                if v[ia] != 0 or v[ib] != 0:
                    tip = (p[ia] + v[ia] * scale, p[ib] + v[ib] * scale)
                    ax.annotate("", xy=tip, xytext=(p[ia], p[ib]),
                                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4), zorder=2)
                    ax.scatter([tip[0]], [tip[1]], marker="D", s=24, color=col, edgecolors="white",
                               linewidths=0.6, zorder=5)  # draggable handle
            ax.set_xlim(centre[0] - half, centre[0] + half)
            ax.set_ylim(centre[1] - half, centre[1] + half)
            ax.set_aspect("equal", adjustable="box")
        self.mpl.draw_idle()

    # ------------------------------------------------------------------ mouse
    def _plane(self, ax):
        return (0, 1) if ax is self.ax_xy else (0, 2) if ax is self.ax_xz else None

    def _hit(self, ax, event, bodies) -> int | None:
        ia, ib = self._plane(ax)
        best, best_d = None, PICK_PIXELS
        for i, _m, p, _v in bodies:
            px, py = ax.transData.transform((p[ia], p[ib]))
            d = math.hypot(px - event.x, py - event.y)
            if d < best_d:
                best, best_d = i, d
        return best

    def _tip_hit(self, ax, event, bodies, scale) -> int | None:
        """The body whose arrow-tip handle is under the mouse (only bodies that are moving in this view)."""
        ia, ib = self._plane(ax)
        for i, _m, p, v in bodies:
            if v[ia] == 0 and v[ib] == 0:
                continue
            px, py = ax.transData.transform((p[ia] + v[ia] * scale, p[ib] + v[ib] * scale))
            if math.hypot(px - event.x, py - event.y) < PICK_PIXELS:
                return i
        return None

    def _on_key(self, event) -> None:
        if event.key in ("delete", "backspace"):
            self.delete_selected()

    def _on_press(self, event) -> None:
        ax = event.inaxes
        if ax is None or self._plane(ax) is None or event.xdata is None:
            return
        bodies = self._valid()
        scale = self._arrow_scale(bodies)
        hit = self._hit(ax, event, bodies)
        mode = self.mode_var.get()
        if event.button == 3:  # right-click deletes whatever body is under the mouse
            if hit is not None:
                self.remove_body(hit)
            return
        if event.button != 1:
            return
        tip = self._tip_hit(ax, event, bodies, scale)
        if tip is not None and tip != hit:  # grabbed an arrow handle: reshape that body's velocity
            self.selected = tip
            self._drag = {"i": tip, "ax": ax, "mode": "velocity", "scale": scale}
            self.refresh_preview(fit=False)
            return
        if hit is None:
            if mode == "move":
                ia, ib = self._plane(ax)
                pos = [0.0, 0.0, 0.0]
                pos[ia], pos[ib] = float(event.xdata), float(event.ydata)
                self.add_body(pos=pos)
            elif self.selected is not None:
                self.selected = None
                self.refresh_preview(fit=False)
            return
        if mode == "delete":
            self.remove_body(hit)
            return
        self.selected = hit
        pull_arrow = mode == "velocity" or event.key == "shift"  # shift+drag also pulls out an arrow
        self._drag = {"i": hit, "ax": ax, "mode": "velocity" if pull_arrow else "move", "scale": scale}
        self.refresh_preview(fit=False)

    def _on_motion(self, event) -> None:
        d = self._drag
        if d is None or event.inaxes is not d["ax"] or event.xdata is None:
            return
        ia, ib = self._plane(d["ax"])
        keys_p = ("x", "y", "z")
        keys_v = ("vx", "vy", "vz")
        row = self.rows[d["i"]]
        if d["mode"] == "move":
            row[keys_p[ia]].set(_fmt(event.xdata))
            row[keys_p[ib]].set(_fmt(event.ydata))
        else:
            try:
                pa, pb = float(row[keys_p[ia]].get()), float(row[keys_p[ib]].get())
            except ValueError:
                return
            row[keys_v[ia]].set(_fmt((event.xdata - pa) / d["scale"]))
            row[keys_v[ib]].set(_fmt((event.ydata - pb) / d["scale"]))
        self.refresh_preview(fit=False)

    def _on_release(self, _event) -> None:
        if self._drag is None:
            return
        self._drag = None
        self.refresh_preview(fit=True)
        self._changed()
