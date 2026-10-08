"""
gui/view3d.py

The 3D viewer (matplotlib, embedded in Tk) and the loader for a finished experiment.

What you see:
    * every body as a dot in its own colour; the ring around it is the integrator that took the
      last step (blue = Leapfrog, orange = IAS15). The integrator switches for the whole system,
      so all rings change together. The body that is driving the switch (highest own risk score)
      gets a larger ring.
    * each body's trail, coloured segment by segment by the integrator in use at the time
    * a timeline under the 3D view: each body's risk score, the switching thresholds, shading
      where IAS15 was active, rewinds as red ticks, and the energy drift on a log axis.
      Click the timeline to jump to that time.
"""

from __future__ import annotations

import json
import os

import numpy as np

BG = "#14161c"
FG = "#c9d1d9"
GRID = (1.0, 1.0, 1.0, 0.10)
LEAPFROG = "#4da3ff"
IAS15 = "#ff9f43"
REWIND = "#ff5c5c"
BODY_COLORS = ["#ff6b6b", "#51cf66", "#cc5de8", "#ffd43b", "#22b8cf", "#f783ac",
               "#a9e34b", "#74c0fc", "#ff922b", "#b197fc", "#63e6be", "#e8590c"]


def body_color(i: int) -> str:
    return BODY_COLORS[i % len(BODY_COLORS)]


def _rgba(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)] + [1.0])


# =========================================================================== data
class RunData:
    """One finished experiment, loaded from the worker's .npz + .json."""

    def __init__(self, meta_path: str) -> None:
        with open(meta_path, "r", encoding="utf-8") as fh:
            self.meta = json.load(fh)
        d = np.load(os.path.splitext(meta_path)[0] + ".npz")
        big = 1e12
        self.times = d["times"]
        self.masses = d["masses"]
        self.pos = np.clip(np.nan_to_num(d["positions"], nan=0.0, posinf=big, neginf=-big), -big, big)
        self.active = d["active"].astype(bool)
        self.score = d["score"]
        self.body_score = d["body_score"] if "body_score" in d.files else None
        self.energy_drift = d["energy_drift"]
        self.switches = d["switches"]
        self.rewinds = d["rewinds"]
        self.resyncs = d["resyncs"]
        self.n = self.pos.shape[1]
        self.t_count = self.pos.shape[0]
        names = self.meta.get("names") or []
        self.names = [names[i] if i < len(names) and names[i] else str(i + 1) for i in range(self.n)]
        self.threshold = float(self.meta["safety_threshold"])
        self.release = self.threshold * float(self.meta["release_fraction"])
        self.summary = self.meta["summary"]
        self.sim_id = self.meta["simulation_id"]

        # which body is driving the switch at each sample (-1 = none above the release level)
        self.driver = np.full(self.t_count, -1, dtype=int)
        if self.body_score is not None:
            with np.errstate(invalid="ignore"):
                bs = np.nan_to_num(self.body_score, nan=-1.0)
                arg = bs.argmax(axis=1)
                top = bs.max(axis=1)
            self.driver = np.where(top >= self.release, arg, -1)

        # robust whole-run view: centre on the median, half-width = 99.5th percentile distance
        flat = self.pos.reshape(-1, 3)
        self.center = np.median(flat, axis=0)
        dev = np.abs(flat - self.center).max(axis=1)
        self.half = float(max(np.percentile(dev, 99.5), 1e-6) * 1.15)

    def frame_at_time(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.times, t), 0, self.t_count - 1))


# =========================================================================== viewer
class Viewer:
    def __init__(self, parent, on_seek=None) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.on_seek = on_seek
        self.fig = Figure(figsize=(7.2, 6.6), dpi=100, facecolor=BG)
        grid = self.fig.add_gridspec(5, 1, hspace=0.32, left=0.07, right=0.93, top=0.985, bottom=0.07)
        self.ax = self.fig.add_subplot(grid[:4, 0], projection="3d")
        self.axt = self.fig.add_subplot(grid[4, 0])
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.widget = self.canvas.get_tk_widget()
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_click)

        self.run: RunData | None = None
        self.trail = 150
        self.follow = False
        self.show_labels = True
        self.zoom = 1.0
        self._follow_c = None
        self._follow_h = None
        self.frame = 0
        self._style_empty()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ styling
    def _style3d(self) -> None:
        ax = self.ax
        ax.set_facecolor(BG)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.09, 0.10, 0.13, 1.0))
            axis.label.set_color(FG)
            try:
                axis._axinfo["grid"]["color"] = GRID
            except Exception:
                pass
        ax.tick_params(colors=FG, labelsize=7)
        ax.set_xlabel("x", labelpad=-6)
        ax.set_ylabel("y", labelpad=-6)
        ax.set_zlabel("z", labelpad=-6)
        ax.set_box_aspect((1, 1, 1))

    def _style_empty(self) -> None:
        self.ax.cla()
        self._style3d()
        self.ax.set_xlim3d(-1, 1)
        self.ax.set_ylim3d(-1, 1)
        self.ax.set_zlim3d(-1, 1)
        self.ax.text2D(0.5, 0.5, "Run a simulation, then pick an experiment\nto watch it here.",
                       transform=self.ax.transAxes, ha="center", va="center", color=FG, fontsize=10)
        self.axt.cla()
        self.axt.set_facecolor(BG)
        self.axt.set_xticks([])
        self.axt.set_yticks([])
        for sp in self.axt.spines.values():
            sp.set_color("#30363d")

    # ------------------------------------------------------------------ loading
    def set_data(self, run: RunData) -> None:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        self.run = run
        self.frame = 0
        self._follow_c = self._follow_h = None
        self.ax.cla()
        self._style3d()
        n = run.n
        dummy = [np.zeros((2, 3))]
        self.trails = []
        for b in range(n):
            col = Line3DCollection(dummy, linewidths=1.8)
            self.ax.add_collection3d(col)
            self.trails.append(col)
        mmax = float(run.masses.max()) if run.masses.size else 1.0
        self.dots = []
        self.labels = []
        for b in range(n):
            size = 6.0 + 7.0 * (float(run.masses[b]) / mmax) ** (1.0 / 3.0)
            (dot,) = self.ax.plot([0], [0], [0], "o", color=body_color(b), ms=size, mec=LEAPFROG, mew=2.2)
            self.dots.append(dot)
            self.labels.append(self.ax.text(0, 0, 0, "  " + run.names[b], color=body_color(b), fontsize=8))
        self._base_ms = [d.get_markersize() for d in self.dots]
        self.info = self.ax.text2D(0.02, 0.98, "", transform=self.ax.transAxes, va="top", fontsize=9,
                                   color=FG, family="monospace")
        self._state_rgba = np.where(run.active[:, None], _rgba(IAS15), _rgba(LEAPFROG))
        self._build_timeline()
        self.set_frame(0)

    def _build_timeline(self) -> None:
        run = self.run
        ax = self.axt
        ax.cla()
        ax.set_facecolor(BG)
        t = run.times
        ax.fill_between(t, 0, 1, where=run.active, step="post", color=IAS15, alpha=0.16, lw=0,
                        transform=ax.get_xaxis_transform())
        if run.body_score is not None:
            for b in range(run.n):
                ax.plot(t, run.body_score[:, b], color=body_color(b), lw=0.9, alpha=0.95)
        else:
            ax.plot(t, run.score, color=FG, lw=1.0)
        ax.axhline(run.threshold, color=IAS15, lw=1.0, ls="--")
        ax.axhline(run.release, color=IAS15, lw=0.8, ls=":")
        if len(run.rewinds):
            ax.vlines(run.rewinds[:, 2], 0, 0.1, transform=ax.get_xaxis_transform(), colors=REWIND, lw=0.8)
        finite = run.score[np.isfinite(run.score)]
        top = max(run.threshold * 1.5, float(finite.max()) * 1.08 if finite.size else 1.0)
        ax.set_ylim(0, min(top, 1.0))
        ax.set_xlim(t[0], t[-1] if t[-1] > t[0] else t[0] + 1)
        ax.set_ylabel("risk score", color=FG, fontsize=7)
        ax.set_xlabel("time", color=FG, fontsize=7, labelpad=1)
        ax.tick_params(colors=FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#30363d")

        self.axe = ax.twinx()
        self.axe.semilogy(t, np.maximum(run.energy_drift, 1e-17), color="#8b949e", lw=0.7, alpha=0.8)
        self.axe.set_ylabel("|dE/E|", color="#8b949e", fontsize=7)
        self.axe.tick_params(colors="#8b949e", labelsize=6)
        for sp in self.axe.spines.values():
            sp.set_color("#30363d")
        self.cursor = ax.axvline(t[0], color="white", lw=1.0, alpha=0.9)

    # ------------------------------------------------------------------ options
    def set_options(self, trail: int | None = None, follow: bool | None = None,
                    labels: bool | None = None) -> None:
        if trail is not None:
            self.trail = max(2, int(trail))
        if follow is not None:
            if follow != self.follow:
                self._follow_c = self._follow_h = None
            self.follow = follow
        if labels is not None:
            self.show_labels = labels
        if self.run is not None:
            self.set_frame(self.frame)

    def reset_zoom(self) -> None:
        self.zoom = 1.0
        if self.run is not None:
            self.set_frame(self.frame)

    # ------------------------------------------------------------------ drawing
    def set_frame(self, k: int) -> None:
        run = self.run
        if run is None:
            return
        k = int(np.clip(k, 0, run.t_count - 1))
        self.frame = k
        i0 = max(0, k - self.trail)
        ias = bool(run.active[k])
        ring = IAS15 if ias else LEAPFROG
        drv = int(run.driver[k])

        window = run.pos[i0:k + 1]
        for b in range(run.n):
            pts = window[:, b, :]
            if len(pts) >= 2:
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                cols = self._state_rgba[i0 + 1:k + 1].copy()
                cols[:, 3] = np.linspace(0.12, 1.0, len(cols))
            else:
                segs = [np.zeros((2, 3))]
                cols = np.zeros((1, 4))
            self.trails[b].set_segments(segs)
            self.trails[b].set_color(cols)
            x, y, z = run.pos[k, b]
            dot = self.dots[b]
            dot.set_data_3d([x], [y], [z])
            dot.set_markeredgecolor(ring)
            big = b == drv
            dot.set_markersize(self._base_ms[b] + (4 if big else 0))
            dot.set_markeredgewidth(3.4 if big else 2.2)
            lab = self.labels[b]
            lab.set_visible(self.show_labels)
            try:
                lab.set_position_3d((x, y, z))
            except AttributeError:
                lab.set_position((x, y))
                lab.set_3d_properties(z)

        self._set_limits(k, i0)
        t = run.times[k]
        sc = run.score[k]
        e = run.energy_drift[k]
        who = run.names[drv] if drv >= 0 else "-"
        self.info.set_text(
            f"t = {t:.3f}\n"
            f"integrator: {'IAS15' if ias else 'Leapfrog'}\n"
            f"score {sc:.3f}  (switch at {run.threshold:.3f})\n"
            f"driver: {who}\n"
            f"|dE/E| {e:.1e}")
        self.info.set_color(IAS15 if ias else LEAPFROG)
        self.cursor.set_xdata([t, t])
        self.canvas.draw_idle()

    def _set_limits(self, k: int, i0: int) -> None:
        run = self.run
        if self.follow:
            w = run.pos[i0:k + 1]
            c = w.reshape(-1, 3).mean(axis=0)
            h = float(max(np.abs(w - c).max() * 1.25, run.half * 0.02, 1e-9))
            if self._follow_c is None:
                self._follow_c, self._follow_h = c, h
            else:  # smooth, so the view does not jitter
                self._follow_c = 0.85 * self._follow_c + 0.15 * c
                self._follow_h = 0.85 * self._follow_h + 0.15 * h
            c, h = self._follow_c, self._follow_h
        else:
            c, h = run.center, run.half
        h = h * self.zoom
        self.ax.set_xlim3d(c[0] - h, c[0] + h)
        self.ax.set_ylim3d(c[1] - h, c[1] + h)
        self.ax.set_zlim3d(c[2] - h, c[2] + h)

    # ------------------------------------------------------------------ mouse
    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.ax or self.run is None:
            return
        self.zoom = float(np.clip(self.zoom * (0.88 if event.step > 0 else 1.14), 0.02, 50.0))
        self.set_frame(self.frame)

    def _on_click(self, event) -> None:
        if self.run is None or event.xdata is None:
            return
        if event.inaxes in (self.axt, getattr(self, "axe", None)) and event.button == 1:
            k = self.run.frame_at_time(event.xdata)
            if self.on_seek:
                self.on_seek(k)
            else:
                self.set_frame(k)
