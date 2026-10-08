# IRIS adaptive-integrator GUI

```
python iris_gui.py
```

Runs **only** the adaptive Leapfrog <-> IAS15 integrator (`core.adaptive.run_adaptive`). There is no
IAS15-only or Leapfrog-only reference run and no correction, so accuracy is judged by the drift of
total energy and angular momentum (shown on the timeline).

## Systems

* **Random** (default): the same systems `validate_adaptive.py` generates (`--seed`, `--n-experiments`,
  `--three-body`). Experiments run in parallel (`--workers`); watch any finished one while the rest run.
* **Custom**: choose *System source = custom*, then place bodies in the **Setup** tab. Edit mass,
  position and velocity in the table, or use the two 2D views (top x-y, side x-z):
  * **Add / move:** click empty space to add a body; drag a body to move it.
  * **Velocity:** drag the ◆ handle at the tip of a body's arrow to reshape it; Shift+drag from a
    body (or the *Set velocity* mode) pulls out a new arrow. The side view keeps the y component.
  * **Delete:** right-click a body, use *Delete bodies* mode, or select a body (click it) and press
    *Delete selected* / the Delete key. The ✕ in the table row works too.
  Presets and "random from seed" give a starting point; "Centre of mass" zeroes the COM position
  and total momentum.

## What the 3D view shows

* Ring around each body, and the trail colour: **blue = Leapfrog, orange = IAS15**. The integrator
  switches for the *whole system* (the system score is the worst body's), so all bodies change
  together. The larger ring marks the **driver**: the body whose own risk score is highest while
  the system is at or above the release level.
* Timeline: each body's risk score, the switch (dashed) and release (dotted) thresholds, orange
  shading where IAS15 was active, red ticks at rewinds, and |dE/E| on a log axis. Click it to jump.
* Mouse: drag to rotate, scroll to zoom. *Follow bodies* keeps the view on the moving region.

Per-body scores are rebuilt after the run from the sampled states (jerk from a backward Taylor
step) and checked against the score the integrator logged; the worst mismatch is stored in the
summary. They are not available with `--situation-index` (stateful index).

## Settings

* Every `validate_adaptive.py` flag is in the left panel. Defaults are the latest run's settings,
  reconstructed from `outputs/validation/` (flags were not recorded); **◆** marks the ones that
  could not be recovered exactly. After the first GUI run the defaults are the previous GUI run's.
* Flags that need a reference run (`--correct`, `--analyze-rollback` and its sub-options, `--plot`,
  `--ias15-stepped`) are shown but locked off.
* *Copy command* gives the equivalent `validate_adaptive.py` command (non-default flags only);
  *Paste command...* fills the form from one.
* Each run is saved to `outputs/gui_runs/<timestamp>/` (`settings.json`, `configs/`, `traj/`,
  `summary.csv`); nothing is written to `outputs/validation/`. *Open run...* reloads one.
