"""
core/jit_kernels.py

Optional compiled (Numba) kernels for the from-scratch Leapfrog code.

WHY THIS EXISTS
---------------
IAS15 (REBOUND) is compiled C: one `sim.integrate(...)` call per dt costs a few
microseconds. The Leapfrog in core/leapfrog.py used to be interpreted NumPy:
every step built several tiny temporary arrays plus an (N, N, 3) broadcast for
the forces, ~40 microseconds of pure interpreter / allocation overhead per step.
That made "Leapfrog-only" look ~3x SLOWER than "IAS15-only stepped every dt",
which cannot be true of the algorithms themselves (a Leapfrog step is ONE force
evaluation; an IAS15 step is 7 substeps x several predictor-corrector
iterations, i.e. tens of force evaluations). The inversion was a cost-model
artifact of comparing Python against C, not a counting bug.

These kernels compile the exact same arithmetic, in the exact same operation
order as the NumPy code (so results agree to rounding), so the comparison is
between algorithms rather than between interpreters.

If Numba is not installed (or IRIS_NO_NUMBA=1 is set) HAVE_NUMBA is False and
every caller falls back to the original NumPy code, unchanged.
"""

from __future__ import annotations

import math
import os

import numpy as np

HAVE_NUMBA = False
if os.environ.get("IRIS_NO_NUMBA", "") != "1":
    try:
        from numba import njit

        HAVE_NUMBA = True
    except Exception:  # noqa: BLE001 - numba missing or broken: fall back to NumPy
        HAVE_NUMBA = False

if HAVE_NUMBA:

    @njit(cache=True)
    def acc_kernel(pos, masses, g, eps2):
        """Accelerations, same operation order as core.physics.compute_accelerations (NumPy)."""
        n = pos.shape[0]
        out = np.empty((n, 3))
        for i in range(n):
            sx = 0.0
            sy = 0.0
            sz = 0.0
            for j in range(n):
                if j == i:
                    continue
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                inv = 1.0 / ((dist * dist + eps2) ** 1.5)
                mj = masses[j]
                sx += mj * dx * inv
                sy += mj * dy * inv
                sz += mj * dz * inv
            out[i, 0] = g * sx
            out[i, 1] = g * sy
            out[i, 2] = g * sz
        return out

    @njit(cache=True)
    def vv_advance_kernel(pos, vel, acc, masses, g, eps2, dt, n_steps):
        """
        Advance n_steps velocity-Verlet steps of length dt. `pos` and `vel` are updated IN PLACE;
        the final acceleration is returned (it is the `acc` the next step needs).
        """
        n = pos.shape[0]
        for _ in range(n_steps):
            for i in range(n):
                for k in range(3):
                    pos[i, k] = (pos[i, k] + vel[i, k] * dt) + ((0.5 * acc[i, k]) * dt) * dt
            new_acc = acc_kernel(pos, masses, g, eps2)
            for i in range(n):
                for k in range(3):
                    vel[i, k] = vel[i, k] + (0.5 * (acc[i, k] + new_acc[i, k])) * dt
            acc = new_acc
        return acc

    @njit(cache=True)
    def vv_step_kernel(pos, vel, acc, cp, cv, masses, g, eps2, h, comp):
        """
        One velocity-Verlet step of length h returning FRESH arrays (pos, vel, acc, cp, cv),
        optionally Kahan-compensated. Mirrors core.adaptive._vv_step exactly.
        """
        n = pos.shape[0]
        new_pos = np.empty((n, 3))
        new_cp = np.empty((n, 3))
        for i in range(n):
            for k in range(3):
                inc = vel[i, k] * h + ((0.5 * acc[i, k]) * h) * h
                if comp:
                    y = inc - cp[i, k]
                    t = pos[i, k] + y
                    new_cp[i, k] = (t - pos[i, k]) - y
                    new_pos[i, k] = t
                else:
                    new_pos[i, k] = pos[i, k] + inc
                    new_cp[i, k] = cp[i, k]
        new_acc = acc_kernel(new_pos, masses, g, eps2)
        new_vel = np.empty((n, 3))
        new_cv = np.empty((n, 3))
        for i in range(n):
            for k in range(3):
                inc = (0.5 * (acc[i, k] + new_acc[i, k])) * h
                if comp:
                    y = inc - cv[i, k]
                    t = vel[i, k] + y
                    new_cv[i, k] = (t - vel[i, k]) - y
                    new_vel[i, k] = t
                else:
                    new_vel[i, k] = vel[i, k] + inc
                    new_cv[i, k] = cv[i, k]
        return new_pos, new_vel, new_acc, new_cp, new_cv


def warm_up() -> None:
    """Compile (and cache to disk) every kernel once, so parallel workers load them instead of
    all compiling at the same moment. No-op without Numba."""
    if not HAVE_NUMBA:
        return
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    vel = np.zeros((3, 3))
    m = np.ones(3)
    acc = acc_kernel(pos, m, 1.0, 0.0)
    vv_advance_kernel(pos.copy(), vel.copy(), acc, m, 1.0, 0.0, 1e-3, 2)
    z = np.zeros((3, 3))
    vv_step_kernel(pos, vel, acc, z, z, m, 1.0, 0.0, 1e-3, True)
    vv_step_kernel(pos, vel, acc, z, z, m, 1.0, 0.0, 1e-3, False)
