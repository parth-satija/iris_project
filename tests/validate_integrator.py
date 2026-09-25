"""
tests/validate_integrator.py

Standalone validation script (no pytest dependency) that checks basic
physical correctness properties of the integrators before they are
trusted for calibration experiments. Run directly:

    python tests/validate_integrator.py

Prints PASS/FAIL for each test and exits with a non-zero status code if
any test fails.

Tests:
    1. IAS15 vs IAS15   -- running the same system through run_ias15 twice
                            should produce (approximately) zero error.
    2. Earth-Sun orbit  -- Leapfrog with a tiny timestep should conserve
                            energy well over one full orbit.
    3. Momentum conservation -- total linear momentum should stay constant
                                 (to numerical precision) under Leapfrog.
    4. Angular momentum conservation -- total angular momentum should stay
                                         constant (to numerical precision)
                                         under Leapfrog.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Allow running this script directly (python tests/validate_integrator.py)
# by ensuring the project root is on sys.path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.leapfrog import run_leapfrog  # noqa: E402
from core.physics import (  # noqa: E402
    Body,
    System,
    compute_angular_momentum,
    compute_total_energy,
    compute_total_momentum,
)
from core.rebound_reference import run_ias15  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"


def _print_result(test_name: str, passed: bool, detail: str) -> None:
    status = PASS if passed else FAIL
    print(f"[{status}] {test_name}: {detail}")


def make_earth_sun_system() -> System:
    """
    Build a simple two-body Earth-Sun-like system in simulation units
    where G = 1, chosen so that a circular orbit at radius 1 has period
    2*pi (i.e. v_circ = sqrt(G * M / r) with M = 1, r = 1 => v = 1).

    Returns:
        A System with a heavy central "sun" and a light orbiting "earth"
        on a circular orbit.
    """
    sun = Body(mass=1.0, position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], name="sun")
    earth = Body(
        mass=1.0e-3, position=[1.0, 0.0, 0.0], velocity=[0.0, 1.0, 0.0], name="earth"
    )
    return System(bodies=[sun, earth], g=1.0, softening=0.0)


def test_ias15_vs_ias15() -> bool:
    """
    Test 1: Run the same system through IAS15 twice; since IAS15 is
    deterministic, the resulting trajectories should be identical (error
    approximately zero, allowing for tiny floating point noise).

    Returns:
        True if the test passes, False otherwise.
    """
    system = make_earth_sun_system()
    traj_a = run_ias15(system, total_time=2.0, sample_interval=0.1)
    traj_b = run_ias15(system, total_time=2.0, sample_interval=0.1)

    pos_diff = np.max(np.abs(traj_a.positions - traj_b.positions))
    vel_diff = np.max(np.abs(traj_a.velocities - traj_b.velocities))

    passed = pos_diff < 1e-10 and vel_diff < 1e-10
    _print_result(
        "IAS15 vs IAS15",
        passed,
        f"max_pos_diff={pos_diff:.3e}, max_vel_diff={vel_diff:.3e}",
    )
    return passed


def test_leapfrog_energy_conservation() -> bool:
    """
    Test 2: Earth-Sun orbit under Leapfrog with a tiny timestep should
    conserve total energy well (small relative drift) over one full orbit
    (2*pi in these units).

    Returns:
        True if the test passes, False otherwise.
    """
    system = make_earth_sun_system()
    masses = system.masses()

    initial_energy = compute_total_energy(
        system.positions(), system.velocities(), masses, g=system.g, softening=system.softening
    )

    dt = 1.0e-4
    # Snap total_time to an exact integer number of steps of dt so that
    # sample_interval (also snapped) divides it evenly.
    n_total_steps = int(round((2.0 * np.pi) / dt))
    total_time = n_total_steps * dt
    steps_per_sample = max(1, n_total_steps // 100)
    sample_interval = steps_per_sample * dt
    traj = run_leapfrog(system, dt=dt, total_time=total_time, sample_interval=sample_interval)

    final_energy = compute_total_energy(
        traj.positions[-1], traj.velocities[-1], masses, g=system.g, softening=system.softening
    )

    relative_drift = abs((final_energy - initial_energy) / abs(initial_energy))
    passed = relative_drift < 1e-4
    _print_result(
        "Leapfrog energy conservation (Earth-Sun)",
        passed,
        f"relative_energy_drift={relative_drift:.3e}",
    )
    return passed


def test_momentum_conservation() -> bool:
    """
    Test 3: Total linear momentum of an isolated system should remain
    constant (to numerical precision) throughout a Leapfrog integration,
    since gravity is an internal force (Newton's third law).

    Returns:
        True if the test passes, False otherwise.
    """
    system = make_earth_sun_system()
    masses = system.masses()

    initial_momentum = compute_total_momentum(system.velocities(), masses)

    dt = 1.0e-3
    n_total_steps = int(round((2.0 * np.pi) / dt))
    total_time = n_total_steps * dt
    steps_per_sample = max(1, n_total_steps // 50)
    sample_interval = steps_per_sample * dt
    traj = run_leapfrog(system, dt=dt, total_time=total_time, sample_interval=sample_interval)

    max_drift = 0.0
    for idx in range(traj.velocities.shape[0]):
        p = compute_total_momentum(traj.velocities[idx], masses)
        drift = float(np.linalg.norm(p - initial_momentum))
        max_drift = max(max_drift, drift)

    passed = max_drift < 1e-8
    _print_result("Momentum conservation (Leapfrog)", passed, f"max_drift={max_drift:.3e}")
    return passed


def test_angular_momentum_conservation() -> bool:
    """
    Test 4: Total angular momentum of an isolated system should remain
    constant (to numerical precision) throughout a Leapfrog integration,
    since gravity is a central force.

    Returns:
        True if the test passes, False otherwise.
    """
    system = make_earth_sun_system()
    masses = system.masses()

    initial_l = compute_angular_momentum(system.positions(), system.velocities(), masses)

    dt = 1.0e-3
    n_total_steps = int(round((2.0 * np.pi) / dt))
    total_time = n_total_steps * dt
    steps_per_sample = max(1, n_total_steps // 50)
    sample_interval = steps_per_sample * dt
    traj = run_leapfrog(system, dt=dt, total_time=total_time, sample_interval=sample_interval)

    max_relative_drift = 0.0
    initial_l_mag = float(np.linalg.norm(initial_l))
    for idx in range(traj.positions.shape[0]):
        l_vec = compute_angular_momentum(traj.positions[idx], traj.velocities[idx], masses)
        drift = abs(float(np.linalg.norm(l_vec)) - initial_l_mag) / initial_l_mag
        max_relative_drift = max(max_relative_drift, drift)

    passed = max_relative_drift < 1e-6
    _print_result(
        "Angular momentum conservation (Leapfrog)",
        passed,
        f"max_relative_drift={max_relative_drift:.3e}",
    )
    return passed


def run_all_tests() -> bool:
    """
    Run every validation test in sequence and print a final summary.

    Returns:
        True if all tests passed, False if any failed.
    """
    print("Running IRIS calibration integrator validation tests...\n")

    results = [
        test_ias15_vs_ias15(),
        test_leapfrog_energy_conservation(),
        test_momentum_conservation(),
        test_angular_momentum_conservation(),
    ]

    n_passed = sum(results)
    n_total = len(results)
    print(f"\n{n_passed}/{n_total} tests passed.")
    return n_passed == n_total


if __name__ == "__main__":
    all_passed = run_all_tests()
    sys.exit(0 if all_passed else 1)
