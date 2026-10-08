"""
IRIS adaptive-integrator GUI.

    python iris_gui.py

Runs only the adaptive Leapfrog <-> IAS15 integrator (no reference runs, no correction) on random
systems from validate_adaptive.py's generator or on bodies you place by hand, and plays the result
back in 3D, showing which integrator is in use and which body is driving each switch.
See gui/__init__.py for the module layout.
"""

import multiprocessing
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    multiprocessing.freeze_support()  # needed for frozen Windows builds; harmless otherwise
    from gui.app import main as run_gui

    run_gui()


if __name__ == "__main__":  # worker processes re-import this file; they must not open a window
    main()
