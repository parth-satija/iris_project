"""
gui/

Desktop GUI for the IRIS adaptive-integrator validation experiment
(validate_adaptive.py). Launch it with:

    python iris_gui.py

Modules:
    settings    every validate_adaptive.py flag (defaults = the latest run), validation,
                command-line <-> settings conversion, persistence
    worker      one experiment in one child process (calls validation.runner unchanged)
    controller  schedules the child processes, progress, cancel
    view3d      the matplotlib 3D viewer + timeline
    app         the Tkinter window
"""
