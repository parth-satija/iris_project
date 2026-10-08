"""
gui/controller.py

Runs the experiments of one GUI run, each in its own child process (spawned, so it works the
same on Windows), at most `workers` at a time. The GUI calls `poll()` from its timer: that
launches queued experiments, collects finished ones, and notices crashed children. `cancel()`
terminates everything (a child cannot be interrupted any other way: run_adaptive has no
cancellation hook).
"""

from __future__ import annotations

import multiprocessing
import queue as queue_mod
import time

from gui import worker

CRASH_GRACE_S = 1.5  # a dead child gets this long for its result to arrive before it counts as crashed


class RunController:
    def __init__(self) -> None:
        self._ctx = multiprocessing.get_context("spawn")
        self.reset()

    def reset(self) -> None:
        self._queue = None
        self._shared = None
        self._procs: dict[int, multiprocessing.Process] = {}
        self._dead_since: dict[int, float] = {}
        self._pending: list[int] = []
        self.status: dict[int, str] = {}  # idx -> queued | running | done | failed | cancelled
        self.results: dict[int, dict] = {}
        self._args = None
        self._workers = 1

    # ------------------------------------------------------------------ control
    def start(self, s: dict, run_dir: str, cfg_paths: list[str], names: list[list[str]],
              indices: list[int]) -> None:
        self.reset()
        self._queue = self._ctx.Queue()
        self._shared = self._ctx.Array("d", 2 * len(cfg_paths), lock=False)
        self._args = (s, run_dir, cfg_paths, names)
        self._workers = max(1, int(s["workers"]))
        self._pending = list(indices)
        self.status = {i: "queued" for i in indices}
        self._launch()

    def cancel(self) -> None:
        for p in self._procs.values():
            if p.is_alive():
                p.terminate()
        for p in self._procs.values():
            p.join(timeout=2.0)
        for i, st in self.status.items():
            if st in ("queued", "running"):
                self.status[i] = "cancelled"
        self._procs.clear()
        self._pending.clear()

    @property
    def active(self) -> bool:
        return bool(self._pending or self._procs)

    # ------------------------------------------------------------------ progress
    def progress(self, idx: int) -> float:
        if self.status.get(idx) == "done":
            return 1.0
        return float(self._shared[2 * idx]) if self._shared is not None else 0.0

    def phase(self, idx: int) -> str:
        if self._shared is None:
            return "queued"
        return worker.PHASE_NAMES.get(int(self._shared[2 * idx + 1]), "?")

    def overall_progress(self) -> float:
        if not self.status:
            return 0.0
        return sum(self.progress(i) for i in self.status) / len(self.status)

    # ------------------------------------------------------------------ the timer hook
    def poll(self) -> list[dict]:
        """Launch queued work, collect results. Returns the results that arrived since last call."""
        new: list[dict] = []
        if self._queue is None:
            return new
        while True:
            try:
                res = self._queue.get_nowait()
            except queue_mod.Empty:
                break
            idx = res["idx"]
            self.status[idx] = "done" if res.get("ok") else "failed"
            self.results[idx] = res
            new.append(res)
            proc = self._procs.pop(idx, None)
            if proc is not None:
                proc.join(timeout=2.0)
            self._dead_since.pop(idx, None)

        now = time.monotonic()
        for idx, proc in list(self._procs.items()):
            if proc.is_alive():
                continue
            first = self._dead_since.setdefault(idx, now)
            if now - first > CRASH_GRACE_S and self.status.get(idx) == "running":
                self.status[idx] = "failed"
                res = {"idx": idx, "ok": False, "error": f"worker process exited with code {proc.exitcode}"}
                self.results[idx] = res
                new.append(res)
                del self._procs[idx]
        self._launch()
        return new

    # ------------------------------------------------------------------ internals
    def _launch(self) -> None:
        s, run_dir, cfg_paths, names = self._args
        while self._pending and len(self._procs) < self._workers:
            idx = self._pending.pop(0)
            proc = self._ctx.Process(
                target=worker.run_experiment,
                args=(cfg_paths[idx], run_dir, s, idx, names[idx], self._shared, self._queue),
                daemon=True,
            )
            proc.start()
            self._procs[idx] = proc
            self.status[idx] = "running"
