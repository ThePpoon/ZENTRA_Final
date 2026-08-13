"""
server/jobs.py — background job runner for ZENTRA (training / upload).

Runs ONE long task at a time as a separate Python subprocess (so it never
blocks the web server or competes with the live pipeline's event loop),
captures its stdout into a ring buffer, and parses coarse progress
(epoch x/y, mAP50) from the ultralytics output.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

# Backend (AI) project dir — jobs run as `python -m training.<mod>` with this as
# cwd, so it must be the package root that actually contains training/.
# Was parent.parent.parent/"ZENTRA" (the old two-repo layout), which resolved to
# the repo root: `python -m training.trainer` there raised ModuleNotFoundError,
# so every Train/Upload button in the app failed.
_ZENTRA_BACKEND = Path(__file__).parent.parent / "backend"

_EPOCH_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s")          # "  12/100  2.1G ..."
_MAP_RE   = re.compile(r"mAP50[^0-9]*([01]\.\d+)", re.I)    # "mAP50: 0.8123"


class JobManager:
    def __init__(self):
        self._lock   = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._lines: deque[str] = deque(maxlen=400)
        self._state  = "idle"          # idle | running | done | error | stopped
        self._label  = ""
        self._started: float = 0.0
        self._epoch  = 0
        self._total  = 0
        self._map50: Optional[float] = None

    # ── Public ────────────────────────────────────────────────
    def is_running(self) -> bool:
        return self._state == "running"

    def start(self, args: list[str], label: str) -> tuple[bool, str]:
        """Start `python -m <args...>` in the backend dir. One job at a time."""
        with self._lock:
            if self._state == "running":
                return False, "มีงานกำลังทำงานอยู่แล้ว"
            if not _ZENTRA_BACKEND.exists():
                return False, f"ไม่พบโฟลเดอร์ backend: {_ZENTRA_BACKEND}"

            self._lines.clear()
            self._state   = "running"
            self._label   = label
            self._started = time.time()
            self._epoch   = 0
            self._total   = 0
            self._map50   = None

            env = dict(os.environ)
            env["PYTHONUTF8"] = "1"
            env["PYTHONUNBUFFERED"] = "1"

            cmd = [sys.executable, "-m", *args]
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=str(_ZENTRA_BACKEND),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except Exception as e:
                self._state = "error"
                self._push(f"[Job] start failed: {e}")
                return False, str(e)

            self._thread = threading.Thread(target=self._reader, daemon=True, name="JobReader")
            self._thread.start()
            self._push(f"[Job] ▶ {label}: {' '.join(cmd)}")
            return True, "started"

    def stop(self) -> bool:
        with self._lock:
            proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
            self._push("[Job] ⏹ ผู้ใช้สั่งหยุด")
            self._state = "stopped"
            return True
        return False

    def status(self) -> dict:
        with self._lock:
            return {
                "state":   self._state,
                "label":   self._label,
                "elapsed": int(time.time() - self._started) if self._started else 0,
                "epoch":   self._epoch,
                "total":   self._total,
                "map50":   self._map50,
                "lines":   list(self._lines)[-40:],
            }

    # ── Private ───────────────────────────────────────────────
    def _push(self, line: str):
        self._lines.append(line.rstrip())

    def _reader(self):
        proc = self._proc
        try:
            for raw in iter(proc.stdout.readline, ""):
                if raw == "" and proc.poll() is not None:
                    break
                line = raw.rstrip()
                if not line:
                    continue
                self._push(line)

                m = _EPOCH_RE.match(line)
                if m:
                    self._epoch, self._total = int(m.group(1)), int(m.group(2))
                mp = _MAP_RE.search(line)
                if mp:
                    try:
                        self._map50 = float(mp.group(1))
                    except ValueError:
                        pass
        except Exception as e:
            self._push(f"[Job] reader error: {e}")
        finally:
            code = proc.wait() if proc else -1
            # Don't overwrite an explicit 'stopped'
            if self._state == "running":
                self._state = "done" if code == 0 else "error"
            self._push(f"[Job] จบการทำงาน (exit {code}) → {self._state}")


# Singleton
manager = JobManager()
