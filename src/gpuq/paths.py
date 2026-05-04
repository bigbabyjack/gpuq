"""XDG-aware filesystem paths for gpuq state."""

from __future__ import annotations

import os
from pathlib import Path


def _xdg_state_home() -> Path:
    raw = os.environ.get("XDG_STATE_HOME")
    if raw:
        return Path(raw)
    return Path(os.environ["HOME"]) / ".local" / "state"


def state_dir() -> Path:
    return _xdg_state_home() / "gpuq"


def socket_path() -> Path:
    return state_dir() / "gpuq.sock"


def db_path() -> Path:
    return state_dir() / "state.db"


def log_dir(job_id: str) -> Path:
    return state_dir() / "logs" / job_id


def ensure() -> None:
    (state_dir() / "logs").mkdir(parents=True, exist_ok=True)
