"""Priority FIFO over JobRepo."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from gpuq.state import JobRecord, Store


def _now() -> str:
    return datetime.now(UTC).isoformat()


def pid_alive(pid: int) -> bool:
    """True iff a process with `pid` exists. Uses kill(pid, 0)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def recent_mean_runtime(store: Store, *, tag: str = "", limit: int = 10) -> float | None:
    """Mean runtime in seconds of the last `limit` done jobs (optionally tag-filtered).

    Returns None if fewer than 3 samples are available.
    """
    sql = (
        "SELECT started_at, finished_at FROM jobs "
        "WHERE state='done' AND started_at IS NOT NULL AND finished_at IS NOT NULL"
    )
    args: list = []
    if tag:
        sql += " AND tag=?"
        args.append(tag)
    sql += " ORDER BY finished_at DESC LIMIT ?"
    args.append(limit)
    durations: list[float] = []
    for row in store.jobs.raw_query(sql, args):
        try:
            s = datetime.fromisoformat(row["started_at"])
            f = datetime.fromisoformat(row["finished_at"])
        except ValueError:
            continue
        durations.append((f - s).total_seconds())
    if len(durations) < 3:
        return None
    return sum(durations) / len(durations)


def wedge_signature(
    job: JobRecord,
    *,
    log_dir: Path,
    now: datetime,
    min_age_s: float = 60.0,
    alive: Callable[[int], bool] = pid_alive,
) -> str | None:
    """Classify a running job as wedged. Returns a short signature string or None.

    Signatures:
        "wedged-no-pid"   — state=running, pid IS NULL, no log output, age >= min_age
        "wedged-dead-pid" — state=running, kill(pid,0)==ESRCH, no log output, age >= min_age
    """
    if job.state != "running" or job.started_at is None:
        return None
    try:
        started = datetime.fromisoformat(job.started_at)
    except ValueError:
        return None
    if (now - started).total_seconds() < min_age_s:
        return None
    combined = log_dir / "combined.log"
    if combined.exists() and combined.stat().st_size > 0:
        return None
    if job.pid is None:
        return "wedged-no-pid"
    if not alive(job.pid):
        return "wedged-dead-pid"
    return None


class Queue:
    def __init__(self, store: Store) -> None:
        self.store = store
        self.notify = asyncio.Event()

    def enqueue(self, job: JobRecord) -> None:
        self.store.jobs.insert(job)
        self.notify.set()

    def pop(self) -> JobRecord | None:
        nxt = self.store.jobs.next_queued()
        if nxt is None:
            self.notify.clear()
            return None
        self.store.jobs.update_state(nxt.id, state="starting", started_at=_now())
        return nxt

    def position(self, jid: str) -> int | None:
        for i, j in enumerate(self.store.jobs.find(state="queued"), start=1):
            if j.id == jid:
                return i
        return None

    def cancel(self, jid: str) -> bool:
        rec = self.store.jobs.get(jid)
        if rec is None or rec.state != "queued":
            return False
        self.store.jobs.update_state(jid, state="cancelled", finished_at=_now())
        return True
