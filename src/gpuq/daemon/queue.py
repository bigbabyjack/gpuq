"""Priority FIFO over JobRepo."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from gpuq.state import JobRecord, Store


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        for i, j in enumerate(self.store.jobs.list(state="queued"), start=1):
            if j.id == jid:
                return i
        return None

    def cancel(self, jid: str) -> bool:
        rec = self.store.jobs.get(jid)
        if rec is None or rec.state != "queued":
            return False
        self.store.jobs.update_state(jid, state="cancelled", finished_at=_now())
        return True
