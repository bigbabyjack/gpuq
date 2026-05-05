"""Prune RPC: deletes terminal jobs older than cutoff."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from gpuq import paths
from gpuq import protocol as p
from gpuq.state import JobRecord

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("daemon")]


async def _send_recv(req: p.Request) -> list[p.Event]:
    reader, writer = await asyncio.open_unix_connection(str(paths.socket_path()))
    writer.write(p.encode_request(req).encode())
    await writer.drain()
    out: list[p.Event] = []
    while line := await reader.readline():
        out.append(p.decode_event(line.decode()))
    writer.close()
    await writer.wait_closed()
    return out


def _seed(daemon, jid: str, *, state: str, age_seconds: float, tag: str = "") -> None:
    when = (datetime.now(UTC) - timedelta(seconds=age_seconds)).isoformat()
    daemon.store.jobs.insert(
        JobRecord(
            id=jid,
            cmd=["true"],
            cwd="/tmp",
            env={},
            tag=tag,
            priority=0,
            state=state,
            pid=None,
            exit_code=0 if state == "done" else 1,
            submitted_at=when,
            started_at=when,
            finished_at=when if state in ("done", "failed", "cancelled") else None,
        )
    )
    paths.log_dir(jid).mkdir(parents=True, exist_ok=True)


async def test_prune_removes_old_terminal_jobs(daemon):
    _seed(daemon, "j_old", state="done", age_seconds=86400 * 30)
    _seed(daemon, "j_new", state="done", age_seconds=60)
    events = await _send_recv(p.Prune(older_than_s=86400 * 7))
    payload = next(e for e in events if isinstance(e, p.ResultEvent)).payload
    assert "j_old" in payload["deleted"]
    assert "j_new" not in payload["deleted"]
    assert daemon.store.jobs.get("j_old") is None
    assert daemon.store.jobs.get("j_new") is not None
    assert not paths.log_dir("j_old").exists()


async def test_prune_dry_run_does_not_mutate(daemon):
    _seed(daemon, "j_x", state="failed", age_seconds=86400 * 30)
    events = await _send_recv(p.Prune(older_than_s=86400 * 7, dry_run=True))
    payload = next(e for e in events if isinstance(e, p.ResultEvent)).payload
    assert "j_x" in payload["deleted"]
    assert payload["dry_run"] is True
    assert daemon.store.jobs.get("j_x") is not None


async def test_prune_refuses_active_states(daemon):
    _seed(daemon, "j_q", state="queued", age_seconds=86400 * 30)
    _seed(daemon, "j_r", state="running", age_seconds=86400 * 30)
    events = await _send_recv(p.Prune(older_than_s=86400 * 7))
    payload = next(e for e in events if isinstance(e, p.ResultEvent)).payload
    assert "j_q" not in payload["deleted"]
    assert "j_r" not in payload["deleted"]


async def test_prune_keep_tag(daemon):
    _seed(daemon, "j_keep", state="done", age_seconds=86400 * 30, tag="precious")
    _seed(daemon, "j_drop", state="done", age_seconds=86400 * 30, tag="meh")
    events = await _send_recv(p.Prune(older_than_s=86400 * 7, keep_tag="precious"))
    payload = next(e for e in events if isinstance(e, p.ResultEvent)).payload
    assert "j_keep" not in payload["deleted"]
    assert "j_drop" in payload["deleted"]
