"""Retry RPC: resubmits a terminal job, links via parent_id."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

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


def _terminal(daemon, jid: str, *, state: str, tag: str = "") -> None:
    now = datetime.now(UTC).isoformat()
    daemon.store.jobs.insert(
        JobRecord(
            id=jid,
            cmd=["echo", "hi"],
            cwd="/tmp",
            env={},
            tag=tag,
            priority=0,
            state=state,
            pid=None,
            exit_code=1 if state == "failed" else 0,
            submitted_at=now,
            started_at=now,
            finished_at=now,
            description="orig",
        )
    )


async def test_retry_failed_creates_new_job_with_parent_id(daemon):
    _terminal(daemon, "j_orig", state="failed", tag="t")
    events = await _send_recv(p.Retry(id="j_orig"))
    payload = next(e for e in events if isinstance(e, p.ResultEvent)).payload
    assert payload["ok"] is True
    new_id = payload["job"]["id"]
    assert payload["job"]["parent_id"] == "j_orig"
    assert payload["job"]["tag"] == "t"
    assert payload["job"]["description"] == "orig"
    assert new_id != "j_orig"


async def test_retry_done_refused_without_force(daemon):
    _terminal(daemon, "j_done", state="done")
    events = await _send_recv(p.Retry(id="j_done"))
    payload = next(e for e in events if isinstance(e, p.ResultEvent)).payload
    assert payload["ok"] is False
    assert "done" in payload.get("reason", "")


async def test_retry_done_with_force(daemon):
    _terminal(daemon, "j_done2", state="done")
    events = await _send_recv(p.Retry(id="j_done2", force=True))
    payload = next(e for e in events if isinstance(e, p.ResultEvent)).payload
    assert payload["ok"] is True


async def test_retry_unknown_id(daemon):
    events = await _send_recv(p.Retry(id="j_nope"))
    payload = next(e for e in events if isinstance(e, p.ResultEvent)).payload
    assert payload["ok"] is False
