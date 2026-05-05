"""Doctor RPC: detect and (optionally) clear wedged jobs."""

from __future__ import annotations

import asyncio
import os

import pytest

from gpuq import paths
from gpuq import protocol as p

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("daemon")]


async def _send_recv(req: p.Request) -> list[p.Event]:
    reader, writer = await asyncio.open_unix_connection(str(paths.socket_path()))
    writer.write(p.encode_request(req).encode())
    await writer.drain()
    out: list[p.Event] = []
    while True:
        line = await reader.readline()
        if not line:
            break
        out.append(p.decode_event(line.decode()))
    writer.close()
    await writer.wait_closed()
    return out


def _seed_wedged_job(daemon, *, jid: str = "j_wedge") -> None:
    """Insert a synthetic wedged-running row directly into the store."""
    from datetime import UTC, datetime, timedelta

    from gpuq.state import JobRecord

    started = (datetime.now(UTC) - timedelta(seconds=300)).isoformat()
    daemon.store.jobs.insert(
        JobRecord(
            id=jid,
            cmd=["true"],
            cwd="/tmp",
            env={},
            tag="",
            priority=0,
            state="running",
            pid=None,
            exit_code=None,
            submitted_at=started,
            started_at=started,
            finished_at=None,
        )
    )
    paths.log_dir(jid).mkdir(parents=True, exist_ok=True)
    (paths.log_dir(jid) / "combined.log").write_bytes(b"")


async def test_doctor_detects_wedged_job(daemon):
    _seed_wedged_job(daemon)
    events = await _send_recv(p.Doctor())
    payload = next(e for e in events if isinstance(e, p.ResultEvent)).payload
    ids = [w["id"] for w in payload["wedged"]]
    assert "j_wedge" in ids
    sig = next(w["signature"] for w in payload["wedged"] if w["id"] == "j_wedge")
    assert sig == "wedged-no-pid"


async def test_doctor_no_op_when_healthy(daemon):
    events = await _send_recv(p.Doctor())
    payload = next(e for e in events if isinstance(e, p.ResultEvent)).payload
    assert payload["wedged"] == []


async def test_doctor_fix_transitions_to_failed(daemon):
    _seed_wedged_job(daemon)
    events = await _send_recv(p.Doctor(fix=True))
    payload = next(e for e in events if isinstance(e, p.ResultEvent)).payload
    fixed = [w for w in payload["wedged"] if w["id"] == "j_wedge"]
    assert fixed and fixed[0]["fixed"] is True
    show = await _send_recv(p.Show(id="j_wedge"))
    job = next(e for e in show if isinstance(e, p.ResultEvent)).payload["job"]
    assert job["state"] == "failed"
    assert job["exit_code"] == -1


async def test_doctor_fix_idempotent(daemon):
    _seed_wedged_job(daemon)
    await _send_recv(p.Doctor(fix=True))
    events = await _send_recv(p.Doctor(fix=True))
    payload = next(e for e in events if isinstance(e, p.ResultEvent)).payload
    # Already terminal — no longer reported as wedged.
    assert payload["wedged"] == []


async def test_doctor_excludes_under_min_age(daemon):
    from datetime import UTC, datetime

    from gpuq.state import JobRecord

    fresh = datetime.now(UTC).isoformat()
    daemon.store.jobs.insert(
        JobRecord(
            id="j_fresh",
            cmd=["true"],
            cwd="/tmp",
            env={},
            tag="",
            priority=0,
            state="running",
            pid=None,
            exit_code=None,
            submitted_at=fresh,
            started_at=fresh,
            finished_at=None,
        )
    )
    paths.log_dir("j_fresh").mkdir(parents=True, exist_ok=True)
    events = await _send_recv(p.Doctor())
    payload = next(e for e in events if isinstance(e, p.ResultEvent)).payload
    assert "j_fresh" not in [w["id"] for w in payload["wedged"]]


async def test_doctor_excludes_live_pid(daemon):
    """A running row whose pid is alive isn't wedged regardless of empty logs."""
    from datetime import UTC, datetime, timedelta

    from gpuq.state import JobRecord

    started = (datetime.now(UTC) - timedelta(seconds=300)).isoformat()
    daemon.store.jobs.insert(
        JobRecord(
            id="j_live",
            cmd=["true"],
            cwd="/tmp",
            env={},
            tag="",
            priority=0,
            state="running",
            pid=os.getpid(),  # us — definitely alive
            exit_code=None,
            submitted_at=started,
            started_at=started,
            finished_at=None,
        )
    )
    paths.log_dir("j_live").mkdir(parents=True, exist_ok=True)
    events = await _send_recv(p.Doctor())
    payload = next(e for e in events if isinstance(e, p.ResultEvent)).payload
    assert "j_live" not in [w["id"] for w in payload["wedged"]]
