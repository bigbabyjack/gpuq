import asyncio

import pytest

from gpuq import paths
from gpuq import protocol as p

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("daemon")]


async def _send_recv(req: p.Request) -> list[p.Event]:
    reader, writer = await asyncio.open_unix_connection(str(paths.socket_path()))
    writer.write(p.encode_request(req).encode())
    await writer.drain()
    events: list[p.Event] = []
    while True:
        line = await reader.readline()
        if not line:
            break
        events.append(p.decode_event(line.decode()))
    writer.close()
    await writer.wait_closed()
    return events


async def test_submit_runs_command_and_streams_events():
    events = await _send_recv(
        p.Submit(
            cmd=["bash", "-c", "echo hi"],
            cwd="/tmp",
            env={},
            tag="",
            priority=0,
        )
    )
    states = [e.state for e in events if isinstance(e, p.StateEvent)]
    assert "queued" in states
    assert "running" in states
    assert "done" in states
    log_lines = [e.line for e in events if isinstance(e, p.LogEvent)]
    assert any("hi" in line for line in log_lines)


async def test_submit_propagates_exit_code():
    events = await _send_recv(
        p.Submit(cmd=["bash", "-c", "exit 3"], cwd="/tmp", env={}),
    )
    terminal = [e for e in events if isinstance(e, p.StateEvent) and e.exit_code is not None]
    assert terminal[-1].exit_code == 3
    assert terminal[-1].state == "failed"


async def test_ps_lists_jobs():
    await _send_recv(p.Submit(cmd=["true"], cwd="/tmp", env={}))
    result = await _send_recv(p.Ps())
    payloads = [e for e in result if isinstance(e, p.ResultEvent)]
    assert payloads
    assert "jobs" in payloads[0].payload
    assert len(payloads[0].payload["jobs"]) == 1


async def test_jobs_serialized():
    """Two submits run sequentially, not in parallel."""

    async def submit_and_collect():
        return await _send_recv(
            p.Submit(
                cmd=["bash", "-c", "date +%s%N; sleep 0.2; date +%s%N"],
                cwd="/tmp",
                env={},
            )
        )

    e1, e2 = await asyncio.gather(submit_and_collect(), submit_and_collect())
    times1 = [e.line.strip() for e in e1 if isinstance(e, p.LogEvent) and e.stream == "stdout"]
    times2 = [e.line.strip() for e in e2 if isinstance(e, p.LogEvent) and e.stream == "stdout"]
    assert len(times1) == 2 and len(times2) == 2
    end1, start2 = int(times1[1]), int(times2[0])
    end2, start1 = int(times2[1]), int(times1[0])
    assert end1 <= start2 or end2 <= start1


async def test_show_returns_job():
    submit_events = await _send_recv(p.Submit(cmd=["true"], cwd="/tmp", env={}))
    jid = next(e.id for e in submit_events if isinstance(e, p.StateEvent))
    events = await _send_recv(p.Show(id=jid))
    payload = next(e for e in events if isinstance(e, p.ResultEvent)).payload
    assert payload["job"]["id"] == jid


async def test_submit_next_jumps_ahead_of_queued():
    """A blocker runs; we queue A normally, then B with --next; B should run before A."""
    blocker_events: list[p.Event] = []

    async def blocker():
        blocker_events.extend(
            await _send_recv(p.Submit(cmd=["bash", "-c", "sleep 0.4"], cwd="/tmp", env={}))
        )

    blocker_task = asyncio.create_task(blocker())
    await asyncio.sleep(0.05)

    a_events = await _send_recv(
        p.Submit(cmd=["bash", "-c", "echo A"], cwd="/tmp", env={}, detach=True),
    )
    b_events = await _send_recv(
        p.Submit(cmd=["bash", "-c", "echo B"], cwd="/tmp", env={}, detach=True, next_=True),
    )
    a_id = next(e.id for e in a_events if isinstance(e, p.StateEvent))
    b_id = next(e.id for e in b_events if isinstance(e, p.StateEvent))

    await blocker_task

    # Wait for both to finish
    for _ in range(50):
        ps = await _send_recv(p.Ps())
        jobs = next(e for e in ps if isinstance(e, p.ResultEvent)).payload["jobs"]
        states = {j["id"]: j for j in jobs}
        if states[a_id]["state"] == "done" and states[b_id]["state"] == "done":
            break
        await asyncio.sleep(0.05)

    a_show = await _send_recv(p.Show(id=a_id))
    b_show = await _send_recv(p.Show(id=b_id))
    a_job = next(e for e in a_show if isinstance(e, p.ResultEvent)).payload["job"]
    b_job = next(e for e in b_show if isinstance(e, p.ResultEvent)).payload["job"]
    assert b_job["started_at"] < a_job["started_at"]
    assert b_job["priority"] > a_job["priority"]


async def test_bump_queued_job_to_front():
    blocker_events: list[p.Event] = []

    async def blocker():
        blocker_events.extend(
            await _send_recv(p.Submit(cmd=["bash", "-c", "sleep 0.4"], cwd="/tmp", env={}))
        )

    blocker_task = asyncio.create_task(blocker())
    await asyncio.sleep(0.05)

    a_events = await _send_recv(
        p.Submit(cmd=["bash", "-c", "echo A"], cwd="/tmp", env={}, detach=True),
    )
    b_events = await _send_recv(
        p.Submit(cmd=["bash", "-c", "echo B"], cwd="/tmp", env={}, detach=True),
    )
    a_id = next(e.id for e in a_events if isinstance(e, p.StateEvent))
    b_id = next(e.id for e in b_events if isinstance(e, p.StateEvent))

    bump_events = await _send_recv(p.Bump(id=b_id))
    payload = next(e for e in bump_events if isinstance(e, p.ResultEvent)).payload
    assert payload["ok"] is True
    assert payload["job"]["priority"] > 0

    await blocker_task

    for _ in range(50):
        ps = await _send_recv(p.Ps())
        jobs = next(e for e in ps if isinstance(e, p.ResultEvent)).payload["jobs"]
        states = {j["id"]: j for j in jobs}
        if states[a_id]["state"] == "done" and states[b_id]["state"] == "done":
            break
        await asyncio.sleep(0.05)

    a_job = next(
        e for e in (await _send_recv(p.Show(id=a_id))) if isinstance(e, p.ResultEvent)
    ).payload["job"]
    b_job = next(
        e for e in (await _send_recv(p.Show(id=b_id))) if isinstance(e, p.ResultEvent)
    ).payload["job"]
    assert b_job["started_at"] < a_job["started_at"]


async def test_bump_running_job_returns_false():
    submit_events = await _send_recv(p.Submit(cmd=["true"], cwd="/tmp", env={}))
    jid = next(e.id for e in submit_events if isinstance(e, p.StateEvent))
    # Wait for it to finish; bumping a done job should also fail.
    bump_events = await _send_recv(p.Bump(id=jid))
    payload = next(e for e in bump_events if isinstance(e, p.ResultEvent)).payload
    assert payload["ok"] is False


async def test_cancel_running_waits_for_terminal_state():
    """Cancel of a running job returns ok=True only after state has transitioned."""
    submit_reader, submit_writer = await asyncio.open_unix_connection(str(paths.socket_path()))
    submit_writer.write(
        p.encode_request(
            p.Submit(cmd=["bash", "-c", "sleep 5"], cwd="/tmp", env={}, detach=True)
        ).encode()
    )
    await submit_writer.drain()
    queued = p.decode_event((await submit_reader.readline()).decode())
    submit_writer.close()
    await submit_writer.wait_closed()
    jid = queued.id

    # Wait for it to reach running
    for _ in range(50):
        events = await _send_recv(p.Show(id=jid))
        job = next(e for e in events if isinstance(e, p.ResultEvent)).payload["job"]
        if job["state"] == "running":
            break
        await asyncio.sleep(0.05)
    assert job["state"] == "running"

    cancel_events = await _send_recv(p.Cancel(id=jid, timeout=5.0))
    payload = next(e for e in cancel_events if isinstance(e, p.ResultEvent)).payload
    assert payload["ok"] is True
    # State must already be terminal when cancel returns ok=True
    show_events = await _send_recv(p.Show(id=jid))
    job = next(e for e in show_events if isinstance(e, p.ResultEvent)).payload["job"]
    assert job["state"] in ("failed", "cancelled", "done")


async def test_cancel_running_escalates_to_sigkill_when_sigterm_ignored():
    submit_reader, submit_writer = await asyncio.open_unix_connection(str(paths.socket_path()))
    # Trap TERM so SIGTERM is ignored; only SIGKILL ends it.
    cmd = ["bash", "-c", "trap '' TERM; sleep 30"]
    submit_writer.write(
        p.encode_request(p.Submit(cmd=cmd, cwd="/tmp", env={}, detach=True)).encode()
    )
    await submit_writer.drain()
    queued = p.decode_event((await submit_reader.readline()).decode())
    submit_writer.close()
    await submit_writer.wait_closed()
    jid = queued.id

    for _ in range(50):
        events = await _send_recv(p.Show(id=jid))
        job = next(e for e in events if isinstance(e, p.ResultEvent)).payload["job"]
        if job["state"] == "running":
            break
        await asyncio.sleep(0.05)
    assert job["state"] == "running"

    cancel_events = await _send_recv(p.Cancel(id=jid, timeout=0.5))
    payload = next(e for e in cancel_events if isinstance(e, p.ResultEvent)).payload
    assert payload["ok"] is True


async def test_cancel_queued_job():
    """Submit a long-running blocker, queue a second job, cancel the second."""
    blocker_events: list[p.Event] = []

    async def blocker():
        events = await _send_recv(
            p.Submit(cmd=["bash", "-c", "sleep 0.5"], cwd="/tmp", env={}),
        )
        blocker_events.extend(events)

    blocker_task = asyncio.create_task(blocker())
    await asyncio.sleep(0.05)

    reader, writer = await asyncio.open_unix_connection(str(paths.socket_path()))
    writer.write(p.encode_request(p.Submit(cmd=["true"], cwd="/tmp", env={})).encode())
    await writer.drain()
    queued_event = p.decode_event((await reader.readline()).decode())
    assert isinstance(queued_event, p.StateEvent)
    jid = queued_event.id

    cancel_events = await _send_recv(p.Cancel(id=jid))
    ok_payload = next(e for e in cancel_events if isinstance(e, p.ResultEvent)).payload
    assert ok_payload["ok"] is True

    saw_cancelled = False
    while line := await reader.readline():
        ev = p.decode_event(line.decode())
        if isinstance(ev, p.StateEvent) and ev.state == "cancelled":
            saw_cancelled = True
            break
    assert saw_cancelled
    writer.close()
    await writer.wait_closed()
    await blocker_task
