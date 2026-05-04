import asyncio

import pytest

import gpuq
from gpuq import paths
from gpuq import protocol as p


@pytest.mark.asyncio
async def test_lease_acquires_and_releases(daemon):
    """The library lease() acquires a lease via the socket and releases on exit."""
    loop = asyncio.get_running_loop()

    def hold_briefly() -> str:
        with gpuq.lease(reason="test") as lid:
            return lid

    lid = await loop.run_in_executor(None, hold_briefly)
    assert lid.startswith("l_")

    # Lease record should exist with released_at populated.
    rec = daemon.store.leases.get(lid)
    assert rec is not None
    assert rec.released_at is not None


@pytest.mark.asyncio
@pytest.mark.usefixtures("daemon")
async def test_lease_blocks_queued_jobs():
    """While the lease is held, a queued submit must not start running."""
    loop = asyncio.get_running_loop()
    progress: dict[str, bool] = {"job_started": False, "lease_released": False}

    def hold_lease():
        with gpuq.lease(reason="blocker"):
            # Sleep long enough for a queued job to attempt to start.
            import time

            time.sleep(0.3)
        progress["lease_released"] = True

    lease_task = loop.run_in_executor(None, hold_lease)
    await asyncio.sleep(0.05)  # let the lease be acquired

    # Now submit a job; it should remain queued until the lease releases.
    r, w = await asyncio.open_unix_connection(str(paths.socket_path()))
    w.write(
        p.encode_request(
            p.Submit(
                cmd=["bash", "-c", "echo done"],
                cwd="/tmp",
                env={},
            )
        ).encode()
    )
    await w.drain()

    states: list[str] = []
    while line := await r.readline():
        ev = p.decode_event(line.decode())
        if isinstance(ev, p.StateEvent):
            states.append(ev.state)
            if ev.state == "running" and not progress["lease_released"]:
                progress["job_started"] = True
            if ev.state in ("done", "failed", "cancelled"):
                break
    w.close()
    await w.wait_closed()
    await lease_task

    assert "queued" in states
    assert states[-1] == "done"
    assert (
        progress["job_started"] is False
    ), "job started while lease was held — scheduler did not respect lease"
