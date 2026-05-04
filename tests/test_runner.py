import asyncio

import pytest

from gpuq.daemon.runner import Runner
from gpuq.logs import JobLogs


@pytest.mark.asyncio
async def test_run_simple_command_captures_output(tmp_path):
    logs = JobLogs(tmp_path / "j_x")
    logs.open()
    r = Runner()
    rc, _dur = await r.run(
        cmd=["bash", "-c", "echo hello; echo err 1>&2; exit 0"],
        cwd=str(tmp_path), env={"PATH": "/usr/bin:/bin"}, logs=logs,
    )
    logs.close()
    assert rc == 0
    out = (tmp_path / "j_x" / "stdout.log").read_text()
    err = (tmp_path / "j_x" / "stderr.log").read_text()
    assert "hello" in out
    assert "err" in err


@pytest.mark.asyncio
async def test_run_propagates_nonzero_exit(tmp_path):
    logs = JobLogs(tmp_path / "j_y")
    logs.open()
    r = Runner()
    rc, _ = await r.run(
        cmd=["bash", "-c", "exit 7"], cwd=str(tmp_path), env={}, logs=logs,
    )
    logs.close()
    assert rc == 7


@pytest.mark.asyncio
async def test_run_can_be_signalled(tmp_path):
    logs = JobLogs(tmp_path / "j_z")
    logs.open()
    r = Runner()

    task = asyncio.create_task(r.run(
        cmd=["bash", "-c", "sleep 30"], cwd=str(tmp_path), env={}, logs=logs,
    ))
    await asyncio.sleep(0.05)
    await r.signal("TERM")
    rc, _ = await task
    logs.close()
    assert rc != 0
