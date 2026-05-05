import asyncio
import json

import pytest
from click.testing import CliRunner

from gpuq import paths
from gpuq import protocol as p
from gpuq.cli import main


@pytest.mark.asyncio
@pytest.mark.usefixtures("daemon")
async def test_submit_streams_output_and_propagates_exit():
    runner = CliRunner()

    def invoke():
        return runner.invoke(
            main,
            ["submit", "--", "bash", "-c", "echo hello; exit 4"],
            catch_exceptions=False,
        )

    result = await asyncio.get_running_loop().run_in_executor(None, invoke)
    assert result.exit_code == 4
    assert "hello" in result.output


@pytest.mark.asyncio
@pytest.mark.usefixtures("daemon")
async def test_ps_json():
    # Submit a job through the socket directly so we don't need to wait.
    r, w = await asyncio.open_unix_connection(str(paths.socket_path()))
    w.write(p.encode_request(p.Submit(cmd=["true"], cwd="/tmp", env={})).encode())
    await w.drain()
    while await r.readline():
        pass
    w.close()
    await w.wait_closed()

    runner = CliRunner()
    result = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: runner.invoke(main, ["ps", "--json"], catch_exceptions=False),
    )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert "jobs" in parsed
    assert len(parsed["jobs"]) == 1


def test_submit_next_and_priority_mutually_exclusive():
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["submit", "--next", "-p", "5", "--", "true"],
        catch_exceptions=False,
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


@pytest.mark.asyncio
@pytest.mark.usefixtures("daemon")
async def test_bump_command_succeeds_on_queued_job():
    # Block the runner with a slow job, then queue and bump a second.
    r1, w1 = await asyncio.open_unix_connection(str(paths.socket_path()))
    w1.write(
        p.encode_request(
            p.Submit(cmd=["bash", "-c", "sleep 0.4"], cwd="/tmp", env={}, detach=True)
        ).encode()
    )
    await w1.drain()
    await r1.readline()  # queued/running event
    w1.close()

    r2, w2 = await asyncio.open_unix_connection(str(paths.socket_path()))
    w2.write(p.encode_request(p.Submit(cmd=["true"], cwd="/tmp", env={}, detach=True)).encode())
    await w2.drain()
    queued_line = await r2.readline()
    queued_ev = p.decode_event(queued_line.decode())
    assert isinstance(queued_ev, p.StateEvent)
    jid = queued_ev.id
    w2.close()

    runner = CliRunner()
    result = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: runner.invoke(main, ["bump", jid], catch_exceptions=False),
    )
    assert result.exit_code == 0
    assert "bumped" in result.output


@pytest.mark.asyncio
@pytest.mark.usefixtures("daemon")
async def test_bump_unknown_job_exits_nonzero():
    runner = CliRunner()
    result = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: runner.invoke(main, ["bump", "j_nope"], catch_exceptions=False),
    )
    assert result.exit_code == 1


@pytest.mark.asyncio
async def test_cli_no_daemon_exits_64(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    runner = CliRunner()
    result = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: runner.invoke(main, ["ps"], catch_exceptions=False),
    )
    assert result.exit_code == 64
