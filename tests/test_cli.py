import asyncio
import json

import pytest
from click.testing import CliRunner

from gpuq import paths, protocol as p
from gpuq.cli import main


@pytest.mark.asyncio
async def test_submit_streams_output_and_propagates_exit(daemon):
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
async def test_ps_json(daemon):
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


@pytest.mark.asyncio
async def test_cli_no_daemon_exits_64(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    runner = CliRunner()
    result = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: runner.invoke(main, ["ps"], catch_exceptions=False),
    )
    assert result.exit_code == 64
