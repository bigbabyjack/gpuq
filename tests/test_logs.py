import asyncio

import pytest
from click.testing import CliRunner

from gpuq import paths
from gpuq import protocol as p
from gpuq.cli import main
from gpuq.logs import JobLogs


def test_append_and_read_combined(tmp_path):
    j = JobLogs(tmp_path / "j_x")
    j.open()
    j.append("stdout", "hello\n")
    j.append("stderr", "err\n")
    j.close()
    assert (tmp_path / "j_x" / "combined.log").read_text() == "hello\nerr\n"
    assert (tmp_path / "j_x" / "stdout.log").read_text() == "hello\n"
    assert (tmp_path / "j_x" / "stderr.log").read_text() == "err\n"


def test_tail_last_n(tmp_path):
    j = JobLogs(tmp_path / "j_x")
    j.open()
    for i in range(50):
        j.append("stdout", f"line {i}\n")
    j.close()
    tail = JobLogs(tmp_path / "j_x").tail(n=5)
    assert tail == [("stdout", f"line {i}\n") for i in range(45, 50)]


def test_read_all_preserves_stream_and_order(tmp_path):
    j = JobLogs(tmp_path / "j_x")
    j.open()
    j.append("stdout", "a\n")
    j.append("stderr", "b\n")
    j.append("stdout", "c\n")
    j.close()
    assert JobLogs(tmp_path / "j_x").read_all() == [
        ("stdout", "a\n"),
        ("stderr", "b\n"),
        ("stdout", "c\n"),
    ]


@pytest.mark.asyncio
@pytest.mark.usefixtures("daemon")
async def test_logs_emits_placeholder_when_files_empty():
    """A queued job has 0-byte logs → CLI prints placeholder on stderr."""
    r, w = await asyncio.open_unix_connection(str(paths.socket_path()))
    w.write(
        p.encode_request(
            p.Submit(cmd=["bash", "-c", "sleep 5"], cwd="/tmp", env={}, detach=True)
        ).encode()
    )
    await w.drain()
    queued_ev = p.decode_event((await r.readline()).decode())
    w.close()
    await w.wait_closed()
    jid = queued_ev.id

    # Block other queue access; before the job runs its logs are empty.
    runner = CliRunner()
    result = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: runner.invoke(main, ["logs", jid], catch_exceptions=False),
    )
    assert result.exit_code == 0
    assert "no output yet" in result.output


@pytest.mark.asyncio
async def test_follow_yields_new_lines(tmp_path):
    j = JobLogs(tmp_path / "j_x")
    j.open()
    out: list[tuple[str, str]] = []

    async def consume():
        async for stream, line in j.follow():
            out.append((stream, line))

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    j.append("stdout", "live\n")
    await asyncio.sleep(0.01)
    j.eof()
    await consumer
    j.close()
    assert ("stdout", "live\n") in out
