from unittest.mock import patch

import pytest

from gpuq.ollama import OllamaController


def _proc(stdout: bytes = b"", returncode: int = 0):
    class P:
        def __init__(self):
            self.returncode = returncode

        async def communicate(self):
            return stdout, b""

    return P()


@pytest.mark.asyncio
async def test_evict_calls_stop_for_each_loaded_model():
    ps_out = (
        b"NAME              ID    SIZE   PROCESSOR    UNTIL\n"
        b"magnum-v4:latest  abc   13 GB  100% GPU     5 minutes\n"
        b"qwen2:7b          def   4 GB   100% GPU     1 minute\n"
    )
    calls: list[list[str]] = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[1] == "ps":
            return _proc(stdout=ps_out)
        return _proc()

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        ctl = OllamaController()
        evicted = await ctl.evict()
    assert evicted == ["magnum-v4:latest", "qwen2:7b"]
    stop_calls = [c for c in calls if "stop" in c]
    assert {c[2] for c in stop_calls} == {"magnum-v4:latest", "qwen2:7b"}


@pytest.mark.asyncio
async def test_evict_noop_when_ollama_missing():
    async def fake_exec(*cmd, **kwargs):
        raise FileNotFoundError

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        ctl = OllamaController()
        assert await ctl.evict() == []


@pytest.mark.asyncio
async def test_evict_noop_when_ps_fails():
    async def fake_exec(*cmd, **kwargs):
        return _proc(returncode=1)

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        ctl = OllamaController()
        assert await ctl.evict() == []


@pytest.mark.asyncio
async def test_restore_is_noop():
    ctl = OllamaController()
    assert await ctl.restore() is None
