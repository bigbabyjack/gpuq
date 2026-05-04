"""Spawn a job, stream stdout/stderr to JobLogs, await exit."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from collections.abc import Mapping

from gpuq.logs import JobLogs


class Runner:
    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None

    async def run(self, *, cmd: list[str], cwd: str, env: Mapping[str, str],
                  logs: JobLogs) -> tuple[int, float]:
        full_env = {**os.environ, **env}
        full_env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        t0 = time.monotonic()
        self._proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=cwd, env=full_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def pump(reader: asyncio.StreamReader, stream: str) -> None:
            while True:
                line = await reader.readline()
                if not line:
                    return
                logs.append(stream, line.decode(errors="replace"))

        assert self._proc.stdout and self._proc.stderr
        await asyncio.gather(
            pump(self._proc.stdout, "stdout"),
            pump(self._proc.stderr, "stderr"),
        )
        rc = await self._proc.wait()
        return rc, time.monotonic() - t0

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    async def signal(self, sig: str) -> None:
        if self._proc is None or self._proc.returncode is not None:
            return
        s = signal.SIGTERM if sig == "TERM" else signal.SIGKILL
        try:
            self._proc.send_signal(s)
        except ProcessLookupError:
            pass
