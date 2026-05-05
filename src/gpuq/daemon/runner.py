"""Spawn a job, stream stdout/stderr to JobLogs, await exit."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from collections.abc import Mapping

from gpuq.logs import JobLogs


class Runner:
    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None

    async def run(
        self, *, cmd: list[str], cwd: str, env: Mapping[str, str], logs: JobLogs
    ) -> tuple[int, float]:
        full_env = {**os.environ, **env}
        full_env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        t0 = time.monotonic()
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            env=full_env,
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
        # Pipes have closed; the child should be dead or imminently dead. If
        # asyncio's child watcher misses SIGCHLD (a known-ish hazard around
        # forks/grandchildren that inherit the PID), `_proc.wait()` can hang
        # forever. Bound it, then SIGKILL and bail.
        try:
            rc = await asyncio.wait_for(self._proc.wait(), timeout=10.0)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                self._proc.kill()
            try:
                rc = await asyncio.wait_for(self._proc.wait(), timeout=2.0)
            except TimeoutError:
                rc = -1
        return rc, time.monotonic() - t0

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    async def signal(self, sig: str) -> None:
        if self._proc is None or self._proc.returncode is not None:
            return
        s = signal.SIGTERM if sig == "TERM" else signal.SIGKILL
        with contextlib.suppress(ProcessLookupError):
            self._proc.send_signal(s)
