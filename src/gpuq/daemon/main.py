"""Foreground daemon entrypoint (`gpuq daemon`)."""

from __future__ import annotations

import asyncio
import logging
import signal

from gpuq.daemon.server import Server


async def _run() -> None:
    logging.basicConfig(level=logging.INFO, format="[gpuq] %(message)s")
    server = Server()
    await server.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    await server.stop()


def main() -> None:
    asyncio.run(_run())
