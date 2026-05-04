"""Drive the `ollama` CLI to evict loaded models when a GPU job needs the card."""

from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)


class OllamaController:
    """Evict / restore ollama models. Restore is a no-op — ollama lazy-loads."""

    async def evict(self) -> list[str]:
        models = await self._loaded_models()
        for m in models:
            await self._stop(m)
        if models:
            log.info("evicted ollama models: %s", models)
        return models

    async def restore(self) -> None:
        # Ollama reloads on next request via OLLAMA_KEEP_ALIVE. No-op.
        return None

    async def _loaded_models(self) -> list[str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "ollama", "ps",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            log.debug("ollama not installed")
            return []
        out, err = await proc.communicate()
        if proc.returncode != 0:
            log.debug("ollama ps failed: %s", err.decode(errors="replace"))
            return []
        return _parse_ps(out.decode())

    async def _stop(self, model: str) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "ollama", "stop", model,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return
        await proc.communicate()


def _parse_ps(text: str) -> list[str]:
    lines = text.splitlines()
    if len(lines) < 2:
        return []
    names = []
    for line in lines[1:]:
        line = line.rstrip()
        if not line:
            continue
        names.append(line.split()[0])
    return names
