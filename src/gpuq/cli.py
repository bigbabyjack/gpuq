"""gpuq CLI (click-based, thin client over the unix socket)."""

from __future__ import annotations

import asyncio
import json
import os
import sys

import click

from gpuq import paths, protocol as p


def _run(coro):
    return asyncio.run(coro)


async def _open():
    sock = paths.socket_path()
    if not sock.exists():
        click.echo(f"[gpuq] daemon not running at {sock}", err=True)
        click.echo("[gpuq] start it with: systemctl --user start gpuq", err=True)
        sys.exit(64)
    return await asyncio.open_unix_connection(str(sock))


async def _send(req: p.Request):
    reader, writer = await _open()
    writer.write(p.encode_request(req).encode())
    await writer.drain()
    return reader, writer


async def _recv_one(reader) -> p.Event | None:
    line = await reader.readline()
    return p.decode_event(line.decode()) if line else None


@click.group()
def main() -> None:
    """gpuq — single-GPU job scheduler."""


@main.command()
@click.option("-t", "--tag", default="")
@click.option("-p", "--priority", type=int, default=0)
@click.option("--detach", is_flag=True)
@click.argument("cmd", nargs=-1, required=True)
def submit(tag: str, priority: int, detach: bool, cmd: tuple[str, ...]) -> None:
    """Submit a command to the queue. Streams output and exits with its code."""

    async def run():
        req = p.Submit(
            cmd=list(cmd), cwd=os.getcwd(), env=dict(os.environ),
            tag=tag, priority=priority, detach=detach,
        )
        reader, writer = await _send(req)
        exit_code = 0
        try:
            while (ev := await _recv_one(reader)) is not None:
                if isinstance(ev, p.LogEvent):
                    target = sys.stdout if ev.stream == "stdout" else sys.stderr
                    target.write(ev.line)
                    target.flush()
                elif isinstance(ev, p.StateEvent):
                    parts = [f"[gpuq] {ev.state} {ev.id}"]
                    if ev.position is not None:
                        parts.append(f"pos={ev.position}")
                    if ev.exit_code is not None:
                        parts.append(f"exit={ev.exit_code}")
                    click.echo(" ".join(parts), err=True)
                    if ev.exit_code is not None:
                        exit_code = ev.exit_code
                elif isinstance(ev, p.OllamaEvent):
                    click.echo(f"[gpuq] ollama {ev.action} {ev.models}", err=True)
                elif isinstance(ev, p.ErrorEvent):
                    click.echo(f"[gpuq] error: {ev.message}", err=True)
                    exit_code = 1
        finally:
            writer.close()
        sys.exit(exit_code)

    _run(run())


@main.command()
@click.option("--tag", default=None)
@click.option("--state", default=None)
@click.option("--json", "as_json", is_flag=True)
def ps(tag: str | None, state: str | None, as_json: bool) -> None:
    """List jobs."""

    async def run():
        reader, writer = await _send(p.Ps(tag=tag, state=state))
        ev = await _recv_one(reader)
        writer.close()
        payload = ev.payload if isinstance(ev, p.ResultEvent) else {"jobs": []}
        if as_json:
            click.echo(json.dumps(payload))
        else:
            for j in payload.get("jobs", []):
                click.echo(f"{j['id']}  {j['state']:9}  p={j['priority']}  "
                           f"{j.get('tag', '')}  {' '.join(j['cmd'])}")

    _run(run())


@main.command()
@click.argument("id")
@click.option("--json", "as_json", is_flag=True)
def show(id: str, as_json: bool) -> None:
    """Show one job."""

    async def run():
        reader, writer = await _send(p.Show(id=id))
        ev = await _recv_one(reader)
        writer.close()
        payload = ev.payload if isinstance(ev, p.ResultEvent) else {"job": None}
        click.echo(json.dumps(payload, indent=None if as_json else 2))

    _run(run())


@main.command()
@click.argument("id")
@click.option("-f", "--follow", is_flag=True)
@click.option("--from", "from_", type=click.Choice(["start", "tail"]), default="tail")
def logs(id: str, follow: bool, from_: str) -> None:
    """Stream job logs."""

    async def run():
        reader, writer = await _send(p.Attach(id=id, follow=follow, from_=from_))
        try:
            while (ev := await _recv_one(reader)) is not None:
                if isinstance(ev, p.LogEvent):
                    target = sys.stdout if ev.stream == "stdout" else sys.stderr
                    target.write(ev.line)
                    target.flush()
        finally:
            writer.close()

    _run(run())


@main.command()
@click.argument("id")
@click.option("--kill", is_flag=True)
def cancel(id: str, kill: bool) -> None:
    """Cancel a job."""

    async def run():
        reader, writer = await _send(p.Cancel(
            id=id, signal="KILL" if kill else "TERM",
        ))
        ev = await _recv_one(reader)
        writer.close()
        if isinstance(ev, p.ResultEvent) and ev.payload.get("ok"):
            click.echo(f"[gpuq] cancelled {id}", err=True)
        else:
            click.echo(f"[gpuq] could not cancel {id}", err=True)
            sys.exit(1)

    _run(run())


@main.group()
def ollama() -> None:
    """Ollama controls."""


@ollama.command("evict")
def ollama_evict() -> None:
    async def run():
        reader, writer = await _send(p.EvictOllama())
        ev = await _recv_one(reader)
        writer.close()
        click.echo(json.dumps(ev.payload if isinstance(ev, p.ResultEvent) else {}))

    _run(run())


@ollama.command("restore")
def ollama_restore() -> None:
    async def run():
        reader, writer = await _send(p.RestoreOllama())
        await _recv_one(reader)
        writer.close()

    _run(run())


@main.command()
def daemon() -> None:
    """Run the daemon in the foreground (used by systemd)."""
    from gpuq.daemon.main import main as daemon_main

    daemon_main()
