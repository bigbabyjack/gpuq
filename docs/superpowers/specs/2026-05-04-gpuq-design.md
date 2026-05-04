# gpuq — design

A single-GPU job scheduler for one box. Queues commands, runs them one at a time on the GPU, and gets ollama out of the way when a heavier job arrives. Built for Jack's 3090 Ti workload: atelier (ComfyUI / SDXL), gpt-from-scratch (PyTorch training), sd-scripts (Kohya LoRA), and ollama (persistent chat daemon).

## Problem

Three workloads all want the full 24 GB at different times:

- **Batch jobs** (atelier, gpt-from-scratch, sd-scripts) are exclusive 24 GB consumers, kicked off ad hoc, sometimes minutes apart, sometimes hours.
- **Ollama** is a persistent daemon parked on ~14 GB whenever it's warm. It needs to be available for chat, but should disappear the moment a batch job is queued.
- **Manual sessions** (interactive ComfyUI, ad-hoc REPLs, notebooks) want the GPU exclusively for some bounded window.

Today this is managed by hand: starting a training run with ollama warm OOMs; starting ComfyUI while a sd-script is running thrashes. The fix is a queue plus an eviction policy, not a co-scheduler.

## Goals / non-goals

**Goals.**
- Serialize batch jobs on the GPU. FIFO with priority + tags.
- Make ollama yield to any queued batch job; restore on idle.
- One CLI verb (`gpuq submit`) that feels exactly like running the command directly — blocks, streams logs, propagates exit code. Existing scripts and Claude Code's `Bash(run_in_background)` pattern work unchanged.
- Foreground / interactive work cooperates via a held lease.
- Inspectable: `gpuq ps`, `gpuq show`, `gpuq logs -f` with JSON output.

**Non-goals.**
- VRAM-aware co-scheduling, MIG, time-slicing. Single GPU, exclusive runs.
- Multi-host. One box, one daemon.
- Anything fancy with checkpointing or preemption of running jobs (ollama is the only thing that gets evicted, and only because it's stateless to evict).

## Architecture

```
┌─────────────┐     unix socket      ┌─────────────────┐
│ gpuq <verb> │ ◀──── JSON lines ───▶│ gpuq daemon     │
│  (CLI)      │                      │ (systemd user)  │
└─────────────┘                      └────────┬────────┘
                                              │
                          spawns child   ┌────┴────────────┐
                          ────────────▶  │ job process     │
                                         │ (atelier / torch│
                                         │  / comfy / ...) │
                                         └─────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────┐
                       │ ~/.local/share/gpuq/                 │
                       │   gpuq.sock                          │
                       │   state.db   (sqlite)                │
                       │   logs/<id>/{stdout,stderr,combined} │
                       └──────────────────────────────────────┘
```

Two units of code:

- **Daemon** (`gpuq.daemon`). Owns the queue, the GPU lock, the ollama eviction policy, the sqlite state, the log directory, and the unix socket. Exactly one runs per user, under `systemd --user`.
- **CLI** (`gpuq.cli`). Thin client. Connects to the socket, sends a request, streams events to stdout/stderr, exits with the job's exit code. Every verb is a thin wrapper around one socket op.

Optional **library hook** (`gpuq.lease`) for jobs that want to cooperate at sub-process granularity (e.g., release the GPU while doing CPU-side preprocessing). Ships in the same package, opt-in.

### Module layout

```
src/gpuq/
  __init__.py
  cli.py             # click entrypoint, one function per verb
  daemon/
    __init__.py
    server.py        # asyncio unix-socket server
    queue.py         # priority-aware FIFO, in-memory + sqlite-backed
    runner.py        # spawns child processes, owns the GPU lock
    ollama.py        # eviction policy (stop / restart / probe)
    state.py         # sqlite schema, query helpers
    logs.py          # per-job log directory, append + tail
  protocol.py        # JSON-line frames, shared by cli + daemon
  lease.py           # optional library hook
tests/
  test_protocol.py
  test_queue.py
  test_runner.py     # gpu-marked
  test_ollama.py
  test_cli.py
  test_lease.py
```

Boundaries are tight on purpose: `queue.py` doesn't know about subprocesses; `runner.py` doesn't know about sqlite; `ollama.py` doesn't know about the queue. `server.py` wires them together. Each file should fit on a screen.

## Wire protocol

Unix socket at `~/.local/share/gpuq/gpuq.sock`. Line-delimited JSON, one frame per line, both directions.

### Client → daemon ops

| op | fields | reply pattern |
|----|--------|---------------|
| `submit` | `cmd: [str]`, `cwd: str`, `env: {str:str}`, `tag: str`, `priority: int`, `detach: bool` | one `{state:queued, id, position}` then a stream of state + log events until terminal |
| `attach` | `id: str`, `follow: bool`, `from: "start"\|"tail"\|"cursor"` | stream of log + state events |
| `ps` | `tag: str?`, `state: str?` | one `{jobs: [...]}` |
| `show` | `id: str` | one `{job: {...}}` |
| `cancel` | `id: str`, `signal: "TERM"\|"KILL"` | one `{ok: bool, job: {...}}` |
| `lease` | `duration: str` (e.g. `"30m"`), `reason: str?` | one `{lease_id, granted_at, expires_at}` |
| `release` | `lease_id: str` | one `{ok: bool}` |
| `evict_ollama` | — | one `{evicted: bool, was_loaded: [str]}` |
| `restore_ollama` | — | one `{ok: bool}` |

### Daemon → client events

```jsonc
{"type": "state", "id": "j_abc", "state": "queued", "position": 2}
{"type": "state", "id": "j_abc", "state": "starting"}
{"type": "state", "id": "j_abc", "state": "running", "pid": 12345, "started_at": "..."}
{"type": "ollama", "action": "evicted", "models": ["magnum-v4-22b"]}
{"type": "log", "id": "j_abc", "stream": "stdout", "line": "epoch 1 ..."}
{"type": "log", "id": "j_abc", "stream": "stderr", "line": "[gpuq] starting j_abc"}
{"type": "state", "id": "j_abc", "state": "done", "exit_code": 0, "duration_s": 2530}
```

The `submit` connection stays open through the lifecycle so the CLI can stream logs straight to the user's terminal. If the client disconnects, the daemon keeps the job running and keeps writing to the log file; reattach replays from `start`, `tail`, or a saved `cursor`.

## CLI

```
gpuq submit [-t TAG] [-p PRIORITY] [--detach] [--lease DURATION] -- <cmd>...
gpuq lease [DURATION] [--reason TEXT]              # hold the GPU exclusively
gpuq release [LEASE_ID]                            # release a held lease
gpuq ps [--tag TAG] [--state STATE] [--json]
gpuq show ID [--json]
gpuq logs ID [-f] [--from start|tail]
gpuq cancel ID [--kill]
gpuq ollama evict
gpuq ollama restore
gpuq daemon                                        # foreground daemon (for systemd)
```

**Default is blocking.** `gpuq submit -- python train.py` queues, runs when its turn comes, streams the job's stdout/stderr to your terminal, and exits with the job's exit code. From the caller's perspective the queue is invisible — the command "just takes longer to start sometimes."

**Status lines on stderr, output on stdout.** Daemon-side events (`[gpuq] queued j_abc pos=2`, `[gpuq] starting j_abc`, `[gpuq] evicted ollama`, `[gpuq] done j_abc exit=0 elapsed=42m`) are emitted on stderr, prefixed `[gpuq]`. The job's own output is unmodified on stdout. Pipelines work: `gpuq submit -- ./gen.sh 2>/dev/null | tee out.png`.

**JSON for inspection.** `gpuq ps --json` and `gpuq show <id> --json` return parseable state. This is the path Claude Code uses to check "is it queued or running" without grepping logs.

## Scheduling rules

The runner holds at most one of {queued batch job running, manual lease held}. Ollama yields to either; manual leases are coequal to queued jobs and respect FIFO + priority.

```
if no job running and no lease held:
    pop highest-priority queued job
    if ollama is loaded: evict
    spawn job; mark RUNNING
on job exit:
    record exit_code, duration
    if queue empty and no manual lease: restore ollama (background, idempotent)
on lease request:
    if no job running: grant immediately, evict ollama
    else: queue the lease at given priority; grant when GPU free
on lease release / expiry:
    same as job exit
```

Priority is an integer (0 default, higher = sooner). Within a priority, FIFO. No starvation guard in v1 — single user, low load.

Eviction is `ollama stop <model>` for each loaded model (discovered via `ollama ps`). Restore is just letting `ollama` reload on next request — the daemon doesn't pre-warm. (`OLLAMA_KEEP_ALIVE` settings on the user's ollama service are respected.)

## State

SQLite at `~/.local/share/gpuq/state.db`. Schema is small:

```sql
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,            -- j_<6-char nanoid>
    cmd TEXT NOT NULL,              -- json array
    cwd TEXT NOT NULL,
    env TEXT NOT NULL,              -- json object, filtered
    tag TEXT NOT NULL DEFAULT '',
    priority INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL,            -- queued|starting|running|done|failed|cancelled
    pid INTEGER,
    exit_code INTEGER,
    submitted_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);

CREATE TABLE leases (
    id TEXT PRIMARY KEY,
    reason TEXT,
    granted_at TEXT,
    expires_at TEXT,
    released_at TEXT
);

CREATE INDEX jobs_state_priority ON jobs(state, priority DESC, submitted_at);
```

Logs live on disk, not in sqlite — `~/.local/share/gpuq/logs/<id>/{stdout.log,stderr.log,combined.log}`. The combined log is what the CLI streams by default; stdout/stderr are kept separate for downstream tooling.

## Library hook (opt-in)

```python
import gpuq

# Cooperative lease — block until GPU is free, run, release.
with gpuq.lease(reason="train run", priority=10):
    train()

# Sub-step release — give up the GPU during long CPU work.
with gpuq.lease() as lease:
    train_one_epoch()
    with lease.released():
        prepare_next_dataset()   # CPU-only
    train_one_epoch()
```

`gpuq.lease()` talks to the same daemon over the same socket. Out of scope for v1: anything that requires the daemon to know about CUDA contexts.

## Failure modes (deliberately small list)

- **Daemon crash mid-run.** The child process is reparented to PID 1. On restart the daemon scans `jobs` for `state=running` rows, checks the pid, and reconciles (`done` if exited, `running` if still alive, `failed` with exit_code=-1 if pid is gone but no completion event was logged).
- **Socket gone.** CLI prints a clear error and exits 64. `systemctl --user start gpuq` is the documented fix.
- **Ollama not installed / not running.** Eviction is a no-op; logged, not fatal.
- **GPU not present.** Daemon refuses to start with a clear error. CI runs without a GPU; `tests/test_runner.py` is `@pytest.mark.gpu` and skipped.

## Testing

- `test_protocol.py` — round-trip every frame; reject malformed input.
- `test_queue.py` — FIFO, priority, tag filtering, snapshot/restore.
- `test_runner.py` (`gpu`) — submit a tiny torch op; assert it runs, logs are captured, exit code propagates.
- `test_ollama.py` — mock `ollama` CLI; verify evict / restore call the right commands.
- `test_cli.py` — spin up daemon in a tmp dir, exercise verbs end-to-end with a stub command (`bash -c "echo hi; sleep 0.1"`).
- `test_lease.py` — `with gpuq.lease(): ...` blocks the queue.

Integration tests use a real daemon in a temp `XDG_STATE_HOME`; no monkeypatching the socket.

## Build order (informational — actual plan lives elsewhere)

1. Protocol + sqlite state. No daemon yet, just types and persistence.
2. Daemon skeleton + queue (no runner). `submit` returns ids; `ps` lists.
3. Runner: spawn, capture, exit-code propagation. `submit` blocks end-to-end.
4. Ollama eviction.
5. Lease (CLI + library hook).
6. systemd user unit + install docs.

Each step ends with a green test pass and a commit.

## Open questions

- Cross-tag fairness — punt to v2; single-user load doesn't need it.
- GPU 0 vs multiple GPUs — out of scope; assume `CUDA_VISIBLE_DEVICES=0` everywhere.
- ollama eviction granularity — v1 evicts everything, v2 could be model-aware if a "tiny model fits alongside" use case shows up.
