# gpuq

GPU job scheduler — queue, run, and de-conflict torch / comfyui / ollama jobs on a single GPU.

## Setup

```bash
uv sync
uv run pre-commit install
```

## Install the daemon (systemd --user)

```bash
mkdir -p ~/.config/systemd/user
cp scripts/gpuq.service ~/.config/systemd/user/gpuq.service
systemctl --user daemon-reload
systemctl --user enable --now gpuq
```

The daemon listens on `$XDG_STATE_HOME/gpuq/gpuq.sock`
(default `~/.local/state/gpuq/gpuq.sock`) and persists state in `state.db`
alongside per-job logs in `logs/<id>/`.

## Use

```bash
gpuq submit -- python train.py             # blocks until done, propagates exit code
gpuq submit -t train -p 5 -- ./run.sh      # tagged + higher priority
gpuq submit --next -- ./urgent.sh          # jump ahead of all queued jobs
gpuq submit --describe "ablation A" -- ... # long-form note for history
gpuq ps                                     # active jobs (queued + running)
gpuq ps --all                               # include terminal history
gpuq ps --states running,failed --since 1d  # filter by state(s) and age
gpuq ps --json                              # parseable
gpuq show j_abc123 --json                   # one job
gpuq logs j_abc123 -f                       # tail logs (streams)
gpuq logs j_abc123 -n 50                    # last 50 lines
gpuq bump j_abc123                          # move queued job to the front
gpuq cancel j_abc123                        # SIGTERM, then SIGKILL on timeout
gpuq retry j_abc123                         # resubmit a failed/cancelled job
gpuq doctor                                 # report wedged jobs (no pid + no logs)
gpuq doctor --fix                           # transition wedged jobs to failed
gpuq prune --older-than 7d                  # delete done/failed older than 7d
gpuq prune --older-than 7d --dry-run        # preview only
gpuq ollama evict                           # force ollama out
gpuq ollama restore                         # no-op (ollama lazy-loads)
```

### Recovery and history

- **`gpuq doctor`** — detects jobs stuck in `state=running` with no live pid and
  zero-byte logs (a "wedged" job). Reports without changing state; pass
  `--fix` to transition them to `failed`. Use this instead of restarting the
  daemon when a job hangs without spawning.
- **`gpuq cancel`** is synchronous: it returns `ok=true` only after the state
  has actually transitioned. SIGTERM is sent first; if the child doesn't exit
  within `--timeout` (default 10s) SIGKILL follows. If neither succeeds the
  CLI exits non-zero with `reason=wedged` — run `gpuq doctor --fix --id <id>`
  to clear it.
- **`gpuq logs`** prints `[gpuq] no output yet for <id> (state=..., started=...)`
  on stderr when the log files are empty, so you can tell "no output yet"
  apart from "command failed instantly".
- **`gpuq ps`** defaults to active jobs (`queued`, `starting`, `running`).
  Use `--all` to recover the previous "everything" behaviour.
- **`gpuq retry`** resubmits with the same cmd/cwd/tag/priority/description and
  records the original id as `parent_id` for chain-following. By default only
  `failed`/`cancelled` jobs are retryable; pass `--force` to retry a `done`
  job.
- **`gpuq prune`** deletes terminal jobs (and their `logs/<id>/` directories)
  older than the cutoff. Refuses to prune `queued`/`running`. Always preview
  with `--dry-run` first.

`submit` is the only verb most callers need. It looks like a normal subprocess —
queues the command, runs it when the GPU is free (evicting ollama first if
loaded), streams stdout to stdout and stderr to stderr, and exits with the
job's exit code. Status lines from the daemon (`[gpuq] queued j_abc pos=2
eta_running=8m eta_start=11m`, `[gpuq] done j_abc exit=0`) are emitted on
stderr prefixed `[gpuq]`. ETA estimates appear once there are at least three
recent same-tag (or overall) runs to draw a rolling mean from.

## Library

```python
import gpuq

with gpuq.lease(reason="train"):
    train()  # holds the GPU exclusively for the duration of the block
```

## Test

```bash
uv run pytest
```
