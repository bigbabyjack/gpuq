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
gpuq submit -- python train.py        # blocks until done, propagates exit code
gpuq submit -t train -p 5 -- ./run.sh # tagged + higher priority
gpuq ps                                # list jobs
gpuq ps --json                         # parseable
gpuq show j_abc123 --json              # one job
gpuq logs j_abc123 -f                  # tail logs
gpuq cancel j_abc123                   # SIGTERM (--kill for SIGKILL)
gpuq ollama evict                      # force ollama out
gpuq ollama restore                    # no-op (ollama lazy-loads)
```

`submit` is the only verb most callers need. It looks like a normal subprocess —
queues the command, runs it when the GPU is free (evicting ollama first if
loaded), streams stdout to stdout and stderr to stderr, and exits with the
job's exit code. Status lines from the daemon (`[gpuq] queued j_abc pos=2`,
`[gpuq] done j_abc exit=0`) are emitted on stderr prefixed `[gpuq]`.

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
