"""JSON-line frame protocol shared by CLI and daemon."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any


class ProtocolError(ValueError):
    """Malformed frame or unknown op."""


# ---- requests ---------------------------------------------------------------


@dataclass(frozen=True)
class Submit:
    cmd: list[str]
    cwd: str
    env: dict[str, str] = field(default_factory=dict)
    tag: str = ""
    priority: int = 0
    detach: bool = False
    next_: bool = False
    description: str = ""
    parent_id: str = ""


@dataclass(frozen=True)
class Bump:
    id: str


@dataclass(frozen=True)
class Attach:
    id: str
    follow: bool = True
    from_: str = "tail"


@dataclass(frozen=True)
class Ps:
    tag: str | None = None
    state: str | None = None
    states: list[str] = field(default_factory=list)
    since: str | None = None


@dataclass(frozen=True)
class Show:
    id: str


@dataclass(frozen=True)
class Cancel:
    id: str
    signal: str = "TERM"
    timeout: float = 10.0


@dataclass(frozen=True)
class Doctor:
    fix: bool = False
    ids: list[str] = field(default_factory=list)
    min_age_s: float = 60.0


@dataclass(frozen=True)
class Retry:
    id: str
    force: bool = False


@dataclass(frozen=True)
class Prune:
    older_than_s: float = 0.0
    states: list[str] = field(default_factory=list)
    keep_tag: str = ""
    dry_run: bool = False


@dataclass(frozen=True)
class Lease:
    duration: str = ""
    reason: str = ""
    priority: int = 0


@dataclass(frozen=True)
class Release:
    lease_id: str


@dataclass(frozen=True)
class EvictOllama:
    pass


@dataclass(frozen=True)
class RestoreOllama:
    pass


Request = (
    Submit
    | Bump
    | Attach
    | Ps
    | Show
    | Cancel
    | Doctor
    | Retry
    | Prune
    | Lease
    | Release
    | EvictOllama
    | RestoreOllama
)

_REQ_BY_OP: dict[str, type] = {
    "submit": Submit,
    "bump": Bump,
    "attach": Attach,
    "ps": Ps,
    "show": Show,
    "cancel": Cancel,
    "doctor": Doctor,
    "retry": Retry,
    "prune": Prune,
    "lease": Lease,
    "release": Release,
    "evict_ollama": EvictOllama,
    "restore_ollama": RestoreOllama,
}
_OP_BY_REQ = {v: k for k, v in _REQ_BY_OP.items()}


def encode_request(req: Request) -> str:
    op = _OP_BY_REQ[type(req)]
    payload: dict[str, Any] = {"op": op}
    payload.update(_dump(req))
    return json.dumps(payload) + "\n"


def decode_request(line: str) -> Request:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"bad json: {e}") from None
    op = obj.get("op")
    cls = _REQ_BY_OP.get(op)
    if cls is None:
        raise ProtocolError(f"unknown op: {op!r}")
    kwargs = {k: v for k, v in obj.items() if k != "op"}
    if "from" in kwargs:
        kwargs["from_"] = kwargs.pop("from")
    if "next" in kwargs:
        kwargs["next_"] = kwargs.pop("next")
    return _load(cls, kwargs)


# ---- events -----------------------------------------------------------------


@dataclass(frozen=True)
class StateEvent:
    id: str
    state: str
    position: int | None = None
    pid: int | None = None
    started_at: str | None = None
    exit_code: int | None = None
    duration_s: float | None = None


@dataclass(frozen=True)
class LogEvent:
    id: str
    stream: str
    line: str


@dataclass(frozen=True)
class OllamaEvent:
    action: str
    models: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResultEvent:
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ErrorEvent:
    message: str
    code: str = "error"


Event = StateEvent | LogEvent | OllamaEvent | ResultEvent | ErrorEvent

_EV_BY_TYPE: dict[str, type] = {
    "state": StateEvent,
    "log": LogEvent,
    "ollama": OllamaEvent,
    "result": ResultEvent,
    "error": ErrorEvent,
}
_TYPE_BY_EV = {v: k for k, v in _EV_BY_TYPE.items()}


def encode_event(ev: Event) -> str:
    payload: dict[str, Any] = {"type": _TYPE_BY_EV[type(ev)]}
    payload.update(_dump(ev))
    return json.dumps(payload) + "\n"


def decode_event(line: str) -> Event:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"bad json: {e}") from None
    cls = _EV_BY_TYPE.get(obj.get("type"))
    if cls is None:
        raise ProtocolError(f"unknown event type: {obj.get('type')!r}")
    kwargs = {k: v for k, v in obj.items() if k != "type"}
    return _load(cls, kwargs)


# ---- helpers ----------------------------------------------------------------


def _dump(d: Any) -> dict[str, Any]:
    out = asdict(d) if is_dataclass(d) else dict(d)
    rename = {"from_": "from", "next_": "next"}
    return {rename.get(k, k): v for k, v in out.items()}


def _load(cls: type, kwargs: dict[str, Any]) -> Any:
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in kwargs.items() if k in valid})
