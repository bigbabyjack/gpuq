"""Short opaque ids: j_<6> for jobs, l_<6> for leases."""

from __future__ import annotations

import secrets

_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def _short(prefix: str) -> str:
    return prefix + "".join(secrets.choice(_ALPHABET) for _ in range(6))


def new_job_id() -> str:
    return _short("j_")


def new_lease_id() -> str:
    return _short("l_")
