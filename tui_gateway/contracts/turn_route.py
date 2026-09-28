"""Host-owned first-turn route binding read contract."""

from __future__ import annotations

from typing import Literal

from .base import Result
from .registry import method
from .sessions import SessionParams


class SessionTurnRouteReadParams(SessionParams):
    stored_session_id: str
    profile: str | None = None


class SessionTurnRouteReadResult(Result):
    schema_version: Literal["hermes.turn_route.binding.v1"]
    session_id: str
    stored_session_id: str
    evidence: Literal["session_binding"]
    status: Literal["pending", "unrecorded", "default", "routed", "user"]
    owner: Literal["default", "middleware", "user"]
    model: str
    provider: str
    requested_provider: str
    middleware_plugins: list[str]
    middleware_reason: str
    reasoning_effort: Literal["", "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]
    reasoning_owner: Literal["default", "middleware", "user"]


method(
    "session.turn_route.read",
    params=SessionTurnRouteReadParams,
    result=SessionTurnRouteReadResult,
    doc="Read one live session's persisted route binding without activating storage or constructing an agent.",
)


__all__ = ["SessionTurnRouteReadParams", "SessionTurnRouteReadResult"]
