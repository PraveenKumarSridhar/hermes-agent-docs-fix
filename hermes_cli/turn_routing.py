"""Shared host-owned resolution for public turn-route middleware selections."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def _selected_provider(payload: dict, runtime: dict) -> str:
    """Return the middleware-selected public provider identity, preserving aliases."""
    public_runtime = payload.get("runtime")
    public_runtime = public_runtime if isinstance(public_runtime, dict) else {}
    current_requested = runtime.get("requested_provider") or runtime.get("provider")
    current_canonical = runtime.get("provider")
    top_requested = payload.get("requested_provider")
    nested_requested = public_runtime.get("requested_provider")
    requested = (
        nested_requested
        if nested_requested and nested_requested != current_requested
        else (top_requested or nested_requested)
    )
    canonical = payload.get("provider") or public_runtime.get("provider")
    return str(
        requested
        if requested and requested != current_requested
        else (
            canonical
            if canonical and canonical != current_canonical
            else (requested or canonical or current_requested or "")
        )
    ).strip()


def resolve_turn_route(
    model: str,
    runtime: dict,
    *,
    resolve_runtime: Callable[[str, str], dict],
    user_message: Any,
    session_id: str | None,
    session_key: str | None,
    source: str,
    is_first_turn: bool,
    internal: bool,
    tool_continuation: bool = False,
    reasoning_config: dict | None = None,
    resolve_reasoning: Callable[[str], dict | None] | None = None,
    preserve_reasoning: bool = False,
) -> dict:
    """Resolve one public middleware selection back into a host-owned runtime.

    The callback receives only the selected provider and model. Credentials stay in
    ``runtime`` or in the callback result and never cross the middleware boundary.
    Invalid or raising middleware fails open to the supplied route.
    """
    current_runtime = dict(runtime)
    route = {
        "model": model,
        "runtime": current_runtime,
        "middleware_trace": [],
        "middleware_changed": False,
        "reasoning_config": reasoning_config,
        "reasoning_owner": "user" if preserve_reasoning else "default",
    }
    if internal:
        return route
    try:
        from hermes_cli.middleware import apply_turn_route_middleware, public_turn_route

        result = apply_turn_route_middleware(
            public_turn_route(model, current_runtime, reasoning_config),
            user_message=user_message,
            session_id=session_id,
            session_key=session_key,
            source=source,
            is_user_turn=True,
            is_first_turn=is_first_turn,
            internal=False,
            tool_continuation=tool_continuation,
        )
        if not isinstance(result.payload, dict):
            return route
        selected_model = result.payload.get("model")
        selected_provider = _selected_provider(result.payload, current_runtime)
        if not isinstance(selected_model, str) or not selected_model.strip() or not selected_provider:
            return route
        selected_model = selected_model.strip()
        current_requested = current_runtime.get("requested_provider") or current_runtime.get("provider")
        if selected_provider != current_requested or selected_model != model:
            current_runtime = dict(resolve_runtime(selected_provider, selected_model))
        current_runtime["requested_provider"] = selected_provider
        selected_reasoning = reasoning_config
        reasoning_owner = "user" if preserve_reasoning else "default"
        retain_reasoning = preserve_reasoning or result.payload.get("preserve_reasoning") is True
        if not retain_reasoning and "reasoning_effort" in result.payload:
            effort = str(result.payload.get("reasoning_effort") or "").strip().lower()
            from agent.reasoning_effort import route_supported_efforts
            from hermes_constants import parse_reasoning_effort

            if effort not in route_supported_efforts(selected_provider, selected_model):
                raise ValueError("turn-route middleware selected an unsupported reasoning effort")
            selected_reasoning = parse_reasoning_effort(effort)
            if selected_reasoning is None:
                raise ValueError("turn-route middleware selected an invalid reasoning effort")
            reasoning_owner = "middleware"
        elif not retain_reasoning and selected_model != model and resolve_reasoning is not None:
            selected_reasoning = resolve_reasoning(selected_model)
        route.update(
            model=selected_model,
            runtime=current_runtime,
            middleware_trace=list(result.trace),
            middleware_changed=bool(result.changed),
            reasoning_config=selected_reasoning,
            reasoning_owner=reasoning_owner,
        )
    except Exception as exc:
        logger.warning("Turn-route middleware failed open: %s", exc)
    return route


__all__ = ["resolve_turn_route"]
