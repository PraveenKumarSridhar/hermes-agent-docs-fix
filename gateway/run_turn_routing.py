"""Turn-route selection for the Gateway runner.

The route is resolved before agent construction.  Hermes keeps credentials and transport
ownership; middleware only selects a public model/provider identity for this turn.
"""

from __future__ import annotations

import logging
from typing import Optional

from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")


def _project_runtime_agent_kwargs(runtime_kwargs: dict) -> tuple[dict, dict]:
    """Split provider resolution output into public agent runtime and request overrides."""
    runtime = {
        key: runtime_kwargs.get(key) for key in (
            "api_key", "base_url", "provider", "requested_provider", "api_mode", "command", "args",
            "credential_pool", "max_tokens", "capabilities",
        )
    }
    runtime["args"] = list(runtime["args"] or [])
    runtime["capabilities"] = dict(runtime["capabilities"] or {})
    return runtime, dict(runtime_kwargs.get("request_overrides") or {})


class GatewayTurnRoutingMixin:
    """Resolve the effective public route and host-owned runtime for one Gateway turn."""

    def _resolve_turn_agent_config(
        self, user_message: str, model: str, runtime_kwargs: dict,
        *, session_id: Optional[str] = None, session_key: Optional[str] = None,
        source: Optional[SessionSource] = None, conversation_history: Optional[list] = None,
        # Every caller must classify the turn explicitly. Omitting this would silently bypass
        # external-user routing when a new entry point is added.
        internal: bool,
    ) -> dict:
        """Build one turn route; middleware is fail-open and never owns credentials.

        The default keeps internal/background turns out of user-turn middleware. The external
        TurnRunner passes the event's internal identity after loading persisted turn history.
        """
        from gateway.run import _deep_merge_request_overrides
        from hermes_cli.models import resolve_fast_mode_overrides

        from gateway.run import _resolve_runtime_agent_kwargs_for_provider
        from hermes_cli.turn_routing import resolve_turn_route

        selected = resolve_turn_route(
            model,
            runtime_kwargs,
            resolve_runtime=lambda provider, target_model: _resolve_runtime_agent_kwargs_for_provider(
                provider, target_model=target_model),
            user_message=user_message,
            session_id=session_id,
            session_key=session_key,
            source=source.platform.value if source and source.platform else "gateway",
            is_first_turn=not bool(conversation_history),
            internal=internal,
        )
        runtime, base_request_overrides = _project_runtime_agent_kwargs(selected["runtime"])
        route = {
            "model": selected["model"],
            "runtime": runtime,
            "signature": (
                selected["model"], runtime["provider"], runtime["requested_provider"], runtime["base_url"],
                runtime["api_mode"], runtime["command"], tuple(runtime["args"]),
            ),
        }
        if selected.get("middleware_trace"):
            route["middleware_trace"] = selected["middleware_trace"]
        if getattr(self, "_service_tier", None) != "priority":
            route["request_overrides"] = base_request_overrides
            return route
        try:
            overrides = resolve_fast_mode_overrides(
                route["model"], provider=runtime["provider"], base_url=runtime["base_url"],
            )
        except Exception:
            overrides = None
        route["request_overrides"] = _deep_merge_request_overrides(base_request_overrides, overrides or {})
        return route
