"""Desktop/TUI first-turn routing and host-owned session binding reads."""

from __future__ import annotations

import json
import logging
import threading

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method

logger = logging.getLogger(__name__)

TURN_ROUTE_BINDING_SCHEMA_VERSION = "hermes.turn_route.binding.v1"
_BINDING_FIELDS = (
    "schema_version",
    "status",
    "owner",
    "model",
    "provider",
    "requested_provider",
    "middleware_plugins",
    "middleware_reason",
    "reasoning_effort",
    "reasoning_owner",
)


def _middleware_plugin_names(trace) -> list[str]:
    """Bounded manifest identities only, never arbitrary middleware metadata."""
    names: list[str] = []
    for entry in trace if isinstance(trace, list) else []:
        if not isinstance(entry, dict):
            continue
        raw = entry.get("plugin")
        if not isinstance(raw, str) or not (name := raw.strip()[:128]) or name == "plugin":
            continue
        if name not in names:
            names.append(name)
        if len(names) >= 16:
            break
    return names


def _middleware_reason_code(trace) -> str:
    """Return one bounded machine-readable code, never plugin prose."""
    allowed = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:/-")
    for entry in reversed(trace if isinstance(trace, list) else []):
        raw = entry.get("reason") if isinstance(entry, dict) else None
        if isinstance(raw, str) and 0 < len(raw) <= 64 and all(char in allowed for char in raw):
            return raw
    return ""


def _binding(
    status: str,
    owner: str,
    model: str = "",
    runtime: dict | None = None,
    trace: list | None = None,
    reasoning_config: dict | None = None,
    reasoning_owner: str = "default",
) -> dict:
    runtime = runtime or {}
    trace = trace if isinstance(trace, list) else []
    reason = _middleware_reason_code(trace)
    effort = ""
    if isinstance(reasoning_config, dict):
        effort = ("none" if reasoning_config.get("enabled") is False
                  else str(reasoning_config.get("effort") or "").strip().lower())
    return {
        "schema_version": TURN_ROUTE_BINDING_SCHEMA_VERSION,
        "status": status,
        "owner": owner,
        "model": str(model or ""),
        "provider": str(runtime.get("provider") or ""),
        "requested_provider": str(runtime.get("requested_provider") or runtime.get("provider") or ""),
        "middleware_plugins": _middleware_plugin_names(trace),
        "middleware_reason": reason,
        "reasoning_effort": effort,
        "reasoning_owner": reasoning_owner,
    }


def _safe_binding(raw) -> dict | None:
    """Detached allowlisted binding, or None for an invalid in-memory value."""
    if not isinstance(raw, dict):
        return None
    value = {key: raw.get(key) for key in _BINDING_FIELDS}
    if value.get("schema_version") != TURN_ROUTE_BINDING_SCHEMA_VERSION:
        return None
    if value.get("status") not in {"default", "routed", "user"}:
        return None
    if value.get("owner") not in {"default", "middleware", "user"}:
        return None
    value["model"] = str(value.get("model") or "")
    value["provider"] = str(value.get("provider") or "")
    value["requested_provider"] = str(value.get("requested_provider") or "")
    value["middleware_reason"] = _middleware_reason_code(
        [{"reason": value.get("middleware_reason")}])
    value["reasoning_effort"] = str(value.get("reasoning_effort") or "")
    if value["reasoning_effort"] not in {
        "", "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
    }:
        value["reasoning_effort"] = ""
    if value.get("reasoning_owner") not in {"default", "middleware", "user"}:
        value["reasoning_owner"] = "default"
    plugins = value.get("middleware_plugins")
    plugins = plugins if isinstance(plugins, list) else []
    value["middleware_plugins"] = _middleware_plugin_names(
        [{"plugin": item} for item in plugins])
    return value


def _persist_turn_route_state(session: dict) -> bool:
    """Merge pending/binding state into a new or existing session row."""
    if not session.get("session_key"):
        return True
    if session.get("agent") is not None:
        # One atomic live-runtime write must carry the new binding and any
        # composer provenance. This also gives callers a real failure result.
        return _persist_live_session_runtime(session) is not False
    if _ensure_session_db_row(session) is False:
        return False
    try:
        with _session_db(session) as db:
            if db is None:
                return False
            key = session["session_key"]
            row = db.get_session(key) or {}
            model_config = _parse_model_config(row.get("model_config"), quiet=True)
            if safe := _safe_binding(session.get("turn_route_binding")):
                model_config["turn_route_binding"] = safe
                model_config.pop("turn_route_pending", None)
            elif session.get("turn_route_pending"):
                model_config["turn_route_pending"] = True
            else:
                model_config.pop("turn_route_pending", None)
            override = session.get("model_override") if isinstance(session.get("model_override"), dict) else {}
            runtime = session.get("_turn_route_runtime")
            runtime = runtime if isinstance(runtime, dict) else override
            model = str(override.get("model") or row.get("model") or "").strip()
            provider = str(runtime.get("requested_provider") or runtime.get("provider") or "").strip()
            safe_runtime = {
                "model": model,
                "provider": provider,
                "base_url": str(runtime.get("base_url") or "").strip(),
                "api_mode": str(runtime.get("api_mode") or "").strip(),
            }
            reasoning = session.get("create_reasoning_override")
            if isinstance(reasoning, dict):
                model_config["reasoning_config"] = reasoning
            # A TUI binding supersedes any older messaging-gateway snapshot, which
            # otherwise wins resume precedence over these top-level fields.
            model_config.pop("gateway_runtime", None)
            for runtime_key, value in safe_runtime.items():
                if value:
                    model_config[runtime_key] = value
                else:
                    model_config.pop(runtime_key, None)
            if hasattr(db, "update_session_meta"):
                db.update_session_meta(key, json.dumps(model_config), model or None)
            else:
                return False
        return True
    except Exception:
        logger.debug("failed to persist turn-route binding", exc_info=True)
        return False


def _resolve_public_runtime(model_override: dict | None = None) -> tuple[str, dict]:
    """Resolve with host credentials while retaining only public identity in callers."""
    try:
        return _resolve_agent_model_runtime_strict(model_override, None)
    except Exception as exc:
        from hermes_cli.auth import AuthError

        if not isinstance(exc, AuthError) or getattr(exc, "code", None) in {
            "invalid_provider", "no_provider_configured", "corrupt_config",
        }:
            raise
        # The normal agent build owns configured provider fallback and its user
        # notice. Keep only the requested primary route here so a transient fallback
        # cannot become a durable session selection.
        logger.debug("strict turn-route runtime resolution failed; retaining requested route", exc_info=True)
        if isinstance(model_override, dict) and model_override.get("model"):
            model = str(model_override.get("model") or "").strip()
            provider = str(model_override.get("provider") or "auto").strip() or "auto"
        else:
            model, requested = _resolve_startup_runtime()
            provider = str(requested or "auto").strip() or "auto"
        return model, {"provider": provider, "requested_provider": provider}


def _user_binding_from_override(session: dict) -> dict:
    override = session.get("model_override") if isinstance(session.get("model_override"), dict) else {}
    model, runtime = _resolve_public_runtime(override or None)
    reasoning = session.get("create_reasoning_override")
    if not isinstance(reasoning, dict):
        reasoning = _load_reasoning_config(model)
    return _binding(
        "user", "user", model, runtime,
        reasoning_config=reasoning,
        reasoning_owner="user" if session.get("create_reasoning_override") is not None else "default",
    )


def _arm_initial_turn_route(session: dict) -> bool:
    """Stamp a fresh session as eligible. Absence is never interpreted as eligibility."""
    if session.get("history") or session.get("resume_session_id"):
        return False
    if isinstance(session.get("model_override"), dict) and session["model_override"].get("model"):
        try:
            with _session_profile_runtime_scope(session):
                session["turn_route_binding"] = _user_binding_from_override(session)
        except Exception:
            override = session["model_override"]
            session["turn_route_binding"] = _binding(
                "user", "user", str(override.get("model") or ""),
                {"provider": override.get("provider"), "requested_provider": override.get("provider")})
        session["turn_route_pending"] = False
        return False
    try:
        with _session_profile_runtime_scope(session):
            from hermes_cli.middleware import TURN_ROUTE_MIDDLEWARE
            from hermes_cli.plugins import has_middleware

            pending = has_middleware(TURN_ROUTE_MIDDLEWARE)
    except Exception:
        logger.debug("turn-route availability probe failed", exc_info=True)
        pending = False
    session["turn_route_pending"] = bool(pending)
    return bool(pending)


def _resolve_initial_turn_route(
    sid: str,
    session: dict,
    user_message,
    *,
    internal: bool = False,
) -> dict | None:
    """Freeze and persist one first-turn route before any agent can be built."""
    if not session.get("turn_route_pending"):
        return _safe_binding(session.get("turn_route_binding"))
    lock = session.setdefault("agent_build_lock", threading.Lock())
    with lock:
        if not session.get("turn_route_pending"):
            return _safe_binding(session.get("turn_route_binding"))
        with _session_profile_runtime_scope(session):
            selected_reasoning = None
            explicit = session.get("model_override")
            if isinstance(explicit, dict) and explicit.get("model"):
                selected_model, selected_runtime = _resolve_public_runtime(explicit)
                selected_reasoning = session.get("create_reasoning_override")
                if not isinstance(selected_reasoning, dict):
                    selected_reasoning = _load_reasoning_config(selected_model)
                binding = _binding(
                    "user", "user", selected_model, selected_runtime,
                    reasoning_config=selected_reasoning,
                    reasoning_owner=(
                        "user" if session.get("create_reasoning_override") is not None else "default"),
                )
            else:
                default_model, default_runtime = _resolve_public_runtime()
                default_reasoning = (
                    session.get("create_reasoning_override")
                    if isinstance(session.get("create_reasoning_override"), dict)
                    else _load_reasoning_config(default_model)
                )
                from hermes_cli.turn_routing import resolve_turn_route

                selected = resolve_turn_route(
                    default_model,
                    default_runtime,
                    resolve_runtime=lambda provider, model: _resolve_public_runtime(
                        {"model": model, "provider": provider})[1],
                    user_message=user_message,
                    session_id=sid,
                    session_key=str(session.get("session_key") or "") or None,
                    source=_session_source(session),
                    is_first_turn=True,
                    internal=internal,
                    reasoning_config=default_reasoning,
                    resolve_reasoning=_load_reasoning_config,
                    preserve_reasoning=session.get("create_reasoning_override") is not None,
                )
                selected_model = selected["model"]
                selected_runtime = selected["runtime"]
                trace = selected.get("middleware_trace") or []
                selected_reasoning = selected.get("reasoning_config")
                reasoning_owner = selected.get("reasoning_owner") or "default"
                status, owner = (
                    ("routed", "middleware")
                    if selected.get("middleware_changed")
                    else ("default", "default")
                )
                binding = _binding(
                    status, owner, selected_model, selected_runtime, trace,
                    reasoning_config=selected_reasoning,
                    reasoning_owner=reasoning_owner,
                )
        model_override = {
            "model": selected_model,
            "provider": selected_runtime.get("requested_provider") or selected_runtime.get("provider"),
        }
        persisted = dict(session)
        persisted.update(
            model_override=model_override,
            turn_route_binding=binding,
            turn_route_pending=False,
            _turn_route_runtime=selected_runtime,
        )
        if isinstance(selected_reasoning, dict):
            persisted["create_reasoning_override"] = selected_reasoning
        else:
            persisted.pop("create_reasoning_override", None)
        if not _persist_turn_route_state(persisted):
            raise RuntimeError("turn-route binding could not be persisted")
        session["model_override"] = model_override
        session["turn_route_binding"] = binding
        session["turn_route_pending"] = False
        if isinstance(selected_reasoning, dict):
            session["create_reasoning_override"] = selected_reasoning
        else:
            session.pop("create_reasoning_override", None)
        return dict(binding)


def _mark_turn_route_user_owned(
    session: dict,
    *,
    model: str,
    provider: str | None,
    model_override: dict | None = None,
    reasoning_config: dict | None = None,
) -> None:
    """Commit a persistent user model choice without changing one-turn/internal ownership."""
    if not isinstance(session, dict):
        return
    agent = session.get("agent")
    if agent is not None:
        selected_model = str(getattr(agent, "model", None) or model or "")
        runtime = {
            "provider": str(getattr(agent, "provider", None) or provider or ""),
            "requested_provider": str(getattr(agent, "requested_provider", None) or provider or ""),
            "base_url": str(getattr(agent, "base_url", None) or ""),
            "api_mode": str(getattr(agent, "api_mode", None) or ""),
        }
    else:
        try:
            with _session_profile_runtime_scope(session):
                selected_model, runtime = _resolve_public_runtime({"model": model, "provider": provider})
        except Exception:
            selected_model = str(model or "")
            runtime = {"provider": str(provider or ""), "requested_provider": str(provider or "")}
    explicit_reasoning = isinstance(reasoning_config, dict)
    reasoning = reasoning_config if explicit_reasoning else (
        getattr(agent, "reasoning_config", None) if agent is not None else session.get(
            "create_reasoning_override")
    )
    if not isinstance(reasoning, dict):
        reasoning = _load_reasoning_config(selected_model)
    binding = _binding(
        "user", "user", selected_model, runtime,
        reasoning_config=reasoning,
        reasoning_owner=(
            "user"
            if explicit_reasoning or session.get("create_reasoning_override") is not None
            else "default"
        ),
    )
    persisted = dict(session)
    if model_override is not None:
        persisted["model_override"] = model_override
    if explicit_reasoning:
        # Keep this conversation pinned to the selected effort until /new clears
        # the override. Otherwise a later global config edit can change an
        # agentless session between selection and construction.
        persisted["create_reasoning_override"] = reasoning
    persisted["turn_route_binding"] = binding
    persisted["turn_route_pending"] = False
    persisted["_turn_route_runtime"] = runtime
    previous_agent_reasoning = getattr(agent, "reasoning_config", None) if agent is not None else None
    if explicit_reasoning and agent is not None:
        agent.reasoning_config = reasoning
    if not _persist_turn_route_state(persisted):
        if explicit_reasoning and agent is not None:
            agent.reasoning_config = previous_agent_reasoning
        raise RuntimeError("user turn-route binding could not be persisted")
    if model_override is not None:
        session["model_override"] = model_override
    if explicit_reasoning:
        session["create_reasoning_override"] = reasoning
    session["turn_route_binding"] = binding
    session["turn_route_pending"] = False


def _mark_turn_route_reasoning_owned(
    session: dict,
    reasoning_config: dict,
) -> None:
    """Persist a user reasoning choice and its public binding in the same row write."""
    binding = _safe_binding(session.get("turn_route_binding"))
    if binding is None and not session.get("turn_route_pending"):
        agent = session.get("agent")
        override = session.get("model_override") if isinstance(session.get("model_override"), dict) else {}
        model = str(getattr(agent, "model", None) or override.get("model") or "")
        provider = str(
            getattr(agent, "requested_provider", None)
            or getattr(agent, "provider", None)
            or override.get("provider")
            or ""
        )
        binding = _binding(
            "user", "user", model,
            {"provider": provider, "requested_provider": provider},
        )
    if binding is not None:
        binding = dict(binding)
        binding["reasoning_effort"] = (
            "none"
            if reasoning_config.get("enabled") is False
            else str(reasoning_config.get("effort") or "").strip().lower()
        )
        binding["reasoning_owner"] = "user"

    agent = session.get("agent")
    previous_agent_reasoning = getattr(agent, "reasoning_config", None) if agent is not None else None
    persisted = dict(session)
    if binding is not None:
        persisted["turn_route_binding"] = binding
    # Persist and retain the effective value for this bound conversation on both
    # prebuild and live paths. /new clears the conversation pin.
    persisted["create_reasoning_override"] = reasoning_config
    if agent is not None:
        agent.reasoning_config = reasoning_config
    if not _persist_turn_route_state(persisted):
        if agent is not None:
            agent.reasoning_config = previous_agent_reasoning
        raise RuntimeError("user reasoning binding could not be persisted")

    session["create_reasoning_override"] = reasoning_config
    if binding is not None:
        session["turn_route_binding"] = binding


def _read_binding(session: dict) -> dict:
    if safe := _safe_binding(session.get("turn_route_binding")):
        return safe
    if session.get("turn_route_pending"):
        return {
            "schema_version": TURN_ROUTE_BINDING_SCHEMA_VERSION,
            "status": "pending",
            "owner": "default",
            "model": "",
            "provider": "",
            "requested_provider": "",
            "middleware_plugins": [],
            "middleware_reason": "",
            "reasoning_effort": "",
            "reasoning_owner": "default",
        }
    return {
        "schema_version": TURN_ROUTE_BINDING_SCHEMA_VERSION,
        "status": "unrecorded",
        "owner": "default",
        "model": "",
        "provider": "",
        "requested_provider": "",
        "middleware_plugins": [],
        "middleware_reason": "",
        "reasoning_effort": "",
        "reasoning_owner": "default",
    }


@method("session.turn_route.read")
def _(rid, params: dict) -> dict:
    """Read one live binding without storage, activation, waiting, or agent construction."""
    runtime_id = params.get("session_id")
    stored_id = params.get("stored_session_id")
    if not isinstance(runtime_id, str) or not runtime_id.strip():
        return _err(rid, 4004, "session_id is required")
    if not isinstance(stored_id, str) or not stored_id.strip():
        return _err(rid, 4004, "stored_session_id is required")
    session, err = _sess_nowait({"session_id": runtime_id}, rid)
    if err:
        return err
    if str(session.get("session_key") or "") != stored_id:
        return _err(rid, 4007, "stored_session_id does not match the live session")
    if profile := params.get("profile"):
        if not isinstance(profile, str):
            return _err(rid, 4004, "profile must be a string")
        requested_profile = _canonical_profile_request(profile.strip())
        actual_profile = (
            profile_name_for_home(session.get("profile_home"))
            or str(_current_profile_name() or "default").strip()
        )
        if requested_profile != actual_profile:
            return _err(rid, 4007, "profile does not match the live session")
    return _ok(rid, {
        "session_id": runtime_id,
        "stored_session_id": stored_id,
        "evidence": "session_binding",
        **_read_binding(session),
    })


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))


__all__ = [
    "TURN_ROUTE_BINDING_SCHEMA_VERSION",
    "_arm_initial_turn_route",
    "_mark_turn_route_reasoning_owned",
    "_mark_turn_route_user_owned",
    "_resolve_initial_turn_route",
]
