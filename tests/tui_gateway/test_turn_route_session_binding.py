"""Desktop turn-route ordering and durable binding regressions."""

from __future__ import annotations

import contextlib
import copy
import json
import threading
import types

import pytest

from hermes_cli.middleware import RequestMiddlewareResult
from tui_gateway import server


class _DeferredThread:
    """Capture the post-build turn worker without running it."""

    def __init__(self, target=None, **_kwargs):
        self.target = target

    def start(self):
        return None

    def is_alive(self):
        return True


def _fresh_session() -> dict:
    return {
        "agent": None,
        "agent_error": None,
        "agent_ready": threading.Event(),
        "history": [],
        "history_lock": threading.Lock(),
        "running": False,
        "session_key": "stored-route-1",
        "source": "desktop",
        "model_override": None,
        "turn_route_pending": True,
        "profile_home": None,
        "transport": None,
    }


def test_first_prompt_routes_before_agent_construction_and_persists_binding(monkeypatch):
    """Catch agent prebuild or row persistence before the first Desktop route is frozen."""
    session = _fresh_session()
    server._sessions["runtime-route-1"] = session
    observed: list[tuple] = []

    def apply_route(route, **context):
        observed.append(("middleware", route, context))
        selected = {
            "model": "gpt-6-luna",
            "provider": "openai-codex",
            "requested_provider": "openai-codex",
            "runtime": {"provider": "openai-codex", "requested_provider": "openai-codex"},
        }
        return RequestMiddlewareResult(
            payload=selected,
            original_payload=route,
            changed=True,
            trace=[{"plugin": "jev-router", "reason": "economical"}],
        )

    monkeypatch.setattr("hermes_cli.plugins.has_middleware", lambda kind: kind == "turn_route")
    monkeypatch.setattr("hermes_cli.middleware.apply_turn_route_middleware", apply_route)
    monkeypatch.setattr(
        server,
        "_resolve_agent_model_runtime_strict",
        lambda model_override, _provider_override: (
            str((model_override or {}).get("model") or "gpt-6-sol"),
            {
                "provider": str((model_override or {}).get("provider") or "openai-codex"),
                "requested_provider": str((model_override or {}).get("provider") or "openai-codex"),
                "api_mode": "responses",
                "api_key": "host-secret",
            },
        ),
    )
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_args: None)
    monkeypatch.setattr(server, "_legacy_group_fence_error", lambda *_args: None)
    monkeypatch.setattr(server, "_reattach_refusal", lambda *_args: None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_args: False)
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {})
    monkeypatch.setattr(
        server,
        "_lock_in_submit_turn",
        lambda *_args: (session.__setitem__("running", True) or None, {}),
    )

    def persist(_rid, current, _text, _display_kind):
        observed.append(("persist", current.get("model_override"), current.get("turn_route_binding")))
        return None

    def build(_sid, current):
        observed.append(("build", current.get("model_override"), current.get("turn_route_binding")))

    monkeypatch.setattr(server, "_persist_session_row_for_submit", persist)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *_args: False)
    monkeypatch.setattr(server, "_start_agent_build", build)
    monkeypatch.setattr(server.threading, "Thread", _DeferredThread)

    try:
        response = server._methods["prompt.submit"](
            "request-1", {"session_id": "runtime-route-1", "text": "Summarize this report"})
    finally:
        server._sessions.pop("runtime-route-1", None)

    assert response["id"] == "request-1"
    assert response["result"] == {"status": "streaming"}
    assert [item[0] for item in observed] == ["middleware", "persist", "build"]
    for stage in observed[1:]:
        assert stage[1] == {"model": "gpt-6-luna", "provider": "openai-codex"}
        assert stage[2] == {
            "schema_version": "hermes.turn_route.binding.v1",
            "status": "routed",
            "owner": "middleware",
            "model": "gpt-6-luna",
            "provider": "openai-codex",
            "requested_provider": "openai-codex",
            "middleware_plugins": ["jev-router"],
            "middleware_reason": "economical",
            "reasoning_effort": "",
            "reasoning_owner": "default",
        }
    _, _, context = observed[0]
    assert context["session_id"] == "runtime-route-1"
    assert context["session_key"] == "stored-route-1"
    assert context["source"] == "desktop"
    assert context["is_first_turn"] is True


def test_create_with_turn_router_cannot_prewarm_agent(monkeypatch, tmp_path):
    """Catch any create-time build path that bypasses the pending marker."""
    attempted = []
    monkeypatch.setattr("hermes_cli.plugins.has_middleware", lambda kind: kind == "turn_route")
    monkeypatch.setattr(server, "_completion_cwd", lambda _params=None: str(tmp_path))
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_session_default_model", lambda _session: "gpt-6-sol")
    monkeypatch.setattr(server.git_probe, "branch", lambda _cwd: "")
    monkeypatch.setattr(
        server,
        "_make_agent",
        lambda *_args, **_kwargs: attempted.append("built") or (_ for _ in ()).throw(
            AssertionError("prewarm built before the first route")),
    )

    def prewarm_now(sid):
        server._start_agent_build(sid, server._sessions[sid])

    monkeypatch.setattr(server, "_schedule_agent_build", prewarm_now)
    response = server._methods["session.create"]("create-route", {"cols": 80})
    sid = response["result"]["session_id"]
    try:
        session = server._sessions[sid]
        assert session["turn_route_pending"] is True
        assert not session.get("agent_build_started")
        assert attempted == []
    finally:
        server._sessions.pop(sid, None)


def test_prewarm_timer_waits_for_binding_persistence(monkeypatch):
    """A timer firing inside a slow callback cannot construct the agent first."""
    session = _fresh_session()
    server._sessions["runtime-race"] = session
    entered = threading.Event()
    release = threading.Event()
    persisted = threading.Event()
    inner_started = []
    real_thread = threading.Thread

    def apply_route(route, **_context):
        entered.set()
        assert release.wait(timeout=5)
        return RequestMiddlewareResult(
            payload={
                "model": "gpt-6-luna",
                "provider": "openai-codex",
                "runtime": {"provider": "openai-codex", "requested_provider": "openai-codex"},
            },
            original_payload=route,
            changed=True,
            trace=[{"plugin": "jev-router"}],
        )

    class _BuildThread:
        def __init__(self, target=None, **_kwargs):
            self.target = target

        def start(self):
            assert persisted.is_set()
            assert session["turn_route_binding"]["model"] == "gpt-6-luna"
            inner_started.append(True)

        def is_alive(self):
            return True

    monkeypatch.setattr("hermes_cli.middleware.apply_turn_route_middleware", apply_route)
    monkeypatch.setattr(
        server,
        "_resolve_agent_model_runtime_strict",
        lambda override, _provider: (
            str((override or {}).get("model") or "gpt-6-sol"),
            {
                "provider": str((override or {}).get("provider") or "openai-codex"),
                "requested_provider": str((override or {}).get("provider") or "openai-codex"),
            },
        ),
    )
    monkeypatch.setattr(server, "_persist_turn_route_state", lambda _session: persisted.set() or True)
    monkeypatch.setattr(server.threading, "Thread", _BuildThread)

    route_thread = real_thread(
        target=lambda: server._resolve_initial_turn_route(
            "runtime-race", session, "Classify this request"))
    build_thread = real_thread(target=lambda: server._start_agent_build("runtime-race", session))
    try:
        route_thread.start()
        assert entered.wait(timeout=5)
        build_thread.start()
        build_thread.join(timeout=0.05)
        assert build_thread.is_alive()
        assert not session.get("agent_build_started")
        assert inner_started == []
        release.set()
        route_thread.join(timeout=5)
        build_thread.join(timeout=5)
        assert not route_thread.is_alive()
        assert not build_thread.is_alive()
        assert inner_started == [True]
    finally:
        release.set()
        server._sessions.pop("runtime-race", None)


def test_route_binding_merges_into_an_existing_row(monkeypatch):
    """INSERT OR IGNORE rows still receive the committed binding before a build."""
    session = _fresh_session()
    row = {"model": "gpt-6-sol", "model_config": json.dumps({
        "existing": "kept", "provider": "old-provider", "base_url": "https://old.invalid",
    })}
    writes = []

    class _Db:
        def get_session(self, key):
            assert key == "stored-route-1"
            return dict(row)

        def update_session_meta(self, key, model_config, model):
            writes.append((key, json.loads(model_config), model))

    @contextlib.contextmanager
    def session_db(_session):
        yield _Db()

    monkeypatch.setattr("hermes_cli.middleware.apply_turn_route_middleware", lambda route, **_context: RequestMiddlewareResult(
        payload={
            "model": "gpt-6-luna",
            "provider": "openai-codex",
            "runtime": {"provider": "openai-codex", "requested_provider": "openai-codex"},
        },
        original_payload=route,
        changed=True,
        trace=[{"plugin": "jev-router", "reason": "private explanation"}],
    ))
    monkeypatch.setattr(
        server,
        "_resolve_agent_model_runtime_strict",
        lambda override, _provider: (
            str((override or {}).get("model") or "gpt-6-sol"),
            {
                "provider": str((override or {}).get("provider") or "openai-codex"),
                "requested_provider": str((override or {}).get("provider") or "openai-codex"),
                "api_mode": "responses",
            },
        ),
    )
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: True)
    monkeypatch.setattr(server, "_session_db", session_db)

    server._resolve_initial_turn_route("runtime-route-1", session, "Summarize this")

    assert len(writes) == 1
    key, model_config, model = writes[0]
    assert key == "stored-route-1"
    assert model == "gpt-6-luna"
    assert model_config["existing"] == "kept"
    assert model_config["model"] == "gpt-6-luna"
    assert model_config["provider"] == "openai-codex"
    assert model_config["api_mode"] == "responses"
    assert "base_url" not in model_config
    assert model_config["turn_route_binding"] == session["turn_route_binding"]
    assert "turn_route_pending" not in model_config
    assert "private explanation" not in json.dumps(model_config)


def test_pending_database_row_routes_and_resumes_with_the_committed_runtime(monkeypatch, tmp_path):
    """Exercise the real row parser, metadata write, restart read, and cold-resume projection."""
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session(
        "stored-route-1",
        source="desktop",
        model="gpt-6-sol",
        model_config={
            "turn_route_pending": True,
            "gateway_runtime": {"provider": "old-provider", "base_url": "https://old.invalid"},
        },
    )
    pending = server._stored_session_runtime_overrides(db.get_session("stored-route-1"))
    assert pending == {"turn_route_pending": True}

    session = _fresh_session()
    session.update(
        model_override=pending.get("model_override"),
        turn_route_pending=pending["turn_route_pending"],
    )

    def strict(override, _provider):
        model = str((override or {}).get("model") or "gpt-6-sol")
        provider = str((override or {}).get("provider") or "openai-codex")
        return model, {
            "provider": provider,
            "requested_provider": provider,
            "base_url": "https://new.example/v1",
            "api_mode": "responses",
            "api_key": "never-persist-this",
        }

    @contextlib.contextmanager
    def session_db(_session):
        yield db

    monkeypatch.setattr(server, "_resolve_agent_model_runtime_strict", strict)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: True)
    monkeypatch.setattr(server, "_session_db", session_db)
    monkeypatch.setattr("hermes_cli.middleware.apply_turn_route_middleware", lambda route, **_context: RequestMiddlewareResult(
        payload={
            **route,
            "model": "gpt-6-luna",
            "provider": "openai-codex",
            "requested_provider": "openai-codex",
        },
        original_payload=route,
        changed=True,
        trace=[{"plugin": "jev-router"}],
    ))

    binding = server._resolve_initial_turn_route("runtime", session, "route after restart")
    db.close()

    with SessionDB(db_path=db_path) as reopened:
        row = reopened.get_session("stored-route-1")
    config = json.loads(row["model_config"])
    resumed = server._stored_session_runtime_overrides(row)

    assert binding["status"] == "routed"
    assert row["model"] == "gpt-6-luna"
    assert config["provider"] == "openai-codex"
    assert config["base_url"] == "https://new.example/v1"
    assert config["api_mode"] == "responses"
    assert "gateway_runtime" not in config
    assert "api_key" not in json.dumps(config)
    assert "turn_route_pending" not in config
    assert resumed["turn_route_binding"] == binding
    assert resumed["model_override"] == {
        "model": "gpt-6-luna",
        "provider": "openai-codex",
        "base_url": "https://new.example/v1",
        "api_mode": "responses",
    }


def test_resume_metadata_distinguishes_committed_pending_and_legacy_rows(monkeypatch):
    binding = {
        "schema_version": "hermes.turn_route.binding.v1",
        "status": "routed",
        "owner": "middleware",
        "model": "gpt-6-luna",
        "provider": "openai-codex",
        "requested_provider": "openai-codex",
        "middleware_plugins": ["jev-router"],
        "middleware_reason": "",
        "reasoning_effort": "",
        "reasoning_owner": "default",
    }
    monkeypatch.setattr(server, "_is_routable_provider", lambda provider: bool(provider))
    committed = server._stored_session_runtime_overrides({
        "model": "gpt-6-luna",
        "model_config": json.dumps({
            "model": "gpt-6-luna",
            "provider": "openai-codex",
            "turn_route_binding": binding,
        }),
    })
    pending = server._stored_session_runtime_overrides({
        "model": "gpt-6-sol",
        "model_config": json.dumps({"turn_route_pending": True}),
    })
    legacy = server._stored_session_runtime_overrides({
        "model": "gpt-6-sol",
        "model_config": "{}",
    })

    assert committed["turn_route_binding"] == binding
    assert "turn_route_pending" not in committed
    assert pending["turn_route_pending"] is True
    assert "model_override" not in pending
    assert "provider_override" not in pending
    assert "turn_route_binding" not in pending
    assert "turn_route_pending" not in legacy
    assert "turn_route_binding" not in legacy


def test_binding_persistence_failure_keeps_session_pending(monkeypatch):
    session = _fresh_session()
    monkeypatch.setattr(
        server,
        "_resolve_agent_model_runtime_strict",
        lambda _override, _provider: (
            "gpt-6-sol", {"provider": "openai-codex", "requested_provider": "openai-codex"}),
    )
    monkeypatch.setattr("hermes_cli.middleware.apply_turn_route_middleware", lambda route, **_context: RequestMiddlewareResult(
        payload={**route, "model": "gpt-6-luna"}, original_payload=route,
        changed=True, trace=[{"plugin": "jev-router"}],
    ))
    monkeypatch.setattr(server, "_persist_turn_route_state", lambda _session: False)

    with pytest.raises(RuntimeError, match="turn-route binding"):
        server._resolve_initial_turn_route("runtime", session, "route this")

    assert session["turn_route_pending"] is True
    assert "turn_route_binding" not in session
    assert session["model_override"] is None


def test_route_resolution_failure_after_admission_releases_turn(monkeypatch):
    session = _fresh_session()
    session["inflight_turn"] = None
    server._sessions["runtime-route-error"] = session
    released = []
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_args: None)
    monkeypatch.setattr(server, "_legacy_group_fence_error", lambda *_args: None)
    monkeypatch.setattr(server, "_reattach_refusal", lambda *_args: None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_args: False)
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {})
    monkeypatch.setattr(
        server,
        "_lock_in_submit_turn",
        lambda *_args: (
            session.update(running=True, inflight_turn={"user": "route this"}) or None,
            {},
        ),
    )
    monkeypatch.setattr(
        server,
        "_resolve_initial_turn_route",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("storage unavailable")),
    )
    monkeypatch.setattr(server, "_release_active_session_slot", lambda current: released.append(current) or True)

    try:
        response = server._methods["prompt.submit"](
            "route-error", {"session_id": "runtime-route-error", "text": "route this"})
    finally:
        server._sessions.pop("runtime-route-error", None)

    assert response["error"]["code"] == 5071
    assert session["running"] is False
    assert session["inflight_turn"] is None
    assert released == [session]


def test_public_route_resolution_never_pins_auth_fallback(monkeypatch):
    from hermes_cli.auth import AuthError

    monkeypatch.setattr(
        server,
        "_resolve_agent_model_runtime_strict",
        lambda *_args: (_ for _ in ()).throw(
            AuthError("primary unavailable", provider="primary-provider", code="missing_api_key")),
    )
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: ("primary-model", "primary-provider"))
    monkeypatch.setattr(
        server,
        "_resolve_agent_model_runtime",
        lambda *_args: pytest.fail("routing must not consume the operational fallback chain"),
    )

    model, runtime = server._resolve_public_runtime()

    assert model == "primary-model"
    assert runtime == {"provider": "primary-provider", "requested_provider": "primary-provider"}


def test_invalid_selected_provider_fails_open_to_default_binding(monkeypatch):
    session = _fresh_session()

    def strict(override, _provider):
        if override:
            raise ValueError("unknown selected provider")
        return "gpt-6-sol", {
            "provider": "openai-codex", "requested_provider": "openai-codex"}

    monkeypatch.setattr(server, "_resolve_agent_model_runtime_strict", strict)
    monkeypatch.setattr(server, "_persist_turn_route_state", lambda _session: True)
    monkeypatch.setattr(
        "hermes_cli.middleware.apply_turn_route_middleware",
        lambda route, **_context: RequestMiddlewareResult(
            payload={**route, "model": "bad-model", "provider": "unknown-provider"},
            original_payload=route,
            changed=True,
            trace=[{"plugin": "jev-router"}],
        ),
    )

    binding = server._resolve_initial_turn_route(
        "runtime", session, "route this invalid selection")

    assert binding["status"] == "default"
    assert binding["owner"] == "default"
    assert binding["model"] == "gpt-6-sol"
    assert binding["provider"] == "openai-codex"
    assert binding["middleware_plugins"] == []


def test_committed_binding_never_reruns_middleware(monkeypatch):
    session = _fresh_session()
    session["turn_route_pending"] = False
    session["turn_route_binding"] = {
        "schema_version": "hermes.turn_route.binding.v1",
        "status": "routed",
        "owner": "middleware",
        "model": "gpt-6-luna",
        "provider": "openai-codex",
        "requested_provider": "openai-codex",
        "middleware_plugins": ["jev-router"],
        "middleware_reason": "",
        "reasoning_effort": "",
        "reasoning_owner": "default",
    }
    monkeypatch.setattr(
        "hermes_cli.middleware.apply_turn_route_middleware",
        lambda *_args, **_kwargs: pytest.fail("committed resume must not rerun middleware"),
    )

    first = server._resolve_initial_turn_route("runtime", session, "one")
    second = server._resolve_initial_turn_route("runtime", session, "two")

    assert first == second == session["turn_route_binding"]


def test_pending_binding_routes_exactly_once(monkeypatch):
    session = _fresh_session()
    calls = []
    monkeypatch.setattr("hermes_cli.middleware.apply_turn_route_middleware", lambda route, **_context: (
        calls.append(1) or RequestMiddlewareResult(
            payload={
                "model": "gpt-6-luna",
                "provider": "openai-codex",
                "runtime": {"provider": "openai-codex", "requested_provider": "openai-codex"},
            },
            original_payload=route,
            changed=True,
            trace=[{"plugin": "jev-router"}],
        )
    ))
    monkeypatch.setattr(
        server,
        "_resolve_agent_model_runtime_strict",
        lambda override, _provider: (
            str((override or {}).get("model") or "gpt-6-sol"),
            {
                "provider": str((override or {}).get("provider") or "openai-codex"),
                "requested_provider": str((override or {}).get("provider") or "openai-codex"),
            },
        ),
    )
    monkeypatch.setattr(server, "_persist_turn_route_state", lambda _session: True)

    server._resolve_initial_turn_route("runtime", session, "first")
    server._resolve_initial_turn_route("runtime", session, "second")

    assert calls == [1]
    assert session["turn_route_pending"] is False


def test_internal_first_submit_freezes_default_without_invoking_middleware(monkeypatch):
    session = _fresh_session()
    monkeypatch.setattr(
        "hermes_cli.middleware.apply_turn_route_middleware",
        lambda *_args, **_kwargs: pytest.fail("internal submit must bypass user turn routing"),
    )
    monkeypatch.setattr(
        server,
        "_resolve_agent_model_runtime_strict",
        lambda _override, _provider: (
            "gpt-6-sol",
            {"provider": "openai-codex", "requested_provider": "openai-codex"},
        ),
    )
    monkeypatch.setattr(server, "_persist_turn_route_state", lambda _session: True)

    binding = server._resolve_initial_turn_route(
        "runtime", session, "internal task", internal=True)

    assert binding["status"] == "default"
    assert binding["owner"] == "default"
    assert session["turn_route_pending"] is False


def test_compute_host_dispatch_observes_selected_public_route(monkeypatch):
    session = _fresh_session()
    server._sessions["runtime-compute"] = session
    dispatched = []
    monkeypatch.setattr("hermes_cli.middleware.apply_turn_route_middleware", lambda route, **_context: RequestMiddlewareResult(
        payload={
            "model": "gpt-6-luna",
            "provider": "openai-codex",
            "runtime": {"provider": "openai-codex", "requested_provider": "openai-codex"},
        },
        original_payload=route,
        changed=True,
        trace=[{"plugin": "jev-router"}],
    ))
    monkeypatch.setattr(
        server,
        "_resolve_agent_model_runtime_strict",
        lambda override, _provider: (
            str((override or {}).get("model") or "gpt-6-sol"),
            {
                "provider": str((override or {}).get("provider") or "openai-codex"),
                "requested_provider": str((override or {}).get("provider") or "openai-codex"),
                "api_key": "host-only-secret",
            },
        ),
    )
    monkeypatch.setattr(server, "_persist_turn_route_state", lambda _session: True)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_args: None)
    monkeypatch.setattr(server, "_legacy_group_fence_error", lambda *_args: None)
    monkeypatch.setattr(server, "_reattach_refusal", lambda *_args: None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_args: True)
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {"enabled": True})
    monkeypatch.setattr(
        server,
        "_lock_in_submit_turn",
        lambda *_args: (session.__setitem__("running", True) or None, {}),
    )

    def dispatch(_rid, _sid, current, _text, **_kwargs):
        dispatched.append((current.get("model_override"), current.get("turn_route_binding")))
        return {"id": "compute", "result": {"status": "streaming"}}

    monkeypatch.setattr(server, "_submit_prompt_to_compute_host", dispatch)
    try:
        response = server._methods["prompt.submit"](
            "compute", {"session_id": "runtime-compute", "text": "Analyze this"})
    finally:
        server._sessions.pop("runtime-compute", None)

    assert response["result"]["status"] == "streaming"
    assert dispatched[0][0] == {"model": "gpt-6-luna", "provider": "openai-codex"}
    assert dispatched[0][1]["status"] == "routed"
    assert "host-only-secret" not in json.dumps(dispatched[0])


def test_preprompt_user_model_selection_cancels_pending_route(monkeypatch):
    session = _fresh_session()
    server._sessions["runtime-user"] = session
    monkeypatch.setattr(
        server,
        "_start_agent_build",
        lambda *_args: pytest.fail("a pre-prompt user choice must not build the default agent"),
    )
    monkeypatch.setattr(
        server,
        "_resolve_agent_model_runtime_strict",
        lambda override, _provider: (
            str((override or {}).get("model") or "gpt-6-sol"),
            {
                "provider": str((override or {}).get("provider") or "openai-codex"),
                "requested_provider": str((override or {}).get("provider") or "openai-codex"),
            },
        ),
    )
    monkeypatch.setattr(server, "_persist_turn_route_state", lambda _session: True)

    def switch(_sid, current, _raw, **_kwargs):
        current["model_override"] = {"model": "gpt-6-astra", "provider": "openai-codex"}
        server._mark_turn_route_user_owned(
            current, model="gpt-6-astra", provider="openai-codex")
        return {"value": "gpt-6-astra", "warning": "", "scope": "session"}

    monkeypatch.setattr(server, "_apply_model_switch", switch)
    try:
        response = server._methods["config.set"](
            "user-route",
            {"session_id": "runtime-user", "key": "model", "value": "gpt-6-astra"},
        )
    finally:
        server._sessions.pop("runtime-user", None)

    assert response["result"]["value"] == "gpt-6-astra"
    assert session["turn_route_pending"] is False
    assert session["turn_route_binding"]["status"] == "user"
    assert session["turn_route_binding"]["owner"] == "user"


def test_concurrent_preprompt_user_selection_wins_after_inflight_route(monkeypatch):
    session = _fresh_session()
    route_entered = threading.Event()
    release_route = threading.Event()
    route_result = []
    switch_result = []

    def apply_route(route, **_context):
        route_entered.set()
        assert release_route.wait(timeout=5)
        return RequestMiddlewareResult(
            payload={**route, "model": "gpt-6-luna", "provider": "openai-codex"},
            original_payload=route,
            changed=True,
            trace=[{"plugin": "jev-router"}],
        )

    def strict(override, _provider):
        model = str((override or {}).get("model") or "gpt-6-sol")
        provider = str((override or {}).get("provider") or "openai-codex")
        return model, {"provider": provider, "requested_provider": provider}

    switch = types.SimpleNamespace(
        success=True,
        error_message="",
        warning_message="",
        new_model="gpt-6-astra",
        target_provider="openai-codex",
        base_url="",
        api_key="secret",
        api_mode="responses",
        model_info=None,
    )
    monkeypatch.setattr("hermes_cli.middleware.apply_turn_route_middleware", apply_route)
    monkeypatch.setattr(server, "_resolve_agent_model_runtime_strict", strict)
    monkeypatch.setattr(server, "_persist_turn_route_state", lambda _session: True)
    monkeypatch.setattr(server, "_current_model_runtime", lambda *_args: (
        "openai-codex", "gpt-6-sol", "", "secret"))
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kwargs: switch)
    monkeypatch.setattr(server, "_expensive_model_confirm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics_events.record_model_switch",
        lambda **_kwargs: None,
    )

    route_thread = threading.Thread(target=lambda: route_result.append(
        server._resolve_initial_turn_route("runtime", session, "route this")))
    switch_thread = threading.Thread(target=lambda: switch_result.append(
        server._apply_model_switch("runtime", session, "gpt-6-astra")))
    route_thread.start()
    assert route_entered.wait(timeout=5)
    switch_thread.start()
    switch_thread.join(timeout=0.05)
    assert switch_thread.is_alive()
    release_route.set()
    route_thread.join(timeout=5)
    switch_thread.join(timeout=5)

    assert not route_thread.is_alive()
    assert not switch_thread.is_alive()
    assert route_result[0]["status"] == "routed"
    assert switch_result[0]["value"] == "gpt-6-astra"
    assert session["model_override"]["model"] == "gpt-6-astra"
    assert session["turn_route_binding"]["status"] == "user"
    assert session["turn_route_binding"]["model"] == "gpt-6-astra"


@pytest.mark.parametrize("turn_route_pending", [False, True])
def test_user_selection_waits_for_inflight_agent_build(monkeypatch, turn_route_pending):
    """A routed build must attach before config.set mutates the live runtime."""
    session = _fresh_session()
    session.update(turn_route_pending=turn_route_pending, agent_build_started=True)
    switched = []

    class _Agent:
        model = "gpt-6-luna"
        provider = "openai-codex"
        requested_provider = "openai-codex"
        base_url = ""
        api_key = "old-secret"
        api_mode = "responses"
        reasoning_config = {"effort": "medium"}
        _primary_runtime = None
        session_id = "stored-route-1"

        def switch_model(self, *, new_model, new_provider, api_key, base_url, api_mode, **_kwargs):
            switched.append((new_model, new_provider))
            self.model = new_model
            self.provider = new_provider
            self.requested_provider = new_provider
            self.api_key = api_key
            self.base_url = base_url
            self.api_mode = api_mode

    result = types.SimpleNamespace(
        success=True,
        error_message="",
        warning_message="",
        new_model="gpt-6-astra",
        target_provider="openai-codex",
        base_url="",
        api_key="new-secret",
        api_mode="responses",
        model_info=None,
    )
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kwargs: result)
    monkeypatch.setattr(server, "_expensive_model_confirm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_persist_turn_route_state", lambda _session: True)
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda _session: True)
    monkeypatch.setattr(server, "_emit_session_info", lambda *_args: None)
    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics_events.record_model_switch",
        lambda **_kwargs: None,
    )

    values, errors = [], []

    def run_switch():
        try:
            values.append(server._apply_model_switch(
                "runtime", session, "gpt-6-astra --provider openai-codex"))
        except Exception as exc:  # pragma: no cover - assertion below reports the failure
            errors.append(exc)

    worker = threading.Thread(target=run_switch)
    worker.start()
    worker.join(timeout=0.05)
    assert worker.is_alive()

    session["agent"] = _Agent()
    session["agent_ready"].set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert errors == []
    assert values[0]["value"] == "gpt-6-astra"
    assert switched == [("gpt-6-astra", "openai-codex")]
    assert session["turn_route_binding"]["model"] == "gpt-6-astra"


def test_first_prompt_persists_reasoning_choice_and_default_trace(monkeypatch):
    session = _fresh_session()

    def apply_route(route, **_context):
        assert route["current_reasoning_effort"] == "medium"
        return RequestMiddlewareResult(
            payload={**route, "model": "gpt-6-luna", "reasoning_effort": "high"},
            original_payload=route,
            changed=True,
            trace=[{"plugin": "jev-router", "status": "routed", "reason": "economical/high"}],
        )

    monkeypatch.setattr("hermes_cli.middleware.apply_turn_route_middleware", apply_route)
    monkeypatch.setattr(server, "_load_reasoning_config", lambda _model: {"effort": "medium"})
    monkeypatch.setattr(
        server,
        "_resolve_agent_model_runtime_strict",
        lambda override, _provider: (
            str((override or {}).get("model") or "gpt-6-sol"),
            {"provider": "openai-codex", "requested_provider": "openai-codex"},
        ),
    )
    monkeypatch.setattr(server, "_persist_turn_route_state", lambda _session: True)

    binding = server._resolve_initial_turn_route("runtime", session, "route this")

    assert session["create_reasoning_override"] == {"enabled": True, "effort": "high"}
    assert binding["reasoning_effort"] == "high"
    assert binding["reasoning_owner"] == "middleware"
    assert binding["middleware_reason"] == "economical/high"


def test_unchanged_route_retains_plugin_abstention_identity(monkeypatch):
    session = _fresh_session()

    monkeypatch.setattr(
        "hermes_cli.middleware.apply_turn_route_middleware",
        lambda route, **_context: RequestMiddlewareResult(
            payload=dict(route),
            original_payload=route,
            changed=False,
            trace=[{"plugin": "jev-router", "status": "default", "reason": "low_confidence"}],
        ),
    )
    monkeypatch.setattr(server, "_load_reasoning_config", lambda _model: {"effort": "medium"})
    monkeypatch.setattr(
        server,
        "_resolve_agent_model_runtime_strict",
        lambda _override, _provider: (
            "gpt-6-sol",
            {"provider": "openai-codex", "requested_provider": "openai-codex"},
        ),
    )
    monkeypatch.setattr(server, "_persist_turn_route_state", lambda _session: True)

    binding = server._resolve_initial_turn_route("runtime", session, "route this")

    assert binding["status"] == "default"
    assert binding["middleware_plugins"] == ["jev-router"]
    assert binding["middleware_reason"] == "low_confidence"
    assert binding["reasoning_effort"] == "medium"
    assert binding["reasoning_owner"] == "default"


def test_invalid_middleware_reasoning_fails_open(monkeypatch):
    session = _fresh_session()
    monkeypatch.setattr(
        "hermes_cli.middleware.apply_turn_route_middleware",
        lambda route, **_context: RequestMiddlewareResult(
            payload={**route, "model": "gpt-6-luna", "reasoning_effort": "ultra"},
            original_payload=route,
            changed=True,
            trace=[{"plugin": "jev-router", "status": "routed", "reason": "economical/ultra"}],
        ),
    )
    monkeypatch.setattr(server, "_load_reasoning_config", lambda _model: {"effort": "medium"})
    monkeypatch.setattr(
        server,
        "_resolve_agent_model_runtime_strict",
        lambda override, _provider: (
            str((override or {}).get("model") or "gpt-6-sol"),
            {"provider": "openai-codex", "requested_provider": "openai-codex"},
        ),
    )
    monkeypatch.setattr(server, "_persist_turn_route_state", lambda _session: True)

    binding = server._resolve_initial_turn_route("runtime", session, "route this")

    assert binding["status"] == "default"
    assert binding["model"] == "gpt-6-sol"
    assert binding["reasoning_effort"] == "medium"
    assert binding["middleware_plugins"] == []


def test_explicit_reasoning_override_wins_over_middleware(monkeypatch):
    session = _fresh_session()
    session["create_reasoning_override"] = {"enabled": True, "effort": "low"}

    def apply_route(route, **_context):
        assert route["current_reasoning_effort"] == "low"
        return RequestMiddlewareResult(
            payload={**route, "model": "gpt-6-luna", "reasoning_effort": "high"},
            original_payload=route,
            changed=True,
            trace=[{"plugin": "jev-router", "status": "routed", "reason": "economical/high"}],
        )

    monkeypatch.setattr("hermes_cli.middleware.apply_turn_route_middleware", apply_route)
    monkeypatch.setattr(server, "_load_reasoning_config", lambda _model: {"effort": "medium"})
    monkeypatch.setattr(
        server,
        "_resolve_agent_model_runtime_strict",
        lambda override, _provider: (
            str((override or {}).get("model") or "gpt-6-sol"),
            {"provider": "openai-codex", "requested_provider": "openai-codex"},
        ),
    )
    monkeypatch.setattr(server, "_persist_turn_route_state", lambda _session: True)

    binding = server._resolve_initial_turn_route("runtime", session, "route this")

    assert binding["status"] == "routed"
    assert binding["model"] == "gpt-6-luna"
    assert binding["reasoning_effort"] == "low"
    assert binding["reasoning_owner"] == "user"
    assert session["create_reasoning_override"] == {"enabled": True, "effort": "low"}


def test_middleware_can_preserve_profile_reasoning_across_model_change(monkeypatch):
    session = _fresh_session()
    monkeypatch.setattr(
        "hermes_cli.middleware.apply_turn_route_middleware",
        lambda route, **_context: RequestMiddlewareResult(
            payload={**route, "model": "gpt-6-luna", "preserve_reasoning": True},
            original_payload=route,
            changed=True,
            trace=[{"plugin": "jev-router", "status": "routed", "reason": "economical/manual"}],
        ),
    )
    monkeypatch.setattr(server, "_load_reasoning_config", lambda _model: {"effort": "medium"})
    monkeypatch.setattr(
        server,
        "_resolve_agent_model_runtime_strict",
        lambda override, _provider: (
            str((override or {}).get("model") or "gpt-6-sol"),
            {"provider": "openai-codex", "requested_provider": "openai-codex"},
        ),
    )
    monkeypatch.setattr(server, "_persist_turn_route_state", lambda _session: True)

    binding = server._resolve_initial_turn_route("runtime", session, "route this")

    assert binding["model"] == "gpt-6-luna"
    assert binding["reasoning_effort"] == "medium"
    assert binding["reasoning_owner"] == "default"
    assert session["create_reasoning_override"] == {"effort": "medium"}


def test_preprompt_user_binding_write_failure_keeps_pending_state(monkeypatch):
    session = _fresh_session()
    switch = types.SimpleNamespace(
        success=True,
        error_message="",
        warning_message="",
        new_model="gpt-6-astra",
        target_provider="openai-codex",
        base_url="",
        api_key="secret",
        api_mode="responses",
        model_info=None,
    )
    monkeypatch.setattr(server, "_current_model_runtime", lambda *_args: (
        "openai-codex", "gpt-6-sol", "", "secret"))
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kwargs: switch)
    monkeypatch.setattr(server, "_expensive_model_confirm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_resolve_agent_model_runtime_strict",
        lambda override, _provider: (
            str((override or {}).get("model") or "gpt-6-sol"),
            {"provider": "openai-codex", "requested_provider": "openai-codex"},
        ),
    )
    monkeypatch.setattr(server, "_persist_turn_route_state", lambda _session: False)

    with pytest.raises(RuntimeError, match="user turn-route binding"):
        server._apply_model_switch("runtime", session, "gpt-6-astra")

    assert session["turn_route_pending"] is True
    assert session["model_override"] is None
    assert "turn_route_binding" not in session


def test_live_user_binding_write_failure_rolls_back_agent(monkeypatch):
    class _Agent:
        model = "gpt-6-luna"
        provider = "openai-codex"
        requested_provider = "openai-codex"
        base_url = "https://old.example/v1"
        api_key = "old-secret"
        api_mode = "responses"
        reasoning_config = {"effort": "medium"}
        _primary_runtime = None

        def switch_model(self, *, new_model, new_provider, api_key, base_url, api_mode, **_kwargs):
            self.model = new_model
            self.provider = new_provider
            self.requested_provider = new_provider
            self.api_key = api_key
            self.base_url = base_url
            self.api_mode = api_mode

    old_binding = {
        "schema_version": "hermes.turn_route.binding.v1",
        "status": "routed",
        "owner": "middleware",
        "model": "gpt-6-luna",
        "provider": "openai-codex",
        "requested_provider": "openai-codex",
        "middleware_plugins": ["jev-router"],
    }
    agent = _Agent()
    session = {
        "agent": agent,
        "model_override": {"model": "gpt-6-luna", "provider": "openai-codex"},
        "turn_route_binding": copy.deepcopy(old_binding),
        "turn_route_pending": False,
    }
    switch = types.SimpleNamespace(
        success=True,
        error_message="",
        warning_message="",
        new_model="gpt-6-astra",
        target_provider="openai-codex",
        base_url="https://new.example/v1",
        api_key="new-secret",
        api_mode="responses",
        model_info=None,
    )
    monkeypatch.setattr(server, "_current_model_runtime", lambda *_args: (
        "openai-codex", "gpt-6-luna", "https://old.example/v1", "old-secret"))
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kwargs: switch)
    monkeypatch.setattr(server, "_expensive_model_confirm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_persist_turn_route_state", lambda _session: False)

    with pytest.raises(RuntimeError, match="user turn-route binding"):
        server._apply_model_switch("runtime", session, "gpt-6-astra")

    assert agent.model == "gpt-6-luna"
    assert agent.provider == "openai-codex"
    assert agent.base_url == "https://old.example/v1"
    assert session["model_override"] == {
        "model": "gpt-6-luna", "provider": "openai-codex"}
    assert session["turn_route_binding"] == old_binding


@pytest.mark.parametrize(("one_turn", "count_switch"), [(True, True), (False, False)])
def test_one_turn_and_internal_model_switches_preserve_binding(monkeypatch, one_turn, count_switch):
    binding = {
        "schema_version": "hermes.turn_route.binding.v1",
        "status": "routed",
        "owner": "middleware",
        "model": "gpt-6-luna",
        "provider": "openai-codex",
        "requested_provider": "openai-codex",
        "middleware_plugins": ["jev-router"],
    }
    agent = types.SimpleNamespace(model="gpt-6-luna", provider="openai-codex")
    session = {"agent": agent, "turn_route_binding": copy.deepcopy(binding)}
    result = types.SimpleNamespace(
        success=True,
        error_message="",
        warning_message="",
        new_model="gpt-6-sol",
        target_provider="openai-codex",
        base_url="",
        api_key="secret",
        api_mode="responses",
        model_info=None,
    )
    monkeypatch.setattr(
        server,
        "_switch_request",
        lambda *_args, **_kwargs: ("gpt-6-sol", "openai-codex", one_turn, False, None),
    )
    monkeypatch.setattr(
        server,
        "_current_model_runtime",
        lambda *_args: ("openai-codex", "gpt-6-luna", "", "secret"),
    )
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kwargs: result)
    monkeypatch.setattr(server, "_expensive_model_confirm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_commit_agent_switch", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics_events.record_model_switch",
        lambda **_kwargs: None,
    )

    server._apply_model_switch(
        "runtime", session, "gpt-6-sol", count_switch=count_switch)

    assert session["turn_route_binding"] == binding


def test_attachment_lookup_does_not_build_while_route_is_pending(monkeypatch):
    session = _fresh_session()
    server._sessions["runtime-attachment"] = session
    monkeypatch.setattr(
        server,
        "_start_agent_build",
        lambda *_args: pytest.fail("attachment lookup must remain mutation-only while routing is pending"),
    )
    try:
        found, error = server._sess_building(
            {"session_id": "runtime-attachment"}, "attachment")
        dependent, dependent_error = server._sess(
            {"session_id": "runtime-attachment"}, "dependent")
    finally:
        server._sessions.pop("runtime-attachment", None)

    assert found is session and error is None
    assert dependent is None
    assert dependent_error["error"]["code"] == 5036


def test_deferred_resume_runtime_does_not_pass_binding_fields_to_agent(monkeypatch):
    binding = {
        "schema_version": "hermes.turn_route.binding.v1",
        "status": "routed",
        "owner": "middleware",
        "model": "gpt-6-luna",
        "provider": "openai-codex",
        "requested_provider": "openai-codex",
        "middleware_plugins": ["jev-router"],
    }
    session = _fresh_session()
    session["resume_runtime_overrides"] = {
        "model_override": {"model": "gpt-6-luna", "provider": "openai-codex"},
        "provider_override": "openai-codex",
        "turn_route_binding": binding,
        "turn_route_pending": False,
    }
    monkeypatch.setattr(server, "_overrides_have_routable_provider", lambda _overrides: True)

    kwargs = server._deferred_build_agent_kwargs(session, None)

    assert kwargs["model_override"]["model"] == "gpt-6-luna"
    assert "turn_route_binding" not in kwargs
    assert "turn_route_pending" not in kwargs


@pytest.mark.parametrize(("stored_tier", "expected_tier"), [
    ("normal", ""),
    ("priority", "priority"),
])
def test_pending_resume_keeps_reasoning_and_service_tier_on_first_routed_build(
    monkeypatch, stored_tier, expected_tier,
):
    monkeypatch.setattr(server, "_is_routable_provider", lambda provider: bool(provider))
    overrides = server._stored_session_runtime_overrides({
        "model": "creation-default",
        "model_config": json.dumps({
            "provider": "openai-codex",
            "reasoning_config": {"effort": "high"},
            "service_tier": stored_tier,
            "turn_route_pending": True,
        }),
    })
    assert "model_override" not in overrides
    assert overrides["turn_route_pending"] is True

    session = _fresh_session()
    session["resume_runtime_overrides"] = overrides
    session["model_override"] = {
        "model": "gpt-6-luna", "provider": "openai-codex"}
    kwargs = server._deferred_build_agent_kwargs(session, None)

    assert kwargs["model_override"]["model"] == "gpt-6-luna"
    assert kwargs["reasoning_config_override"] == {"effort": "high"}
    assert kwargs["service_tier_override"] == expected_tier


def test_gateway_runtime_mismatch_suppresses_stale_desktop_binding(monkeypatch):
    monkeypatch.setattr(server, "_is_routable_provider", lambda provider: bool(provider))
    stale_binding = {
        "schema_version": "hermes.turn_route.binding.v1",
        "status": "routed",
        "owner": "middleware",
        "model": "gpt-6-luna",
        "provider": "openai-codex",
        "requested_provider": "openai-codex",
        "middleware_plugins": ["jev-router"],
    }
    overrides = server._stored_session_runtime_overrides({
        "model": "gateway-model",
        "model_config": json.dumps({
            "gateway_runtime": {"provider": "nous", "api_mode": "chat_completions"},
            "turn_route_binding": stale_binding,
        }),
    })

    assert overrides["model_override"]["model"] == "gateway-model"
    assert overrides["provider_override"] == "nous"
    assert "turn_route_binding" not in overrides


def test_gateway_runtime_change_invalidates_desktop_binding_before_tui_resume(monkeypatch):
    from gateway.run_turn import GatewayTurnMixin

    row = {
        "model": "gpt-6-luna",
        "model_config": json.dumps({
            "provider": "openai-codex",
            "turn_route_binding": {
                "schema_version": "hermes.turn_route.binding.v1",
                "status": "routed",
                "owner": "middleware",
                "model": "gpt-6-luna",
                "provider": "openai-codex",
                "requested_provider": "openai-codex",
                "middleware_plugins": ["jev-router"],
            },
        }),
    }

    class _Db:
        def get_session(self, _session_id):
            return dict(row)

        def update_session_meta(self, _session_id, model_config, model):
            row["model_config"] = model_config
            row["model"] = model

    runner = object.__new__(GatewayTurnMixin)
    runner._session_db = types.SimpleNamespace(_db=_Db())
    runner._sync_session_model_from_agent(
        "stored-route-1",
        types.SimpleNamespace(
            model="gateway-model",
            provider="nous",
            base_url="https://inference-api.nousresearch.com/v1",
            api_mode="chat_completions",
            _fallback_activated=False,
        ),
    )

    persisted = json.loads(row["model_config"])
    assert row["model"] == "gateway-model"
    assert persisted["gateway_runtime"]["provider"] == "nous"
    assert "turn_route_binding" not in persisted
    monkeypatch.setattr(server, "_is_routable_provider", lambda provider: bool(provider))
    resumed = server._stored_session_runtime_overrides(row)
    assert resumed["model_override"]["model"] == "gateway-model"
    assert resumed["provider_override"] == "nous"
    assert "turn_route_binding" not in resumed


def test_matching_gateway_runtime_still_invalidates_desktop_binding():
    from gateway.run_turn import GatewayTurnMixin

    row = {
        "model": "gpt-6-sol",
        "model_config": json.dumps({
            "gateway_runtime": {
                "provider": "openai-codex",
                "api_mode": "responses",
                "fallback_active": False,
            },
            "turn_route_binding": {
                "schema_version": "hermes.turn_route.binding.v1",
                "status": "routed",
                "owner": "middleware",
                "model": "gpt-6-sol",
                "provider": "openai-codex",
                "requested_provider": "openai-codex",
                "middleware_plugins": ["jev-router"],
            },
        }),
    }

    class _Db:
        def get_session(self, _session_id):
            return dict(row)

        def update_session_meta(self, _session_id, model_config, model):
            row["model_config"] = model_config
            row["model"] = model

    runner = object.__new__(GatewayTurnMixin)
    runner._session_db = types.SimpleNamespace(_db=_Db())
    runner._sync_session_model_from_agent(
        "stored-route-1",
        types.SimpleNamespace(
            model="gpt-6-sol",
            provider="openai-codex",
            base_url=None,
            api_mode="responses",
            _fallback_activated=False,
        ),
    )

    persisted = json.loads(row["model_config"])
    assert "turn_route_binding" not in persisted


@pytest.mark.parametrize(("model_config", "expected"), [
    ({"turn_route_pending": True}, "cold"),
    ({}, "eager"),
])
def test_eager_resume_defers_only_an_explicit_pending_marker(monkeypatch, model_config, expected):
    row = {"id": "stored", "model_config": json.dumps(model_config), "cwd": ""}

    class _Db:
        def close(self):
            return None

    class _Resume:
        def __init__(self, rid, params, target):
            self.rid = rid
            self.params = params
            self.target = target
            self.db = None
            self.owns_db = False
            self.found = None
            self.profile_home = None
            self.profile_resume_cwd = ""
            self.lazy = False
            self.eager_build = True
            self.defer_history = False

    monkeypatch.setattr(server, "_Resume", _Resume)
    monkeypatch.setattr(server, "_profile_session_db", lambda _home: (_Db(), False))
    monkeypatch.setattr(server, "_resume_locate", lambda ctx: setattr(ctx, "found", row) or None)
    monkeypatch.setattr(server, "_resume_follow_tip", lambda _ctx: None)
    monkeypatch.setattr(server, "_resume_guard", lambda _ctx: None)
    monkeypatch.setattr(server, "_profile_workspace_cwd", lambda _home: "")
    monkeypatch.setattr(server, "_find_live_session_by_key", lambda *_args: None)
    monkeypatch.setattr(server, "_resume_cold", lambda _ctx: {"path": "cold"})
    monkeypatch.setattr(server, "_resume_eager", lambda _ctx: {"path": "eager"})

    response = server._methods["session.resume"](
        "resume", {"session_id": "stored", "eager_build": True})

    assert response == {"path": expected}


def test_profile_a_b_a_turn_routes_do_not_leak(monkeypatch):
    active_profiles = []
    routed = []

    @contextlib.contextmanager
    def profile_scope(session):
        active_profiles.append(session.get("profile_home") or "launch")
        try:
            yield
        finally:
            active_profiles.pop()

    def resolve_runtime(override, _provider):
        profile = active_profiles[-1]
        model = str((override or {}).get("model") or f"default-{profile}")
        provider = str((override or {}).get("provider") or f"provider-{profile}")
        return model, {"provider": provider, "requested_provider": provider}

    def apply_route(route, **_context):
        profile = active_profiles[-1]
        model = f"routed-{profile}"
        provider = f"provider-{profile}"
        routed.append(profile)
        return RequestMiddlewareResult(
            payload={
                "model": model,
                "provider": provider,
                "runtime": {"provider": provider, "requested_provider": provider},
            },
            original_payload=route,
            changed=True,
            trace=[{"plugin": f"router-{profile}"}],
        )

    monkeypatch.setattr(server, "_session_profile_runtime_scope", profile_scope)
    monkeypatch.setattr(server, "_resolve_agent_model_runtime_strict", resolve_runtime)
    monkeypatch.setattr("hermes_cli.middleware.apply_turn_route_middleware", apply_route)
    monkeypatch.setattr(server, "_persist_turn_route_state", lambda _session: True)

    results = []
    for profile in ("A", "B", "A"):
        session = _fresh_session()
        session["profile_home"] = profile
        results.append(server._resolve_initial_turn_route(
            f"runtime-{profile}", session, f"prompt-{profile}"))

    assert routed == ["A", "B", "A"]
    assert [item["model"] for item in results] == ["routed-A", "routed-B", "routed-A"]
    assert [item["middleware_plugins"] for item in results] == [
        ["router-A"], ["router-B"], ["router-A"]]
    assert active_profiles == []
