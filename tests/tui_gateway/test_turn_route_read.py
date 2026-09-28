"""Addressed reads of host-owned Desktop turn-route bindings."""

from __future__ import annotations

import pytest

from tui_gateway import server


def test_session_turn_route_read_is_addressed_detached_and_side_effect_free(monkeypatch):
    binding = {
        "schema_version": "hermes.turn_route.binding.v1",
        "status": "routed",
        "owner": "middleware",
        "model": "gpt-6-luna",
        "provider": "openai-codex",
        "requested_provider": "openai-codex",
        "middleware_plugins": ["jev-router"],
        "private_plugin_metadata": "must not cross the boundary",
    }
    session = {
        "session_key": "stored-route-1",
        "profile_home": None,
        "turn_route_binding": binding,
        "agent": None,
    }
    server._sessions["runtime-route-1"] = session
    monkeypatch.setattr(
        server,
        "_start_agent_build",
        lambda *_args: (_ for _ in ()).throw(AssertionError("read must not build an agent")),
    )
    monkeypatch.setattr(
        server,
        "_session_db",
        lambda *_args: (_ for _ in ()).throw(AssertionError("read must not open session storage")),
    )

    try:
        response = server._methods["session.turn_route.read"](
            "request-1",
            {"session_id": "runtime-route-1", "stored_session_id": "stored-route-1"},
        )
        binding["model"] = "mutated-after-read"
    finally:
        server._sessions.pop("runtime-route-1", None)

    assert response["result"] == {
        "schema_version": "hermes.turn_route.binding.v1",
        "session_id": "runtime-route-1",
        "stored_session_id": "stored-route-1",
        "evidence": "session_binding",
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


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"stored_session_id": "stored"}, "session_id is required"),
        ({"session_id": "runtime"}, "stored_session_id is required"),
        ({"session_id": 7, "stored_session_id": "stored"}, "session_id is required"),
        ({"session_id": "runtime", "stored_session_id": []}, "stored_session_id is required"),
    ],
)
def test_session_turn_route_read_requires_both_string_identities(params, message):
    response = server._methods["session.turn_route.read"]("request", params)
    assert response["error"]["message"] == message


def test_session_turn_route_read_rejects_stale_or_mismatched_identity():
    stale = server._methods["session.turn_route.read"](
        "stale", {"session_id": "missing", "stored_session_id": "stored"})
    assert stale["error"]["code"] == 4001

    server._sessions["runtime"] = {"session_key": "stored", "profile_home": None}
    try:
        mismatch = server._methods["session.turn_route.read"](
            "mismatch", {"session_id": "runtime", "stored_session_id": "other"})
    finally:
        server._sessions.pop("runtime", None)
    assert mismatch["error"]["code"] == 4007


def test_session_turn_route_read_reports_pending_and_legacy_unrecorded():
    server._sessions.update({
        "pending-runtime": {
            "session_key": "pending-stored", "profile_home": None, "turn_route_pending": True},
        "legacy-runtime": {"session_key": "legacy-stored", "profile_home": None},
    })
    try:
        pending = server._methods["session.turn_route.read"](
            "pending", {"session_id": "pending-runtime", "stored_session_id": "pending-stored"})
        legacy = server._methods["session.turn_route.read"](
            "legacy", {"session_id": "legacy-runtime", "stored_session_id": "legacy-stored"})
    finally:
        server._sessions.pop("pending-runtime", None)
        server._sessions.pop("legacy-runtime", None)

    assert pending["result"]["status"] == "pending"
    assert legacy["result"]["status"] == "unrecorded"


def test_session_turn_route_read_rejects_the_wrong_explicit_profile(monkeypatch):
    server._sessions["runtime"] = {
        "session_key": "stored", "profile_home": "/profiles/right", "turn_route_pending": True}
    monkeypatch.setattr(
        server, "_profile_home",
        lambda *_args: pytest.fail("an addressed read must not activate a profile"),
    )
    monkeypatch.setattr(
        server, "profile_name_for_home",
        lambda home: "right" if home == "/profiles/right" else None,
    )
    try:
        response = server._methods["session.turn_route.read"](
            "profile",
            {"session_id": "runtime", "stored_session_id": "stored", "profile": "wrong"},
        )
    finally:
        server._sessions.pop("runtime", None)
    assert response["error"]["code"] == 4007


def test_session_turn_route_read_accepts_matching_profile_without_activation(monkeypatch):
    server._sessions["runtime"] = {
        "session_key": "stored", "profile_home": "/profiles/right", "turn_route_pending": True}
    monkeypatch.setattr(
        server, "_profile_home",
        lambda *_args: pytest.fail("an addressed read must not activate a profile"),
    )
    monkeypatch.setattr(
        server, "profile_name_for_home",
        lambda home: "right" if home == "/profiles/right" else None,
    )
    try:
        response = server._methods["session.turn_route.read"](
            "profile",
            {"session_id": "runtime", "stored_session_id": "stored", "profile": "right"},
        )
    finally:
        server._sessions.pop("runtime", None)
    assert response["result"]["status"] == "pending"
