"""A call the service refuses must raise, on every mcp release this package admits.

mcp 1.x reports a refused tool call as ``isError``; mcp 2.x renamed it ``is_error``. A client
that reads only one name passes every refusal off as a success: writes are lost, recall comes
back empty and a refused erasure reads as "nothing to erase". These tests drive the real
sidecar against a local stand-in for the hosted endpoint (``fake_endpoint``), so the refusal
travels the whole way through the installed mcp release.
"""
from __future__ import annotations

import importlib.metadata
import time

import pytest
from mcp import types

from saihm_adapters import SaihmMemoryClient, SaihmNoIdentityError, SaihmStartupError
from saihm_adapters.client import _call_failed

MCP_VERSION = importlib.metadata.version("mcp")
HOST_CHAT = ("Tell the operator", "saihm_join", "in the chat", "Join SAIHM")


def test_error_flag_is_read_on_the_installed_mcp():
    refused = types.CallToolResult.model_validate(
        {"content": [{"type": "text", "text": "refused"}], "isError": True}
    )
    accepted = types.CallToolResult.model_validate({"content": [{"type": "text", "text": "ok"}]})
    assert _call_failed(refused) is True, MCP_VERSION
    assert _call_failed(accepted) is False, MCP_VERSION


def test_result_without_an_error_flag_fails_loudly():
    class Unknown:
        content = []

    with pytest.raises(RuntimeError, match="no error flag"):
        _call_failed(Unknown())


def test_either_error_flag_marks_a_refusal():
    # mcp 1.x results accept extra fields, so a result can carry both names. A false value under
    # one name must never hide a true value under the other.
    class Both:
        content = []
        is_error = False
        isError = True

    class OneUnset:
        content = []
        is_error = None
        isError = False

    assert _call_failed(Both()) is True
    assert _call_failed(OneUnset()) is False


@pytest.fixture
def refusing_client(fake_endpoint, clean_identity_env, make_key, tmp_path):
    """A live client whose every memory call the endpoint refuses with a hard quota cap."""
    fake_endpoint.call_reply = (429, {"error": "quota_hard_cap"})
    env = {
        "HOME": str(tmp_path),
        "SAIHM_ENDPOINT_URL": fake_endpoint.url,
        "SAIHM_MASTER_SECRET_FILE": str(make_key(tmp_path / "keys" / "test.key")),
        "SAIHM_AUTH_HEADER": "Bearer test",  # a fixed header: no sign-in round trip
        "SAIHM_RECALL_CACHE": "0",
    }
    c = SaihmMemoryClient(env=env, timeout=60)
    try:
        yield c
    finally:
        c.close()


def test_refused_remember_raises(refusing_client):
    with pytest.raises(RuntimeError, match="quota_hard_cap"):
        refusing_client.remember("must not be lost in silence")


def test_refused_recall_raises_instead_of_returning_empty(refusing_client):
    with pytest.raises(RuntimeError, match="quota_hard_cap"):
        refusing_client.recall()


def test_refused_forget_raises_instead_of_returning_false(refusing_client):
    with pytest.raises(RuntimeError, match="quota_hard_cap"):
        refusing_client.forget("0" * 32)
    with pytest.raises(RuntimeError, match="quota_hard_cap"):
        refusing_client._forget_raw("0" * 32)


def test_refused_status_raises(refusing_client):
    with pytest.raises(RuntimeError, match="quota_hard_cap"):
        refusing_client.status()


def _live_env(tmp_path, url):
    return {"HOME": str(tmp_path), "SAIHM_ENDPOINT_URL": url}


def test_unfinished_free_join_raises_no_identity(fake_endpoint, clean_identity_env, make_key, tmp_path):
    # free-join saves the key before the sign-in is confirmed; until it is, sign-in is refused.
    make_key(tmp_path / ".saihm" / "free-identity.key")
    fake_endpoint.onboard_reply = (401, {"error": "verification_failed", "reason": "no_free_entitlement"})
    with pytest.raises(SaihmNoIdentityError) as exc:
        SaihmMemoryClient(env=_live_env(tmp_path, fake_endpoint.url), timeout=60)
    err = exc.value
    assert isinstance(err, SaihmStartupError)
    assert (err.kind, err.reason) == ("identity-not-active", "no_free_entitlement")
    assert "no_free_entitlement" in err.message  # the service's own text stays available
    text = str(err)
    assert "not active yet" in text and "free-join" in text and "reuses your key" in text
    assert not any(phrase in text for phrase in HOST_CHAT), text
    assert err.__cause__ is None and err.__suppress_context__
    assert ("POST", "/api/onboard") in fake_endpoint.seen  # it asked the service; no guess


def test_inactive_subscription_raises_no_identity(fake_endpoint, clean_identity_env, make_key, tmp_path):
    make_key(tmp_path / ".saihm" / "free-identity.key")
    fake_endpoint.onboard_reply = (401, {"error": "verification_failed", "reason": "no_active_subscription"})
    with pytest.raises(SaihmNoIdentityError) as exc:
        SaihmMemoryClient(env=_live_env(tmp_path, fake_endpoint.url), timeout=60)
    assert exc.value.reason == "no_active_subscription"
    assert "SAIHM_TIER" in str(exc.value)


def test_active_identity_starts_after_one_sign_in(fake_endpoint, clean_identity_env, make_key, tmp_path):
    # The start-up check signs in once; the calls that follow reuse that sign-in.
    make_key(tmp_path / ".saihm" / "free-identity.key")
    fake_endpoint.onboard_reply = (200, {"jwt": "test-token"})
    fake_endpoint.call_reply = (200, {"tier": "FREE", "activeShardCount": 0, "custody": "non-custodial"})
    env = {**_live_env(tmp_path, fake_endpoint.url), "SAIHM_RECALL_CACHE": "0"}
    with SaihmMemoryClient(env=env, timeout=60) as c:
        assert fake_endpoint.seen.count(("POST", "/api/onboard")) == 1
        assert fake_endpoint.seen.count(("POST", "/mcp")) == 1  # the start-up status check
        assert c.status() == "tier=FREE memories=0 custody=non-custodial"
    assert fake_endpoint.seen.count(("POST", "/api/onboard")) == 1


def _refusal(reason, delay_s=None):
    body = {"error": "verification_failed", "reason": reason}
    return (401, body) if delay_s is None else (401, body, delay_s)


# The refusals about the one-time challenge, as the service sends them.
CHALLENGE_CODES = ("unknown_nonce", "nonce_expired", "nonce_replay")


@pytest.mark.parametrize("code", CHALLENGE_CODES)
def test_challenge_refusal_gets_one_more_sign_in(code, fake_endpoint, clean_identity_env, make_key, tmp_path):
    # The service lost its one-time challenge, for example in a restart between the two sign-in
    # steps. That says nothing about the identity, so start-up signs in once more.
    make_key(tmp_path / ".saihm" / "free-identity.key")
    fake_endpoint.onboard_reply = [_refusal(code), (200, {"jwt": "test-token"})]
    fake_endpoint.call_reply = (200, {"tier": "FREE", "activeShardCount": 0, "custody": "non-custodial"})
    env = {**_live_env(tmp_path, fake_endpoint.url), "SAIHM_RECALL_CACHE": "0"}
    with SaihmMemoryClient(env=env, timeout=60) as c:
        assert fake_endpoint.seen.count(("POST", "/api/onboard")) == 2
        assert c.status() == "tier=FREE memories=0 custody=non-custodial"


@pytest.mark.parametrize("code", CHALLENGE_CODES)
def test_challenge_refused_twice_does_not_block_start_up(code, fake_endpoint, clean_identity_env, make_key, tmp_path):
    make_key(tmp_path / ".saihm" / "free-identity.key")
    fake_endpoint.onboard_reply = _refusal(code)
    env = {**_live_env(tmp_path, fake_endpoint.url), "SAIHM_RECALL_CACHE": "0"}
    with SaihmMemoryClient(env=env, timeout=60) as c:
        assert fake_endpoint.seen.count(("POST", "/api/onboard")) == 2  # one more, then go ahead
        with pytest.raises(RuntimeError, match=code):
            c.remember("x")


@pytest.mark.parametrize("code", CHALLENGE_CODES)
def test_hard_refusal_on_the_second_sign_in_still_raises(code, fake_endpoint, clean_identity_env, make_key, tmp_path):
    # The one more sign-in is a real check: refused for a reason about the identity, start-up
    # stops with the typed error, after exactly two sign-ins.
    make_key(tmp_path / ".saihm" / "free-identity.key")
    fake_endpoint.onboard_reply = [_refusal(code), _refusal("no_free_entitlement")]
    with pytest.raises(SaihmNoIdentityError) as exc:
        SaihmMemoryClient(env=_live_env(tmp_path, fake_endpoint.url), timeout=60)
    assert (exc.value.kind, exc.value.reason) == ("identity-not-active", "no_free_entitlement")
    assert fake_endpoint.seen.count(("POST", "/api/onboard")) == 2


def test_late_challenge_refusal_shares_the_20_s_limit(fake_endpoint, clean_identity_env, make_key, tmp_path):
    # The second sign-in gets what is left of the 20 s, not 20 s of its own: refused at 10 s,
    # then a sign-in that never answers, and the client still starts near 20 s (not 30 s).
    make_key(tmp_path / ".saihm" / "free-identity.key")
    fake_endpoint.onboard_reply = [_refusal("unknown_nonce", delay_s=10), fake_endpoint.HANG]
    env = {**_live_env(tmp_path, fake_endpoint.url), "SAIHM_RECALL_CACHE": "0"}
    began = time.monotonic()
    with SaihmMemoryClient(env=env, timeout=60):
        waited = time.monotonic() - began
    assert 19 <= waited < 27, waited
    assert fake_endpoint.seen.count(("POST", "/api/onboard")) == 2


def test_401_after_a_good_sign_in_does_not_block_start_up(fake_endpoint, clean_identity_env, make_key, tmp_path):
    # Only a refused sign-in says the identity is not active. A memory call answered 401 after a
    # good sign-in lets the client start, and the calls report it.
    make_key(tmp_path / ".saihm" / "free-identity.key")
    fake_endpoint.onboard_reply = (200, {"jwt": "test-token"})
    fake_endpoint.call_reply = (401, {"error": "unauthorized"})
    env = {**_live_env(tmp_path, fake_endpoint.url), "SAIHM_RECALL_CACHE": "0"}
    with SaihmMemoryClient(env=env, timeout=60) as c:
        with pytest.raises(RuntimeError, match="401|unauthorized"):
            c.status()


def test_settings_refusal_names_its_reason_code(fake_endpoint, clean_identity_env, make_key, tmp_path):
    # A paid-plan setting the service rejects is not a free-join problem: the reason code leads.
    make_key(tmp_path / ".saihm" / "free-identity.key")
    fake_endpoint.onboard_reply = _refusal("bad_tier")
    with pytest.raises(SaihmNoIdentityError) as exc:
        SaihmMemoryClient(env=_live_env(tmp_path, fake_endpoint.url), timeout=60)
    err = exc.value
    assert (err.kind, err.reason) == ("identity-not-active", "bad_tier")
    text = str(err)
    assert "reason code: bad_tier" in text and "SAIHM_TIER" in text
    assert not any(phrase in text for phrase in HOST_CHAT), text
    assert fake_endpoint.seen.count(("POST", "/api/onboard")) == 1  # no second sign-in for it


def test_endpoint_that_never_answers_delays_start_up_by_at_most_20_s(
    fake_endpoint, clean_identity_env, make_key, tmp_path
):
    # The start-up check gives up after 20 s and the client starts. The sign-in request itself
    # would give up only after 30 s, so a start-up near 20 s shows the cap at work.
    make_key(tmp_path / ".saihm" / "free-identity.key")
    fake_endpoint.onboard_reply = fake_endpoint.HANG
    env = {**_live_env(tmp_path, fake_endpoint.url), "SAIHM_RECALL_CACHE": "0"}
    began = time.monotonic()
    with SaihmMemoryClient(env=env, timeout=60):
        waited = time.monotonic() - began
    assert 19 <= waited < 27, waited
    assert fake_endpoint.seen.count(("POST", "/api/onboard")) == 1


def test_other_sign_in_failures_do_not_block_start_up(fake_endpoint, clean_identity_env, make_key, tmp_path):
    # Only the service's typed "this identity is not active" stops start-up. A server error
    # lets the client start as before, and each call then reports it.
    make_key(tmp_path / ".saihm" / "free-identity.key")
    fake_endpoint.onboard_reply = (503, {"error": "unavailable"})
    env = {**_live_env(tmp_path, fake_endpoint.url), "SAIHM_RECALL_CACHE": "0"}
    with SaihmMemoryClient(env=env, timeout=60) as c:
        assert fake_endpoint.seen.count(("POST", "/api/onboard")) == 1  # no second sign-in for it
        with pytest.raises(RuntimeError, match="503"):
            c.remember("x")


def test_unreachable_endpoint_does_not_block_start_up(clean_identity_env, make_key, tmp_path):
    make_key(tmp_path / ".saihm" / "free-identity.key")
    env = {**_live_env(tmp_path, "http://127.0.0.1:9/mcp"), "SAIHM_RECALL_CACHE": "0"}
    with SaihmMemoryClient(env=env, timeout=60) as c:
        with pytest.raises(RuntimeError, match="could not (be )?reach"):
            c.recall()
