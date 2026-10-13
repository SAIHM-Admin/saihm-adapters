"""Start-up failures reach the caller as typed errors carrying the sidecar's own message.

Live mode is pointed at an unreachable local port and HOME at an empty directory, so the sidecar
finds no identity and stops before any request leaves the machine.
"""
from __future__ import annotations

import pytest

from saihm_adapters import SaihmMemoryClient, SaihmNoIdentityError, SaihmStartupError

_IDENTITY_VARS = (
    "SAIHM_IDENTITY", "SAIHM_IDENTITY_PASSPHRASE", "SAIHM_MASTER_SECRET_FILE",
    "SAIHM_MASTER_SECRET_HEX", "SAIHM_HOME", "SAIHM_SELF_JOIN",
)


@pytest.fixture
def live_env(tmp_path, monkeypatch):
    for name in _IDENTITY_VARS:
        monkeypatch.delenv(name, raising=False)
    return {"HOME": str(tmp_path), "SAIHM_ENDPOINT_URL": "http://127.0.0.1:9/mcp"}


def test_no_identity_raises_typed_error_with_server_message(live_env):
    with pytest.raises(SaihmNoIdentityError) as exc:
        SaihmMemoryClient(env=live_env, timeout=30)
    err = exc.value
    assert isinstance(err, SaihmStartupError) and isinstance(err, RuntimeError)
    assert err.kind == "no-identity"
    assert err.message.startswith("No SAIHM memory yet on this device")
    assert "free-join" in str(err)  # the Python-side steps are appended
    assert err.__cause__ is None and err.__suppress_context__  # no ExceptionGroup traceback


def test_unreadable_key_file_raises_startup_error(live_env, tmp_path):
    live_env["SAIHM_MASTER_SECRET_FILE"] = str(tmp_path / "missing.key")
    with pytest.raises(SaihmStartupError) as exc:
        SaihmMemoryClient(env=live_env, timeout=30)
    assert not isinstance(exc.value, SaihmNoIdentityError)
    assert "SAIHM_MASTER_SECRET_FILE" in exc.value.message
