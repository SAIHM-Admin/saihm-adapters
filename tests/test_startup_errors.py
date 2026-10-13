"""Start-up failures reach the caller as typed errors carrying the sidecar's own message.

Live mode is pointed at an unreachable local port and HOME at an empty directory, so the sidecar
finds no identity and stops before any request leaves the machine.
"""
from __future__ import annotations

import shutil
import warnings

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
    text = str(err)
    assert text.index("free-join") < text.index(err.message)  # Python steps first
    assert err.__cause__ is None and err.__suppress_context__  # no ExceptionGroup traceback


def test_unreadable_key_file_raises_startup_error(live_env, tmp_path):
    live_env["SAIHM_MASTER_SECRET_FILE"] = str(tmp_path / "missing.key")
    with pytest.raises(SaihmStartupError) as exc:
        SaihmMemoryClient(env=live_env, timeout=30)
    assert not isinstance(exc.value, SaihmNoIdentityError)
    assert "SAIHM_MASTER_SECRET_FILE" in exc.value.message


def _bundle_copy(dest):
    from saihm_adapters._bootstrap import _BUNDLE_FILES, _bundle_dir
    dest.mkdir()
    for name in _BUNDLE_FILES:
        shutil.copy2(_bundle_dir() / name, dest / name)
    return dest


def test_dependency_load_failure_raises_startup_error(tmp_path, monkeypatch):
    # The bundle without node_modules: the sidecar's dependencies cannot load.
    monkeypatch.setenv("SAIHM_SIDECAR_DIR", str(_bundle_copy(tmp_path / "sidecar")))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # same lockfile as the bundle: no staleness warning
        with pytest.raises(SaihmStartupError) as exc:
            SaihmMemoryClient(env={"HOME": str(tmp_path)}, timeout=30)
    assert exc.value.kind == "startup"
    assert "@modelcontextprotocol/sdk" in exc.value.message
    assert exc.value.__cause__ is None  # reported by the sidecar itself, not the output fallback


def test_stale_sidecar_dir_warns(tmp_path, monkeypatch):
    from saihm_adapters._bootstrap import ensure_sidecar
    d = _bundle_copy(tmp_path / "sidecar")
    (d / "package-lock.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SAIHM_SIDECAR_DIR", str(d))
    with pytest.warns(RuntimeWarning, match="npm ci --omit=dev"):
        assert ensure_sidecar() == str((d / "server.mjs").resolve())


def test_sandbox_writes_nothing_under_saihm_home(tmp_path, capfd, monkeypatch):
    monkeypatch.delenv("SAIHM_ENDPOINT_URL", raising=False)
    with SaihmMemoryClient(env={"HOME": str(tmp_path)}, timeout=60) as c:
        cid = c.remember("sandbox only")
        assert [m.text for m in c.recall()] == ["sandbox only"]
        assert c.forget(cid) is True
    assert not (tmp_path / ".saihm").exists()
    assert "recall is fetching" not in capfd.readouterr().err
