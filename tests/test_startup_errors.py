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


def test_no_identity_raises_typed_error_written_for_developers(live_env):
    with pytest.raises(SaihmNoIdentityError) as exc:
        SaihmMemoryClient(env=live_env, timeout=30)
    err = exc.value
    assert isinstance(err, SaihmStartupError) and isinstance(err, RuntimeError)
    assert err.kind == "no-identity" and err.reason == ""
    assert err.message.startswith("No SAIHM memory yet on this device")  # kept for logs
    text = str(err)
    assert "free-join" in text and "SAIHM_MASTER_SECRET_FILE" in text and "sandbox" in text
    # The sidecar's message is written for MCP hosts (editors); a developer never sees it.
    for phrase in ("Tell the operator", "saihm_join", "in the chat", "Join SAIHM"):
        assert phrase not in text, phrase
    assert err.__cause__ is None and err.__suppress_context__  # no ExceptionGroup traceback


def test_no_identity_with_self_join_off_names_the_key_file_setting(live_env):
    live_env["SAIHM_SELF_JOIN"] = "0"
    with pytest.raises(SaihmNoIdentityError) as exc:
        SaihmMemoryClient(env=live_env, timeout=30)
    assert "SAIHM_SELF_JOIN=0 is set" in str(exc.value)


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


def test_missing_node_raises_startup_error(tmp_path):
    with pytest.raises(SaihmStartupError) as exc:
        SaihmMemoryClient(node="saihm-no-such-node", timeout=30)
    assert exc.value.kind == "node-missing" and "Node.js >= 20" in str(exc.value)
    with pytest.raises(SaihmStartupError) as exc:
        SaihmMemoryClient(env={"PATH": str(tmp_path)}, timeout=30)  # a PATH without node
    assert exc.value.kind == "node-missing"


def test_missing_node_at_install_raises_startup_error(tmp_path, monkeypatch):
    import saihm_adapters._bootstrap as bootstrap
    monkeypatch.delenv("SAIHM_SIDECAR_DIR", raising=False)
    monkeypatch.delenv("SAIHM_SKIP_BOOTSTRAP", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))  # nothing provisioned here yet
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name, *a, **k: None)
    with pytest.raises(SaihmStartupError) as exc:
        SaihmMemoryClient(timeout=30)
    assert exc.value.kind == "node-missing"


def test_skip_bootstrap_without_a_sidecar_raises_startup_error(tmp_path, monkeypatch):
    monkeypatch.delenv("SAIHM_SIDECAR_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("SAIHM_SKIP_BOOTSTRAP", "1")
    with pytest.raises(SaihmStartupError) as exc:
        SaihmMemoryClient(timeout=30)
    assert exc.value.kind == "not-provisioned" and "SAIHM_SKIP_BOOTSTRAP" in str(exc.value)


def test_sidecar_dir_without_a_sidecar_raises_startup_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SAIHM_SIDECAR_DIR", str(tmp_path))
    with pytest.raises(SaihmStartupError) as exc:
        SaihmMemoryClient(timeout=30)
    assert exc.value.kind == "not-provisioned"


def test_unwritable_cache_raises_startup_error(tmp_path, monkeypatch):
    monkeypatch.delenv("SAIHM_SIDECAR_DIR", raising=False)
    monkeypatch.delenv("SAIHM_SKIP_BOOTSTRAP", raising=False)
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setenv("XDG_CACHE_HOME", str(blocker))  # the cache root cannot be created
    with pytest.raises(SaihmStartupError) as exc:
        SaihmMemoryClient(timeout=30)
    assert exc.value.kind == "install-failed" and exc.value.__cause__ is not None


def test_silent_sidecar_exit_raises_startup_error(tmp_path):
    quiet = tmp_path / "quiet.mjs"
    quiet.write_text("process.exit(3);\n", encoding="utf-8")
    with pytest.raises(SaihmStartupError) as exc:
        SaihmMemoryClient(server_path=str(quiet), timeout=30)
    assert exc.value.kind == "startup" and exc.value.__cause__ is not None
