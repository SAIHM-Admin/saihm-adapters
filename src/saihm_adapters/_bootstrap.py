"""Provision the bundled Node sidecar into a per-user cache on first use.

The wheel ships only the tiny ``server.mjs`` / ``sandbox.mjs`` plus a pinned
``package.json`` + ``package-lock.json`` — never ``node_modules``. On first launch this
module materializes a runnable sidecar directory under the user cache and installs the
pinned Node dependencies there once (``npm ci``); later launches reuse it.

Design notes:
- **Reproducible:** ``npm ci`` against the bundled lockfile (exact versions), not ``npm install``.
- **Version-keyed:** the cache dir is namespaced by this package's version, so upgrading the
  Python package provisions a fresh sidecar instead of reusing stale ``node_modules``.
- **Race-safe without a lock dependency:** each caller installs into a private temp dir and
  then ``os.replace``-renames it into place atomically; a loser (target already provisioned)
  simply discards its temp copy. No file locks, no extra dependency.
- **Escape hatches:** ``SAIHM_SIDECAR_DIR`` points at a pre-provisioned dir (air-gapped / CI
  cache) and skips all npm work; ``SAIHM_SKIP_BOOTSTRAP=1`` forbids network install and
  requires the cache to already exist (fail loud rather than silently reach the registry).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from importlib import resources
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

_BUNDLE_FILES = ("server.mjs", "sandbox.mjs", "package.json", "package-lock.json")
_READY = ".saihm-sidecar-ready"
_DEFAULT_INSTALL_TIMEOUT = 600  # seconds; a cold `npm ci` on a slow link is minutes, not hours


def _truthy(name: str) -> bool:
    """Parse a boolean env var permissively.

    Deliberately generous: these flags gate *whether we reach the network*, so anything a
    user plausibly writes to mean "yes" ("1", "true", "yes", "on", "Y") must be honored.
    Matching only ("1", "true", "TRUE") would silently fail open — the user asks us not to
    install and we install anyway.
    """
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on", "y")


def _install_timeout() -> int:
    raw = os.environ.get("SAIHM_BOOTSTRAP_TIMEOUT", "").strip()
    try:
        val = int(raw)
        return val if val > 0 else _DEFAULT_INSTALL_TIMEOUT
    except ValueError:
        return _DEFAULT_INSTALL_TIMEOUT


def _package_version() -> str:
    try:
        return _pkg_version("saihm-adapters")
    except PackageNotFoundError:  # running from a source checkout, not an installed dist
        return "src"


def _cache_root() -> Path:
    """Per-user cache root, honoring XDG_CACHE_HOME; version-namespaced by the caller."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(Path.home(), ".cache")
    return Path(base) / "saihm-adapters"


def _bundle_dir() -> Path:
    """Directory of the bundled sidecar sources inside the installed package."""
    return Path(str(resources.files("saihm_adapters"))) / "_sidecar"


def _looks_ready(d: Path) -> bool:
    return (d / _READY).is_file() and (d / "server.mjs").is_file() and (d / "node_modules").is_dir()


def _provision(dest: Path, allow_install: bool) -> None:
    """Populate a fresh temp dir with the bundle + installed node_modules, then atomically
    move it to ``dest``. Idempotent under concurrency via os.replace."""
    if not allow_install:
        raise RuntimeError(
            "SAIHM sidecar is not provisioned and SAIHM_SKIP_BOOTSTRAP is set. "
            f"Provision {dest} out-of-band (npm ci) or unset SAIHM_SKIP_BOOTSTRAP."
        )
    npm = shutil.which("npm")
    node = shutil.which("node")
    if not npm or not node:
        raise RuntimeError(
            "The saihm-adapters sidecar needs Node.js (node + npm) on PATH — install Node >=20 "
            "(https://nodejs.org). Python performs no cryptography; the Node sidecar seals every cell."
        )
    src = _bundle_dir()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".tmp-{dest.name}-", dir=str(dest.parent)))
    try:
        for name in _BUNDLE_FILES:
            shutil.copy2(src / name, tmp / name)
        # npm ci = clean, lockfile-exact, reproducible; prod deps only.
        # Bounded: an unreachable/hanging registry must fail loudly rather than wedge the
        # caller's first client call forever with no output.
        try:
            proc = subprocess.run(
                [npm, "ci", "--omit=dev", "--no-audit", "--no-fund", "--loglevel=error"],
                cwd=str(tmp), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                timeout=_install_timeout(),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Timed out after {_install_timeout()}s installing the SAIHM Node sidecar "
                "dependencies (npm ci). Check network/registry access, or pre-provision the "
                "sidecar and point SAIHM_SIDECAR_DIR at it. Raise the limit with "
                "SAIHM_BOOTSTRAP_TIMEOUT (seconds)."
            ) from exc
        if proc.returncode != 0:
            raise RuntimeError(
                "Failed to install the SAIHM Node sidecar dependencies "
                f"(npm ci exit {proc.returncode}):\n{proc.stdout.strip()[-2000:]}"
            )
        (tmp / _READY).write_text("ok\n", encoding="utf-8")
        try:
            os.replace(tmp, dest)  # atomic on the same filesystem
            tmp = None  # ownership transferred
        except OSError:
            # Another process won the race and populated dest first — fine if it's ready.
            if not _looks_ready(dest):
                raise
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def ensure_sidecar() -> str:
    """Return the absolute path to a runnable ``server.mjs`` whose ``node_modules`` are present,
    provisioning it into the per-user cache on first use. Idempotent and concurrency-safe."""
    override = os.environ.get("SAIHM_SIDECAR_DIR")
    if override:
        server = Path(override).expanduser().resolve() / "server.mjs"
        if not server.is_file():
            raise RuntimeError(f"SAIHM_SIDECAR_DIR={override} has no server.mjs")
        return str(server)

    dest = _cache_root() / _package_version()
    if not _looks_ready(dest):
        _provision(dest, allow_install=not _truthy("SAIHM_SKIP_BOOTSTRAP"))
    return str(dest / "server.mjs")
