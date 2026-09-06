"""Shared test fixtures for the offline blind sandbox — framework-agnostic (core client only).

Everything runs against the OFFLINE blind sandbox (no account, no live endpoint): a fresh
SaihmMemoryClient with no SAIHM_ENDPOINT_URL spins up a local in-process blind endpoint via the
bundled Node sidecar. One sandbox client is shared for the whole session (a Node subprocess
spawn is not free); each test starts from a clean slate via the autouse ``_wipe`` fixture.

Per-framework fixtures (memory / history / retriever / backend / …) live in the conftest of each
``tests/<framework>/`` subdirectory, so importing this module needs only the core client — not
any framework — and the two frameworks that both name a class ``SaihmMemory`` never collide here.
"""
from __future__ import annotations

import pytest

from saihm_adapters import SaihmMemoryClient


@pytest.fixture(scope="session")
def client():
    c = SaihmMemoryClient()  # sandbox mode (no env) -> local blind endpoint
    try:
        yield c
    finally:
        c.close()


@pytest.fixture(autouse=True)
def _wipe(client):
    """Erase every cell before each test so the shared sandbox is isolated per-test."""
    for m in client.recall():
        client._forget_raw(m.cell_id)
    yield
