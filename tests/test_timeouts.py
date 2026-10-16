"""A call that runs out of time raises SaihmTimeout on every supported Python.

On Python 3.10 a future's timeout is its own class (``concurrent.futures.TimeoutError``), not
the built-in ``TimeoutError``, so a client that catches only the built-in one raises
``RuntimeError`` there. The CI leg on 3.10 is the one that can catch that regression.
"""
from __future__ import annotations

import re

import pytest

from saihm_adapters import SaihmMemoryClient, SaihmTimeout


@pytest.fixture
def hasty_client(tmp_path, monkeypatch):
    """A sandbox client whose calls get no time at all. Start-up does not use the timeout."""
    monkeypatch.delenv("SAIHM_ENDPOINT_URL", raising=False)
    c = SaihmMemoryClient(env={"HOME": str(tmp_path)}, timeout=1e-9)
    try:
        yield c
    finally:
        c.close()


def test_timed_out_call_raises_saihm_timeout(hasty_client):
    with pytest.raises(SaihmTimeout):
        hasty_client.recall()


def test_timed_out_remember_keeps_its_cell_id(hasty_client):
    with pytest.raises(SaihmTimeout) as exc:
        hasty_client.remember("this write may still land")
    assert re.fullmatch(r"[0-9a-f]{32}", exc.value.cell_id or ""), exc.value.cell_id
