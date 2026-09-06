"""AutoGen adapter fixtures. Inherits the shared ``client`` / ``_wipe`` from the parent conftest."""
from __future__ import annotations

import pytest

from saihm_adapters.autogen_memory import SaihmMemory


@pytest.fixture
def memory(client):
    # AutoGen adapter; client passed in => the memory does not own/close it (session fixture does).
    return SaihmMemory(client=client)
