"""LangChain + LlamaIndex adapter fixtures. Inherits ``client`` / ``_wipe`` from the parent conftest.

The LlamaIndex ``SaihmMemory`` is imported from its submodule (``saihm_adapters.llamaindex_memory``),
not the top level, because the AutoGen adapter defines a different class of the same name.
"""
from __future__ import annotations

import pytest

from saihm_adapters import SaihmChatMessageHistory
from saihm_adapters.llamaindex_memory import SaihmMemory


@pytest.fixture
def history(client):
    # LangChain adapter; client passed in => it does not own/close the shared client.
    return SaihmChatMessageHistory(client=client)


@pytest.fixture
def memory(client):
    # LlamaIndex adapter; client passed in => it does not own/close the shared client.
    return SaihmMemory.from_defaults(client=client)
