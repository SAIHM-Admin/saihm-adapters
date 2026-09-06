"""RAG retriever fixtures. Inherits the shared ``client`` / ``_wipe`` from the parent conftest."""
from __future__ import annotations

import pytest

from saihm_adapters import SaihmRetriever


@pytest.fixture
def retriever(client):
    # client passed in => the retriever does not own/close it (session fixture handles that).
    return SaihmRetriever(client=client, similarity_top_k=3)
