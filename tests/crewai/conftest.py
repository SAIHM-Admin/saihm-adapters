"""Test fixtures. Everything runs against the OFFLINE blind sandbox (no account, no live
endpoint): a fresh SaihmMemoryClient with no SAIHM_ENDPOINT_URL spins up a local in-process
blind endpoint via the bundled Node sidecar.

One sandbox client is shared for the whole session (a Node subprocess spawn is not free); each
test starts from a clean slate via the autouse ``_wipe`` fixture.

Targets the CURRENT CrewAI memory API (``crewai.memory.storage.backend.StorageBackend``,
``crewai.memory.types.MemoryRecord``, ``crewai.memory.storage.factory``). CrewAI replaced the
older ``storage.interface.Storage`` / ``ExternalMemory`` surface between 1.9 and 1.15, so there
is no ``ExternalMemory`` fixture here -- the drop-in is proven through the factory hook instead.
"""
from __future__ import annotations

import pytest
from crewai.memory.types import MemoryRecord

from saihm_adapters import SaihmMemoryClient, SaihmStorageBackend


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


@pytest.fixture
def backend(client):
    # client passed in => the backend does not own/close it (session fixture handles that).
    return SaihmStorageBackend(client=client)


@pytest.fixture
def rec():
    """Build a MemoryRecord. CrewAI -- not the adapter -- owns id/embedding in this API, so the
    tests construct records the way CrewAI would."""
    def make(content: str, **kw) -> MemoryRecord:
        return MemoryRecord(content=content, **kw)
    return make


@pytest.fixture
def embed():
    """A deterministic offline embedder mapping cat/dog synonyms onto two axes, so cosine
    ranking is *semantic* ('feline' matches 'cat') where lexical overlap would be zero.

    In this API CrewAI computes embeddings and hands them in: on the record as
    ``MemoryRecord.embedding`` and on the query as ``search(query_embedding=...)``. The adapter
    never embeds anything itself.
    """
    def _embed(text: str):
        t = text.lower()
        axes = {"cat": 0, "feline": 0, "kitten": 0, "dog": 1, "canine": 1, "puppy": 1}
        v = [0.0, 0.0]
        for word, i in axes.items():
            if word in t:
                v[i] += 1.0
        return v
    return _embed
