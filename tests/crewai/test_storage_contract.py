"""SaihmStorageBackend <-> CrewAI ``StorageBackend`` contract, plus the SAIHM-specific
invariants (client-side ranking, coexistence, crypto-shred safety).

CrewAI's own reference backends need a vector DB + an embedding model + network, so they are not
offline-testable. These tests instead exercise the genuine protocol -- ``StorageBackend`` is
``@runtime_checkable``, so conformance is asserted directly -- and drive the documented wiring
through the real ``crewai.memory.storage.factory`` hook, entirely offline against the blind
sandbox.

API NOTE: CrewAI replaced the older ``storage.interface.Storage`` / ``ExternalMemory`` surface
between 1.9 and 1.15. ``search`` now takes a QUERY EMBEDDING (CrewAI embeds; the blind store
cannot) and returns ``(MemoryRecord, score)`` tuples rather than dicts.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from crewai.memory.storage.backend import StorageBackend
from crewai.memory.types import MemoryRecord, ScopeInfo

from saihm_adapters import SaihmMemoryClient, SaihmStorageBackend


# ---- protocol conformance ------------------------------------------------

def test_satisfies_runtime_checkable_protocol(backend):
    """The whole integration rests on this: CrewAI accepts any object satisfying the protocol."""
    assert isinstance(backend, StorageBackend)


# ---- core save / search --------------------------------------------------

def test_save_then_search_returns_record(backend, rec, embed):
    backend.save([rec("Alice prefers dark mode", metadata={"topic": "prefs"},
                      embedding=embed("cat"))])
    results = backend.search(embed("cat"), limit=5)
    assert len(results) == 1
    record, score = results[0]
    assert isinstance(record, MemoryRecord)
    assert record.content == "Alice prefers dark mode"
    assert record.metadata == {"topic": "prefs"}
    assert score == pytest.approx(1.0)  # identical vectors -> cosine 1.0


def test_search_empty_store_is_empty(backend, embed):
    assert backend.search(embed("anything")) == []


def test_search_min_score_filters(backend, rec, embed):
    backend.save([rec("a loyal puppy", embedding=embed("dog"))])
    # orthogonal axes -> cosine 0.0, so a 0.5 floor excludes it and a 0.0 floor keeps it
    assert backend.search(embed("cat"), min_score=0.5) == []
    assert len(backend.search(embed("cat"), min_score=0.0)) == 1


def test_search_ranks_and_limits(backend, rec, embed):
    backend.save([
        rec("a feline companion", embedding=embed("cat")),          # cosine 1.0 vs "cat"
        rec("a cat and a dog", embedding=embed("cat dog")),          # cosine ~0.707
        rec("a loyal puppy", embedding=embed("dog")),                # cosine 0.0
    ])
    top = backend.search(embed("cat"), limit=1)
    assert len(top) == 1 and top[0][0].content == "a feline companion"
    two = backend.search(embed("cat"), limit=2, min_score=0.01)
    assert [r.content for r, _ in two] == ["a feline companion", "a cat and a dog"]


def test_embedding_ranking_is_semantic_not_lexical(backend, rec, embed):
    """'a feline companion' shares no words with 'cat' -- only the embedding can rank it first."""
    backend.save([
        rec("a feline companion", embedding=embed("feline")),
        rec("a loyal puppy", embedding=embed("puppy")),
    ])
    results = backend.search(embed("cat"), min_score=0.5)
    assert [r.content for r, _ in results] == ["a feline companion"]


def test_recency_fallback_when_no_embeddings(backend, rec):
    """A blind store with no embeddings must still return something useful, newest first."""
    backend.save([rec("older", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))])
    backend.save([rec("newer", created_at=datetime(2026, 6, 1, tzinfo=timezone.utc))])
    results = backend.search([], min_score=0.0)
    assert [r.content for r, _ in results] == ["newer", "older"]


def test_structured_metadata_and_categories_roundtrip(backend, rec, embed):
    """content is a plain str in this API, but metadata/categories carry structure -- and they
    must survive the seal/unseal intact."""
    backend.save([rec("profile", categories=["prefs", "ui"],
                      metadata={"user": "alice", "nested": {"pref": "dark"}, "n": 3},
                      embedding=embed("cat"))])
    record, _ = backend.search(embed("cat"), min_score=0.0)[0]
    assert record.categories == ["prefs", "ui"]
    assert record.metadata == {"user": "alice", "nested": {"pref": "dark"}, "n": 3}


# ---- exact recall --------------------------------------------------------

def test_get_record_by_id_and_miss(backend, rec):
    r = rec("findable")
    backend.save([r])
    got = backend.get_record(r.id)
    assert got is not None and got.content == "findable"
    assert backend.get_record("no-such-id") is None


def test_list_records_newest_first_with_offset(backend, rec):
    for i, day in enumerate([1, 2, 3]):
        backend.save([rec(f"item{i}", created_at=datetime(2026, 1, day, tzinfo=timezone.utc))])
    assert [r.content for r in backend.list_records()] == ["item2", "item1", "item0"]
    assert [r.content for r in backend.list_records(limit=1, offset=1)] == ["item1"]


def test_save_upserts_by_id(backend, rec):
    r = rec("original")
    backend.save([r])
    r.content = "revised"
    backend.update(r)
    assert backend.count() == 1                       # replaced, not duplicated
    assert backend.get_record(r.id).content == "revised"


# ---- delete / erasure ----------------------------------------------------

def test_delete_by_record_ids(backend, rec):
    a, b = rec("keep"), rec("drop")
    backend.save([a, b])
    assert backend.delete(record_ids=[b.id]) == 1
    assert [r.content for r in backend.list_records()] == ["keep"]


def test_delete_by_category_and_metadata(backend, rec):
    backend.save([
        rec("tagged", categories=["x"], metadata={"keep": False}),
        rec("untagged", categories=["y"], metadata={"keep": True}),
    ])
    assert backend.delete(categories=["x"]) == 1
    assert backend.count() == 1
    assert backend.delete(metadata_filter={"keep": True}) == 1
    assert backend.count() == 0


def test_delete_older_than_handles_naive_datetimes(backend, rec):
    """MemoryRecord.created_at defaults to a NAIVE datetime, so a naive/aware comparison would
    raise (or silently drop matches). Both sides must be normalized."""
    backend.save([rec("ancient", created_at=datetime(2020, 1, 1))])            # naive
    backend.save([rec("fresh", created_at=datetime.now(timezone.utc))])        # aware
    cutoff = datetime.now(timezone.utc) - timedelta(days=1)
    assert backend.delete(older_than=cutoff) == 1
    assert [r.content for r in backend.list_records()] == ["fresh"]


def test_scope_prefix_respects_path_boundaries(backend, rec):
    """THE dangerous one: delete()/reset() crypto-shred irreversibly. A naive startswith would
    make '/demo' also match '/demo2' and '/demonstrate' and destroy unrelated scopes."""
    backend.save([
        rec("in scope", scope="/demo"),
        rec("child", scope="/demo/inner"),
        rec("sibling", scope="/demo2"),
        rec("lookalike", scope="/demonstrate"),
    ])
    assert backend.count(scope_prefix="/demo") == 2
    assert backend.delete(scope_prefix="/demo") == 2
    assert sorted(r.content for r in backend.list_records()) == ["lookalike", "sibling"]


# ---- introspection -------------------------------------------------------

def test_count_scopes_categories_and_scope_info(backend, rec):
    backend.save([
        rec("a", scope="/proj/alpha", categories=["c1"],
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc)),
        rec("b", scope="/proj/beta", categories=["c1", "c2"],
            created_at=datetime(2026, 3, 1, tzinfo=timezone.utc)),
    ])
    assert backend.count() == 2
    assert backend.list_scopes("/proj") == ["/proj/alpha", "/proj/beta"]
    assert backend.list_categories() == {"c1": 2, "c2": 1}

    info = backend.get_scope_info("/proj")
    assert isinstance(info, ScopeInfo)
    assert info.record_count == 2 and info.categories == ["c1", "c2"]
    assert info.oldest_record.year == 2026 and info.newest_record.month == 3


# ---- async ---------------------------------------------------------------

async def test_async_asave_asearch_adelete(backend, rec, embed):
    await backend.asave([rec("async memory item", embedding=embed("cat"))])
    results = await backend.asearch(embed("cat"), min_score=0.0)
    assert len(results) == 1 and results[0][0].content == "async memory item"
    assert await backend.adelete() == 1


# ---- documented wiring: the real factory hook ----------------------------

def test_factory_hook_resolves_our_backend(client):
    """The README tells users to register via set_memory_storage_factory; prove that path."""
    from crewai.memory.storage.factory import (
        resolve_memory_storage,
        set_memory_storage_factory,
    )
    try:
        set_memory_storage_factory(
            lambda spec: SaihmStorageBackend(client=client) if spec == "saihm" else None
        )
        resolved = resolve_memory_storage("saihm")
        assert isinstance(resolved, SaihmStorageBackend)
        assert isinstance(resolved, StorageBackend)
        assert resolve_memory_storage("something-else") is None  # defers to CrewAI
    finally:
        set_memory_storage_factory(None)  # never leak global state into other tests


# ---- SAIHM-specific: coexistence + crypto-shred safety -------------------

def test_ignores_foreign_cells(backend, client, rec, embed):
    """A cell written outside the adapter (no _saihm_crewai envelope) is invisible to the
    backend -- a shared SAIHM store can hold facts from other adapters without collision."""
    client.remember("a raw non-adapter memory cell")
    backend.save([rec("adapter-owned memory", embedding=embed("cat"))])
    results = backend.search(embed("cat"), min_score=0.0)
    assert len(results) == 1 and results[0][0].content == "adapter-owned memory"


def test_reset_does_not_shred_foreign_cells(backend, client, rec):
    """reset() must crypto-shred ONLY the backend's own entries -- never a sibling adapter's or
    a user's other cells. A naive 'forget everything' here would be catastrophic."""
    client.remember("a precious foreign cell")   # not ours
    backend.save([rec("ours #1"), rec("ours #2")])
    assert len(client.recall()) == 3

    backend.reset()

    remaining = client.recall()
    assert len(remaining) == 1
    assert remaining[0].text == "a precious foreign cell"   # foreign survived


def test_save_then_shred_is_observable(backend, client, rec):
    """Erasure removes the underlying cell from the client's own recall -- the observable of a
    crypto-shred, not a soft hide."""
    backend.save([rec("sensitive pii", metadata={"pii": True})])
    before = len(client.recall())
    backend.reset()
    assert len(client.recall()) == before - 1
