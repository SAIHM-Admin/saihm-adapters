"""SaihmStore ⇄ LangGraph InMemoryStore parity + SAIHM-specific invariants.

The adapter reuses LangGraph's own ``_compare_values`` / ``_does_match`` so filter and
namespace semantics match InMemoryStore exactly; the one documented difference is search
ordering (SaihmStore is newest-first; InMemoryStore is insertion order), so search assertions
compare membership, not order.
"""
from __future__ import annotations

import pytest

NS = ("users", "alice")


def _members(items):
    return {(it.namespace, it.key) for it in items}


def _seed(store):
    store.put(("users", "alice"), "profile", {"name": "Alice", "score": 10})
    store.put(("users", "bob"), "profile", {"name": "Bob", "score": 3})
    store.put(("users", "alice"), "prefs", {"theme": "dark", "score": 7})
    store.put(("docs", "x"), "meta", {"kind": "note", "score": 10})


# ---- core get/put/delete -------------------------------------------------

def test_put_get_roundtrip(saihm, mem):
    for s in (saihm, mem):
        s.put(NS, "profile", {"name": "Alice", "score": 10})
    a, b = saihm.get(NS, "profile"), mem.get(NS, "profile")
    assert a is not None and b is not None
    assert a.value == b.value == {"name": "Alice", "score": 10}
    assert a.key == b.key == "profile"
    assert a.namespace == b.namespace == NS


def test_get_missing_is_none(saihm, mem):
    assert saihm.get(NS, "nope") is None
    assert mem.get(NS, "nope") is None


def test_upsert_preserves_created_at_updates_updated_at(saihm):
    saihm.put(NS, "profile", {"v": 1})
    first = saihm.get(NS, "profile")
    saihm.put(NS, "profile", {"v": 2})
    second = saihm.get(NS, "profile")
    assert second.value == {"v": 2}
    assert second.created_at == first.created_at      # created_at preserved across upsert
    assert second.updated_at >= first.updated_at      # updated_at advances


def test_delete_removes(saihm, mem):
    for s in (saihm, mem):
        s.put(NS, "profile", {"name": "Alice"})
        s.delete(NS, "profile")
    assert saihm.get(NS, "profile") is None
    assert mem.get(NS, "profile") is None


def test_put_value_none_deletes(saihm):
    saihm.put(NS, "profile", {"name": "Alice"})
    saihm.put(NS, "profile", None)  # PutOp(value=None) == delete
    assert saihm.get(NS, "profile") is None


# ---- search: namespace prefix, filter, paging ----------------------------

def test_search_namespace_prefix_membership(saihm, mem):
    for s in (saihm, mem):
        _seed(s)
    a = saihm.search(("users",))
    b = mem.search(("users",))
    assert _members(a) == _members(b)
    assert _members(a) == {
        (("users", "alice"), "profile"),
        (("users", "bob"), "profile"),
        (("users", "alice"), "prefs"),
    }


def test_search_filter_operator_parity(saihm, mem):
    for s in (saihm, mem):
        _seed(s)
    a = saihm.search(("users",), filter={"score": {"$gt": 5}})
    b = mem.search(("users",), filter={"score": {"$gt": 5}})
    assert _members(a) == _members(b)
    assert _members(a) == {(("users", "alice"), "profile"), (("users", "alice"), "prefs")}


def test_search_exact_filter_parity(saihm, mem):
    for s in (saihm, mem):
        _seed(s)
    assert _members(saihm.search(("users",), filter={"name": "Bob"})) == \
        _members(mem.search(("users",), filter={"name": "Bob"}))


def test_search_paging(saihm):
    _seed(saihm)
    page1 = saihm.search(("users",), limit=2, offset=0)
    page2 = saihm.search(("users",), limit=2, offset=2)
    assert len(page1) == 2
    assert _members(page1).isdisjoint(_members(page2))


def test_search_is_newest_first(saihm):
    # documented SaihmStore ordering (differs from InMemoryStore insertion order)
    saihm.put(("t",), "a", {"i": 1})
    saihm.put(("t",), "b", {"i": 2})
    saihm.put(("t",), "c", {"i": 3})
    keys = [it.key for it in saihm.search(("t",))]
    assert keys == ["c", "b", "a"]


def test_search_score_is_none(saihm):
    saihm.put(NS, "profile", {"name": "Alice"})
    (item,) = saihm.search(NS)
    assert item.score is None  # blind store: no server-side semantic ranking


# ---- list_namespaces -----------------------------------------------------

def test_list_namespaces_parity(saihm, mem):
    for s in (saihm, mem):
        _seed(s)
    assert saihm.list_namespaces() == mem.list_namespaces()


def test_list_namespaces_max_depth_parity(saihm, mem):
    for s in (saihm, mem):
        _seed(s)
    assert saihm.list_namespaces(max_depth=1) == mem.list_namespaces(max_depth=1)


# ---- async ---------------------------------------------------------------

async def test_async_abatch_roundtrip(saihm):
    await saihm.aput(NS, "profile", {"name": "Alice"})
    got = await saihm.aget(NS, "profile")
    assert got is not None and got.value == {"name": "Alice"}
    await saihm.adelete(NS, "profile")
    assert await saihm.aget(NS, "profile") is None


# ---- SAIHM-specific: coexistence + erasure -------------------------------

def test_ignores_foreign_cells(saihm, client):
    """A cell written outside the adapter (no _saihm_lg envelope) is invisible to the store —
    so a shared SAIHM store can hold facts from other adapters without collision."""
    client.remember("a raw non-adapter memory cell")
    saihm.put(NS, "profile", {"name": "Alice"})
    assert _members(saihm.search(())) == {(NS, "profile")}  # empty prefix = all adapter items only
    assert saihm.list_namespaces() == [NS]                  # foreign cell contributes no namespace
    assert len(saihm.search(NS)) == 1


def test_delete_crypto_shreds(saihm, client):
    """delete() erases the underlying cell (not just hides it): the cell is gone from the
    client's own recall, which is the observable of a crypto-shred."""
    saihm.put(NS, "secret", {"pii": "sensitive"})
    before = len(client.recall())
    saihm.delete(NS, "secret")
    after = len(client.recall())
    assert after == before - 1
    assert saihm.get(NS, "secret") is None
