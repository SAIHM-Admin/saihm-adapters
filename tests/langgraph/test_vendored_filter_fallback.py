"""The vendored filter fallback must be PROVEN exercised, not merely present.

``_lg_filter`` binds LangGraph's private ``_compare_values`` / ``_does_match`` when they import, and
a vendored copy when they do not. A test that only ran alongside the real import would prove nothing
about the fallback, so ``forced_vendored`` makes the upstream import actually raise ImportError
(``sys.modules[...] = None`` is the documented way to do that) and reloads the module, then asserts:

  1. the vendored branch is the one that got taken (``USING_VENDORED is True``), and
  2. the vendored copies agree with upstream across a filter/namespace matrix, and
  3. ``SaihmStore`` itself still matches ``InMemoryStore`` while the fallback is live.

(2) is the point: "the fallback ran" is worth little without "the fallback is equivalent".
"""
from __future__ import annotations

import importlib
import sys

import pytest
from langgraph.store.base import MatchCondition

import saihm_adapters._lg_filter as lg_filter

# InMemoryStore is public API and is bound here rather than via conftest's `mem` fixture on purpose:
# forced_vendored nulls langgraph.store.memory in sys.modules, so anything importing from it during
# fixture setup would raise ImportError.
from langgraph.store.memory import InMemoryStore

# The upstream private symbols are the reference the vendored copies must match — but importing them
# unconditionally would make THIS file fail to collect in exactly the scenario the fallback exists
# for (a langgraph-checkpoint that dropped them). So guard the import and skip the equivalence tests
# when there is nothing to compare against; the vendored-behaviour tests still run.
try:
    from langgraph.store.memory import _compare_values as up_compare  # type: ignore[attr-defined]
    from langgraph.store.memory import _does_match as up_match  # type: ignore[attr-defined]

    UPSTREAM_AVAILABLE = True
except ImportError:  # pragma: no cover - only on a langgraph that removed the private helpers
    up_compare = up_match = None  # type: ignore[assignment]
    UPSTREAM_AVAILABLE = False

requires_upstream = pytest.mark.skipif(
    not UPSTREAM_AVAILABLE,
    reason="langgraph no longer exposes _compare_values/_does_match — nothing to compare against",
)


@pytest.fixture
def forced_vendored(monkeypatch):
    """Reload _lg_filter with the upstream import broken, so the except-ImportError branch runs.

    Rebinds ``langgraph_store`` too, and restores in the reverse order. That order matters: reloading
    the store while ``langgraph.store.memory`` is still nulled re-binds it to the VENDORED helpers
    and leaves it that way, so every later test would silently exercise the fallback instead of
    upstream. Hostile review caught exactly that leak here — conftest's ``_no_vendored_leak`` guard
    now fails the suite if it ever comes back.
    """
    import saihm_adapters.langgraph_store as store_mod

    monkeypatch.setitem(sys.modules, "langgraph.store.memory", None)
    with pytest.warns(RuntimeWarning, match="vendored copies"):
        reloaded = importlib.reload(lg_filter)
    assert reloaded.USING_VENDORED is True, "fallback branch did not run — test proves nothing"
    importlib.reload(store_mod)  # point SaihmStore at the vendored helpers
    assert store_mod._compare_values is reloaded._vendored_compare_values
    try:
        yield reloaded
    finally:
        monkeypatch.undo()           # 1. langgraph.store.memory importable again
        importlib.reload(lg_filter)  # 2. _lg_filter re-binds upstream
        importlib.reload(store_mod)  # 3. and only now does the store re-bind upstream


# a real ImportError, not a stub: setting the sys.modules entry to None makes import raise
def test_breaking_the_import_really_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "langgraph.store.memory", None)
    with pytest.raises(ImportError):
        from langgraph.store.memory import _compare_values  # noqa: F401


@requires_upstream
def test_normal_install_binds_upstream_not_vendored():
    assert lg_filter.USING_VENDORED is False
    assert lg_filter.compare_values is up_compare
    assert lg_filter.does_match is up_match
    # and the store is bound to the same objects, not to a stale copy
    import saihm_adapters.langgraph_store as store_mod

    assert store_mod._compare_values is up_compare
    assert store_mod._does_match is up_match


def test_fallback_warns_so_it_is_never_silent(monkeypatch):
    """A silently-engaged fallback would hide a real upstream semantics change. It must warn."""
    import saihm_adapters.langgraph_store as store_mod

    monkeypatch.setitem(sys.modules, "langgraph.store.memory", None)
    try:
        with pytest.warns(RuntimeWarning, match=r"vendored copies taken from langgraph-checkpoint"):
            importlib.reload(lg_filter)
        importlib.reload(store_mod)
    finally:
        monkeypatch.undo()
        importlib.reload(lg_filter)
        importlib.reload(store_mod)


@requires_upstream
def test_forced_fallback_binds_vendored(forced_vendored):
    assert forced_vendored.compare_values is forced_vendored._vendored_compare_values
    assert forced_vendored.does_match is forced_vendored._vendored_does_match
    assert forced_vendored.compare_values is not up_compare
    assert forced_vendored.does_match is not up_match


# ---- equivalence: vendored vs upstream ------------------------------------------------------

FILTER_CASES = [
    # (item_value, filter_value)
    (10, 10),                                  # scalar equal
    (10, 11),                                  # scalar unequal
    ("dark", "dark"),
    (None, None),
    (10, {"$eq": 10}),
    (10, {"$ne": 10}),
    (10, {"$gt": 5}),
    (10, {"$gt": 10}),
    (10, {"$gte": 10}),
    (10, {"$lt": 20}),
    (10, {"$lte": 10}),
    (10, {"$gt": 5, "$lt": 20}),               # multiple operators, all must hold
    (10, {"$gt": 5, "$lt": 7}),
    ({"a": 1}, {"a": 1}),                      # nested dict equal
    ({"a": 1, "b": 2}, {"a": 1}),              # partial nested match
    ({"a": 1}, {"a": 2}),
    ({"a": {"b": 3}}, {"a": {"b": 3}}),        # deep nesting
    ({"a": {"b": 3}}, {"a": {"b": {"$gte": 3}}}),
    ("scalar", {"a": 1}),                      # dict filter vs non-dict item
    ({"a": 1}, {"missing": 1}),                # key absent -> compares against None
    ([1, 2, 3], [1, 2, 3]),                    # list equal
    ([1, 2], [1, 2, 3]),                       # length mismatch
    ((1, 2), [1, 2]),                          # tuple/list cross-type
    ("nope", [1, 2]),                          # list filter vs non-list item
    ([{"a": 1}], [{"a": 1}]),                  # list of dicts
]


@requires_upstream
@pytest.mark.parametrize("item_value,filter_value", FILTER_CASES)
def test_vendored_compare_matches_upstream(item_value, filter_value, forced_vendored):
    assert forced_vendored.compare_values(item_value, filter_value) == up_compare(
        item_value, filter_value
    )


@requires_upstream
@pytest.mark.parametrize(
    "op,bad_value",
    [("$gt", "not-a-number"), ("$lte", None)],
)
def test_vendored_compare_raises_like_upstream_on_bad_operands(op, bad_value, forced_vendored):
    """Non-numeric operands hit float() in both implementations — same exception type."""
    with pytest.raises((TypeError, ValueError)):
        up_compare(bad_value, {op: 1})
    with pytest.raises((TypeError, ValueError)):
        forced_vendored.compare_values(bad_value, {op: 1})


@requires_upstream
def test_vendored_compare_rejects_unknown_operator_like_upstream(forced_vendored):
    for fn in (up_compare, forced_vendored.compare_values):
        with pytest.raises(ValueError, match="Unsupported operator"):
            fn(1, {"$regex": "x"})


NAMESPACE_CASES = [
    # (match_type, path, key)
    ("prefix", ("users",), ("users", "alice")),
    ("prefix", ("users", "alice"), ("users", "alice")),
    ("prefix", ("users", "bob"), ("users", "alice")),
    ("prefix", ("users", "alice", "deep"), ("users", "alice")),   # path longer than key
    ("prefix", ("*", "alice"), ("users", "alice")),               # wildcard
    ("prefix", ("users", "*"), ("users", "bob", "x")),
    ("prefix", (), ("users", "alice")),                           # empty path matches everything
    ("suffix", ("alice",), ("users", "alice")),
    ("suffix", ("bob",), ("users", "alice")),
    ("suffix", ("*",), ("users", "alice")),
    ("suffix", ("users", "*"), ("x", "users", "alice")),
    ("suffix", ("a", "b", "c"), ("b", "c")),                      # path longer than key
]


@requires_upstream
@pytest.mark.parametrize("match_type,path,key", NAMESPACE_CASES)
def test_vendored_does_match_matches_upstream(match_type, path, key, forced_vendored):
    cond = MatchCondition(match_type=match_type, path=path)
    assert forced_vendored.does_match(cond, key) == up_match(cond, key)


@requires_upstream
def test_vendored_does_match_rejects_unknown_match_type_like_upstream(forced_vendored):
    cond = MatchCondition(match_type="sideways", path=("users",))  # NamedTuple: no validation
    for fn in (up_match, forced_vendored.does_match):
        with pytest.raises(ValueError, match="Unsupported match type"):
            fn(cond, ("users", "alice"))


# ---- end-to-end: the store still matches InMemoryStore on the fallback path -------------------

def test_store_search_and_namespaces_stay_correct_on_fallback(forced_vendored, client):
    """The strongest proof: rebuild SaihmStore against the reloaded module and check real parity."""
    import saihm_adapters.langgraph_store as store_mod

    # forced_vendored already rebound the store; assert it rather than reloading again here (doing
    # that reload inside the test was the leak hostile review found).
    assert store_mod._compare_values is forced_vendored._vendored_compare_values
    mem = InMemoryStore()
    saihm = store_mod.SaihmStore(client=client)

    for s in (saihm, mem):
        s.put(("users", "alice"), "profile", {"name": "Alice", "score": 10})
        s.put(("users", "bob"), "profile", {"name": "Bob", "score": 3})
        s.put(("users", "alice"), "prefs", {"theme": "dark", "score": 7})

    def members(items):
        return {(it.namespace, it.key) for it in items}

    assert members(saihm.search(("users",))) == members(mem.search(("users",)))
    assert members(saihm.search(("users",), filter={"name": "Bob"})) == members(
        mem.search(("users",), filter={"name": "Bob"})
    )
    # Boundary filters on purpose: `$gt: 5` over scores {10,3,7} returns the same set under `>` and
    # `>=`, so it cannot tell a correct comparator from an off-by-one one. `$gt: 10` / `$gte: 10`
    # straddle an actual value and do discriminate.
    for flt in (
        {"score": {"$gt": 10}},
        {"score": {"$gte": 10}},
        {"score": {"$lt": 3}},
        {"score": {"$lte": 3}},
        {"score": {"$gt": 3, "$lte": 7}},
    ):
        assert members(saihm.search(("users",), filter=flt)) == members(
            mem.search(("users",), filter=flt)
        ), f"filter parity broke on {flt}"

    assert saihm.list_namespaces() == mem.list_namespaces()
    assert saihm.list_namespaces(max_depth=1) == mem.list_namespaces(max_depth=1)
    assert saihm.list_namespaces(suffix=("alice",)) == mem.list_namespaces(suffix=("alice",))
    assert saihm.list_namespaces(prefix=("users", "*")) == mem.list_namespaces(
        prefix=("users", "*")
    )
