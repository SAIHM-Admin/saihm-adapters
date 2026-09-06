"""SaihmMemory ⇄ ``autogen_core.memory.Memory`` contract + real-AutoGen integration
(``update_context`` against a real model context, and Component dump/load through AutoGen's own
serialization), plus the SAIHM-specific invariants (coexistence with foreign cells, scoped
``clear``, and observable crypto-shred).

All tests run offline against the blind sandbox — no account, no live endpoint, no model.
"""
from __future__ import annotations

import pytest

from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import SystemMessage

from saihm_adapters.autogen_memory import SaihmMemory


# ---- protocol basics -----------------------------------------------------

def test_is_an_autogen_memory(memory):
    # A real drop-in: isinstance of AutoGen's own Memory protocol/ABC.
    assert isinstance(memory, Memory)


async def test_add_then_query_roundtrip(memory):
    await memory.add(MemoryContent(content="Alice prefers dark mode", mime_type=MemoryMimeType.TEXT))
    res = await memory.query()
    assert isinstance(res, MemoryQueryResult)
    assert len(res.results) == 1
    assert res.results[0].content == "Alice prefers dark mode"


async def test_query_empty_store_is_empty(memory):
    assert (await memory.query()).results == []


# ---- typed content round-trips (own cells keep mime_type + metadata) ------

async def test_text_metadata_roundtrips(memory):
    await memory.add(
        MemoryContent(content="v", mime_type=MemoryMimeType.TEXT, metadata={"topic": "prefs"})
    )
    m = (await memory.query()).results[0]
    assert m.content == "v"
    assert m.mime_type == MemoryMimeType.TEXT
    assert m.metadata == {"topic": "prefs"}


async def test_json_dict_content_roundtrips(memory):
    await memory.add(MemoryContent(content={"user": "alice", "pref": "dark"}, mime_type=MemoryMimeType.JSON))
    m = (await memory.query()).results[0]
    assert m.content == {"user": "alice", "pref": "dark"}
    assert m.mime_type == MemoryMimeType.JSON


async def test_bytes_content_roundtrips(memory):
    payload = bytes(range(8))
    await memory.add(MemoryContent(content=payload, mime_type=MemoryMimeType.BINARY))
    m = (await memory.query()).results[0]
    assert m.content == payload  # base64 round-trip back to the exact bytes


# ---- update_context against a REAL AutoGen model context ------------------

async def test_update_context_injects_one_system_message(memory):
    await memory.add(MemoryContent(content="Dana uses metric units", mime_type=MemoryMimeType.TEXT))
    ctx = BufferedChatCompletionContext(buffer_size=10)
    result = await memory.update_context(ctx)
    msgs = await ctx.get_messages()
    assert len(msgs) == 1
    assert isinstance(msgs[0], SystemMessage)
    assert "metric units" in str(msgs[0].content)
    assert len(result.memories.results) == 1


async def test_update_context_empty_adds_nothing(memory):
    ctx = BufferedChatCompletionContext(buffer_size=10)
    result = await memory.update_context(ctx)
    assert await ctx.get_messages() == []
    assert result.memories.results == []


# ---- coexistence: a shared store may hold other adapters'/users' cells ----

async def test_foreign_cells_read_as_opaque_text(memory, client):
    """A cell written outside the adapter (no _saihm envelope) is still visible in the owned
    store, but is read as opaque text/plain with no metadata — memorized untrusted text can't
    forge a typed mime_type or metadata."""
    client.remember("a raw non-adapter memory cell")
    await memory.add(
        MemoryContent(content="adapter-owned fact", mime_type=MemoryMimeType.TEXT, metadata={"m": 1})
    )
    results = (await memory.query()).results
    contents = [str(m.content) for m in results]
    assert "a raw non-adapter memory cell" in contents
    assert "adapter-owned fact" in contents
    foreign = next(m for m in results if str(m.content) == "a raw non-adapter memory cell")
    assert foreign.mime_type == MemoryMimeType.TEXT
    assert foreign.metadata is None


# ---- SAIHM-specific safety: clear() is scoped, erasure is a crypto-shred ---

async def test_clear_shreds_only_this_instances_cells(client):
    """clear() must crypto-shred ONLY what THIS instance added — never a sibling memory's cells
    or the user's other cells, even when several share one client. A naive 'forget everything'
    here would be catastrophic."""
    a = SaihmMemory(client=client)
    b = SaihmMemory(client=client)
    client.remember("a precious foreign cell")           # neither instance's
    await a.add(MemoryContent(content="a-owned", mime_type=MemoryMimeType.TEXT))
    await b.add(MemoryContent(content="b-owned", mime_type=MemoryMimeType.TEXT))
    assert len(client.recall()) == 3

    await a.clear()

    remaining = {m.text for m in client.recall()}
    assert len(remaining) == 2
    assert "a precious foreign cell" in remaining  # foreign survived
    # b's cell survived (its content is JSON-enveloped, so match by substring)
    assert any("b-owned" in t for t in remaining)


async def test_forget_single_cell_by_id(memory, client):
    await memory.add(MemoryContent(content="keep me", mime_type=MemoryMimeType.TEXT))
    await memory.add(MemoryContent(content="shred me", mime_type=MemoryMimeType.TEXT))
    target = next(m.cell_id for m in client.recall() if "shred me" in m.text)
    assert await memory.forget(target) is True
    remaining = [m.text for m in client.recall()]
    assert not any("shred me" in t for t in remaining)
    assert any("keep me" in t for t in remaining)


async def test_erasure_is_observable_in_recall(memory, client):
    """Erasure removes the underlying cell from the client's own recall — the observable of a
    crypto-shred, not a soft hide."""
    await memory.add(MemoryContent(content="sensitive pii", mime_type=MemoryMimeType.TEXT))
    before = len(client.recall())
    await memory.clear()
    assert len(client.recall()) == before - 1


# ---- drop-in through AutoGen's own Component serialization -----------------

async def test_component_serialization_roundtrips(memory):
    """SaihmMemory is an AutoGen Component: dump_component()/load_component() round-trip through
    AutoGen's real serialization machinery (only the instance name is serialized — never a key
    or cell id)."""
    model = memory.dump_component()
    assert model.config["name"] == "saihm"
    loaded = SaihmMemory.load_component(model)
    try:
        assert isinstance(loaded, SaihmMemory)
        assert loaded.name == "saihm"
    finally:
        await loaded.close()  # load_component builds a fresh client; close its sidecar
