"""SaihmMemory ⇄ LlamaIndex ``BaseMemory`` contract + the SAIHM-specific invariants (role only
trusted for own messages, scoped reset, observable crypto-shred).

All tests run offline against the blind sandbox — no account, no live endpoint, no model.
"""
from __future__ import annotations

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import BaseMemory

from saihm_adapters.llamaindex_memory import SaihmMemory


def test_is_a_llamaindex_memory(memory):
    assert isinstance(memory, BaseMemory)


def test_empty_memory(memory):
    assert memory.get_all() == []


def test_put_and_get_all_roundtrip(memory):
    memory.put(ChatMessage(role=MessageRole.USER, content="hello"))
    memory.put(ChatMessage(role=MessageRole.ASSISTANT, content="hi there"))
    msgs = memory.get_all()
    assert [(m.role, m.content) for m in msgs] == [
        (MessageRole.USER, "hello"),
        (MessageRole.ASSISTANT, "hi there"),
    ]


def test_get_matches_get_all(memory):
    memory.put(ChatMessage(role=MessageRole.USER, content="x"))
    assert [m.content for m in memory.get()] == [m.content for m in memory.get_all()]


def test_system_role_preserved_for_own(memory):
    memory.put(ChatMessage(role=MessageRole.SYSTEM, content="be terse"))
    m = memory.get_all()[0]
    assert m.role == MessageRole.SYSTEM
    assert m.content == "be terse"


def test_foreign_cell_read_as_opaque_user(memory, client):
    """A cell written outside the adapter has no trusted role — read as an opaque USER message,
    so memorized untrusted text can't forge a privileged system/assistant role."""
    client.remember("a raw non-adapter cell")
    memory.put(ChatMessage(role=MessageRole.SYSTEM, content="own system msg"))
    foreign = next(m for m in memory.get_all() if m.content == "a raw non-adapter cell")
    assert foreign.role == MessageRole.USER  # NOT elevated to SYSTEM


def test_from_defaults_seeds_chat_history(client):
    seed = [ChatMessage(role=MessageRole.USER, content="seeded fact")]
    mem = SaihmMemory.from_defaults(client=client, chat_history=seed)
    assert any(m.content == "seeded fact" for m in mem.get_all())


def test_reset_shreds_only_this_memorys_messages(client):
    """reset() must crypto-shred ONLY what THIS memory added — never a sibling memory's messages
    or the user's other cells, even when several share one client."""
    a = SaihmMemory.from_defaults(client=client)
    b = SaihmMemory.from_defaults(client=client)
    client.remember("a precious foreign cell")
    a.put(ChatMessage(role=MessageRole.USER, content="a-owned"))
    b.put(ChatMessage(role=MessageRole.USER, content="b-owned"))
    assert len(client.recall()) == 3

    a.reset()

    texts = [m.text for m in client.recall()]
    assert len(texts) == 2
    assert any(t == "a precious foreign cell" for t in texts)  # foreign (raw) survived
    assert any("b-owned" in t for t in texts)                  # sibling (enveloped) survived


def test_forget_single_message_by_id(memory, client):
    memory.put(ChatMessage(role=MessageRole.USER, content="keep me"))
    memory.put(ChatMessage(role=MessageRole.USER, content="shred me"))
    target = next(m.cell_id for m in client.recall() if "shred me" in m.text)
    assert memory.forget(target) is True
    remaining = [m.text for m in client.recall()]
    assert not any("shred me" in t for t in remaining)
    assert any("keep me" in t for t in remaining)


def test_erasure_is_observable_in_recall(memory, client):
    memory.put(ChatMessage(role=MessageRole.USER, content="sensitive"))
    before = len(client.recall())
    memory.reset()
    assert len(client.recall()) == before - 1
