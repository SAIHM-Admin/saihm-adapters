"""SaihmChatMessageHistory ⇄ LangChain ``BaseChatMessageHistory`` contract + the SAIHM-specific
invariants (role only trusted for own messages, scoped clear, observable crypto-shred).

All tests run offline against the blind sandbox — no account, no live endpoint, no model.
"""
from __future__ import annotations

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from saihm_adapters import SaihmChatMessageHistory


def test_is_a_langchain_chat_history(history):
    assert isinstance(history, BaseChatMessageHistory)


def test_empty_history(history):
    assert history.messages == []


def test_add_and_read_roundtrip(history):
    history.add_user_message("hello")
    history.add_ai_message("hi there")
    msgs = history.messages
    assert [type(m) for m in msgs] == [HumanMessage, AIMessage]
    assert [m.content for m in msgs] == ["hello", "hi there"]


def test_system_role_preserved_for_own(history):
    history.add_message(SystemMessage(content="be terse"))
    m = history.messages[0]
    assert isinstance(m, SystemMessage)
    assert m.content == "be terse"


def test_foreign_cell_read_as_opaque_human(history, client):
    """A cell written outside the adapter has no trusted role — it is read as an opaque
    HumanMessage, so memorized untrusted text can't forge a privileged system/assistant turn."""
    client.remember("a raw non-adapter cell")
    history.add_message(SystemMessage(content="own system msg"))
    foreign = next(m for m in history.messages if m.content == "a raw non-adapter cell")
    assert isinstance(foreign, HumanMessage)  # NOT elevated to SystemMessage/AIMessage


def test_clear_shreds_only_this_historys_messages(client):
    """clear() must crypto-shred ONLY what THIS history added — never a sibling history's
    messages or the user's other cells, even when several share one client."""
    a = SaihmChatMessageHistory(client=client)
    b = SaihmChatMessageHistory(client=client)
    client.remember("a precious foreign cell")
    a.add_user_message("a-owned")
    b.add_user_message("b-owned")
    assert len(client.recall()) == 3

    a.clear()

    texts = [m.text for m in client.recall()]
    assert len(texts) == 2
    assert any(t == "a precious foreign cell" for t in texts)  # foreign (raw) survived
    assert any("b-owned" in t for t in texts)                  # sibling (enveloped) survived


def test_forget_single_message_by_id(history, client):
    history.add_user_message("keep me")
    history.add_user_message("shred me")
    target = next(m.cell_id for m in client.recall() if "shred me" in m.text)
    assert history.forget(target) is True
    remaining = [m.text for m in client.recall()]
    assert not any("shred me" in t for t in remaining)
    assert any("keep me" in t for t in remaining)


def test_erasure_is_observable_in_recall(history, client):
    history.add_user_message("sensitive")
    before = len(client.recall())
    history.clear()
    assert len(client.recall()) == before - 1
