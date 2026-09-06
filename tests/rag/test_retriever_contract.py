"""SaihmRetriever ⇄ LlamaIndex ``BaseRetriever`` contract: ingestion, client-side ranking
(lexical + cosine), the relevance floor, node-id dedup, async parity, and the SAIHM-specific
invariants (exact crypto-shred erasure, scoped clear that spares foreign cells).

All tests run offline against the blind sandbox — no account, no live endpoint, no LLM.
"""
from __future__ import annotations

import asyncio

import pytest

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle, TextNode

from saihm_adapters import SaihmRetriever


def test_is_a_llamaindex_retriever(retriever):
    assert isinstance(retriever, BaseRetriever)


def test_retrieve_empty_corpus(retriever):
    assert retriever.retrieve("anything at all") == []


def test_add_and_ranked_retrieve(retriever):
    retriever.add_texts([
        "Dana is allergic to penicillin and avoids beta-lactam antibiotics.",
        "The cafeteria serves soup on Fridays.",
        "Confirm the patient is allergic to penicillin antibiotics.",
    ])
    hits = retriever.retrieve("which antibiotics is the patient allergic to — penicillin?")
    assert len(hits) >= 2
    assert "penicillin" in hits[0].node.get_content()          # most relevant first
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)              # descending by score


def test_irrelevant_query_returns_nothing(retriever):
    """The relevance floor (score > 0) means an off-topic query returns nothing rather than
    padding the context with irrelevant chunks."""
    retriever.add_texts(["Dana is allergic to penicillin."])
    assert retriever.retrieve("quarterly financial derivatives volatility") == []


def test_top_k_caps_results(client):
    r = SaihmRetriever(client=client, similarity_top_k=1)
    r.add_texts(["alpha beta gamma", "alpha beta delta", "alpha epsilon"])
    assert len(r.retrieve("alpha beta gamma")) == 1


def test_add_texts_metadata_roundtrips(retriever):
    retriever.add_texts(["encrypted portable memory chunk"], metadatas=[{"src": "doc1"}])
    hits = retriever.retrieve("encrypted portable memory chunk")
    assert hits and hits[0].node.metadata.get("src") == "doc1"


def test_add_texts_metadata_length_mismatch_raises(retriever):
    with pytest.raises(ValueError):
        retriever.add_texts(["a", "b"], metadatas=[{"only": 1}])


def test_duplicate_nodes_collapse(retriever):
    """A node added more than once (each a distinct cell) still ranks as ONE distinct chunk, so
    similarity_top_k counts distinct chunks rather than being starved by copies."""
    node = TextNode(text="unique alpha beta gamma content")
    retriever.add_nodes([node])
    retriever.add_nodes([node])  # same node_id, second cell
    ids = [h.node.node_id for h in retriever.retrieve("unique alpha beta gamma content")]
    assert ids.count(node.node_id) == 1


def test_async_retrieve_matches_sync(retriever):
    retriever.add_texts(["alpha beta gamma delta", "unrelated soup fridays cafeteria"])
    sync = retriever.retrieve("alpha beta gamma")
    asyncres = asyncio.run(retriever._aretrieve(QueryBundle(query_str="alpha beta gamma")))
    assert [h.node.get_content() for h in sync] == [h.node.get_content() for h in asyncres]


def test_cosine_ranking_when_embeddings_present(retriever):
    """When the query and every candidate carry same-dimension embeddings, ranking is cosine
    (semantic), not lexical — 'feline' wins for a query on the same axis though lexical overlap
    is zero."""
    retriever.add_nodes([
        TextNode(text="a feline companion", embedding=[1.0, 0.0]),
        TextNode(text="a loyal puppy", embedding=[0.0, 1.0]),
    ])
    hits = retriever._retrieve(QueryBundle(query_str="pet", embedding=[1.0, 0.0]))
    assert hits and hits[0].node.get_content() == "a feline companion"
    assert hits[0].score == pytest.approx(1.0)


def test_forget_removes_from_retrieval(retriever):
    retriever.add_texts([
        "keep alpha beta gamma",
        "shred me: penicillin allergy record",
    ])
    hits = retriever.retrieve("penicillin allergy record")
    target = hits[0].node.node_id
    assert retriever.forget(target) is True
    again = retriever.retrieve("penicillin allergy record")
    assert not any("penicillin" in h.node.get_content() for h in again)


def test_forget_unknown_id_returns_false(retriever):
    retriever.add_texts(["alpha beta gamma"])
    assert retriever.forget("no-such-node-id") is False


def test_clear_shreds_only_rag_chunks(retriever, client):
    """clear() must crypto-shred ONLY the chunks this retriever wrote (the _saihm_rag envelope) —
    never a sibling adapter's or the user's other cells in the same owned store."""
    client.remember("a precious foreign non-RAG cell")
    retriever.add_texts(["rag chunk one", "rag chunk two"])
    assert len(client.recall()) == 3

    retriever.clear()

    remaining = [m.text for m in client.recall()]
    assert remaining == ["a precious foreign non-RAG cell"]  # only the RAG chunks were shredded
