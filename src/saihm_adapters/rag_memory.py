"""LlamaIndex integration: a SAIHM-backed ``BaseRetriever`` for RAG over a corpus you own.

Retrieval-augmented generation, but the knowledge base is yours: every chunk is sealed
client-side by the bundled SAIHM Node sidecar (Python holds no key), portable across models
*and* frameworks, and provably erasable (GDPR Art. 17) — erase a source document and it is
crypto-shredded, not merely de-indexed.

    from saihm_adapters import SaihmRetriever
    from llama_index.core.schema import TextNode

    r = SaihmRetriever()
    r.add_nodes([TextNode(text="Dana is allergic to penicillin.")])   # seal chunks into the corpus
    nodes = r.retrieve("what is the patient allergic to?")            # the RAG retrieval step

Wire it into a full RAG query loop with your own LLM:

    from llama_index.core.query_engine import RetrieverQueryEngine
    engine = RetrieverQueryEngine.from_args(r, llm=your_llm)
    print(engine.query("what is the patient allergic to?"))

Each chunk is one encrypted SAIHM cell. SAIHM is a *blind* store: the endpoint holds ciphertext
only and cannot run a server-side vector index, so retrieval **ranking happens client-side**.
Offline (the default) that is a transparent, stopword-filtered content-word overlap score; give the
query and *every* candidate chunk a same-dimension embedding (e.g. via a LlamaIndex embed model) and
the whole result set is ranked by cosine instead. The two scales are never mixed: if any candidate
lacks a matching embedding, the entire query ranks lexically. Erasure (:meth:`forget`) is always exact.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from llama_index.core.callbacks import CallbackManager
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, IndexNode, NodeWithScore, QueryBundle, TextNode
from llama_index.core.storage.docstore.utils import doc_to_json, json_to_doc

from .client import SaihmMemoryClient

_ENVELOPE = "_saihm_rag"
_TOKEN = re.compile(r"[a-z0-9]+")
# Common words carry no retrieval signal; drop them so the offline overlap score reflects content.
_STOP = frozenset(
    "the a an and or of to in on at for with from into is are was were be been being this that "
    "these those it its as by it's i you he she we they them his her their our your my me do does "
    "did has have had will would can could should which what who whom when where why how not no "
    "but if then than so such about over under again here there all any each more most other some "
    "only own same too very".split()
)


def _content_tokens(text: str) -> List[str]:
    """Lowercased alphanumeric tokens, minus stopwords and very short tokens."""
    return [t for t in _TOKEN.findall(text.lower()) if len(t) >= 3 and t not in _STOP]


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v)) if v else 0.0


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    val = sum(x * y for x, y in zip(a, b)) / (na * nb)
    return val if math.isfinite(val) else 0.0  # guard non-finite components (inf/nan)


class SaihmRetriever(BaseRetriever):
    """A LlamaIndex ``BaseRetriever`` whose corpus lives in SAIHM, sealed client-side.

    Manages only the chunks it wrote (cells carrying the ``_saihm_rag`` envelope); other cells in
    the same owned store — e.g. facts written from the LangChain, CrewAI, AutoGen, or LangGraph
    adapters — are left untouched. Pass a ``client`` to reuse a session, or omit it for a local
    blind sandbox (paid live endpoint via env — see
    :class:`~saihm_adapters.client.SaihmMemoryClient`).
    """

    def __init__(
        self,
        client: Optional[SaihmMemoryClient] = None,
        *,
        similarity_top_k: int = 3,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[Dict] = None,
        objects: Optional[List[IndexNode]] = None,
        verbose: bool = False,
        **client_kwargs,
    ) -> None:
        # Forward the standard BaseRetriever kwargs (incl. objects/object_map for recursive
        # retrieval) to the base ctor; only the remaining kwargs configure SaihmMemoryClient.
        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )
        self._client = client or SaihmMemoryClient(**client_kwargs)
        self._owns = client is None
        self.similarity_top_k = similarity_top_k

    # ---- ingestion -------------------------------------------------------
    def add_nodes(self, nodes: Sequence[BaseNode]) -> List[str]:
        """Seal each node into the owned corpus as one encrypted cell. Returns the cell ids.

        Uses the canonical LlamaIndex node serializer, so any ``BaseNode`` subclass — including
        the ``IndexNode`` used by recursive retrieval — round-trips as itself, not a plain
        ``TextNode``.
        """
        ids: List[str] = []
        for node in nodes:
            ids.append(self._client.remember(json.dumps({_ENVELOPE: 1, "node": doc_to_json(node)})))
        return ids

    def add_texts(self, texts: Sequence[str], metadatas: Optional[Sequence[dict]] = None) -> List[str]:
        """Convenience: seal raw strings (with optional per-text metadata) as chunks."""
        texts = list(texts)
        if metadatas is None:
            metas: List[dict] = [{} for _ in texts]
        else:
            metas = list(metadatas)
            if len(metas) != len(texts):
                raise ValueError(f"metadatas length {len(metas)} != texts length {len(texts)}")
        return self.add_nodes([TextNode(text=t, metadata=m) for t, m in zip(texts, metas)])

    # ---- storage helpers -------------------------------------------------
    def _all(self) -> List[Tuple[str, BaseNode]]:
        """Every chunk this retriever wrote, as (cell_id, node); preserves the node subclass."""
        out: List[Tuple[str, BaseNode]] = []
        for m in self._client.recall():
            try:
                obj = json.loads(m.text)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(obj, dict) and obj.get(_ENVELOPE) and isinstance(obj.get("node"), dict):
                try:
                    out.append((m.cell_id, json_to_doc(obj["node"])))
                except Exception:
                    continue
        return out

    # ---- retrieval -------------------------------------------------------
    @staticmethod
    def _lexical(query_tokens: set, node: BaseNode) -> float:
        if not query_tokens:
            return 0.0
        toks = _content_tokens(node.get_content())
        if not toks:
            return 0.0
        overlap = sum(1 for t in toks if t in query_tokens)
        return overlap / math.sqrt(len(toks))  # term-frequency-ish, length-normalized

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = [node for _, node in self._all()]
        # Coerce the query embedding to a list of plain floats: an embed pipeline may hand back a
        # numpy array, whose truthiness is ambiguous (`bool(np.ndarray)` raises).
        qe = query_bundle.embedding
        qe = [float(x) for x in qe] if qe is not None else None
        # Pick ONE ranking scale for the whole result set, never a mix: cosine only when the query
        # carries a non-zero embedding AND every candidate carries a same-dimension embedding;
        # otherwise lexical overlap for all. (Cosine is bounded in [-1,1] while lexical overlap is
        # unbounded — different scales, so a mixed corpus must not co-sort them; a degenerate zero-norm
        # query embedding would score every candidate 0, so it falls back to lexical as well.)
        use_cosine = bool(qe) and _norm(qe) > 0.0 and bool(nodes) and all(
            n.embedding and len(n.embedding) == len(qe) for n in nodes
        )
        if use_cosine:
            scored = [(_cosine(qe, n.embedding), n) for n in nodes]
        else:
            q = set(_content_tokens(query_bundle.query_str))
            scored = [(self._lexical(q, n), n) for n in nodes]
        # Collapse duplicates: a node may have been added more than once (each is its own cell), but
        # retrieval ranks DISTINCT nodes — keep the best score per node_id so similarity_top_k counts
        # distinct chunks rather than being starved by several copies of one node.
        best: dict = {}
        for s, n in scored:
            if n.node_id not in best or s > best[n.node_id][0]:
                best[n.node_id] = (s, n)
        scored = list(best.values())
        # Relevance floor (a fixed similarity_cutoff of 0): keep only positively-relevant chunks.
        # A query unrelated to the corpus therefore returns fewer than similarity_top_k — possibly
        # none — rather than padding context with noise.
        scored = [(s, n) for s, n in scored if s > 0.0]
        # Most relevant first; node_id ascending is a deterministic, self-contained tie-break.
        scored.sort(key=lambda sn: sn[1].node_id)            # stable secondary: node_id ascending
        scored.sort(key=lambda sn: sn[0], reverse=True)      # primary: score descending
        return [NodeWithScore(node=n, score=s) for s, n in scored[: max(0, self.similarity_top_k)]]

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # The SAIHM client is synchronous; offload so the event loop is never blocked.
        return await asyncio.to_thread(self._retrieve, query_bundle)

    # ---- erasure / lifecycle --------------------------------------------
    def forget(self, node_id: str) -> bool:
        """Crypto-shred every chunk with this node id (GDPR Art. 17).

        Shreds *all* matching cells (a node may have been added more than once), then verifies
        none remain. Returns True only if at least one was found and nothing with that id survives.
        """
        matches = [cid for cid, node in self._all() if node.node_id == node_id]
        if not matches:
            return False
        for cid in matches:
            self._client._forget_raw(cid)
        return not any(node.node_id == node_id for _, node in self._all())

    def clear(self) -> None:
        """Crypto-shred every chunk this retriever wrote (leaves other adapters' cells untouched)."""
        for cid, _ in self._all():
            self._client._forget_raw(cid)

    @property
    def client(self) -> SaihmMemoryClient:
        return self._client

    def close(self) -> None:
        if self._owns:
            self._client.close()
