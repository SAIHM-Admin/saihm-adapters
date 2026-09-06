"""LangGraph integration: a SAIHM-backed ``BaseStore`` for long-term, cross-thread memory.

Give a LangGraph app long-term memory it owns: portable across models and frameworks,
non-custodial (sealed client-side by the bundled SAIHM Node sidecar — Python never holds a
key), and provably erasable (GDPR Art. 17). Plug it into the graph:

    from saihm_adapters import SaihmStore

    graph = builder.compile(store=SaihmStore())   # nodes get the store injected

Each ``(namespace, key)`` is one encrypted SAIHM cell. ``BaseStore`` implements
``get`` / ``put`` / ``search`` / ``delete`` / ``list_namespaces`` (sync **and** async) on top
of ``batch`` / ``abatch``, so this class only implements those two.

SAIHM is a *blind* store: the endpoint holds ciphertext only and cannot run a server-side
vector index, so :meth:`search` applies the namespace prefix + ``filter`` and returns matches
newest-first with ``score=None`` (no semantic ranking). ``filter`` ($-operators) and namespace
matching reuse LangGraph's own helpers, so those match ``InMemoryStore`` exactly; ordering is
newest-first (vs ``InMemoryStore``'s insertion order). Erasure (``delete`` / ``put(value=None)``)
crypto-shreds the cell.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)
# Exact parity with LangGraph's reference InMemoryStore: JSONB-style filter comparison (incl.
# $-operators) and namespace match-condition matching. Bound to LangGraph's own private helpers
# when importable, else to a vendored copy — see _lg_filter for why that fallback exists.
from ._lg_filter import compare_values as _compare_values, does_match as _does_match

from .client import SaihmMemoryClient

_ENVELOPE = "_saihm_lg"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(s) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


class SaihmStore(BaseStore):
    """A LangGraph ``BaseStore`` whose items live in SAIHM, sealed client-side.

    The store manages only the items it wrote (cells carrying the ``_saihm_lg`` envelope);
    other cells in the same owned store — e.g. facts written from the LangChain, CrewAI, or
    AutoGen adapters — are left untouched. Pass a ``client`` to reuse a session, or omit it for
    a local blind sandbox (paid live endpoint via env — see
    :class:`~saihm_adapters.client.SaihmMemoryClient`).
    """

    supports_ttl = False  # blind store: no server-side TTL/expiry

    def __init__(self, client: Optional[SaihmMemoryClient] = None, **client_kwargs) -> None:
        self._client = client or SaihmMemoryClient(**client_kwargs)
        self._owns = client is None

    # ---- storage helpers -------------------------------------------------
    @staticmethod
    def _encode(ns: Tuple[str, ...], key: str, value: dict, created: str, updated: str) -> str:
        return json.dumps(
            {
                _ENVELOPE: 1,
                "namespace": list(ns),
                "key": key,
                "value": value,
                "created_at": created,
                "updated_at": updated,
            }
        )

    def _all(self) -> List[Tuple[str, dict]]:
        """Every SaihmStore record in the owned store, as (cell_id, record dict)."""
        out: List[Tuple[str, dict]] = []
        for m in self._client.recall():
            try:
                obj = json.loads(m.text)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(obj, dict) and obj.get(_ENVELOPE) and "namespace" in obj and "key" in obj:
                out.append((m.cell_id, obj))
        return out

    def _find(self, ns: Tuple[str, ...], key: str):
        for cid, rec in self._all():
            if tuple(rec["namespace"]) == ns and rec["key"] == key:
                return cid, rec
        return None

    @staticmethod
    def _to_item(rec: dict) -> Item:
        return Item(
            value=rec.get("value") or {},
            key=rec["key"],
            namespace=tuple(rec["namespace"]),
            created_at=_parse_dt(rec.get("created_at")),
            updated_at=_parse_dt(rec.get("updated_at")),
        )

    @staticmethod
    def _to_search_item(rec: dict) -> SearchItem:
        return SearchItem(
            namespace=tuple(rec["namespace"]),
            key=rec["key"],
            value=rec.get("value") or {},
            created_at=_parse_dt(rec.get("created_at")),
            updated_at=_parse_dt(rec.get("updated_at")),
            score=None,  # blind store: no server-side semantic ranking
        )

    # ---- op handlers -----------------------------------------------------
    def _do_get(self, op: GetOp) -> Optional[Item]:
        found = self._find(tuple(op.namespace), op.key)
        return self._to_item(found[1]) if found else None

    def _do_put(self, op: PutOp) -> None:
        ns = tuple(op.namespace)
        existing = self._find(ns, op.key)
        if op.value is None:  # delete
            if existing:
                self._client._forget_raw(existing[0])
            return None
        created = existing[1].get("created_at") if existing else _now_iso()
        if existing:
            self._client._forget_raw(existing[0])  # upsert: replace the prior cell
        self._client.remember(self._encode(ns, op.key, op.value, created, _now_iso()))
        return None

    def _do_search(self, op: SearchOp) -> List[SearchItem]:
        prefix = tuple(op.namespace_prefix)
        matched = [
            rec
            for _, rec in self._all()
            if tuple(rec["namespace"])[: len(prefix)] == prefix
            and (
                not op.filter
                or all(
                    _compare_values((rec.get("value") or {}).get(k), v)
                    for k, v in op.filter.items()
                )
            )
        ]
        matched.sort(key=lambda r: r.get("updated_at") or "", reverse=True)  # newest first
        page = matched[op.offset : op.offset + op.limit]
        return [self._to_search_item(rec) for rec in page]

    def _do_list(self, op: ListNamespacesOp) -> List[Tuple[str, ...]]:
        nss = [tuple(rec["namespace"]) for _, rec in self._all()]
        if op.match_conditions:
            nss = [ns for ns in nss if all(_does_match(c, ns) for c in op.match_conditions)]
        if op.max_depth is not None:
            nss = sorted({ns[: op.max_depth] for ns in nss})
        else:
            nss = sorted(set(nss))
        return nss[op.offset : op.offset + op.limit]

    # ---- BaseStore abstract ---------------------------------------------
    def batch(self, ops: Iterable[Op]) -> List[Result]:
        out: List[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                out.append(self._do_get(op))
            elif isinstance(op, PutOp):
                out.append(self._do_put(op))
            elif isinstance(op, SearchOp):
                out.append(self._do_search(op))
            elif isinstance(op, ListNamespacesOp):
                out.append(self._do_list(op))
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")
        return out

    async def abatch(self, ops: Iterable[Op]) -> List[Result]:
        # The SAIHM client is synchronous; offload so the event loop is never blocked.
        return await asyncio.to_thread(self.batch, list(ops))

    # ---- lifecycle -------------------------------------------------------
    @property
    def client(self) -> SaihmMemoryClient:
        return self._client

    def close(self) -> None:
        if self._owns:
            self._client.close()
