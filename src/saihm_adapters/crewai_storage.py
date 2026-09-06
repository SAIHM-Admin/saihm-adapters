"""CrewAI integration: a SAIHM-backed ``StorageBackend`` for CrewAI's unified memory.

Route CrewAI's long-term memory through SAIHM so it is **yours**: portable across models and
frameworks, non-custodial (sealed client-side by the bundled SAIHM Node sidecar — CrewAI never
sees a key), and provably erasable (GDPR Art. 17). Register it once at startup:

    from crewai.memory.storage.factory import set_memory_storage_factory
    from saihm_adapters import SaihmStorageBackend

    set_memory_storage_factory(lambda spec: SaihmStorageBackend() if spec == "saihm" else None)

Any CrewAI memory built afterwards for the ``"saihm"`` storage spec is then backed by SAIHM;
return ``None`` for other specs to defer to CrewAI's built-in selection.

SAIHM is a *blind* store: the endpoint only ever holds ciphertext, so it cannot run a
server-side vector index. Retrieval therefore happens client-side — :meth:`search` ranks by
cosine over the embedding CrewAI stored on each record (persisted in the sealed cell), and
falls back to recency when no embedding is present (e.g. the offline demo). Exact recall (:meth:`get_record`,
:meth:`list_records`) and erasure (:meth:`delete`, :meth:`reset` → crypto-shred) are always
exact.
"""
from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from crewai.memory.types import MemoryRecord, ScopeInfo

from .client import SaihmMemoryClient

_ENVELOPE = "_saihm_crewai"


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _created_key(r: MemoryRecord) -> str:
    # ISO-8601 strings sort chronologically and sidestep naive/aware datetime comparison errors.
    return r.created_at.isoformat() if isinstance(r.created_at, datetime) else ""


def _aware(t: datetime) -> datetime:
    """Coerce a datetime to aware-UTC so naive/aware values compare safely (min/max/<)."""
    return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)


class SaihmStorageBackend:
    """A CrewAI ``StorageBackend`` (Protocol) whose records live in SAIHM, sealed client-side.

    Each CrewAI ``MemoryRecord`` becomes one encrypted SAIHM cell. The backend manages only the
    records it wrote (typed CrewAI records); other cells in the same owned store — e.g. facts
    written from the LangChain or AutoGen adapters — are left untouched. Pass a ``client`` to
    reuse a session, or omit it for a local blind sandbox (paid live endpoint via env — see
    :class:`~saihm_adapters.client.SaihmMemoryClient`).
    """

    def __init__(self, client: Optional[SaihmMemoryClient] = None, **client_kwargs: Any) -> None:
        self._client = client or SaihmMemoryClient(**client_kwargs)
        self._owns = client is None

    # ---- encode / decode -------------------------------------------------
    @staticmethod
    def _encode(record: MemoryRecord) -> str:
        # MemoryRecord.embedding is exclude=True, so model_dump() drops it; persist it
        # explicitly so client-side cosine search works live (CrewAI supplies the embedding).
        data = record.model_dump(mode="json")
        data["embedding"] = record.embedding
        return json.dumps({_ENVELOPE: 1, "record": data})

    def _all(self) -> List[Tuple[str, MemoryRecord]]:
        """Every CrewAI record in the owned store, as (cell_id, record)."""
        out: List[Tuple[str, MemoryRecord]] = []
        for m in self._client.recall():
            try:
                obj = json.loads(m.text)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(obj, dict) and obj.get(_ENVELOPE) and isinstance(obj.get("record"), dict):
                try:
                    out.append((m.cell_id, MemoryRecord.model_validate(obj["record"])))
                except Exception:
                    continue
        return out

    # ---- filters ---------------------------------------------------------
    @staticmethod
    def _in_scope(r: MemoryRecord, scope_prefix: Optional[str]) -> bool:
        # Path-boundary match: "/demo" covers "/demo" and "/demo/x" but NOT "/demo2" or
        # "/demonstrate". Critical for delete()/reset() — a naive startswith would crypto-shred
        # sibling scopes irreversibly.
        if scope_prefix is None or scope_prefix in ("", "/"):
            return True  # whole store
        p = scope_prefix.rstrip("/")
        sp = r.scope or "/"
        return sp == p or sp.startswith(p + "/")

    @staticmethod
    def _cat_ok(r: MemoryRecord, categories: Optional[List[str]]) -> bool:
        return not categories or any(c in (r.categories or []) for c in categories)

    @staticmethod
    def _meta_ok(r: MemoryRecord, metadata_filter: Optional[Dict[str, Any]]) -> bool:
        if not metadata_filter:
            return True
        meta = r.metadata or {}
        return all(meta.get(k) == v for k, v in metadata_filter.items())

    # ---- StorageBackend protocol (sync) ----------------------------------
    def save(self, records: List[MemoryRecord]) -> None:
        if not records:
            return
        existing = {r.id: cid for cid, r in self._all()}
        for rec in records:
            old = existing.get(rec.id)
            if old:
                self._client._forget_raw(old)  # upsert: replace any record with the same id
            existing[rec.id] = self._client.remember(self._encode(rec))  # upsert within-batch too

    def update(self, record: MemoryRecord) -> None:
        self.save([record])

    def get_record(self, record_id: str) -> Optional[MemoryRecord]:
        for _, r in self._all():
            if r.id == record_id:
                return r
        return None

    def list_records(
        self, scope_prefix: Optional[str] = None, limit: int = 200, offset: int = 0
    ) -> List[MemoryRecord]:
        recs = [r for _, r in self._all() if self._in_scope(r, scope_prefix)]
        recs.sort(key=_created_key, reverse=True)  # newest first
        return recs[offset : offset + limit]

    def search(
        self,
        query_embedding: List[float],
        scope_prefix: Optional[str] = None,
        categories: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[Tuple[MemoryRecord, float]]:
        cands = [
            r
            for _, r in self._all()
            if self._in_scope(r, scope_prefix)
            and self._cat_ok(r, categories)
            and self._meta_ok(r, metadata_filter)
        ]
        if query_embedding and any(r.embedding for r in cands):
            scored = [(r, _cosine(query_embedding, r.embedding or [])) for r in cands]
        else:
            # blind store, no embeddings available: rank by recency (newest -> 1.0)
            ordered = sorted(cands, key=_created_key, reverse=True)
            n = len(ordered) or 1
            scored = [(r, (n - i) / n) for i, r in enumerate(ordered)]
        scored = [(r, s) for r, s in scored if s >= min_score]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:limit]

    @staticmethod
    def _older(r: MemoryRecord, cutoff: datetime) -> bool:
        t = r.created_at
        if not isinstance(t, datetime):
            return False
        return _aware(t) < _aware(cutoff)  # normalize both so naive/aware can't drop matches

    def delete(
        self,
        scope_prefix: Optional[str] = None,
        categories: Optional[List[str]] = None,
        record_ids: Optional[List[str]] = None,
        older_than: Optional[datetime] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> int:
        n = 0
        for cid, r in self._all():
            if record_ids is not None and r.id not in record_ids:
                continue
            if not self._in_scope(r, scope_prefix):
                continue
            if not self._cat_ok(r, categories):
                continue
            if not self._meta_ok(r, metadata_filter):
                continue
            if older_than is not None and not self._older(r, older_than):
                continue
            self._client._forget_raw(cid)  # crypto-shred
            n += 1
        return n

    def reset(self, scope_prefix: Optional[str] = None) -> None:
        """Crypto-shred every record in scope (None = all CrewAI records). Irreversible."""
        self.delete(scope_prefix=scope_prefix)

    def count(self, scope_prefix: Optional[str] = None) -> int:
        return sum(1 for _, r in self._all() if self._in_scope(r, scope_prefix))

    def list_scopes(self, parent: str = "/") -> List[str]:
        base = parent if parent.endswith("/") else parent + "/"
        children = set()
        for _, r in self._all():
            sp = r.scope or "/"
            if sp.startswith(base) and len(sp) > len(base):
                children.add(base + sp[len(base) :].strip("/").split("/")[0])
        return sorted(children)

    def list_categories(self, scope_prefix: Optional[str] = None) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for _, r in self._all():
            if not self._in_scope(r, scope_prefix):
                continue
            for c in r.categories or []:
                out[c] = out.get(c, 0) + 1
        return out

    def get_scope_info(self, scope: str) -> ScopeInfo:
        recs = [r for _, r in self._all() if self._in_scope(r, scope)]
        times = [_aware(r.created_at) for r in recs if isinstance(r.created_at, datetime)]
        cats = sorted({c for r in recs for c in (r.categories or [])})
        oldest, newest = (min(times), max(times)) if times else (None, None)
        return ScopeInfo(
            path=scope,
            record_count=len(recs),
            categories=cats,
            oldest_record=oldest,
            newest_record=newest,
            child_scopes=self.list_scopes(scope),
        )

    # ---- StorageBackend protocol (async) ---------------------------------
    async def asave(self, records: List[MemoryRecord]) -> None:
        await asyncio.to_thread(self.save, records)

    async def asearch(self, *args: Any, **kwargs: Any) -> List[Tuple[MemoryRecord, float]]:
        return await asyncio.to_thread(lambda: self.search(*args, **kwargs))

    async def adelete(self, *args: Any, **kwargs: Any) -> int:
        return await asyncio.to_thread(lambda: self.delete(*args, **kwargs))

    # ---- lifecycle -------------------------------------------------------
    @property
    def client(self) -> SaihmMemoryClient:
        return self._client

    def close(self) -> None:
        if self._owns:
            self._client.close()
