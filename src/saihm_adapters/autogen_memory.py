"""AutoGen integration: an ``autogen_core.memory.Memory`` backed by SAIHM.

Give any AutoGen agent a memory that is **yours**: portable across models, non-custodial
(sealed client-side by the bundled SAIHM Node sidecar — AutoGen never sees a key), and
provably erasable. Plug it straight into an ``AssistantAgent``:

    from autogen_agentchat.agents import AssistantAgent
    from saihm_adapters import SaihmMemory

    agent = AssistantAgent("assistant", model_client=..., memory=[SaihmMemory()])

AutoGen's ``Memory`` API is asynchronous; the underlying SAIHM client is synchronous and
blocking, so every call is offloaded to a worker thread (``asyncio.to_thread``) and never
blocks the running event loop.
"""
from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, List, Optional

from autogen_core import CancellationToken, Component, Image
from pydantic import BaseModel
from autogen_core.memory import (
    Memory,
    MemoryContent,
    MemoryMimeType,
    MemoryQueryResult,
    UpdateContextResult,
)
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage

from .client import SaihmMemoryClient


def _mime_value(mt: Any) -> str:
    if isinstance(mt, MemoryMimeType):
        return mt.value
    return str(mt) if mt is not None else MemoryMimeType.TEXT.value


def _mime_from_str(mt: Any) -> Any:
    # Round-trip mime back to the MemoryMimeType enum for known members, so adapter-written
    # cells compare equal to foreign/fallback cells (MemoryMimeType is a plain Enum, so
    # MemoryMimeType.TEXT != "text/plain"). Unknown custom mime strings are returned as-is.
    if isinstance(mt, MemoryMimeType):
        return mt
    try:
        return MemoryMimeType(mt)
    except (ValueError, TypeError):
        return mt if mt is not None else MemoryMimeType.TEXT


def _encode_content(content: Any):
    # Lossless round-trip for every MemoryContent.content type (str|bytes|dict|Image):
    # Image -> base64 via its own codec; bytes -> base64; JSON-native passes through; other -> str().
    if isinstance(content, Image):
        return content.to_base64(), "image"
    if isinstance(content, bytes):
        return base64.b64encode(content).decode("ascii"), "base64"
    if isinstance(content, (str, int, float, bool, type(None), list, dict)):
        return content, None
    return str(content), None


def _encode(c: MemoryContent) -> str:
    # Tag cells this adapter writes with a JSON envelope. The marker is NOT an authenticity
    # proof (stored content can mimic it); mime_type/metadata are therefore only trusted on
    # read for cells this adapter actually wrote — see `query`.
    content, enc = _encode_content(c.content)
    return json.dumps(
        {
            "_saihm": 1,
            "content": content,
            "enc": enc,
            "mime_type": _mime_value(c.mime_type),
            "metadata": c.metadata,
        }
    )


def _decode(text: str) -> MemoryContent:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        obj = None
    if isinstance(obj, dict) and obj.get("_saihm") and "content" in obj:
        content = obj["content"]
        enc = obj.get("enc")
        if enc == "base64" and isinstance(content, str):
            try:
                content = base64.b64decode(content.encode("ascii"), validate=True)
            except (ValueError, TypeError):
                pass
        elif enc == "image" and isinstance(content, str):
            try:
                content = Image.from_base64(content)
            except Exception:  # malformed image payload: fall back to the raw string
                pass
        return MemoryContent(
            content=content,
            mime_type=_mime_from_str(obj.get("mime_type")),
            metadata=obj.get("metadata"),
        )
    return MemoryContent(content=text, mime_type=MemoryMimeType.TEXT)  # opaque plain fact


class SaihmMemoryConfig(BaseModel):
    """Declarative config for :class:`SaihmMemory` (AutoGen Component serialization).

    Only the instance name is serialized — never the master secret or cell ids; live
    credentials come from the ``SAIHM_*`` environment at load time.
    """

    name: str = "saihm"


class SaihmMemory(Memory, Component[SaihmMemoryConfig]):
    """AutoGen ``Memory`` whose contents live in SAIHM, sealed client-side.

    :meth:`query` reads the whole owned store (your memory opens from any framework), but only
    contents *this* instance wrote carry their declared ``mime_type``/``metadata``; every other
    cell is read as opaque ``text/plain``, so memorized untrusted text can't forge typed
    metadata. :meth:`clear` is scoped: it crypto-shreds only what this instance added, even when
    several share one client — so it never wipes the rest of your memory by surprise. Pass a
    ``client`` to reuse a session, or omit it for a local blind sandbox (paid live endpoint via
    env — see :class:`~saihm_adapters.client.SaihmMemoryClient`).
    """

    component_type = "memory"
    component_provider_override = "saihm_adapters.autogen_memory.SaihmMemory"
    component_config_schema = SaihmMemoryConfig

    def __init__(
        self,
        client: Optional[SaihmMemoryClient] = None,
        name: Optional[str] = None,
        **client_kwargs: Any,
    ) -> None:
        self._client = client or SaihmMemoryClient(**client_kwargs)
        self._owns = client is None
        self._name = name or "saihm"
        self._ids: List[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def client(self) -> SaihmMemoryClient:
        return self._client

    def _to_config(self) -> SaihmMemoryConfig:
        return SaihmMemoryConfig(name=self._name)

    @classmethod
    def _from_config(cls, config: SaihmMemoryConfig) -> "SaihmMemory":
        # Reconstruct from declarative config; live creds come from SAIHM_* env, not config.
        return cls(name=config.name)

    async def add(
        self, content: MemoryContent, cancellation_token: Optional[CancellationToken] = None
    ) -> None:
        cid = await asyncio.to_thread(self._client.remember, _encode(content))
        self._ids.append(cid)

    async def query(
        self,
        query: str | MemoryContent = "",
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        # A blind store returns the full owned working set; rank/filter client-side as needed.
        _ = query, cancellation_token, kwargs
        rows = await asyncio.to_thread(self._client.recall)
        own = set(self._ids)
        results = [
            _decode(m.text)
            if m.cell_id in own
            else MemoryContent(content=m.text, mime_type=MemoryMimeType.TEXT)
            for m in rows
        ]
        return MemoryQueryResult(results=results)

    async def update_context(
        self, model_context: ChatCompletionContext
    ) -> UpdateContextResult:
        """Inject the owned memory into a model context as a single SystemMessage."""
        memories = (await self.query()).results
        if not memories:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))
        lines = [f"{i}. {str(m.content)}" for i, m in enumerate(memories, 1)]
        block = "\nRelevant memory (SAIHM, in chronological order):\n" + "\n".join(lines) + "\n"
        await model_context.add_message(SystemMessage(content=block))
        return UpdateContextResult(memories=MemoryQueryResult(results=memories))

    async def clear(self) -> None:
        """Crypto-shred every memory this instance added (irreversible; GDPR Art. 17)."""
        ids, self._ids = self._ids, []

        def _shred() -> None:
            for fid in ids:
                self._client._forget_raw(fid)

        await asyncio.to_thread(_shred)

    async def forget(self, cell_id: str) -> bool:
        """Crypto-shred a single memory by its full cell id."""
        return await asyncio.to_thread(self._client.forget, cell_id)

    async def close(self) -> None:
        if self._owns:
            await asyncio.to_thread(self._client.close)
