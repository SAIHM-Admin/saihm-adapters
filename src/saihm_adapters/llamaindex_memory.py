"""LlamaIndex integration: a ``BaseMemory`` backed by SAIHM.

Drop it into a LlamaIndex chat engine / agent (``memory=SaihmMemory.from_defaults()``)
to get memory you own: portable across models, non-custodial (sealed client-side by the
SAIHM sidecar — LlamaIndex never sees a key), and provably erasable.

    from saihm_adapters import SaihmMemory
    from llama_index.core.llms import ChatMessage, MessageRole
    mem = SaihmMemory.from_defaults()              # local blind sandbox by default
    mem.put(ChatMessage(role=MessageRole.USER, content="My name is Dana."))
    mem.get_all()                                  # -> [ChatMessage(USER, "My name is Dana.")]
    mem.reset()                                    # crypto-shreds the messages it added
"""
from __future__ import annotations

import json
from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import BaseMemory

from .client import SaihmMemoryClient


def _encode(m: ChatMessage) -> str:
    # Tag messages this adapter writes with a JSON envelope. The marker is NOT an
    # authenticity proof (stored content can mimic it); role is therefore only trusted on
    # read for cells this adapter actually wrote — see `get_all`.
    role = m.role.value if isinstance(m.role, MessageRole) else str(m.role)
    return json.dumps({"_saihm": 1, "role": role, "content": m.content})


def _decode(text: str) -> ChatMessage:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        obj = None
    if isinstance(obj, dict) and obj.get("_saihm") and "role" in obj and "content" in obj:
        try:
            return ChatMessage(role=MessageRole(obj["role"]), content=str(obj["content"]))
        except ValueError:
            return ChatMessage(role=MessageRole.USER, content=str(obj["content"]))
    return ChatMessage(role=MessageRole.USER, content=text)  # opaque: a plain fact


class SaihmMemory(BaseMemory):
    """LlamaIndex memory whose messages live in SAIHM, sealed client-side.

    :meth:`get_all` reads the whole owned store (your memory opens from any framework), but
    only messages *this* memory wrote carry their role; any other cell is read as opaque user
    content, so memorized untrusted text can't forge a privileged role. :meth:`reset` is also
    scoped: it only crypto-shreds the messages *this* memory added, even when several share one
    client — so it never wipes the rest of your memory by surprise.
    """

    _client: SaihmMemoryClient = PrivateAttr()
    _ids: List[str] = PrivateAttr(default_factory=list)
    _owns: bool = PrivateAttr(default=False)

    def __init__(self, client: Optional[SaihmMemoryClient] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = client or SaihmMemoryClient()
        self._owns = client is None
        self._ids = []

    @classmethod
    def class_name(cls) -> str:
        return "SaihmMemory"

    @classmethod
    def from_defaults(  # type: ignore[override]
        cls,
        chat_history: Optional[List[ChatMessage]] = None,
        llm: Any = None,
        client: Optional[SaihmMemoryClient] = None,
        **kwargs: Any,
    ) -> "SaihmMemory":
        inst = cls(client=client)
        if chat_history:
            inst.set(chat_history)
        return inst

    @property
    def client(self) -> SaihmMemoryClient:
        return self._client

    def get(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        return self.get_all()

    def get_all(self) -> List[ChatMessage]:
        # Read the whole owned store, but only trust the ROLE of messages this memory wrote;
        # every other cell (including arbitrary facts) is opaque user content, so memorized
        # untrusted text cannot forge a privileged role. (Destruction via reset() is likewise
        # scoped to what this memory added.)
        own = set(self._ids)
        return [
            _decode(m.text) if m.cell_id in own else ChatMessage(role=MessageRole.USER, content=m.text)
            for m in self._client.recall()
        ]

    def put(self, message: ChatMessage) -> None:
        self._ids.append(self._client.remember(_encode(message)))

    def set(self, messages: List[ChatMessage]) -> None:
        for m in messages:
            self.put(m)

    def reset(self) -> None:
        """Crypto-shred every message this memory added (irreversible; GDPR Art. 17)."""
        for fid in self._ids:
            self._client._forget_raw(fid)
        self._ids = []

    def forget(self, cell_id: str) -> bool:
        """Crypto-shred a single memory by its full cell id."""
        return self._client.forget(cell_id)
