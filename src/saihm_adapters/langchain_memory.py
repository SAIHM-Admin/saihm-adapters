"""LangChain integration: a ``BaseChatMessageHistory`` backed by SAIHM.

Use it on its own, or plug it into ``RunnableWithMessageHistory`` to give any LangChain
chain a memory that is **yours**: portable across models, non-custodial (sealed
client-side by the SAIHM sidecar — LangChain never sees a key), and provably erasable.

    from saihm_adapters import SaihmChatMessageHistory
    history = SaihmChatMessageHistory()            # local blind sandbox by default
    history.add_user_message("My name is Dana.")
    history.messages                               # -> [HumanMessage("My name is Dana.")]
    history.clear()                                # crypto-shreds the messages it added
"""
from __future__ import annotations

import json
from typing import List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .client import SaihmMemoryClient

_ROLE_TO_CLS = {"human": HumanMessage, "ai": AIMessage, "system": SystemMessage}


def _encode(m: BaseMessage) -> str:
    # Tag messages this adapter writes with a JSON envelope. The marker is NOT an
    # authenticity proof (stored content can mimic it); role is therefore only trusted on
    # read for cells this adapter actually wrote — see `messages`.
    return json.dumps({"_saihm": 1, "role": m.type, "content": m.content})


def _decode(text: str) -> BaseMessage:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        obj = None
    if isinstance(obj, dict) and obj.get("_saihm") and "role" in obj and "content" in obj:
        return _ROLE_TO_CLS.get(obj["role"], HumanMessage)(content=str(obj["content"]))
    return HumanMessage(content=text)  # opaque: a plain fact, no role inference


class SaihmChatMessageHistory(BaseChatMessageHistory):
    """Chat-message history whose messages live in SAIHM, sealed client-side.

    :pyattr:`messages` reads the whole owned store (your memory opens from any framework), but
    only messages *this* history wrote carry their role; any other cell is read as opaque user
    content, so memorized untrusted text can't forge a privileged turn. :meth:`clear` is also
    scoped: it only crypto-shreds the messages *this* history added, even when several share one
    client — so it never wipes the rest of your memory by surprise. Pass a ``client`` to reuse a
    session, or omit it to spawn a local blind sandbox (paid live endpoint via env — see
    :class:`~saihm_adapters.client.SaihmMemoryClient`).
    """

    def __init__(self, client: Optional[SaihmMemoryClient] = None, **client_kwargs) -> None:
        self._client = client or SaihmMemoryClient(**client_kwargs)
        self._owns = client is None
        self._ids: List[str] = []

    @property
    def client(self) -> SaihmMemoryClient:
        return self._client

    @property
    def messages(self) -> List[BaseMessage]:
        # Read the whole owned store, but only trust the ROLE of messages this history wrote;
        # every other cell (including arbitrary facts) is opaque user content, so memorized
        # untrusted text cannot forge a privileged system/assistant turn. (Destruction via
        # clear() is likewise scoped to what this history added.)
        own = set(self._ids)
        return [
            _decode(m.text) if m.cell_id in own else HumanMessage(content=m.text)
            for m in self._client.recall()
        ]

    def add_message(self, message: BaseMessage) -> None:
        self._ids.append(self._client.remember(_encode(message)))

    def clear(self) -> None:
        """Crypto-shred every message this history added (irreversible; GDPR Art. 17)."""
        for fid in self._ids:
            self._client._forget_raw(fid)
        self._ids = []

    def forget(self, cell_id: str) -> bool:
        """Crypto-shred a single memory by its full cell id."""
        return self._client.forget(cell_id)

    def close(self) -> None:
        if self._owns:
            self._client.close()
