"""SAIHM memory for Python — a thin, synchronous client over the SAIHM MCP node sidecar.

All cryptography happens in the Node sidecar (``server.mjs`` / ``@saihm/mcp-server-pro``):
every cell is sealed client-side (ML-DSA-65 + AES-256-GCM, its key wrapped under *your*
master secret) before it leaves the process, and the endpoint only ever sees ciphertext.
Python holds no keys and performs no cryptography — it simply drives the sidecar over MCP
stdio. This keeps one audited crypto implementation instead of porting it to a second language.

This client is **synchronous and blocking**. Inside an async app (FastAPI, etc.), call its
methods in a worker thread, e.g. ``await loop.run_in_executor(None, client.remember, text)``.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from secrets import token_hex
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_CELL_RE = re.compile(r"cellId=([0-9a-f]+)")

# Default per-call timeout (seconds). A live-but-wedged sidecar must not hang the caller.
DEFAULT_TIMEOUT = 60.0


@dataclass(frozen=True)
class Memory:
    """One recalled memory. ``cell_id`` is the **full** cell id — pass it straight to
    :meth:`SaihmMemoryClient.forget`."""

    cell_id: str
    text: str


class SaihmTimeout(TimeoutError):
    """A call exceeded the timeout. For ``remember``, the write may still have landed; the
    client-generated :attr:`cell_id` lets you :meth:`forget` it, so it is never an
    unforgettable orphan."""

    def __init__(self, message: str, cell_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.cell_id = cell_id


# --- Module-level loop driver so the background thread/task never holds a strong reference
# --- to the client; that lets a dropped client be garbage-collected and reaped (no leak).
async def _serve(node, server, env, ready, stop_box, session_box, exc_box):
    stop = asyncio.Event()
    stop_box.append(stop)
    params = StdioServerParameters(command=node, args=[server], env=env, cwd=str(Path(server).parent))
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                session_box.append(session)
                ready.set()
                await stop.wait()
    except BaseException as e:  # surface startup failures back to __init__
        exc_box.append(e)
        ready.set()


def _run_loop(loop, node, server, env, ready, stop_box, session_box, exc_box):
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_serve(node, server, env, ready, stop_box, session_box, exc_box))
    finally:
        loop.close()


def _reap(loop, thread, stop_box):
    """Stop the background loop and join its thread. Used by close() and registered as a
    finalizer so a dropped/forgotten client never leaks the node subprocess."""
    try:
        if stop_box and loop.is_running():
            loop.call_soon_threadsafe(stop_box[0].set)
    except Exception:
        pass
    if thread.is_alive():
        thread.join(timeout=5)


class SaihmMemoryClient:
    """Synchronous facade over the SAIHM MCP sidecar.

    Spawns ``node server.mjs`` once and keeps a single long-lived MCP session on a
    background event loop, so blocking ``remember`` / ``recall`` / ``forget`` / ``status``
    calls all share the same in-process memory.

    Sandbox mode by default (a local, in-process *blind* endpoint that lasts for the life
    of this client). For the real, hosted, blind, non-custodial endpoint, pass ``env`` with
    ``SAIHM_ENDPOINT_URL`` + ``SAIHM_MASTER_SECRET_HEX`` + either ``SAIHM_TIER=FREE`` (free tier;
    run ``npx -y @saihm/mcp-server-pro free-join`` once first) or ``SAIHM_AUTH_HEADER`` (paid Pro; join
    https://saihm.coti.global/join).

    Always :meth:`close` it (or use it as a context manager). A forgotten client is reaped
    on garbage collection / interpreter exit, but explicit close is cleaner.
    """

    def __init__(
        self,
        server_path: Optional[str] = None,
        env: Optional[dict] = None,
        node: str = "node",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if server_path:
            self._server = str(Path(server_path).resolve())
        else:
            from ._bootstrap import ensure_sidecar
            self._server = ensure_sidecar()  # provisions the bundled Node sidecar on first use
        self._timeout = timeout
        self._closed = False
        self._lock = threading.Lock()  # serialize calls against close()
        # Merge over the full environment so PATH (and thus `node`) is always present.
        env = {**os.environ, **(env or {})}
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._stop_box: list = []
        self._session_box: list = []
        self._exc_box: list = []
        self._thread = threading.Thread(
            target=_run_loop,
            args=(self._loop, node, self._server, env, self._ready,
                  self._stop_box, self._session_box, self._exc_box),
            name="saihm-mcp",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._exc_box:
            self._thread.join(timeout=5)
            raise self._exc_box[0]
        # Safety net: reap subprocess/thread if the caller forgets to close().
        self._finalizer = weakref.finalize(self, _reap, self._loop, self._thread, self._stop_box)

    def _call(self, coro):
        with self._lock:
            if self._closed:
                coro.close()  # avoid "coroutine was never awaited"
                raise RuntimeError("SaihmMemoryClient is closed")
            fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
            try:
                return fut.result(timeout=self._timeout)
            except TimeoutError:
                fut.cancel()
                raise SaihmTimeout(f"SAIHM sidecar did not respond within {self._timeout}s") from None
            except Exception as e:
                raise RuntimeError(f"SAIHM sidecar call failed: {str(e) or type(e).__name__}") from e

    async def _tool(self, name: str, args: dict) -> str:
        session = self._session_box[0]
        res = await session.call_tool(name, args or {})
        text = "\n".join(getattr(c, "text", "") for c in res.content)
        if getattr(res, "isError", False):
            raise RuntimeError(text or f"{name} failed")
        return text

    # --- public, synchronous API ---
    def remember(self, content: str, cell_id: Optional[str] = None) -> str:
        """Seal and store ``content``; returns the full cell id.

        The id is generated client-side, so even if the call times out you still hold the id
        (on :class:`SaihmTimeout`) and can :meth:`forget` it — a timed-out write never
        becomes an unforgettable orphan."""
        cid = cell_id or token_hex(16)
        try:
            text = self._call(self._tool("saihm_remember", {"content": content, "cellId": cid}))
        except SaihmTimeout as e:
            e.cell_id = cid
            raise
        m = _CELL_RE.search(text)
        return m.group(1) if m else cid

    def recall(self, query: Optional[str] = None) -> list[Memory]:
        """Return decrypted memories with their full cell ids, optionally filtered by ``query``.

        The id on each :class:`Memory` is the full id — pass it straight to :meth:`forget`."""
        text = self._call(self._tool("saihm_recall", {"query": query} if query else {}))
        try:
            rows = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(rows, list):
            return []
        return [
            Memory(cell_id=str(r["id"]), text=str(r["text"]))
            for r in rows
            if isinstance(r, dict) and "id" in r and "text" in r
        ]

    def _forget_raw(self, cell_id: str) -> None:
        """Fire the crypto-shred without before/after verification (so erasing many cells —
        e.g. an adapter's clear()/reset() — stays linear, not quadratic)."""
        self._call(self._tool("saihm_forget", {"cellId": cell_id}))

    def forget(self, cell_id: str) -> bool:
        """Crypto-shred a memory by its full cell id (irreversible; GDPR Art. 17).

        Returns ``True`` only if a memory with this id existed and is now gone, so a typo'd
        or already-erased id reports ``False`` rather than a misleading success."""
        existed = any(m.cell_id == cell_id for m in self.recall())
        self._forget_raw(cell_id)
        gone = not any(m.cell_id == cell_id for m in self.recall())
        return existed and gone

    def status(self) -> str:
        """One-line non-custodial status (tier, active memory count, custody)."""
        return self._call(self._tool("saihm_status", {}))

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._finalizer()  # idempotent: stop the loop + join the thread

    def __enter__(self) -> "SaihmMemoryClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
