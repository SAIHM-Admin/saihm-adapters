"""SAIHM memory for Python — a thin, synchronous client over the SAIHM MCP node sidecar.

All cryptography happens in the Node sidecar (``server.mjs`` / ``@saihm/mcp-server-pro``):
every cell is sealed client-side (ML-DSA-65 + AES-256-GCM, its key wrapped under *your*
master secret) before it leaves the process, and the endpoint only ever sees ciphertext.
Python performs no cryptography and, when the key is in a key file, never reads your key; it
drives the sidecar over MCP stdio. This keeps one audited crypto implementation instead of porting it to a second language.

This client is **synchronous and blocking**. Inside an async app (FastAPI, etc.), call its
methods in a worker thread, e.g. ``await loop.run_in_executor(None, client.remember, text)``.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
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


# The sidecar writes this prefix + JSON {kind, message} to stderr when it cannot start.
_BOOT_MARKER = "saihm-sidecar-error "
_STDERR_TAIL = 8 * 1024
# server.mjs prints this once its MCP transport is connected.
_READY_LINE = "saihm-adapters MCP server ready"

_PYTHON_JOIN_HINT = (
    "No SAIHM identity on this machine yet. From Python, use a key file: run "
    "`npx -y @saihm/mcp-server-pro free-join` once (a one-time browser sign-in) and start the "
    "client again. The sidecar finds the key file free-join saves; if you keep it elsewhere, set "
    "SAIHM_MASTER_SECRET_FILE to its full path. With the key file, Python never reads your key. "
    "To try SAIHM without an identity, leave SAIHM_ENDPOINT_URL unset for a local sandbox whose "
    "memory ends with the process. The server's own message follows; its chat steps apply to "
    "MCP hosts such as editors."
)


class SaihmStartupError(RuntimeError):
    """The Node sidecar could not start. :attr:`message` is the sidecar's own explanation."""

    def __init__(self, message: str, kind: str = "startup") -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind


class SaihmNoIdentityError(SaihmStartupError):
    """Live mode (``SAIHM_ENDPOINT_URL`` set) found no SAIHM identity on this device.

    :attr:`message` carries the server's message; ``str(error)`` puts how to create an identity
    from Python first, then that message."""

    def __init__(self, message: str) -> None:
        super().__init__(message, kind="no-identity")

    def __str__(self) -> str:
        return f"{_PYTHON_JOIN_HINT}\n\n{self.message}"


class _StderrTee:
    """Pipe for the sidecar's stderr: echoes each line to ``sys.stderr`` as before, keeps a
    bounded tail, and holds back the boot-error marker line so it can be raised instead."""

    def __init__(self) -> None:
        r, w = os.pipe()
        self.writer = os.fdopen(w, "w")
        self._reader = os.fdopen(r, "r", encoding="utf-8", errors="replace")
        self.boot_error: Optional[dict] = None
        self.tail = ""
        self._thread = threading.Thread(target=self._pump, name="saihm-stderr", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        with self._reader:
            for line in self._reader:
                if line.startswith(_BOOT_MARKER):
                    try:
                        self.boot_error = json.loads(line[len(_BOOT_MARKER):])
                        continue
                    except ValueError:
                        pass
                self.tail = (self.tail + line)[-_STDERR_TAIL:]
                try:
                    sys.stderr.write(line)
                    sys.stderr.flush()
                except Exception:
                    pass

    def close_writer(self) -> None:
        try:
            self.writer.close()
        except OSError:
            pass

    def join(self, timeout: float) -> None:
        self._thread.join(timeout)


class SaihmTimeout(TimeoutError):
    """A call exceeded the timeout. For ``remember``, the write may still have landed; the
    client-generated :attr:`cell_id` lets you :meth:`forget` it, so it is never an
    unforgettable orphan."""

    def __init__(self, message: str, cell_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.cell_id = cell_id


# --- Module-level loop driver so the background thread/task never holds a strong reference
# --- to the client; that lets a dropped client be garbage-collected and reaped (no leak).
async def _serve(node, server, env, ready, stop_box, session_box, exc_box, tee):
    stop = asyncio.Event()
    stop_box.append(stop)
    params = StdioServerParameters(command=node, args=[server], env=env, cwd=str(Path(server).parent))
    try:
        async with stdio_client(params, errlog=tee.writer) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                session_box.append(session)
                ready.set()
                await stop.wait()
    except BaseException as e:  # surface startup failures back to __init__
        exc_box.append(e)
    finally:
        tee.close_writer()  # the sidecar has exited: let the stderr reader reach EOF
        ready.set()


def _run_loop(loop, node, server, env, ready, stop_box, session_box, exc_box, tee):
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_serve(node, server, env, ready, stop_box, session_box, exc_box, tee))
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

    Sandbox mode by default: with no ``SAIHM_ENDPOINT_URL``, memory lives in a local,
    in-process *blind* endpoint and is gone when this client closes. For the real, hosted,
    blind, non-custodial endpoint, run ``npx -y @saihm/mcp-server-pro free-join`` once on this
    machine, then set ``SAIHM_ENDPOINT_URL=https://saihm.net/mcp``. The sidecar finds the key
    file free-join saves; set ``SAIHM_MASTER_SECRET_FILE`` if you keep it elsewhere. The key
    stays in the file and the sidecar; Python never reads it. With no identity, live mode
    raises :class:`SaihmNoIdentityError`.

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
        tee = _StderrTee()
        self._thread = threading.Thread(
            target=_run_loop,
            args=(self._loop, node, self._server, env, self._ready,
                  self._stop_box, self._session_box, self._exc_box, tee),
            name="saihm-mcp",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._exc_box:
            self._thread.join(timeout=5)
            tee.join(timeout=5)
            boot = tee.boot_error
            if isinstance(boot, dict) and boot.get("message"):
                if boot.get("kind") == "no-identity":
                    raise SaihmNoIdentityError(str(boot["message"])) from None
                raise SaihmStartupError(str(boot["message"]), str(boot.get("kind") or "startup")) from None
            if tee.tail.strip():
                if _READY_LINE in tee.tail:
                    what = "The SAIHM sidecar started, but the MCP handshake with it failed"
                else:
                    what = "The SAIHM sidecar exited during start-up"
                raise SaihmStartupError(
                    f"{what} ({type(self._exc_box[0]).__name__}). Its last output:\n" + tee.tail.strip()
                ) from self._exc_box[0]
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
