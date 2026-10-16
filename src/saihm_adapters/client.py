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
import concurrent.futures
import json
import os
import re
import shutil
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

_FREE_JOIN = "npx -y @saihm/mcp-server-pro free-join"

# What a developer reads when live mode has no usable identity. The sidecar's own message is
# written for MCP hosts such as editors, so it stays on the error's ``message`` attribute.
_NO_IDENTITY_TEXT = (
    "SAIHM live mode is on (SAIHM_ENDPOINT_URL is set), but this machine has no SAIHM identity "
    "yet.\n"
    f"Create one once, in a terminal:  {_FREE_JOIN}\n"
    "Open the link it shows, sign in to GitHub, enter the code and approve, then start the "
    "client again. The client finds the key file that free-join saves; if you keep it somewhere "
    "else, set SAIHM_MASTER_SECRET_FILE to its full path. Python never reads the key.\n"
    "To try SAIHM first, unset SAIHM_ENDPOINT_URL: the client then runs a local sandbox whose "
    "memory ends when the client closes."
)
_SELF_JOIN_OFF_TEXT = (
    "\nSAIHM_SELF_JOIN=0 is set, so the key file free-join saves is not read: set "
    "SAIHM_MASTER_SECRET_FILE to its full path, or unset SAIHM_SELF_JOIN."
)
_NOT_ACTIVE_TEXT = {
    "no_free_entitlement": (
        "Your SAIHM identity is not active yet: its key file is on this machine, but its sign-in "
        "was not approved.\n"
        f"Run  {_FREE_JOIN}  again: it reuses your key and shows a new code. Open the link, sign "
        "in to GitHub, enter the code and approve, then start the client again.\n"
        "If this key belongs to a paid plan, set SAIHM_TIER and SAIHM_PAYMENT_METHOD for that "
        "plan instead."
    ),
    "no_active_subscription": (
        "SAIHM found no active subscription for this identity on the plan that SAIHM_TIER names.\n"
        "Check that SAIHM_TIER names the plan this identity was bought on, then start the "
        "client again."
    ),
}
_NOT_ACTIVE_DEFAULT = (
    "SAIHM did not accept this identity, so live mode cannot start.\n"
    f"If you created it with free-join, run  {_FREE_JOIN}  again: it reuses your key and shows a "
    "new code. Open the link, sign in to GitHub, enter the code and approve, then start the "
    "client again."
)


class SaihmStartupError(RuntimeError):
    """The client could not start. ``str(error)`` says what to do; :attr:`message` is the
    explanation from the sidecar (or from the client, for failures before the sidecar runs).

    :attr:`kind` names the failure: ``"node-missing"`` (no Node.js to run the sidecar),
    ``"not-provisioned"`` (the sidecar is not installed and installing is switched off, or
    ``SAIHM_SIDECAR_DIR`` has no sidecar), ``"install-failed"`` (installing the sidecar's Node
    dependencies failed), ``"no-identity"`` / ``"identity-not-active"`` (see
    :class:`SaihmNoIdentityError`) or ``"startup"`` (anything else)."""

    def __init__(self, message: str, kind: str = "startup") -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind


class SaihmNoIdentityError(SaihmStartupError):
    """Live mode (``SAIHM_ENDPOINT_URL`` set) has no usable SAIHM identity on this device.

    :attr:`kind` is ``"no-identity"`` when there is no key at all, and
    ``"identity-not-active"`` when there is a key that SAIHM does not accept yet, for example
    because its free-join sign-in was not finished; :attr:`reason` then carries the service's
    reason code. ``str(error)`` is written for you, the developer: what is missing and the
    command that fixes it. :attr:`message` keeps the sidecar's own text for logs."""

    def __init__(self, message: str, kind: str = "no-identity", reason: str = "") -> None:
        super().__init__(message, kind=kind)
        self.reason = reason

    def __str__(self) -> str:
        if self.kind == "identity-not-active":
            return _NOT_ACTIVE_TEXT.get(self.reason, _NOT_ACTIVE_DEFAULT)
        text = _NO_IDENTITY_TEXT
        if "SAIHM_SELF_JOIN=0" in self.message:
            text += _SELF_JOIN_OFF_TEXT
        return text


def _resolve_node(node: str, path: Optional[str]) -> str:
    """The Node.js executable that runs the sidecar, or a typed error that names the fix."""
    found = shutil.which(node, path=path)
    if found:
        return found
    raise SaihmStartupError(
        f"Node.js was not found: no executable {node!r} on PATH. The SAIHM sidecar needs "
        "Node.js >= 20. Install it from https://nodejs.org, or pass node='/full/path/to/node' "
        "to SaihmMemoryClient.",
        kind="node-missing",
    )


def _describe(exc: BaseException) -> str:
    """``Type: message`` of the first underlying error, looking inside exception groups."""
    while getattr(exc, "exceptions", None):
        exc = exc.exceptions[0]
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _call_failed(res) -> bool:
    """Whether a tool result carries the MCP error flag. mcp 1.x names it ``isError``; mcp 2.x
    names it ``is_error`` and has no ``isError`` attribute. A result with neither is refused, so
    a future rename fails loudly instead of passing a refused call off as a success."""
    for name in ("is_error", "isError"):
        flag = getattr(res, name, None)
        if flag is not None:
            return bool(flag)
    raise RuntimeError(f"unrecognized MCP tool result ({type(res).__name__}): it has no error flag")


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
    stays in the file and the sidecar; Python never reads it. With no identity, or with a key
    whose free-join sign-in was not finished, live mode raises :class:`SaihmNoIdentityError`.
    Every other start-up failure raises :class:`SaihmStartupError`, and a call the service
    refuses raises ``RuntimeError`` with the service's explanation.

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
            try:
                self._server = ensure_sidecar()  # provisions the bundled Node sidecar on first use
            except SaihmStartupError:
                raise
            except Exception as exc:  # e.g. a cache directory that cannot be written
                raise SaihmStartupError(
                    f"The SAIHM sidecar could not be installed ({_describe(exc)}).",
                    kind="install-failed",
                ) from exc
        self._timeout = timeout
        self._closed = False
        self._lock = threading.Lock()  # serialize calls against close()
        # Merge over the full environment so PATH (and thus `node`) is always present.
        env = {**os.environ, **(env or {})}
        # Find Node.js the way the sidecar's launch will (on the PATH it gets), so a missing
        # Node.js is a typed start-up error rather than an OS error from deep inside the launch.
        node = _resolve_node(node, env.get("PATH"))
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
                kind = str(boot.get("kind") or "startup")
                if kind in ("no-identity", "identity-not-active"):
                    raise SaihmNoIdentityError(
                        str(boot["message"]), kind=kind, reason=str(boot.get("reason") or "")
                    ) from None
                raise SaihmStartupError(str(boot["message"]), kind) from None
            if tee.tail.strip():
                if _READY_LINE in tee.tail:
                    what = "The SAIHM sidecar started, but the MCP handshake with it failed"
                else:
                    what = "The SAIHM sidecar exited during start-up"
                raise SaihmStartupError(
                    f"{what} ({type(self._exc_box[0]).__name__}). Its last output:\n" + tee.tail.strip()
                ) from self._exc_box[0]
            raise SaihmStartupError(
                f"The SAIHM sidecar could not start ({_describe(self._exc_box[0])})."
            ) from self._exc_box[0]
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
            # On Python 3.10 the futures timeout is its own class, not the built-in TimeoutError.
            except (TimeoutError, concurrent.futures.TimeoutError):
                fut.cancel()
                raise SaihmTimeout(f"SAIHM sidecar did not respond within {self._timeout}s") from None
            except Exception as e:
                raise RuntimeError(f"SAIHM sidecar call failed: {str(e) or type(e).__name__}") from e

    async def _tool(self, name: str, args: dict) -> str:
        session = self._session_box[0]
        res = await session.call_tool(name, args or {})
        text = "\n".join(getattr(c, "text", "") for c in getattr(res, "content", None) or ())
        # A refused call (no active identity, a used-up allowance, an expired plan, an auth
        # error) must raise; returning its text as a result would lose writes without a word.
        if _call_failed(res):
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
            rows = None
        if not isinstance(rows, list):
            # The sidecar always answers a recall with a JSON list. Anything else is a failure
            # to report, never an empty memory.
            raise RuntimeError("SAIHM sidecar call failed: the recall result is not a JSON list")
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
