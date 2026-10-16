"""Shared test fixtures for the offline blind sandbox — framework-agnostic (core client only).

Everything runs against the OFFLINE blind sandbox (no account, no live endpoint): a fresh
SaihmMemoryClient with no SAIHM_ENDPOINT_URL spins up a local in-process blind endpoint via the
bundled Node sidecar. One sandbox client is shared for the whole session (a Node subprocess
spawn is not free); each test starts from a clean slate via the autouse ``_wipe`` fixture.

Per-framework fixtures (memory / history / retriever / backend / …) live in the conftest of each
``tests/<framework>/`` subdirectory, so importing this module needs only the core client — not
any framework — and the two frameworks that both name a class ``SaihmMemory`` never collide here.
"""
from __future__ import annotations

import pytest

from saihm_adapters import SaihmMemoryClient


@pytest.fixture(scope="session")
def client():
    c = SaihmMemoryClient()  # sandbox mode (no env) -> local blind endpoint
    try:
        yield c
    finally:
        c.close()


@pytest.fixture(autouse=True)
def _wipe(client):
    """Erase every cell before each test so the shared sandbox is isolated per-test."""
    for m in client.recall():
        client._forget_raw(m.cell_id)
    yield


# --- A local stand-in for the hosted endpoint, for the refusal paths -------------------------
# The sandbox accepts every well-formed call, so it cannot show what happens when the service
# refuses one. This server answers the sign-in and the memory calls with whatever reply a test
# sets, in the hosted endpoint's wire format, so the real sidecar and the installed mcp release
# carry the refusal end to end.
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# A reply that never comes: the request waits until the test ends.
HANG = "hang"


class _EndpointHandler(BaseHTTPRequestHandler):
    def _reply(self, status, body):
        data = json.dumps(body).encode()
        try:
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client gave up on this request

    def do_GET(self):
        ep = self.server.endpoint
        ep.seen.append(("GET", self.path))
        if self.path == "/api/onboard/challenge":
            return self._reply(200, {"nonce": secrets.token_hex(32)})
        self._reply(404, {"error": "not_found"})

    def do_POST(self):
        ep = self.server.endpoint
        self.rfile.read(int(self.headers.get("content-length") or 0))
        ep.seen.append(("POST", self.path))
        reply = ep.next_reply(ep.onboard_reply if self.path == "/api/onboard" else ep.call_reply)
        if reply == HANG:
            ep.released.wait(120)
            reply = (503, {"error": "unavailable"})
        self._reply(*reply)

    def log_message(self, *args):  # keep the test output clean
        pass


class FakeEndpoint:
    """``url`` is the endpoint; ``seen`` lists the requests it answered. ``onboard_reply`` and
    ``call_reply`` are each a ``(status, json)`` reply, a list of replies answered in turn (the
    last one repeats), or ``HANG``."""

    HANG = HANG

    def __init__(self):
        self.seen = []
        self.onboard_reply = (500, {"error": "unset"})
        self.call_reply = (500, {"error": "unset"})
        self.released = threading.Event()
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _EndpointHandler)
        self._server.daemon_threads = True
        self._server.endpoint = self
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/mcp"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def next_reply(self, reply):
        with self._lock:
            if isinstance(reply, list):
                return reply.pop(0) if len(reply) > 1 else reply[0]
            return reply

    def close(self):
        self.released.set()
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def fake_endpoint():
    srv = FakeEndpoint()
    try:
        yield srv
    finally:
        srv.close()


_IDENTITY_VARS = (
    "SAIHM_IDENTITY", "SAIHM_IDENTITY_PASSPHRASE", "SAIHM_MASTER_SECRET_FILE",
    "SAIHM_MASTER_SECRET_HEX", "SAIHM_HOME", "SAIHM_SELF_JOIN", "SAIHM_AUTH_HEADER",
    "SAIHM_TIER", "SAIHM_PAYMENT_METHOD", "SAIHM_ENDPOINT_URL", "SAIHM_RECALL_CACHE",
)


@pytest.fixture
def clean_identity_env(monkeypatch):
    """No identity setting from the developer's own shell may reach a live-mode test."""
    for name in _IDENTITY_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def make_key():
    """Write a throwaway key file shaped like the one free-join saves (mode 600)."""
    def make(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32), encoding="utf-8")
        path.chmod(0o600)
        return path
    return make
