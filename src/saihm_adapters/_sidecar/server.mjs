#!/usr/bin/env node
// SAIHM memory as an MCP server — drop persistent, client-side-encrypted memory into
// Claude Code, Cursor, or any MCP host. Exposes four tools (remember / recall / forget /
// status) that seal client-side via @saihm/mcp-server-pro; the endpoint only sees ciphertext.
//
// Modes:
//   LIVE     (recommended)  set SAIHM_ENDPOINT_URL=https://saihm.net/mcp. Your identity comes from
//                           the key file `npx -y @saihm/mcp-server-pro free-join` saves (found
//                           automatically, or named by SAIHM_MASTER_SECRET_FILE)
//                           -> durable, hosted, blind memory (https://saihm.net)
//   SANDBOX  (default)      no SAIHM_ENDPOINT_URL -> a local, in-process blind endpoint; memory
//                           lasts only as long as this process. Good for trying it with zero signup.
//
// If the sidecar cannot start (for example, no identity on this device yet, an identity whose
// sign-in was never finished, or a dependency that fails to load), it writes one line,
// BOOT_MARKER + JSON {kind, message, reason?}, to stderr and exits, so the Python client can
// raise a typed error that carries the message.

import { randomBytes } from 'node:crypto';

// Loaded inside the boot try block below, so a dependency that fails to load is reported the
// same way as any other start-up failure.
let Server, StdioServerTransport, CallToolRequestSchema, ListToolsRequestSchema;
let deriveIdentity, toHex, fromHex, SaihmProClient, startSandbox;

const BOOT_MARKER = 'saihm-sidecar-error ';
const NO_IDENTITY = /^No SAIHM (memory yet on this device|identity is configured)/;
// How long start-up waits to learn whether a live identity is active before going ahead anyway.
const IDENTITY_CHECK_MS = 20_000;
// Sign-in refusals about the one-time challenge. The service lost, expired or already used it
// (for example, it restarted between the two sign-in steps), which says nothing about the
// identity. A fresh sign-in clears them, so start-up tries once more and never stops on them.
const CHALLENGE_REASONS = new Set(['unknown_nonce', 'nonce_expired', 'nonce_replay']);

class BootError extends Error {
  constructor(kind, message, reason) {
    super(message);
    this.kind = kind;
    this.reason = reason;
  }
}

// The reason code of a sign-in refusal (401 verification_failed), '' when it names none, or null
// when the error is not a sign-in refusal.
function refusalReason(e) {
  if (e?.status !== 401 || e?.code !== 'verification_failed') return null;
  return /\(verification_failed: ([a-z0-9_]+)\)/.exec(String(e?.message ?? ''))?.[1] ?? '';
}

// Live mode: ask the service whether this identity is active before reporting ready. A key that
// free-join saved before its sign-in was finished is refused at sign-in with 401
// verification_failed; report that as an identity that is not active, so the caller gets a typed
// error at start-up instead of a client that fails every call. A refusal about the challenge
// gets one more sign-in. Anything else (no network, a timeout, a server error, a used-up
// allowance, or a challenge refused twice) does not stop start-up: the calls report it.
async function checkIdentityActive(client) {
  let timer;
  const timeUp = new Promise((resolve) => { timer = setTimeout(resolve, IDENTITY_CHECK_MS); });
  try {
    for (let attempt = 1; attempt <= 2; attempt++) {
      const probe = client.status();
      probe.catch(() => {}); // after the time limit below, a late failure is the calls' to report
      try {
        await Promise.race([probe, timeUp]);
        return;
      } catch (e) {
        const reason = refusalReason(e);
        if (reason === null) return;
        if (!CHALLENGE_REASONS.has(reason)) {
          throw new BootError('identity-not-active', String(e?.message ?? e), reason);
        }
      }
    }
  } finally {
    clearTimeout(timer);
  }
}

async function makeClient() {
  if (process.env.SAIHM_ENDPOINT_URL) {
    const client = SaihmProClient.bootFromEnv();
    await checkIdentityActive(client);
    return { client, mode: 'live', close: async () => {} };
  }
  // A sandbox identity lasts one process. Keep it off disk: no erasure feed under ~/.saihm for a
  // throwaway tenant, and no recall cache (which also silences the client's notice about it).
  process.env.SAIHM_ERASURE_FEED = '0';
  process.env.SAIHM_RECALL_CACHE = '0';
  const { url, close } = await startSandbox();
  const master = process.env.SAIHM_MASTER_SECRET_HEX
    ? fromHex(process.env.SAIHM_MASTER_SECRET_HEX.trim())
    : randomBytes(32);
  const client = new SaihmProClient(url, `Bearer ${toHex(deriveIdentity(master).agentIdHash)}`, master, { tier: 'SANDBOX' });
  return { client, mode: 'sandbox', close };
}

let booted;
try {
  ({ Server } = await import('@modelcontextprotocol/sdk/server/index.js'));
  ({ StdioServerTransport } = await import('@modelcontextprotocol/sdk/server/stdio.js'));
  ({ CallToolRequestSchema, ListToolsRequestSchema } = await import('@modelcontextprotocol/sdk/types.js'));
  ({ deriveIdentity, toHex, fromHex } = await import('@saihm/client-pro'));
  ({ SaihmProClient } = await import('@saihm/mcp-server-pro'));
  ({ startSandbox } = await import('./sandbox.mjs'));
  booted = await makeClient();
} catch (e) {
  const message = String(e?.message ?? e);
  const kind = e instanceof BootError ? e.kind : NO_IDENTITY.test(message) ? 'no-identity' : 'startup';
  const out = { kind, message };
  if (e instanceof BootError && e.reason) out.reason = e.reason;
  process.stderr.write(BOOT_MARKER + JSON.stringify(out) + '\n', () => process.exit(1));
  await new Promise(() => {}); // wait for the write callback to exit
}
const { client, mode, close } = booted;

const TOOLS = [
  {
    name: 'saihm_remember',
    description: 'Store a memory in SAIHM (sealed client-side before it leaves this process). Pass cellId to update an existing memory.',
    inputSchema: { type: 'object', properties: { content: { type: 'string', description: 'The text to remember.' }, cellId: { type: 'string', description: 'Optional: update this existing memory.' } }, required: ['content'] },
  },
  {
    name: 'saihm_recall',
    description: 'Recall SAIHM memories (decrypted client-side), optionally filtered by a keyword.',
    inputSchema: { type: 'object', properties: { query: { type: 'string', description: 'Optional keyword filter.' } } },
  },
  {
    name: 'saihm_forget',
    description: 'Permanently crypto-shred a memory by cellId (irreversible; GDPR Art. 17 erasure).',
    inputSchema: { type: 'object', properties: { cellId: { type: 'string' } }, required: ['cellId'] },
  },
  {
    name: 'saihm_status',
    description: 'Non-custodial status: tier, active memory count, and custody (no plaintext).',
    inputSchema: { type: 'object', properties: {} },
  },
];

const server = new Server({ name: 'saihm-adapters', version: '0.1.3' }, { capabilities: { tools: {} } });

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const { name, arguments: args = {} } = req.params;
  const run = async () => {
    if (name === 'saihm_remember') {
      const r = await client.remember(String(args.content ?? ''), args.cellId ? { cellId: String(args.cellId) } : {});
      return `Remembered. cellId=${r.cellId} seq=${r.seq}`;
    }
    if (name === 'saihm_recall') {
      const cells = await client.recall(args.query ? String(args.query) : undefined);
      // Machine-readable: full cell ids (so callers can forget what they recall) and
      // newline-safe (JSON), since this sidecar is driven by the Python client.
      return JSON.stringify(cells.map((c) => ({ id: c.cellId, text: c.plaintext })));
    }
    if (name === 'saihm_forget') {
      const r = await client.forget(String(args.cellId ?? ''));
      return `Forgot ${r.cellId} — crypto-shredded (${r.complete}).`;
    }
    if (name === 'saihm_status') {
      const s = await client.status();
      return `tier=${s.tier} memories=${s.activeShardCount} custody=${s.custody}`;
    }
    return `unknown tool: ${name}`;
  };
  try {
    return { content: [{ type: 'text', text: await run() }] };
  } catch (e) {
    return { content: [{ type: 'text', text: `error${e?.code ? ' (' + e.code + ')' : ''}: ${e?.message ?? e}` }], isError: true };
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
// stdout is the JSON-RPC channel — log only to stderr.
process.stderr.write(`saihm-adapters MCP server ready (mode=${mode}).\n`);

const bye = async () => { try { await close(); } finally { process.exit(0); } };
process.on('SIGINT', bye);
process.on('SIGTERM', bye);
