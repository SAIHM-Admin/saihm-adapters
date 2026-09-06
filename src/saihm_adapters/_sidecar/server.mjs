#!/usr/bin/env node
// SAIHM memory as an MCP server — drop persistent, client-side-encrypted memory into
// Claude Code, Cursor, or any MCP host. Exposes four tools (remember / recall / forget /
// status) that seal client-side via @saihm/mcp-server-pro; the endpoint only sees ciphertext.
//
// Modes:
//   LIVE     (recommended)  set SAIHM_ENDPOINT_URL + SAIHM_MASTER_SECRET_HEX + SAIHM_TIER=FREE (free tier; run `npx -y @saihm/mcp-server-pro free-join` once first) OR SAIHM_AUTH_HEADER (paid Pro)
//                           -> durable, hosted, blind memory (join: https://saihm.coti.global/join)
//   SANDBOX  (default)      no SAIHM_ENDPOINT_URL -> a local, in-process blind endpoint; memory
//                           lasts for the editor session. Great for trying it with zero signup.

import { randomBytes } from 'node:crypto';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { deriveIdentity, toHex, fromHex } from '@saihm/client-pro';
import { SaihmProClient } from '@saihm/mcp-server-pro';
import { startSandbox } from './sandbox.mjs';

async function makeClient() {
  if (process.env.SAIHM_ENDPOINT_URL) {
    return { client: SaihmProClient.bootFromEnv(), mode: 'live', close: async () => {} };
  }
  const { url, close } = await startSandbox();
  const master = process.env.SAIHM_MASTER_SECRET_HEX
    ? fromHex(process.env.SAIHM_MASTER_SECRET_HEX.trim())
    : randomBytes(32);
  const client = new SaihmProClient(url, `Bearer ${toHex(deriveIdentity(master).agentIdHash)}`, master, { tier: 'SANDBOX' });
  return { client, mode: 'sandbox', close };
}

const { client, mode, close } = await makeClient();

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

const server = new Server({ name: 'saihm-adapters', version: '0.1.0' }, { capabilities: { tools: {} } });

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
