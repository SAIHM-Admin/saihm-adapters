// A LOCAL, BLIND sandbox endpoint for this demo.
//
// It speaks the same wire as the hosted SAIHM `/mcp` endpoint, but it stores
// only OPAQUE CIPHERTEXT — exactly like the production service. Every
// cryptographic operation runs in the CLIENT (`@saihm/mcp-server-pro` +
// `@saihm/client-pro`); this server never holds a key and cannot read one byte
// of your memory. That non-custodial, "blind operator" property is the whole
// point of SAIHM — made concrete here in ~130 lines you can read end to end.
//
// It exists so you can run the demo fully offline with zero signup. To talk to
// the real, hosted service instead, set SAIHM_ENDPOINT_URL=https://saihm.coti.global/mcp
// (see the README "Go live" section) and join at https://saihm.coti.global/join.

import { createServer } from 'node:http';

export function startSandbox() {
  const store = new Map();   // tenant -> Map<cellId, wire>   (wire = opaque encoded envelope)
  const hwm = new Map();     // `${tenant}/${cellId}` -> bigint  (monotonic anti-replay)
  const graves = new Set();  // `${tenant}/${cellId}`            (a forgotten id stays retired)

  const tenantStore = (t) => {
    let m = store.get(t);
    if (!m) { m = new Map(); store.set(t, m); }
    return m;
  };
  const k = (t, c) => `${t}/${c}`;

  const json = (res, status, body) => {
    const s = JSON.stringify(body);
    res.writeHead(status, { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(s)) });
    res.end(s);
  };

  const server = createServer((req, res) => {
    void (async () => {
      try {
        if (req.method !== 'POST') return json(res, 405, { error: 'method_not_allowed' });
        const auth = req.headers['authorization'] || '';
        const tenant = auth.startsWith('Bearer ') ? auth.slice(7) : '';
        if (!tenant) return json(res, 401, { error: 'unauthorized' });

        let raw = '';
        for await (const chunk of req) raw += chunk;
        const { method, params } = JSON.parse(raw || '{}');

        if (method === 'saihm_remember') {
          const wire = params?.wire;
          if (!wire || typeof wire.cellId !== 'string' || typeof wire.seq !== 'string') {
            return json(res, 400, { error: 'bad_request' });
          }
          // Attribution: the signed agentIdHash inside the (opaque) envelope must match the tenant.
          if (wire.agentIdHash !== tenant) return json(res, 403, { error: 'BLIND_ATTRIBUTION_MISMATCH' });
          const key = k(tenant, wire.cellId);
          if (graves.has(key)) return json(res, 409, { error: 'BLIND_STALE_SEQ' });
          const seq = BigInt(wire.seq);
          const prev = hwm.get(key);
          if (prev !== undefined && seq <= prev) return json(res, 409, { error: 'BLIND_STALE_SEQ' });
          tenantStore(tenant).set(wire.cellId, wire);
          hwm.set(key, seq);
          return json(res, 200, {
            cellId: wire.cellId,
            shardId: 'sandbox-' + wire.cellId.slice(0, 8),
            seq: wire.seq,
            commitmentHash: wire.publicMeta?.commitmentHash ?? wire.cellId,
          });
        }

        if (method === 'saihm_recall') {
          const m = tenantStore(tenant);
          if (params && typeof params.cellId === 'string') {
            const wire = m.get(params.cellId);
            return json(res, 200, wire ? { found: true, wire } : { found: false });
          }
          const rows = [];
          for (const [cellId, wire] of m) rows.push({ cellId, found: true, wire });
          return json(res, 200, rows);
        }

        if (method === 'saihm_forget') {
          const id = params?.id;
          const existed = tenantStore(tenant).delete(id);
          graves.add(k(tenant, id)); // anti-resurrection: the id cannot be re-used to bring it back
          return json(res, 200, {
            cellId: id,
            shardId: 'sandbox',
            complete: true,
            sharesPurged: 0,
            steps: [{ step: 'destroy_dek', success: existed, detail: 'sandbox crypto-shred: wrapped DEK destroyed' }],
            epoch: String(Math.floor(Date.now() / 1000)),
          });
        }

        if (method === 'saihm_status') {
          const m = tenantStore(tenant);
          return json(res, 200, {
            agentIdHashHex: tenant,
            tier: 'SANDBOX',
            activeShardCount: m.size,
            activeSharingContracts: 0,
            bfsi: 0, bfsi_R: '0', bfsi_M: '0',
            prsInstrumented: false,
            snapshotEpoch: String(Math.floor(Date.now() / 1000)),
            custody: 'non-custodial',
          });
        }

        if (method === 'saihm_share' || method === 'saihm_revoke_share') {
          return json(res, 501, { error: 'not_implemented_in_sandbox' });
        }
        if (method === 'saihm_governance_propose' || method === 'saihm_governance_vote') {
          return json(res, 403, { error: 'governance_unavailable' });
        }
        return json(res, 400, { error: 'unknown_method' });
      } catch {
        json(res, 500, { error: 'sandbox_error' });
      }
    })();
  });

  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      resolve({
        url: `http://127.0.0.1:${port}/mcp`,
        close: () => new Promise((r) => server.close(() => r())),
      });
    });
  });
}
