# saihm-adapters

**One memory store your agents own — portable across every model and framework, sealed
client-side, and provably erasable.**

<a href="https://saihm.net/overview"><img src="https://saihm.net/media/saihm-short-overview-play.jpg" alt="Watch: SAIHM in 6 minutes" width="480"></a>

**New to SAIHM?** [Watch the 6-minute overview](https://saihm.net/overview) (captions and transcript), or [read the SAIHM manual (PDF)](https://saihm.net/manual).

`saihm-adapters` gives a Python AI agent a single persistent memory backed by
[SAIHM](https://saihm.net): every cell is encrypted *inside your process* by a bundled
Node sidecar. With your key in a key file, **Python never reads your key**, and the store
belongs to you and your users. Point
LangChain, LlamaIndex, a RAG retriever, CrewAI, and AutoGen at the *same* store and the memory
travels with your user across frameworks and across model providers. Erase a memory and it is
cryptographically shredded (GDPR Art. 17): its key is destroyed, so it cannot be read again.

- **Owned & sovereign** — you hold the keys; the server stores ciphertext it cannot read.
- **Portable across models & frameworks** — one source of truth behind every adapter below.
- **Provably erasable** — deletion is cryptographic and irreversible, and observable in recall.
- **Drop-in** — each adapter implements the framework's own memory/history/retriever/storage
  interface, so it slots into code you already have.

## Install

The core client needs only [`mcp`](https://pypi.org/project/mcp/). Each framework adapter is an
optional extra, imported lazily — **install only what you use**:

```bash
pip install saihm-adapters                 # core client (any Python app)
pip install "saihm-adapters[langchain]"    # + LangChain chat history
pip install "saihm-adapters[langgraph]"    # + LangGraph BaseStore
pip install "saihm-adapters[llamaindex]"   # + LlamaIndex memory
pip install "saihm-adapters[rag]"          # + LlamaIndex retriever for RAG
pip install "saihm-adapters[crewai]"       # + CrewAI storage backend
pip install "saihm-adapters[autogen]"      # + AutoGen memory
pip install "saihm-adapters[all]"          # every adapter
```

> **Node.js ≥ 20** must be on `PATH`. The wheel ships only the tiny sidecar sources
> (`server.mjs` / `sandbox.mjs` + a pinned lockfile); its Node dependencies are installed once
> into a per-user cache on first use (`npm ci`). Python performs no cryptography — the Node
> sidecar seals every cell. For air-gapped or CI installs, pre-provision the sidecar and set
> `SAIHM_SIDECAR_DIR`; `SAIHM_SKIP_BOOTSTRAP=1` forbids any network install.

## Quickstart

The core client works on its own — remember, recall, and erase, from any Python app:

```python
from saihm_adapters import SaihmMemoryClient

client = SaihmMemoryClient()          # no SAIHM_ENDPOINT_URL: a local sandbox (see below)
cid = client.remember("Ada prefers metric units.")
for cell in client.recall("units"):
    print(cell.text)
client.forget(cid)                    # cryptographic erase
client.close()
```

With no `SAIHM_ENDPOINT_URL` set, the client runs against a **local sandbox**: a blind endpoint
inside the sidecar process, with no sign-up and no calls to SAIHM. (The first use of the package
runs `npm ci` once to install the sidecar.) It is a throwaway store, so its
memories are gone when the client closes. Use it to try the API, then go live.

## Go live

Your memory on the hosted service is tied to an identity: a key that stays on your machine.
Create one once with the SAIHM command-line tool (Node.js ≥ 20):

```bash
npx -y @saihm/mcp-server-pro free-join
```

It shows a link and a short code, asks you to confirm with a quick GitHub sign-in in your
browser, and then prints where it saved your key (by default `~/.saihm/free-identity.key`).
Then point the client at the service:

```bash
export SAIHM_ENDPOINT_URL=https://saihm.net/mcp
# Only if you keep the key somewhere other than the default:
export SAIHM_MASTER_SECRET_FILE=/full/path/to/free-identity.key
```

The sidecar reads the key file itself, so your Python process never sees the key. Prefer the
key file to pasting a hex secret into an environment variable.

With the default free-join key file, the client also keeps a local recall cache: memories this
device has opened are stored unencrypted beside the key, in a file only you can read (mode 600),
so later recalls fetch only what changed. `forget` removes a memory from it, and
`SAIHM_RECALL_CACHE=0` turns the cache off. With a key file in any other place the cache is
off, and the first recall prints a one-time note about it on stderr; `SAIHM_RECALL_CACHE=0`
silences that note too. On a paid plan, follow the
[@saihm/mcp-server-pro README](https://github.com/SAIHM-Admin/saihm-mcp-server-pro#readme)
to set your tier, then use the same two variables.

If live mode finds no identity, the client raises `SaihmNoIdentityError` with the server's own
explanation and these steps:

```python
from saihm_adapters import SaihmMemoryClient, SaihmNoIdentityError

try:
    client = SaihmMemoryClient()
except SaihmNoIdentityError as e:
    print(e)                          # the free-join steps, then the server's message
```

Any other sidecar start-up failure that the sidecar can explain raises `SaihmStartupError`
(the parent class) with its message. A sidecar that exits during start-up without one raises
`SaihmStartupError` with its last output.

## Per-framework adapters

Each adapter accepts a shared `client=` so a whole app (or a whole crew) reads and writes one
store:

```python
from saihm_adapters import SaihmMemoryClient
client = SaihmMemoryClient()

# LangChain — a BaseChatMessageHistory
from saihm_adapters import SaihmChatMessageHistory
history = SaihmChatMessageHistory(client=client)

# LangGraph — a BaseStore for long-term, cross-thread memory
from saihm_adapters import SaihmStore
store = SaihmStore(client=client)

# RAG — a LlamaIndex BaseRetriever over a corpus you own
from saihm_adapters import SaihmRetriever
retriever = SaihmRetriever(client=client, similarity_top_k=3)

# CrewAI (1.15+) — a memory StorageBackend, installed with set_memory_storage_factory
from saihm_adapters import SaihmStorageBackend
backend = SaihmStorageBackend(client=client)

# AutoGen — an autogen_core.memory.Memory
from saihm_adapters.autogen_memory import SaihmMemory as AutoGenMemory
memory = AutoGenMemory(client=client)

# LlamaIndex — a BaseMemory
from saihm_adapters.llamaindex_memory import SaihmMemory as LlamaIndexMemory
memory = LlamaIndexMemory.from_defaults(client=client)
```

> **Why `SaihmMemory` is imported from a submodule:** AutoGen and LlamaIndex each define a
> class named `SaihmMemory`, and they are different classes. Importing from
> `saihm_adapters.autogen_memory` / `saihm_adapters.llamaindex_memory` keeps the two unambiguous.
> The other adapter classes have unique names and import straight from the top level.

## How it works

Python drives a bundled **Node MCP sidecar** over stdio. Your plaintext is sealed
(ML-KEM-768 + AES-256-GCM, signed with ML-DSA-65) inside that sidecar before anything leaves
the process; the SAIHM service stores only ciphertext. Recall decrypts in-process. Because the
store is keyed to *you*, the same memory is reachable from any framework and any model — and an
erase removes the key material, so the data cannot be recovered.

## Live demos

Runnable notebooks and end-to-end examples: **<https://saihm-admin.github.io/saihm-demos/>**

## License

Apache-2.0.
