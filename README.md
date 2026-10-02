# saihm-adapters

**One memory store your agents own — portable across every model and framework, sealed
client-side, and provably erasable.**

<a href="https://saihm.net/overview"><img src="https://saihm.net/media/saihm-short-overview-play.jpg" alt="Watch: SAIHM in 6 minutes" width="480"></a>

**New to SAIHM?** [Watch the 6-minute overview](https://saihm.net/overview) (captions and transcript), or [read the SAIHM manual (PDF)](https://saihm.net/manual).

`saihm-adapters` gives a Python AI agent a single persistent memory backed by
[SAIHM](https://saihm.coti.global): every cell is encrypted *inside your process* by a bundled
Node sidecar — **Python never holds a key** — so the store is yours, not a vendor's. Point
LangChain, LlamaIndex, a RAG retriever, CrewAI, and AutoGen at the *same* store and the memory
travels with your user across frameworks and across model providers. Erase a memory and it is
cryptographically shredded (GDPR Art. 17), not merely de-indexed.

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

client = SaihmMemoryClient()          # reads SAIHM_* env for your account
cid = client.remember("Ada prefers metric units.")
for cell in client.recall("units"):
    print(cell.content)
client.forget(cid)                    # cryptographic erase
```

### Per-framework adapters

Each adapter accepts a shared `client=` so a whole app (or a whole crew) reads and writes one
store:

```python
# LangChain — a BaseChatMessageHistory
from saihm_adapters import SaihmChatMessageHistory
history = SaihmChatMessageHistory(client=client)

# LangGraph — a BaseStore for long-term, cross-thread memory
from saihm_adapters import SaihmStore
store = SaihmStore(client=client)

# RAG — a LlamaIndex BaseRetriever over a corpus you own
from saihm_adapters import SaihmRetriever
retriever = SaihmRetriever(client=client, similarity_top_k=3)

# CrewAI — a Storage backend for ExternalMemory
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

Runnable notebooks and end-to-end examples: **<https://citw2.github.io/saihm-demos/>**

## License

Apache-2.0.
