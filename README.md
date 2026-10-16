# saihm-adapters — memory for Python AI agents

**Portable memory for Python AI agents.** Your agents remember what matters — across
sessions, across models, across frameworks. One store backs LangChain, LangGraph,
LlamaIndex, CrewAI and AutoGen, and you can erase any memory for good.

[![PyPI version](https://img.shields.io/pypi/v/saihm-adapters.svg)](https://pypi.org/project/saihm-adapters/)
[![license](https://img.shields.io/pypi/l/saihm-adapters.svg)](https://github.com/SAIHM-Admin/saihm-adapters/blob/main/LICENSE)
[![Star on GitHub](https://img.shields.io/badge/GitHub-Star-181717?logo=github&logoColor=white)](https://github.com/SAIHM-Admin/saihm-adapters)
[![Sponsor SAIHM](https://img.shields.io/badge/Sponsor-SAIHM-ea4aaa?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/SAIHM-Admin)
[![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/@SAIHM-Memory?sub_confirmation=1)
[![Follow on X](https://img.shields.io/badge/X-Follow-000000?logo=x&logoColor=white)](https://x.com/SAIHMemory)

<a href="https://saihm.net/overview"><img src="https://saihm.net/media/saihm-short-overview-play.jpg" alt="Watch: SAIHM in 6 minutes" width="480"></a>

**New to SAIHM?** [Watch the 6-minute overview](https://saihm.net/overview) (captions and transcript).
**Prefer to read?** [The SAIHM manual](https://saihm.net/manual) (PDF) is a decision-maker's guide to SAIHM.

AI agents forget everything when the process ends. `saihm-adapters` gives yours a memory
that doesn't — one store that follows your user to a different framework, a different
model, or a different company's product.

**Encrypted at rest, by default.** Every memory is encrypted on your own machine, inside a
bundled Node.js sidecar, before it goes anywhere, and SAIHM stores only that ciphertext.
Nobody else can read your memories — not the storage, not SAIHM. With your key in a key
file, Python never reads the key. With the default free-join key file, the client also keeps
a local copy of your memories, readable only by your user account, so recall is fast; set
`SAIHM_RECALL_CACHE=0` to keep none ([details](#how-it-works)).

## Start free — one command

```sh
npx -y @saihm/mcp-server-pro free-join
```

No card and no wallet. You need Node.js 20 or later, and a browser and a GitHub account
once, for a quick sign-in that confirms you're a real person:

1. The command creates your key on this machine and shows a link and a short code.
2. Open the link, sign in to GitHub, enter the code and approve.
3. The command finishes by itself and shows where your key is saved.

**No GitHub account?** Start at <https://saihm.net/free> instead: you sign in with
Google, and the page gives you your new key. Save it to a file only you can read, and set
`SAIHM_MASTER_SECRET_FILE` to that file's full path where your app runs. Never paste the
key itself into code or a settings file. Your key is ready, so skip free-join: a join would
make a second, different key.

**If a join is interrupted**, join again on the same machine: it reuses the key it
already created, so you keep the same identity. Until the sign-in is approved, the key is
not active, and the client tells you so when it starts. To cancel a join, stop it and
don't enter the code; the code expires on its own. Keep the key file either way.

Then [add SAIHM to your Python app](#add-saihm-to-your-python-app). To try it first, the
[quickstart](#quickstart) runs in a local sandbox with no sign-up.

## Add SAIHM to your Python app

### Install

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

> **Node.js ≥ 20** must be on `PATH`. The wheel ships only the small sidecar sources and a
> pinned lockfile; their Node dependencies are installed once into a per-user cache on first
> use (`npm ci`). Python performs no cryptography — the Node sidecar seals every cell. For
> air-gapped or CI installs, pre-provision the sidecar and set `SAIHM_SIDECAR_DIR`;
> `SAIHM_SKIP_BOOTSTRAP=1` forbids any network install.

### Quickstart

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

### Go live

Once [free-join](#start-free--one-command) has finished, point the client at the service:

```bash
export SAIHM_ENDPOINT_URL=https://saihm.net/mcp
# Only if you keep the key somewhere other than the default (~/.saihm/free-identity.key):
export SAIHM_MASTER_SECRET_FILE=/full/path/to/free-identity.key
```

The sidecar reads the key file itself, so your Python process never sees the key. Prefer the
key file to pasting a hex secret into an environment variable.

When it starts in live mode, the client checks with the service that your identity is
active. If this machine has no identity, or its free-join sign-in was not approved, the
client raises `SaihmNoIdentityError`, and its message says what to run:

```python
from saihm_adapters import SaihmMemoryClient, SaihmNoIdentityError

try:
    client = SaihmMemoryClient()
except SaihmNoIdentityError as e:
    print(e)                          # what is missing, and the command that fixes it
```

Every other start-up failure raises `SaihmStartupError`, the parent class. A call that the
service refuses, for example once the free allowance is used up, raises `RuntimeError` with
the service's reason, so a refused write is never mistaken for a saved one.

## Agent frameworks

Each adapter implements the framework's own memory interface and accepts a shared
`client=`, so a whole app (or a whole crew) reads and writes one store.

| Framework | What you get | Extra |
|---|---|---|
| LangGraph | A long-term memory store for your graph | `[langgraph]` |
| CrewAI | One memory your whole crew shares | `[crewai]` |
| LangChain | Chat history you own | `[langchain]` |
| LlamaIndex | Chat memory, and a retriever for RAG | `[llamaindex]` and `[rag]` |
| AutoGen | A memory your agent owns | `[autogen]` |

**LangGraph.** Compile your graph with the store; every node that takes `store` reads and
writes SAIHM:

```python
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from saihm_adapters import SaihmStore


class State(TypedDict):
    user: str
    answer: str


def remember(state: State, *, store: BaseStore):
    store.put(("users", state["user"]), "drink", {"text": "likes tea"})
    return {}


def recall(state: State, *, store: BaseStore):
    item = store.get(("users", state["user"]), "drink")
    return {"answer": item.value["text"] if item else ""}


graph = StateGraph(State)
graph.add_node("remember", remember)
graph.add_node("recall", recall)
graph.add_edge(START, "remember")
graph.add_edge("remember", "recall")
graph.add_edge("recall", END)

app = graph.compile(store=SaihmStore())
print(app.invoke({"user": "ada", "answer": ""}))  # {'user': 'ada', 'answer': 'likes tea'}
```

**CrewAI (1.15+).** Register the backend once at start-up, then ask for SAIHM storage:

```python
from crewai import Memory
from crewai.memory.storage.factory import set_memory_storage_factory

from saihm_adapters import SaihmStorageBackend

set_memory_storage_factory(lambda spec: SaihmStorageBackend() if spec == "saihm" else None)
memory = Memory(storage="saihm")  # give it to a crew: Crew(agents=..., tasks=..., memory=memory)
```

**The other frameworks** take the shared client the same way:

```python
from saihm_adapters import SaihmMemoryClient
client = SaihmMemoryClient()

# LangChain — a BaseChatMessageHistory
from saihm_adapters import SaihmChatMessageHistory
history = SaihmChatMessageHistory(client=client)

# RAG — a LlamaIndex BaseRetriever over a corpus you own
from saihm_adapters import SaihmRetriever
retriever = SaihmRetriever(client=client, similarity_top_k=3)

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

## Your memories follow your key

Your memory belongs to your key, not to a computer — that's what makes it portable.
The key is created on your machine and never sent anywhere, which is exactly why
nobody else can read your memories. **Keep a copy of the key file somewhere safe:**
SAIHM cannot make you another one. Upgrading to a paid plan keeps the same key and
every memory.

**Running on a server, in CI, or on a second computer?** Two ways:

- **Same memory, anywhere** — on the machine that has your key, run:

  ```bash
  npx -y @saihm/mcp-server-pro export-identity
  ```

  It saves two values to a file only you can read: `SAIHM_IDENTITY` (your key,
  sealed) and `SAIHM_IDENTITY_PASSPHRASE` (the passphrase that opens it). Set both as
  environment variables or secrets where your app runs. Together the two values **are**
  your identity: keep them in a password manager or your platform's secret store, never
  in a repository, and delete the export file. On a paid plan, see
  [exporting your identity](https://github.com/SAIHM-Admin/saihm-mcp-server-pro/blob/main/REFERENCE.md#exporting-your-identity) first.
- **Copy the key file** — put it at `~/.saihm/free-identity.key` on the other machine
  (or point `SAIHM_MASTER_SECRET_FILE` at it), and use the same settings as the first
  machine (on a paid plan, `SAIHM_TIER` and `SAIHM_PAYMENT_METHOD` too).

## See it run

Runnable demos and end-to-end examples: **<https://saihm-admin.github.io/saihm-demos/>**

## What it costs

Start free. The free tier is a fixed, one-time allowance of writes, reads, and shares
for trying SAIHM on real infrastructure — it doesn't reset or refill. **No card, and
nothing to cancel.** When it is used up, calls raise an error that names the reason.

Paid plans are monthly; prices are on the [pricing page](https://saihm.net/pricing).
Upgrading keeps the same key and every memory you already have:

```sh
SAIHM_MASTER_SECRET_FILE=$HOME/.saihm/free-identity.key \
SAIHM_TIER=FREE \
  npx -y @saihm/mcp-server-pro upgrade PRO
```

That prints a checkout link tied to your identity. Pay, then set two variables where your
app runs, and start it again:

```bash
export SAIHM_TIER=PRO
export SAIHM_PAYMENT_METHOD=stripe
```

A paid plan needs both: without `SAIHM_PAYMENT_METHOD`, which names the payment rail to
check, the client raises `SaihmStartupError` at start-up. The upgrade command's link is card
checkout, and goes with `stripe`. It keeps your key and every memory on it. Stablecoin is
paid on the website's [join page](https://saihm.net/join) and goes with `stablecoin`.
Paying there starts a new identity with its own key. The memories on a key you already
have stay on that key and do not move to the new one, so to upgrade the key you have, use
the card link.

## If something isn't working

| What you see | What to do |
|---|---|
| `SaihmNoIdentityError` with `kind == "no-identity"` | This machine has no identity: run free-join, or set `SAIHM_MASTER_SECRET_FILE` to your key file |
| `SaihmNoIdentityError` with `kind == "identity-not-active"` | SAIHM refused the sign-in, and `reason` holds its code. With `no_free_entitlement`, the free-join sign-in was not approved: run free-join again (it reuses your key) and approve the new code. With another code, check `SAIHM_TIER` and `SAIHM_PAYMENT_METHOD` against your plan |
| `SaihmStartupError` with `kind == "node-missing"` | Install Node.js 20 or later, or pass `node="/full/path/to/node"` to the client |
| `SaihmStartupError` with `kind == "install-failed"` or `"not-provisioned"` | The sidecar could not be installed: allow access to the npm registry (and unset `SAIHM_SKIP_BOOTSTRAP`), or provision the sidecar yourself and set `SAIHM_SIDECAR_DIR` |
| A call raises `RuntimeError` with the service's reason | The service refused the call, for example because the free allowance is used up: see [What it costs](#what-it-costs) |
| A call raises `RuntimeError` saying the endpoint could not be reached | The machine cannot reach `saihm.net`: allow it in the network settings; if traffic must go through a proxy, set `HTTPS_PROXY` |
| A different memory than you expected | This machine has its own key rather than yours: see [Your memories follow your key](#your-memories-follow-your-key) |

## How it works

Python drives a bundled **Node.js sidecar** over MCP stdio, and the sidecar runs the same
SAIHM client as `@saihm/mcp-server-pro`. Everything is encrypted inside that sidecar, on
your machine, before it is sent, and decrypted there after it comes back. What SAIHM
stores is unreadable ciphertext, and no key that opens it. To erase something, its key is
destroyed, which is why erasure at SAIHM is immediate and final rather than a promise that a
copy was deleted somewhere.

- **Post-quantum by design.** ML-DSA-65 signs every memory.
- **Non-custodial.** The hosted service at <https://saihm.net> provides the durable
  storage, yet only ever holds ciphertext and never a key.
- **Portable across models & frameworks.** Every adapter reads and writes the same store,
  so the memory travels with your user.
- **Fast recall on your device.** With the default free-join key file, the client keeps a
  local copy of your memories, so a recall fetches only what is new. Each recall adds every
  memory it fetches to that copy, whatever its query, and `forget` recalls before and after
  it erases. That copy is not encrypted, only your user account can read it, and
  `forget` removes a memory from it too. A second device that uses the same key file keeps
  a forgotten memory in its own copy until its next recall. A backup of `~/.saihm` holds that copy as well, and
  erasure cannot reach a backup.
- **Everything encrypted at rest, if you prefer.** Set `SAIHM_RECALL_CACHE=0` and the
  client keeps no local copy, so every memory SAIHM holds at rest is ciphertext. The
  trade-off is speed: each recall then fetches all your memories again. With a key file
  in any other place, the cache is off, and the first recall prints a one-time note
  about it on stderr; `SAIHM_RECALL_CACHE=0` silences that note too.

## Configuration

Most apps need only `SAIHM_ENDPOINT_URL`. These are the main settings:

| Variable | What it does |
|---|---|
| `SAIHM_ENDPOINT_URL` | `https://saihm.net/mcp` for the hosted service; unset, a local sandbox |
| `SAIHM_MASTER_SECRET_FILE` | Your key file, when it is not at `~/.saihm/free-identity.key` |
| `SAIHM_IDENTITY`, `SAIHM_IDENTITY_PASSPHRASE` | An exported identity, for servers and CI |
| `SAIHM_TIER`, `SAIHM_PAYMENT_METHOD` | Your plan and its payment rail, on a paid plan |
| `SAIHM_RECALL_CACHE` | `0` keeps no local copy of your memories |
| `SAIHM_SIDECAR_DIR` | A sidecar you provisioned yourself, for air-gapped or CI installs |
| `SAIHM_SKIP_BOOTSTRAP` | `1` forbids the first-run network install |
| `SAIHM_BOOTSTRAP_TIMEOUT` | Seconds the first-run install may take (default 600) |

## Support SAIHM

- **Star it on [GitHub](https://github.com/SAIHM-Admin/saihm-adapters).** Stars help other
  developers find SAIHM.
- **Share it** with a colleague, your team, or anyone whose agents keep forgetting.
- **Follow [@SAIHMemory on X](https://x.com/SAIHMemory)** for news and updates.
- **Sponsor it** through [GitHub Sponsors](https://github.com/sponsors/SAIHM-Admin), monthly or one-time.
- **Follow along on YouTube:** like the videos,
  [subscribe to the channel](https://www.youtube.com/@SAIHM-Memory?sub_confirmation=1), and turn on notifications
  with the bell 🔔 for new videos and updates.
- **Questions or problems?** See [contact and support](https://saihm.net/contact), or open an
  issue on [GitHub](https://github.com/SAIHM-Admin/saihm-adapters/issues).

## License

Apache-2.0 © SAIHM
