# Changelog

## 0.1.3

- **A call the service refuses now raises, on every `mcp` release this package supports.**
  `mcp` 2.x renamed the error flag on tool results (`isError` became `is_error`), and 0.1.2
  read only the old name. With `mcp` 2.x installed, which is what a fresh install picks, a
  refused call looked like a success: `remember()` returned an id for a write that was never
  stored, `recall()` returned an empty list, and `forget()` returned `False`. The client now
  reads both names, so every refused call raises `RuntimeError` with the service's reason, and
  `forget()` raises instead of returning `False` when an erase is refused. A tool result with
  neither flag raises too, and `recall()` raises on a result that is not a list of memories
  instead of returning an empty one. If you are on 0.1.2, upgrade, or run
  `pip install "mcp<2"` until you can.
- **An unfinished free-join is reported at start-up.** `free-join` saves your key before you
  approve the sign-in. Until you do, the service does not accept the key, but 0.1.2 started
  as if it were ready. In live mode the client now checks with the service at start-up, and a
  key that is not active yet raises `SaihmNoIdentityError` (`kind == "identity-not-active"`)
  with the steps to finish: run free-join again, which reuses your key. Other sign-in
  refusals, such as a paid identity without an active subscription or a plan setting the
  service rejects, raise the same error with the service's reason code in `reason`, and
  `str(error)` names the code when the free-join steps may not apply. A refusal of the one-time
  sign-in challenge, which a restart of the service can cause, gets one more sign-in and never
  stops start-up. If the service cannot be reached, the client starts as before and each call
  reports the problem; if it does not answer, start-up goes ahead after 20 seconds.
- **Errors written for Python developers.** `str(SaihmNoIdentityError)` now carries only the
  steps for a Python app. The sidecar's own message, written for MCP hosts such as editors,
  stays on the error's `message` attribute for logs.
- **Every start-up failure raises `SaihmStartupError`.** A missing Node.js (before or after the
  sidecar is installed), a sidecar that is not installed while `SAIHM_SKIP_BOOTSTRAP` is set, a
  failed or timed-out sidecar install, a `SAIHM_SIDECAR_DIR` without a sidecar, and a sidecar
  that exits without a message all raise it now. Its `kind` attribute names the failure.
- **Sidecar dependencies updated.** The bundled lockfile moves to `@modelcontextprotocol/sdk`
  1.32.1, `proxy-addr` 2.0.8, `hono` 4.13.13, `ip-address` 10.7.4 and `fast-uri` 3.1.8. When
  this release was built, `npm audit --omit=dev` reported no known vulnerabilities in it. The
  sidecar talks MCP over stdio and does not load its dependencies' HTTP-server code. The first
  run after upgrading installs the new sidecar by itself.
- **On Python 3.10, a call that times out raises `SaihmTimeout`** as documented. It raised a
  generic `RuntimeError` there before, because 3.10 has a separate timeout class for futures.
- **README.** It follows the `@saihm/mcp-server-pro` README: start free, add SAIHM to your
  app, agent frameworks, your key, what it costs (with the
  [pricing page](https://saihm.net/pricing)), troubleshooting, how it works, configuration
  and support. New: a runnable LangGraph graph (`graph.compile(store=SaihmStore())`), the
  CrewAI registration (`set_memory_storage_factory`), and a table of the main settings. The
  local-copy note now says that the copy holds the memories this device has written and every
  memory a recall fetches, whatever its query, and that a second device keeps a forgotten
  memory until its next recall.
- **Tests** run the full client against `mcp` 1.x and 2.x, including the refusal paths and the
  start-up time limit, with a local stand-in for the hosted endpoint. A timed-out call is
  tested to raise `SaihmTimeout`.

### Upgrading from 0.1.2

- Code that relied on `forget()` returning `False`, or `recall()` returning `[]`, when the
  service refused the call now gets a `RuntimeError` instead. An id that does not exist still
  returns `False` from `forget()`.
- In live mode, `SaihmMemoryClient()` now signs in to the service at start-up. A key that is
  not active yet raises `SaihmNoIdentityError` there, where it used to fail on the first call.
  If the service does not answer, start-up waits up to 20 seconds before going ahead.
- Start-up failures that used to raise a plain `RuntimeError` or an OS error now raise
  `SaihmStartupError`, a `RuntimeError` subclass, so existing `except RuntimeError` clauses
  still catch them.
- The previous sidecar stays in `~/.cache/saihm-adapters/0.1.2-*` (under `$XDG_CACHE_HOME` if
  set). Nothing loads it; delete it so security scans of the cache see only the current one.
- In live mode with the default free-join key file, 0.1.2 kept a local copy of your memories
  beside your key file, or in the file that `SAIHM_RECALL_CACHE_PATH` names, as 0.1.3 does, so
  recall is fast. That copy is not encrypted, and only your user account can read it. Setting
  `SAIHM_RECALL_CACHE=0` in every app that uses this key, MCP hosts included, stops new copies
  and leaves the existing one in place. To remove it, delete `~/.saihm/recall.*`, or
  `recall.*` in `SAIHM_HOME` if you set it; with `SAIHM_RECALL_CACHE_PATH`, delete that file
  and its `.lock` and `.tmp.*` files too.
- If pip reports conflicting dependencies with an `mcp` older than 1.21.1, upgrade `mcp` to
  1.21.1 or later. Until you can, `saihm-adapters==0.1.1` still installs. It lacks the fixes
  above, so move to 0.1.3 when you upgrade `mcp`.

## 0.1.2

- **A clear error when there is no identity yet.** In live mode (`SAIHM_ENDPOINT_URL` set) with
  no SAIHM identity on the machine, `SaihmMemoryClient()` now raises `SaihmNoIdentityError`
  carrying the server's own message, plus the steps to create an identity from Python. Before,
  the message went only to stderr and the caller saw a bare "Connection closed". Other start-up
  failures the sidecar can explain raise `SaihmStartupError`, the parent class. The sidecar's
  other stderr output still reaches your terminal as before.
- **Current SAIHM client.** The bundled Node sidecar now uses `@saihm/mcp-server-pro` 0.14
  (was 0.5). The sidecar provisions itself again on first use after the upgrade.
- **Go live, documented.** The README has a new "Go live" section: create an identity once with
  `npx -y @saihm/mcp-server-pro free-join`, set `SAIHM_ENDPOINT_URL`, and keep the key in its
  file (`SAIHM_MASTER_SECRET_FILE`), so Python never reads it. The quickstart now says that the
  default, with no endpoint, is a local throwaway sandbox.
- **Fixes in the README.** The quickstart reads `cell.text` (it said `cell.content`, which does
  not exist), and the CrewAI line names the 1.15 storage-factory API the adapter uses.
- **Start-up errors from the sidecar's dependencies** (a Node package that fails to load) now
  raise `SaihmStartupError` too, and a sidecar that exits early without a message raises it
  with the sidecar's last output.
- **Requires `mcp` 1.21.1 or later.** Older `mcp` releases fail to import with current
  pydantic.
- **Local recall cache, documented.** With the default free-join key file, the 0.14 client keeps
  the memories this device has opened in a local file beside the key; the README says where,
  how `forget` treats it, and how to turn it off.
- **Links.** Project links now point to https://saihm.net and the GitHub repository; the demos
  link points to https://saihm-admin.github.io/saihm-demos/.

### Upgrading from 0.1.1

- Start-up failures used to surface as an `ExceptionGroup` around an `mcp`
  "Connection closed" error. They now raise `SaihmStartupError`, a `RuntimeError` subclass, or its subclass
  `SaihmNoIdentityError`. Update any `except` clause that caught the old types.
- The sidecar cache is keyed by the package version and the bundled Node lockfile, so the first
  run after upgrading installs the new sidecar (`npm ci`) by itself. With
  `SAIHM_SKIP_BOOTSTRAP=1`, the first run after upgrading stops with an error that names the
  new directory: provision it there (copy the bundled sidecar files and run
  `npm ci --omit=dev`), or point `SAIHM_SIDECAR_DIR` at a directory you provision.
- A directory you provisioned yourself and named with `SAIHM_SIDECAR_DIR` keeps the old
  sidecar until you refresh it; the client warns when it differs from the installed release.
  To refresh it, copy `server.mjs`, `sandbox.mjs`, `package.json` and `package-lock.json`
  from the installed `saihm_adapters/_sidecar/` folder into it and run `npm ci --omit=dev`
  there.
