# Changelog

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
