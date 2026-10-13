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
  not exist), and the CrewAI line names the 1.15 storage-factory API it actually uses.
- **Links.** Project links now point to https://saihm.net and the GitHub repository; the demos
  link points to https://saihm-admin.github.io/saihm-demos/.

