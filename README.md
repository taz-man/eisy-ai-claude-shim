# eisy-ai-claude-shim

A local Anthropic Messages API-compatible HTTP endpoint that lets
[eisy-ai](https://github.com/UniversalDevicesInc/eisy-ai) use your Claude
Pro/Max **subscription** for inference instead of a metered Anthropic API key.

## How it works (not an OAuth-token hack)

Every request spawns the real, unmodified `claude` binary — the official
Claude Code CLI, authenticated via your own `claude auth login` — through
Nous Research's `directsdk.Client`. That module is vendored here unmodified
(MIT-licensed, see `vendor/LICENSE`) from
[`NousResearch/hermes-official-plugins/claude-subscription-directsdk`](https://github.com/NousResearch/hermes-official-plugins/tree/main/claude-subscription-directsdk)
— the same engine behind Hermes's own "Claude Subscription DirectSDK" plugin.

That module drives Claude Code's native stream-json protocol directly and
**guarantees exactly one real upstream request per call** (admission-gated —
see `vendor/admission.py`), so Claude Code's own internal retries/extra turns
never silently multiply your subscription usage.

`server.py` only translates wire formats between what eisy-ai sends
(Anthropic Messages API shape, including the `beta.messages.stream()` variant
eisy-ai actually calls) and what `directsdk.Client` expects (an
OpenAI-chat-completions-shaped interface). It never touches credentials,
never extracts an OAuth token, and never replays one against the API as a
different client — which is exactly why this is allowed, while the
"extract-the-token-and-replay-it" approach other tools used is both against
Anthropic's Consumer Terms of Service and server-enforced (blocked since
January 2026).

## What's proven working

- Plain text completions
- Tool-calling (`tools` in the request → real `tool_use` blocks in the
  response) — this is what eisy-ai's device-control customer mode needs
- Streaming (`stream: true`, including the `?beta=true` query parameter
  eisy-ai's SDK appends) — currently implemented as one non-incremental SSE
  flush rather than true token-by-token streaming; good enough not to break
  eisy-ai's client, but responses arrive all at once
- **Prompt-cache reuse across turns of the same conversation**, confirmed
  against real eisy-ai traffic: `cache_read_input_tokens` went from 0% to
  38-53%+ within a session after fixing how `system` content is split (see
  below). Anthropic's cache-read price is roughly 10-20x cheaper than a
  fresh input token, so this matters a lot for both subscription-window
  consumption and raw cost if you ever do run this against a metered key.
- Every response's usage carries Claude Code's own `native_admission` data
  (`upstream_requests`, `blocked_requests`, `rate_limit.using_overage`,
  `rate_limit.utilization`) — the shim logs these per request to
  `requests.log` so you can confirm you're staying inside your subscription
  window and never hitting overage billing

## Known limitations

- No image/vision content blocks (dropped silently if eisy-ai sends one)
- ~1-2s latency per request from the fresh `claude` CLI process spawn —
  fine for occasional chat, slower than a direct API call under load
- CLI version sensitivity: `vendor/directsdk.py`'s native wire-format
  assumptions are version-pinned by Nous's own CI; a future `claude update`
  could break something this vendored snapshot doesn't handle. Re-test after
  updating, and re-pull `vendor/` from upstream if things break
- This wraps Nous's own **"Experimental"**-labeled engine — treat it the
  same way

## Setup

### 1. Prerequisites on the host that will run this

- Docker + Docker Compose
- The official Claude Code CLI installed and logged into your Pro/Max
  subscription will happen *inside* the container (step 3) — you don't need
  it on the host itself, just Docker.
- **A Linux host, or Docker Desktop on Windows/macOS with its WSL2/Linux VM
  working correctly.** If you're on Windows with WSL2 in *mirrored*
  networking mode (`networkingMode=mirrored` in `.wslconfig`), Docker's
  published ports may become unreachable from other devices on your LAN even
  though `localhost` works fine. If so, switch to `networkingMode=nat`
  (or remove the line — `nat` is the default), run `wsl --shutdown`, then
  fully quit and relaunch Docker Desktop (not just the WSL VM) before
  testing LAN reachability again.

### 2. Build and run

```bash
cp .env.example .env
# Edit .env and set EISY_SHIM_API_KEY to a real random value, e.g.:
#   python3 -c "import secrets; print(secrets.token_urlsafe(24))"

docker compose build
```

### 3. One-time subscription login

Run this **yourself, interactively, in your own terminal** — not through an
agent or automation — so the OAuth code never leaves your machine:

```bash
docker run --rm -it --name eisy-claude-login \
  -v eisy-ai-claude-shim_claude-auth:/root/.claude \
  eisy-ai-claude-shim:latest claude auth login
```

Approve in the browser; if prompted, paste the code into that same terminal.
Verify it landed:

```bash
docker run --rm -v eisy-ai-claude-shim_claude-auth:/root/.claude \
  eisy-ai-claude-shim:latest claude auth status
```

You should see `"loggedIn": true` and your subscription's `planTier`.

### 4. Start it

```bash
docker compose up -d
```

Logs (including per-request admission/overage data) go to `docker logs -f
eisy-ai-claude-shim` and to `requests.log` inside the container.

### 5. Point eisy-ai at it

In eisy-ai's plugin configuration, for `unified`, `plugin_authoring`, or
both:

```json
{
  "provider": "claude",
  "model": "claude-haiku-5-5",
  "api_key": "<the value you put in .env>",
  "url": "http://<this-host's-LAN-IP-or-hostname>:8799",
  "stream": true
}
```

Model ids/aliases pass straight through to Claude Code
(`claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`, or the short
aliases `haiku`/`sonnet`/`opus`). Make sure eisy can actually reach that
host and port on your LAN — test with `curl` directly from eisy's own shell
before troubleshooting inside eisy-ai itself.

### 6. Check your Claude account's "extra usage" setting

This shim surfaces `using_overage` per request in its logs but does not
enforce anything. If overage billing is enabled on your account and you
exceed your subscription's included window, Anthropic will bill you
per-token exactly like the metered API — which defeats the point of this
setup. Disable it in your Claude account settings if you want a hard
guarantee against that.

## Troubleshooting

- **`eisy-ai` shows a generic "something went wrong" error with no detail**:
  check eisy-ai's own plugin logs for the actual traceback — the error shown
  in chat is often truncated.
- **404 `unknown path`**: eisy-ai's Anthropic client calls the `beta`
  streaming endpoint, which appends `?beta=true` to the URL. Make sure
  you're running a version of `server.py` that strips the query string
  before matching the path (fixed in this repo).
- **Connection timeout / connection refused from eisy specifically, but
  `curl localhost:<port>` on the Docker host works fine**: test from a
  *different* device on the LAN (not the Docker host itself — a hairpin
  NAT test back to the same box can succeed even when real external
  devices can't connect). If that also fails, suspect the WSL2
  mirrored-networking issue above.
- **A Docker-published port gets stuck** even after killing whatever was
  using it: `docker port`/`NetworkSettings.Ports` comes back empty even
  though `HostConfig.PortBindings` looks correct. This is a host-side
  docker-proxy binding failure, not a config mistake — pick a different
  host port and move on rather than re-debugging the same one.
- **Prompt caching never hits (`cache_read_input_tokens` always 0), even
  across turns of one conversation**: this was a real bug, fixed in this
  repo. eisy-ai sends `system` as multiple content blocks — some marked
  `cache_control` (static: tool manifests, device lists), some not (dynamic:
  live device state, timestamps). `directsdk.py`'s `conversation_session_id()`
  derives a stable `--session-id` by hashing `[system, first_frame]` — if any
  byte of the dynamic block ends up inside that hashed `system` value, the
  derived session id changes on *every single turn*, native Claude Code
  never resumes the same session, and nothing ever gets cached. The fix:
  split `system` blocks by whether they carry `cache_control`, keep only the
  static ones in the hashed system string, and move dynamic content into the
  actual per-turn message instead, where changing bytes don't matter.

## Attribution

`vendor/*.py` is an unmodified copy of Nous Research's
`claude-subscription-directsdk` plugin (MIT, see `vendor/LICENSE`), pulled
from the `main` branch of `NousResearch/hermes-official-plugins`.
`server.py`, `Dockerfile`, and `docker-compose.yml` are original to this
repo (MIT, see `LICENSE`).

This is not an official Anthropic, Nous Research, or Universal Devices
project. Use at your own risk and in accordance with Anthropic's Consumer
Terms of Service.
