#!/usr/bin/env python3
"""
eisy-claude-shim: a local Anthropic-Messages-API-compatible HTTP endpoint
for eisy-ai, backed by your Claude Pro/Max subscription via the official
Claude Code CLI.

Mechanism (not an OAuth-token hack): every request spawns the real,
unmodified `claude` binary (logged in via `claude auth login`) through
Nous Research's vendored `directsdk.Client` (MIT-licensed, vendored
under ./vendor/ from NousResearch/hermes-official-plugins). That module
drives Claude Code's native stream-json protocol directly and guarantees
exactly one real upstream request per call (admission-gated), so Claude
Code's own internal retries/extra turns never multiply subscription usage.
This shim only translates wire formats; it never touches credentials.

eisy-ai's `claude` provider config should point its `url` at this
server, e.g.:
    "url": "http://<this-machine>:8787"
and `api_key` at the SHIM_API_KEY value configured below (NOT a real
Anthropic key -- this shim never uses one).

Endpoints:
    POST /v1/messages        Anthropic Messages API, non-streaming and streaming
"""
import json
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vendor"))

from directsdk import Client, ClaudeAPIError, ClaudeCodeLoggedOut, ClaudeCodeMissing, RequestRejected  # noqa: E402

HOST = os.environ.get("EISY_SHIM_HOST", "0.0.0.0")
PORT = int(os.environ.get("EISY_SHIM_PORT", "8787"))
SHIM_API_KEY = os.environ.get("EISY_SHIM_API_KEY")  # if set, required as x-api-key
DEFAULT_CACHE_TTL = os.environ.get("EISY_SHIM_CACHE_TTL", "5m")  # "5m", "1h", or "" to disable
LOG_PATH = os.environ.get("EISY_SHIM_LOG", "/app/logs/requests.log" if os.path.isdir("/app/logs")
                           else os.path.join(os.path.dirname(__file__), "requests.log"))

client = Client()


def log(event: dict):
    event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = json.dumps(event, default=str)
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Anthropic Messages API  <->  OpenAI chat.completions  translation
# ---------------------------------------------------------------------------

def anthropic_messages_to_openai(system, messages):
    """Anthropic `messages` (+ top-level `system`) -> OpenAI-style `messages`.

    Splits `system` blocks by whether nucore marked them `cache_control`:
    cacheable (static) blocks stay in the system message; everything else
    (dynamic per-turn content, e.g. live device state) is moved into the
    latest message instead. This matters because directsdk's
    conversation_session_id() hashes [system, first_frame] to derive a
    stable --session-id for native Claude Code to resume across calls --
    if any dynamic byte lives inside `system`, the hash (and therefore the
    session, and therefore any cache reuse) changes on every single turn.
    """
    out = []
    dynamic_text = ""
    if system:
        if isinstance(system, list):  # Anthropic allows system as content blocks too
            static_parts = [b.get("text", "") for b in system if isinstance(b, dict) and b.get("cache_control")]
            dynamic_parts = [b.get("text", "") for b in system if isinstance(b, dict) and not b.get("cache_control")]
            system_str = "\n".join(static_parts)
            dynamic_text = "\n".join(dynamic_parts)
        else:
            system_str = system
        if system_str:
            out.append({"role": "system", "content": system_str})

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            out.append({"role": role, "content": ""})
            continue

        text_parts = []
        tool_calls = []
        tool_results = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })
            elif btype == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    result_content = "\n".join(
                        b.get("text", "") for b in result_content if isinstance(b, dict)
                    )
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id"),
                    "content": result_content if isinstance(result_content, str) else json.dumps(result_content),
                })
            # image blocks etc.: dropped (unsupported in this shim for now)

        # tool_result blocks arrive on a "user" turn in Anthropic's protocol but
        # must become separate role="tool" messages in OpenAI's.
        out.extend(tool_results)

        if role == "assistant":
            out.append({
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else None,
                **({"tool_calls": tool_calls} if tool_calls else {}),
            })
        elif text_parts:
            out.append({"role": role, "content": "\n".join(text_parts)})

    if dynamic_text:
        # Prepend the non-cacheable system content to the latest message
        # with plain string content, rather than back into `system` where
        # it would poison the session-id hash on every turn. Walk backward
        # for the nearest string-content message (an assistant tool_calls
        # entry has content=None and isn't eligible).
        for entry in reversed(out):
            if isinstance(entry.get("content"), str):
                entry["content"] = dynamic_text + "\n\n" + entry["content"]
                break
        else:
            out.append({"role": "user", "content": dynamic_text})

    return out


def _dump_request_shape(body):
    """Diagnostic only (EISY_SHIM_DEBUG_SHAPE=1): log the structural shape of
    a real incoming request -- never its text content -- so we can see
    whether nucore sends `system` as a single string or multiple blocks,
    and where (if anywhere) it places cache_control markers. This is what
    determines whether our flattening in anthropic_messages_to_openai() is
    destroying a static/dynamic cache split nucore intended to preserve.
    """
    system = body.get("system")
    shape = {"system_type": type(system).__name__}
    if isinstance(system, list):
        shape["system_block_count"] = len(system)
        shape["system_block_types"] = [b.get("type") if isinstance(b, dict) else type(b).__name__ for b in system]
        shape["system_cache_control"] = [b.get("cache_control") if isinstance(b, dict) else None for b in system]
        shape["system_block_lengths"] = [len(b.get("text", "")) if isinstance(b, dict) else None for b in system]
    messages = body.get("messages", [])
    shape["message_count"] = len(messages)
    msg_shapes = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            msg_shapes.append({
                "role": m.get("role"), "block_count": len(content),
                "block_types": [b.get("type") if isinstance(b, dict) else type(b).__name__ for b in content],
                "cache_control": [b.get("cache_control") if isinstance(b, dict) else None for b in content],
            })
        else:
            msg_shapes.append({"role": m.get("role"), "content_type": type(content).__name__})
    shape["messages"] = msg_shapes
    shape["top_level_keys"] = sorted(body.keys())
    log({"event": "debug_shape", **shape})


def detect_cache_ttl(body):
    """Find an Anthropic cache_control hint anywhere in the request and map
    it to directsdk's `hermes_cache_ttl` kwarg -- the thing that actually
    tells native Claude Code to set CLAUDE_CODE_PROMPT_CACHE_TTL and request
    caching from Anthropic at all. Without this, nucore's own cache_control
    markers (and its "cache_ttl": "5m" config) were being silently dropped
    on the floor, forcing every request to pay full, non-cached input-token
    rates (~10-20x more expensive, and ~10-20x faster subscription-window
    consumption) even on repeated turns of the same conversation.
    """
    def blocks():
        system = body.get("system")
        if isinstance(system, list):
            yield from system
        for msg in body.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                yield from content

    for block in blocks():
        if isinstance(block, dict) and isinstance(block.get("cache_control"), dict):
            ttl = block["cache_control"].get("ttl", "5m")
            if ttl == "5m":
                return "5m"
    return None


def anthropic_tools_to_openai(tools):
    out = []
    for t in tools or []:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return out


STOP_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
    None: "end_turn",
}


def openai_response_to_anthropic(resp_dict, requested_model):
    choice = resp_dict["choices"][0]
    message = choice["message"]
    content_blocks = []
    if message.get("content"):
        content_blocks.append({"type": "text", "text": message["content"]})
    for tc in message.get("tool_calls") or []:
        try:
            tool_input = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, TypeError):
            tool_input = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex}",
            "name": tc["function"]["name"],
            "input": tool_input,
        })
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    usage = resp_dict.get("usage", {})
    admission = usage.get("native_admission", {})
    native_usage = usage.get("native_usage", {})
    # The actual model Anthropic's backend answered with, independent of
    # whatever alias/string was requested -- keys of native_cost.modelUsage
    # are the ground truth, confirmed live against a real response.
    model_usage = usage.get("native_cost", {}).get("modelUsage", {})
    canonical_model = next(iter(model_usage), None)

    return {
        "id": resp_dict.get("id") or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content_blocks,
        "stop_reason": STOP_REASON_MAP.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": native_usage.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": native_usage.get("cache_read_input_tokens", 0),
        },
    }, admission, canonical_model


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "eisy-claude-shim/0.1"

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_wrapped(self, anthropic_resp):
        """Fake a minimal valid Anthropic SSE stream carrying one full response.

        eisy-ai's config sets stream: true. Rather than reimplement native's
        token-by-token streaming wire format, this emits the complete message
        as a single content_block_delta, which is valid SSE event-wise even
        though it doesn't incrementally stream tokens. Good enough for a first
        working version; real incremental streaming is a follow-up.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def emit(event, data):
            chunk = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
            self.wfile.write(chunk)

        msg = dict(anthropic_resp)
        content = msg.pop("content")
        emit("message_start", {"type": "message_start", "message": {**msg, "content": []}})
        for i, block in enumerate(content):
            emit("content_block_start", {"type": "content_block_start", "index": i, "content_block": block})
            if block["type"] == "text":
                emit("content_block_delta", {"type": "content_block_delta", "index": i,
                                              "delta": {"type": "text_delta", "text": block["text"]}})
            elif block["type"] == "tool_use":
                emit("content_block_delta", {"type": "content_block_delta", "index": i,
                                              "delta": {"type": "input_json_delta",
                                                        "partial_json": json.dumps(block["input"])}})
            emit("content_block_stop", {"type": "content_block_stop", "index": i})
        emit("message_delta", {"type": "message_delta",
                                "delta": {"stop_reason": anthropic_resp["stop_reason"], "stop_sequence": None},
                                "usage": anthropic_resp["usage"]})
        emit("message_stop", {"type": "message_stop"})

    def do_POST(self):
        if urlsplit(self.path).path.rstrip("/") != "/v1/messages":
            self._send_json(404, {"type": "error", "error": {"type": "not_found_error", "message": "unknown path"}})
            return

        if SHIM_API_KEY and self.headers.get("x-api-key") != SHIM_API_KEY:
            self._send_json(401, {"type": "error", "error": {"type": "authentication_error", "message": "bad key"}})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "bad json"}})
            return

        if os.environ.get("EISY_SHIM_DEBUG_SHAPE"):
            _dump_request_shape(body)

        model = body.get("model", "sonnet")
        openai_messages = anthropic_messages_to_openai(body.get("system"), body.get("messages", []))
        openai_tools = anthropic_tools_to_openai(body.get("tools"))
        wants_stream = bool(body.get("stream"))

        kwargs = {
            "model": model,
            "messages": openai_messages,
            "max_tokens": body.get("max_tokens", 4096),
            "stream": False,  # always fetch complete; see _send_sse_wrapped for the stream:true path
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        cache_ttl = detect_cache_ttl(body) or (DEFAULT_CACHE_TTL or None)
        if cache_ttl == "5m":
            kwargs["hermes_cache_ttl"] = "5m"

        t0 = time.monotonic()
        try:
            resp = client.create(**kwargs)
            resp_dict = resp.model_dump()
        except ClaudeCodeMissing as e:
            log({"event": "error", "kind": "claude_missing", "detail": str(e)})
            self._send_json(503, {"type": "error", "error": {"type": "api_error", "message": str(e)}})
            return
        except ClaudeCodeLoggedOut as e:
            log({"event": "error", "kind": "logged_out", "detail": str(e)})
            self._send_json(401, {"type": "error", "error": {"type": "authentication_error", "message": str(e)}})
            return
        except ClaudeAPIError as e:
            log({"event": "error", "kind": "api_error", "detail": str(e),
                 "status_code": getattr(e, "status_code", None)})
            self._send_json(getattr(e, "status_code", None) or 502,
                             {"type": "error", "error": {"type": "api_error", "message": str(e)}})
            return
        except RequestRejected as e:
            log({"event": "error", "kind": "request_rejected", "detail": str(e)})
            self._send_json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}})
            return
        except Exception as e:
            # Never let an unanticipated exception just drop the connection
            # (curl/eisy-ai then sees a bare connection failure, not a
            # diagnosable error) -- always return *some* clean JSON error.
            log({"event": "error", "kind": "unexpected", "detail": repr(e)})
            self._send_json(500, {"type": "error", "error": {"type": "api_error", "message": f"shim internal error: {e}"}})
            return

        anthropic_resp, admission, canonical_model = openai_response_to_anthropic(resp_dict, model)
        dt_ms = int((time.monotonic() - t0) * 1000)
        log({
            "event": "request", "model_requested": model, "model_answered": canonical_model, "duration_ms": dt_ms,
            "input_tokens": anthropic_resp["usage"]["input_tokens"],
            "output_tokens": anthropic_resp["usage"]["output_tokens"],
            "cache_creation_input_tokens": anthropic_resp["usage"]["cache_creation_input_tokens"],
            "cache_read_input_tokens": anthropic_resp["usage"]["cache_read_input_tokens"],
            "stop_reason": anthropic_resp["stop_reason"],
            "upstream_requests": admission.get("upstream_requests"),
            "blocked_requests": admission.get("blocked_requests"),
            "using_overage": (admission.get("rate_limit") or {}).get("using_overage"),
            "utilization": (admission.get("rate_limit") or {}).get("utilization"),
        })

        if wants_stream:
            self._send_sse_wrapped(anthropic_resp)
        else:
            self._send_json(200, anthropic_resp)

    def log_message(self, fmt, *args):
        pass  # quiet; we log structured JSON ourselves above


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"eisy-claude-shim listening on http://{HOST}:{PORT}/v1/messages", flush=True)
    print(f"logging requests to {LOG_PATH}", flush=True)
    if not SHIM_API_KEY:
        print("WARNING: EISY_SHIM_API_KEY not set -- anyone on the network who can reach "
              "this port can spend your Claude subscription quota. Set it before exposing "
              "this beyond localhost.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
