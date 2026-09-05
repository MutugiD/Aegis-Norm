# HTTP API and generation runtime

Status: specified for the first demo. This is a documented chat-completions subset, not a claim of full compatibility with another provider. No endpoint is running yet.

## Routes and configuration

`POST /v1/chat/completions`, `GET /healthz`, `GET /readyz`. Run one application process and one model-owning worker thread; multiple Uvicorn workers would duplicate GPU weights and are unsupported. Bind to `127.0.0.1:8000` by default. A permitted public endpoint requires an operator-supplied bearer token; do not print or export it. Prompt text is not logged by default.

| Setting | Default / behavior |
|---|---|
| Model alias | `tinyllama`; maps only to the configured pinned model, never downloads a request-supplied ID |
| Backend | `cuda`; explicit `reference` mode is used for baseline serving |
| Active generation | 1 |
| Waiting queue | 4; FIFO admission under a lock |
| Maximum body | 64 KiB; enforce while reading, including chunked bodies |
| Maximum messages | 32 |
| Model context cap | 2048 tokens including template and maximum generation; may be lowered by operator |
| Queue deadline | 30 seconds from admission |
| Generation deadline | 90 seconds from activation |
| Overall deadline | 120 seconds from receipt; the earliest applicable deadline wins |
| Stop grace | 5 seconds; a worker that does not return makes readiness false until restarted |
| Text handoff | 64 chunks per active request, at most 5 seconds producer wait when full |
| SSE heartbeat | Comment every 15 seconds without data; excluded from output and latency metrics |

`/healthz` returns 200 when the event loop responds. `/readyz` returns 200 only after model/native smoke/warmup and worker startup, otherwise 503. Readiness need not turn false simply because the queue is full. On process shutdown, stop admissions, cancel queued requests, signal the active request, and wait only the stop grace before terminating the process.

## Request contract

```json
{
  "model": "tinyllama",
  "messages": [{"role": "user", "content": "Explain CUDA memory coalescing."}],
  "stream": true,
  "max_tokens": 128,
  "temperature": 0,
  "seed": 0
}
```

`model` and `messages` are required. `stream` defaults to false. `max_tokens` is an integer 1..512, default 128; booleans are not integers for validation. `temperature` is a finite number 0..2, default 0; zero uses greedy decoding. `seed` is an integer 0..2^32-1, default 0. Use a saved/restored RNG scope inside the single worker for sampling. Determinism is assessed on the same environment, not promised across CUDA/library versions.

Messages contain exactly `role` and string `content`. Permit an optional initial nonempty system message followed by nonempty user/assistant messages in alternating order, beginning and ending with user. Reject empty/whitespace-only content, misplaced system messages, tools, images, unknown fields and unsupported roles. Reject unknown top-level fields rather than silently ignoring them. No tools, logprobs, multiple choices, arbitrary stop sequences, model hot-swap or request-level backend selection in v0.1.

Apply the pinned chat template once with `add_generation_prompt=True` and tokenize without duplicating special tokens. Require `prompt_tokens + max_tokens <= context_cap`; return a validation error otherwise. No silent truncation. Bound concurrent CPU preprocessing with a semaphore of 5 so requests cannot bypass the bounded queue by accumulating in tokenization. Requests beyond this pre-admission capacity receive 429 and release body buffers. Final queue admission is atomic and can also reject with 429.

## Responses

Nonstreaming success uses HTTP 200 and this shape (illustrative, not a recorded generation):

```json
{
  "id": "chatcmpl-example",
  "object": "chat.completion",
  "created": 1788566400,
  "model": "tinyllama",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "Adjacent threads access adjacent memory addresses."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 24, "completion_tokens": 9, "total_tokens": 33}
}
```

Generate a unique request ID and actual Unix timestamp. Example usage numbers are placeholders; implementation counts token IDs, not whitespace words or SSE chunks. Completion tokens include generated terminal special tokens; visible content omits special tokens. Finish reason is `stop` for EOS or `length` for the cap. Prompt text is not echoed.

For streaming, content type is `text/event-stream`, with `Cache-Control: no-cache`. Start response headers only after admission and activation, so queue rejection/timeout still has an HTTP error status. Send an initial role delta, zero or more JSON-escaped content deltas, an empty delta carrying the finish reason, then `[DONE]`, each separated by a blank line:

```text
data: {"id":"chatcmpl-example","object":"chat.completion.chunk","created":1788566400,"model":"tinyllama","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-example","object":"chat.completion.chunk","created":1788566400,"model":"tinyllama","choices":[{"index":0,"delta":{"content":"CUDA memory"},"finish_reason":null}]}

data: {"id":"chatcmpl-example","object":"chat.completion.chunk","created":1788566400,"model":"tinyllama","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```

Streaming usage payloads are excluded from this subset; server evaluation artifacts still retain token counts. UTF-8, newlines, quotes and backslashes must survive JSON serialization. Chunks are decoded text deltas, not guaranteed individual tokens. Do not use per-chunk sleeps to pace output.

## Errors and cancellation

Errors use `{"error":{"code":"...","message":"...","request_id":"..."}}`. Never include secrets, prompts or native stack traces in client messages.

| Condition | Before stream headers | After headers |
|---|---|---|
| Malformed JSON | 400 `invalid_json` | Not applicable |
| Missing/invalid bearer token when configured | 401 `unauthorized` | Not applicable |
| Body exceeds cap | 413 `body_too_large` | Not applicable |
| Invalid schema, model alias or context | 422 `invalid_request` | Not applicable |
| Queue/preprocessing capacity full | 429 `queue_full`, `Retry-After: 1` | Not applicable |
| Not ready/shutting down | 503 `not_ready` | Error event if active execution is terminated |
| Queue or execution timeout | 504 `request_timeout` | `event: error` with JSON error, then `[DONE]` |
| Worker exception | 500 `generation_failed` | Error event then `[DONE]`; never a success finish reason |
| Client disconnect | Cancel; no response possible | Cancel; do not attempt a terminal send |

Remove cancelled queued requests rather than retaining occupied capacity. Active cancellation sets a thread-safe flag checked by a generation stopping criterion between steps. A worker exception always completes the request future and wakes the async consumer. A custom bounded streamer/handoff avoids blocking the event loop on a synchronous iterator. A producer stalled by a slow consumer is cancelled after 5 seconds; if connected, best-effort error delivery follows, otherwise close.

Do not release the active generation slot merely because the HTTP consumer ended. Release only after worker completion and cleanup. If the worker fails to stop within grace, mark unavailable and reject new work; a Python thread cannot safely preempt an in-flight CUDA kernel. OOM fails the request and readiness; restart/warmup before serving again instead of automatically retrying and hiding instability.

## Observability

Record request ID, backend, model revision, status, prompt/generated token counts, queue time, activation time, first generated token time, first text delta time, finish time, cancellation cause and error code. Kernel profiling is a separate diagnostic mode. No prompt/output text is required in logs. Fixed public evaluation prompts are stored separately with hashes.

Client TTFT starts at request dispatch and ends at first nonempty content delta; it excludes role events and heartbeats. Server timing includes queueing in end-to-end figures but reports it separately. Tunnel and localhost runs are different result categories.
