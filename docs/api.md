# Laya Go Launcher HTTP API

Base URL: `http://127.0.0.1:8420`

All endpoints are under `/api/v1`. Requests and responses are JSON
(`Content-Type: application/json`). Errors use a non-2xx status and a JSON body:

```json
{ "error": { "code": "engine_not_loaded", "message": "no engine is loaded" } }
```

The same surface is exposed two ways: over HTTP (for scripts and other
processes) and as Wails bindings (for the bundled GUI). Both call the same
`internal/inference` use cases, so behaviour is identical.

---

## GET /api/v1/health

Liveness and runtime capability. Always available, never fails.

```json
{
  "status": "ok",
  "version": "0.1.0",
  "kernel": { "available": true, "tensorrt": "101600" },
  "backend": {
    "requested": "auto",
    "selected": "tensorrt",
    "note": "auto prefers TensorRT; an .onnx model is served by ONNX Runtime",
    "onnx": { "available": true, "version": "1.25.0", "error": "" }
  },
  "device": {
    "name": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
    "compute_capability": "12.0",
    "vram_free_mb": 11042,
    "vram_total_mb": 12199
  },
  "engine": { "loaded": true, "path": "C:\\...\\laya_e2e.engine", "backend": "TensorRT", "contexts": 2 }
}
```

When no engine is loaded, `engine.loaded` is `false` and `engine.path` is `""`.
When the kernel failed to initialise, `kernel.available` is `false` and
`kernel.error` carries the reason.

`kernel` describes TensorRT specifically, and is kept for compatibility with
clients written before a second kernel existed. `backend` is the general answer
and is the object to read:

| field | meaning |
|---|---|
| `requested` | the configured kernel, after parsing: `auto`, `tensorrt` or `onnx` |
| `selected` | the kernel a load would use right now, given what this machine has |
| `note` | why `selected` is not what was requested, or what `auto` will do |
| `onnx.available` | whether the bridge and an ONNX Runtime library were both found |
| `onnx.version` | the ONNX Runtime version, read from the library itself |
| `onnx.error` | why ONNX Runtime is unavailable |

`onnx.version` comes from `OrtGetApiBase()->GetVersionString()`, read by the bridge
without creating a session, so a stale or mismatched `onnxruntime.dll` is visible in
diagnostics rather than only showing up as a failed inference. `tensorrt.version` is
the TensorRT version the process is linked against.

`requested` is what was configured and `selected` is always a concrete kernel
(`tensorrt` or `onnx`). The two differ whenever `requested` is `auto`, and that is
normal: `auto` is a policy rather than a kernel, and which kernel it lands on
depends on the model file — a plan can only be read by TensorRT, a graph only by
ONNX Runtime. Because this endpoint is answered without a model, it reports
`auto`'s *preference* and says so in `note`:

```
"note": "auto prefers TensorRT; an .onnx model is served by ONNX Runtime"
```

The case to act on is `auto` on a machine where TensorRT cannot come up:
`requested: "auto"`, `selected: "onnx"`, and a `note` explaining the substitution.
A kernel named explicitly is reported as itself in both fields, because it is never
substituted — a load will fail instead.

`engine.backend` is the display name of the kernel actually serving the model
(`TensorRT` or `ONNX Runtime`), so a client can tell which one a request will run
on without re-deriving it from configuration. It is absent when nothing is
loaded.

---

## GET /api/v1/backends

Which kernels this build and this machine can offer, answered without loading a
model. It is the endpoint to call before committing to a load, and the same
diagnostics `/health` embeds.

```json
{
  "requested": "auto",
  "tensorrt": { "name": "TensorRT", "available": true, "version": "101600" },
  "onnx": { "name": "ONNX Runtime", "available": true, "version": "1.25.0" },
  "selected": "tensorrt",
  "note": "auto prefers TensorRT; an .onnx model is served by ONNX Runtime"
}
```

Each kernel reports `name`, `available`, and optionally `version` and `error`.
`version` is read from each runtime itself — the TensorRT version the process is
linked against, and the ONNX Runtime version the bridge reports. `error` explains an
unavailable kernel and is absent when it is available. `selected` is the kernel a
load would use right now; `note` explains what that means when `requested` is `auto`
or when a fallback is in effect, and is absent when there is nothing to add.

Availability here is about the *runtime*, not about a model. TensorRT reports
available when CUDA and the driver come up; ONNX Runtime reports available when
both the bridge library and `onnxruntime.dll` resolve. A kernel can be available
and still fail to load a particular file, which is why the load endpoint reports
its own errors.

This endpoint always returns `200`: an unavailable kernel is a fact to report,
not a request failure.

---

## GET /api/v1/engine

Describes the loaded engine, including every IO tensor.

```json
{
  "loaded": true,
  "path": "C:\\Users\\lingxin\\Documents\\laya-ort-probe\\laya_e2e.engine",
  "backend": "TensorRT",
  "runtime": "101600",
  "device": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
  "contexts": 2,
  "activation_memory_mb": 1.6,
  "inputs": [
    { "name": "input_ids",      "dtype": "int64",   "shape": [1, 64] },
    { "name": "attention_mask", "dtype": "int64",   "shape": [1, 64] },
    { "name": "marker_pos",     "dtype": "int64",   "shape": [1, 3] },
    { "name": "marker_mask",    "dtype": "bool",    "shape": [1, 3] },
    { "name": "qtype",          "dtype": "int64",   "shape": [1] }
  ],
  "outputs": [
    { "name": "logits",     "dtype": "float32", "shape": [1, 3] },
    { "name": "act_logits", "dtype": "float32", "shape": [1, 2] }
  ]
}
```

`backend` names the kernel serving the model. `runtime` and `device` are present
when the kernel reports them: TensorRT supplies its version and the CUDA device
name, ONNX Runtime supplies its version and the execution provider (`cuda`,
`directml`, or `cpu`). Without an explicit provider, the legacy automatic path
may fall back to CPU and include the reason in `device`.

A `shape` dimension of `-1` is dynamic — the model accepts a range there, and
`/api/v1/limits` reports what it settled on. An ONNX graph exported with dynamic
axes declares most of its dimensions this way, so `-1` is normal rather than a
fault.

Returns `404` with `engine_not_loaded` when nothing is loaded.

---

## POST /api/v1/engine/load

Loads a model and pre-allocates its execution contexts.

Request:

```json
{
  "path": "C:\\models\\laya_fp16.engine",
  "contexts": 2,
  "backend": "tensorrt",
  "provider": "cuda"
}
```

`contexts` is optional; `0` or omitted picks a count from free VRAM. Only the
TensorRT path uses it — an ONNX Runtime session reports the concurrency it
actually allows, which is one run at a time.

`backend` and `provider` are optional and override the process configuration for
**this load only**. An unrecognised `backend` is rejected with `400
invalid_backend` rather than ignored. `provider` selects the ONNX Runtime
execution provider (`cuda`, `cpu`, `directml`).

The override is deliberately not persisted. A load that names a kernel is
describing that model, so folding the name into the switcher's configuration
would silently re-point every later load: after one `"backend": "onnx"` request,
a subsequent plain `.engine` load was refused by the ONNX kernel, which is what
made switching back look broken. Each load instead resolves its kernel from the
file's format when `backend` is absent — `.engine` goes to TensorRT, `.onnx` to
ONNX Runtime — so naming a kernel is never necessary just to go back.

The kernel can also be switched in place: a load naming a different backend than
the one currently resident installs the new kernel and then releases the old one,
so no request observes a backend with nothing loaded. Reloading into the same
kernel reuses it, which keeps the TensorRT path at one model of peak VRAM.

Because the format decides, the load never probes the other kernel first. An
`.onnx` load does not open the TensorRT engine at all — which matters when a plan
already holds the GPU, since that wasted attempt could exhaust memory before the
real load was ever tried.

Response: the same body as `GET /api/v1/engine`, plus a `backend_note` field when
`auto` fell back:

```json
{
  "loaded": true,
  "path": "C:\\models\\laya_dyn.onnx",
  "backend": "ONNX Runtime",
  "runtime": "1.25.0",
  "device": "cpu",
  "contexts": 1,
  "activation_memory_mb": 0,
  "backend_note": "tensorrt unavailable (...); using ONNX Runtime",
  "inputs": [ ... ],
  "outputs": [ ... ]
}
```

`backend_note` is present only when a fallback happened, and is how a client
learns that its `auto` load landed on a kernel other than the preferred one
instead of having to infer it from `backend`.

Errors: `400 invalid_request` (missing path), `400 invalid_backend` (unknown
kernel name), `404 engine_not_found`, `409 engine_wrong_kernel` (a `.engine` handed
to the ONNX kernel, or an `.onnx` handed to TensorRT — select the other kernel),
`409 engine_incompatible` (built for another TensorRT version or GPU), `503
runtime_missing` (the kernel's runtime library is not present), `500 load_failed`.

---

## POST /api/v1/engine/unload

Drains the context pool and releases the engine. Idempotent.

```json
{ "unloaded": true }
```

---

## GET /api/v1/device

```json
{
  "name": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
  "compute_capability": "12.0",
  "vram_free_mb": 11042,
  "vram_total_mb": 12199
}
```

---

## POST /api/v1/predict

The main use case: evaluate typed questions over a state in one forward pass.

Request:

```json
{
  "state": {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "We were billed twice for March. Please refund the duplicate today."
  },
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this request?",
      "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "sales": "pricing, new contracts"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this request?",
      "criteria": ["not urgent", "soon", "critical deadline"]
    },
    "churn_risk": {
      "type": "noul",
      "instructions": "Does the user threaten to cancel or leave?"
    }
  }
}
```

`state` accepts a string, an object, or an array of turns. An object or array is
serialised to compact JSON before tokenising, exactly as the Python SDK does.

`questions` maps a question id to a definition. Three types:

| type | `criteria` | answer field |
|---|---|---|
| `choice` | object `{label: description}` or array of labels | `choice`, `probabilities` |
| `score` | array of level descriptions, low to high | `score`, `legend`, `probabilities` |
| `noul` | omitted | `noul` |

Response:

```json
{
  "model": "laya-rl-agent",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": { "billing": 0.9929, "technical": 0.0056, "sales": 0.0014 },
      "confidence": 0.9755,
      "action": { "act_probability": 1.0 }
    },
    "urgency": {
      "type": "score",
      "score": 1.8412,
      "legend": { "0": "not urgent", "1": "soon", "2": "critical deadline" },
      "probabilities": { "0": 0.18, "1": 0.62, "2": 0.2 },
      "confidence": 0.4122,
      "action": { "act_probability": 0.83 }
    },
    "churn_risk": {
      "type": "noul",
      "noul": 0.8538,
      "confidence": 0.8538,
      "action": { "act_probability": 1.0 }
    }
  },
  "usage": { "input_tokens": 64, "output_tokens": 0 },
  "timing": { "total_ms": 4.49, "tokenize_ms": 0.21, "inference_ms": 4.1 }
}
```

`confidence` is normalised Shannon entropy: `1 - H(p)/log(k)`, so it is `1.0`
for a single-option question and near `0` for a uniform distribution.

Errors: `409 engine_not_loaded`, `400 invalid_request` (unknown question type,
empty criteria, a question whose options do not fit the head budget),
`500 inference_failed`.

---

## POST /api/v1/tokenize

Inspect how a string is tokenised, for debugging prompt construction.

Request:

```json
{ "text": "Duplicate charge on invoice #4411", "with_specials": false }
```

Response:

```json
{
  "count": 9,
  "ids": [13759, 4073, 319, 11475, 4758, 489, 15827, 26, 20554],
  "tokens": ["Duplicate", "Ġcharge", "Ġon", "Ġinvoice", "Ġ#", "44", "11", ""]
}
```

`with_specials` wraps the result in `[CLS] ... [SEP]` when true.

---

## POST /api/v1/sequence

Builds the full token sequence for one question, so the prompt layout can be
verified without running inference.

Request:

```json
{
  "state": { "body": "Billed twice" },
  "question": {
    "type": "choice",
    "instructions": "Which department?",
    "criteria": { "billing": "invoices", "technical": "bugs" }
  }
}
```

Response:

```json
{
  "ids": [50281, 13759, 4073, 50282, 50284, 13759, 50284, 4073, 50282, 1234, 50282],
  "length": 11,
  "marker_positions": [4, 6],
  "options": ["billing: invoices", "technical: bugs"]
}
```

`marker_positions` are the indices of the `[MASK]` tokens the decision head
scores.

---

## GET /api/v1/metrics

Counters and latency percentiles since start.

```json
{
  "requests_total": 128,
  "requests_failed": 0,
  "predict_total": 120,
  "latency_ms": { "p50": 4.49, "p90": 6.29, "p99": 9.1, "min": 2.36, "max": 14.2 },
  "last_run": {
    "at": "2026-09-22T14:31:02Z",
    "total_ms": 4.49,
    "tokenize_ms": 0.21,
    "inference_ms": 4.1
  }
}
```

`latency_ms` covers the last 512 predictions (a bounded ring, so memory does
not grow with uptime).

---

## Conventions

* Field names are `snake_case` throughout, matching the Python SDK's dict keys.
* Timestamps are RFC 3339 UTC.
* Probabilities and confidences are rounded to 4 decimals, as the SDK does.
* `POST` bodies larger than 1 MiB are rejected with `413 payload_too_large`.
* The server binds `127.0.0.1` by default; `--addr 0.0.0.0:8420` exposes it.

### Error codes

The distinctions are meant to be actionable, so the status tells a client whether
retrying or reconfiguring is the fix.

| code | status | meaning |
|---|---|---|
| `invalid_request` | 400 | malformed body, missing `path`, or an unusable question |
| `invalid_backend` | 400 | a kernel name that is not `auto`, `tensorrt` or `onnx` |
| `engine_not_loaded` | 404 / 409 | nothing is resident; `404` from a description, `409` from a call that needs a model |
| `engine_not_found` | 404 | the model file does not exist |
| `engine_wrong_kernel` | 409 | the file belongs to the other kernel: a `.engine` handed to ONNX Runtime, or an `.onnx` handed to TensorRT. Select the other kernel rather than replacing the file |
| `engine_incompatible` | 409 | the model is not one this kernel can serve: built for another TensorRT version or GPU |
| `manager_closed` | 409 | the backend was closed and cannot be reused |
| `runtime_missing` | 503 | the kernel's own runtime library is absent — `onnxruntime.dll`, or the bridge itself |
| `payload_too_large` | 413 | body over 1 MiB |
| `inference_failed` | 500 | the forward pass failed |
| `load_failed` | 500 | a load failed for a reason not covered above |
| `encode_failed` | 500 | the response could not be encoded |

`runtime_missing` is worth separating from `load_failed`: it says this machine
cannot run that kernel at all, which is fixable by installing the runtime or
pointing `onnx_runtime_path` at it, whereas `engine_incompatible` says the file
is wrong and no install will help. See [kernels.md](kernels.md) for what each one
means in practice.
