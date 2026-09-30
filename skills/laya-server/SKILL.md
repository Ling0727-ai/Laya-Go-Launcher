---
name: laya-server
description: Call or operate the local Laya inference server (layatrt-server / layatrt-gui) to classify text with laya's choice / score / noul questions, and to check or change its model, fallback chain (tensorrt → onnx-cuda → onnx-directml → onnx-cpu), base config, and TensorRT conversion. Use when a task needs laya predictions over HTTP, or needs the Laya server started, reconfigured, or diagnosed.
---

# Laya Server

Local HTTP service that answers structured questions about a text. Default base URL `http://127.0.0.1:8420`; the API lives under `/api/v1`, a web console at `/`, and the full contract at `/api/v1/openapi.json`.

## Start

From the repo root (`laya-trt`), no arguments needed:

```powershell
.\layatrt-server.exe                      # console: http://127.0.0.1:8420/
go build -o layatrt-server.exe ./cmd/layatrt-server   # rebuild (CGO_ENABLED=1)
```

Defaults: model must accept 8192 tokens, backend `auto`, chain `tensorrt,onnx-cuda,onnx-directml,onnx-cpu`, `auto_convert` on. The model loads in the background after the listener is up, so poll readiness before predicting:

```powershell
Invoke-RestMethod http://127.0.0.1:8420/api/v1/load   # phase: loading | ready | failed
```

Useful flags: `--addr`, `--engine <file>` (try first), `--backend onnx --provider cpu` (one kernel, no fallback), `--fallback onnx-cuda,onnx-cpu`, `--seq 2048`, `--no-convert`, `--no-load`, `--config <file>`, `--write-config`. Each has a `LAYA_TRT_*` env equivalent (e.g. `LAYA_TRT_FALLBACK`, `LAYA_TRT_MODEL_SEQ`, `LAYA_TRT_ADMIN_TOKEN`).

If port 8420 is taken the server exits instead of moving ports; pick another `--addr`. The desktop app (`cmd/layatrt-gui`) runs the same launcher and API in-process, but it moves to the next free port when the configured one is taken.

## Predict

`POST /api/v1/predict`, body `{ "state": <text>, "questions": { <name>: <spec> } }`.

| type | spec | answer fields |
|---|---|---|
| `choice` | `criteria`: object label → description | `choice`, `probabilities`, `confidence` |
| `score` | `criteria`: array of level descriptions, low → high | `score`, `legend`, `confidence` |
| `noul` | no criteria; `instructions` is a yes/no question | `noul` (probability of yes), `confidence` |

```powershell
$body = @{
  state = 'I was charged twice this month and want my money back.'
  questions = @{
    department = @{ type='choice'; instructions='Which team should handle this ticket?'
      criteria=@{ billing='payments, refunds'; technical='bugs, outages'; sales='pricing, plans' } }
    refund = @{ type='noul'; instructions='Is the customer asking for a refund?' }
  }
} | ConvertTo-Json -Depth 6
Invoke-RestMethod http://127.0.0.1:8420/api/v1/predict -Method Post -ContentType application/json -Body $body
```

Put several questions in one request instead of one request per question: they share one forward pass, and each extra question costs about 1 ms, against roughly 7 ms of fixed cost per pass. Treat `confidence` below ~0.5 as uncertain. Unknown JSON fields are rejected with 400.

## Operate

Read-only (all under `/api/v1`): `/health` (kernel, device, engine, load state), `/load` (last load and every fallback attempt), `/models` (discovered files), `/models/plan` (the order an auto load would try), `/config`, `/convert` (build jobs), `/metrics`.

Mutating (admin; need `Authorization: Bearer <admin_token>` when a token is configured; cross-origin browser requests are refused):

- `PUT /api/v1/config`: full config object; add `"reload": true` to rerun the chain. Saved to `layatrt.config.json`. `http_addr` changes need a restart. Read with GET first, edit fields, and send it back. `admin_token` comes back as `***`; send `***` to keep it.
- `POST /api/v1/load/auto`: rerun the fallback chain now.
- `POST /api/v1/engine/load` `{ "path", "backend"?, "provider"?, "contexts"? }`: load one file explicitly. No fallback applies.
- `POST /api/v1/convert` `{ "seq"?, "precision"?, "source"?, "builder_opt"? }`: build a TensorRT plan with trtexec in the background (202). A build takes minutes. Only one build runs at a time (409 if busy). `POST /api/v1/convert/cancel` stops it. A finished build is picked up automatically if the server is running on a slower fallback.

## Selection rules worth knowing

- An auto load never picks a model whose max sequence is below `model_seq`, and never picks an `.engine` whose seq cannot be read from its file name, `engines/manifest.json`, or the `<file>.meta.json` sidecar. A pinned `engine_path` is tried first.
- In the TensorRT step, two-profile plans (`*_p2.engine`) rank first. They are several times faster on short inputs.
- Fused `.opt` ONNX graphs are preferred on CUDA/CPU, plain graphs on DirectML. Conversion always builds from a plain graph, because the TensorRT parser rejects the fused contrib ops.
- A step is skipped, not failed, when its runtime is missing. The reason is in `/load` → `attempts[].error`.
- An explicit `backend` (not `auto`) disables the chain. A failed load is then an error rather than a silent CPU fallback.

## Diagnose

- `phase: failed`: read `attempts`. Plan deserialize errors usually mean a TensorRT or GPU mismatch, and with `auto_convert` on a rebuild starts automatically, once per file. Out-of-memory: close other GPU users or lower `contexts`.
- `no tokenizer.json found`: `huggingface-cli download convaiinnovations/laya --include "tokenizer/*" rl_agent_config.json`.
- Native runtime discovery is logged at startup on the `native:` line. `layatrt-doctor.exe` gives a full environment check.
