# Kernels

A **kernel** here is the library that turns a model file into answers. laya-trt
supports two, and the rest of the program does not care which one is running:
the predict use case asks the same questions of either, through the contract in
`internal/backend`.

The reason to have two is not that one is better. TensorRT is the fast path and
the historical one; ONNX Runtime is the portable path, and it is what lets this
program run a model on a machine that has no TensorRT install at all. Which one
you get should be a deliberate choice, so this document is about making that
choice and about what each error means when it goes wrong.

## The two kernels

|  | TensorRT (`tensorrt`) | ONNX Runtime (`onnx`) |
|---|---|---|
| model file | a `.engine` plan | a `.onnx` graph |
| needs | TensorRT 10.x, CUDA, an NVIDIA GPU | `onnxruntime.dll`, and the bridge |
| concurrency | a pool of contexts, sized from free VRAM | one run at a time, serialised |
| activation memory | reported, and reserved per the plan's profile | not reported (`0`) |
| typical use | production latency | portability, CPU-only machines, comparison |

TensorRT is the faster path, and it is the default for that reason: its plan is
compiled with fused kernels and runs on the GPU, while the ONNX path on the CPU
provider has neither. What ONNX Runtime buys is that the program starts and runs
when TensorRT is not installed — including on a machine with no NVIDIA GPU, where
the ONNX path falls back to the CPU provider and still produces the same decision.

The one number this document can state honestly is the TensorRT one: the reference
engine answers in about 24 ms. The ONNX path's latency depends on the provider and
the machine, so measure it with `bench/apibench` rather than assuming it from
here. The cross-kernel check that matters is agreement, not speed, and it is
covered below.

A third name, `auto`, is not a kernel. It is a policy, described below.

## Choosing a kernel

There are four places to make the choice, and they are checked in this order:
the per-request field wins over the CLI flag, which wins over the environment
variable, which wins over the config file. `auto` is the default everywhere, so
not choosing is itself a choice — and it means the TensorRT path, which is what
this project did before the option existed.

### Config file

`layatrt.config.json`:

```json
{
  "backend": "auto",
  "provider": "cuda",
  "onnx_runtime_path": "",
  "onnx_bridge_path": ""
}
```

| field | meaning |
|---|---|
| `backend` | `auto`, `tensorrt` or `onnx`. Empty or absent means `auto` |
| `provider` | ONNX Runtime execution provider: `cuda`, `cpu`, `directml`. Empty lets the backend choose |
| `onnx_runtime_path` | where `onnxruntime.dll` is; empty discovers it |
| `onnx_bridge_path` | where `layatrt_onnx.dll` is; empty resolves it beside the executable |

A config file written before these keys existed has none of them, which parses as
`auto` — the TensorRT path. That is deliberate: the option was added without
changing what an existing file means.

### Environment variable

| variable | overrides |
|---|---|
| `LAYA_TRT_BACKEND` | `backend` |
| `LAYA_TRT_PROVIDER` | `provider` |
| `LAYA_TRT_ONNX_RUNTIME` | `onnx_runtime_path` |
| `LAYA_TRT_ONNX_BRIDGE` | `onnx_bridge_path` |

An unparseable `LAYA_TRT_BACKEND` is reported at startup rather than at the first
load, because a typo in configuration should not look like a broken model.

### CLI flag

`layatrt-server` takes `-backend`, `-provider` and `-onnx-runtime`;
`predictprobe` takes `-backend` and `-provider`. Both validate the name before
doing any work, so a bad value is a usage error:

```powershell
.\layatrt-server.exe --engine laya_1k.engine --backend tensorrt --provider cuda
.\layatrt-server.exe --engine laya_dyn.onnx  --backend onnx --provider cpu
```

`--provider` only affects the ONNX path. TensorRT has no equivalent knob here.

### Per request

The GUI exposes `auto`, `TensorRT`, `ONNX CUDA`, `ONNX DirectML` and `ONNX CPU` as separate manual choices. Switching between ONNX modes reloads the same `.onnx` graph with the selected provider; TensorRT needs a corresponding `.engine` plan. The active `device` field shows which provider was actually loaded. CUDA and DirectML require matching ONNX Runtime distributions under `Assets/onnx/cuda12`, `Assets/onnx/cuda13`, or `Assets/onnx/directml`. Set `LAYA_TRT_ONNX_ROOT` to another directory containing that `Assets/onnx` tree to share existing packages across projects. A fixed `onnx_runtime_path` selects one DLL only and should be left empty for CUDA/DirectML switching.

`POST /api/v1/engine/load` accepts optional `backend` and `provider` fields that
apply to that load only. This is what makes the switch a runtime operation rather
than a startup decision: a server started on `auto` can be handed an ONNX graph
and will move to the ONNX kernel, then be handed a plan and move back. The old
kernel is released only after the new one has its model resident, so no request
sees an empty backend.

```bash
curl -X POST http://127.0.0.1:8420/api/v1/engine/load \
  -H 'Content-Type: application/json' \
  -d '{ "path": "C:\\models\\laya_dyn.onnx", "backend": "onnx", "provider": "cpu" }'
```

`GET /api/v1/backends` reports what is available before you commit, and
`GET /api/v1/health` reports what is loaded now. Both are described in
[api.md](api.md).

## What `auto` means

`auto` is a policy, and it is deliberately narrow. The model's **format** is
resolved first, because the two formats are not interchangeable and the file name
already says which kernel is even possible:

1. An `.onnx` path goes straight to ONNX Runtime. TensorRT is never opened.
2. Any other path goes to TensorRT. If it loads the model, that is the answer.
3. If TensorRT fails, and the path is not a `.engine` plan, and the failure looks
   like a format rejection rather than a missing file or an exhausted GPU, try
   ONNX Runtime once.
4. If the path is a `.engine` plan, report the TensorRT error.

Step 1 is what makes switching practical. Trying TensorRT first "just in case"
means reading a multi-gigabyte graph into memory only to have deserialization
reject it — seconds of work and a spike in peak memory for an answer the file name
already gave. When a plan already holds the GPU, that wasted attempt can exhaust
memory before the real load is ever tried, which is what made switching from
TensorRT to ONNX look impossible.

Step 4 is the part worth understanding. A plan cannot be opened by ONNX Runtime —
the formats are different — so falling back would turn a clear "this plan is
broken" into a confusing double failure. `auto` also refuses to fall back when
TensorRT was asked for by name, and a kernel named explicitly is never
substituted at all.

That last rule is the most important one in this document. A request to run on
the GPU that quietly lands on the CPU is worse than a failure, because it looks
like it worked: the timings are wrong, the capacity planning is wrong, and the
only symptom is that everything is slow. So `--backend tensorrt` fails loudly if
TensorRT cannot serve the model, and `auto` reports a `backend_note` when it
falls back, so the substitution is visible rather than inferred.

## Exporting an ONNX graph

The ONNX path needs a `.onnx` file with the same IO contract the TensorRT path
expects: five inputs (`input_ids`, `attention_mask`, `marker_pos`, `marker_mask`,
`qtype`) and two float32 outputs (`logits`, `act_logits`).

Use `tools/export_onnx.py`; do not call `torch.onnx.export` directly.

```powershell
python tools\export_onnx.py --out laya_dyn.onnx --seq-len 512 --markers 32
```

The script exists because the laya head uses `nn.TransformerEncoder`, and
PyTorch's exporter folds the trace length into a reshape constant. The resulting
graph is only correct at the length it was traced at, which makes a TensorRT
engine built from it accept exactly one sequence length and makes an ONNX graph
with `dynamic_axes` declare a shape that is not actually dynamic. The script
reimplements the head's attention with the layer's own weights so the shape stays
symbolic, and it verifies the exported graph against the unmodified model at the
traced length **and** at a different one. That second check is the one that
catches the bug.

The export is the same file the TensorRT path consumes, so a single `.onnx` can
serve both kernels — which is also what makes the two comparable.

## Where the runtime libraries go

The ONNX kernel has two moving parts, and they are separate on purpose:

* `layatrt_onnx.dll` — the bridge, built from this repository. It is opened with
  `LoadLibrary` and every entry point is resolved by name, so it is a *runtime*
  dependency, not a load-time one.
* `onnxruntime.dll` — the ONNX Runtime itself, which the bridge opens and
  resolves `OrtGetApiBase` from.

Neither is linked. That is why a machine without ONNX Runtime still starts the
GUI, the server and the doctor command and reports what is missing, instead of
dying before `main()` with `0xC0000135` when a TensorRT DLL is genuinely missing.

Resolution order for each:

| library | searched in order |
|---|---|
| `layatrt_onnx.dll` | `onnx_bridge_path` / `LAYA_TRT_ONNX_BRIDGE`, then beside the executable, then `build\bin\Release` under the executable's directory, then the working directory |
| `onnxruntime.dll` | `onnx_runtime_path` / `LAYA_TRT_ONNX_RUNTIME`, then beside the executable, then `runtime\` and `Assets\` beside it, then the DLL search path |

Putting `onnxruntime.dll` next to the executable is the simplest working layout,
and it is what the discovery order expects first. The version matters: the
vendored header declares `ORT_API_VERSION 17`, and a runtime that reports an
older API fails with "ONNX Runtime C API version is incompatible" rather than
misbehaving. Upgrade the header and the DLL together.

## Building the bridge

```powershell
.\build.ps1
```

This builds both kernels into `build\bin\Release`. The TensorRT kernel needs
TensorRT and CUDA; the ONNX bridge needs only the C API header, which is vendored
in `third_party/onnxruntime` so the build works with no network access and no ORT
SDK install. When that header is missing, CMake warns and skips the bridge while
still building the TensorRT kernel, so a TensorRT-only machine is not blocked by
the second kernel.

To build against a different ONNX Runtime:

```powershell
.\build.ps1 -ORTHeaderDir "C:\path\to\onnxruntime\include"
```

or, directly through CMake, `-DORT_HEADER_DIR=...`. Only the header is needed —
never the import library, because nothing links against it.

The ONNX bridge is Windows-only. On other platforms `internal/ortbackend` still
compiles and reports the ONNX kernel as unavailable with a clear reason, so
`go build ./...` and `go vet ./...` work anywhere.

## Troubleshooting

Start with `GET /api/v1/backends` or `layatrt-doctor`. The endpoint reports which
kernels are available and why one is not, without loading a model; the doctor
command links no kernel at all, so it runs even when nothing else will.

### `runtime_missing` (503)

The kernel's own runtime library is not there. For the ONNX path this is either
the bridge or `onnxruntime.dll`; the message names which.

```
backend: runtime library is missing: onnxruntime.dll not found
  (put it beside the executable or set RuntimePath)
```

Put `onnxruntime.dll` beside the executable, or set `onnx_runtime_path`. If the
message instead says `cannot load ...layatrt_onnx.dll`, the bridge has not been
built — run `.\build.ps1`. A message about a missing *export* means the bridge is
older than this build of the Go code, which happens after pulling changes: rebuild
it.

### `invalid_backend` (400)

The name is not `auto`, `tensorrt` or `onnx`. The aliases `trt`, `plan`, `engine`,
`onnxruntime` and `ort` are also accepted, case-insensitively. Note that `cuda`
is **not** a backend name — it is a provider, and belongs in `provider`.

### `engine_wrong_kernel` (409)

The file belongs to the other kernel. This is a configuration mistake, not a
corrupt file, and the message says so: *"... is a TensorRT plan; select the
tensorrt kernel for it"*, or the mirror case for an `.onnx` handed to TensorRT.
Select the other kernel, or let `auto` pick — it resolves the kernel from the
format, so it will not make this mistake.

It is separate from `engine_incompatible` because the remedy is different: pick
the other kernel, rather than rebuild or replace the file.

### `engine_incompatible` (409)

The model is a valid file for this kernel but not one it can serve — typically a
plan built for a different TensorRT version or a different GPU architecture.
Rebuild the engine on this machine.

### "ONNX Runtime C API version is incompatible"

The `onnxruntime.dll` is older than the vendored header's `ORT_API_VERSION 17`.
This is reported rather than tolerated because an older `OrtApi` would be read
through a newer struct layout. Replace the DLL with a matching build, or vendor a
matching header and rebuild the bridge.

### An explicit ONNX provider cannot load

Selecting `ONNX CUDA` or `ONNX DirectML` is strict. If the matching provider DLL or its dependencies are unavailable, the load returns an error and keeps the previous model resident; it does **not** report a successful CPU session. Check the error and ensure the matching ONNX Runtime distribution is installed. A load with no explicit provider retains the legacy CUDA-to-CPU fallback; check `device` to see what it actually used. Selecting `ONNX CPU` requests CPU explicitly.

### The two kernels disagree

`internal/inference` → `TestKernelsAgree` runs the same request through both and
compares. It is not a bit-exactness test: the runtimes use different kernels and
different precisions, so logits legitimately differ in the low bits. What must
hold is that the choice is identical and the probabilities are within 0.05.

If you are comparing by hand and see a larger gap, check the calibration before
suspecting the kernel: `rl_agent_config.json` supplies the temperatures, and
without it both paths compute probabilities at temperature 1.0, which is wrong in
a way that can look like a kernel difference. Also confirm both are reading the
same graph — an `.onnx` and a `.engine` built from a different export are not
expected to agree.

## Verified

The numbers below are from real runs on the reference machine (RTX 5070 Ti
Laptop), not estimates.

**TensorRT, end to end.** A predict request through the plan returned
`choice = billing` with `billing = 0.9983`, in 23.7 ms.

**ONNX Runtime.** Loads the real 1.57 GB laya graph, reports the correct
contract — 5 inputs, 2 outputs — and runs inference. `GET /api/v1/engine` after
an ONNX load shows the symbolic dimensions the graph declares, which is what
`InputBounds` refuses to turn into a maximum:

```json
{
  "backend": "ONNX Runtime",
  "runtime": "1.25.0",
  "device": "cpu",
  "contexts": 1,
  "activation_memory_mb": 0,
  "inputs": [
    { "name": "input_ids",      "dtype": "int64", "shape": [-1, -1] },
    { "name": "attention_mask", "dtype": "int64", "shape": [-1, -1] },
    { "name": "marker_pos",     "dtype": "int64", "shape": [-1, -1] },
    { "name": "marker_mask",    "dtype": "bool",  "shape": [-1, -1] },
    { "name": "qtype",          "dtype": "int64", "shape": [-1] }
  ],
  "outputs": [
    { "name": "logits",     "dtype": "float32", "shape": [-1, -1] },
    { "name": "act_logits", "dtype": "float32", "shape": [-1, 2] }
  ]
}
```

`contexts` is `1` because that is the concurrency the ONNX path really allows,
and `activation_memory_mb` is `0` because ONNX Runtime does not report one — not
because the model uses none.

**Cross-kernel agreement.** `TestKernelsAgree` passes with both kernels agreeing
exactly:

```
choice = billing
account = 0.0001   billing = 0.9997   technical = 0.0002
confidence = 0.9976
```

Same choice, probabilities equal to four decimals, so the switch changes the
runtime without changing the answer.
