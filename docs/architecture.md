# Architecture

## What this is

Three things that share one core:

1. **Two interchangeable kernels** — the QualityScaler-go `ai_tensorrt.cpp`,
   copied and stripped of its image/video pipeline, with the tensor contract
   generalised so a transformer's `int64`/`bool` inputs can be expressed; and an
   ONNX Runtime kernel built on a native bridge written for the same contract.
2. **A Go runtime** — the execution contract, the kernel selection policy,
   engine registry, context pool, host-memory allocators, and a pure-Go
   implementation of the laya tokenizer and sequence builder.
3. **Two front doors** — a Wails desktop GUI and an HTTP API, both calling the
   same use cases.

The point of (2) is that nothing in the request path leaves the process. The
kernel takes 4 ms; shelling out to Python for tokenisation would cost more than
that, so the tokenizer is ported and verified against the reference.

The point of (1) is that the kernel is a choice, not an assumption. The use case
asks for a model to be loaded and a forward pass to be run; which library does
that — a serialized TensorRT plan or an ONNX graph — is decided below it, and a
process that starts on one kernel can switch to the other without the use case
being rebuilt. The two kernels are not equivalent, and the selection policy is
written to keep that visible rather than paper over it.

## Layers and dependency direction

```
cmd/layatrt-gui          Wails window          ┐
cmd/layatrt-server       headless HTTP         ├ entrypoints
cmd/predictprobe         bisection harness     ┘
        │
        ├── internal/app             config, HTTP listener, Wails bindings
        └── internal/transport/httpapi   decode → use case → encode
                    │
                    └── internal/inference   predict use case
                                │
                                ├── internal/sequence   token layout
                                ├── internal/tokenizer  ByteLevel BPE
                                └── internal/backend    the execution contract
                                            │
                                            └── internal/backends   factory, policy, switcher
                                                        │
                                                        ├── internal/trtbackend
                                                        │       └── internal/engine → internal/kernel
                                                        │                   └── qualityscaler_tensorrt.dll
                                                        └── internal/ortbackend
                                                                └── layatrt_onnx.dll → onnxruntime.dll
```

Dependencies point one way only: entrypoint → transport → use case → contract →
kernel. Nothing below `internal/backend` knows the HTTP or GUI layers exist, and
neither kernel package knows anything about laya.

`internal/metrics` is a leaf that the transport writes and the GUI reads.

The arrow that used to run from the use case straight into `internal/engine` now
stops at `internal/backend`. That is the whole change: `internal/inference` names
the contract, and `internal/backends` is the only package that knows both
concrete kernels exist and imports either adapter.

## Why the pieces are where they are

**`internal/backend`** is the seam. It declares what the predict use case needs
from a kernel and nothing more: load a model, describe its IO, report the range
an input dimension accepts, run one forward pass over named tensors, release it.
`Backend` is an interface over `Options`, `Info`, `TensorInfo`, `Tensor`,
`RunInput` and `Output`, all expressed in kernel-neutral terms, so
`internal/inference` can be written against it and never learn whether a plan or
a graph is resident. The sentinel errors (`ErrNotLoaded`, `ErrClosed`,
`ErrUnsupported`, `ErrRuntimeMissing`, `ErrIncompatible`) are what lets a caller
classify a failure without matching on message text — which is what the transport
does when it maps a load failure onto a wire code.

The types carry two decisions worth naming. `DType` uses the ONNX Runtime
element-type numbering, so the ONNX backend passes it straight through and only
the TensorRT adapter has to translate; it is an internal convention, not a wire
format. `TensorInfo.Shape` reports a dynamic dimension as `-1`, and
`NumElements` returns `0` for such a shape rather than a wrong count.

**`internal/backends`** is the factory and the policy. It is the one place that
imports both adapters, which is what keeps the dependency arrow pointing the
right way. Its policy is deliberately boring, and each clause is there for a
reason:

* The default is `auto`, which means the TensorRT path. A config file written
  before this option existed has no `backend` key, so it must keep meaning what
  it meant.
* A kernel named explicitly is used as named. If it cannot start, that is an
  error, not a silent substitution: a request to run on the GPU that quietly
  lands on the CPU is worse than a failure, because it looks like it worked.
* `auto` may fall back, and reports a note when it does. The fallback is only
  attempted when the model is an ONNX graph, because a plan file cannot be opened
  by ONNX Runtime and trying would turn a clear "this plan is broken" into a
  confusing double failure.

`Probe` answers "which kernels could run here" without loading a model, and is
what `/api/v1/backends` and the startup log report. `Inspect` opens a model just
far enough to read its IO tensors and releases it again; the kernel is chosen
from the file extension there, because that is a property of the formats rather
than a preference, so it does not consult the configured backend.

**`internal/backends.Switcher`** is what makes the switch a runtime operation
rather than a startup decision. It holds the active kernel in one place and
implements `backend.Backend` itself, so the use case is handed a single object
that can later be pointed at a different kernel. It exists for a second, more
basic reason too: loading a model means creating a kernel, and if the caller that
loads is not the caller that predicts, the model lands in a backend nobody reads.
One holder makes that mistake impossible.

Swapping is atomic from a request's point of view. A caller sees either the old
kernel or the new one, never a half-installed pair, because the active pointer is
only replaced once the new kernel has a model resident. Reloading into the *same*
kernel reuses the existing backend, so the TensorRT path keeps its
one-model-of-peak-VRAM property; switching kernels builds and loads the new one
first and releases the old one after, so peak memory is briefly two models but no
request can observe an empty backend.

**`internal/trtbackend`** implements the contract over `internal/engine`. This is
the historical path and its behaviour is unchanged: a plan is loaded through
`internal/kernel`, a pool of execution contexts is pre-allocated, and a run
borrows one. Everything TensorRT-specific — the context pool, the VRAM-derived
context count, the activation-memory accounting — stays inside this package, so
the use case never sees a plan, a context or a CUDA stream. It also translates
the manager's own sentinels onto the neutral ones, so a caller classifies a
failure without knowing which kernel produced it.

**`internal/ortbackend`** implements the contract over ONNX Runtime through the
native bridge. An ONNX Runtime session is not safe to run concurrently from
several goroutines, so runs are serialised on a mutex and `Info.Contexts` reports
`1` — the concurrency the path really allows — rather than a VRAM-derived guess.
The same guarantee the TensorRT path gets from its context pool, expressed as a
lock because ORT does its own intra-op scheduling.

**`internal/kernel`** is the only package with cgo. Every call goes through an
exported function; no C struct is declared or laid out on the Go side. Descriptors,
staging buffers and output buffers are owned by the native run builder, so the
boundary is the documented C ABI and nothing else. The ONNX path deliberately
does *not* use cgo — see below.

**`internal/engine`** carries the ported allocation strategy. It is deliberately
separate from `internal/kernel`: the kernel knows how to run a plan, the engine
knows how to keep one resident and hand out contexts and buffers.

**`internal/tokenizer`** and **`internal/sequence`** are pure Go with no kernel
dependency, which is what lets them be tested against Python without a GPU.

**`internal/inference`** is the only place that knows the model's IO contract —
five inputs named `input_ids`/`attention_mask`/`marker_pos`/`marker_mask`/`qtype`,
two outputs named `logits`/`act_logits`. It names the kernel through
`backend.Backend` and never the concrete type.

**`internal/transport/httpapi`** decodes, calls, encodes. It holds no model
logic, so the GUI and a REST client cannot behave differently.

## The ONNX Runtime bridge

The second kernel is served by `include/layatrt_onnx.h` and
`src/layatrt_onnx.cpp`, a generic-tensor bridge ported from QualityScaler-go's
`pkg/func/ortcuda`. The loading strategy is copied deliberately: `onnxruntime.dll`
is opened with `LoadLibraryW`, `OrtGetApiBase` is resolved with `GetProcAddress`,
and the session is driven through the `OrtApi` that returns. Nothing links
against `onnxruntime.lib`.

What differs from QualityScaler-go is the tensor contract. That kernel was
written for one image model — a single float32 `[1,3,H,W]` input and one output,
sized from width/height/scale. laya's decision model takes five named inputs of
mixed dtype (`int64`, `int64`, `int64`, `bool`, `int64`) and returns two float32
outputs, so this ABI is generic over a flat descriptor list: name, dtype, rank,
shape and a host pointer. A run is described, not hard-coded.

**Why runtime symbol resolution instead of cgo linking.** The alternative — the
`internal/kernel` approach, a cgo package linking the import library — makes the
DLL a *load-time* dependency of every binary that imports the package. That is
already the source of the `0xC0000279` failure mode on this project: a machine
without TensorRT cannot run `--help`, because the process dies before `main()`.
Linking ONNX Runtime the same way would extend that failure to a second library
and make the ONNX path a prerequisite for starting the GUI, the server and the
doctor command — on a machine that may never use it.

Resolving at runtime inverts that. `internal/ortbackend` uses
`syscall.LoadDLL` and `FindProc`; a missing bridge or a missing `onnxruntime.dll`
is a diagnosis returned from `Load`, not a crash. The `internal/app` config
validates the backend name at startup, `Probe` reports ONNX availability, and
`layatrt-doctor` — which links no kernel at all — keeps working. The price is
that every entry point is a resolved function pointer and every error is a return
code plus a caller-supplied message buffer, since there is no cgo to translate
between representations.

Three consequences of the ownership rule are deliberate and load-bearing:

* **No allocation crosses the ABI in either direction.** Every string the bridge
  produces is copied into a caller-supplied buffer, and every failure is a return
  code plus a message in that buffer. Nothing returns a pointer the caller would
  have to free or convert back into a Go pointer.
* **Outputs are read through ONNX Runtime-owned result values and copied out.**
  A graph exported with dynamic axes declares its outputs symbolically (`[-1,-1]`),
  so there is no shape to size a buffer from until the graph has actually run. The
  run produces results, `LayaOnnx_GetResultInfo` reports their concrete shapes,
  and `LayaOnnx_CopyResult` copies into memory the Go side allocates. The input
  buffers are still the caller's, bound directly with no Go-side copy.
* **`InputBounds` reports `ok == false` for a symbolic dimension** rather than
  inventing a maximum. The caller clamps its token budget to whatever bound it is
  given, so a fabricated one would silently cap (or fail to cap) the sequence
  length. `internal/inference` keeps its configured budget when the model declares
  none.

The bridge is Windows-only: `native_windows.go` holds the syscall plumbing and
`native_stub.go` keeps the package compiling elsewhere, reporting
`ErrRuntimeMissing` with a clear reason. The ONNX Runtime C API header is vendored
in `third_party/onnxruntime` — it is the bridge's only build-time dependency — so
`cmake` can build both kernels with no network access and no ORT SDK install.
`CMakeLists.txt` gates the `layatrt_onnx` target on finding that header and warns
rather than failing when it is absent, so a TensorRT-only machine is not blocked
by the second kernel; `-DORT_HEADER_DIR=...` points at a different ONNX Runtime.

## The request path

```
POST /api/v1/predict
  │
  ├─ httpapi: decode JSON into inference.Request
  │
  ├─ inference.Predict
  │    ├─ serialize the state (string passes through, else compact JSON)
  │    ├─ for each question (sorted, so batch rows are deterministic)
  │    │    ├─ BuildQuestion   → option texts, rendered like the SDK
  │    │    └─ sequence.Build  → ids + [MASK] marker positions
  │    ├─ pad every sequence to one length, pad markers to the engine's K
  │    ├─ backend.InputBounds  → how long and how many options this model accepts
  │    ├─ backend.Run ×5 inputs → the active kernel: pooled context, or a locked ORT session
  │    └─ softmax each logits row → typed Answer (choice/score/noul)
  │
  └─ httpapi: encode the response
```

Two properties of the engine shape this path:

* **A fixed-shape plan pins batch and K.** `input_ids` is `[1,64]` and
  `marker_pos` is `[1,3]` in the reference engine, so the runtime pads to those
  exact values and marks the padding with `marker_mask`. Multi-question requests
  are chunked to the batch dimension the engine declares.
* **Padding is safe.** The encoder honours `attention_mask`, so a padded run
  returns the same decision as an unpadded one (measured: ~1e-5 on logits).

The two properties that differ between kernels are both read from the contract
rather than assumed. `InputBounds` decides the padding: a TensorRT plan reports
the range from its optimisation profile, a fixed plan reports `min == max`, and a
symbolic ONNX dimension reports no bound at all, which leaves the configured
budget in place. `Info.Contexts` describes the concurrency the kernel actually
allows — a VRAM-derived pool for TensorRT, `1` for ONNX Runtime — so the batching
decision above stays honest on either path.

## Memory model

Ported from QualityScaler-go, because it is what keeps the per-run path free of
allocation. It applies to the TensorRT path; the ONNX path has a different shape,
described after the table.

| Mechanism | Where | Purpose |
|---|---|---|
| Context pool | `engine.Manager` | Contexts pre-created at load, borrowed and returned; a run never creates one |
| Pinned host buffers | kernel, per context | Registered once via `TensorRT_SetHostBuffers`, grown only when a shape needs it |
| Size-keyed scratch pool | `engine.Resources` | Bounded map of `sync.Pool`, so a spread of sizes cannot grow without limit |
| Per-shape pair pool | `engine.Resources` | Bounded creation with a blocking wait, so concurrent workers share buffers |
| Native run builder | kernel | Owns descriptors and staging, so no foreign allocation crosses the boundary |

The engine's activation memory is `kUSER_MANAGED`: the kernel sizes it from
`updateDeviceMemorySizeForShapes()` and reallocates only when a shape change
makes it grow.

The ONNX path has no pool to port, because there is no equivalent of an
`IExecutionContext` to pre-create: ONNX Runtime owns its own arena and intra-op
threads, and one `OrtSession` is not safe to run concurrently. What it does keep
is the ownership rule. Input buffers are the caller's and are bound directly, with
no copy on the Go side; output buffers are allocated by the Go side after the run
reports the concrete shapes, then filled by `LayaOnnx_CopyResult`. Nothing is
returned across the ABI for the caller to free, and nothing is retained past the
call. `Info.ActivationMemoryMB` is reported as `0` here, which means "not
reported" rather than "no memory used".

## Threading

Each TensorRT context has its own `IExecutionContext`, CUDA stream and mutex, so
concurrent requests run in parallel rather than serialising on a shared context.
`Manager.WithContext` blocks when every context is busy, which bounds concurrency
by VRAM instead of by luck.

The ONNX path is the opposite by construction: runs take a mutex and queue. That
is a property of the runtime, not a simplification — an `OrtSession` is not
documented as safe for concurrent `Run` calls, and ORT already schedules work
across cores internally. So a process on the ONNX kernel has a concurrency of one
forward pass, and a process on TensorRT has as many as VRAM allows. `Info.Contexts`
reports which, so a client can size its batching rather than guess.

The GUI's bindings and the HTTP handlers both go through the same `Switcher`, so
the two front doors share one active kernel and one pool.

## Frontend

`cmd/layatrt-gui/frontend` follows a strict structure, enforced by
`npm run lint:struct`:

* `src/components/<Parent>/<Child>/` — a child is a subdirectory of its parent,
  never a sibling
* every component is four files: `.tsx` (view), `.data.ts` (types/constants),
  `.api.ts` (contract), `.ts` (logic)
* `src/utils/<domain>/` with an `index.ts` per domain

One consequence worth knowing: because a component's `.ts` and `.tsx` share a
name, TypeScript resolves a bare `./Foo` to the `.ts` logic file. View-layer
imports therefore write the extension explicitly (`./Foo.tsx`), and
`allowImportingTsExtensions` is enabled to permit it.

## Testing

| Test | What it proves |
|---|---|
| `internal/tokenizer` → `TestParityAgainstPython` | ids match the reference tokenizer exactly, across Latin, CJK, Arabic, emoji, multi-space and specials |
| `internal/sequence` → `TestParityAgainstPython` | ids **and** marker positions match `build_sequence`, including truncation |
| `internal/inference` → `TestRawOutputs` | the engine's real IO dtypes and output values |
| `internal/inference` → `TestKernelsAgree` | the same request through both kernels yields the same choice, with probabilities within 0.05 |
| `internal/backend` → `TestParseKind`, `TestTensor*` | the name/alias table, dtype widths and `Validate` rejections |
| `internal/backends` → `TestNew*`, `TestAuto*` | an explicit kernel is never substituted; `auto` does not fall back for a plan path |
| `internal/ortbackend` → `TestLoadDescribesModel`, `TestDynamicInputBoundsAreNotInvented`, `TestRunProducesLayaShapedOutputs`, `TestUnloadAndReload`, `TestRunWithoutLoadFails`, `TestEnginePathRejected` | the bridge reports the 5-in/2-out contract, does not invent a bound for a symbolic dimension, runs real inference, and fails cleanly on a `.engine` |
| `tests/test_kernel.cpp` | the kernel works without Go in the way |
| `go/cmd/laya-gotest` | the pool and allocators behave |
| `npm run lint:struct` | the frontend structure rules hold |

The two parity tests are the load-bearing ones: a tokenizer that is off by one
id produces a confidently wrong answer, so it is checked against the reference
rather than against itself.

`TestKernelsAgree` is the load-bearing one for the kernel switch. It is not a
bit-exactness test — the two runtimes use different kernels and different
precisions, so the logits legitimately differ in the low bits. What must hold is
that the ranking and the calibrated probabilities agree closely enough that no
answer changes, which is the property a switch has to preserve and the one a
silent regression would break. Both it and the `internal/ortbackend` tests skip
rather than fail when the bridge, the runtime or a graph is absent, so a machine
without ONNX Runtime still gets a green `go test ./...`.

The `internal/backend` and `internal/backends` tests need no kernel at all: they
cover the name table, the dtype widths, the tensor validation that keeps a bad
shape away from a native call, and the selection policy — including the two
clauses that matter most, that an explicit kernel is never substituted and that
`auto` will not fall back for a `.engine` path.
