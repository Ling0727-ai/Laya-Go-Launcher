// Package ortbackend implements backend.Backend over ONNX Runtime.
//
// The runtime is loaded with LoadLibrary and driven through the small C ABI in
// include/layatrt_onnx.h, which is itself a port of QualityScaler-go's
// pkg/func/ortcuda: dlopen onnxruntime.dll, fetch OrtApi via OrtGetApiBase, run
// through an OrtApi session. Two properties of that approach matter here and are
// preserved deliberately:
//
//   - Nothing links against onnxruntime.lib. A machine without ONNX Runtime
//     still starts layatrt-gui, layatrt-server and layatrt-doctor and gets a
//     diagnosis, instead of dying before main() with 0xC0000279.
//   - Every buffer stays on the caller's side. The bridge is generic over named
//     tensors, so the same code serves laya's five mixed-dtype inputs and two
//     float32 outputs without a shape-specific binding.
//
// This file is the platform-independent half. The syscall plumbing lives in
// native_windows.go with a stub for other platforms, so the package builds
// everywhere and only the ONNX path is unavailable off Windows.
package ortbackend

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/local/laya-go-launcher/internal/backend"
)

// DefaultDLL is the bridge library's name, resolved next to the executable.
const DefaultDLL = "layatrt_onnx.dll"

// Backend serves models through ONNX Runtime.
type Backend struct {
	mu sync.RWMutex

	dllPath string
	session *nativeSession
	info    backend.Info

	// opts records what was loaded, so diagnostics can explain a fall back.
	opts backend.Options

	closed bool
}

// New builds an unloaded backend. dllPath is the native bridge library; empty
// means DefaultDLL, resolved beside the executable.
func New(dllPath string) *Backend {
	return &Backend{dllPath: dllPath}
}

// Kind names the kernel.
func (b *Backend) Kind() backend.Kind { return backend.KindONNX }

// Load creates an ONNX Runtime session for the graph at opts.Path.
func (b *Backend) Load(ctx context.Context, opts backend.Options) (backend.Info, error) {
	if strings.TrimSpace(opts.Path) == "" {
		return backend.Info{}, fmt.Errorf("onnx: model path is required")
	}
	if !strings.EqualFold(filepath.Ext(opts.Path), ".onnx") {
		// Not fatal on its own — a graph can be named anything — but the ONNX
		// path is only ever correct for a serialized graph, and a .engine here
		// is a user pointing the wrong kernel at a plan.
		if strings.EqualFold(filepath.Ext(opts.Path), ".engine") {
			return backend.Info{}, fmt.Errorf(
				"%w: %s is a TensorRT plan; select the tensorrt kernel for it",
				backend.ErrWrongFormat, opts.Path)
		}
	}
	// Check the file exists before handing it to ONNX Runtime. The format guard
	// above runs first because "wrong kernel" is the more useful answer for a
	// plan regardless of whether it exists.
	//
	// This is not an optimisation. ORT reports a missing model as its own
	// sentence ("Load model ... failed. File doesn't exist") with no errno, so
	// the neutral os.ErrNotExist never reaches the caller and the transport
	// cannot map it to 404 — a missing file would be reported as a 500 load
	// failure. The TensorRT path already pre-checks for exactly this reason
	// (see engine.Manager.Load), so this restores parity between the kernels:
	// both now classify a missing model the same way.
	if _, err := os.Stat(opts.Path); err != nil {
		return backend.Info{}, fmt.Errorf("onnx: %w", err)
	}

	sess, err := newNativeSession(nativeConfig{
		dllPath:         b.dllPath,
		RuntimePath:     opts.RuntimePath,
		ModelPath:       opts.Path,
		Provider:        opts.Provider,
		ProviderOptions: opts.ProviderOptions,
	})
	if err != nil {
		// Tag an allocation failure so a caller can tell "there was not enough
		// memory" from "this graph is broken": the first is fixed by releasing
		// whatever else is resident, the second is not.
		return backend.Info{}, backend.WrapAllocation(err)
	}

	if err := requireProvider(opts.Provider, sess.providerName(), sess.providerNote()); err != nil {
		destroyNativeSession(sess)
		return backend.Info{}, err
	}

	info, err := describe(sess, opts.Path, opts.Provider)
	if err != nil {
		destroyNativeSession(sess)
		return backend.Info{}, err
	}

	b.mu.Lock()
	prev := b.session
	b.session = sess
	b.info = info
	b.opts = opts
	b.closed = false
	b.mu.Unlock()

	// Release the previous session outside the lock, after the new one is
	// resident: peak memory is briefly two models, but no request can observe a
	// half-installed backend, which is the property that matters here.
	if prev != nil {
		destroyNativeSession(prev)
	}
	return info, nil
}

// Unload releases the session.
func (b *Backend) Unload() {
	b.mu.Lock()
	sess := b.session
	b.session = nil
	b.info = backend.Info{}
	b.mu.Unlock()
	if sess != nil {
		destroyNativeSession(sess)
	}
}

// Close releases the session and marks the backend unusable.
func (b *Backend) Close() {
	b.mu.Lock()
	b.closed = true
	b.mu.Unlock()
	b.Unload()
}

// Loaded reports whether a session is resident.
func (b *Backend) Loaded() bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.session != nil
}

// Info describes the resident session.
func (b *Backend) Info() (backend.Info, bool) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.session == nil {
		return backend.Info{}, false
	}
	return b.info, true
}

// InputBounds reports the range one input dimension accepts.
//
// An ONNX graph declares a fixed dimension as a number and a dynamic one as a
// symbol, which carries no maximum: the graph accepts whatever the kernel
// underneath can hold. So a symbolic dimension reports ok == false rather than
// inventing a bound, and the caller keeps its configured budget. A fixed
// dimension reports min == max, which is what makes a statically shaped export
// pad to its exact size.
func (b *Backend) InputBounds(name string, dim int) (int, int, bool) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.session == nil {
		return 0, 0, false
	}
	t, ok := backend.Find(b.info.Inputs, name)
	if !ok || dim >= len(t.Shape) {
		return 0, 0, false
	}
	if t.Shape[dim] <= 0 {
		// Dynamic: the shape is not a promise about the range.
		return 0, 0, false
	}
	return t.Shape[dim], t.Shape[dim], true
}

// Dir is the directory the graph came from.
func (b *Backend) Dir() string {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.info.Path == "" {
		return ""
	}
	return filepath.Dir(b.info.Path)
}

// Run executes one forward pass.
//
// The bridge binds the caller's buffers directly, so no copy happens on the Go
// side; outputs are written into memory this function allocates and then hands
// back. An ONNX Runtime session is not safe to run concurrently from several
// goroutines, so runs are serialised here — the same guarantee the TensorRT path
// gets from its context pool, expressed as a lock because ORT does its own
// intra-op scheduling.
func (b *Backend) Run(ctx context.Context, in backend.RunInput) ([]backend.Output, error) {
	b.mu.RLock()
	sess, info, closed := b.session, b.info, b.closed
	b.mu.RUnlock()
	if closed {
		return nil, backend.ErrClosed
	}
	if sess == nil {
		return nil, backend.ErrNotLoaded
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	for _, t := range in.Inputs {
		if err := t.Validate(); err != nil {
			return nil, err
		}
	}

	// Which outputs to compute. An empty list asks for the model's own order.
	want := in.Outputs
	if len(want) == 0 {
		want = info.OutputNames()
	}
	// Every requested output must exist, so a typo fails here rather than being
	// silently ignored by the runtime.
	for _, name := range want {
		if _, ok := backend.Find(info.Outputs, name); !ok {
			return nil, fmt.Errorf("%w: model has no output %q", backend.ErrIncompatible, name)
		}
	}

	if err := sess.run(ctx, in.Inputs, want); err != nil {
		return nil, err
	}

	// The run has happened, so the shapes are concrete now even when the graph
	// declared them symbolically. Size each buffer from the result itself.
	results, err := sess.results()
	if err != nil {
		return nil, err
	}
	outputs := make([]backend.Output, 0, len(results))
	for _, r := range results {
		buf := make([]byte, r.elements*r.dtype.Size())
		if err := sess.copyResult(r.index, buf); err != nil {
			return nil, err
		}
		outputs = append(outputs, backend.Output{
			Name:  r.name,
			DType: r.dtype,
			Shape: r.shape,
			Data:  buf,
		})
	}
	return outputs, nil
}

// resultInfo is one entry of a completed run, in neutral terms.
//
// It is declared here rather than beside the Windows session because both the
// real session and the non-Windows stub name it, and keeping it in the
// platform-specific file left the stub unable to compile — which contradicted
// the promise that this package builds everywhere.
type resultInfo struct {
	index    int
	name     string
	dtype    backend.DType
	shape    []int
	elements int
}

// describe turns the session's own IO declaration into the neutral description.
func describe(sess *nativeSession, path, provider string) (backend.Info, error) {
	inputs, err := sess.inputs()
	if err != nil {
		return backend.Info{}, err
	}
	outputs, err := sess.outputs()
	if err != nil {
		return backend.Info{}, err
	}

	info := backend.Info{
		Path:    path,
		Backend: "ONNX Runtime",
		Runtime: sess.runtimeVersion(),
		Device:  sess.providerName(),
		Inputs:  inputs,
		Outputs: outputs,
	}
	// The ONNX path has no context pool; report the concurrency it really
	// allows (one run at a time) rather than a VRAM-derived guess.
	info.Contexts = 1
	if note := sess.providerNote(); note != "" {
		// Surface a provider fall back in the device field so /health and the
		// GUI show why this is running on the CPU.
		info.Device = fmt.Sprintf("%s (%s)", sess.providerName(), note)
	}
	_ = provider
	return info, nil
}

func requireProvider(requested, actual, note string) error {
	requested = strings.ToLower(strings.TrimSpace(requested))
	actual = strings.ToLower(strings.TrimSpace(actual))
	if requested == "" {
		return nil // legacy automatic CUDA-to-CPU fallback
	}
	if requested == "dml" {
		requested = "directml"
	}
	if actual == "dml" {
		actual = "directml"
	}
	switch requested {
	case "cpu", "cuda", "directml":
	default:
		return fmt.Errorf("onnx: unknown execution provider %q", requested)
	}
	if actual != requested {
		if note != "" {
			return fmt.Errorf("onnx: requested %s provider, got %s: %s", requested, actual, note)
		}
		return fmt.Errorf("onnx: requested %s provider, got %s", requested, actual)
	}
	return nil
}

// compile-time check that the ONNX path satisfies the contract.
var _ backend.Backend = (*Backend)(nil)
