// Package backend defines the execution contract every inference kernel
// implements, and the policy that picks one.
//
// The predict use case needs four things from a kernel: load a model, describe
// its IO, run one forward pass over named tensors, and release it. Nothing in
// this package knows about TensorRT plans, ONNX graphs, CUDA streams or native
// handles, so internal/inference can be written against it and stay independent
// of which kernel is resident.
//
// This is the seam that makes a kernel switch possible: internal/trtbackend
// implements it over the TensorRT plan manager, internal/ortbackend over the
// ONNX Runtime native session. A caller selects one and the use case above never
// learns which it got.
package backend

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// Errors a caller can classify without matching on message text.
var (
	// ErrNotLoaded means an operation needed a resident model and none was.
	ErrNotLoaded = errors.New("backend: no model is loaded")
	// ErrClosed means the backend was released and cannot be reused.
	ErrClosed = errors.New("backend: backend is closed")
	// ErrUnsupported means the kernel cannot express the requested operation.
	ErrUnsupported = errors.New("backend: operation is not supported")
	// ErrRuntimeMissing means the kernel's own runtime library was not found.
	ErrRuntimeMissing = errors.New("backend: runtime library is missing")
	// ErrIncompatible means the model loaded but does not expose the tensors the
	// caller needs.
	ErrIncompatible = errors.New("backend: model is incompatible")
	// ErrWrongFormat means the model file belongs to the other kernel: a plan
	// handed to ONNX Runtime, or a graph handed to TensorRT. It is separate from
	// ErrIncompatible because the remedy is different — pick the other kernel,
	// rather than replace the file.
	ErrWrongFormat = errors.New("backend: model file belongs to the other kernel")
	// ErrOutOfMemory means the kernel could not allocate what the model needs.
	// It is its own sentinel because it is the one load failure that a retry can
	// fix: releasing the resident model first frees exactly the resource that was
	// missing, while a corrupt file or a wrong format would fail again.
	ErrOutOfMemory = errors.New("backend: not enough memory to load the model")
)

// classifyAllocation recognises a native allocation failure in a kernel's error
// text.
//
// The two runtimes report exhaustion through strings rather than a status code
// the C ABI here can forward — CUDA says "out of memory", ORT says it could not
// allocate, and the C++ allocation path says "bad_alloc". Matching on the text is
// the only signal available at this boundary, so it lives in one place instead of
// being re-guessed by each caller.
func classifyAllocation(message string) bool {
	m := strings.ToLower(message)
	for _, marker := range []string{
		"out of memory",
		"outofmemory",
		"cuda_error_out_of_memory",
		"failed to allocate",
		"could not allocate",
		"cannot allocate",
		"unable to allocate",
		"bad_alloc",
		"insufficient memory",
		"not enough memory",
	} {
		if strings.Contains(m, marker) {
			return true
		}
	}
	return false
}

// IsOutOfMemory reports whether err is, or was caused by, an allocation failure.
//
// It accepts both the sentinel and a raw message, because a kernel that has not
// wrapped its error yet still has to be classifiable.
func IsOutOfMemory(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, ErrOutOfMemory) {
		return true
	}
	return classifyAllocation(err.Error())
}

// WrapAllocation tags a native error with ErrOutOfMemory when its text says the
// failure was an allocation, and returns it unchanged otherwise.
func WrapAllocation(err error) error {
	if err == nil || errors.Is(err, ErrOutOfMemory) {
		return err
	}
	if !classifyAllocation(err.Error()) {
		return err
	}
	return fmt.Errorf("%w: %v", ErrOutOfMemory, err)
}

// Kind names an execution kernel.
type Kind string

const (
	// KindAuto picks TensorRT when it is usable and ONNX Runtime otherwise.
	KindAuto Kind = "auto"
	// KindTensorRT is the TensorRT plan path (the historical default).
	KindTensorRT Kind = "tensorrt"
	// KindONNX is the ONNX Runtime path.
	KindONNX Kind = "onnx"
)

// ParseKind reads a backend name from configuration or a command line.
//
// An empty value is auto, so a config file written before this option existed
// keeps working and keeps meaning what it meant: use the TensorRT path.
func ParseKind(s string) (Kind, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", string(KindAuto):
		return KindAuto, nil
	case string(KindTensorRT), "trt", "plan", "engine":
		return KindTensorRT, nil
	case string(KindONNX), "onnxruntime", "ort":
		return KindONNX, nil
	default:
		return "", fmt.Errorf("backend: unknown kernel %q (want auto, tensorrt or onnx)", s)
	}
}

// String names the kernel.
func (k Kind) String() string { return string(k) }

// Options describes what to load and how.
type Options struct {
	// Path is the model file: a .engine plan for TensorRT, a .onnx graph for
	// ONNX Runtime.
	Path string
	// Kind overrides the kernel for this load only. Empty means "use the
	// caller's configured default".
	//
	// It lives here rather than in the configuration because a per-load choice
	// is a property of the request, not of the process. Treating the two as one
	// is what let a single request that named a kernel reconfigure every later
	// load.
	Kind Kind
	// Contexts is how many execution contexts to pre-allocate. Zero or negative
	// picks a count from free VRAM. Only the TensorRT path uses it; the ONNX
	// path reports the concurrency it actually allows.
	Contexts int
	// Provider selects the ONNX Runtime execution provider, e.g. "cuda", "cpu",
	// "directml". Empty means the backend's own default. Ignored by TensorRT.
	Provider string
	// RuntimePath overrides discovery of the kernel's runtime library
	// (onnxruntime.dll for the ONNX path). Empty means discover it.
	RuntimePath string
	// ProviderOptions are passed through to the execution provider.
	ProviderOptions map[string]string
}

// Info describes a resident model in kernel-neutral terms.
type Info struct {
	// Path is the model file that was loaded.
	Path string
	// Backend is the kernel serving it, for display and diagnostics.
	Backend string
	// Contexts is the concurrency the kernel pre-allocated.
	Contexts int
	// ActivationMemoryMB is the kernel's worst-case activation memory, when it
	// reports one. Zero means "not reported", not "no memory used".
	ActivationMemoryMB float64
	Inputs             []TensorInfo
	Outputs            []TensorInfo
	// Runtime is the kernel runtime's own version string, when known.
	Runtime string
	// Device names the compute device, when known.
	Device string
}

// InputNames lists the model's input tensor names in declaration order.
func (i Info) InputNames() []string { return names(i.Inputs) }

// OutputNames lists the model's output tensor names in declaration order.
func (i Info) OutputNames() []string { return names(i.Outputs) }

func names(ts []TensorInfo) []string {
	out := make([]string, 0, len(ts))
	for _, t := range ts {
		out = append(out, t.Name)
	}
	return out
}

// Find returns the named tensor's description.
func Find(ts []TensorInfo, name string) (TensorInfo, bool) {
	for _, t := range ts {
		if t.Name == name {
			return t, true
		}
	}
	return TensorInfo{}, false
}

// Backend is one execution kernel with at most one model resident.
//
// Implementations must be safe for concurrent use: the HTTP handlers and the
// GUI bindings share one instance, and Predict may be called from several
// goroutines at once. Load and Unload may be called while runs are in flight;
// an implementation must drain rather than free memory another goroutine is
// still reading.
type Backend interface {
	// Kind names the kernel.
	Kind() Kind
	// Load makes a model resident, replacing any current one.
	Load(ctx context.Context, opts Options) (Info, error)
	// Unload releases the model. Idempotent.
	Unload()
	// Close releases the model and marks the backend unusable. Idempotent.
	Close()
	// Loaded reports whether a model is resident.
	Loaded() bool
	// Info describes the resident model, or reports false when none is.
	Info() (Info, bool)
	// InputBounds returns the (min, max) range of one input dimension.
	//
	// A fixed dimension reports min == max. A dynamic dimension reports the
	// range the kernel actually accepts; when the kernel cannot express a bound
	// (an ONNX graph with a symbolic dimension has no declared maximum) it
	// reports ok == false rather than inventing one, and the caller keeps its
	// configured budget.
	InputBounds(name string, dim int) (min, max int, ok bool)
	// Dir is the directory the model came from, where its checkpoint
	// configuration is expected to sit. Empty when nothing is loaded.
	Dir() string
	// Run executes one forward pass over named tensors and returns the
	// requested outputs.
	//
	// Outputs names the tensors to read back. An empty list asks for every
	// output the model declares. Cancelling ctx abandons the wait for a
	// context/slot; an already-started native call finishes before returning.
	Run(ctx context.Context, in RunInput) ([]Output, error)
}

// RunInput is one forward pass.
type RunInput struct {
	// Inputs are the named input tensors, in the model's declared order.
	Inputs []Tensor
	// Outputs names the tensors to read back; empty means all of them.
	Outputs []string
}

// Input returns the named input tensor.
func (in RunInput) Input(name string) (Tensor, bool) {
	for _, t := range in.Inputs {
		if t.Name == name {
			return t, true
		}
	}
	return Tensor{}, false
}
