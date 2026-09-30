// Package backends builds and selects execution kernels.
//
// It is the one place that knows every concrete backend exists, which keeps the
// dependency arrow pointing the right way: internal/inference depends only on
// internal/backend's contract, and only this package imports the TensorRT and
// ONNX Runtime adapters.
//
// Selection policy is deliberately boring:
//
//   - The default is auto, which means "the TensorRT path" — the historical
//     behaviour, so an existing config file keeps meaning what it meant.
//   - A kernel named explicitly is used as named. If it cannot start, that is an
//     error, not a silent substitution: a request to run on the GPU that quietly
//     lands on the CPU is worse than a failure, because it looks like it worked.
//   - auto may fall back, and says so in the returned Info.
package backends

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/local/laya-go-launcher/internal/backend"
	"github.com/local/laya-go-launcher/internal/ortbackend"
	"github.com/local/laya-go-launcher/internal/trtbackend"
)

// Config is what the process knows before a model is chosen.
type Config struct {
	// Kind is the requested kernel. Empty means auto.
	Kind backend.Kind
	// Provider selects the ONNX Runtime execution provider ("cuda", "cpu",
	// "directml"). Empty means the backend's default.
	Provider string
	// RuntimePath overrides ONNX Runtime discovery.
	RuntimePath string
	// DLLPath overrides the native ONNX bridge library's location.
	DLLPath string
	// ProviderOptions are passed through to the execution provider.
	ProviderOptions map[string]string
}

// New builds the requested kernel without loading a model.
//
// A named kernel is returned even when its runtime is unavailable, so that the
// failure is reported at load time with a real reason instead of at
// construction. auto resolves to TensorRT; use Auto to get the fallback
// behaviour.
func New(cfg Config) (backend.Backend, error) {
	return NewFor(cfg, backend.Options{})
}

// NewFor builds the kernel a load of opts.Path would use, without loading a
// model.
//
// Unlike New, it resolves auto against the file: an .onnx path yields the ONNX
// kernel, so a caller that wants the right object for a known model does not have
// to guess and does not get a TensorRT backend it can never use. A per-load Kind
// in opts wins over the configuration, matching LoadWith.
func NewFor(cfg Config, opts backend.Options) (backend.Backend, error) {
	kind, err := effectiveKind(cfg, opts)
	if err != nil {
		return nil, err
	}
	switch concreteKind(kind, opts.Path) {
	case backend.KindONNX:
		return newONNX(cfg), nil
	case backend.KindTensorRT:
		return newTensorRT(), nil
	default:
		return nil, fmt.Errorf("backends: unhandled kernel %q", kind)
	}
}

// newONNX builds the ONNX Runtime kernel. Kept as a named helper so the
// switcher can construct one without duplicating the configuration plumbing.
func newONNX(cfg Config) backend.Backend { return ortbackend.New(cfg.DLLPath) }

// newTensorRT builds the TensorRT kernel.
func newTensorRT() backend.Backend { return trtbackend.New() }

// Auto picks a kernel and loads a model into it, falling back from TensorRT to
// ONNX Runtime when TensorRT cannot serve the model.
//
// The fallback only applies to auto. It is attempted when the model is an ONNX
// graph (a plan cannot be opened by ONNX Runtime, so trying would be pointless).
//
// Returns the loaded backend, its description, and a note when a fallback
// happened so the caller can report it rather than hide it.
//
// Callers that must keep the kernel they chose should use Switcher, which is
// built on the same policy but owns the result.
func Auto(ctx context.Context, cfg Config, opts backend.Options) (backend.Backend, backend.Info, string, error) {
	return autoWith(ctx, cfg, opts)
}

func withProvider(cfg Config, opts backend.Options) backend.Options {
	if opts.Provider == "" {
		opts.Provider = cfg.Provider
	}
	if opts.RuntimePath == "" {
		opts.RuntimePath = cfg.RuntimePath
	}
	if len(opts.ProviderOptions) == 0 {
		opts.ProviderOptions = cfg.ProviderOptions
	}
	return opts
}

// effectiveKind resolves the kernel a load should use, honouring a per-load
// override over the configured default.
func effectiveKind(cfg Config, opts backend.Options) (backend.Kind, error) {
	if opts.Kind != "" {
		return backend.ParseKind(string(opts.Kind))
	}
	return backend.ParseKind(string(cfg.Kind))
}

func isONNXPath(path string) bool {
	return strings.EqualFold(ext(path), ".onnx")
}

// isPlanPath reports whether the file is a TensorRT plan by extension. A file
// with no recognised extension is treated as a plan, because that is the
// historical default and the only other format this program reads.
func isPlanPath(path string) bool {
	return strings.EqualFold(ext(path), ".engine")
}

func ext(path string) string {
	if i := strings.LastIndexByte(path, '.'); i >= 0 {
		return path[i:]
	}
	return ""
}

// Diagnostics describes which kernels this build and this machine can offer,
// without loading a model. It is what /health and the doctor command report.
type Diagnostics struct {
	// Requested is the configured kernel name.
	Requested string `json:"requested"`
	// TensorRT and ONNX report per-kernel availability.
	TensorRT KernelStatus `json:"tensorrt"`
	ONNX     KernelStatus `json:"onnx"`
	// Selected is the kernel a load would use right now.
	Selected string `json:"selected"`
	// Note explains a selection that is not what was requested.
	Note string `json:"note,omitempty"`
}

// KernelStatus is one kernel's availability.
type KernelStatus struct {
	// Name is the display name.
	Name string `json:"name"`
	// Available is true when the kernel's runtime can be brought up.
	Available bool `json:"available"`
	// Version is the runtime version, when known.
	Version string `json:"version,omitempty"`
	// Error explains an unavailable kernel.
	Error string `json:"error,omitempty"`
}

// Probe reports kernel availability without loading a model.
func Probe(cfg Config) Diagnostics {
	kind, err := backend.ParseKind(string(cfg.Kind))
	if err != nil {
		kind = backend.KindAuto
	}
	out := Diagnostics{Requested: string(kind)}

	out.TensorRT = probeTensorRT()
	out.ONNX = probeONNX(cfg)

	switch kind {
	case backend.KindONNX:
		out.Selected = string(backend.KindONNX)
	case backend.KindTensorRT:
		out.Selected = string(backend.KindTensorRT)
	default:
		// auto is a policy, not a kernel, and which kernel it lands on depends on
		// the model file: a plan can only be read by TensorRT, a graph only by
		// ONNX Runtime. Without a file there is nothing to resolve against, so
		// report the preference and say so rather than implying a commitment.
		if out.TensorRT.Available {
			out.Selected = string(backend.KindTensorRT)
			out.Note = "auto prefers TensorRT; an .onnx model is served by ONNX Runtime"
		} else if out.ONNX.Available {
			out.Selected = string(backend.KindONNX)
			out.Note = "TensorRT is unavailable; auto will use ONNX Runtime"
		} else {
			out.Selected = string(backend.KindTensorRT)
			out.Note = "no kernel is available"
		}
	}
	return out
}

// probeTensorRT checks the TensorRT runtime.
//
// kernel.Initialize is cgo-linked, so on a machine missing the DLL this process
// could not have started at all; reaching here means the library resolved, and
// the remaining question is whether the runtime comes up (driver, GPU).
func probeTensorRT() KernelStatus {
	status := KernelStatus{Name: "TensorRT"}
	// Guarded because Initialize is a native call that touches CUDA.
	func() {
		defer func() {
			if rec := recover(); rec != nil {
				status.Error = fmt.Sprintf("panic: %v", rec)
			}
		}()
		if err := trtbackend.Initialize(); err != nil {
			status.Error = err.Error()
			return
		}
		status.Available = true
		status.Version = trtbackend.RuntimeVersion()
	}()
	return status
}

// probeONNX checks whether the ONNX bridge and a runtime library are present.
//
// The configured provider is passed through, because which runtime is acceptable
// depends on it: a CUDA request must not be satisfied by a DirectML build, and a
// diagnostics panel that reported "available" for one would be lying.
func probeONNX(cfg Config) KernelStatus {
	status := KernelStatus{Name: "ONNX Runtime"}
	path, err := ortbackend.ProbeFor(cfg.DLLPath, cfg.RuntimePath, cfg.Provider)
	if err != nil {
		status.Error = err.Error()
		return status
	}
	status.Available = true
	status.Version = path
	return status
}

// IsUnavailable reports whether a load failure means "this kernel cannot run on
// this machine", as opposed to "this model is wrong".
func IsUnavailable(err error) bool {
	return errors.Is(err, backend.ErrRuntimeMissing)
}

// Inspect opens a model just far enough to read its IO tensors, then releases
// it. It is what the engine picker uses to tell a laya model from any other
// file, and it is the only reliable test: filenames say nothing, and a model
// built for something else is a perfectly valid file for its own kernel.
//
// The kernel is chosen from the file's extension, because a .engine plan can
// only be read by TensorRT and a .onnx graph only by ONNX Runtime. That is a
// property of the formats, not a preference, so it does not consult the
// configured backend.
//
// This is not free. Reading a plan's IO means deserializing it, which allocates
// its weights on the GPU, and reading a graph's IO means building a session. So
// the result is cached (see inspectionCache) and the picker calls this once per
// file rather than once per scan: a directory of five plans would otherwise
// deserialize five multi-hundred-megabyte engines every time the list is
// refreshed, while a model is already holding the GPU.
func Inspect(ctx context.Context, cfg Config, path string) (backend.Info, error) {
	return inspectCache.get(ctx, cfg, path, func() (backend.Info, error) {
		if isONNXPath(path) {
			return inspectONNX(ctx, cfg, path)
		}
		return inspectPlan(path)
	})
}

// inspectPlan reads a plan's IO contract through the TensorRT kernel.
func inspectPlan(path string) (backend.Info, error) {
	be := trtbackend.New()
	defer be.Close()
	return be.Load(context.Background(), backend.Options{Path: path, Contexts: 1})
}

// inspectONNX reads a graph's IO contract through a CPU session. Metadata
// inspection must not depend on the selected GPU provider: an unavailable CUDA
// runtime must not hide a valid graph that can be loaded with DirectML.
func inspectONNX(ctx context.Context, cfg Config, path string) (backend.Info, error) {
	be := ortbackend.New(cfg.DLLPath)
	defer be.Close()
	opts := withProvider(cfg, backend.Options{Path: path, Contexts: 1})
	opts.Provider = "cpu"
	return be.Load(ctx, opts)
}

// MissingInputs reports which of the required inputs a model does not declare.
// An empty result means the model matches the contract.
func MissingInputs(info backend.Info, required []string) []string {
	var missing []string
	for _, want := range required {
		if _, ok := backend.Find(info.Inputs, want); !ok {
			missing = append(missing, want)
		}
	}
	return missing
}
