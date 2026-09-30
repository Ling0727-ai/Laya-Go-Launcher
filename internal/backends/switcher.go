package backends

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"

	"github.com/local/laya-go-launcher/internal/backend"
)

// Switcher is a backend that can change which kernel serves requests.
//
// This is what makes kernel switching a runtime operation rather than a startup
// decision: the process builds one Switcher, hands it to the use case, and later
// swaps the kernel underneath — from a config change, a CLI flag, or a per-load
// request on the HTTP API — without the inference service learning about it.
//
// It exists for a second, more basic reason too. Loading a model means creating
// a kernel; if the caller that loads is not the caller that predicts, the model
// lands in a backend nobody reads. Holding the active kernel in one place makes
// that mistake impossible: there is exactly one object to load into and to run
// on.
//
// Swapping is atomic from a request's point of view. A caller either sees the
// old kernel or the new one, never a half-installed pair, because the active
// pointer is only replaced once the new kernel has a model resident.
type Switcher struct {
	mu      sync.RWMutex
	cfg     Config
	active  backend.Backend
	lastErr error
}

// NewSwitcher builds an empty switcher with a default kernel selection.
func NewSwitcher(cfg Config) *Switcher {
	return &Switcher{cfg: cfg}
}

// Config returns the switcher's current default selection.
func (s *Switcher) Config() Config {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.cfg
}

// Active returns the kernel currently serving requests, or nil when nothing is
// loaded.
//
// The concrete type is exposed so the bench and probe commands, and the model
// picker, can reach kernel-specific metadata. Everything on the request path
// should go through the Backend methods instead.
func (s *Switcher) Active() backend.Backend {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.active
}

// Kind reports the active kernel, or the configured one when nothing is loaded.
func (s *Switcher) Kind() backend.Kind {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.active != nil {
		return s.active.Kind()
	}
	kind, err := backend.ParseKind(string(s.cfg.Kind))
	if err != nil {
		return backend.KindAuto
	}
	return kind
}

// Load loads a model using the switcher's own configuration.
func (s *Switcher) Load(ctx context.Context, opts backend.Options) (backend.Info, error) {
	info, _, err := s.LoadWith(ctx, s.Config(), opts)
	return info, err
}

// LoadWith loads a model into the requested kernel, replacing whatever is
// resident, and reports a note when auto fell back.
//
// The kernel is chosen from the model's own format before anything is opened:
// a plan is only readable by TensorRT and a graph only by ONNX Runtime, so a
// load never pays for a guaranteed-failure probe of the other kernel first.
//
// Two cases are handled differently on purpose:
//
//   - Same kernel: the existing backend is reused, so its own Load runs and the
//     previous model is released before the new one is created. That is what
//     keeps peak VRAM at one model for the TensorRT path.
//   - Different kernel: a new backend is built and loaded first, then installed,
//     and only then is the old one released. Peak memory is briefly two models,
//     but no request can observe a backend with nothing loaded, which matters
//     more: a swap that briefly breaks inference is worse than a moment of extra
//     memory. When that extra model does not fit, the resident one is released
//     and the load is retried once — see loadSwapping.
//
// cfg.Kind is the process default and is persisted. A per-load override travels
// in opts.Kind, which is what keeps one request's kernel choice from silently
// becoming every later request's default.
func (s *Switcher) LoadWith(ctx context.Context, cfg Config, opts backend.Options) (backend.Info, string, error) {
	if err := ctx.Err(); err != nil {
		return backend.Info{}, "", err
	}

	want, err := backend.ParseKind(string(cfg.Kind))
	if err != nil {
		return backend.Info{}, "", err
	}
	// A per-load override wins for this load only. It is deliberately not folded
	// into cfg: a request that names a kernel is describing that request, not
	// reconfiguring the process. Folding it in is what used to leave the process
	// pinned to onnx after one such request, so a later plain .engine load was
	// refused by the onnx kernel.
	if opts.Kind != "" {
		kind, err := backend.ParseKind(string(opts.Kind))
		if err != nil {
			return backend.Info{}, "", err
		}
		want = kind
	}
	// Which kernel this request will actually end up on. For auto that depends on
	// the file: a plan can only be read by TensorRT and a graph only by ONNX
	// Runtime, so the format decides.
	concrete := concreteKind(want, opts.Path)

	s.mu.Lock()
	current := s.active
	s.mu.Unlock()

	// Reusing the resident kernel keeps a same-kernel reload on the
	// memory-efficient path: its own Load releases the previous model before
	// creating the new one, so peak memory is one model rather than two.
	//
	// That efficiency has a cost this branch has to handle: a reload that fails
	// has already released the model it was replacing. The backend is left
	// loaded-but-empty, so the switcher must stop pointing at it — otherwise a
	// later load would reuse a kernel whose model is gone, and a caller would see
	// "loaded" for a backend with nothing resident.
	if current != nil && current.Kind() == concrete {
		info, err := current.Load(ctx, withProvider(cfg, opts))
		if err != nil {
			s.mu.Lock()
			s.lastErr = err
			if !current.Loaded() {
				// The failed reload dropped the previous model; drop the backend
				// too, so the next load starts from a clean kernel.
				s.active = nil
			}
			s.mu.Unlock()
			if !current.Loaded() {
				current.Close()
			}
			return backend.Info{}, "", err
		}
		s.mu.Lock()
		s.cfg = cfg
		s.lastErr = nil
		s.mu.Unlock()
		return info, "", nil
	}

	// A different kernel: resolve which one to use, then install it.
	be, info, note, err := s.loadSwapping(ctx, cfg, opts, current)
	if err != nil {
		s.mu.Lock()
		s.lastErr = err
		s.mu.Unlock()
		return backend.Info{}, "", err
	}

	s.mu.Lock()
	previous := s.active
	s.active = be
	s.cfg = cfg
	s.lastErr = nil
	s.mu.Unlock()

	// Release the old kernel after the new one is installed, so no request can
	// observe a switcher with nothing loaded.
	if previous != nil {
		previous.Close()
	}
	return info, note, nil
}

// loadSwapping builds the requested kernel, and when the first attempt runs out
// of memory with a model still resident, releases that model and tries once
// more.
//
// This is the case a plain "load the new kernel before releasing the old one"
// rule cannot serve. A TensorRT plan holds its activation arenas for as long as
// it is loaded — several gigabytes for a large profile — so on a card that is
// already full, building an ONNX Runtime CUDA session on top of it fails, and
// the switch appears impossible. It is not: the memory the new kernel needs is
// exactly the memory the old one is holding. Releasing first makes the switch
// succeed.
//
// The release is deliberately narrow. It happens only when the failure is an
// allocation failure and only when something is actually resident; a corrupt
// file, a wrong format or a missing runtime is reported without disturbing a
// working model. Retrying is safe because the failed load released everything it
// had already built.
func (s *Switcher) loadSwapping(ctx context.Context, cfg Config, opts backend.Options, current backend.Backend) (backend.Backend, backend.Info, string, error) {
	be, info, note, err := autoWith(ctx, cfg, opts)
	if err == nil {
		return be, info, note, nil
	}
	if current == nil || !backend.IsOutOfMemory(err) {
		return nil, backend.Info{}, "", err
	}

	// Drop the resident model to make room, then retry. The switcher is briefly
	// empty here, which is the trade this makes: a switch that fails outright
	// leaves the caller with nothing usable either, so preferring the working
	// new kernel is strictly better than refusing.
	s.mu.Lock()
	if s.active == current {
		s.active = nil
	}
	s.mu.Unlock()
	current.Close()

	retryBackend, retryInfo, retryNote, retryErr := autoWith(ctx, cfg, opts)
	if retryErr != nil {
		return nil, backend.Info{}, "", fmt.Errorf(
			"%w (after releasing the resident model: %v)", retryErr, err)
	}
	return retryBackend, retryInfo, retryNote, nil
}

// concreteKind resolves a kernel selection to the kernel that will actually
// serve a given model file.
//
// auto is a policy rather than a kernel: it means "TensorRT, falling back to
// ONNX Runtime". The file decides which is even possible, because the two
// formats are not interchangeable, so auto resolves by extension. An explicit
// choice is returned as-is — naming a kernel is a promise this code does not
// second-guess.
func concreteKind(kind backend.Kind, path string) backend.Kind {
	if kind != backend.KindAuto {
		return kind
	}
	if isONNXPath(path) {
		return backend.KindONNX
	}
	return backend.KindTensorRT
}

// Unload releases the resident model but keeps the switcher usable.
func (s *Switcher) Unload() {
	s.mu.RLock()
	active := s.active
	s.mu.RUnlock()
	if active != nil {
		active.Unload()
	}
}

// Close releases the resident kernel and marks the switcher unusable.
func (s *Switcher) Close() {
	s.mu.Lock()
	active := s.active
	s.active = nil
	s.mu.Unlock()
	if active != nil {
		active.Close()
	}
}

// Loaded reports whether a model is resident.
func (s *Switcher) Loaded() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.active != nil && s.active.Loaded()
}

// Info describes the resident model.
func (s *Switcher) Info() (backend.Info, bool) {
	s.mu.RLock()
	active := s.active
	s.mu.RUnlock()
	if active == nil {
		return backend.Info{}, false
	}
	return active.Info()
}

// InputBounds reports the range one input dimension accepts.
func (s *Switcher) InputBounds(name string, dim int) (int, int, bool) {
	s.mu.RLock()
	active := s.active
	s.mu.RUnlock()
	if active == nil {
		return 0, 0, false
	}
	return active.InputBounds(name, dim)
}

// Dir is the directory the resident model came from.
func (s *Switcher) Dir() string {
	s.mu.RLock()
	active := s.active
	s.mu.RUnlock()
	if active == nil {
		return ""
	}
	return active.Dir()
}

// Run executes one forward pass on the resident kernel.
//
// The active backend is captured once, so a concurrent swap cannot move the run
// to a different kernel mid-call: the run either completes on the kernel it
// started with, or fails because that kernel was released.
func (s *Switcher) Run(ctx context.Context, in backend.RunInput) ([]backend.Output, error) {
	s.mu.RLock()
	active := s.active
	s.mu.RUnlock()
	if active == nil {
		return nil, backend.ErrNotLoaded
	}
	return active.Run(ctx, in)
}

// LastError is the most recent load failure, or nil. The GUI reports it when a
// configured model could not be loaded at startup.
func (s *Switcher) LastError() error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.lastErr
}

// autoWith is Auto, but it also returns the backend it loaded into.
//
// Auto alone cannot serve a caller that has to keep the kernel it chose, which
// is exactly the Switcher's situation.
//
// The kernel is resolved from the model's format *before* anything is opened.
// That ordering is the point: probing TensorRT first with an .onnx path means
// reading a multi-gigabyte graph into memory only to have deserialization reject
// it, which costs seconds and a spike in peak memory for an answer the file name
// already gave. The probe was also what made a switch look impossible — the
// wasted attempt could exhaust memory before the real load was tried.
func autoWith(ctx context.Context, cfg Config, opts backend.Options) (backend.Backend, backend.Info, string, error) {
	kind, err := effectiveKind(cfg, opts)
	if err != nil {
		return nil, backend.Info{}, "", err
	}
	if kind == backend.KindONNX {
		be := newONNX(cfg)
		info, err := be.Load(ctx, withProvider(cfg, opts))
		if err != nil {
			be.Close()
			return nil, backend.Info{}, "", err
		}
		return be, info, "", nil
	}

	// TensorRT, or auto. Auto resolves by format: a graph cannot be read by
	// TensorRT, so it goes straight to ONNX Runtime rather than failing there
	// first.
	if kind == backend.KindAuto && isONNXPath(opts.Path) {
		be := newONNX(cfg)
		info, err := be.Load(ctx, withProvider(cfg, opts))
		if err != nil {
			be.Close()
			return nil, backend.Info{}, "", err
		}
		return be, info, "", nil
	}

	trt := newTensorRT()
	info, trtErr := trt.Load(ctx, opts)
	if trtErr == nil {
		return trt, info, "", nil
	}
	if kind == backend.KindTensorRT {
		// Explicitly requested: never substitute.
		trt.Close()
		return nil, backend.Info{}, "", trtErr
	}
	// auto, on a path that is not an .onnx graph. A plan is not an ONNX graph, so
	// a failure to load one is not something the other kernel can fix — unless
	// the file's extension is simply absent or wrong, which is the one case where
	// trying ONNX Runtime is worth it. That attempt is made only when TensorRT's
	// failure looks like a format mismatch rather than a missing file, a broken
	// plan or an exhausted GPU.
	if !worthONNXFallback(opts.Path, trtErr) {
		trt.Close()
		return nil, backend.Info{}, "", trtErr
	}
	ort := newONNX(cfg)
	ortInfo, ortErr := ort.Load(ctx, withProvider(cfg, opts))
	if ortErr != nil {
		trt.Close()
		// Report both failures: the second alone would hide why TensorRT, the
		// preferred kernel, was skipped.
		return nil, backend.Info{}, "", fmt.Errorf(
			"backends: tensorrt: %v; onnx: %w", trtErr, ortErr)
	}
	trt.Close()
	return ort, ortInfo, fmt.Sprintf("tensorrt unavailable (%v); using ONNX Runtime", trtErr), nil
}

// worthONNXFallback decides whether an auto load that failed on TensorRT should
// be retried on ONNX Runtime.
//
// The answer is no for a `.engine` file: it is a plan by name, and the two
// formats are not interchangeable, so a failure is about that plan — corrupt,
// built for another TensorRT version, or too large for the GPU — and retrying
// would only replace a clear diagnosis with a confusing double failure.
//
// For any other extension the file is unclassified, so a TensorRT format
// rejection means "this is probably a graph"; that is worth one attempt. Failures
// that describe something other than the format are not retried, because they
// would fail identically on the other kernel and cost another full load.
func worthONNXFallback(path string, trtErr error) bool {
	if isPlanPath(path) {
		return false
	}
	// A missing runtime, a missing file or an exhausted GPU is not a format
	// problem: ONNX Runtime cannot fix any of them.
	if errors.Is(trtErr, backend.ErrRuntimeMissing) ||
		errors.Is(trtErr, os.ErrNotExist) ||
		backend.IsOutOfMemory(trtErr) {
		return false
	}
	return true
}

// compile-time check that the switcher is usable everywhere a backend is.
var _ backend.Backend = (*Switcher)(nil)
