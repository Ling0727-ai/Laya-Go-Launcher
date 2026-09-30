package backends

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/local/laya-go-launcher/internal/backend"
)

// TestAutoResolvesONNXBeforeProbingTensorRT is the load-bearing test for the
// switch.
//
// The defect it pins down: auto used to try TensorRT first regardless of the
// file, so loading an .onnx read the whole graph into memory only to have
// deserialization reject it, and then fell back. That wasted seconds and a
// memory spike, and on a machine where the resident model already held the GPU it
// could exhaust memory before the real load was ever attempted — which is what
// made switching from TensorRT to ONNX look impossible.
//
// The observable property is that an .onnx path never reaches TensorRT. It is
// checked without a GPU or a runtime by asserting on the error: a missing file
// produces a TensorRT-flavoured message if the wrong kernel was consulted, and an
// ONNX-flavoured one if the format was resolved correctly.
func TestAutoResolvesONNXBeforeProbingTensorRT(t *testing.T) {
	// A .onnx path that does not exist. If auto resolved by format, the ONNX
	// kernel is asked and reports its own failure.
	_, _, note, err := Auto(context.Background(), Config{Kind: backend.KindAuto},
		backend.Options{Path: "definitely-missing.onnx"})
	if err == nil {
		t.Fatal("Auto on a missing .onnx returned no error")
	}
	if note != "" {
		t.Errorf("resolving an .onnx by format should not be a fallback, got note %q", note)
	}
	if !errors.Is(err, os.ErrNotExist) {
		t.Errorf("error = %v, want a not-exist error from the ONNX path", err)
	}
	// The ONNX path names itself; the TensorRT path would not.
	if !contains(err.Error(), "onnx") {
		t.Errorf("error %q does not look like it came from the ONNX kernel", err)
	}
}

// TestAutoPlanPathNeverTriesONNX pins the other half of the resolution: a
// .engine plan is TensorRT's, and a failure there is not something ONNX Runtime
// can fix.
func TestAutoPlanPathNeverTriesONNX(t *testing.T) {
	_, _, note, err := Auto(context.Background(), Config{Kind: backend.KindAuto},
		backend.Options{Path: "definitely-missing.engine"})
	if err == nil {
		t.Fatal("Auto on a missing .engine returned no error")
	}
	if note != "" {
		t.Errorf("a plan path produced a fallback note %q", note)
	}
	if contains(err.Error(), "onnx") {
		t.Errorf("error %q mentions the ONNX kernel for a .engine path", err)
	}
}

// TestWorthONNXFallback covers the decision directly, including the cases that
// must not be retried.
func TestWorthONNXFallback(t *testing.T) {
	formatErr := errors.New("tensorrt: load x: Failed to deserialize engine")
	cases := []struct {
		name string
		path string
		err  error
		want bool
	}{
		{"a plan is never retried", "model.engine", formatErr, false},
		{"an unclassified file is retried", "model.bin", formatErr, true},
		{"an extensionless file is retried", "model", formatErr, true},
		{"a missing runtime is not retried", "model.bin",
			backend.ErrRuntimeMissing, false},
		{"a missing file is not retried", "model.bin", os.ErrNotExist, false},
		{"an allocation failure is not retried", "model.bin",
			backend.ErrOutOfMemory, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := worthONNXFallback(c.path, c.err); got != c.want {
				t.Errorf("worthONNXFallback(%q, %v) = %v, want %v", c.path, c.err, got, c.want)
			}
		})
	}
}

// TestPerLoadKindDoesNotLeakIntoTheDefault is the regression test for the
// configuration leak.
//
// A request that named a kernel used to have that name written into the
// switcher's configuration, so one "load this .onnx as onnx" left the whole
// process pinned to onnx: every later load, including a plain .engine, was then
// sent to the ONNX kernel and refused. The per-load choice has to stay per-load.
func TestPerLoadKindDoesNotLeakIntoTheDefault(t *testing.T) {
	s := NewSwitcher(Config{Kind: backend.KindAuto})
	defer s.Close()

	before := s.Config().Kind

	// This load fails (no such file), but the choice must not be recorded
	// regardless of the outcome.
	_, _, err := s.LoadWith(context.Background(), s.Config(), backend.Options{
		Path: "definitely-missing.onnx",
		Kind: backend.KindONNX,
	})
	if err == nil {
		t.Fatal("loading a missing file returned no error")
	}

	if after := s.Config().Kind; after != before {
		t.Errorf("a per-load Kind changed the default: %q -> %q", before, after)
	}
}

// TestConcreteKindResolvesByFormat checks the resolution the switch rests on.
func TestConcreteKindResolvesByFormat(t *testing.T) {
	cases := []struct {
		name string
		kind backend.Kind
		path string
		want backend.Kind
	}{
		{"auto takes an .onnx to onnx", backend.KindAuto, "m.onnx", backend.KindONNX},
		{"auto takes an .engine to tensorrt", backend.KindAuto, "m.engine", backend.KindTensorRT},
		{"auto takes an unknown extension to tensorrt", backend.KindAuto, "m", backend.KindTensorRT},
		{"an explicit kernel is not second-guessed", backend.KindTensorRT, "m.onnx", backend.KindTensorRT},
		{"an explicit kernel is not second-guessed (onnx)", backend.KindONNX, "m.engine", backend.KindONNX},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := concreteKind(c.kind, c.path); got != c.want {
				t.Errorf("concreteKind(%q, %q) = %q, want %q", c.kind, c.path, got, c.want)
			}
		})
	}
}

// TestNewForPicksTheKernelForTheFile checks the constructor a caller uses when it
// already knows the model, so it does not build a TensorRT backend for a graph it
// can never load there.
func TestNewForPicksTheKernelForTheFile(t *testing.T) {
	cfg := Config{Kind: backend.KindAuto}

	be, err := NewFor(cfg, backend.Options{Path: "model.onnx"})
	if err != nil {
		t.Fatalf("NewFor(.onnx): %v", err)
	}
	defer be.Close()
	if be.Kind() != backend.KindONNX {
		t.Errorf("NewFor(.onnx).Kind() = %q, want onnx", be.Kind())
	}

	be2, err := NewFor(cfg, backend.Options{Path: "model.engine"})
	if err != nil {
		t.Fatalf("NewFor(.engine): %v", err)
	}
	defer be2.Close()
	if be2.Kind() != backend.KindTensorRT {
		t.Errorf("NewFor(.engine).Kind() = %q, want tensorrt", be2.Kind())
	}
}

// TestNewForHonoursPerLoadOverride checks the override reaches the constructor
// too, so an explicit choice is consistent between NewFor and LoadWith.
func TestNewForHonoursPerLoadOverride(t *testing.T) {
	cfg := Config{Kind: backend.KindAuto}
	be, err := NewFor(cfg, backend.Options{Path: "model.onnx", Kind: backend.KindTensorRT})
	if err != nil {
		t.Fatalf("NewFor: %v", err)
	}
	defer be.Close()
	if be.Kind() != backend.KindTensorRT {
		t.Errorf("Kind() = %q, want tensorrt for an explicit override", be.Kind())
	}
}

// TestWrongFormatIsItsOwnSentinel checks the two kernels agree on how a
// mismatched file is reported, because the transport turns that sentinel into a
// wire code and the message has to be actionable.
func TestWrongFormatIsItsOwnSentinel(t *testing.T) {
	dir := t.TempDir()

	// A real .onnx handed to the ONNX kernel's own guard is not a mismatch, so
	// exercise the plan-to-onnx direction, which is checked before any native
	// call and needs no runtime.
	plan := filepath.Join(dir, "model.engine")
	if err := os.WriteFile(plan, []byte("not a plan"), 0o600); err != nil {
		t.Fatal(err)
	}

	be, err := NewFor(Config{Kind: backend.KindONNX}, backend.Options{Path: plan})
	if err != nil {
		t.Fatalf("NewFor: %v", err)
	}
	defer be.Close()

	_, err = be.Load(context.Background(), backend.Options{Path: plan})
	if err == nil {
		t.Fatal("loading a .engine through the onnx kernel returned no error")
	}
	if !errors.Is(err, backend.ErrWrongFormat) {
		t.Errorf("error = %v, want ErrWrongFormat", err)
	}
}

// TestOutOfMemoryClassification covers the classifier the switch retry depends
// on: it has to recognise the shapes the native runtimes actually report, and it
// must not mistake an ordinary failure for one.
func TestOutOfMemoryClassification(t *testing.T) {
	oom := []string{
		"CUDA error: out of memory",
		"cuda_error_out_of_memory",
		"onnx: create session: Failed to allocate memory for the model",
		"std::bad_alloc",
		"insufficient memory available",
		"not enough memory to load",
	}
	for _, msg := range oom {
		if !backend.IsOutOfMemory(errors.New(msg)) {
			t.Errorf("IsOutOfMemory(%q) = false, want true", msg)
		}
	}

	notOOM := []string{
		"Failed to deserialize engine",
		"no such file or directory",
		"ONNX Runtime C API version is incompatible",
		"invalid model: bad protobuf",
	}
	for _, msg := range notOOM {
		if backend.IsOutOfMemory(errors.New(msg)) {
			t.Errorf("IsOutOfMemory(%q) = true, want false", msg)
		}
	}

	if backend.IsOutOfMemory(nil) {
		t.Error("IsOutOfMemory(nil) = true, want false")
	}
	if !backend.IsOutOfMemory(backend.ErrOutOfMemory) {
		t.Error("the sentinel itself was not classified")
	}
}

// TestWrapAllocationPreservesTheOriginal checks the wrap adds a classification
// without destroying the message a user needs to read.
func TestWrapAllocationPreservesTheOriginal(t *testing.T) {
	base := errors.New("onnx: create session: out of memory")
	wrapped := backend.WrapAllocation(base)
	if !errors.Is(wrapped, backend.ErrOutOfMemory) {
		t.Error("wrapped error is not classified as out of memory")
	}
	if !contains(wrapped.Error(), "create session") {
		t.Errorf("wrapped error lost the original text: %q", wrapped)
	}

	// An unrelated error is returned unchanged, not wrapped in a false claim.
	other := errors.New("failed to deserialize engine")
	if got := backend.WrapAllocation(other); got != other {
		t.Errorf("WrapAllocation changed an unrelated error: %v", got)
	}
}

func contains(haystack, needle string) bool {
	return len(needle) == 0 || len(haystack) >= len(needle) &&
		indexOf(haystack, needle) >= 0
}

func indexOf(haystack, needle string) int {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return i
		}
	}
	return -1
}
