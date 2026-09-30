package backends

import (
	"context"
	"testing"

	"github.com/local/laya-go-launcher/internal/backend"
)

// TestSwitcherStartsEmpty checks the state before any load, because that is what
// the GUI and the HTTP API observe on a fresh process.
func TestSwitcherStartsEmpty(t *testing.T) {
	s := NewSwitcher(Config{Kind: backend.KindAuto})
	defer s.Close()

	if s.Loaded() {
		t.Error("a fresh switcher reports loaded")
	}
	if _, ok := s.Info(); ok {
		t.Error("a fresh switcher reports model info")
	}
	if s.Active() != nil {
		t.Error("a fresh switcher has an active backend")
	}
	if s.Dir() != "" {
		t.Errorf("Dir() = %q, want empty", s.Dir())
	}
	if _, _, ok := s.InputBounds("input_ids", 1); ok {
		t.Error("a fresh switcher reported input bounds")
	}
	if s.LastError() != nil {
		t.Errorf("LastError() = %v, want nil", s.LastError())
	}
}

// TestSwitcherRunWithoutLoad checks a predict before any load reports the
// sentinel the transport maps to 409, rather than dereferencing a nil kernel.
func TestSwitcherRunWithoutLoad(t *testing.T) {
	s := NewSwitcher(Config{Kind: backend.KindAuto})
	defer s.Close()

	if _, err := s.Run(context.Background(), backend.RunInput{}); err == nil {
		t.Fatal("Run without a load returned no error")
	} else if err != backend.ErrNotLoaded {
		t.Errorf("Run error = %v, want ErrNotLoaded", err)
	}
}

// TestSwitcherKindBeforeLoad checks the reported kernel falls back to the
// configured one, so a diagnostics panel can say what a load would use.
func TestSwitcherKindBeforeLoad(t *testing.T) {
	cases := []struct {
		cfg  string
		want backend.Kind
	}{
		{"", backend.KindAuto},
		{"auto", backend.KindAuto},
		{"tensorrt", backend.KindTensorRT},
		{"onnx", backend.KindONNX},
		{"nonsense", backend.KindAuto},
	}
	for _, c := range cases {
		s := NewSwitcher(Config{Kind: backend.Kind(c.cfg)})
		if got := s.Kind(); got != c.want {
			t.Errorf("Kind() with config %q = %q, want %q", c.cfg, got, c.want)
		}
		s.Close()
	}
}

// TestSwitcherConfigRoundTrip checks the configured selection is readable, since
// the HTTP load handler merges a per-request override onto it.
func TestSwitcherConfigRoundTrip(t *testing.T) {
	cfg := Config{Kind: backend.KindONNX, Provider: "cuda", RuntimePath: "rt.dll"}
	s := NewSwitcher(cfg)
	defer s.Close()

	got := s.Config()
	if got.Kind != cfg.Kind || got.Provider != cfg.Provider || got.RuntimePath != cfg.RuntimePath {
		t.Errorf("Config() = %+v, want %+v", got, cfg)
	}
}

// TestSwitcherLoadFailureIsRecorded checks a failed load leaves the switcher
// empty and records the reason, which is what the GUI reports at startup.
//
// A missing file fails in every kernel, so this needs no GPU and no runtime.
func TestSwitcherLoadFailureIsRecorded(t *testing.T) {
	s := NewSwitcher(Config{Kind: backend.KindAuto})
	defer s.Close()

	if _, err := s.Load(context.Background(), backend.Options{
		Path: "definitely-missing.engine",
	}); err == nil {
		t.Fatal("loading a missing file returned no error")
	}

	if s.Loaded() {
		t.Error("a failed load left the switcher loaded")
	}
	if s.Active() != nil {
		t.Error("a failed load installed a backend")
	}
	if s.LastError() == nil {
		t.Error("a failed load did not record LastError")
	}
}

// TestConcreteKind covers the resolution that decides whether a reload can reuse
// the resident kernel.
//
// The subtle case is auto: it is a policy, not a kernel, and which kernel it
// lands on depends on the file. Getting this wrong would make a reload of an
// .onnx through a resident TensorRT kernel try to open a graph as a plan.
func TestConcreteKind(t *testing.T) {
	cases := []struct {
		kind backend.Kind
		path string
		want backend.Kind
	}{
		// auto resolves by format, because the formats are not interchangeable.
		{backend.KindAuto, "model.onnx", backend.KindONNX},
		{backend.KindAuto, "MODEL.ONNX", backend.KindONNX},
		{backend.KindAuto, "model.engine", backend.KindTensorRT},
		{backend.KindAuto, "model", backend.KindTensorRT},
		// An explicit choice is never second-guessed, even when the extension
		// disagrees: the user asked for that kernel.
		{backend.KindTensorRT, "model.onnx", backend.KindTensorRT},
		{backend.KindONNX, "model.engine", backend.KindONNX},
		{backend.KindTensorRT, "model.engine", backend.KindTensorRT},
		{backend.KindONNX, "model.onnx", backend.KindONNX},
	}
	for _, c := range cases {
		if got := concreteKind(c.kind, c.path); got != c.want {
			t.Errorf("concreteKind(%q, %q) = %q, want %q", c.kind, c.path, got, c.want)
		}
	}
}

// TestSwitcherRejectsAnUnknownKernel checks a bad kernel name is refused before
// anything is constructed.
func TestSwitcherRejectsAnUnknownKernel(t *testing.T) {
	s := NewSwitcher(Config{Kind: backend.Kind("cuda")})
	defer s.Close()

	if _, err := s.Load(context.Background(), backend.Options{Path: "x.engine"}); err == nil {
		t.Error("loading with an unknown kernel returned no error")
	}
}

// TestSwitcherHonoursCancellation checks a cancelled context stops a load before
// it reaches the kernel, which matters because a load is expensive.
func TestSwitcherHonoursCancellation(t *testing.T) {
	s := NewSwitcher(Config{Kind: backend.KindAuto})
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := s.Load(ctx, backend.Options{Path: "x.engine"}); err == nil {
		t.Error("a cancelled load returned no error")
	}
}

// TestSwitcherUnloadAndCloseAreIdempotent checks repeated teardown is safe: the
// HTTP API and the GUI both unload, and Wails calls Shutdown on every close.
func TestSwitcherUnloadAndCloseAreIdempotent(t *testing.T) {
	s := NewSwitcher(Config{Kind: backend.KindAuto})

	s.Unload()
	s.Unload()
	s.Close()
	s.Close()

	if s.Loaded() {
		t.Error("a closed switcher reports loaded")
	}
	if s.Active() != nil {
		t.Error("a closed switcher still has an active backend")
	}
}

// TestSwitcherSatisfiesTheBackendContract checks the switcher can stand in for a
// kernel anywhere the contract is expected — which is what lets the inference
// service hold one object and still have the kernel change underneath it.
func TestSwitcherSatisfiesTheBackendContract(t *testing.T) {
	var _ backend.Backend = NewSwitcher(Config{})
}
