package backends

import (
	"context"
	"testing"

	"github.com/local/laya-go-launcher/internal/backend"
)

// TestNewSelectsTheRequestedKernel checks the factory honours an explicit
// choice and defaults to TensorRT.
//
// The backends are not loaded here, only constructed: a named kernel must be
// returned even when its runtime is unavailable, so the failure surfaces at
// load time with a real reason instead of at construction.
func TestNewSelectsTheRequestedKernel(t *testing.T) {
	cases := []struct {
		name string
		kind string
		want backend.Kind
	}{
		{"default is tensorrt", "", backend.KindTensorRT},
		{"auto is tensorrt", "auto", backend.KindTensorRT},
		{"explicit tensorrt", "tensorrt", backend.KindTensorRT},
		{"explicit onnx", "onnx", backend.KindONNX},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			be, err := New(Config{Kind: backend.Kind(c.kind)})
			if err != nil {
				t.Fatalf("New(%q): %v", c.kind, err)
			}
			defer be.Close()
			if be.Kind() != c.want {
				t.Errorf("Kind() = %q, want %q", be.Kind(), c.want)
			}
		})
	}
}

// TestNewRejectsAnUnknownKernel checks a typo is a construction error, not a
// silent fall back to something the user did not ask for.
func TestNewRejectsAnUnknownKernel(t *testing.T) {
	if _, err := New(Config{Kind: backend.Kind("cuda")}); err == nil {
		t.Error("New(cuda) returned no error")
	}
}

// TestAutoDoesNotFallBackForAPlanPath is the important policy check.
//
// auto may fall back from TensorRT to ONNX Runtime, but only when the model is
// an ONNX graph. A TensorRT plan cannot be opened by ONNX Runtime, so a failure
// to load one is not something the other kernel can fix — and trying would turn
// a clear "this plan is broken" into a confusing double failure.
func TestAutoDoesNotFallBackForAPlanPath(t *testing.T) {
	// A path that does not exist fails in both kernels; the point is that the
	// error is the TensorRT one, not a combined message.
	_, _, note, err := Auto(context.Background(), Config{Kind: backend.KindAuto},
		backend.Options{Path: "definitely-missing.engine"})
	if err == nil {
		t.Fatal("Auto on a missing plan returned no error")
	}
	if note != "" {
		t.Errorf("Auto reported a fall back note %q for a plan path", note)
	}
	t.Logf("error (informational): %v", err)
}

// TestAutoNeverSubstitutesAnExplicitKernel checks that naming TensorRT means
// TensorRT: a GPU request that quietly lands elsewhere is worse than a failure,
// because it looks like it worked.
func TestAutoNeverSubstitutesAnExplicitKernel(t *testing.T) {
	_, _, note, err := Auto(context.Background(), Config{Kind: backend.KindTensorRT},
		backend.Options{Path: "definitely-missing.onnx"})
	if err == nil {
		t.Fatal("Auto with an explicit tensorrt request returned no error for a missing file")
	}
	if note != "" {
		t.Errorf("an explicit kernel produced a fall back note %q", note)
	}
}

// TestMissingInputs covers the model-compatibility check the picker uses.
func TestMissingInputs(t *testing.T) {
	laya := []string{"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"}

	full := backend.Info{Inputs: []backend.TensorInfo{
		{Name: "input_ids"}, {Name: "attention_mask"}, {Name: "marker_pos"},
		{Name: "marker_mask"}, {Name: "qtype"},
	}}
	if got := MissingInputs(full, laya); len(got) != 0 {
		t.Errorf("complete model reported missing %v", got)
	}

	partial := backend.Info{Inputs: []backend.TensorInfo{
		{Name: "input_ids"}, {Name: "attention_mask"},
	}}
	got := MissingInputs(partial, laya)
	if len(got) != 3 {
		t.Errorf("partial model reported missing %v, want 3 entries", got)
	}

	none := backend.Info{Inputs: []backend.TensorInfo{{Name: "input"}}}
	if got := MissingInputs(none, laya); len(got) != len(laya) {
		t.Errorf("unrelated model reported missing %v, want all %d", got, len(laya))
	}
}

// TestIsUnavailable checks the classifier the transport uses to decide between
// "install a runtime" and "this file is wrong".
func TestIsUnavailable(t *testing.T) {
	if !IsUnavailable(backend.ErrRuntimeMissing) {
		t.Error("ErrRuntimeMissing was not classified as unavailable")
	}
	if IsUnavailable(backend.ErrIncompatible) {
		t.Error("ErrIncompatible was classified as unavailable")
	}
	if IsUnavailable(nil) {
		t.Error("nil was classified as unavailable")
	}
}

// TestIsONNXPath covers the extension test the fallback decision rests on.
func TestIsONNXPath(t *testing.T) {
	cases := map[string]bool{
		"model.onnx":     true,
		"MODEL.ONNX":     true,
		"a/b/model.onnx": true,
		"model.engine":   false,
		"model":          false,
		"onnx":           false,
		"model.onnx.tmp": false,
	}
	for path, want := range cases {
		if got := isONNXPath(path); got != want {
			t.Errorf("isONNXPath(%q) = %v, want %v", path, got, want)
		}
	}
}

// TestWithProvider checks the precedence: a per-load option wins over the
// process configuration, and the configuration fills in what the load omitted.
func TestWithProvider(t *testing.T) {
	cfg := Config{Provider: "cuda", RuntimePath: "cfg.dll", ProviderOptions: map[string]string{"a": "1"}}

	// Nothing on the load: the configuration supplies everything.
	got := withProvider(cfg, backend.Options{})
	if got.Provider != "cuda" || got.RuntimePath != "cfg.dll" || got.ProviderOptions["a"] != "1" {
		t.Errorf("configuration did not fill the load: %+v", got)
	}

	// A per-load choice wins.
	got = withProvider(cfg, backend.Options{Provider: "cpu", RuntimePath: "load.dll"})
	if got.Provider != "cpu" {
		t.Errorf("Provider = %q, want cpu", got.Provider)
	}
	if got.RuntimePath != "load.dll" {
		t.Errorf("RuntimePath = %q, want load.dll", got.RuntimePath)
	}
}
