package ortbackend

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/local/laya-go-launcher/internal/backend"
)

// findBridge locates layatrt_onnx.dll. It is built by build.ps1 into
// build/bin/Release; a test binary runs from its own package directory, so the
// repository root has to be walked back to.
func findBridge(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("LAYA_ONNX_BRIDGE"); p != "" {
		return p
	}
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for i := 0; i < 5; i++ {
		candidate := filepath.Join(dir, "build", "bin", "Release", DefaultDLL)
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Skip("build/bin/Release/layatrt_onnx.dll not found; run .\\build.ps1")
	return ""
}

// findRuntime locates an onnxruntime.dll. The tests need a real runtime, so an
// environment override comes first and a couple of known layouts after it.
func findRuntime(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("LAYA_ORT_RUNTIME"); p != "" {
		return p
	}
	for _, c := range []string{
		filepath.Join("Assets", "onnxruntime.dll"),
		`C:\Users\lingxin\Downloads\QualityScaler-go\Assets\onnxruntime.dll`,
	} {
		if _, err := os.Stat(c); err == nil {
			abs, absErr := filepath.Abs(c)
			if absErr == nil {
				return abs
			}
			return c
		}
	}
	t.Skip("set LAYA_ORT_RUNTIME to an onnxruntime.dll")
	return ""
}

// findModel locates a laya ONNX graph.
func findModel(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("LAYA_ONNX_MODEL"); p != "" {
		return p
	}
	for _, c := range []string{
		`C:\Users\lingxin\Documents\laya-ort-probe\laya_dyn.onnx`,
		`C:\Users\lingxin\Documents\laya-ort-probe\laya_ctx8192.onnx`,
	} {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	t.Skip("set LAYA_ONNX_MODEL to a laya .onnx graph")
	return ""
}

func TestRequireProvider(t *testing.T) {
	cases := []struct {
		name, requested, actual, note string
		wantErr                       bool
	}{
		{name: "legacy automatic fallback", actual: "cpu", note: "CUDA unavailable"},
		{name: "cuda selected", requested: "cuda", actual: "cuda"},
		{name: "directml selected", requested: "directml", actual: "directml"},
		{name: "dml alias", requested: "dml", actual: "directml"},
		{name: "cpu selected", requested: "cpu", actual: "cpu"},
		{name: "cuda fell back", requested: "cuda", actual: "cpu", note: "CUDA unavailable", wantErr: true},
		{name: "directml fell back", requested: "directml", actual: "cpu", wantErr: true},
		{name: "unknown provider", requested: "vulkan", actual: "cpu", wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := requireProvider(tc.requested, tc.actual, tc.note)
			if (err != nil) != tc.wantErr {
				t.Fatalf("requireProvider(%q, %q, %q) = %v, want error %t", tc.requested, tc.actual, tc.note, err, tc.wantErr)
			}
			if tc.note != "" && tc.wantErr && !strings.Contains(err.Error(), tc.note) {
				t.Errorf("error %q omitted provider failure detail", err)
			}
		})
	}
}

// TestLoadDescribesModel checks the bridge reports the model's IO declaration
// correctly: the five named inputs with their dtypes, and the two outputs.
func TestLoadDescribesModel(t *testing.T) {
	be := New(findBridge(t))
	defer be.Close()

	info, err := be.Load(context.Background(), backend.Options{
		Path:        findModel(t),
		RuntimePath: findRuntime(t),
		Provider:    "cpu",
	})
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	t.Logf("backend=%s runtime=%s device=%s", info.Backend, info.Runtime, info.Device)
	for _, in := range info.Inputs {
		t.Logf("in  %-16s %-8s %v", in.Name, in.DType, in.Shape)
	}
	for _, out := range info.Outputs {
		t.Logf("out %-16s %-8s %v", out.Name, out.DType, out.Shape)
	}

	// The contract the predict use case depends on.
	wantInputs := map[string]backend.DType{
		"input_ids":      backend.Int64,
		"attention_mask": backend.Int64,
		"marker_pos":     backend.Int64,
		"marker_mask":    backend.Bool,
		"qtype":          backend.Int64,
	}
	for name, dtype := range wantInputs {
		got, ok := backend.Find(info.Inputs, name)
		if !ok {
			t.Errorf("input %q is missing", name)
			continue
		}
		if got.DType != dtype {
			t.Errorf("input %q dtype = %s, want %s", name, got.DType, dtype)
		}
	}
	for _, name := range []string{"logits", "act_logits"} {
		got, ok := backend.Find(info.Outputs, name)
		if !ok {
			t.Errorf("output %q is missing", name)
			continue
		}
		if got.DType != backend.Float32 {
			t.Errorf("output %q dtype = %s, want float32", name, got.DType)
		}
	}

	if be.Kind() != backend.KindONNX {
		t.Errorf("kind = %s, want onnx", be.Kind())
	}
	if !be.Loaded() {
		t.Error("backend reports not loaded after a successful load")
	}
}

// TestDynamicInputBoundsAreNotInvented checks a symbolic dimension reports no
// bound rather than a fabricated maximum.
//
// This matters because the caller clamps its token budget to whatever bound it
// is given: inventing one would silently cap (or fail to cap) the sequence
// length. The graph declares "batch" and "seq", which carry no number.
func TestDynamicInputBoundsAreNotInvented(t *testing.T) {
	be := New(findBridge(t))
	defer be.Close()

	if _, err := be.Load(context.Background(), backend.Options{
		Path:        findModel(t),
		RuntimePath: findRuntime(t),
		Provider:    "cpu",
	}); err != nil {
		t.Fatalf("load: %v", err)
	}

	// input_ids is [batch, seq], both symbolic in the dynamic export.
	if lo, hi, ok := be.InputBounds("input_ids", 1); ok {
		t.Errorf("dynamic seq reported bounds %d..%d; want ok=false", lo, hi)
	}
	// qtype is [batch] — also symbolic.
	if lo, hi, ok := be.InputBounds("qtype", 0); ok {
		t.Errorf("dynamic batch reported bounds %d..%d; want ok=false", lo, hi)
	}
	// An input that does not exist must not claim a bound either.
	if _, _, ok := be.InputBounds("nope", 0); ok {
		t.Error("unknown input reported bounds")
	}
}

// TestRunProducesLayaShapedOutputs is the real end-to-end check: a forward pass
// over the five inputs returns two finite float32 outputs of the right shape.
func TestRunProducesLayaShapedOutputs(t *testing.T) {
	be := New(findBridge(t))
	defer be.Close()

	if _, err := be.Load(context.Background(), backend.Options{
		Path:        findModel(t),
		RuntimePath: findRuntime(t),
		Provider:    "cpu",
	}); err != nil {
		t.Fatalf("load: %v", err)
	}

	const (
		batch   = 1
		length  = 32
		markers = 3
	)

	ids := make([]int64, batch*length)
	att := make([]int64, batch*length)
	for i := 0; i < length; i++ {
		ids[i] = int64(100 + (i*7919)%19000)
		att[i] = 1
	}
	ids[0] = 50281 // CLS, as the sequence builder emits
	pos := []int64{5, 10, 15}
	mask := []byte{1, 1, 1}
	qt := []int64{0}

	outputs, err := be.Run(context.Background(), backend.RunInput{
		Inputs: []backend.Tensor{
			{Name: "input_ids", DType: backend.Int64, Data: ids, Shape: []int{batch, length}},
			{Name: "attention_mask", DType: backend.Int64, Data: att, Shape: []int{batch, length}},
			{Name: "marker_pos", DType: backend.Int64, Data: pos, Shape: []int{batch, markers}},
			{Name: "marker_mask", DType: backend.Bool, Data: mask, Shape: []int{batch, markers}},
			{Name: "qtype", DType: backend.Int64, Data: qt, Shape: []int{batch}},
		},
		Outputs: []string{"logits", "act_logits"},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(outputs) != 2 {
		t.Fatalf("got %d outputs, want 2", len(outputs))
	}

	byName := map[string]backend.Output{}
	for _, out := range outputs {
		byName[out.Name] = out
	}

	logits, err := byName["logits"].Floats()
	if err != nil {
		t.Fatalf("logits: %v", err)
	}
	if len(logits) != batch*markers {
		t.Errorf("logits has %d values, want %d", len(logits), batch*markers)
	}
	for i, v := range logits {
		if v != v {
			t.Errorf("logits[%d] is NaN", i)
		}
	}

	act, err := byName["act_logits"].Floats()
	if err != nil {
		t.Fatalf("act_logits: %v", err)
	}
	if len(act) != batch*2 {
		t.Errorf("act_logits has %d values, want %d", len(act), batch*2)
	}

	t.Logf("logits     = %v", logits)
	t.Logf("act_logits = %v", act)
}

// TestUnloadAndReload checks the session lifecycle, because the HTTP API lets a
// client unload and load again while requests may be in flight.
func TestUnloadAndReload(t *testing.T) {
	be := New(findBridge(t))
	defer be.Close()

	model, runtime := findModel(t), findRuntime(t)
	for i := 0; i < 2; i++ {
		if _, err := be.Load(context.Background(), backend.Options{
			Path: model, RuntimePath: runtime, Provider: "cpu",
		}); err != nil {
			t.Fatalf("load %d: %v", i, err)
		}
		if !be.Loaded() {
			t.Fatalf("load %d: not loaded", i)
		}
		be.Unload()
		if be.Loaded() {
			t.Fatalf("unload %d: still loaded", i)
		}
		if _, ok := be.Info(); ok {
			t.Fatalf("unload %d: Info still reports a model", i)
		}
	}
}

// TestRunWithoutLoadFails checks a run before any load reports the sentinel the
// transport classifies, rather than panicking on a nil session.
func TestRunWithoutLoadFails(t *testing.T) {
	be := New(findBridge(t))
	defer be.Close()

	_, err := be.Run(context.Background(), backend.RunInput{})
	if err == nil {
		t.Fatal("run without a load returned no error")
	}
	if !isErr(err, backend.ErrNotLoaded) {
		t.Errorf("error = %v, want ErrNotLoaded", err)
	}
}

// TestProbeReportsRuntimeVersion checks the diagnostics path reports the actual
// installed ONNX Runtime version.
//
// A probe that always answers "available, version unknown" is worse than
// useless for the case it exists to catch — a stale or mismatched DLL — so this
// asserts a real version string comes back.
func TestProbeReportsRuntimeVersion(t *testing.T) {
	bridge := findBridge(t)
	runtimePath := findRuntime(t)

	version, err := Probe(bridge, runtimePath)
	if err != nil {
		t.Fatalf("Probe: %v", err)
	}
	if version == "" {
		t.Fatal("Probe reported an empty version for a runtime that loads")
	}
	t.Logf("ONNX Runtime version: %s", version)

	// A version looks like "1.25.0"; require at least one dot so a placeholder
	// or a path cannot pass.
	if !strings.Contains(version, ".") {
		t.Errorf("version %q does not look like a runtime version", version)
	}
}

// TestProbeWithoutBridgeFails checks a missing bridge library is reported as a
// missing runtime, which the transport maps to 503 rather than 500.
func TestProbeWithoutBridgeFails(t *testing.T) {
	_, err := Probe(filepath.Join(t.TempDir(), "no-such-bridge.dll"), "")
	if err == nil {
		t.Fatal("Probe with a missing bridge returned no error")
	}
	if !isErr(err, backend.ErrRuntimeMissing) {
		t.Errorf("error = %v, want ErrRuntimeMissing", err)
	}
}

// TestProbeWithMissingRuntimePathFails checks an explicitly named runtime that
// does not exist is reported rather than silently falling back to discovery.
func TestProbeWithMissingRuntimePathFails(t *testing.T) {
	_, err := Probe(findBridge(t), filepath.Join(t.TempDir(), "no-such-runtime.dll"))
	if err == nil {
		t.Fatal("Probe with a missing runtime path returned no error")
	}
	if !isErr(err, backend.ErrRuntimeMissing) {
		t.Errorf("error = %v, want ErrRuntimeMissing", err)
	}
}

// isErr reports whether err matches target through the standard error chain.
//
// This deliberately delegates to errors.Is rather than walking Unwrap by hand:
// a missing-file error arrives as *os.PathError wrapping a syscall.Errno, and
// Errno matches os.ErrNotExist through its own Is method rather than through
// Unwrap. A hand-rolled walk therefore reports "not a match" for the very case
// these tests check.
func isErr(err, target error) bool {
	return errors.Is(err, target)
}

// TestEnginePathRejected checks pointing the ONNX kernel at a TensorRT plan is
// reported as the wrong kernel instead of failing deep inside ONNX Runtime.
//
// The distinction is ErrWrongFormat rather than ErrIncompatible: the file is a
// valid plan, just not a graph, so the remedy is to select the other kernel
// rather than to replace the file. The transport turns the two sentinels into
// different wire codes for that reason.
//
// The path deliberately does not exist. The format guard has to run before the
// existence check, because "wrong kernel" stays the more useful answer for a
// plan whether or not it is present.
func TestEnginePathRejected(t *testing.T) {
	be := New(findBridge(t))
	defer be.Close()

	_, err := be.Load(context.Background(), backend.Options{
		Path:        "model.engine",
		RuntimePath: findRuntime(t),
		Provider:    "cpu",
	})
	if err == nil {
		t.Fatal("loading a .engine through the ONNX backend succeeded")
	}
	if !isErr(err, backend.ErrWrongFormat) {
		t.Errorf("error = %v, want ErrWrongFormat", err)
	}
}

// TestMissingModelIsNotExist checks a graph that is not there is reported as a
// missing file rather than an opaque load failure.
//
// ONNX Runtime describes this case in prose ("File doesn't exist") with no
// errno, so without a check ahead of the native call the neutral os.ErrNotExist
// never reaches the caller. The transport maps that sentinel to 404, so losing
// it turns "you pointed at a file that is not there" into a 500.
func TestMissingModelIsNotExist(t *testing.T) {
	be := New(findBridge(t))
	defer be.Close()

	missing := filepath.Join(t.TempDir(), "definitely-missing.onnx")
	_, err := be.Load(context.Background(), backend.Options{
		Path:        missing,
		RuntimePath: findRuntime(t),
		Provider:    "cpu",
	})
	if err == nil {
		t.Fatal("loading a missing .onnx succeeded")
	}
	if !isErr(err, os.ErrNotExist) {
		t.Errorf("error = %v, want os.ErrNotExist", err)
	}
}
