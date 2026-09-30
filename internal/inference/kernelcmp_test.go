package inference

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/local/laya-go-launcher/internal/backend"
	"github.com/local/laya-go-launcher/internal/ortbackend"
	"github.com/local/laya-go-launcher/internal/trtbackend"
)

// findONNXModel locates a laya ONNX graph for the cross-kernel comparison.
func findONNXModel(t *testing.T) string {
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

func findONNXRuntime(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("LAYA_ORT_RUNTIME"); p != "" {
		return p
	}
	for _, c := range []string{
		`C:\Users\lingxin\Downloads\QualityScaler-go\Assets\onnxruntime.dll`,
	} {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	t.Skip("set LAYA_ORT_RUNTIME to an onnxruntime.dll")
	return ""
}

func findONNXBridge(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("LAYA_ONNX_BRIDGE"); p != "" {
		return p
	}
	// A test binary runs from its own package directory, so walk back to the
	// repository root rather than assuming the working directory.
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for i := 0; i < 5; i++ {
		candidate := filepath.Join(dir, "build", "bin", "Release", "layatrt_onnx.dll")
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

// TestKernelsAgree is the load-bearing check for the kernel switch: the same
// request through the TensorRT plan and through the ONNX graph must produce the
// same decision.
//
// It is not a bit-exactness test — the two runtimes use different kernels and
// different precisions, so the logits legitimately differ in the low bits. What
// must hold is that the ranking and the calibrated probabilities agree closely
// enough that no answer changes. That is the property a kernel switch has to
// preserve, and the one a silent regression would break.
func TestKernelsAgree(t *testing.T) {
	tok := findTok(t)
	enginePath := findEngine(t)
	onnxPath := findONNXModel(t)
	runtimePath := findONNXRuntime(t)
	bridgePath := findONNXBridge(t)

	question := map[string]QuestionSpec{
		"department": {
			Type:         "choice",
			Instructions: "Which department should handle this?",
			Criteria: map[string]any{
				"billing":   "invoices, refunds, payment problems",
				"technical": "bugs, crashes, errors",
				"account":   "login, password, profile",
			},
		},
	}
	state := map[string]any{"body": "I was charged twice for the same order and need a refund."}

	// TensorRT first.
	trt := trtbackend.New()
	defer trt.Close()
	if _, err := trt.Load(context.Background(),
		backend.Options{Path: enginePath, Contexts: 1}); err != nil {
		t.Fatalf("load tensorrt: %v", err)
	}
	trtSvc, err := NewService(Config{Backend: trt, Tokenizer: tok, MaxLen: 64, HeadMaxLen: 48})
	if err != nil {
		t.Fatalf("tensorrt service: %v", err)
	}
	trtSvc.AdaptToEngine()
	trtResp, err := trtSvc.Predict(context.Background(), Request{State: state, Questions: question})
	if err != nil {
		t.Fatalf("tensorrt predict: %v", err)
	}

	// Then ONNX Runtime, through the same use case.
	ort := ortbackend.New(bridgePath)
	defer ort.Close()
	if _, err := ort.Load(context.Background(), backend.Options{
		Path: onnxPath, RuntimePath: runtimePath, Provider: "cpu",
	}); err != nil {
		t.Fatalf("load onnx: %v", err)
	}
	ortSvc, err := NewService(Config{Backend: ort, Tokenizer: tok, MaxLen: 64, HeadMaxLen: 48})
	if err != nil {
		t.Fatalf("onnx service: %v", err)
	}
	ortSvc.AdaptToEngine()
	ortResp, err := ortSvc.Predict(context.Background(), Request{State: state, Questions: question})
	if err != nil {
		t.Fatalf("onnx predict: %v", err)
	}

	trtAns := trtResp.Answers["department"]
	ortAns := ortResp.Answers["department"]

	t.Logf("tensorrt: choice=%q probs=%v conf=%.4f", trtAns.Choice, trtAns.Probabilities, trtAns.Confidence)
	t.Logf("onnx    : choice=%q probs=%v conf=%.4f", ortAns.Choice, ortAns.Probabilities, ortAns.Confidence)

	if trtAns.Choice != ortAns.Choice {
		t.Errorf("the kernels disagree on the answer: tensorrt=%q onnx=%q",
			trtAns.Choice, ortAns.Choice)
	}

	// Probabilities must agree closely: a divergence large enough to flip a
	// close decision is a real defect, not rounding.
	const tol = 0.05
	for label, trtP := range trtAns.Probabilities {
		ortP, ok := ortAns.Probabilities[label]
		if !ok {
			t.Errorf("onnx is missing probability for %q", label)
			continue
		}
		if d := math.Abs(trtP - ortP); d > tol {
			t.Errorf("probability for %q differs by %.4f (tensorrt=%.4f onnx=%.4f)",
				label, d, trtP, ortP)
		}
	}
}
