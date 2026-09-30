package inference

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/local/laya-go-launcher/internal/backend"
	"github.com/local/laya-go-launcher/internal/kernel"
	"github.com/local/laya-go-launcher/internal/tokenizer"
	"github.com/local/laya-go-launcher/internal/trtbackend"
)

func findEngine(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("LAYA_ENGINE"); p != "" {
		return p
	}
	home, _ := os.UserHomeDir()
	for _, c := range []string{
		filepath.Join(home, "Documents", "laya-ort-probe", "laya_e2e.engine"),
		filepath.Join(home, "Documents", "laya-ort-probe", "laya_fp16_fixed.engine"),
	} {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	t.Skip("set LAYA_ENGINE to a built laya engine")
	return ""
}

func findTok(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	if p := os.Getenv("LAYA_TOKENIZER"); p != "" {
		tk, err := tokenizer.Load(p)
		if err != nil {
			t.Fatalf("load tokenizer: %v", err)
		}
		return tk
	}
	home, _ := os.UserHomeDir()
	pattern := filepath.Join(home, ".cache", "huggingface", "hub",
		"models--convaiinnovations--laya", "snapshots", "*", "tokenizer", "tokenizer.json")
	matches, _ := filepath.Glob(pattern)
	best := ""
	for _, m := range matches {
		if strings.Compare(m, best) > 0 {
			best = m
		}
	}
	if best == "" {
		t.Skip("laya tokenizer.json not found")
	}
	tk, err := tokenizer.Load(best)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	return tk
}

// TestRawOutputs dumps the engine's outputs so the act head's scale is visible.
func TestRawOutputs(t *testing.T) {
	eng, err := kernel.LoadEngine(findEngine(t))
	if err != nil {
		t.Fatalf("load engine: %v", err)
	}
	defer eng.Close()

	tensors, err := eng.Tensors()
	if err != nil {
		t.Fatalf("tensors: %v", err)
	}
	for _, ti := range tensors {
		t.Logf("%-6s %-16s %-8s %v", map[bool]string{true: "in", false: "out"}[ti.IsInput],
			ti.Name, ti.DType, ti.Shape)
	}

	ctx, err := eng.NewContext()
	if err != nil {
		t.Fatalf("context: %v", err)
	}
	defer ctx.Close()

	const L, K = 64, 3
	ids := make([]int64, L)
	att := make([]int64, L)
	for i := 0; i < L; i++ {
		ids[i] = int64(100 + (i*7919)%19000)
		att[i] = 1
	}
	ids[0] = 50281
	pos := []int64{5, 10, 15}
	mask := []byte{1, 1, 1}
	qt := []int64{0}

	run, err := ctx.NewRun()
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	defer run.Close()

	for _, in := range []struct {
		name  string
		dtype kernel.DType
		data  any
		shape []int
	}{
		{"input_ids", kernel.Int64, ids, []int{1, L}},
		{"attention_mask", kernel.Int64, att, []int{1, L}},
		{"marker_pos", kernel.Int64, pos, []int{1, K}},
		{"marker_mask", kernel.Bool, mask, []int{1, K}},
		{"qtype", kernel.Int64, qt, []int{1}},
	} {
		if err := run.AddInput(in.name, in.dtype, in.data, in.shape); err != nil {
			t.Fatalf("add %s: %v", in.name, err)
		}
	}
	if err := run.AddOutput("logits"); err != nil {
		t.Fatalf("add logits: %v", err)
	}
	if err := run.AddOutput("act_logits"); err != nil {
		t.Fatalf("add act_logits: %v", err)
	}
	if _, err := run.Resolve(); err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if err := run.Execute(); err != nil {
		t.Fatalf("execute: %v", err)
	}

	for i, name := range run.OutputNames() {
		flat, err := run.ReadOutput(i)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		t.Logf("%s = %v", name, flat)
	}
}

// TestPredictAnswerShape checks the JSON-facing fields are populated.
func TestPredictAnswerShape(t *testing.T) {
	be := trtbackend.New()
	defer be.Close()

	svc, err := NewService(Config{
		Backend: be, Tokenizer: findTok(t),
		MaxLen: 64, HeadMaxLen: 48, PadLen: 64,
	})
	if err != nil {
		t.Fatalf("service: %v", err)
	}
	if _, err := be.Load(context.Background(),
		backend.Options{Path: findEngine(t), Contexts: 1}); err != nil {
		t.Fatalf("load: %v", err)
	}

	resp, err := svc.Predict(context.Background(), Request{
		State: map[string]any{"body": "Billed twice"},
		Questions: map[string]QuestionSpec{
			"department": {
				Type: "choice", Instructions: "Which?",
				Criteria: map[string]any{"billing": "invoices", "technical": "bugs"},
			},
		},
	})
	if err != nil {
		t.Fatalf("predict: %v", err)
	}

	ans, ok := resp.Answers["department"]
	if !ok {
		t.Fatalf("missing answer for department")
	}
	if ans.Type != "choice" {
		t.Errorf("type = %q, want choice", ans.Type)
	}
	if ans.Choice == "" {
		t.Errorf("choice is empty; probabilities=%v", ans.Probabilities)
	}
	if len(ans.Probabilities) != 2 {
		t.Errorf("got %d probabilities, want 2", len(ans.Probabilities))
	}
	if ans.Confidence <= 0 || ans.Confidence > 1 {
		t.Errorf("confidence = %v, want (0,1]", ans.Confidence)
	}
	if ans.Action.ActProbability < 0 || ans.Action.ActProbability > 1 {
		t.Errorf("act_probability = %v, want [0,1]", ans.Action.ActProbability)
	}
	t.Logf("answer: %+v", ans)
	t.Logf("timing: %+v", resp.Timing)
}
