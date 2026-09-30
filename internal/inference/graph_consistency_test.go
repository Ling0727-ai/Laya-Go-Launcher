package inference

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/local/laya-go-launcher/internal/backend"
	"github.com/local/laya-go-launcher/internal/trtbackend"
)

// findMultiProfileEngine locates a plan built with more than one optimisation
// profile (bench/build-engines.ps1 -MultiProfile).
func findMultiProfileEngine(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("LAYA_ENGINE_MULTIPROFILE"); p != "" {
		return p
	}
	dir, _ := os.Getwd()
	for i := 0; i < 5; i++ {
		c := filepath.Join(dir, "engines", "laya_s8192_fp16_p2.engine")
		if _, err := os.Stat(c); err == nil {
			return c
		}
		dir = filepath.Dir(dir)
	}
	t.Skip("set LAYA_ENGINE_MULTIPROFILE to a multi-profile laya engine")
	return ""
}

// TestGraphReplayMatchesPlainEnqueue interleaves requests of different shapes
// and checks every repeat returns exactly what the first run of that request
// returned.
//
// The first run of a shape is always a plain enqueueV3 (a change is pending);
// the second is captured into a CUDA graph and later ones replay it. Any stale
// graph, a replay after a shape or profile switch, or a moved buffer that the
// graph still points at would show up here as a changed answer.
//
// It also covers the batch split: two long questions fit neither the batched
// profile (too long) nor the long profile (batch 1), so they must be run as two
// single-row passes on profile 1.
func TestGraphReplayMatchesPlainEnqueue(t *testing.T) {
	tok := findTok(t)
	be := trtbackend.New()
	defer be.Close()
	if _, err := be.Load(context.Background(),
		backend.Options{Path: findMultiProfileEngine(t), Contexts: 1}); err != nil {
		t.Fatalf("load: %v", err)
	}
	svc, err := NewService(Config{Backend: be, Tokenizer: tok, MaxLen: 2048, HeadMaxLen: 256})
	if err != nil {
		t.Fatalf("service: %v", err)
	}
	svc.AdaptToEngine()

	two := map[string]QuestionSpec{
		"dept": {Type: "choice", Instructions: "Which team should handle `message`?",
			Criteria: map[string]any{"billing": "invoices, refunds", "technical": "bugs, outages",
				"sales": "pricing, demos"}},
		"refund": {Type: "noul", Instructions: "Does the customer ask for money back?"},
	}
	one := map[string]QuestionSpec{"refund": two["refund"]}
	long := strings.Repeat("The invoice was charged twice and I would like the duplicate refunded. ", 180)

	reqs := []struct {
		name string
		req  Request
	}{
		{"short-b2", Request{State: map[string]any{"message": "I was charged twice, please refund."}, Questions: two}},
		{"short-b1", Request{State: map[string]any{"message": "The API returns 500 errors."}, Questions: one}},
		{"other-len-b2", Request{State: map[string]any{"message": "What does the enterprise plan cost for 200 seats, and can we get a demo next week?"}, Questions: two}},
		{"long-b1", Request{State: map[string]any{"message": long}, Questions: one}},
		{"long-b2-split", Request{State: map[string]any{"message": long}, Questions: two}},
	}

	type result map[string]Answer
	first := map[string]result{}
	// Each request three times in a row (plain, capture, replay), then the whole
	// set again in a different order so every replay follows a shape change.
	order := []int{0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 0, 3, 0, 0, 2, 4, 1, 1, 0, 0}
	for step, i := range order {
		r := reqs[i]
		resp, err := svc.Predict(context.Background(), r.req)
		if err != nil {
			t.Fatalf("step %d %s: %v", step, r.name, err)
		}
		got := result(resp.Answers)
		want, seen := first[r.name]
		if !seen {
			first[r.name] = got
			t.Logf("%-14s tokens=%d answers=%+v", r.name, resp.Usage.InputTokens, got)
			continue
		}
		for id, a := range want {
			b := got[id]
			if a.Choice != b.Choice || a.Action.ActProbability != b.Action.ActProbability ||
				a.Confidence != b.Confidence || a.NoUL != b.NoUL {
				t.Errorf("step %d %s/%s changed: first=%+v now=%+v", step, r.name, id, a, b)
			}
			for label, p := range a.Probabilities {
				if b.Probabilities[label] != p {
					t.Errorf("step %d %s/%s p[%s] changed: %v -> %v", step, r.name, id, label, p, b.Probabilities[label])
				}
			}
		}
	}
}
