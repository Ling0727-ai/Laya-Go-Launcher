package inference

import (
	"context"
	"os"
	"testing"

	"github.com/local/laya-go-launcher/internal/backend"
	"github.com/local/laya-go-launcher/internal/backends"
)

// TestManualEnginesAgree exercises real inference through the same service after
// each manual selection. It is opt-in because two GPU ORT distributions and the
// full model must be installed on the test machine.
func TestManualEnginesAgree(t *testing.T) {
	root := os.Getenv("LAYA_TRT_TEST_ONNX_ROOT")
	if root == "" {
		t.Skip("set LAYA_TRT_TEST_ONNX_ROOT to an Assets/onnx runtime tree")
	}
	t.Setenv("LAYA_TRT_ONNX_ROOT", root)
	model, plan, tok := findONNXModel(t), findEngine(t), findTok(t)
	switcher := backends.NewSwitcher(backends.Config{Kind: backend.KindAuto, DLLPath: findONNXBridge(t)})
	defer switcher.Close()
	svc, err := NewService(Config{Backend: switcher, Tokenizer: tok, MaxLen: 64, HeadMaxLen: 48})
	if err != nil {
		t.Fatal(err)
	}
	question := map[string]QuestionSpec{
		"department": {
			Type: "choice", Instructions: "Which department should handle this?",
			Criteria: map[string]any{
				"billing":   "invoices, refunds, payment problems",
				"technical": "bugs, crashes, errors",
				"account":   "login, password, profile",
			},
		},
	}
	request := Request{
		State:     map[string]any{"body": "I was charged twice for the same order and need a refund."},
		Questions: question,
	}
	var first string
	for _, choice := range []struct {
		path, provider string
		kind           backend.Kind
	}{
		{model, "cuda", backend.KindONNX},
		{model, "directml", backend.KindONNX},
		{plan, "", backend.KindTensorRT},
	} {
		info, _, err := switcher.LoadWith(context.Background(), switcher.Config(), backend.Options{
			Path: choice.path, Kind: choice.kind, Provider: choice.provider, Contexts: 1,
		})
		if err != nil {
			t.Fatalf("load %s/%s: %v", choice.kind, choice.provider, err)
		}
		svc.AdaptToEngine()
		response, err := svc.Predict(context.Background(), request)
		if err != nil {
			t.Fatalf("predict %s/%s: %v", choice.kind, choice.provider, err)
		}
		answer := response.Answers["department"]
		t.Logf("%s/%s: device=%s choice=%s", choice.kind, choice.provider, info.Device, answer.Choice)
		if first == "" {
			first = answer.Choice
		}
		if answer.Choice != first {
			t.Fatalf("answer changed: %s vs %s", first, answer.Choice)
		}
	}
}
