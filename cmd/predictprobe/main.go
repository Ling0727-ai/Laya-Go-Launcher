// Command predictprobe runs the predict path directly, so a native crash can be
// bisected without the HTTP layer in the way.
//
//	go run ./cmd/predictprobe -engine <path> -pad-len 64 -max-len 64
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/local/laya-go-launcher/internal/backend"
	"github.com/local/laya-go-launcher/internal/backends"
	"github.com/local/laya-go-launcher/internal/inference"
	"github.com/local/laya-go-launcher/internal/tokenizer"
)

func main() {
	enginePath := flag.String("engine", "", "model path (.engine or .onnx)")
	tokPath := flag.String("tokenizer", "", "tokenizer.json")
	padLen := flag.Int("pad-len", 64, "fixed sequence length")
	maxLen := flag.Int("max-len", 64, "max_len")
	headMaxLen := flag.Int("head-max-len", 48, "head_max_len")
	nQuestions := flag.Int("questions", 1, "how many questions to send")
	backendName := flag.String("backend", "", "execution kernel: auto, tensorrt or onnx")
	provider := flag.String("provider", "", "ONNX Runtime provider: cuda, cpu or directml")
	flag.Parse()

	kind, err := backend.ParseKind(*backendName)
	if err != nil {
		log.Fatalf("backend: %v", err)
	}
	cfg := backends.Config{Kind: kind, Provider: *provider}

	tok, err := findTokenizer(*tokPath)
	if err != nil {
		log.Fatalf("tokenizer: %v", err)
	}
	step("tokenizer loaded: vocab=%d", tok.VocabSize())

	// One switcher owns the model, and the service predicts through it. Building
	// a backend with New and then loading with Auto put the model in a second,
	// unreachable object, so Predict failed with "no model is loaded" after a
	// load that had just reported success.
	model := backends.NewSwitcher(cfg)
	defer model.Close()

	svc, err := inference.NewService(inference.Config{
		Backend: model, Tokenizer: tok,
		MaxLen: *maxLen, HeadMaxLen: *headMaxLen, PadLen: *padLen,
	})
	if err != nil {
		log.Fatalf("service: %v", err)
	}
	step("service built")

	info, note, err := model.LoadWith(context.Background(), cfg, backend.Options{
		Path: *enginePath, Provider: *provider,
	})
	if err != nil {
		log.Fatalf("load: %v", err)
	}
	if note != "" {
		step("backend note: %s", note)
	}
	svc.AdaptToEngine()
	step("model loaded: %s [%s] %d contexts", info.Path, info.Backend, info.Contexts)

	questions := map[string]inference.QuestionSpec{
		"department": {
			Type:         "choice",
			Instructions: "Which?",
			Criteria:     map[string]any{"billing": "invoices, refunds", "technical": "bugs"},
		},
	}
	if *nQuestions > 1 {
		questions["churn"] = inference.QuestionSpec{
			Type: "noul", Instructions: "Will they cancel?",
		}
	}

	step("calling Predict with %d question(s)", len(questions))
	resp, err := svc.Predict(context.Background(), inference.Request{
		State:     map[string]any{"body": "Billed twice"},
		Questions: questions,
	})
	if err != nil {
		fmt.Printf("Predict error: %v\n", err)
		os.Exit(1)
	}
	step("Predict returned")
	fmt.Printf("model=%s tokens=%d total=%.2fms tokenize=%.2fms infer=%.2fms\n",
		resp.Model, resp.Usage.InputTokens,
		resp.Timing.TotalMS, resp.Timing.TokenizeMS, resp.Timing.InferenceMS)
	for id, a := range resp.Answers {
		fmt.Printf("  %-12s type=%-6s choice=%-10q noul=%.4f conf=%.4f probs=%v\n",
			id, a.Type, a.Choice, a.NoUL, a.Confidence, a.Probabilities)
	}
	step("done")
}

func step(format string, args ...any) {
	fmt.Printf("  %s\n", fmt.Sprintf(format, args...))
	os.Stdout.Sync()
}

func findTokenizer(explicit string) (*tokenizer.Tokenizer, error) {
	if explicit != "" {
		return tokenizer.Load(explicit)
	}
	home, _ := os.UserHomeDir()
	pattern := filepath.Join(home, ".cache", "huggingface", "hub",
		"models--convaiinnovations--laya", "snapshots", "*", "tokenizer", "tokenizer.json")
	matches, _ := filepath.Glob(pattern)
	var best string
	for _, m := range matches {
		if strings.Compare(m, best) > 0 {
			best = m
		}
	}
	if best == "" {
		return nil, fmt.Errorf("no tokenizer.json found")
	}
	return tokenizer.Load(best)
}
