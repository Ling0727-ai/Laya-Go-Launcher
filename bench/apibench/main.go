// Command apibench measures a running laya-trt HTTP API.
//
// It exists because the numbers that decide the next optimisation are not
// visible from a single p50: what matters is how latency splits into a fixed
// per-run cost and a per-token cost. If the fixed part dominates, the win is in
// batching and in the per-run setup path; if the per-token part dominates, the
// win is in the engine (precision, tactics, profiles).
//
// The tool sweeps two axes:
//
//	sequence length  — one question over a state of growing size, which moves
//	                   tokens without changing the number of forward passes
//	question count   — a fixed short state with more questions, which moves the
//	                   number of forward passes at a nearly constant token count
//
// A least-squares fit over the length sweep gives inference_ms ≈ a + b·tokens,
// so `a` is the fixed cost of one run and `b` the marginal cost per token.
//
// It is read-only: it only POSTs /api/v1/predict and GETs status endpoints, so
// it can run against an instance someone else started. It never starts, stops,
// or reconfigures a server.
//
//	go run ./bench/apibench -reps 15 -out bench\sweep.json
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type predictRequest struct {
	State     any            `json:"state"`
	Questions map[string]any `json:"questions"`
}

type predictTiming struct {
	TotalMS     float64 `json:"total_ms"`
	TokenizeMS  float64 `json:"tokenize_ms"`
	InferenceMS float64 `json:"inference_ms"`
}

type predictUsage struct {
	InputTokens int `json:"input_tokens"`
}

type predictResponse struct {
	Usage  predictUsage  `json:"usage"`
	Timing predictTiming `json:"timing"`
}

// sample is one measured request.
type sample struct {
	wallMS      float64
	totalMS     float64
	tokenizeMS  float64
	inferenceMS float64
	tokens      int
}

type result struct {
	name    string
	state   string
	nq      int
	samples []sample
}

// sentence is roughly 11 words, so a word budget converts to repeats.
const sentence = "We were billed twice for March. Please refund the duplicate today. "

func stateFor(words int) string {
	reps := words / 11
	if reps < 1 {
		reps = 1
	}
	return strings.Repeat(sentence, reps)
}

// buildQuestions mixes the three question types so one scenario exercises the
// choice, score and noul head paths rather than only the cheapest one.
func buildQuestions(n int) map[string]any {
	q := make(map[string]any, n)
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("q%d", i)
		switch i % 3 {
		case 0:
			q[id] = map[string]any{
				"type":         "choice",
				"instructions": "Which department should handle this request?",
				"criteria": map[string]any{
					"billing":   "invoices, payments, refunds",
					"technical": "bugs, outages, system errors",
					"sales":     "pricing, new contracts",
				},
			}
		case 1:
			q[id] = map[string]any{
				"type":         "score",
				"instructions": "How urgent is this request?",
				"criteria":     []string{"not urgent", "soon", "critical deadline"},
			}
		default:
			q[id] = map[string]any{
				"type":         "noul",
				"instructions": "Does the user threaten to cancel or leave?",
			}
		}
	}
	return q
}

func percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}
	s := append([]float64(nil), values...)
	sort.Float64s(s)
	idx := int(math.Ceil(p/100*float64(len(s)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(s) {
		idx = len(s) - 1
	}
	return s[idx]
}

// leastSquares fits y = intercept + slope*x.
func leastSquares(xs, ys []float64) (intercept, slope float64) {
	n := float64(len(xs))
	if n < 2 {
		return 0, 0
	}
	var sx, sy, sxx, sxy float64
	for i := range xs {
		sx += xs[i]
		sy += ys[i]
		sxx += xs[i] * xs[i]
		sxy += xs[i] * ys[i]
	}
	den := n*sxx - sx*sx
	if den == 0 {
		return 0, 0
	}
	slope = (n*sxy - sx*sy) / den
	intercept = (sy - slope*sx) / n
	return intercept, slope
}

func main() {
	var (
		base    = flag.String("base", "http://127.0.0.1:8420/api/v1", "API base URL")
		reps    = flag.Int("reps", 15, "measured repetitions per scenario")
		warmup  = flag.Int("warmup", 3, "warm-up requests per scenario")
		timeout = flag.Duration("timeout", 180*time.Second, "per-request timeout")
		out     = flag.String("out", "", "write results as JSON to this path")
	)
	flag.Parse()

	client := &http.Client{Timeout: *timeout}

	getJSON := func(path string, dst any) error {
		resp, err := client.Get(*base + path)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("GET %s: HTTP %d", path, resp.StatusCode)
		}
		return json.NewDecoder(resp.Body).Decode(dst)
	}

	var health struct {
		Device map[string]any `json:"device"`
		Kernel map[string]any `json:"kernel"`
	}
	if err := getJSON("/health", &health); err != nil {
		fmt.Fprintf(os.Stderr, "health: %v\n", err)
		os.Exit(1)
	}
	var engineInfo struct {
		Path               string `json:"path"`
		Contexts           int    `json:"contexts"`
		ActivationMemoryMB float64 `json:"activation_memory_mb"`
	}
	if err := getJSON("/engine", &engineInfo); err != nil {
		fmt.Fprintf(os.Stderr, "engine: %v\n", err)
		os.Exit(1)
	}
	var limits map[string]any
	if err := getJSON("/limits", &limits); err != nil {
		fmt.Fprintf(os.Stderr, "limits: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("engine   : %s\n", engineInfo.Path)
	fmt.Printf("contexts : %d   activation_mb: %.1f\n", engineInfo.Contexts, engineInfo.ActivationMemoryMB)
	fmt.Printf("device   : %v\n\n", health.Device["name"])

	predict := func(state string, nq int) (sample, error) {
		body, err := json.Marshal(predictRequest{State: state, Questions: buildQuestions(nq)})
		if err != nil {
			return sample{}, err
		}
		start := time.Now()
		resp, err := client.Post(*base+"/predict", "application/json", bytes.NewReader(body))
		if err != nil {
			return sample{}, err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			var buf bytes.Buffer
			_, _ = buf.ReadFrom(resp.Body)
			return sample{}, fmt.Errorf("predict: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(buf.String()))
		}
		var pr predictResponse
		if err := json.NewDecoder(resp.Body).Decode(&pr); err != nil {
			return sample{}, err
		}
		return sample{
			wallMS:      float64(time.Since(start).Microseconds()) / 1000,
			totalMS:     pr.Timing.TotalMS,
			tokenizeMS:  pr.Timing.TokenizeMS,
			inferenceMS: pr.Timing.InferenceMS,
			tokens:      pr.Usage.InputTokens,
		}, nil
	}

	// The two axes. Length moves tokens at one forward pass; question count
	// moves forward passes at a nearly constant token count.
	var scenarios []struct {
		name  string
		state string
		nq    int
		axis  string
	}
	for _, words := range []int{20, 100, 250, 500, 1000, 2000, 4000} {
		scenarios = append(scenarios, struct {
			name  string
			state string
			nq    int
			axis  string
		}{fmt.Sprintf("len-%d", words), stateFor(words), 1, "length"})
	}
	for _, nq := range []int{1, 2, 4, 8, 16} {
		scenarios = append(scenarios, struct {
			name  string
			state string
			nq    int
			axis  string
		}{fmt.Sprintf("q-%d", nq), stateFor(20), nq, "questions"})
	}

	var results []result
	for _, sc := range scenarios {
		for i := 0; i < *warmup; i++ {
			if _, err := predict(sc.state, sc.nq); err != nil {
				fmt.Fprintf(os.Stderr, "%s warmup: %v\n", sc.name, err)
				os.Exit(1)
			}
		}
		r := result{name: sc.name, state: sc.state, nq: sc.nq}
		for i := 0; i < *reps; i++ {
			s, err := predict(sc.state, sc.nq)
			if err != nil {
				fmt.Fprintf(os.Stderr, "%s: %v\n", sc.name, err)
				os.Exit(1)
			}
			r.samples = append(r.samples, s)
		}
		results = append(results, r)
	}

	fmt.Printf("%-10s %6s %8s %10s %10s %10s %10s\n",
		"scenario", "tokens", "infer/tok", "tok_p50", "infer_p50", "infer_p95", "total_p50")
	fmt.Println(strings.Repeat("-", 72))

	collect := func(r result, f func(sample) float64) []float64 {
		out := make([]float64, 0, len(r.samples))
		for _, s := range r.samples {
			out = append(out, f(s))
		}
		return out
	}

	var lenTokens, lenInfer []float64
	var qCounts, qInfer []float64

	for _, r := range results {
		tokens := r.samples[0].tokens
		infP50 := percentile(collect(r, func(s sample) float64 { return s.inferenceMS }), 50)
		perTok := infP50 / float64(max(1, tokens))

		fmt.Printf("%-10s %6d %8.3f %10.2f %10.2f %10.2f %10.2f\n",
			r.name, tokens, perTok,
			percentile(collect(r, func(s sample) float64 { return s.tokenizeMS }), 50),
			infP50,
			percentile(collect(r, func(s sample) float64 { return s.inferenceMS }), 95),
			percentile(collect(r, func(s sample) float64 { return s.totalMS }), 50),
		)

		if strings.HasPrefix(r.name, "len-") {
			lenTokens = append(lenTokens, float64(tokens))
			lenInfer = append(lenInfer, infP50)
		}
		if strings.HasPrefix(r.name, "q-") {
			qCounts = append(qCounts, float64(r.nq))
			qInfer = append(qInfer, infP50)
		}
	}

	intercept, slope := leastSquares(lenTokens, lenInfer)
	fmt.Println()
	fmt.Printf("fit over the length sweep: inference_ms ≈ %.2f + %.5f · tokens\n", intercept, slope)
	if slope > 0 {
		fmt.Printf("  fixed cost per forward pass : %.2f ms\n", intercept)
		fmt.Printf("  marginal cost per token     : %.3f ms  (= %.1f ms per 512 tokens)\n",
			slope, slope*512)
		fmt.Printf("  a 512-token run at zero fixed cost would take %.2f ms\n", slope*512)
	}

	qIntercept, qSlope := leastSquares(qCounts, qInfer)
	fmt.Println()
	fmt.Printf("fit over the question sweep: inference_ms ≈ %.2f + %.2f · questions\n", qIntercept, qSlope)
	fmt.Printf("  cost of one additional question (one more forward pass): %.2f ms\n", qSlope)

	if *out != "" {
		if dir := filepath.Dir(*out); dir != "" && dir != "." {
			_ = os.MkdirAll(dir, 0o755)
		}
		doc := map[string]any{
			"captured_at": time.Now().UTC().Format(time.RFC3339),
			"base_url":    *base,
			"reps":        *reps,
			"warmup":      *warmup,
			"device":      health.Device,
			"kernel":      health.Kernel,
			"engine": map[string]any{
				"path":                 engineInfo.Path,
				"contexts":             engineInfo.Contexts,
				"activation_memory_mb": engineInfo.ActivationMemoryMB,
			},
			"limits": limits,
			"fit": map[string]any{
				"fixed_ms_per_run":        intercept,
				"ms_per_token":            slope,
				"ms_per_extra_question":   qSlope,
			},
			"scenarios": func() []map[string]any {
				var out []map[string]any
				for _, r := range results {
					out = append(out, map[string]any{
						"name":        r.name,
						"questions":   r.nq,
						"tokens":      r.samples[0].tokens,
						"tok_p50_ms":  percentile(collect(r, func(s sample) float64 { return s.tokenizeMS }), 50),
						"infer_p50_ms": percentile(collect(r, func(s sample) float64 { return s.inferenceMS }), 50),
						"infer_p95_ms": percentile(collect(r, func(s sample) float64 { return s.inferenceMS }), 95),
						"total_p50_ms": percentile(collect(r, func(s sample) float64 { return s.totalMS }), 50),
					})
				}
				return out
			}(),
		}
		raw, _ := json.MarshalIndent(doc, "", "  ")
		if err := os.WriteFile(*out, raw, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "write %s: %v\n", *out, err)
			os.Exit(1)
		}
		fmt.Printf("\nwrote %s\n", *out)
	}
}
