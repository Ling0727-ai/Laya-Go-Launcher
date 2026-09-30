// Command autobench measures end-to-end Predict latency (P50/P95/P99) and answer
// accuracy across several engines over one fixed case set, one engine at a time.
// It is the Go counterpart of laya/research/scripts/bench_auto.py; output is log only.
//
//	go run ./bench/autobench                                  # default: trt + onnx(cuda)
//	go run ./bench/autobench -engines trt,onnx-cpu -reps 100 -log bench\autobench.log
//	go run ./bench/autobench -engine "x=onnx:cuda:C:\m.onnx"  # custom spec, repeatable
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/local/laya-go-launcher/internal/backend"
	"github.com/local/laya-go-launcher/internal/backends"
	"github.com/local/laya-go-launcher/internal/inference"
	"github.com/local/laya-go-launcher/internal/tokenizer"
)

const (
	defaultONNX = `C:\Users\lingxin\Documents\laya-trt\onnx\laya_ctx8192.onnx`
	// fusedONNX is defaultONNX after tools/optimize_onnx.py --fp16. It is what
	// the GPU presets run when present; the CPU preset keeps the fp32 graph,
	// because fp16 is slower than fp32 on the CPU provider.
	fusedONNX     = `C:\Users\lingxin\Documents\laya-trt\onnx\laya_ctx8192.opt.fp16.onnx`
	// multiProfileEngine is bench/build-engines.ps1 -MultiProfile: batch <= 8 up
	// to 512 tokens on profile 0, batch 1 up to 8192 on profile 1. It is what the
	// trt preset runs when present; legacyEngine is the single-profile batch-1
	// plan it replaces.
	multiProfileEngine = `C:\Users\lingxin\Documents\laya-trt\engines\laya_s8192_fp16_p2.engine`
	legacyEngine       = `C:\Users\lingxin\Documents\laya-trt\engines\laya_s8192_fp16_b1.engine`
	minP99Samples      = 100
)

// defaultEngine is the plan the trt preset uses.
func defaultEngine() string {
	if _, err := os.Stat(multiProfileEngine); err == nil {
		return multiProfileEngine
	}
	return legacyEngine
}

// gpuONNX is the graph the GPU presets use: the fused fp16 graph when it has
// been generated, else the plain export.
func gpuONNX() string {
	if _, err := os.Stat(fusedONNX); err == nil {
		return fusedONNX
	}
	return defaultONNX
}

// target is one engine under test.
type target struct {
	label, kind, provider, path string
}

// presets are selectable by name with -engines.
var presets = map[string]target{
	"trt":           {"trt-fp16-s8192", "tensorrt", "", defaultEngine()},
	"trt-b1":        {"trt-fp16-s8192-b1", "tensorrt", "", legacyEngine},
	"onnx":          {"onnx-cuda", "onnx", "cuda", gpuONNX()},
	"onnx-cuda":     {"onnx-cuda", "onnx", "cuda", gpuONNX()},
	"onnx-fp32":     {"onnx-cuda-fp32", "onnx", "cuda", defaultONNX},
	"onnx-cpu":      {"onnx-cpu", "onnx", "cpu", defaultONNX},
	"onnx-directml": {"onnx-directml", "onnx", "directml", defaultONNX},
}

type multiFlag []string

func (m *multiFlag) String() string     { return strings.Join(*m, ";") }
func (m *multiFlag) Set(v string) error { *m = append(*m, v); return nil }

// parseSpec reads "label=kind:provider:path" or "label=tensorrt:path".
func parseSpec(s string) (target, error) {
	label, rest, ok := strings.Cut(s, "=")
	if !ok {
		return target{}, fmt.Errorf("engine spec %q: want label=kind:[provider:]path", s)
	}
	kind, rest, ok := strings.Cut(rest, ":")
	if !ok {
		return target{}, fmt.Errorf("engine spec %q: missing path", s)
	}
	t := target{label: label, kind: kind, path: rest}
	if kind == "onnx" {
		// provider is optional; a Windows drive letter ("C:") is not a provider.
		if p, path, ok := strings.Cut(rest, ":"); ok && len(p) > 1 {
			t.provider, t.path = p, path
		}
	}
	return t, nil
}

// ---------------------------------------------------------------- case set

var categories = map[string]any{
	"billing":   "invoices, payments, refunds",
	"technical": "bugs, outages, integrations",
	"sales":     "pricing, demos, new purchases",
	"hr":        "hiring, leave, payroll",
}

var questions = map[string]inference.QuestionSpec{
	"dept":   {Type: "choice", Instructions: "Which team should handle `message`?", Criteria: categories},
	"refund": {Type: "noul", Instructions: "Does the customer ask for money back?"},
}

type benchCase struct {
	id, dept string
	refund   bool
	text     string
}

var cases = []benchCase{
	{"en_billing", "billing", true, "I was charged twice for invoice 4411, please refund it today."},
	{"en_technical", "technical", false, "The upload API returns 500 errors on every request since this morning."},
	{"en_sales", "sales", false, "What does the enterprise plan cost for 200 seats, and can we get a demo?"},
	{"en_hr", "hr", false, "How many vacation days do I have left this year, and how do I request leave?"},
	{"en_billing_long", "billing", true, strings.Repeat("We were billed twice for March. Please refund the duplicate today. ", 40)},
}

// ---------------------------------------------------------------- stats

type stats struct {
	n                             int
	p50, p95, p99, mean, min, max float64
}

// percentile is linear interpolation, numpy's default method.
func percentile(sorted []float64, q float64) float64 {
	n := len(sorted)
	if n == 0 {
		return math.NaN()
	}
	pos := float64(n-1) * q
	lo, hi := int(math.Floor(pos)), int(math.Ceil(pos))
	if lo == hi {
		return sorted[lo]
	}
	return sorted[lo]*(float64(hi)-pos) + sorted[hi]*(pos-float64(lo))
}

func summarize(xs []float64) stats {
	if len(xs) == 0 {
		nan := math.NaN()
		return stats{0, nan, nan, nan, nan, nan, nan}
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	sum := 0.0
	for _, v := range s {
		sum += v
	}
	return stats{len(s), percentile(s, .50), percentile(s, .95), percentile(s, .99),
		sum / float64(len(s)), s[0], s[len(s)-1]}
}

// ---------------------------------------------------------------- run

type engineResult struct {
	t        target
	err      error
	loadS    float64
	wall     []float64 // Predict wall time
	infer    []float64 // Timing.InferenceMS
	perCase  map[string]stats
	correct  int
	answered int
}

func runTarget(t target, tok *tokenizer.Tokenizer, cfgPath string, warmup, reps int) (res engineResult) {
	res = engineResult{t: t, perCase: map[string]stats{}}
	kind, err := backend.ParseKind(t.kind)
	if err != nil {
		res.err = err
		return
	}
	cfg := backends.Config{Kind: kind, Provider: t.provider}
	model := backends.NewSwitcher(cfg)
	defer model.Close()

	svc, err := inference.NewService(inference.Config{Backend: model, Tokenizer: tok})
	if err != nil {
		res.err = err
		return
	}
	t0 := time.Now()
	info, note, err := model.LoadWith(context.Background(), cfg,
		backend.Options{Path: t.path, Contexts: 1, Provider: t.provider})
	if err != nil {
		res.err = fmt.Errorf("load: %w", err)
		return
	}
	res.loadS = time.Since(t0).Seconds()
	if note != "" {
		log.Printf("   note: %s", note)
	}
	if cfgPath != "" {
		if err := svc.LoadModelConfigFrom(cfgPath); err != nil {
			log.Printf("   ! model config: %v (temperature 1.0)", err)
		}
	} else {
		svc.AdaptToEngine()
	}
	ml, hl, pl := svc.Budgets()
	log.Printf("   loaded %s [%s] in %.1fs  budgets max=%d head=%d pad=%d",
		filepath.Base(info.Path), info.Backend, res.loadS, ml, hl, pl)

	ctx := context.Background()
	for _, c := range cases {
		req := inference.Request{State: map[string]any{"message": c.text}, Questions: questions}
		resp, err := svc.Predict(ctx, req)
		if err != nil {
			log.Printf("   %-16s ERROR %v", c.id, err)
			continue
		}
		d, r := resp.Answers["dept"], resp.Answers["refund"]
		ok := d.Choice == c.dept && (r.NoUL >= 0.5) == c.refund
		res.answered++
		if ok {
			res.correct++
		}
		for i := 0; i < warmup; i++ {
			_, _ = svc.Predict(ctx, req)
		}
		var wall []float64
		for i := 0; i < reps; i++ {
			ts := time.Now()
			rr, err := svc.Predict(ctx, req)
			if err != nil {
				log.Printf("   %-16s rep %d error: %v", c.id, i, err)
				break
			}
			wall = append(wall, float64(time.Since(ts).Microseconds())/1000)
			res.infer = append(res.infer, rr.Timing.InferenceMS)
		}
		res.wall = append(res.wall, wall...)
		st := summarize(wall)
		res.perCase[c.id] = st
		verdict := "PASS"
		if !ok {
			verdict = "FAIL"
		}
		log.Printf("   %-16s tok=%-5d dept=%-9s(want %-9s p=%.2f) refund=%.3f(want %-5v) %s | p50 %.2f p95 %.2f p99 %.2f ms",
			c.id, resp.Usage.InputTokens, d.Choice, c.dept, d.Confidence, r.NoUL, c.refund, verdict,
			st.p50, st.p95, st.p99)
	}
	return
}

// ---------------------------------------------------------------- main

func main() {
	var specs multiFlag
	names := flag.String("engines", "trt,onnx", "comma-separated presets: trt, onnx(-cuda), onnx-cpu, onnx-directml")
	flag.Var(&specs, "engine", "custom engine label=kind:[provider:]path (repeatable, added to -engines)")
	tokPath := flag.String("tokenizer", "", "tokenizer.json (default: HF cache)")
	cfgPath := flag.String("model-config", "", "rl_agent_config.json (default: HF cache)")
	reps := flag.Int("reps", 50, "timed Predict calls per case")
	warmup := flag.Int("warmup", 5, "untimed calls per case")
	logPath := flag.String("log", "", "also write the log to this file")
	flag.Parse()

	log.SetFlags(log.Ltime)
	if *logPath != "" {
		f, err := os.Create(*logPath)
		if err != nil {
			log.Fatalf("log: %v", err)
		}
		defer f.Close()
		log.SetOutput(io.MultiWriter(os.Stdout, f))
	}

	var targets []target
	for _, n := range strings.Split(*names, ",") {
		if n = strings.TrimSpace(n); n == "" {
			continue
		}
		t, ok := presets[n]
		if !ok {
			log.Fatalf("unknown preset %q", n)
		}
		targets = append(targets, t)
	}
	for _, s := range specs {
		t, err := parseSpec(s)
		if err != nil {
			log.Fatal(err)
		}
		targets = append(targets, t)
	}
	if len(targets) == 0 {
		log.Fatal("no engines selected")
	}

	tok, err := findTokenizer(*tokPath)
	if err != nil {
		log.Fatalf("tokenizer: %v", err)
	}
	if *cfgPath == "" {
		*cfgPath = hfLatest("rl_agent_config.json")
	}
	log.Printf("autobench: %d engine(s), %d cases, warmup %d, reps %d, model-config %q",
		len(targets), len(cases), *warmup, *reps, *cfgPath)

	var results []engineResult
	for _, t := range targets {
		log.Printf("== %s  (%s %s) %s", t.label, t.kind, t.provider, t.path)
		if _, err := os.Stat(t.path); err != nil {
			results = append(results, engineResult{t: t, err: err})
			log.Printf("   ! %v", err)
			continue
		}
		r := runTarget(t, tok, *cfgPath, *warmup, *reps)
		if r.err != nil {
			log.Printf("   ! %v", r.err)
		}
		results = append(results, r)
	}

	log.Printf("== summary (ms, Predict wall time; infer = model-reported inference_ms)")
	log.Printf("%-18s %6s %8s %8s %8s %8s %8s %8s | %8s %8s %8s | %s",
		"engine", "n", "p50", "p95", "p99", "mean", "min", "max", "inf_p50", "inf_p95", "inf_p99", "accuracy")
	for _, r := range results {
		if r.err != nil {
			log.Printf("%-18s ERROR %v", r.t.label, r.err)
			continue
		}
		w, i := summarize(r.wall), summarize(r.infer)
		log.Printf("%-18s %6d %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f | %8.2f %8.2f %8.2f | %d/%d",
			r.t.label, w.n, w.p50, w.p95, w.p99, w.mean, w.min, w.max, i.p50, i.p95, i.p99, r.correct, r.answered)
		if w.n > 0 && w.n < minP99Samples {
			log.Printf("   ! only %d samples; P99 needs >= %d, raise -reps", w.n, minP99Samples)
		}
	}
}

func hfLatest(rel string) string {
	home, _ := os.UserHomeDir()
	m, _ := filepath.Glob(filepath.Join(home, ".cache", "huggingface", "hub",
		"models--convaiinnovations--laya", "snapshots", "*", rel))
	sort.Strings(m)
	if len(m) == 0 {
		return ""
	}
	return m[len(m)-1]
}

func findTokenizer(explicit string) (*tokenizer.Tokenizer, error) {
	if explicit == "" {
		explicit = hfLatest(filepath.Join("tokenizer", "tokenizer.json"))
	}
	if explicit == "" {
		return nil, fmt.Errorf("no tokenizer.json found (pass -tokenizer)")
	}
	return tokenizer.Load(explicit)
}
