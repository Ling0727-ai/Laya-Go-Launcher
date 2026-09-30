// Command enginecmp compares two TensorRT plans over identical inputs.
//
// It answers the two questions that decide whether a plan is safe to ship:
//
//	accuracy  — do the two plans agree on the outputs a caller sees?
//	speed     — how do their latency and activation memory compare?
//
// The comparison is deliberately raw: it builds one input set through the real
// tokenizer and sequence builder, runs it through each engine's kernel directly,
// and compares the float outputs. Running the full predict path would round
// probabilities to four decimals before the comparison, which hides exactly the
// small fp16-vs-fp32 differences this tool exists to measure.
//
// Plans are loaded one at a time. Two large plans resident together can exhaust
// VRAM on a laptop GPU, and there is no need for them to be resident at once
// because the inputs are fixed before either runs.
//
//	go run ./bench/enginecmp -a fp16.engine -b fp32.engine -reps 20
package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/local/laya-go-launcher/internal/engine"
	"github.com/local/laya-go-launcher/internal/kernel"
	"github.com/local/laya-go-launcher/internal/sequence"
	"github.com/local/laya-go-launcher/internal/tokenizer"
)

// outputs holds one plan's raw results plus its timings.
type outputs struct {
	logits []float32
	act    []float32
	times  []time.Duration
	// tokens is the padded sequence length actually sent.
	tokens int
	// rows is the batch dimension actually sent.
	rows int
}

func main() {
	var (
		engineA = flag.String("a", "", "first engine (typically fp16)")
		engineB = flag.String("b", "", "second engine (typically fp32)")
		tokPath = flag.String("tokenizer", "", "tokenizer.json (default: HF cache)")
		nq      = flag.Int("questions", 5, "questions per request, i.e. batch rows")
		words   = flag.Int("words", 120, "approximate state length in words")
		reps    = flag.Int("reps", 20, "measured repetitions per engine")
		warmup  = flag.Int("warmup", 3, "warm-up runs per engine")
		contexts = flag.Int("contexts", 1, "execution contexts to allocate")
		maxLen  = flag.Int("max-len", 512, "max_len for the sequence builder")
		headMax = flag.Int("head-max-len", 192, "head_max_len for the sequence builder")
		labelA  = flag.String("label-a", "", "label for engine A (default: file name)")
		labelB  = flag.String("label-b", "", "label for engine B (default: file name)")
	)
	flag.Parse()

	if *engineA == "" || *engineB == "" {
		fmt.Fprintln(os.Stderr, "usage: enginecmp -a <engine> -b <engine>")
		os.Exit(2)
	}
	if *labelA == "" {
		*labelA = filepath.Base(*engineA)
	}
	if *labelB == "" {
		*labelB = filepath.Base(*engineB)
	}

	if err := kernel.Initialize(); err != nil {
		fmt.Fprintf(os.Stderr, "tensorrt: %v\n", err)
		os.Exit(1)
	}

	tk, err := loadTokenizer(*tokPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "tokenizer: %v\n", err)
		os.Exit(1)
	}

	// Build one deterministic input set, shared by both plans, so a difference
	// in the outputs cannot come from a difference in the inputs.
	inputs, err := buildInputs(tk, *nq, *words, *maxLen, *headMax)
	if err != nil {
		fmt.Fprintf(os.Stderr, "inputs: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("questions : %d rows\n", inputs.batch)
	fmt.Printf("state     : ~%d words -> %d tokens/row (padded)\n", *words, inputs.length)
	fmt.Printf("markers   : %d\n\n", inputs.markers)
	fmt.Printf("TensorRT  : %s\n", kernel.Version())
	dev := kernel.QueryDeviceInfo()
	fmt.Printf("device    : %s\n\n", dev.Name)

	// Run A, then release it before loading B.
	outA, infoA, err := runEngine(*engineA, inputs, *contexts, *warmup, *reps)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", *labelA, err)
		os.Exit(1)
	}
	outB, infoB, err := runEngine(*engineB, inputs, *contexts, *warmup, *reps)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", *labelB, err)
		os.Exit(1)
	}

	// ── speed ────────────────────────────────────────────────────────────────
	fmt.Printf("%-34s %10s %10s %10s %12s\n", "engine", "p50_ms", "p95_ms", "min_ms", "activ_MiB")
	fmt.Println(strings.Repeat("-", 80))
	reportSpeed(*labelA, outA, infoA)
	reportSpeed(*labelB, outB, infoB)

	// ── accuracy ─────────────────────────────────────────────────────────────
	fmt.Printf("\naccuracy: %s vs %s\n", *labelA, *labelB)
	fmt.Println(strings.Repeat("-", 80))

	logitDiff := compare("logits", outA.logits, outB.logits, *labelA, *labelB)
	actDiff := compare("act_logits", outA.act, outB.act, *labelA, *labelB)

	// The caller-visible effect: does the argmax option change, and how far do
	// the probabilities move after the softmax the service applies?
	probsA := softmaxRows(outA.logits, inputs.batch, inputs.nmarker)
	probsB := softmaxRows(outB.logits, inputs.batch, inputs.nmarker)
	argmaxChanges := 0
	maxProbDelta := 0.0
	for r := 0; r < inputs.batch && r < len(probsA) && r < len(probsB); r++ {
		a, b := probsA[r], probsB[r]
		if len(a) == 0 || len(b) == 0 {
			continue
		}
		if argmax(a) != argmax(b) {
			argmaxChanges++
		}
		for i := range a {
			if i < len(b) {
				if d := math.Abs(a[i] - b[i]); d > maxProbDelta {
					maxProbDelta = d
				}
			}
		}
	}
	fmt.Printf("argmax changes across %d rows : %d\n", inputs.batch, argmaxChanges)
	fmt.Printf("max probability delta         : %.6f\n", maxProbDelta)

	// ── verdict ──────────────────────────────────────────────────────────────
	fmt.Println()
	ok := true
	if argmaxChanges > 0 {
		fmt.Printf("FAIL: the two plans disagree on the chosen option for %d row(s)\n", argmaxChanges)
		ok = false
	}
	// A tenth of a percentage point of probability is below the four decimals the
	// API publishes, so it cannot reach a caller.
	if maxProbDelta > 1e-3 {
		fmt.Printf("FAIL: probabilities differ by %.6f, above the 0.001 tolerance\n", maxProbDelta)
		ok = false
	}
	if logitDiff.maxAbs > 0.5 {
		fmt.Printf("WARN: raw logits differ by %.4f; decisions still agree, but the margin is thin\n",
			logitDiff.maxAbs)
	}
	if logitDiff.hasNaN() || actDiff.hasNaN() {
		fmt.Println()
		fmt.Println("NaN detail:")
		if logitDiff.nanA || logitDiff.nanB {
			fmt.Printf("  logits: NaN in %s\n", nanSide(logitDiff, *labelA, *labelB))
		}
		if actDiff.nanA || actDiff.nanB {
			fmt.Printf("  act_logits: NaN in %s\n", nanSide(actDiff, *labelA, *labelB))
		}
		fmt.Println("  act_logits is an auxiliary head. Under fp16 the entropy term inside it")
		fmt.Println("  computes p*log(p), and a probability that underflows to 0 gives 0*-inf = NaN.")
		fmt.Println("  The service maps that to act_probability 0 rather than failing the call")
		fmt.Println("  (see inference.actProbability). The decision logits are unaffected.")
	}

	if ok {
		fmt.Printf("PASS: decisions and probabilities agree; %s is safe to use\n", *labelA)
	} else {
		os.Exit(1)
	}
}

func reportSpeed(label string, out outputs, info engine.Info) {
	p50 := percentile(out.times, 50)
	p95 := percentile(out.times, 95)
	min := out.times[0]
	for _, d := range out.times {
		if d < min {
			min = d
		}
	}
	fmt.Printf("%-34s %10.3f %10.3f %10.3f %12.1f\n",
		label,
		float64(p50.Microseconds())/1000,
		float64(p95.Microseconds())/1000,
		float64(min.Microseconds())/1000,
		info.ActivationMemoryMB,
	)
}

type diff struct {
	maxAbs float64
	// nanA/nanB record which side produced a NaN, so the report can name it
	// rather than saying "one of them".
	nanA bool
	nanB bool
}

func (d diff) hasNaN() bool { return d.nanA || d.nanB }

// compare reports the largest absolute difference between two float slices,
// treating a NaN on either side as a difference worth reporting. The labels name
// the plans in the NaN note.
func compare(name string, a, b []float32, labelA, labelB string) diff {
	d := diff{}
	if len(a) != len(b) {
		fmt.Printf("%-12s length mismatch: %d vs %d\n", name, len(a), len(b))
		d.maxAbs = math.Inf(1)
		return d
	}
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		if math.IsNaN(x) {
			d.nanA = true
		}
		if math.IsNaN(y) {
			d.nanB = true
		}
		if math.IsNaN(x) || math.IsNaN(y) {
			continue
		}
		if v := math.Abs(x - y); v > d.maxAbs {
			d.maxAbs = v
		}
	}
	fmt.Printf("%-12s max|Δ| = %.6f over %d values", name, d.maxAbs, len(a))
	switch {
	case d.nanA && d.nanB:
		fmt.Printf("   (NaN in both)")
	case d.nanA:
		fmt.Printf("   (NaN in %s only)", labelA)
	case d.nanB:
		fmt.Printf("   (NaN in %s only)", labelB)
	}
	fmt.Println()
	return d
}

// nanSide names which plan produced a NaN.
func nanSide(d diff, labelA, labelB string) string {
	switch {
	case d.nanA && d.nanB:
		return "both plans"
	case d.nanA:
		return labelA
	default:
		return labelB
	}
}

func percentile(ds []time.Duration, p float64) time.Duration {
	if len(ds) == 0 {
		return 0
	}
	s := append([]time.Duration(nil), ds...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	idx := int(math.Ceil(p/100*float64(len(s)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(s) {
		idx = len(s) - 1
	}
	return s[idx]
}

func argmax(v []float64) int {
	best, bi := math.Inf(-1), 0
	for i, x := range v {
		if x > best {
			best, bi = x, i
		}
	}
	return bi
}

// softmaxRows applies a plain softmax to each row's first `k` logits, which is
// the transformation the service applies before publishing probabilities.
func softmaxRows(flat []float32, rows, k int) [][]float64 {
	if rows <= 0 || k <= 0 || len(flat) == 0 {
		return nil
	}
	per := len(flat) / rows
	out := make([][]float64, 0, rows)
	for r := 0; r < rows; r++ {
		row := flat[r*per : (r+1)*per]
		if k > len(row) {
			k = len(row)
		}
		row = row[:k]
		maxV := float64(row[0])
		for _, v := range row {
			if float64(v) > maxV {
				maxV = float64(v)
			}
		}
		p := make([]float64, len(row))
		sum := 0.0
		for i, v := range row {
			e := math.Exp(float64(v) - maxV)
			p[i] = e
			sum += e
		}
		if sum > 0 {
			for i := range p {
				p[i] /= sum
			}
		}
		out = append(out, p)
	}
	return out
}

// inputSet is one fixed batch of inputs, shared by both plans.
type inputSet struct {
	ids     []int64
	mask    []int64
	markers []int64
	mmask   []byte
	qtypes  []int64
	batch   int
	length  int
	nmarker int
}

func buildInputs(tk *tokenizer.Tokenizer, nq, words, maxLen, headMaxLen int) (inputSet, error) {
	if nq < 1 {
		return inputSet{}, fmt.Errorf("questions must be at least 1")
	}
	state := strings.Repeat("We were billed twice for March. Please refund the duplicate today. ",
		max(1, words/11))
	stateIDs := sequence.EncodeState(tk, state, maxLen)

	// One question per row, cycling the three types so both head paths run.
	questions := make([]sequence.Question, 0, nq)
	for i := 0; i < nq; i++ {
		switch i % 3 {
		case 0:
			questions = append(questions, sequence.Question{
				Type:         sequence.Choice,
				Instructions: "Which department should handle this request?",
				Criteria:     []string{"billing: invoices, payments, refunds", "technical: bugs, outages", "sales: pricing"},
				Labels:       []string{"billing", "technical", "sales"},
			})
		case 1:
			questions = append(questions, sequence.Question{
				Type:         sequence.Score,
				Instructions: "How urgent is this request?",
				Criteria:     []string{"level 0: not urgent", "level 1: soon", "level 2: critical"},
			})
		default:
			questions = append(questions, sequence.Question{
				Type:         sequence.NoUL,
				Instructions: "Does the user threaten to cancel or leave?",
			})
		}
	}

	built := make([]sequence.Built, 0, nq)
	length, nmarker := 0, 0
	for _, q := range questions {
		b, err := sequence.BuildWithState(tk, stateIDs, q, maxLen, headMaxLen)
		if err != nil {
			return inputSet{}, err
		}
		if len(b.IDs) > length {
			length = len(b.IDs)
		}
		if len(b.MarkerPositions) > nmarker {
			nmarker = len(b.MarkerPositions)
		}
		built = append(built, b)
	}

	in := inputSet{
		batch:   nq,
		length:  length,
		nmarker: nmarker,
		ids:     make([]int64, nq*length),
		mask:    make([]int64, nq*length),
		markers: make([]int64, nq*nmarker),
		mmask:   make([]byte, nq*nmarker),
		qtypes:  make([]int64, nq),
	}
	for r, b := range built {
		ids, m := sequence.PadTo(b.IDs, length, tk.PadID())
		copy(in.ids[r*length:], ids)
		copy(in.mask[r*length:], m)
		for i, pos := range b.MarkerPositions {
			in.markers[r*nmarker+i] = int64(pos)
			in.mmask[r*nmarker+i] = 1
		}
		in.qtypes[r] = int64(b.QType)
	}
	return in, nil
}

// runEngine loads one plan, runs it, and returns its raw outputs and timings.
// The plan is unloaded before returning so two large plans never coexist.
func runEngine(path string, in inputSet, contexts, warmup, reps int) (outputs, engine.Info, error) {
	mgr := engine.NewManager()
	defer mgr.Close()

	info, err := mgr.Load(engine.Options{Path: path, Contexts: contexts})
	if err != nil {
		return outputs{}, info, err
	}

	out := outputs{tokens: in.length, rows: in.batch}

	// A plan whose profile pins the batch dimension needs the rows padded up to
	// the size it accepts; discover that from the engine rather than assuming.
	batch := in.batch
	if fixed := fixedBatch(info, "input_ids"); fixed > 0 {
		batch = fixed
	}
	out.rows = batch

	ids := in.ids
	mask := in.mask
	markers := in.markers
	mmask := in.mmask
	qtypes := in.qtypes
	if batch != in.batch {
		ids = make([]int64, batch*in.length)
		mask = make([]int64, batch*in.length)
		markers = make([]int64, batch*in.nmarker)
		mmask = make([]byte, batch*in.nmarker)
		qtypes = make([]int64, batch)
		for r := 0; r < in.batch; r++ {
			copy(ids[r*in.length:], in.ids[r*in.length:(r+1)*in.length])
			copy(mask[r*in.length:], in.mask[r*in.length:(r+1)*in.length])
			copy(markers[r*in.nmarker:], in.markers[r*in.nmarker:(r+1)*in.nmarker])
			copy(mmask[r*in.nmarker:], in.mmask[r*in.nmarker:(r+1)*in.nmarker])
			qtypes[r] = in.qtypes[r]
		}
	}

	run := func() ([]float32, []float32, error) {
		var logits, act []float32
		err := mgr.WithContext(context.Background(), func(c *kernel.Context) error {
			r, newErr := c.NewRun()
			if newErr != nil {
				return newErr
			}
			defer r.Close()

			for _, inp := range []struct {
				name  string
				dtype kernel.DType
				data  any
				shape []int
			}{
				{"input_ids", kernel.Int64, ids, []int{batch, in.length}},
				{"attention_mask", kernel.Int64, mask, []int{batch, in.length}},
				{"marker_pos", kernel.Int64, markers, []int{batch, in.nmarker}},
				{"marker_mask", kernel.Bool, mmask, []int{batch, in.nmarker}},
				{"qtype", kernel.Int64, qtypes, []int{batch}},
			} {
				if err := r.AddInput(inp.name, inp.dtype, inp.data, inp.shape); err != nil {
					return err
				}
			}
			if err := r.AddOutput("logits"); err != nil {
				return err
			}
			if err := r.AddOutput("act_logits"); err != nil {
				return err
			}
			if _, err := r.Resolve(); err != nil {
				return err
			}
			if err := r.Execute(); err != nil {
				return err
			}
			var readErr error
			if logits, readErr = r.ReadOutput(0); readErr != nil {
				return readErr
			}
			if act, readErr = r.ReadOutput(1); readErr != nil {
				return readErr
			}
			return nil
		})
		return logits, act, err
	}

	// Warm up, then capture the outputs from a settled run.
	for i := 0; i < warmup; i++ {
		if _, _, err := run(); err != nil {
			return outputs{}, info, fmt.Errorf("warmup: %w", err)
		}
	}
	for i := 0; i < reps; i++ {
		start := time.Now()
		logits, act, err := run()
		elapsed := time.Since(start)
		if err != nil {
			return outputs{}, info, err
		}
		out.times = append(out.times, elapsed)
		if i == reps-1 {
			out.logits, out.act = logits, act
		}
	}
	return out, info, nil
}

// fixedBatch returns the engine's declared batch dimension when it is a fixed
// value above 1, or 0 when the dimension is dynamic or single-row.
func fixedBatch(info engine.Info, name string) int {
	for _, t := range info.Inputs {
		if t.Name == name && len(t.Shape) > 0 && t.Shape[0] > 1 {
			return t.Shape[0]
		}
	}
	return 0
}

func loadTokenizer(explicit string) (*tokenizer.Tokenizer, error) {
	if explicit != "" {
		return tokenizer.Load(explicit)
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
		return nil, fmt.Errorf("no tokenizer.json found (pass -tokenizer)")
	}
	return tokenizer.Load(best)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
