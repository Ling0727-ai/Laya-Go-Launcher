package modelstore

import (
	"path/filepath"
	"sort"
	"strings"
)

// Attempt is one step of an auto load: which file, on which kernel/provider.
type Attempt struct {
	Step     string `json:"step"`     // tensorrt, onnx-cuda, onnx-directml, onnx-cpu
	Path     string `json:"path"`     // model file
	Backend  string `json:"backend"`  // tensorrt | onnx
	Provider string `json:"provider"` // cuda | directml | cpu | ""
	Reason   string `json:"reason"`   // why this file was picked
}

// PlanOptions steer candidate ranking.
type PlanOptions struct {
	// Seq is the context the model must accept (e.g. 8192).
	Seq int
	// Precision preferred for TensorRT plans: fp16 or fp32.
	Precision string
	// Steps is the fallback chain.
	Steps []string
	// Pinned, when set, is tried first on the step matching its format.
	Pinned string
	// PerStep caps how many files are tried per step (default 2).
	PerStep int
}

// Plan orders the attempts for an auto load.
//
// Each step contributes its best candidates. A plan that cannot hold opts.Seq
// is never chosen automatically: running a request of 8000 tokens on a 512 plan
// would silently truncate the input, which is worse than falling back.
func Plan(models []Model, opts PlanOptions) []Attempt {
	if opts.PerStep <= 0 {
		opts.PerStep = 2
	}
	var out []Attempt
	seen := map[string]bool{}
	add := func(a Attempt) {
		k := a.Step + "|" + strings.ToLower(a.Path)
		if seen[k] {
			return
		}
		seen[k] = true
		out = append(out, a)
	}
	for _, step := range opts.Steps {
		backend, provider := stepKernel(step)
		if opts.Pinned != "" && formatFor(step) == formatOf(opts.Pinned) {
			add(Attempt{Step: step, Path: opts.Pinned, Backend: backend, Provider: provider, Reason: "configured engine_path"})
		}
		for i, m := range Rank(models, step, opts) {
			if i >= opts.PerStep {
				break
			}
			add(Attempt{Step: step, Path: m.Path, Backend: backend, Provider: provider, Reason: reasonFor(m, opts)})
		}
	}
	return out
}

// Rank returns the models usable for one step, best first.
func Rank(models []Model, step string, opts PlanOptions) []Model {
	want := formatFor(step)
	var c []Model
	for _, m := range models {
		if m.Format != want {
			continue
		}
		if m.Format == FormatEngine && m.SeqMax > 0 && m.SeqMax < opts.Seq {
			continue
		}
		if m.Format == FormatEngine && m.SeqMax == 0 {
			// An unlabelled plan might be anything; never auto-pick it.
			continue
		}
		if m.Format == FormatONNX && m.SeqMax > 0 && m.SeqMax < opts.Seq {
			continue
		}
		c = append(c, m)
	}
	sort.SliceStable(c, func(i, j int) bool {
		si, sj := score(c[i], step, opts), score(c[j], step, opts)
		if si != sj {
			return si > sj
		}
		return c[i].Modified.After(c[j].Modified)
	})
	return c
}

// score ranks candidates within a step. Measured on this project's bench
// (bench/autobench.opt2.log): the two-profile plan runs a 123-token request in
// ~4 ms against ~22 ms for the batch-1 8192 plan, and the fused fp16 graph is the
// fastest ONNX CUDA graph.
func score(m Model, step string, opts PlanOptions) int {
	s := 0
	switch m.Format {
	case FormatEngine:
		if m.MultiProfile {
			s += 100
		}
		if m.Precision == opts.Precision {
			s += 40
		}
		// Tightest ceiling that still fits: less reserved activation memory.
		if m.SeqMax == opts.Seq {
			s += 20
		}
		if m.Origin != "name" {
			s += 5
		}
	case FormatONNX:
		if m.SeqMax >= opts.Seq {
			s += 10
		}
		switch step {
		case "onnx-cuda":
			// Fused fp16 > fused fp32 > plain.
			if m.Optimized {
				s += 50
			}
			if m.Precision == "fp16" {
				s += 30
			}
		case "onnx-directml":
			// DirectML covers the standard opset best; the fused graph's
			// com.microsoft contrib ops are a CUDA/CPU speciality.
			if !m.Optimized {
				s += 50
			}
			if m.Precision == "fp16" {
				s += 5
			}
		case "onnx-cpu":
			// CPU has no fast fp16 path; fused fp32 is quickest.
			if m.Precision != "fp16" {
				s += 50
			}
			if m.Optimized {
				s += 20
			}
		}
	}
	return s
}

// ConversionSource picks the ONNX graph to build a TensorRT plan from: a plain
// (non-fused) graph that covers seq. Fused graphs carry ORT contrib ops that
// the TensorRT parser does not implement.
func ConversionSource(models []Model, seq int) (Model, bool) {
	var best *Model
	for i := range models {
		m := &models[i]
		if m.Format != FormatONNX || m.Optimized {
			continue
		}
		if m.SeqMax > 0 && m.SeqMax < seq {
			continue
		}
		if best == nil || m.Modified.After(best.Modified) {
			best = m
		}
	}
	if best == nil {
		return Model{}, false
	}
	return *best, true
}

// HasEngineFor reports whether a plan for seq/precision is already on disk.
func HasEngineFor(models []Model, seq int, precision string) (Model, bool) {
	r := Rank(models, "tensorrt", PlanOptions{Seq: seq, Precision: precision})
	if len(r) == 0 {
		return Model{}, false
	}
	return r[0], true
}

func stepKernel(step string) (string, string) {
	switch step {
	case "tensorrt":
		return "tensorrt", ""
	case "onnx-cuda":
		return "onnx", "cuda"
	case "onnx-directml":
		return "onnx", "directml"
	case "onnx-cpu":
		return "onnx", "cpu"
	}
	return "", ""
}

func formatFor(step string) Format {
	if step == "tensorrt" {
		return FormatEngine
	}
	return FormatONNX
}

func formatOf(path string) Format {
	if strings.EqualFold(filepath.Ext(path), ".onnx") {
		return FormatONNX
	}
	return FormatEngine
}

func reasonFor(m Model, opts PlanOptions) string {
	var parts []string
	if m.SeqMax > 0 {
		parts = append(parts, "seq≤"+itoa(m.SeqMax))
	}
	if m.MultiProfile {
		parts = append(parts, "two-profile")
	}
	if m.Optimized {
		parts = append(parts, "fused")
	}
	if m.Precision != "" {
		parts = append(parts, m.Precision)
	}
	return strings.Join(parts, ", ")
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}
