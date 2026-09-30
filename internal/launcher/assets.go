package launcher

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/local/laya-go-launcher/internal/config"
	"github.com/local/laya-go-launcher/internal/modelstore"
	"github.com/local/laya-go-launcher/internal/tokenizer"
)

// LoadTokenizer finds tokenizer.json: explicit path, the Hugging Face cache,
// then tokenizer/ and models/laya/tokenizer/ under the working directory.
func LoadTokenizer(explicit string) (*tokenizer.Tokenizer, string, error) {
	if explicit != "" {
		tk, err := tokenizer.Load(explicit)
		return tk, explicit, err
	}
	var candidates []string
	if p := newestHF(filepath.Join("tokenizer", "tokenizer.json")); p != "" {
		candidates = append(candidates, p)
	}
	candidates = append(candidates,
		filepath.Join("tokenizer", "tokenizer.json"),
		filepath.Join("models", "laya", "tokenizer", "tokenizer.json"))
	var lastErr error
	for _, c := range candidates {
		tk, err := tokenizer.Load(c)
		if err == nil {
			return tk, c, nil
		}
		lastErr = err
	}
	return nil, "", fmt.Errorf("no tokenizer.json found (set tokenizer_path, or run: huggingface-cli download convaiinnovations/laya --include \"tokenizer/*\" rl_agent_config.json): %w", lastErr)
}

// newestHF returns rel inside the newest laya snapshot of the HF cache.
func newestHF(rel string) string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	pattern := filepath.Join(home, ".cache", "huggingface", "hub",
		"models--convaiinnovations--laya", "snapshots", "*", rel)
	matches, _ := filepath.Glob(pattern)
	best := ""
	for _, m := range matches {
		if m > best {
			best = m
		}
	}
	return best
}

// ConvertRequest overrides parts of the automatic build.
type ConvertRequest struct {
	Source     string `json:"source"`
	Seq        int    `json:"seq"`
	Precision  string `json:"precision"`
	BuilderOpt int    `json:"builder_opt"`
}

// StartConversion builds a TensorRT plan in the background. Empty fields come
// from the configuration and the catalog.
func (l *Launcher) StartConversion(req ConvertRequest) (modelstore.Job, error) {
	cfg := l.Config()
	seq := req.Seq
	if seq <= 0 {
		seq = cfg.ModelSeq
	}
	precision := req.Precision
	if precision == "" {
		precision = cfg.Precision
	}
	src := req.Source
	if src == "" {
		m, ok := modelstore.ConversionSource(l.Catalog.Models(), seq)
		if !ok {
			return modelstore.Job{}, fmt.Errorf("no plain (non-fused) .onnx graph covering seq %d was found; fused *.opt.onnx graphs cannot be built by TensorRT", seq)
		}
		src = m.Path
	}
	return l.Converter.Start(modelstore.BuildSpec{
		Source:     src,
		OutDir:     l.engineDir(cfg),
		Seq:        seq,
		Precision:  precision,
		BuilderOpt: req.BuilderOpt,
		Trtexec:    cfg.TrtexecPath,
	})
}

// engineDir is where built plans go: configured, else the first existing
// engines/ directory in the catalog, else ./engines.
func (l *Launcher) engineDir(cfg config.Config) string {
	if cfg.EngineDir != "" {
		return cfg.EngineDir
	}
	for _, d := range l.Catalog.Dirs() {
		if strings.EqualFold(filepath.Base(d), "engines") {
			return d
		}
	}
	return "engines"
}

// maybeConvert starts a build when auto_convert is on, TensorRT and trtexec are
// usable, and no loadable plan for model_seq exists. A plan that exists but
// failed to deserialize (built by another TensorRT version or for another GPU)
// is rebuilt once.
func (l *Launcher) maybeConvert() {
	cfg := l.Config()
	if !cfg.AutoConvert || !cfg.AutoChain() {
		return
	}
	st := l.State()
	if st.Step == config.StepTensorRT && st.Phase == "ready" {
		return
	}
	if _, running := l.Converter.Running(); running {
		return
	}
	if why := l.probeStep(cfg, config.StepTensorRT); why != "" {
		return
	}
	if _, err := modelstore.FindTrtexec(cfg.TrtexecPath); err != nil {
		return
	}
	// Did TensorRT get a real chance? A plan that exists and was never tried
	// (loader busy, cancelled) is not a reason to rebuild.
	triedTRT, trtBroken := false, false
	for _, a := range st.Attempts {
		if a.Step != config.StepTensorRT || a.Skipped {
			continue
		}
		triedTRT = true
		// Out of memory is not fixed by a rebuild; a version/arch mismatch is.
		if !a.OK && !strings.Contains(strings.ToLower(a.Error), "memory") {
			trtBroken = true
		}
	}
	if _, has := modelstore.HasEngineFor(l.Catalog.Models(), cfg.ModelSeq, cfg.Precision); has {
		if !triedTRT || !trtBroken {
			return
		}
		out := filepath.Join(l.engineDir(cfg), modelstore.BuildSpec{Seq: cfg.ModelSeq, Precision: cfg.Precision}.OutputName())
		key := strings.ToLower(out)
		l.mu.Lock()
		done := l.rebuilt[key]
		l.rebuilt[key] = true
		l.mu.Unlock()
		if done {
			return
		}
	}
	job, err := l.StartConversion(ConvertRequest{})
	if err != nil {
		log.Printf("auto-convert: %v", err)
		return
	}
	log.Printf("auto-convert: building %s from %s (log: %s)", job.Output, job.Spec.Source, job.Log)
}

// onConversionDone rescans and, when the process is on a slower fallback,
// switches to the new plan.
func (l *Launcher) onConversionDone(job modelstore.Job) {
	l.Catalog.Invalidate()
	if job.State != "done" {
		log.Printf("auto-convert: %s %s: %s", job.Output, job.State, job.Error)
		return
	}
	log.Printf("auto-convert: built %s in %.0fs", job.Output, job.Seconds)
	cfg := l.Config()
	st := l.State()
	if !cfg.AutoChain() || (st.Phase == "ready" && st.Step == config.StepTensorRT) {
		return
	}
	go func() {
		if _, err := l.AutoLoad(bgContext()); err != nil {
			log.Printf("auto-convert: reload failed: %v", err)
		}
	}()
}
