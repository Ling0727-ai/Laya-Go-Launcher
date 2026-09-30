// Package launcher owns the process-level model lifecycle shared by the
// headless server and the desktop shell: configuration, tokenizer, the kernel
// switcher, the automatic load with its fallback chain, model discovery and
// background TensorRT conversion.
//
// It has no GUI dependency, so the pure server build does not link Wails.
package launcher

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/local/laya-go-launcher/internal/backend"
	"github.com/local/laya-go-launcher/internal/backends"
	"github.com/local/laya-go-launcher/internal/config"
	"github.com/local/laya-go-launcher/internal/inference"
	"github.com/local/laya-go-launcher/internal/metrics"
	"github.com/local/laya-go-launcher/internal/modelstore"
	"github.com/local/laya-go-launcher/internal/ortbackend"
	"github.com/local/laya-go-launcher/internal/trtbackend"
)

// AttemptResult is one step of a load as it actually happened.
type AttemptResult struct {
	modelstore.Attempt
	OK      bool    `json:"ok"`
	Skipped bool    `json:"skipped,omitempty"`
	Error   string  `json:"error,omitempty"`
	Ms      float64 `json:"ms"`
}

// LoadState describes the most recent load.
type LoadState struct {
	// Phase is idle, loading, ready or failed.
	Phase       string          `json:"phase"`
	Mode        string          `json:"mode"` // auto, manual
	Step        string          `json:"step,omitempty"`
	Path        string          `json:"path,omitempty"`
	Backend     string          `json:"backend,omitempty"`
	Device      string          `json:"device,omitempty"`
	ModelConfig string          `json:"model_config,omitempty"`
	Note        string          `json:"note,omitempty"`
	Error       string          `json:"error,omitempty"`
	Attempts    []AttemptResult `json:"attempts"`
	StartedAt   time.Time       `json:"started_at"`
	EndedAt     time.Time       `json:"ended_at,omitempty"`
}

// Launcher is the shared runtime.
type Launcher struct {
	Version   string
	Switcher  *backends.Switcher
	Service   *inference.Service
	Metrics   *metrics.Recorder
	Catalog   *modelstore.Catalog
	Converter *modelstore.Converter

	cfgPath string
	tokPath string

	mu    sync.RWMutex
	cfg   config.Config
	state LoadState

	loadMu sync.Mutex
	// rebuilt remembers outputs already rebuilt once, so a plan that keeps
	// failing to deserialize cannot trigger an endless rebuild loop.
	rebuilt map[string]bool
}

// New builds the runtime: tokenizer, switcher, service. No model is loaded.
func New(cfg config.Config, cfgPath, version string) (*Launcher, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	tok, tokPath, err := LoadTokenizer(cfg.TokenizerPath)
	if err != nil {
		return nil, err
	}
	sw := backends.NewSwitcher(BackendConfig(cfg))
	svc, err := inference.NewService(inference.Config{
		Backend:    sw,
		Tokenizer:  tok,
		MaxLen:     cfg.MaxLen,
		HeadMaxLen: cfg.HeadMaxLen,
		PadLen:     cfg.PadLen,
	})
	if err != nil {
		return nil, err
	}
	if cfg.Diagnostics {
		trtbackend.SetDiagnostics(true)
	}
	l := &Launcher{
		Version:   version,
		Switcher:  sw,
		Service:   svc,
		Metrics:   metrics.New(),
		Converter: modelstore.NewConverter(),
		cfgPath:   cfgPath,
		tokPath:   tokPath,
		cfg:       cfg,
		state:     LoadState{Phase: "idle", Attempts: []AttemptResult{}},
		rebuilt:   map[string]bool{},
	}
	l.Catalog = modelstore.NewCatalog(l.searchDirs(cfg))
	l.Converter.OnDone = l.onConversionDone
	return l, nil
}

// Close releases the model and stops a running conversion.
func (l *Launcher) Close() {
	l.Converter.Cancel()
	l.Switcher.Close()
}

// BackendConfig maps the process config onto the kernel selection.
func BackendConfig(c config.Config) backends.Config {
	kind, err := backend.ParseKind(c.Backend)
	if err != nil {
		kind = backend.KindAuto
	}
	return backends.Config{
		Kind:        kind,
		Provider:    c.Provider,
		RuntimePath: c.ONNXRuntimePath,
		DLLPath:     c.ONNXBridgePath,
	}
}

// Config is the configuration in effect.
func (l *Launcher) Config() config.Config {
	l.mu.RLock()
	defer l.mu.RUnlock()
	c := l.cfg
	c.Fallback = append([]string(nil), c.Fallback...)
	c.ModelDirs = append([]string(nil), c.ModelDirs...)
	return c
}

// ConfigPath is the file UpdateConfig writes.
func (l *Launcher) ConfigPath() string { return l.cfgPath }

// TokenizerPath is the tokenizer.json in use.
func (l *Launcher) TokenizerPath() string { return l.tokPath }

// State is the most recent load.
func (l *Launcher) State() LoadState {
	l.mu.RLock()
	defer l.mu.RUnlock()
	s := l.state
	s.Attempts = append([]AttemptResult(nil), s.Attempts...)
	return s
}

// UpdateConfig validates, persists and installs a new configuration. It does
// not reload the model; call AutoLoad for that. Changing http_addr needs a
// restart, which the caller reports.
func (l *Launcher) UpdateConfig(next config.Config) error {
	cur := l.Config()
	if next.AdminToken == "***" {
		next.AdminToken = cur.AdminToken
	}
	if err := next.Validate(); err != nil {
		return err
	}
	if err := config.Write(l.cfgPath, next); err != nil {
		return err
	}
	l.mu.Lock()
	l.cfg = next
	l.mu.Unlock()
	l.Catalog.SetDirs(l.searchDirs(next))
	return nil
}

func (l *Launcher) searchDirs(c config.Config) []string {
	extra := append([]string(nil), c.ModelDirs...)
	if c.EngineDir != "" {
		extra = append(extra, c.EngineDir)
	}
	if c.EnginePath != "" {
		extra = append(extra, filepath.Dir(c.EnginePath))
	}
	return modelstore.DefaultDirs(extra)
}

// Models lists discovered models with the loaded one marked.
func (l *Launcher) Models() []ModelEntry {
	loaded := ""
	if info, ok := l.Switcher.Info(); ok {
		loaded = info.Path
	}
	models := l.Catalog.Models()
	out := make([]ModelEntry, 0, len(models))
	for _, m := range models {
		out = append(out, ModelEntry{Model: m, InUse: strings.EqualFold(m.Path, loaded)})
	}
	return out
}

// ModelEntry is a catalog model plus whether it is the resident one.
type ModelEntry struct {
	modelstore.Model
	InUse bool `json:"in_use"`
}

// PlanAttempts is what AutoLoad would try right now.
func (l *Launcher) PlanAttempts() []modelstore.Attempt {
	cfg := l.Config()
	return l.plan(cfg)
}

func (l *Launcher) plan(cfg config.Config) []modelstore.Attempt {
	models := l.Catalog.Models()
	if !cfg.AutoChain() {
		kind, _ := backend.ParseKind(cfg.Backend)
		step := config.StepTensorRT
		if kind == backend.KindONNX {
			switch strings.ToLower(cfg.Provider) {
			case "cpu":
				step = config.StepONNXCPU
			case "directml", "dml":
				step = config.StepONNXDirectML
			default:
				step = config.StepONNXCUDA
			}
		}
		return modelstore.Plan(models, modelstore.PlanOptions{
			Seq: cfg.ModelSeq, Precision: cfg.Precision, Steps: []string{step},
			Pinned: cfg.EnginePath, PerStep: 1,
		})
	}
	return modelstore.Plan(models, modelstore.PlanOptions{
		Seq: cfg.ModelSeq, Precision: cfg.Precision, Steps: cfg.Fallback, Pinned: cfg.EnginePath,
	})
}

// AutoLoad walks the fallback chain until one attempt loads, then starts a
// background TensorRT build when auto_convert is on and no usable plan exists.
func (l *Launcher) AutoLoad(ctx context.Context) (LoadState, error) {
	l.loadMu.Lock()
	defer l.loadMu.Unlock()
	cfg := l.Config()
	attempts := l.plan(cfg)
	st := l.begin("auto")
	defer func() { go l.maybeConvert() }()

	if len(attempts) == 0 {
		return l.fail(st, fmt.Errorf("no laya model for seq %d found in %s",
			cfg.ModelSeq, strings.Join(l.Catalog.Dirs(), ", ")))
	}
	probes := map[string]string{}
	for _, a := range attempts {
		if err := ctx.Err(); err != nil {
			return l.fail(st, err)
		}
		res := AttemptResult{Attempt: a}
		why, seen := probes[a.Step]
		if !seen {
			why = l.probeStep(cfg, a.Step)
			probes[a.Step] = why
		}
		if why != "" {
			res.Skipped, res.Error = true, why
			st.Attempts = append(st.Attempts, res)
			l.publish(st)
			continue
		}
		start := time.Now()
		info, note, err := l.loadOne(ctx, cfg, a)
		res.Ms = float64(time.Since(start).Microseconds()) / 1000
		if err != nil {
			res.Error = firstLine(err.Error())
			st.Attempts = append(st.Attempts, res)
			l.publish(st)
			log.Printf("load: %s %s failed: %s", a.Step, a.Path, res.Error)
			continue
		}
		res.OK = true
		st.Attempts = append(st.Attempts, res)
		return l.succeed(st, a.Step, info, note), nil
	}
	return l.fail(st, errors.New("every fallback step failed"))
}

// Load loads one file on an explicit kernel. backendName "" or "auto" resolves
// by extension; provider applies to onnx.
func (l *Launcher) Load(ctx context.Context, path, backendName, provider string, contexts int) (LoadState, error) {
	l.loadMu.Lock()
	defer l.loadMu.Unlock()
	cfg := l.Config()
	if contexts > 0 {
		cfg.Contexts = contexts
	}
	st := l.begin("manual")
	a := modelstore.Attempt{Path: path, Backend: backendName, Provider: provider, Reason: "manual"}
	if a.Backend == "" || a.Backend == "auto" {
		if strings.EqualFold(filepath.Ext(path), ".onnx") {
			a.Backend = "onnx"
		} else {
			a.Backend = "tensorrt"
		}
	}
	a.Step = a.Backend
	if a.Backend == "onnx" {
		if a.Provider == "" {
			a.Provider = "cuda"
		}
		a.Step = "onnx-" + a.Provider
	}
	start := time.Now()
	info, note, err := l.loadOne(ctx, cfg, a)
	res := AttemptResult{Attempt: a, Ms: float64(time.Since(start).Microseconds()) / 1000}
	if err != nil {
		res.Error = firstLine(err.Error())
		st.Attempts = append(st.Attempts, res)
		_, ferr := l.fail(st, err)
		return l.State(), ferr
	}
	res.OK = true
	st.Attempts = append(st.Attempts, res)
	return l.succeed(st, a.Step, info, note), nil
}

// Unload releases the model.
func (l *Launcher) Unload() {
	l.loadMu.Lock()
	defer l.loadMu.Unlock()
	l.Switcher.Unload()
	l.mu.Lock()
	l.state.Phase = "idle"
	l.mu.Unlock()
}

func (l *Launcher) loadOne(ctx context.Context, cfg config.Config, a modelstore.Attempt) (backend.Info, string, error) {
	kind, err := backend.ParseKind(a.Backend)
	if err != nil {
		return backend.Info{}, "", err
	}
	bcfg := BackendConfig(cfg)
	info, note, err := l.Switcher.LoadWith(ctx, bcfg, backend.Options{
		Path: a.Path, Kind: kind, Provider: a.Provider, Contexts: cfg.Contexts,
	})
	if err != nil {
		return info, note, err
	}
	return info, note, nil
}

// probeStep says why a step cannot run on this machine, or "" when it can.
// It is cheap: no model is opened.
func (l *Launcher) probeStep(cfg config.Config, step string) string {
	switch step {
	case config.StepTensorRT:
		var msg string
		func() {
			defer func() {
				if r := recover(); r != nil {
					msg = fmt.Sprintf("tensorrt panic: %v", r)
				}
			}()
			if err := trtbackend.Initialize(); err != nil {
				msg = firstLine(err.Error())
			}
		}()
		return msg
	default:
		provider := strings.TrimPrefix(step, "onnx-")
		if _, err := ortbackend.ProbeFor(cfg.ONNXBridgePath, cfg.ONNXRuntimePath, provider); err != nil {
			return firstLine(err.Error())
		}
	}
	return ""
}

func (l *Launcher) begin(mode string) LoadState {
	st := LoadState{Phase: "loading", Mode: mode, StartedAt: time.Now(), Attempts: []AttemptResult{}}
	l.publish(st)
	return st
}

func (l *Launcher) publish(st LoadState) {
	l.mu.Lock()
	l.state = st
	l.mu.Unlock()
}

func (l *Launcher) fail(st LoadState, err error) (LoadState, error) {
	st.Phase = "failed"
	st.Error = err.Error()
	st.EndedAt = time.Now()
	if l.Switcher.Loaded() {
		// A failed switch can leave the previous model resident; say so.
		if info, ok := l.Switcher.Info(); ok {
			st.Phase = "ready"
			st.Path, st.Backend, st.Device = info.Path, info.Backend, info.Device
			st.Note = "load failed; previous model is still serving"
		}
	}
	l.publish(st)
	return st, err
}

func (l *Launcher) succeed(st LoadState, step string, info backend.Info, note string) LoadState {
	st.Phase = "ready"
	st.Step = step
	st.Path, st.Backend, st.Device = info.Path, info.Backend, info.Device
	st.Note = note
	st.EndedAt = time.Now()
	if p, err := l.installModelConfig(); err != nil {
		l.Service.AdaptToEngine()
		st.Note = strings.TrimSpace(st.Note + " model config: " + err.Error() + "; probabilities are uncalibrated")
	} else {
		st.ModelConfig = p
	}
	l.Metrics.RecordEngineLoad()
	l.publish(st)
	lim := l.Service.Limits()
	log.Printf("model: %s [%s %s] seq %d..%d, using max_len=%d head_max_len=%d",
		info.Path, step, info.Device, lim.SequenceMin, lim.SequenceMax, lim.MaxLen, lim.HeadMaxLen)
	return st
}

// installModelConfig finds rl_agent_config.json (configured path, model dir,
// Hugging Face cache) and installs its calibration.
func (l *Launcher) installModelConfig() (string, error) {
	cfg := l.Config()
	var candidates []string
	if cfg.ModelConfigPath != "" {
		candidates = append(candidates, cfg.ModelConfigPath)
	}
	if dir := l.Switcher.Dir(); dir != "" {
		candidates = append(candidates, filepath.Join(dir, "rl_agent_config.json"))
	}
	if p := newestHF("rl_agent_config.json"); p != "" {
		candidates = append(candidates, p)
	}
	lastErr := errors.New("no rl_agent_config.json found")
	for _, c := range candidates {
		if _, err := os.Stat(c); err != nil {
			continue
		}
		if err := l.Service.LoadModelConfigFrom(c); err != nil {
			lastErr = err
			continue
		}
		return c, nil
	}
	return "", lastErr
}

func bgContext() context.Context { return context.Background() }

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}
