package app

import (
	"context"
	"strings"

	"github.com/local/laya-go-launcher/internal/config"
	"github.com/local/laya-go-launcher/internal/launcher"
	"github.com/local/laya-go-launcher/internal/modelstore"
)

// AttachLauncher routes loading, discovery and configuration through the shared
// launcher, so the desktop shell behaves exactly like the pure server. It is a
// function rather than a method so Wails does not expose it to the frontend.
func AttachLauncher(b *Bindings, l *launcher.Launcher, apiAddr string) {
	b.launcher = l
	b.apiAddr = apiAddr
}

// APIAddress is the in-process REST address (the server console lives at /).
func (b *Bindings) APIAddress() string { return b.apiAddr }

// LoadState is the most recent load, with every fallback attempt.
func (b *Bindings) LoadState() launcher.LoadState {
	if b.launcher == nil {
		return launcher.LoadState{Phase: "idle", Attempts: []launcher.AttemptResult{}}
	}
	return b.launcher.State()
}

// AutoLoad runs the fallback chain now.
func (b *Bindings) AutoLoad() (launcher.LoadState, error) {
	if b.launcher == nil {
		return launcher.LoadState{}, errNoLauncher
	}
	return b.launcher.AutoLoad(context.Background())
}

// GetConfig returns the configuration in effect, its file and the defaults.
func (b *Bindings) GetConfig() map[string]any {
	if b.launcher == nil {
		return map[string]any{"config": config.Default(), "defaults": config.Default()}
	}
	return map[string]any{
		"config":    b.launcher.Config().Redacted(),
		"defaults":  config.Default(),
		"path":      b.launcher.ConfigPath(),
		"tokenizer": b.launcher.TokenizerPath(),
		"steps":     config.DefaultFallback,
	}
}

// SaveConfig validates and persists cfg; reload re-runs the fallback chain.
func (b *Bindings) SaveConfig(cfg config.Config, reload bool) (map[string]any, error) {
	if b.launcher == nil {
		return nil, errNoLauncher
	}
	if err := b.launcher.UpdateConfig(cfg); err != nil {
		return nil, err
	}
	out := map[string]any{"config": b.launcher.Config().Redacted(), "saved": true}
	if reload {
		st, err := b.launcher.AutoLoad(context.Background())
		out["state"] = st
		if err != nil {
			out["error"] = err.Error()
		}
	}
	return out, nil
}

// PlanAttempts is what AutoLoad would try right now, in order.
func (b *Bindings) PlanAttempts() []modelstore.Attempt {
	if b.launcher == nil {
		return []modelstore.Attempt{}
	}
	out := b.launcher.PlanAttempts()
	if out == nil {
		out = []modelstore.Attempt{}
	}
	return out
}

// StartConversion builds a TensorRT plan in the background.
func (b *Bindings) StartConversion(seq int, precision string) (modelstore.Job, error) {
	if b.launcher == nil {
		return modelstore.Job{}, errNoLauncher
	}
	return b.launcher.StartConversion(launcher.ConvertRequest{Seq: seq, Precision: precision})
}

// ConversionJobs lists builds, newest last.
func (b *Bindings) ConversionJobs() []modelstore.Job {
	if b.launcher == nil {
		return []modelstore.Job{}
	}
	out := b.launcher.Converter.Jobs()
	if out == nil {
		out = []modelstore.Job{}
	}
	return out
}

// CancelConversion stops the running build.
func (b *Bindings) CancelConversion() bool {
	return b.launcher != nil && b.launcher.Converter.Cancel()
}

func (b *Bindings) loadViaLauncher(path string, contexts int, backendName, provider string) (map[string]any, error) {
	st, err := b.launcher.Load(context.Background(), path, backendName, provider, contexts)
	if err != nil {
		return nil, err
	}
	info, _ := b.service.Backend().Info()
	body := backendInfoBody(info)
	body["limits"] = b.service.Limits()
	body["model_config"] = st.ModelConfig
	body["state"] = st
	if st.Note != "" {
		body["backend_note"] = st.Note
	}
	return body, nil
}

// discoverFromCatalog lists catalog models without deserializing any of them:
// file names, the engine manifest and converter sidecars describe each one.
func (b *Bindings) discoverFromCatalog() []EngineFile {
	out := []EngineFile{}
	for _, m := range b.launcher.Models() {
		e := EngineFile{
			Path:       m.Path,
			Name:       m.Name,
			Dir:        m.Dir,
			SizeMB:     m.SizeMB,
			Modified:   m.Modified.Format("2006-01-02 15:04"),
			InUse:      m.InUse,
			Compatible: true,
			SeqMax:     m.SeqMax,
			Precision:  m.Precision,
			Format:     string(m.Format),
		}
		var tags []string
		if m.MultiProfile {
			tags = append(tags, "双 profile")
		}
		if m.Optimized {
			tags = append(tags, "已融合")
		}
		e.Tags = strings.Join(tags, " · ")
		if m.Format == modelstore.FormatEngine && m.SeqMax == 0 {
			e.Compatible, e.Reason = false, "未知引擎（文件名/manifest 里没有 seq 信息，自动加载不会选它）"
		}
		out = append(out, e)
	}
	return out
}
