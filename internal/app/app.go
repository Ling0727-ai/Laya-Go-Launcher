// Package app holds the process-level wiring: configuration, the embedded HTTP
// server, and the Wails bindings.
//
// The bindings are deliberately a thin adapter. They decode a binding call,
// call internal/inference, and return the same shapes the HTTP API returns, so
// the GUI and a REST client see identical behaviour.
package app

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	wailsruntime "github.com/wailsapp/wails/v2/pkg/runtime"

	"github.com/local/laya-go-launcher/internal/backend"
	"github.com/local/laya-go-launcher/internal/backends"
	"github.com/local/laya-go-launcher/internal/inference"
	"github.com/local/laya-go-launcher/internal/kernel"
	"github.com/local/laya-go-launcher/internal/launcher"
	"github.com/local/laya-go-launcher/internal/metrics"
	"github.com/local/laya-go-launcher/internal/trtbackend"
)

// Config is the process configuration, loaded from a JSON file and overridable
// by environment variables.
type Config struct {
	// EnginePath is loaded at startup when set.
	EnginePath string `json:"engine_path"`
	// ModelConfigPath points at the checkpoint's rl_agent_config.json. When empty
	// it is looked for next to the engine and in the Hugging Face cache.
	ModelConfigPath string `json:"model_config_path"`
	// TokenizerPath overrides tokenizer discovery.
	TokenizerPath string `json:"tokenizer_path"`
	// HTTPAddr is the listen address for the REST API.
	HTTPAddr string `json:"http_addr"`
	// Contexts is the pre-allocated execution context count; 0 picks from VRAM.
	Contexts int `json:"contexts"`
	// MaxLen and HeadMaxLen are the token budgets.
	MaxLen     int `json:"max_len"`
	HeadMaxLen int `json:"head_max_len"`
	// PadLen is the fixed sequence length the engine was built for; 0 means the
	// engine has dynamic shapes.
	PadLen int `json:"pad_len"`
	// Diagnostics turns on the native TensorRT diagnostic log.
	Diagnostics bool `json:"diagnostics"`

	// Backend selects the execution kernel: "auto" (the default, meaning the
	// TensorRT path), "tensorrt" or "onnx".
	Backend string `json:"backend"`
	// Provider selects the ONNX Runtime execution provider: "cuda", "cpu" or
	// "directml". Empty lets the backend choose.
	Provider string `json:"provider"`
	// ONNXRuntimePath overrides discovery of onnxruntime.dll.
	ONNXRuntimePath string `json:"onnx_runtime_path"`
	// ONNXBridgePath overrides discovery of the native ONNX bridge library.
	ONNXBridgePath string `json:"onnx_bridge_path"`
}

// DefaultConfig is what runs with no config file.
//
// MaxLen/HeadMaxLen/PadLen are zero on purpose: the loaded engine's own shapes
// decide them (see inference.Service.AdaptToEngine). A hardcoded default that
// disagrees with the engine is exactly what makes a first run fail.
func DefaultConfig() Config {
	return Config{
		HTTPAddr:   "127.0.0.1:8420",
		Contexts:   0,
		MaxLen:     0,
		HeadMaxLen: 0,
		PadLen:     0,
		Backend:    string(backend.KindAuto),
	}
}

// LoadConfig reads layatrt.config.json from the working directory (or the path
// in LAYA_TRT_CONFIG), then applies environment overrides.
func LoadConfig() (Config, error) {
	cfg := DefaultConfig()

	path := os.Getenv("LAYA_TRT_CONFIG")
	if path == "" {
		path = "layatrt.config.json"
	}
	if raw, err := os.ReadFile(path); err == nil {
		if err := json.Unmarshal(raw, &cfg); err != nil {
			return cfg, fmt.Errorf("parse %s: %w", path, err)
		}
	} else if !os.IsNotExist(err) {
		return cfg, err
	}

	if v := os.Getenv("LAYA_TRT_ENGINE"); v != "" {
		cfg.EnginePath = v
	}
	if v := os.Getenv("LAYA_TRT_HTTP_ADDR"); v != "" {
		cfg.HTTPAddr = v
	}
	if v := os.Getenv("LAYA_TRT_TOKENIZER"); v != "" {
		cfg.TokenizerPath = v
	}
	if v := os.Getenv("LAYA_TRT_MODEL_CONFIG"); v != "" {
		cfg.ModelConfigPath = v
	}
	if v := os.Getenv("LAYA_TRT_BACKEND"); v != "" {
		cfg.Backend = v
	}
	if v := os.Getenv("LAYA_TRT_PROVIDER"); v != "" {
		cfg.Provider = v
	}
	if v := os.Getenv("LAYA_TRT_ONNX_RUNTIME"); v != "" {
		cfg.ONNXRuntimePath = v
	}
	if v := os.Getenv("LAYA_TRT_ONNX_BRIDGE"); v != "" {
		cfg.ONNXBridgePath = v
	}
	if cfg.Diagnostics {
		kernel.SetDiagnostics(true)
	}
	// A backend name that does not parse is a configuration mistake worth
	// reporting now rather than at the first load.
	if _, err := backend.ParseKind(cfg.Backend); err != nil {
		return cfg, err
	}
	return cfg, nil
}

// BackendKind is the parsed kernel selection.
func (c Config) BackendKind() backend.Kind {
	kind, err := backend.ParseKind(c.Backend)
	if err != nil {
		return backend.KindAuto
	}
	return kind
}

// BackendConfig is the kernel configuration this process will use.
func (c Config) BackendConfig() backends.Config {
	return backends.Config{
		Kind:        c.BackendKind(),
		Provider:    c.Provider,
		RuntimePath: c.ONNXRuntimePath,
		DLLPath:     c.ONNXBridgePath,
	}
}

// HTTPServer owns the REST listener.
type HTTPServer struct {
	addr     string
	srv      *http.Server
	listener net.Listener
}

// NewHTTPServer prepares a listener. Binding happens in Start so the caller can
// decide what a port clash means.
func NewHTTPServer(addr string, handler http.Handler) *HTTPServer {
	return &HTTPServer{
		addr: addr,
		srv: &http.Server{
			Handler:           handler,
			ReadHeaderTimeout: 5 * time.Second,
			ReadTimeout:       30 * time.Second,
			WriteTimeout:      120 * time.Second,
			IdleTimeout:       60 * time.Second,
		},
	}
}

// Start binds and serves in the background.
//
// When the configured port is taken it falls back to the next free one instead
// of failing: during `wails dev` a previous instance can still hold the port,
// and refusing to start the window over an API port would be the wrong trade.
// The actual address is available from Addr afterwards.
func (h *HTTPServer) Start() error {
	ln, err := net.Listen("tcp", h.addr)
	if err != nil {
		fallback, ferr := listenNextFree(h.addr, 10)
		if ferr != nil {
			return fmt.Errorf("listen on %s: %w (and no fallback port was free)", h.addr, err)
		}
		log.Printf("http api: %s is in use, using %s instead", h.addr, fallback.Addr())
		ln = fallback
	}
	h.listener = ln
	h.addr = ln.Addr().String()

	go func() {
		_ = h.srv.Serve(ln)
	}()
	return nil
}

// listenNextFree tries the next `attempts` ports after the one in addr.
func listenNextFree(addr string, attempts int) (net.Listener, error) {
	host, portStr, err := net.SplitHostPort(addr)
	if err != nil {
		return nil, err
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		return nil, err
	}
	var lastErr error
	for i := 1; i <= attempts; i++ {
		ln, err := net.Listen("tcp", net.JoinHostPort(host, strconv.Itoa(port+i)))
		if err == nil {
			return ln, nil
		}
		lastErr = err
	}
	return nil, lastErr
}

// Stop shuts the listener down.
func (h *HTTPServer) Stop() {
	if h.listener != nil {
		_ = h.listener.Close()
	}
	if h.srv != nil {
		_ = h.srv.Close()
	}
}

// Addr is the address actually bound, which may differ from the configured one
// when the configured port was taken.
func (h *HTTPServer) Addr() string { return h.addr }

// Bindings is the Wails-exposed surface. Every method is callable from the
// frontend as `window.go.app.Bindings.<Method>`.
type Bindings struct {
	service   *inference.Service
	metrics   *metrics.Recorder
	version   string
	configDir string

	// switcher owns the active kernel, so a load made through the GUI obeys the
	// same selection the process started with and becomes the model the service
	// predicts on.
	switcher *backends.Switcher

	// ctx is the Wails runtime context. The native file dialogs need it.
	ctx context.Context

	// launcher, when set, owns loading: auto fallback, catalog, conversion.
	launcher *launcher.Launcher
	// apiAddr is the bound REST address, shown in the GUI.
	apiAddr string
}

var errNoLauncher = fmt.Errorf("launcher is not attached")

// NewBindings wires the bindings to the same objects the HTTP API uses.
func NewBindings(svc *inference.Service, rec *metrics.Recorder, version string, switcher *backends.Switcher) *Bindings {
	dir, _ := os.Getwd()
	return &Bindings{
		service:   svc,
		metrics:   rec,
		version:   version,
		configDir: dir,
		switcher:  switcher,
	}
}

// Startup is called by Wails once the window exists.
func (b *Bindings) Startup(ctx context.Context) {
	b.ctx = ctx
	if b.launcher != nil {
		// Same startup as the pure server: default config, fallback chain,
		// background conversion. Runs async so the window paints immediately.
		go func() {
			if _, err := b.launcher.AutoLoad(context.Background()); err != nil {
				log.Printf("load: %v", err)
			}
		}()
		return
	}

	// Load the configured model on startup so the window opens ready to use.
	cfg, err := LoadConfig()
	if err != nil || cfg.EnginePath == "" {
		return
	}
	if _, _, err := b.switcher.LoadWith(ctx, cfg.BackendConfig(),
		backend.Options{Path: cfg.EnginePath, Contexts: cfg.Contexts}); err != nil {
		// Surfaced through Status() rather than blocking startup.
		return
	}
	// Calibration temperatures and token budgets both come from the checkpoint
	// config; without it the probabilities are unscaled.
	if _, err := b.LoadModelConfig(); err != nil {
		log.Printf("warning: %v; running without temperature scaling", err)
	}
}

// Shutdown releases the model when the window closes.
func (b *Bindings) Shutdown(ctx context.Context) {
	b.service.Backend().Close()
}

// Status is what the GUI polls for its header.
func (b *Bindings) Status() map[string]any {
	dev := trtbackend.DeviceInfo()
	vram := trtbackend.VRAMInfo()
	probe := backends.Probe(b.switcher.Config())

	out := map[string]any{
		"version": b.version,
		"kernel": map[string]any{
			"available": probe.TensorRT.Available,
			"tensorrt":  probe.TensorRT.Version,
			"error":     probe.TensorRT.Error,
		},
		"backend": map[string]any{
			"requested": probe.Requested,
			"selected":  probe.Selected,
			"note":      probe.Note,
			"onnx": map[string]any{
				"available": probe.ONNX.Available,
				"version":   probe.ONNX.Version,
				"error":     probe.ONNX.Error,
			},
		},
		"device": map[string]any{
			"name":               dev.Name,
			"compute_capability": fmt.Sprintf("%d.%d", dev.ComputeMajor, dev.ComputeMinor),
			"vram_free_mb":       vram.FreeMB,
			"vram_total_mb":      vram.TotalMB,
		},
		"engine": map[string]any{"loaded": false},
	}
	if info, ok := b.service.Backend().Info(); ok {
		out["engine"] = backendInfoBody(info)
	}
	maxLen, headMaxLen, padLen := b.service.Budgets()
	out["budgets"] = map[string]any{
		"max_len":      maxLen,
		"head_max_len": headMaxLen,
		"pad_len":      padLen,
	}
	// How many options the model can score at once, so the UI can warn before a
	// question is written rather than after it fails.
	out["marker_capacity"] = b.service.MarkerCapacity()
	return out
}

// LoadEngine loads a model by path through the configured kernel.
//
// The kernel is resolved from the model's own format, so picking a .onnx graph
// from the list switches to ONNX Runtime and picking a .engine plan switches
// back, without the user having to name a kernel. An explicit choice is still
// possible through LoadEngineWith.
func (b *Bindings) LoadEngine(path string, contexts int) (map[string]any, error) {
	return b.LoadEngineWith(path, contexts, "", "")
}

// LoadEngineWith loads a model, optionally overriding the kernel and provider
// for this load only.
//
// The override is an option rather than a change to the process configuration:
// a load that names a kernel is describing that model, and persisting it would
// silently re-point every later load — which is exactly what made switching
// back from ONNX to TensorRT fail.
func (b *Bindings) LoadEngineWith(path string, contexts int, backendName, provider string) (map[string]any, error) {
	cfg := b.switcher.Config()
	opts := backend.Options{Path: path, Contexts: contexts}
	if backendName != "" {
		kind, err := backend.ParseKind(backendName)
		if err != nil {
			return nil, err
		}
		opts.Kind = kind
	}
	opts.Provider = provider
	if b.launcher != nil {
		return b.loadViaLauncher(path, contexts, backendName, provider)
	}

	info, note, err := b.switcher.LoadWith(context.Background(), cfg, opts)
	if err != nil {
		return nil, err
	}
	// Fit the token budgets and calibration to this model before anything uses
	// them. A failure to find the checkpoint config is reported but not fatal.
	configPath, cfgErr := b.LoadModelConfig()
	if cfgErr != nil {
		b.service.AdaptToEngine()
	}

	lim := b.service.Limits()
	body := backendInfoBody(info)
	body["limits"] = lim
	body["model_config"] = configPath
	body["model_config_error"] = errorString(cfgErr)
	if note != "" {
		body["backend_note"] = note
	}
	return body, nil
}

// UnloadEngine releases the model.
func (b *Bindings) UnloadEngine() map[string]any {
	b.service.Backend().Unload()
	return map[string]any{"unloaded": true}
}

// SetBudgets changes the token budgets.
func (b *Bindings) SetBudgets(maxLen, headMaxLen, padLen int) map[string]any {
	b.service.SetBudgets(maxLen, headMaxLen, padLen)
	m, h, p := b.service.Budgets()
	return map[string]any{"max_len": m, "head_max_len": h, "pad_len": p}
}

// Predict runs the predict use case. The frontend passes the same JSON shape the
// HTTP API accepts, so one request builder serves both.
func (b *Bindings) Predict(request map[string]any) (map[string]any, error) {
	raw, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}
	var req struct {
		State     any                               `json:"state"`
		Questions map[string]inference.QuestionSpec `json:"questions"`
	}
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, fmt.Errorf("decode request: %w", err)
	}
	if !b.service.Backend().Loaded() {
		return nil, fmt.Errorf("no model is loaded")
	}
	resp, err := b.service.Predict(context.Background(), inference.Request{
		State:     req.State,
		Questions: req.Questions,
	})
	if err != nil {
		return nil, err
	}
	out, err := json.Marshal(resp)
	if err != nil {
		return nil, err
	}
	var generic map[string]any
	if err := json.Unmarshal(out, &generic); err != nil {
		return nil, err
	}
	return generic, nil
}

// Tokenize exposes the tokeniser for the inspector panel.
func (b *Bindings) Tokenize(text string, withSpecials bool) map[string]any {
	tk := b.service.Tokenizer()
	if tk == nil {
		return map[string]any{"error": "no tokenizer is loaded"}
	}
	ids := tk.Encode(text)
	if withSpecials {
		ids = tk.EncodeWithSpecials(text)
	}
	return map[string]any{
		"count":  len(ids),
		"ids":    ids,
		"tokens": tk.Tokens(ids),
	}
}

// Metrics returns the counters and latency percentiles.
func (b *Bindings) Metrics() metrics.Snapshot { return b.metrics.Snapshot() }

// WorkingDirectory is the process working directory.
func (b *Bindings) WorkingDirectory() string { return b.configDir }

// DefaultEngineDir is where a file browser should start: the first directory
// that actually contains engine files, else the working directory.
func (b *Bindings) DefaultEngineDir() string {
	for _, dir := range b.engineSearchDirs() {
		if found, _ := listModels(dir); len(found) > 0 {
			return dir
		}
	}
	dir, _ := os.Getwd()
	return dir
}

// engineSearchDirs is where engines are looked for, most specific first.
//
// Only directories that could hold a laya engine are listed. Pointing this at
// another project's model folder is worse than useless: it fills the picker with
// engines that load and then fail, so the user has to guess which of nine
// entries is theirs.
func (b *Bindings) engineSearchDirs() []string {
	var dirs []string
	add := func(p string) {
		if p == "" {
			return
		}
		if info, err := os.Stat(p); err == nil && info.IsDir() {
			for _, existing := range dirs {
				if existing == p {
					return
				}
			}
			dirs = append(dirs, p)
		}
	}

	if cfg, err := LoadConfig(); err == nil && cfg.EnginePath != "" {
		add(filepath.Dir(cfg.EnginePath))
	}
	if home, err := os.UserHomeDir(); err == nil {
		add(filepath.Join(home, "Documents", "laya-trt"))
		add(filepath.Join(home, "Documents", "laya-trt", "models"))
		add(filepath.Join(home, "Documents", "laya-models"))
	}
	add(b.configDir)
	add(filepath.Join(b.configDir, "models"))
	return dirs
}

// EngineFile is one discoverable engine.
type EngineFile struct {
	Path     string  `json:"path"`
	Name     string  `json:"name"`
	Dir      string  `json:"dir"`
	SizeMB   float64 `json:"size_mb"`
	Modified string  `json:"modified"`
	InUse    bool    `json:"in_use"`
	// Compatible is false when the engine's IO tensors are not laya's. Such an
	// entry is still listed (it is a real .engine file) but the UI marks it, so
	// nobody picks an image model by mistake.
	Compatible bool `json:"compatible"`
	// Reason explains an incompatible engine.
	Reason string `json:"reason,omitempty"`
	// Catalog metadata (launcher mode only).
	SeqMax    int    `json:"seq_max,omitempty"`
	Precision string `json:"precision,omitempty"`
	Format    string `json:"format,omitempty"`
	Tags      string `json:"tags,omitempty"`
}

// layaInputs are the tensor names every laya model exposes. Their presence is
// what distinguishes this model from any other file.
var layaInputs = []string{"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"}

// DiscoverEngines scans the usual locations and returns what it finds, so the
// user can pick from a list instead of typing a path.
//
// Each candidate is opened and its IO tensors checked against laya's. Listing a
// directory's worth of unrelated model files would otherwise put image models
// next to real ones, and picking one fails only after the user has committed.
//
// Both kernels' formats are listed: a .onnx graph is as valid a choice as a
// .engine plan now that the ONNX path exists, and hiding it would make the new
// backend unreachable from the UI.
func (b *Bindings) DiscoverEngines() []EngineFile {
	// Non-nil on purpose: a nil slice marshals to JSON null, and the frontend
	// calls .filter() on this array, so "no engines found" would blank the window.
	out := []EngineFile{}
	if b.launcher != nil {
		return b.discoverFromCatalog()
	}
	seen := map[string]bool{}

	loadedPath := ""
	if info, ok := b.service.Backend().Info(); ok {
		loadedPath = info.Path
	}

	for _, dir := range b.engineSearchDirs() {
		found, _ := listModels(dir)
		for _, e := range found {
			if seen[e.Path] {
				continue
			}
			seen[e.Path] = true
			e.InUse = e.Path == loadedPath
			e.Compatible, e.Reason = b.probeLayaModel(e.Path)
			out = append(out, e)
		}
	}
	return out
}

// probeLayaModel opens a model far enough to read its IO tensors, and reports
// whether it is a laya model.
//
// Loading is the only reliable test: filenames say nothing, and a model built
// for another task is a perfectly valid file of its own kind. The model is
// released immediately; this is a metadata read, not a session.
//
// The kernel is chosen from the extension rather than from configuration,
// because a plan can only be read by TensorRT and a graph only by ONNX Runtime.
func (b *Bindings) probeLayaModel(path string) (bool, string) {
	info, err := backends.Inspect(context.Background(), b.switcher.Config(), path)
	if err != nil {
		return false, "无法加载：" + firstLine(err.Error())
	}
	missing := backends.MissingInputs(info, layaInputs)
	if len(missing) == len(layaInputs) {
		return false, "不是 laya 模型（输入里没有 " + layaInputs[0] + "）"
	}
	if len(missing) > 0 {
		return false, "输入不完整，缺少 " + strings.Join(missing, ", ")
	}
	return true, ""
}

// firstLine trims an error to its first line so a picker row stays readable.
func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// listModels returns the model files directly inside dir.
//
// Both kernels' formats are listed: a .engine plan for TensorRT and a .onnx
// graph for ONNX Runtime. Which one serves a given file is decided by its
// extension, not by configuration, because the formats are not interchangeable.
func listModels(dir string) ([]EngineFile, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var out []EngineFile
	for _, entry := range entries {
		if entry.IsDir() || !isModelFile(entry.Name()) {
			continue
		}
		full := filepath.Join(dir, entry.Name())
		info, err := entry.Info()
		if err != nil {
			continue
		}
		out = append(out, EngineFile{
			Path:     full,
			Name:     entry.Name(),
			Dir:      dir,
			SizeMB:   float64(info.Size()) / (1024 * 1024),
			Modified: info.ModTime().Format("2006-01-02 15:04"),
		})
	}
	return out, nil
}

func isModelFile(name string) bool {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".engine", ".onnx":
		return true
	default:
		return false
	}
}

// PickEngineFile opens the native file dialog. Returns "" when cancelled.
func (b *Bindings) PickEngineFile() (string, error) {
	if b.ctx == nil {
		return "", fmt.Errorf("the runtime is not ready yet")
	}
	return wailsruntime.OpenFileDialog(b.ctx, wailsruntime.OpenDialogOptions{
		Title:            "选择模型文件",
		DefaultDirectory: b.DefaultEngineDir(),
		Filters: []wailsruntime.FileFilter{
			{DisplayName: "模型 (*.engine;*.onnx)", Pattern: "*.engine;*.onnx"},
			{DisplayName: "TensorRT engine (*.engine)", Pattern: "*.engine"},
			{DisplayName: "ONNX 模型 (*.onnx)", Pattern: "*.onnx"},
			{DisplayName: "所有文件", Pattern: "*.*"},
		},
	})
}

// PickTokenizerFile opens the native file dialog for a tokenizer.json.
func (b *Bindings) PickTokenizerFile() (string, error) {
	if b.ctx == nil {
		return "", fmt.Errorf("the runtime is not ready yet")
	}
	return wailsruntime.OpenFileDialog(b.ctx, wailsruntime.OpenDialogOptions{
		Title:            "选择 tokenizer.json",
		DefaultDirectory: b.DefaultEngineDir(),
		Filters: []wailsruntime.FileFilter{
			{DisplayName: "tokenizer.json", Pattern: "tokenizer.json"},
			{DisplayName: "JSON", Pattern: "*.json"},
		},
	})
}

// PickDocumentFile opens the native dialog to load a document into the state
// field, so a user can point at a real file instead of pasting.
func (b *Bindings) PickDocumentFile() (string, error) {
	if b.ctx == nil {
		return "", fmt.Errorf("the runtime is not ready yet")
	}
	return wailsruntime.OpenFileDialog(b.ctx, wailsruntime.OpenDialogOptions{
		Title: "选择要分析的文件",
		Filters: []wailsruntime.FileFilter{
			{DisplayName: "文本与邮件", Pattern: "*.txt;*.md;*.eml;*.json;*.csv;*.log"},
			{DisplayName: "所有文件", Pattern: "*.*"},
		},
	})
}

// ReadTextFile loads a picked document as UTF-8 text for the state field.
func (b *Bindings) ReadTextFile(path string) (string, error) {
	const maxBytes = 1 << 20 // matches the API's body limit
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	if len(raw) > maxBytes {
		return "", fmt.Errorf("文件超过 %d KiB，请截取后再试", maxBytes/1024)
	}
	return string(raw), nil
}

// LoadModelConfig finds and installs the checkpoint's rl_agent_config.json.
//
// Search order: the configured path, the engine's own directory, then the
// Hugging Face cache. The file carries the calibration temperatures, so failing
// to find it means probabilities come out unscaled.
func (b *Bindings) LoadModelConfig() (string, error) {
	cfg, _ := LoadConfig()
	var candidates []string
	if cfg.ModelConfigPath != "" {
		candidates = append(candidates, cfg.ModelConfigPath)
	}
	if dir := b.service.Backend().Dir(); dir != "" {
		candidates = append(candidates, filepath.Join(dir, "rl_agent_config.json"))
	}
	if p := findCachedModelConfig(); p != "" {
		candidates = append(candidates, p)
	}

	var lastErr error
	for _, c := range candidates {
		if _, err := os.Stat(c); err != nil {
			lastErr = err
			continue
		}
		if err := b.service.LoadModelConfigFrom(c); err != nil {
			lastErr = err
			continue
		}
		return c, nil
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("no rl_agent_config.json found")
	}
	return "", lastErr
}

// findCachedModelConfig looks in the Hugging Face cache for a laya checkpoint's
// rl_agent_config.json.
func findCachedModelConfig() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	pattern := filepath.Join(home, ".cache", "huggingface", "hub",
		"models--convaiinnovations--laya", "snapshots", "*", "rl_agent_config.json")
	matches, _ := filepath.Glob(pattern)
	best := ""
	for _, m := range matches {
		if strings.Compare(m, best) > 0 {
			best = m
		}
	}
	return best
}

// Limits reports what the loaded engine accepts and what the service is using.
// The GUI shows this so a user can see the real ceiling instead of guessing.
func (b *Bindings) Limits() inference.Limits {
	if !b.service.Backend().Loaded() {
		return inference.Limits{}
	}
	return b.service.Limits()
}

// RevealEngine opens the containing folder in the OS file browser, so a user can
// see what else is next to an engine.
func (b *Bindings) RevealEngine(path string) error {
	if path == "" {
		return fmt.Errorf("没有可打开的路径")
	}
	dir := filepath.Dir(path)
	if _, err := os.Stat(dir); err != nil {
		return err
	}
	return openInFileManager(dir)
}

// backendInfoBody renders a resident model for the GUI, in the same shape the
// HTTP API returns so the two front doors cannot drift apart.
func backendInfoBody(info backend.Info) map[string]any {
	body := map[string]any{
		"loaded":               true,
		"path":                 info.Path,
		"backend":              info.Backend,
		"contexts":             info.Contexts,
		"activation_memory_mb": info.ActivationMemoryMB,
		"inputs":               tensorList(info.Inputs),
		"outputs":              tensorList(info.Outputs),
	}
	if info.Runtime != "" {
		body["runtime"] = info.Runtime
	}
	if info.Device != "" {
		body["device"] = info.Device
	}
	return body
}

func tensorList(ts []backend.TensorInfo) []map[string]any {
	out := make([]map[string]any, 0, len(ts))
	for _, t := range ts {
		out = append(out, map[string]any{
			"name":  t.Name,
			"dtype": t.DType.String(),
			"shape": t.Shape,
		})
	}
	return out
}

// errorString renders an error for a JSON response, or "" when there is none.
func errorString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}
