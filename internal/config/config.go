// Package config is the process configuration shared by the headless server and
// the desktop shell.
//
// It deliberately has no native or GUI dependencies, so the pure server build
// does not link Wails and a config file can be read and written by tools.
//
// Precedence, lowest first: DefaultConfig → config file → environment → command
// line (applied by the caller) → a single request.
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Fallback steps understood by the auto loader, in the default order.
const (
	StepTensorRT     = "tensorrt"
	StepONNXCUDA     = "onnx-cuda"
	StepONNXDirectML = "onnx-directml"
	StepONNXCPU      = "onnx-cpu"
)

// DefaultFallback is the chain used when nothing else is configured.
var DefaultFallback = []string{StepTensorRT, StepONNXCUDA, StepONNXDirectML, StepONNXCPU}

// DefaultModelSeq is the default target context: the encoder's hard ceiling
// (max_position_embeddings), so the default model accepts any input laya can.
const DefaultModelSeq = 8192

// Config is the process configuration.
type Config struct {
	// EnginePath pins one model file. Empty lets the auto loader pick from the
	// catalog; when set with backend auto it is tried first and the rest of the
	// fallback chain still applies.
	EnginePath string `json:"engine_path"`
	// ModelConfigPath points at rl_agent_config.json; empty searches for it.
	ModelConfigPath string `json:"model_config_path"`
	// TokenizerPath overrides tokenizer discovery.
	TokenizerPath string `json:"tokenizer_path"`
	// HTTPAddr is the REST listen address.
	HTTPAddr string `json:"http_addr"`
	// Contexts is the pre-allocated TensorRT context count; 0 picks from VRAM.
	Contexts int `json:"contexts"`
	// Token budgets; 0 lets the model and checkpoint decide (recommended).
	MaxLen     int `json:"max_len"`
	HeadMaxLen int `json:"head_max_len"`
	PadLen     int `json:"pad_len"`
	// Diagnostics turns on the native TensorRT diagnostic log.
	Diagnostics bool `json:"diagnostics"`

	// Backend: "auto" (fallback chain), "tensorrt" or "onnx" (no fallback).
	Backend string `json:"backend"`
	// Provider for an explicit onnx backend: "cuda", "directml" or "cpu".
	Provider        string `json:"provider"`
	ONNXRuntimePath string `json:"onnx_runtime_path"`
	ONNXBridgePath  string `json:"onnx_bridge_path"`

	// ModelSeq is the context the auto loader looks for (default 8192). A model
	// whose ceiling is below it is not chosen automatically.
	ModelSeq int `json:"model_seq"`
	// Fallback is the auto loader's chain. Unknown steps are rejected.
	Fallback []string `json:"fallback"`
	// AutoConvert builds a TensorRT engine from the ONNX graph in the background
	// when no usable engine for ModelSeq exists, then switches to it.
	AutoConvert bool `json:"auto_convert"`
	// Precision of automatically built engines: "fp16" (default) or "fp32".
	Precision string `json:"precision"`
	// ModelDirs are searched before the built-in locations.
	ModelDirs []string `json:"model_dirs"`
	// EngineDir receives built engines; empty uses the first engines/ folder
	// found, else ./engines.
	EngineDir string `json:"engine_dir"`
	// TrtexecPath overrides trtexec discovery.
	TrtexecPath string `json:"trtexec_path"`
	// AdminToken, when set, is required as "Authorization: Bearer <token>" on
	// endpoints that change configuration, load models or start conversions.
	AdminToken string `json:"admin_token"`
}

// Default is what runs with no config file.
func Default() Config {
	return Config{
		HTTPAddr:    "127.0.0.1:8420",
		Backend:     "auto",
		ModelSeq:    DefaultModelSeq,
		Fallback:    append([]string(nil), DefaultFallback...),
		AutoConvert: true,
		Precision:   "fp16",
	}
}

// Path is the config file this process reads: LAYA_TRT_CONFIG or
// ./layatrt.config.json.
func Path() string {
	if p := os.Getenv("LAYA_TRT_CONFIG"); p != "" {
		return p
	}
	return "layatrt.config.json"
}

// ReadFile returns defaults overlaid with the file at path, without environment
// overrides. A missing file is not an error.
func ReadFile(path string) (Config, error) {
	cfg := Default()
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return cfg, err
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return cfg, fmt.Errorf("parse %s: %w", path, err)
	}
	cfg.normalize()
	return cfg, nil
}

// Load reads the config file and applies environment overrides.
func Load() (Config, error) {
	cfg, err := ReadFile(Path())
	if err != nil {
		return cfg, err
	}
	cfg.ApplyEnv()
	return cfg, cfg.Validate()
}

// ApplyEnv applies the LAYA_TRT_* environment overrides.
func (c *Config) ApplyEnv() {
	str := map[string]*string{
		"LAYA_TRT_ENGINE":       &c.EnginePath,
		"LAYA_TRT_HTTP_ADDR":    &c.HTTPAddr,
		"LAYA_TRT_TOKENIZER":    &c.TokenizerPath,
		"LAYA_TRT_MODEL_CONFIG": &c.ModelConfigPath,
		"LAYA_TRT_BACKEND":      &c.Backend,
		"LAYA_TRT_PROVIDER":     &c.Provider,
		"LAYA_TRT_ONNX_RUNTIME": &c.ONNXRuntimePath,
		"LAYA_TRT_ONNX_BRIDGE":  &c.ONNXBridgePath,
		"LAYA_TRT_ENGINE_DIR":   &c.EngineDir,
		"LAYA_TRT_TRTEXEC":      &c.TrtexecPath,
		"LAYA_TRT_ADMIN_TOKEN":  &c.AdminToken,
	}
	for key, dst := range str {
		if v := os.Getenv(key); v != "" {
			*dst = v
		}
	}
	if v := os.Getenv("LAYA_TRT_MODEL_SEQ"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			c.ModelSeq = n
		}
	}
	if v := os.Getenv("LAYA_TRT_FALLBACK"); v != "" {
		c.Fallback = splitList(v)
	}
	if v := os.Getenv("LAYA_TRT_AUTO_CONVERT"); v != "" {
		c.AutoConvert = v == "1" || strings.EqualFold(v, "true")
	}
	if v := os.Getenv("LAYA_TRT_MODEL_DIRS"); v != "" {
		c.ModelDirs = append(filepath.SplitList(v), c.ModelDirs...)
	}
	c.normalize()
}

func splitList(v string) []string {
	var out []string
	for _, p := range strings.Split(v, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func (c *Config) normalize() {
	if c.ModelSeq <= 0 {
		c.ModelSeq = DefaultModelSeq
	}
	if len(c.Fallback) == 0 {
		c.Fallback = append([]string(nil), DefaultFallback...)
	}
	for i, s := range c.Fallback {
		c.Fallback[i] = NormalizeStep(s)
	}
	if c.Precision == "" {
		c.Precision = "fp16"
	}
	if c.Backend == "" {
		c.Backend = "auto"
	}
}

// NormalizeStep accepts the common spellings of a fallback step.
func NormalizeStep(s string) string {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "trt", "tensorrt", "engine":
		return StepTensorRT
	case "onnx-cuda", "cuda", "ort-cuda":
		return StepONNXCUDA
	case "onnx-directml", "onnx-dml", "directml", "dml":
		return StepONNXDirectML
	case "onnx-cpu", "cpu", "ort-cpu":
		return StepONNXCPU
	}
	return strings.ToLower(strings.TrimSpace(s))
}

// Validate reports a configuration mistake before any work is done.
func (c Config) Validate() error {
	switch strings.ToLower(strings.TrimSpace(c.Backend)) {
	case "", "auto", "tensorrt", "trt", "plan", "engine", "onnx", "onnxruntime", "ort":
	default:
		return fmt.Errorf("config: unknown backend %q (want auto, tensorrt or onnx)", c.Backend)
	}
	for _, s := range c.Fallback {
		switch s {
		case StepTensorRT, StepONNXCUDA, StepONNXDirectML, StepONNXCPU:
		default:
			return fmt.Errorf("config: unknown fallback step %q (want %s)", s, strings.Join(DefaultFallback, ", "))
		}
	}
	switch c.Precision {
	case "fp16", "fp32":
	default:
		return fmt.Errorf("config: precision must be fp16 or fp32, got %q", c.Precision)
	}
	if c.ModelSeq < 64 || c.ModelSeq > 8192 {
		return fmt.Errorf("config: model_seq must be within 64..8192, got %d", c.ModelSeq)
	}
	return nil
}

// Write saves cfg to path as indented JSON, atomically.
func Write(path string, cfg Config) error {
	cfg.normalize()
	if err := cfg.Validate(); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(raw, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Redacted is cfg with secrets removed, for display.
func (c Config) Redacted() Config {
	if c.AdminToken != "" {
		c.AdminToken = "***"
	}
	return c
}

// AutoChain reports whether loads should walk the fallback chain.
func (c Config) AutoChain() bool {
	b := strings.ToLower(strings.TrimSpace(c.Backend))
	return b == "" || b == "auto"
}
