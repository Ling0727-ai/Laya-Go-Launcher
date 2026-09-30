// Package inference — model configuration.
//
// Reads the checkpoint's own rl_agent_config.json, which is the authority on
// token budgets and calibration temperatures. The Python SDK reads the same file
// and applies the same rules, so the two paths agree.
package inference

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Temperature bounds. A fitted temperature below 1 sharpens the logits instead
// of softening them; the shipped `choice:11+` bucket is 0.1006, which multiplies
// them ~10x and publishes a coin flip as a certainty. The SDK clamps to this
// range and warns, and so do we — see laya/common.py.
const (
	TempMin = 0.5
	TempMax = 5.0
)

// ModelConfig mirrors rl_agent_config.json.
type ModelConfig struct {
	Encoder      string             `json:"encoder"`
	HeadLayers   int                `json:"head_layers"`
	MaxLen       int                `json:"max_len"`
	HeadMaxLen   int                `json:"head_max_len"`
	AmpDtype     string             `json:"amp_dtype"`
	Temperature  []float64          `json:"temperature"`
	TempByOption map[string]float64 `json:"temperature_by_options"`

	// raw copies hold the file's values before clamping, so the warning can say
	// what was actually shipped.
	rawTemperature  []float64
	rawTempByOption map[string]float64
}

// LoadModelConfig reads rl_agent_config.json from a checkpoint directory.
func LoadModelConfig(path string) (*ModelConfig, error) {
	cfgFile := path
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		cfgFile = filepath.Join(path, "rl_agent_config.json")
	}
	raw, err := os.ReadFile(cfgFile)
	if err != nil {
		return nil, fmt.Errorf("inference: read model config: %w", err)
	}
	var cfg ModelConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("inference: parse %s: %w", cfgFile, err)
	}
	if cfg.MaxLen <= 0 {
		cfg.MaxLen = 512
	}
	if cfg.HeadMaxLen <= 0 {
		cfg.HeadMaxLen = 192
	}
	cfg.rawTemperature = append([]float64(nil), cfg.Temperature...)
	cfg.rawTempByOption = make(map[string]float64, len(cfg.TempByOption))
	for k, v := range cfg.TempByOption {
		cfg.rawTempByOption[k] = v
	}

	// Clamp in place, and collect what was rejected.
	for i, t := range cfg.Temperature {
		cfg.Temperature[i] = ClampTemperature(t)
	}
	for k, v := range cfg.TempByOption {
		cfg.TempByOption[k] = ClampTemperature(v)
	}
	return &cfg, nil
}

// RejectedTemperatures lists the buckets whose shipped value fell outside the
// usable range, so the caller can warn that those confidences are uncalibrated.
func (c *ModelConfig) RejectedTemperatures() []string {
	var out []string
	for i, t := range c.rawTemperature {
		if ClampTemperature(t) != t {
			out = append(out, fmt.Sprintf("temperature[%d]=%.4g", i, t))
		}
	}
	for k, v := range c.rawTempByOption {
		if ClampTemperature(v) != v {
			out = append(out, fmt.Sprintf("%s=%.4g", k, v))
		}
	}
	return out
}

// ClampTemperature confines a fitted temperature to a usable range, falling back
// to 1.0 when it is not a number.
func ClampTemperature(t float64) float64 {
	if t != t || t == 0 || t > 1e308 || t < -1e308 { // NaN / inf / zero
		return 1.0
	}
	if t < TempMin {
		return TempMin
	}
	if t > TempMax {
		return TempMax
	}
	return t
}

// TempBucket names the calibration bucket for a question type and option count.
// Mirrors temp_bucket() in laya/common.py.
func TempBucket(qtype string, k int) string {
	var size string
	switch {
	case k <= 2:
		size = "2"
	case k <= 5:
		size = "3-5"
	case k <= 10:
		size = "6-10"
	default:
		size = "11+"
	}
	return qtype + ":" + size
}

// TemperatureFor picks the scale for one question, preferring the option-count
// bucket and falling back to the per-type value.
func (c *ModelConfig) TemperatureFor(qtype string, k int) float64 {
	if c == nil {
		return 1.0
	}
	if v, ok := c.TempByOption[TempBucket(qtype, k)]; ok {
		return v
	}
	idx := map[string]int{"choice": 0, "score": 1, "noul": 2}[qtype]
	if idx < len(c.Temperature) {
		return c.Temperature[idx]
	}
	return 1.0
}
