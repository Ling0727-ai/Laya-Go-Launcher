// Package modelstore finds laya model files and ranks them for a load.
//
// Discovery is metadata-only: a candidate is described from its file name, the
// engines/manifest.json written by bench/build-engines.ps1, and the
// <file>.meta.json sidecar written by the converter. Nothing is deserialized, so
// a scan of several multi-hundred-megabyte plans costs a few stat calls instead
// of loading each one onto the GPU. Scans are cached and invalidated by the
// directory listing's modification times.
package modelstore

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Format is a model file format.
type Format string

const (
	FormatEngine Format = "engine"
	FormatONNX   Format = "onnx"
)

// Model is one model file with what could be learned about it cheaply.
type Model struct {
	Path     string    `json:"path"`
	Name     string    `json:"name"`
	Dir      string    `json:"dir"`
	Format   Format    `json:"format"`
	SizeMB   float64   `json:"size_mb"`
	Modified time.Time `json:"modified"`

	// SeqMax is the longest sequence the model accepts; 0 when unknown.
	SeqMax int `json:"seq_max"`
	// BatchMax is the batch ceiling of the largest profile; 0 when unknown.
	BatchMax int `json:"batch_max"`
	// MultiProfile is a TensorRT plan with a batched short profile and a
	// batch-1 long profile (the fastest layout for 8192).
	MultiProfile bool `json:"multi_profile"`
	// Precision is "fp16", "fp32" or "".
	Precision string `json:"precision"`
	// Optimized is an ORT-fused graph (tools/optimize_onnx.py). Fused graphs
	// carry com.microsoft contrib ops, so they are not TensorRT build sources.
	Optimized bool `json:"optimized"`
	// Source is the ONNX graph an engine was built from, when recorded.
	Source string `json:"source,omitempty"`
	// Origin says where the metadata came from: "name", "manifest", "sidecar".
	Origin string `json:"origin"`
}

// Sidecar is the <engine>.meta.json the converter writes next to a plan.
type Sidecar struct {
	Source       string    `json:"source"`
	SourceSize   int64     `json:"source_size"`
	SourceMTime  time.Time `json:"source_mtime"`
	SeqMax       int       `json:"seq_max"`
	BatchMax     int       `json:"batch_max"`
	MultiProfile bool      `json:"multi_profile"`
	Precision    string    `json:"precision"`
	TensorRT     string    `json:"tensorrt,omitempty"`
	BuiltAt      time.Time `json:"built_at"`
	BuildSeconds float64   `json:"build_seconds"`
}

// SidecarPath is where the metadata for an engine is stored.
func SidecarPath(enginePath string) string { return enginePath + ".meta.json" }

// WriteSidecar records how an engine was built.
func WriteSidecar(enginePath string, s Sidecar) error {
	raw, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(SidecarPath(enginePath), raw, 0o644)
}

var (
	reEngine = regexp.MustCompile(`(?i)_s(\d+)_(fp16|fp32)_(?:b(\d+)|p(\d+))`)
	reCtx    = regexp.MustCompile(`(?i)(?:ctx|_s|seq)(\d{3,5})`)
	reFP     = regexp.MustCompile(`(?i)\b?(fp16|fp32)\b?`)
)

// describe fills a Model from its file name.
func describe(path string, info os.FileInfo) Model {
	name := filepath.Base(path)
	m := Model{
		Path:     path,
		Name:     name,
		Dir:      filepath.Dir(path),
		SizeMB:   float64(info.Size()) / (1 << 20),
		Modified: info.ModTime(),
		Origin:   "name",
	}
	lower := strings.ToLower(name)
	if strings.HasSuffix(lower, ".onnx") {
		m.Format = FormatONNX
		m.Optimized = strings.Contains(lower, ".opt")
	} else {
		m.Format = FormatEngine
	}
	if g := reEngine.FindStringSubmatch(name); g != nil {
		m.SeqMax, _ = strconv.Atoi(g[1])
		m.Precision = strings.ToLower(g[2])
		if g[3] != "" {
			m.BatchMax, _ = strconv.Atoi(g[3])
		} else {
			m.MultiProfile = true
			m.BatchMax = 8
		}
	} else if g := reCtx.FindStringSubmatch(name); g != nil {
		m.SeqMax, _ = strconv.Atoi(g[1])
	}
	if m.Precision == "" {
		if g := reFP.FindStringSubmatch(lower); g != nil {
			m.Precision = strings.ToLower(g[1])
		} else if m.Format == FormatONNX {
			m.Precision = "fp32"
		}
	}
	return m
}

type manifestDoc struct {
	ONNX    string `json:"onnx"`
	Engines []struct {
		Name      string `json:"name"`
		Precision string `json:"precision"`
		SeqMax    int    `json:"seq_max"`
		BatchMax  int    `json:"batch_max"`
	} `json:"engines"`
}

// Catalog scans directories for models and caches the result.
type Catalog struct {
	mu    sync.Mutex
	dirs  []string
	cache []Model
	stamp string
	at    time.Time
	ttl   time.Duration
}

// NewCatalog builds a catalog over dirs (missing ones are ignored).
func NewCatalog(dirs []string) *Catalog {
	return &Catalog{dirs: dedupe(dirs), ttl: 10 * time.Second}
}

// SetDirs replaces the searched directories.
func (c *Catalog) SetDirs(dirs []string) {
	c.mu.Lock()
	c.dirs = dedupe(dirs)
	c.stamp = ""
	c.mu.Unlock()
}

// Dirs lists the directories that exist.
func (c *Catalog) Dirs() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []string
	for _, d := range c.dirs {
		if st, err := os.Stat(d); err == nil && st.IsDir() {
			out = append(out, d)
		}
	}
	return out
}

// Invalidate forces the next Models call to rescan.
func (c *Catalog) Invalidate() {
	c.mu.Lock()
	c.stamp = ""
	c.mu.Unlock()
}

// Models returns every model found, newest scan reused while the directories
// are unchanged.
func (c *Catalog) Models() []Model {
	c.mu.Lock()
	defer c.mu.Unlock()
	stamp := c.stampLocked()
	if stamp == c.stamp && c.cache != nil && time.Since(c.at) < c.ttl {
		return append([]Model(nil), c.cache...)
	}
	c.cache = scan(c.dirs)
	c.stamp = stamp
	c.at = time.Now()
	return append([]Model(nil), c.cache...)
}

func (c *Catalog) stampLocked() string {
	var b strings.Builder
	for _, d := range c.dirs {
		if st, err := os.Stat(d); err == nil {
			b.WriteString(d)
			b.WriteString(st.ModTime().String())
		}
	}
	return b.String()
}

func scan(dirs []string) []Model {
	out := []Model{}
	seen := map[string]bool{}
	for _, dir := range dirs {
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		manifest := readManifest(dir)
		for _, e := range entries {
			if e.IsDir() {
				continue
			}
			lower := strings.ToLower(e.Name())
			if !strings.HasSuffix(lower, ".engine") && !strings.HasSuffix(lower, ".onnx") {
				continue
			}
			full := filepath.Join(dir, e.Name())
			key := strings.ToLower(full)
			if seen[key] {
				continue
			}
			info, err := e.Info()
			if err != nil || info.Size() == 0 {
				// trtexec leaves a 0-byte file behind when a build fails.
				continue
			}
			seen[key] = true
			m := describe(full, info)
			if manifest != nil && m.Format == FormatEngine {
				for _, me := range manifest.Engines {
					if strings.EqualFold(me.Name, e.Name()) {
						m.SeqMax, m.BatchMax, m.Precision = me.SeqMax, me.BatchMax, me.Precision
						m.Source = manifest.ONNX
						m.Origin = "manifest"
					}
				}
			}
			if s, ok := readSidecar(full); ok {
				m.SeqMax, m.BatchMax, m.Precision = s.SeqMax, s.BatchMax, s.Precision
				m.MultiProfile = s.MultiProfile
				m.Source = s.Source
				m.Origin = "sidecar"
			}
			out = append(out, m)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out
}

func readManifest(dir string) *manifestDoc {
	raw, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil
	}
	raw = trimBOM(raw)
	var doc manifestDoc
	if json.Unmarshal(raw, &doc) != nil {
		return nil
	}
	return &doc
}

func readSidecar(path string) (Sidecar, bool) {
	raw, err := os.ReadFile(SidecarPath(path))
	if err != nil {
		return Sidecar{}, false
	}
	var s Sidecar
	if json.Unmarshal(trimBOM(raw), &s) != nil {
		return Sidecar{}, false
	}
	return s, true
}

func trimBOM(b []byte) []byte {
	if len(b) >= 3 && b[0] == 0xEF && b[1] == 0xBB && b[2] == 0xBF {
		return b[3:]
	}
	return b
}

func dedupe(dirs []string) []string {
	var out []string
	seen := map[string]bool{}
	for _, d := range dirs {
		if d == "" {
			continue
		}
		abs, err := filepath.Abs(d)
		if err != nil {
			abs = d
		}
		k := strings.ToLower(abs)
		if seen[k] {
			continue
		}
		seen[k] = true
		out = append(out, abs)
	}
	return out
}

// DefaultDirs is where models are looked for, most specific first: configured
// directories, then engines/ onnx/ models/ next to the working directory and the
// executable (and a few parents, so a binary under build/bin still finds the
// repository's folders), then ~/Documents/laya-trt.
func DefaultDirs(extra []string) []string {
	dirs := append([]string(nil), extra...)
	var roots []string
	if cwd, err := os.Getwd(); err == nil {
		roots = append(roots, cwd)
	}
	if exe, err := os.Executable(); err == nil {
		d := filepath.Dir(exe)
		for i := 0; i < 4; i++ {
			roots = append(roots, d)
			p := filepath.Dir(d)
			if p == d {
				break
			}
			d = p
		}
	}
	if home, err := os.UserHomeDir(); err == nil {
		roots = append(roots, filepath.Join(home, "Documents", "laya-trt"))
	}
	for _, r := range roots {
		dirs = append(dirs,
			filepath.Join(r, "engines"),
			filepath.Join(r, "onnx"),
			filepath.Join(r, "models"),
		)
	}
	return dedupe(dirs)
}
