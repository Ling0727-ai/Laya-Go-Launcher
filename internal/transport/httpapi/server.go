// Package httpapi exposes the inference use cases over HTTP.
//
// It is a thin transport: decode, call internal/inference, encode. No model
// logic lives here, so the same behaviour is reachable from the Wails bindings
// and from any other process that speaks JSON.
package httpapi

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/local/laya-go-launcher/internal/backend"
	"github.com/local/laya-go-launcher/internal/backends"
	"github.com/local/laya-go-launcher/internal/inference"
	"github.com/local/laya-go-launcher/internal/launcher"
	"github.com/local/laya-go-launcher/internal/metrics"
	"github.com/local/laya-go-launcher/internal/trtbackend"
)

// maxBodyBytes bounds request bodies so a malformed client cannot exhaust
// memory. The largest legitimate body is a long document plus a schema.
const maxBodyBytes = 1 << 20

// Server wires the HTTP surface to the use cases.
type Server struct {
	inference *inference.Service
	metrics   *metrics.Recorder
	version   string
	started   time.Time

	// switcher owns the active kernel, so a load performed here is the model the
	// next predict call runs on, and a per-request backend choice can swap the
	// kernel in place.
	switcher *backends.Switcher
	// kernelCfg is the configured default selection.
	kernelCfg backends.Config

	// launcher, when set, adds the admin surface (see admin.go).
	launcher *launcher.Launcher
	// ui serves the embedded web console at /, when set.
	ui http.Handler
}

// New builds a server.
func New(svc *inference.Service, rec *metrics.Recorder, version string, switcher *backends.Switcher) *Server {
	return &Server{
		inference: svc,
		metrics:   rec,
		version:   version,
		started:   time.Now(),
		switcher:  switcher,
		kernelCfg: switcher.Config(),
	}
}

// backend is the resident execution kernel.
func (s *Server) backend() backend.Backend { return s.inference.Backend() }

// Handler returns the routed http.Handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /api/v1/health", s.handleHealth)
	mux.HandleFunc("GET /api/v1/device", s.handleDevice)
	mux.HandleFunc("GET /api/v1/backends", s.handleBackends)
	mux.HandleFunc("GET /api/v1/engine", s.handleEngineGet)
	mux.HandleFunc("POST /api/v1/engine/load", s.handleEngineLoad)
	mux.HandleFunc("POST /api/v1/engine/unload", s.handleEngineUnload)
	mux.HandleFunc("POST /api/v1/predict", s.handlePredict)
	mux.HandleFunc("POST /api/v1/tokenize", s.handleTokenize)
	mux.HandleFunc("POST /api/v1/sequence", s.handleSequence)
	mux.HandleFunc("GET /api/v1/metrics", s.handleMetrics)
	mux.HandleFunc("GET /api/v1/limits", s.handleLimits)
	mux.HandleFunc("GET /api/v1/openapi.json", s.handleOpenAPI)
	s.registerAdmin(mux)
	if s.ui != nil {
		mux.Handle("GET /", s.ui)
	}

	return withRecovery(withCORS(mux))
}

// ── handlers ────────────────────────────────────────────────────────────────

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	probe := backends.Probe(s.switcher.Config())
	dev := trtbackend.DeviceInfo()
	vram := trtbackend.VRAMInfo()

	engInfo := map[string]any{"loaded": false, "path": "", "contexts": 0}
	if info, ok := s.backend().Info(); ok {
		engInfo = map[string]any{
			"loaded":   true,
			"path":     info.Path,
			"backend":  info.Backend,
			"contexts": info.Contexts,
		}
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"status":   "ok",
		"version":  s.version,
		"uptime_s": int(time.Since(s.started).Seconds()),
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
		"engine": engInfo,
		"load":   s.loadState(),
	})
}

func (s *Server) handleDevice(w http.ResponseWriter, r *http.Request) {
	dev := trtbackend.DeviceInfo()
	vram := trtbackend.VRAMInfo()
	writeJSON(w, http.StatusOK, map[string]any{
		"name":               dev.Name,
		"compute_capability": fmt.Sprintf("%d.%d", dev.ComputeMajor, dev.ComputeMinor),
		"vram_free_mb":       vram.FreeMB,
		"vram_total_mb":      vram.TotalMB,
	})
}

// handleBackends reports which kernels this build and machine can offer,
// without loading a model, so a client can choose before committing.
func (s *Server) handleBackends(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, backends.Probe(s.switcher.Config()))
}

func (s *Server) handleEngineGet(w http.ResponseWriter, r *http.Request) {
	info, ok := s.backend().Info()
	if !ok {
		writeError(w, http.StatusNotFound, "engine_not_loaded", "no model is loaded")
		return
	}
	writeJSON(w, http.StatusOK, s.engineBody(info))
}

type loadRequest struct {
	Path     string `json:"path"`
	Contexts int    `json:"contexts"`
	// Backend overrides the configured kernel for this load: "tensorrt" or
	// "onnx". Empty uses the process configuration.
	Backend string `json:"backend"`
	// Provider overrides the ONNX Runtime execution provider.
	Provider string `json:"provider"`
}

func (s *Server) handleEngineLoad(w http.ResponseWriter, r *http.Request) {
	var req loadRequest
	if !decodeBody(w, r, &req) {
		return
	}
	if strings.TrimSpace(req.Path) == "" {
		writeError(w, http.StatusBadRequest, "invalid_request", "path is required")
		return
	}

	// A per-request kernel choice must still be a valid one, and it applies to
	// this load only: it travels as an option, not as a new process default.
	// Folding it into the switcher's configuration is what used to pin the whole
	// process to one kernel after a single request named it.
	cfg := s.switcher.Config()
	opts := backend.Options{
		Path:     req.Path,
		Contexts: req.Contexts,
	}
	if req.Backend != "" {
		kind, err := backend.ParseKind(req.Backend)
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid_backend", err.Error())
			return
		}
		opts.Kind = kind
	}
	opts.Provider = req.Provider
	if s.launcher != nil {
		s.loadViaLauncher(w, r, req)
		return
	}

	// Load through the switcher, so the model becomes the one the service
	// predicts on. Loading a fresh backend here would leave it unreachable.
	info, note, err := s.switcher.LoadWith(r.Context(), cfg, opts)
	if err != nil {
		code, status := classifyEngineError(err)
		writeError(w, status, code, err.Error())
		return
	}
	// Fit the token budgets to this model's own shapes before anything uses them.
	s.inference.AdaptToEngine()
	s.metrics.RecordEngineLoad()
	body := s.engineBody(info)
	if note != "" {
		body["backend_note"] = note
	}
	writeJSON(w, http.StatusOK, body)
}

// engineBody is the model description plus the limits derived from it, so a
// client can see what the model accepts without having to probe for it.
func (s *Server) engineBody(info backend.Info) map[string]any {
	body := engineInfoBody(info)
	maxLen, headMaxLen, padLen := s.inference.Budgets()
	body["budgets"] = map[string]any{
		"max_len":      maxLen,
		"head_max_len": headMaxLen,
		"pad_len":      padLen,
	}
	body["marker_capacity"] = s.inference.MarkerCapacity()
	return body
}

func (s *Server) handleEngineUnload(w http.ResponseWriter, r *http.Request) {
	s.backend().Unload()
	writeJSON(w, http.StatusOK, map[string]any{"unloaded": true})
}

func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	var req struct {
		State     any                               `json:"state"`
		Questions map[string]inference.QuestionSpec `json:"questions"`
	}
	if !decodeBody(w, r, &req) {
		return
	}
	if len(req.Questions) == 0 {
		writeError(w, http.StatusBadRequest, "invalid_request", "questions must not be empty")
		return
	}
	if !s.backend().Loaded() {
		writeError(w, http.StatusConflict, "engine_not_loaded", "load a model before predicting")
		return
	}

	start := time.Now()
	resp, err := s.inference.Predict(r.Context(), inference.Request{
		State:     req.State,
		Questions: req.Questions,
	})
	if err != nil {
		s.metrics.RecordFailure()
		status, code := http.StatusInternalServerError, "inference_failed"
		if strings.Contains(err.Error(), "sequence:") ||
			strings.Contains(err.Error(), "instructions are required") ||
			strings.Contains(err.Error(), "criteria") {
			status, code = http.StatusBadRequest, "invalid_request"
		}
		writeError(w, status, code, err.Error())
		return
	}

	s.metrics.RecordPredict(time.Since(start), resp.Timing.TotalMS, resp.Timing.TokenizeMS, resp.Timing.InferenceMS)
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleTokenize(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Text         string `json:"text"`
		WithSpecials bool   `json:"with_specials"`
	}
	if !decodeBody(w, r, &req) {
		return
	}
	tk := s.inference.Tokenizer()
	ids := tk.Encode(req.Text)
	if req.WithSpecials {
		ids = tk.EncodeWithSpecials(req.Text)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"count":  len(ids),
		"ids":    ids,
		"tokens": tk.Tokens(ids),
	})
}

func (s *Server) handleSequence(w http.ResponseWriter, r *http.Request) {
	var req struct {
		State    any                    `json:"state"`
		Question inference.QuestionSpec `json:"question"`
	}
	if !decodeBody(w, r, &req) {
		return
	}
	q, err := inference.BuildQuestion(req.Question)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	maxLen, headMaxLen, _ := s.inference.Budgets()
	built, err := sequenceBuild(s, req.State, q, maxLen, headMaxLen)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"ids":              built.IDs,
		"length":           len(built.IDs),
		"marker_positions": built.MarkerPositions,
		"options":          built.Options,
		"truncated":        built.Truncated,
	})
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.metrics.Snapshot())
}

// handleLimits reports what the loaded engine accepts and what the service is
// using, so a client can size its questions without probing.
func (s *Server) handleLimits(w http.ResponseWriter, r *http.Request) {
	if !s.backend().Loaded() {
		writeError(w, http.StatusNotFound, "engine_not_loaded", "no model is loaded")
		return
	}
	writeJSON(w, http.StatusOK, s.inference.Limits())
}

// handleOpenAPI serves the machine-readable contract. docs/api.md is the
// narrative version; this is the one a client generator consumes.
func (s *Server) handleOpenAPI(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, openAPISpec(s.version))
}

// ── helpers ─────────────────────────────────────────────────────────────────

func engineInfoBody(info backend.Info) map[string]any {
	body := map[string]any{
		"loaded":               true,
		"path":                 info.Path,
		"backend":              info.Backend,
		"contexts":             info.Contexts,
		"activation_memory_mb": round2(info.ActivationMemoryMB),
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

func decodeBody(w http.ResponseWriter, r *http.Request, dst any) bool {
	defer r.Body.Close()
	limited := io.LimitReader(r.Body, maxBodyBytes+1)
	raw, err := io.ReadAll(limited)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "cannot read request body")
		return false
	}
	if len(raw) > maxBodyBytes {
		writeError(w, http.StatusRequestEntityTooLarge, "payload_too_large",
			fmt.Sprintf("request body exceeds %d bytes", maxBodyBytes))
		return false
	}
	if len(raw) == 0 {
		writeError(w, http.StatusBadRequest, "invalid_request", "request body is empty")
		return false
	}
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "malformed JSON: "+err.Error())
		return false
	}
	return true
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	// Encode into a buffer first: encoding/json refuses NaN and ±Inf, and a
	// partial write followed by an error would otherwise leave the client with a
	// 200 and an empty body.
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(body); err != nil {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]string{
				"code":    "encode_failed",
				"message": "response could not be encoded: " + err.Error(),
			},
		})
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_, _ = w.Write(buf.Bytes())
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]string{"code": code, "message": message},
	})
}

// classifyEngineError maps a load failure onto a wire code and status.
//
// The distinctions matter to a client: "this machine cannot run that kernel" is
// fixable by installing a runtime, "that file is not a laya model" is not, and
// "there was not enough memory" is fixed by unloading or by a smaller model.
func classifyEngineError(err error) (string, int) {
	switch {
	case errors.Is(err, backend.ErrNotLoaded):
		return "engine_not_loaded", http.StatusConflict
	case errors.Is(err, backend.ErrClosed):
		return "manager_closed", http.StatusConflict
	case errors.Is(err, backend.ErrRuntimeMissing):
		return "runtime_missing", http.StatusServiceUnavailable
	case errors.Is(err, backend.ErrWrongFormat):
		return "engine_wrong_kernel", http.StatusConflict
	case errors.Is(err, backend.ErrIncompatible):
		return "engine_incompatible", http.StatusConflict
	case errors.Is(err, os.ErrNotExist):
		return "engine_not_found", http.StatusNotFound
	case backend.IsOutOfMemory(err):
		return "out_of_memory", http.StatusInsufficientStorage
	}
	msg := err.Error()
	switch {
	case strings.Contains(msg, "no such file"):
		return "engine_not_found", http.StatusNotFound
	case strings.Contains(msg, "deserialize"):
		return "engine_incompatible", http.StatusConflict
	case strings.Contains(msg, "onnxruntime.dll"):
		return "runtime_missing", http.StatusServiceUnavailable
	}
	return "load_failed", http.StatusInternalServerError
}

func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The GUI and the HTTP API share a process; a browser page served from
		// a dev server needs these to call the API directly.
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func withRecovery(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				writeError(w, http.StatusInternalServerError, "internal_error",
					fmt.Sprintf("panic: %v", rec))
			}
		}()
		next.ServeHTTP(w, r)
	})
}

func round2(v float64) float64 {
	return float64(int(v*100+0.5)) / 100
}
