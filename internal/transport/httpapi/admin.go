package httpapi

import (
	"context"
	"crypto/subtle"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/local/laya-go-launcher/internal/config"
	"github.com/local/laya-go-launcher/internal/launcher"
)

// WithLauncher enables the admin surface: config, model catalog, the auto
// loader and TensorRT conversion. Loads made through /engine/load then go
// through the launcher, so its load state stays accurate.
func (s *Server) WithLauncher(l *launcher.Launcher) *Server {
	s.launcher = l
	return s
}

func (s *Server) registerAdmin(mux *http.ServeMux) {
	if s.launcher == nil {
		return
	}
	mux.HandleFunc("GET /api/v1/config", s.handleConfigGet)
	mux.HandleFunc("PUT /api/v1/config", s.admin(s.handleConfigPut))
	mux.HandleFunc("GET /api/v1/models", s.handleModels)
	mux.HandleFunc("GET /api/v1/models/plan", s.handleModelsPlan)
	mux.HandleFunc("POST /api/v1/models/rescan", s.handleModelsRescan)
	mux.HandleFunc("GET /api/v1/load", s.handleLoadState)
	mux.HandleFunc("POST /api/v1/load/auto", s.admin(s.handleAutoLoad))
	mux.HandleFunc("GET /api/v1/convert", s.handleConvertList)
	mux.HandleFunc("POST /api/v1/convert", s.admin(s.handleConvertStart))
	mux.HandleFunc("POST /api/v1/convert/cancel", s.admin(s.handleConvertCancel))
}

// admin guards a mutating endpoint with the configured bearer token. Without a
// token the server is expected to listen on loopback only (the default).
func (s *Server) admin(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !sameOrigin(r) {
			writeError(w, http.StatusForbidden, "cross_origin",
				"admin endpoints only accept same-origin browser requests")
			return
		}
		if s.launcher != nil {
			if token := s.launcher.Config().AdminToken; token != "" {
				got := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
				if subtle.ConstantTimeCompare([]byte(got), []byte(token)) != 1 {
					writeError(w, http.StatusUnauthorized, "unauthorized",
						"this endpoint needs Authorization: Bearer <admin_token>")
					return
				}
			}
		}
		next(w, r)
	}
}

// WithUI serves an embedded web console at / (see internal/webui).
func (s *Server) WithUI(h http.Handler) *Server {
	s.ui = h
	return s
}

// loadState is the launcher's load state for /health, or nil.
func (s *Server) loadState() any {
	if s.launcher == nil {
		return nil
	}
	return s.launcher.State()
}

// sameOrigin rejects a browser request from another site. The API sends
// Access-Control-Allow-Origin: * for predict, so without this any web page the
// user visits could reconfigure a loopback server or start a long build.
func sameOrigin(r *http.Request) bool {
	origin := r.Header.Get("Origin")
	if origin == "" {
		return true // not a browser, or a same-origin GET
	}
	u, err := url.Parse(origin)
	if err != nil {
		return false
	}
	if strings.EqualFold(u.Host, r.Host) {
		return true
	}
	// wails:// and the Wails dev server talk to the in-process API.
	return u.Scheme == "wails" || strings.HasPrefix(u.Host, "wails.localhost") ||
		u.Host == "localhost:5188" || u.Host == "localhost:34116"
}

func (s *Server) handleConfigGet(w http.ResponseWriter, r *http.Request) {
	cfg := s.launcher.Config()
	writeJSON(w, http.StatusOK, map[string]any{
		"config":     cfg.Redacted(),
		"path":       s.launcher.ConfigPath(),
		"tokenizer":  s.launcher.TokenizerPath(),
		"defaults":   config.Default(),
		"steps":      config.DefaultFallback,
		"model_dirs": s.launcher.Catalog.Dirs(),
	})
}

type configPut struct {
	config.Config
	// Reload runs the auto loader after saving.
	Reload bool `json:"reload"`
}

func (s *Server) handleConfigPut(w http.ResponseWriter, r *http.Request) {
	cur := s.launcher.Config()
	req := configPut{Config: cur}
	if !decodeBody(w, r, &req) {
		return
	}
	next := req.Config
	for i, step := range next.Fallback {
		next.Fallback[i] = config.NormalizeStep(step)
	}
	if err := s.launcher.UpdateConfig(next); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_config", err.Error())
		return
	}
	body := map[string]any{"saved": true, "config": s.launcher.Config().Redacted()}
	if next.HTTPAddr != cur.HTTPAddr {
		body["restart_required"] = "http_addr takes effect after a restart"
	}
	if req.Reload {
		go func() { _, _ = s.launcher.AutoLoad(context.Background()) }()
		body["reloading"] = true
	}
	writeJSON(w, http.StatusOK, body)
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"dirs":   s.launcher.Catalog.Dirs(),
		"models": s.launcher.Models(),
	})
}

func (s *Server) handleModelsPlan(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"attempts": s.launcher.PlanAttempts()})
}

func (s *Server) handleModelsRescan(w http.ResponseWriter, r *http.Request) {
	s.launcher.Catalog.Invalidate()
	s.handleModels(w, r)
}

func (s *Server) handleLoadState(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.launcher.State())
}

func (s *Server) handleAutoLoad(w http.ResponseWriter, r *http.Request) {
	// Walking the chain can take tens of seconds; run it detached from the
	// request so a client timeout does not abort a load half-way.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	st, err := s.launcher.AutoLoad(ctx)
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{
			"error": map[string]string{"code": "load_failed", "message": err.Error()},
			"state": st,
		})
		return
	}
	writeJSON(w, http.StatusOK, st)
}

func (s *Server) handleConvertList(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"jobs": s.launcher.Converter.Jobs()})
}

func (s *Server) handleConvertStart(w http.ResponseWriter, r *http.Request) {
	var req launcher.ConvertRequest
	if r.ContentLength != 0 && !decodeBody(w, r, &req) {
		return
	}
	job, err := s.launcher.StartConversion(req)
	if err != nil {
		status := http.StatusBadRequest
		if strings.Contains(err.Error(), "already running") {
			status = http.StatusConflict
		}
		writeError(w, status, "convert_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, job)
}

func (s *Server) handleConvertCancel(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"canceled": s.launcher.Converter.Cancel()})
}

// loadViaLauncher is /engine/load when a launcher is attached.
func (s *Server) loadViaLauncher(w http.ResponseWriter, r *http.Request, req loadRequest) {
	st, err := s.launcher.Load(r.Context(), req.Path, req.Backend, req.Provider, req.Contexts)
	if err != nil {
		code, status := classifyEngineError(err)
		writeJSON(w, status, map[string]any{
			"error": map[string]string{"code": code, "message": err.Error()},
			"state": st,
		})
		return
	}
	info, _ := s.backend().Info()
	body := s.engineBody(info)
	body["state"] = st
	if st.Note != "" {
		body["backend_note"] = st.Note
	}
	writeJSON(w, http.StatusOK, body)
}
