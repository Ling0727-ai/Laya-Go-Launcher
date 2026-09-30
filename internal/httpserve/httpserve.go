// Package httpserve owns a REST listener. It is separate from internal/app so
// the headless server does not link the Wails runtime.
package httpserve

import (
	"fmt"
	"log"
	"net"
	"net/http"
	"strconv"
	"time"
)

// Server owns the listener.
type Server struct {
	addr     string
	srv      *http.Server
	listener net.Listener
	// Strict refuses to move to another port when addr is taken.
	Strict bool
}

// New prepares a listener; binding happens in Start.
func New(addr string, handler http.Handler) *Server {
	return &Server{
		addr: addr,
		srv: &http.Server{
			Handler:           handler,
			ReadHeaderTimeout: 5 * time.Second,
			ReadTimeout:       30 * time.Second,
			WriteTimeout:      15 * time.Minute, // a first-time 8192 ONNX load can be slow
			IdleTimeout:       60 * time.Second,
		},
	}
}

// Start binds and serves in the background. A taken port moves to the next
// free one unless Strict is set; Addr reports what was bound.
func (h *Server) Start() error {
	ln, err := net.Listen("tcp", h.addr)
	if err != nil {
		if h.Strict {
			return fmt.Errorf("listen on %s: %w", h.addr, err)
		}
		fallback, ferr := listenNextFree(h.addr, 10)
		if ferr != nil {
			return fmt.Errorf("listen on %s: %w (and no fallback port was free)", h.addr, err)
		}
		log.Printf("http: %s is in use, using %s instead", h.addr, fallback.Addr())
		ln = fallback
	}
	h.listener = ln
	h.addr = ln.Addr().String()
	go func() { _ = h.srv.Serve(ln) }()
	return nil
}

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

// Stop closes the listener.
func (h *Server) Stop() {
	if h.srv != nil {
		_ = h.srv.Close()
	}
}

// Addr is the bound address.
func (h *Server) Addr() string { return h.addr }
