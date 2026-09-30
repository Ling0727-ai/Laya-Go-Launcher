// Command layatrt-server is the pure server build: HTTP API plus a built-in web
// console at /, no desktop shell and no Wails dependency.
//
// With no arguments it starts from the defaults: find the laya model that
// accepts 8192 tokens and load it on the first kernel that works, in the order
// tensorrt -> onnx-cuda -> onnx-directml -> onnx-cpu. When no TensorRT plan
// exists for the target context it builds one in the background with trtexec
// and switches to it when the build finishes.
//
//	layatrt-server                              defaults, console on http://127.0.0.1:8420/
//	layatrt-server --engine m.engine            pin a model (fallback still applies)
//	layatrt-server --backend onnx --provider cpu  one kernel, no fallback
//	layatrt-server --fallback onnx-cuda,onnx-cpu  custom chain
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/local/laya-go-launcher/internal/config"
	"github.com/local/laya-go-launcher/internal/httpserve"
	"github.com/local/laya-go-launcher/internal/launcher"
	"github.com/local/laya-go-launcher/internal/nativeenv"
	"github.com/local/laya-go-launcher/internal/transport/httpapi"
	"github.com/local/laya-go-launcher/internal/webui"
)

var version = "0.2.0"

func main() {
	var (
		cfgPath     = flag.String("config", "", "config file (default $LAYA_TRT_CONFIG or ./layatrt.config.json)")
		enginePath  = flag.String("engine", "", "model file to try first (.engine or .onnx)")
		addr        = flag.String("addr", "", "listen address (default 127.0.0.1:8420)")
		tokPath     = flag.String("tokenizer", "", "tokenizer.json path (default: HF cache)")
		contexts    = flag.Int("contexts", 0, "pre-allocated TensorRT contexts (0 = from free VRAM)")
		backendName = flag.String("backend", "", "auto (fallback chain), tensorrt or onnx")
		provider    = flag.String("provider", "", "ONNX provider for --backend onnx: cuda, directml or cpu")
		fallback    = flag.String("fallback", "", "comma-separated chain, e.g. tensorrt,onnx-cuda,onnx-directml,onnx-cpu")
		seq         = flag.Int("seq", 0, "context the model must accept (default 8192)")
		noConvert   = flag.Bool("no-convert", false, "never build TensorRT plans automatically")
		noLoad      = flag.Bool("no-load", false, "start without loading a model")
		diag        = flag.Bool("diagnostics", false, "enable native TensorRT diagnostics")
		writeCfg    = flag.Bool("write-config", false, "write the effective config to the config file and exit")
		showVer     = flag.Bool("version", false, "print version and exit")
	)
	flag.Parse()
	if *showVer {
		fmt.Printf("layatrt-server %s\n", version)
		return
	}
	if *cfgPath != "" {
		os.Setenv("LAYA_TRT_CONFIG", *cfgPath)
	}
	path := config.Path()

	cfg, err := config.ReadFile(path)
	if err != nil {
		log.Fatalf("config: %v", err)
	}
	cfg.ApplyEnv()
	set := func(dst *string, v string) {
		if v != "" {
			*dst = v
		}
	}
	set(&cfg.EnginePath, *enginePath)
	set(&cfg.HTTPAddr, *addr)
	set(&cfg.TokenizerPath, *tokPath)
	set(&cfg.Backend, *backendName)
	set(&cfg.Provider, *provider)
	if *contexts != 0 {
		cfg.Contexts = *contexts
	}
	if *seq != 0 {
		cfg.ModelSeq = *seq
	}
	if *fallback != "" {
		cfg.Fallback = nil
		for _, s := range strings.Split(*fallback, ",") {
			if s = config.NormalizeStep(s); s != "" {
				cfg.Fallback = append(cfg.Fallback, s)
			}
		}
	}
	if *noConvert {
		cfg.AutoConvert = false
	}
	if *diag {
		cfg.Diagnostics = true
	}
	if err := cfg.Validate(); err != nil {
		log.Fatalf("config: %v", err)
	}
	if *writeCfg {
		if err := config.Write(path, cfg); err != nil {
			log.Fatalf("write config: %v", err)
		}
		fmt.Println(path)
		return
	}

	log.Printf("layatrt-server %s", version)
	log.Printf("native: %s", nativeenv.Prepare())
	log.Printf("config: %s (seq=%d backend=%s fallback=%s auto_convert=%v)",
		path, cfg.ModelSeq, cfg.Backend, strings.Join(cfg.Fallback, ">"), cfg.AutoConvert)

	l, err := launcher.New(cfg, path, version)
	if err != nil {
		log.Fatalf("startup: %v", err)
	}
	defer l.Close()
	log.Printf("tokenizer: %s", l.TokenizerPath())

	api := httpapi.New(l.Service, l.Metrics, version, l.Switcher).WithLauncher(l).WithUI(webui.Handler())
	srv := httpserve.New(cfg.HTTPAddr, api.Handler())
	// Clients (and AI callers) hard-code the address; moving to another port
	// silently would send them to nothing, so a clash is fatal here.
	srv.Strict = true
	if err := srv.Start(); err != nil {
		log.Fatalf("%v", err)
	}
	log.Printf("console: http://%s/", srv.Addr())
	log.Printf("api:     http://%s/api/v1  (openapi: /api/v1/openapi.json)", srv.Addr())
	if cfg.AdminToken == "" && !loopback(srv.Addr()) {
		log.Printf("warning: listening on %s without admin_token; anyone on the network can reconfigure this server", srv.Addr())
	}

	// Load in the background so the console is reachable while a slow ONNX
	// session builds; /api/v1/load reports progress.
	if !*noLoad {
		go func() {
			if _, err := l.AutoLoad(context.Background()); err != nil {
				log.Printf("load: %v (the console can pick another model)", err)
			}
		}()
	}

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	<-stop
	log.Printf("shutting down")
	srv.Stop()
}

func loopback(addr string) bool {
	return strings.HasPrefix(addr, "127.") || strings.HasPrefix(addr, "[::1]") || strings.HasPrefix(addr, "localhost")
}
