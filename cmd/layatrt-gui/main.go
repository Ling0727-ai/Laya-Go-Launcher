// Command layatrt-gui is the Wails desktop application.
//
// It runs the same launcher as layatrt-server (default config, fallback chain
// tensorrt -> onnx-cuda -> onnx-directml -> onnx-cpu, background TensorRT
// conversion) and serves the same HTTP API and web console in-process, so the
// window, the console and a REST client all see one model.
package main

import (
	"embed"
	"log"

	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"

	"github.com/local/laya-go-launcher/internal/app"
	"github.com/local/laya-go-launcher/internal/config"
	"github.com/local/laya-go-launcher/internal/httpserve"
	"github.com/local/laya-go-launcher/internal/launcher"
	"github.com/local/laya-go-launcher/internal/nativeenv"
	"github.com/local/laya-go-launcher/internal/transport/httpapi"
	"github.com/local/laya-go-launcher/internal/webui"
)

//go:embed all:frontend/dist
var assets embed.FS

// version is overridable at link time: -ldflags "-X main.version=1.2.3"
var version = "0.2.0"

func main() {
	log.Printf("native: %s", nativeenv.Prepare())

	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("config: %v", err)
	}
	l, err := launcher.New(cfg, config.Path(), version)
	if err != nil {
		// Without a tokenizer nothing can predict; say so in a dialog-free way
		// and stop, rather than opening a window that cannot do anything.
		log.Fatalf("startup: %v", err)
	}
	defer l.Close()
	log.Printf("tokenizer: %s", l.TokenizerPath())

	api := httpapi.New(l.Service, l.Metrics, version, l.Switcher).WithLauncher(l).WithUI(webui.Handler())
	httpSrv := httpserve.New(cfg.HTTPAddr, api.Handler())
	if err := httpSrv.Start(); err != nil {
		log.Fatalf("http api: %v", err)
	}
	defer httpSrv.Stop()
	log.Printf("HTTP API + console: http://%s/", httpSrv.Addr())

	bindings := app.NewBindings(l.Service, l.Metrics, version, l.Switcher)
	app.AttachLauncher(bindings, l, httpSrv.Addr())

	err = wails.Run(&options.App{
		Title:            "Laya Go Launcher",
		Width:            1240,
		Height:           840,
		MinWidth:         960,
		MinHeight:        640,
		BackgroundColour: &options.RGBA{R: 15, G: 18, B: 22, A: 255},
		AssetServer: &assetserver.Options{
			Assets: assets,
		},
		OnStartup:  bindings.Startup,
		OnShutdown: bindings.Shutdown,
		Bind:       []any{bindings},
	})
	if err != nil {
		log.Fatalf("wails: %v", err)
	}
}
