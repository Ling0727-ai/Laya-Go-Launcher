// Package webui is the server's built-in console: a single static page that
// drives the admin API (config, model catalog, fallback chain, conversion) and
// a quick predict form. It has no build step and no external assets, so the
// pure server binary stays self-contained.
package webui

import (
	"embed"
	"io/fs"
	"net/http"
)

//go:embed static
var static embed.FS

// Handler serves the console. Unknown paths fall through to index.html.
func Handler() http.Handler {
	sub, _ := fs.Sub(static, "static")
	files := http.FileServer(http.FS(sub))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Content-Security-Policy",
			"default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'")
		if r.URL.Path != "/" {
			if _, err := fs.Stat(sub, r.URL.Path[1:]); err != nil {
				r.URL.Path = "/"
			}
		}
		files.ServeHTTP(w, r)
	})
}
