//go:build !windows

package app

import (
	"os/exec"
	"runtime"
)

// openInFileManager reveals a directory in the platform's file manager.
func openInFileManager(dir string) error {
	switch runtime.GOOS {
	case "darwin":
		return exec.Command("open", dir).Start()
	default:
		return exec.Command("xdg-open", dir).Start()
	}
}
