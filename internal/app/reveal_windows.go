//go:build windows

package app

import "os/exec"

// openInFileManager reveals a directory in Explorer.
func openInFileManager(dir string) error {
	return exec.Command("explorer", dir).Start()
}
