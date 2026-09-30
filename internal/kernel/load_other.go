//go:build !windows

package kernel

// ensureLoaded is a no-op off Windows, where the kernel is linked normally.
func ensureLoaded() error { return nil }
