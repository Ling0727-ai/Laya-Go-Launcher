//go:build !windows

package nativeenv

// Prepare is a no-op off Windows: the shared libraries are found through the
// dynamic linker's own configuration (rpath, ld.so.conf, LD_LIBRARY_PATH).
func Prepare() Dirs { return Dirs{} }
