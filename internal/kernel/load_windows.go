//go:build windows

package kernel

/*
#include <stdlib.h>
#include "kernel_loader.h"
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/local/laya-go-launcher/internal/nativeenv"
)

var (
	loadOnce sync.Once
	loadErr  error
)

// ensureLoaded binds qualityscaler_tensorrt.dll on first use. Discovery runs
// first so TensorRT, CUDA and cuDNN resolve without the caller's PATH.
func ensureLoaded() error {
	loadOnce.Do(func() {
		dirs := nativeenv.Prepare()
		path := dirs.KernelPath()
		if path == "" {
			path = nativeenv.KernelDLL
		}
		cPath := C.CString(path)
		defer C.free(unsafe.Pointer(cPath))
		var buf [1024]C.char
		if C.LayaKernel_Load(cPath, &buf[0], C.int(len(buf))) != 0 {
			loadErr = fmt.Errorf("tensorrt: %s (%s; build it with .\\build.ps1, or set TENSORRT_ROOT / CUDA_ROOT)",
				C.GoString(&buf[0]), dirs)
		}
	})
	return loadErr
}
