// Package nativeenv makes the native runtimes loadable from a plain shell.
//
// The TensorRT kernel, TensorRT itself, the CUDA runtime and cuDNN live in four
// unrelated install directories. Before this package existed the process relied
// on the caller's PATH to reach them (env.ps1, or $env:PATH by hand), and a PATH
// that was merely unlucky — the TensorRT directory missing, or an entry that is
// a file rather than a directory — killed the process before main() with
// 0xC0000279.
//
// Prepare discovers those directories once and puts them at the front of this
// process's PATH, so every later LoadLibrary (the kernel, the ONNX bridge, ONNX
// Runtime's CUDA provider and the cuBLAS/cuDNN it pulls in) resolves the same,
// matching set. It runs once per process, never per model load, so switching
// kernels or providers cannot change which libraries a later load sees.
//
// Explicit settings win over discovery:
//
//	LAYA_TRT_KERNEL_DIR   directory holding qualityscaler_tensorrt.dll
//	TENSORRT_ROOT         TensorRT install (bin\ or lib\ holds nvinfer_10.dll)
//	CUDA_ROOT / CUDA_PATH CUDA toolkit
//	CUDNN_PATH            cuDNN install
package nativeenv

import (
	"fmt"
	"strings"
)

// KernelDLL is the TensorRT kernel's file name.
const KernelDLL = "qualityscaler_tensorrt.dll"

// Dirs is what discovery settled on. An empty field means "not found"; the load
// that needs it then reports its own, specific error.
type Dirs struct {
	Kernel   string
	TensorRT string
	// CUDA holds one runtime directory per CUDA major version found, newest
	// major first. The file names carry the major (cudart64_13.dll,
	// cublas64_12.dll), so several can share PATH without shadowing each other.
	CUDA []string
	// CuDNN matches the first CUDA major. cudnn64_9.dll has the same name for
	// every CUDA build, so only one may be on PATH.
	CuDNN string
	// Dropped lists PATH entries removed because they are files, not
	// directories; such an entry breaks DLL search for everything after it.
	Dropped []string
}

// KernelPath is the full path of the kernel DLL, or "" when it was not found.
func (d Dirs) KernelPath() string {
	if d.Kernel == "" {
		return ""
	}
	return d.Kernel + `\` + KernelDLL
}

// String renders the discovery result for a startup log line.
func (d Dirs) String() string {
	or := func(s string) string {
		if s == "" {
			return "(not found)"
		}
		return s
	}
	cuda := "(not found)"
	if len(d.CUDA) > 0 {
		cuda = strings.Join(d.CUDA, ", ")
	}
	s := fmt.Sprintf("kernel=%s tensorrt=%s cuda=%s cudnn=%s",
		or(d.Kernel), or(d.TensorRT), cuda, or(d.CuDNN))
	if len(d.Dropped) > 0 {
		s += fmt.Sprintf(" dropped-path-files=%d", len(d.Dropped))
	}
	return s
}
