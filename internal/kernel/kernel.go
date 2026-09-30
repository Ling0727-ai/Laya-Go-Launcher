// Package kernel binds the native TensorRT runtime.
//
// Every call here goes through a function the DLL exports. No C struct is
// declared or laid out on the Go side: descriptors, staging buffers, and output
// buffers are owned by the native run builder, and Go only passes names, dtypes
// and shapes as plain arguments. That keeps the boundary to the documented C
// ABI and removes any question of struct layout, allocation, or lifetime
// crossing between the two runtimes.
package kernel

/*
#cgo windows CFLAGS: -I${SRCDIR}/../../include -I${SRCDIR}
// Windows does not link the kernel: kernel_loader_windows.c opens it with
// LoadLibrary after internal/nativeenv has located TensorRT and CUDA, so a
// plain shell PATH no longer kills the process before main().

#cgo linux CXXFLAGS: -std=c++17 -O3 -I${SRCDIR}/../../include -I/usr/local/cuda/include
#cgo linux CFLAGS: -I${SRCDIR}/../../include
#cgo linux LDFLAGS: -L${SRCDIR}/../../build/lib -L/usr/local/cuda/lib64
#cgo linux LDFLAGS: -lqualityscaler_tensorrt -l:libnvinfer.so.10 -lcudart -lpthread

#include <stdlib.h>
#include "ai_tensorrt_cpp.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// DType mirrors TensorIOType. The values are part of the native ABI.
type DType int

const (
	Float32  DType = C.TENSOR_IO_FLOAT32
	Float16  DType = C.TENSOR_IO_FLOAT16
	BFloat16 DType = C.TENSOR_IO_BFLOAT16
	Int32    DType = C.TENSOR_IO_INT32
	Int64    DType = C.TENSOR_IO_INT64
	Bool     DType = C.TENSOR_IO_BOOL
	Int8     DType = C.TENSOR_IO_INT8
	Uint8    DType = C.TENSOR_IO_UINT8
)

// String names the element type as it appears in API payloads.
func (d DType) String() string {
	switch d {
	case Float32:
		return "float32"
	case Float16:
		return "float16"
	case BFloat16:
		return "bfloat16"
	case Int32:
		return "int32"
	case Int64:
		return "int64"
	case Bool:
		return "bool"
	case Int8:
		return "int8"
	case Uint8:
		return "uint8"
	default:
		return fmt.Sprintf("dtype(%d)", int(d))
	}
}

// Size is the byte width of one element, as the kernel reports it.
func (d DType) Size() int { return int(C.TensorRT_DtypeSize(C.int(d))) }

var (
	initOnce sync.Once
	initErr  error
)

// Initialize brings up CUDA and TensorRT. Safe to call from any goroutine and
// any number of times.
func Initialize() error {
	initOnce.Do(func() {
		if err := ensureLoaded(); err != nil {
			initErr = err
			return
		}
		if C.TensorRT_Initialize() != 0 {
			initErr = fmt.Errorf("tensorrt: initialize failed: %s", LastError())
		}
	})
	return initErr
}

// LastError returns the runtime's most recent error string.
func LastError() string { return C.GoString(C.TensorRT_GetLastError()) }

// Version is the TensorRT version the runtime is linked against.
func Version() string {
	if ensureLoaded() != nil {
		return ""
	}
	return C.GoString(C.TensorRT_Version())
}

// SetDiagnostics turns the native diagnostic log on or off.
func SetDiagnostics(enabled bool) {
	if ensureLoaded() != nil {
		return
	}
	v := C.int(0)
	if enabled {
		v = 1
	}
	C.TensorRT_SetDiagnostics(v)
}

// TensorInfo describes one engine IO tensor.
type TensorInfo struct {
	Name    string
	DType   DType
	Shape   []int
	IsInput bool
}

// Engine is a loaded plan. Handles are opaque and owned by the kernel.
type Engine struct {
	handle C.TensorRTEngineHandle
	path   string
	mu     sync.Mutex
	closed bool
}

// LoadEngine deserialises a plan from disk.
func LoadEngine(path string) (*Engine, error) {
	if err := Initialize(); err != nil {
		return nil, err
	}
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	handle := C.TensorRT_LoadEngine(cPath)
	if handle == nil {
		return nil, fmt.Errorf("tensorrt: load %s: %s", path, LastError())
	}
	return &Engine{handle: handle, path: path}, nil
}

// Path is the file this engine was loaded from.
func (e *Engine) Path() string { return e.path }

// Close unloads the plan. Safe to call more than once.
func (e *Engine) Close() {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return
	}
	e.closed = true
	C.TensorRT_UnloadEngine(e.handle)
	e.handle = nil
}

// Tensors enumerates every IO tensor the engine exposes.
func (e *Engine) Tensors() ([]TensorInfo, error) {
	if e == nil || e.handle == nil {
		return nil, fmt.Errorf("tensorrt: engine is not loaded")
	}
	n := int(C.TensorRT_GetNumIOTensors(e.handle))
	if n < 0 {
		return nil, fmt.Errorf("tensorrt: cannot enumerate IO tensors: %s", LastError())
	}
	out := make([]TensorInfo, 0, n)
	for i := 0; i < n; i++ {
		info, err := e.tensorAt(i)
		if err != nil {
			return nil, err
		}
		out = append(out, info)
	}
	return out, nil
}

func (e *Engine) tensorAt(i int) (TensorInfo, error) {
	var (
		nameBuf [256]C.char
		isInput C.int
		dtype   C.int
		shape   [8]C.int64_t
		nbDims  C.int
	)
	if C.TensorRT_GetIOTensorInfoEx(e.handle, C.int(i), &nameBuf[0], &isInput,
		&dtype, &shape[0], &nbDims) != 0 {
		return TensorInfo{}, fmt.Errorf("tensorrt: read IO tensor %d: %s", i, LastError())
	}
	info := TensorInfo{
		Name:    C.GoString(&nameBuf[0]),
		DType:   DType(dtype),
		IsInput: isInput != 0,
	}
	for d := 0; d < int(nbDims); d++ {
		info.Shape = append(info.Shape, int(shape[d]))
	}
	return info, nil
}

// DeviceMemoryBytes is the engine's worst-case activation memory.
func (e *Engine) DeviceMemoryBytes() uint64 {
	if e == nil || e.handle == nil {
		return 0
	}
	return uint64(C.TensorRT_GetDeviceMemorySize(e.handle))
}

// ProfileShape returns an input's optimisation-profile bounds for a dynamic
// tensor: which is 0 = min, 1 = opt, 2 = max.
//
// Returns nil when the tensor has no profile (its dimension is static), so a
// caller can distinguish "dynamic with these bounds" from "fixed".
func (e *Engine) ProfileShape(name string, which int) ([]int, error) {
	if e == nil || e.handle == nil {
		return nil, fmt.Errorf("tensorrt: engine is not loaded")
	}
	if which < 0 || which > 2 {
		return nil, fmt.Errorf("tensorrt: which must be 0 (min), 1 (opt) or 2 (max)")
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	var shape [8]C.int64_t
	var nbDims C.int
	rc := C.TensorRT_GetProfileShape(e.handle, cName, C.int(which), &shape[0], &nbDims)
	if rc < 0 {
		return nil, fmt.Errorf("tensorrt: read profile for %q: %s", name, LastError())
	}
	if rc == 0 {
		return nil, nil
	}
	out := make([]int, 0, int(nbDims))
	for i := 0; i < int(nbDims); i++ {
		out = append(out, int(shape[i]))
	}
	return out, nil
}

// ProfileBounds returns the (min, max) bounds for a dynamic input's dimension.
// ok is false when the tensor is static.
func (e *Engine) ProfileBounds(name string, dim int) (min, max int, ok bool) {
	lo, err := e.ProfileShape(name, 0)
	if err != nil || lo == nil || dim >= len(lo) {
		return 0, 0, false
	}
	hi, err := e.ProfileShape(name, 2)
	if err != nil || hi == nil || dim >= len(hi) {
		return 0, 0, false
	}
	return lo[dim], hi[dim], true
}

// Context is one native execution context plus its CUDA stream.
type Context struct {
	handle C.TensorRTContextHandle
	engine *Engine
	mu     sync.Mutex
	closed bool
}

// NewContext creates a dedicated execution context.
func (e *Engine) NewContext() (*Context, error) {
	if e == nil || e.handle == nil {
		return nil, fmt.Errorf("tensorrt: engine is not loaded")
	}
	h := C.TensorRT_CreateContext(e.handle)
	if h == nil {
		return nil, fmt.Errorf("tensorrt: create context: %s", LastError())
	}
	return &Context{handle: h, engine: e}, nil
}

// Close destroys the context. Safe to call more than once.
func (c *Context) Close() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return
	}
	c.closed = true
	C.TensorRT_DestroyContext(c.handle)
	c.handle = nil
}

// Run is a native-owned inference description. Add inputs, declare outputs,
// resolve, execute, read back. All buffers live on the native side.
type Run struct {
	handle C.TensorRunHandle
	mu     sync.Mutex
	closed bool

	outputs []string
}

// NewRun creates a run bound to a context.
func (c *Context) NewRun() (*Run, error) {
	if c == nil || c.handle == nil {
		return nil, fmt.Errorf("tensorrt: context is closed")
	}
	h := C.TensorRT_RunCreate(c.handle)
	if h == nil {
		return nil, fmt.Errorf("tensorrt: create run: %s", LastError())
	}
	r := &Run{handle: h}
	runtime.SetFinalizer(r, func(r *Run) { r.Close() })
	return r, nil
}

// Close frees the run and everything it owns.
func (r *Run) Close() {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return
	}
	r.closed = true
	runtime.SetFinalizer(r, nil)
	C.TensorRT_RunDestroy(r.handle)
	r.handle = nil
}

// AddInput stages a copy of data under name. The slice is not retained: the
// kernel copies it immediately, so the caller may reuse it on return.
//
// The slice's element type must match dtype. Passing a []byte for anything but
// an 8-bit dtype is a programming error and is rejected.
func (r *Run) AddInput(name string, dtype DType, data any, shape []int) error {
	if r == nil || r.handle == nil {
		return fmt.Errorf("tensorrt: run is closed")
	}
	if name == "" {
		return fmt.Errorf("tensorrt: input name is required")
	}
	if len(shape) == 0 || len(shape) > 8 {
		return fmt.Errorf("tensorrt: input %q needs 1..8 dimensions", name)
	}

	ptr, elems, err := slicePointer(data, dtype)
	if err != nil {
		return fmt.Errorf("tensorrt: input %q: %w", name, err)
	}
	want := 1
	for _, d := range shape {
		if d <= 0 {
			return fmt.Errorf("tensorrt: input %q has a non-positive dimension", name)
		}
		want *= d
	}
	if elems < want {
		return fmt.Errorf("tensorrt: input %q needs %d elements, got %d", name, want, elems)
	}

	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	cShape := make([]C.int64_t, len(shape))
	for i, d := range shape {
		cShape[i] = C.int64_t(d)
	}
	if C.TensorRT_RunAddInput(r.handle, cName, C.int(dtype), ptr,
		&cShape[0], C.int(len(shape))) != 0 {
		return fmt.Errorf("tensorrt: add input %q: %s", name, LastError())
	}
	runtime.KeepAlive(data)
	return nil
}

// AddOutput declares an output by name. Its shape and dtype are read from the
// engine during Resolve.
func (r *Run) AddOutput(name string) error {
	if r == nil || r.handle == nil {
		return fmt.Errorf("tensorrt: run is closed")
	}
	if name == "" {
		return fmt.Errorf("tensorrt: output name is required")
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	if C.TensorRT_RunAddOutput(r.handle, cName) != 0 {
		return fmt.Errorf("tensorrt: add output %q: %s", name, LastError())
	}
	r.outputs = append(r.outputs, name)
	return nil
}

// Resolve computes the concrete shape and dtype of every declared output.
func (r *Run) Resolve() ([]TensorInfo, error) {
	if r == nil || r.handle == nil {
		return nil, fmt.Errorf("tensorrt: run is closed")
	}
	n := int(C.TensorRT_RunOutputCount(r.handle))
	if n <= 0 {
		return nil, fmt.Errorf("tensorrt: no outputs declared")
	}

	const nameStride = 256
	nameBuf := make([]C.char, n*nameStride)
	dtypes := make([]C.int, n)
	shapes := make([]C.int64_t, n*8)
	nbDims := make([]C.int, n)

	if C.TensorRT_RunResolveOutputs(r.handle, &nameBuf[0], C.int(nameStride),
		&dtypes[0], &shapes[0], &nbDims[0], C.int(n)) != 0 {
		return nil, fmt.Errorf("tensorrt: resolve outputs: %s", LastError())
	}

	out := make([]TensorInfo, n)
	for i := 0; i < n; i++ {
		out[i] = TensorInfo{
			Name:  C.GoString(&nameBuf[i*nameStride]),
			DType: DType(dtypes[i]),
		}
		for d := 0; d < int(nbDims[i]); d++ {
			out[i].Shape = append(out[i].Shape, int(shapes[i*8+d]))
		}
	}
	return out, nil
}

// Execute runs the inference and blocks until outputs are in native memory.
func (r *Run) Execute() error {
	if r == nil || r.handle == nil {
		return fmt.Errorf("tensorrt: run is closed")
	}
	if C.TensorRT_RunExecute(r.handle) != 0 {
		return fmt.Errorf("tensorrt: execute: %s", LastError())
	}
	return nil
}

// ReadOutput copies output i into a freshly allocated []float32.
//
// The decision head's outputs are float32; other models may differ, in which
// case use OutputBytes and the raw reader.
func (r *Run) ReadOutput(i int) ([]float32, error) {
	n := r.OutputBytes(i)
	if n == 0 {
		return nil, fmt.Errorf("tensorrt: output %d is unknown or empty", i)
	}
	if n%4 != 0 {
		return nil, fmt.Errorf("tensorrt: output %d is not a whole number of float32", i)
	}
	buf := make([]float32, n/4)
	if err := r.readInto(i, unsafe.Pointer(&buf[0]), n); err != nil {
		return nil, err
	}
	return buf, nil
}

// ReadOutputBytes copies output i into a freshly allocated byte slice, for
// outputs whose element type is not float32.
func (r *Run) ReadOutputBytes(i int) ([]byte, error) {
	n := r.OutputBytes(i)
	if n == 0 {
		return nil, fmt.Errorf("tensorrt: output %d is unknown or empty", i)
	}
	buf := make([]byte, n)
	if err := r.readInto(i, unsafe.Pointer(&buf[0]), n); err != nil {
		return nil, err
	}
	return buf, nil
}

func (r *Run) readInto(i int, dst unsafe.Pointer, bytes int) error {
	if r == nil || r.handle == nil {
		return fmt.Errorf("tensorrt: run is closed")
	}
	if C.TensorRT_RunReadOutput(r.handle, C.int(i), dst, C.size_t(bytes)) != 0 {
		return fmt.Errorf("tensorrt: read output %d: %s", i, LastError())
	}
	return nil
}

// OutputBytes is the byte size of a resolved output, or 0 when unknown.
func (r *Run) OutputBytes(i int) int {
	if r == nil || r.handle == nil {
		return 0
	}
	return int(C.TensorRT_RunOutputBytes(r.handle, C.int(i)))
}

// OutputCount is the number of outputs declared on this run.
func (r *Run) OutputCount() int {
	if r == nil || r.handle == nil {
		return 0
	}
	return int(C.TensorRT_RunOutputCount(r.handle))
}

// OutputNames returns the declared output names, in declaration order.
func (r *Run) OutputNames() []string {
	out := make([]string, len(r.outputs))
	copy(out, r.outputs)
	return out
}

// VRAMInfo reports device memory in MiB.
type VRAMInfo struct {
	FreeMB  int
	TotalMB int
}

// QueryVRAMInfo reads current device memory.
func QueryVRAMInfo() VRAMInfo {
	if ensureLoaded() != nil {
		return VRAMInfo{}
	}
	var freeMB, totalMB C.int
	if C.TensorRT_GetVRAMInfo(&freeMB, &totalMB) != 0 {
		return VRAMInfo{}
	}
	return VRAMInfo{FreeMB: int(freeMB), TotalMB: int(totalMB)}
}

// DeviceInfo reports the active device.
type DeviceInfo struct {
	Name         string
	ComputeMajor int
	ComputeMinor int
}

// QueryDeviceInfo reads the active device's name and compute capability.
func QueryDeviceInfo() DeviceInfo {
	if ensureLoaded() != nil {
		return DeviceInfo{}
	}
	var nameBuf [256]C.char
	var major, minor C.int
	if C.TensorRT_GetDeviceInfo(&nameBuf[0], &major, &minor) != 0 {
		return DeviceInfo{}
	}
	return DeviceInfo{
		Name:         C.GoString(&nameBuf[0]),
		ComputeMajor: int(major),
		ComputeMinor: int(minor),
	}
}

// slicePointer returns the base pointer and element count of a typed slice.
func slicePointer(data any, dtype DType) (unsafe.Pointer, int, error) {
	switch v := data.(type) {
	case []float32:
		if dtype != Float32 {
			return nil, 0, fmt.Errorf("dtype %s does not match []float32", dtype)
		}
		if len(v) == 0 {
			return nil, 0, fmt.Errorf("empty slice")
		}
		return unsafe.Pointer(&v[0]), len(v), nil
	case []float64:
		if dtype != Float32 {
			return nil, 0, fmt.Errorf("dtype %s does not match []float64", dtype)
		}
		if len(v) == 0 {
			return nil, 0, fmt.Errorf("empty slice")
		}
		return unsafe.Pointer(&v[0]), len(v), nil
	case []int64:
		if dtype != Int64 {
			return nil, 0, fmt.Errorf("dtype %s does not match []int64", dtype)
		}
		if len(v) == 0 {
			return nil, 0, fmt.Errorf("empty slice")
		}
		return unsafe.Pointer(&v[0]), len(v), nil
	case []int32:
		if dtype != Int32 {
			return nil, 0, fmt.Errorf("dtype %s does not match []int32", dtype)
		}
		if len(v) == 0 {
			return nil, 0, fmt.Errorf("empty slice")
		}
		return unsafe.Pointer(&v[0]), len(v), nil
	case []bool:
		if dtype != Bool {
			return nil, 0, fmt.Errorf("dtype %s does not match []bool", dtype)
		}
		if len(v) == 0 {
			return nil, 0, fmt.Errorf("empty slice")
		}
		return unsafe.Pointer(&v[0]), len(v), nil
	case []byte:
		if dtype != Bool && dtype != Int8 && dtype != Uint8 {
			return nil, 0, fmt.Errorf("dtype %s does not match []byte", dtype)
		}
		if len(v) == 0 {
			return nil, 0, fmt.Errorf("empty slice")
		}
		return unsafe.Pointer(&v[0]), len(v), nil
	default:
		return nil, 0, fmt.Errorf("unsupported slice type %T", data)
	}
}
