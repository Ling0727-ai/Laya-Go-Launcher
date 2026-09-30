//go:build windows

package ortbackend

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"syscall"
	"unsafe"

	"github.com/local/laya-go-launcher/internal/backend"
	"github.com/local/laya-go-launcher/internal/nativeenv"
)

// nativeTensor mirrors LayaOnnxTensor. Its layout must stay in step with the C
// struct in include/layatrt_onnx.h: a pointer, an int, a pointer, a size_t, an
// eight-element int64 array and an int.
//
// The struct is declared here rather than on the C side of a cgo boundary
// because this package must not link against anything: the bridge is opened
// with LoadLibrary at run time.
type nativeTensor struct {
	name      *byte
	dtype     int32
	data      unsafe.Pointer
	dataBytes uintptr
	shape     [8]int64
	nbDims    int32
}

// nativeSession owns one loaded ONNX Runtime session plus the bridge's entry
// points. Every field is a resolved function pointer, so a missing export is
// reported at load time rather than at the first inference.
type nativeSession struct {
	handle uintptr
	dll    *syscall.DLL
	api    nativeAPI

	// runMu serialises runs. One OrtSession is not safe for concurrent Run
	// calls, and ORT does its own intra-op threading.
	runMu sync.Mutex
}

type nativeAPI struct {
	createSession  *syscall.Proc
	destroySession *syscall.Proc
	getInputCount  *syscall.Proc
	getOutputCount *syscall.Proc
	getTensorInfo  *syscall.Proc
	run            *syscall.Proc
	getResultCount *syscall.Proc
	getResultInfo  *syscall.Proc
	copyResult     *syscall.Proc
	runtimeVersion *syscall.Proc
	// runtimeVersionAt was added after the first bridge build; it is optional so
	// an older DLL still loads and simply reports no version.
	runtimeVersionAt *syscall.Proc
	providerName     *syscall.Proc
	providerNote     *syscall.Proc
}

// nativeConfig is what Load passes down.
type nativeConfig struct {
	dllPath         string
	RuntimePath     string
	ModelPath       string
	Provider        string
	ProviderOptions map[string]string
}

// resolveBridge finds layatrt_onnx.dll: an explicit path, then beside the
// executable, then the working directory, then whatever the loader can find.
func resolveBridge(explicit string) string {
	if explicit != "" {
		return explicit
	}
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		for _, candidate := range []string{
			filepath.Join(dir, DefaultDLL),
			filepath.Join(dir, "build", "bin", "Release", DefaultDLL),
		} {
			if _, err := os.Stat(candidate); err == nil {
				return candidate
			}
		}
	}
	if _, err := os.Stat(DefaultDLL); err == nil {
		abs, absErr := filepath.Abs(DefaultDLL)
		if absErr == nil {
			return abs
		}
		return DefaultDLL
	}
	return DefaultDLL
}

func loadAPI(path string) (*syscall.DLL, nativeAPI, error) {
	dll, err := syscall.LoadDLL(path)
	if err != nil {
		return nil, nativeAPI{}, fmt.Errorf(
			"%w: cannot load %s: %v (build it with .\\build.ps1)",
			backend.ErrRuntimeMissing, path, err)
	}
	var api nativeAPI
	for _, item := range []struct {
		name string
		dst  **syscall.Proc
	}{
		{"LayaOnnx_CreateSession", &api.createSession},
		{"LayaOnnx_DestroySession", &api.destroySession},
		{"LayaOnnx_GetInputCount", &api.getInputCount},
		{"LayaOnnx_GetOutputCount", &api.getOutputCount},
		{"LayaOnnx_GetTensorInfo", &api.getTensorInfo},
		{"LayaOnnx_Run", &api.run},
		{"LayaOnnx_GetResultCount", &api.getResultCount},
		{"LayaOnnx_GetResultInfo", &api.getResultInfo},
		{"LayaOnnx_CopyResult", &api.copyResult},
		{"LayaOnnx_RuntimeVersion", &api.runtimeVersion},
		{"LayaOnnx_ProviderName", &api.providerName},
	} {
		proc, findErr := dll.FindProc(item.name)
		if findErr != nil {
			dll.Release()
			return nil, nativeAPI{}, fmt.Errorf(
				"%w: %s does not export %s", backend.ErrRuntimeMissing, path, item.name)
		}
		*item.dst = proc
	}
	// ProviderNote and RuntimeVersionAt were added after the first bridge build;
	// treat them as optional so an older DLL still loads and simply reports
	// less.
	api.providerNote, _ = dll.FindProc("LayaOnnx_ProviderNote")
	api.runtimeVersionAt, _ = dll.FindProc("LayaOnnx_RuntimeVersionAt")
	return dll, api, nil
}

// errorBuffer is the size of the message buffer passed to the bridge.
//
// The bridge truncates rather than failing, so this only bounds how much of a
// native error reaches the user; 1 KiB is far more than any ORT message needs
// and small enough to keep it on the stack.
const errorBuffer = 1024

// nativeError renders the bridge's message buffer as a Go error.
//
// The message lives in a Go slice that outlives the call, so nothing has to be
// freed and no foreign pointer is ever converted back into a Go pointer.
func nativeError(prefix string, buf []byte) error {
	msg := trimCString(buf)
	if msg == "" {
		return fmt.Errorf("onnx: %s", prefix)
	}
	return fmt.Errorf("onnx: %s: %s", prefix, msg)
}

// trimCString returns the NUL-terminated prefix of buf as a string.
func trimCString(buf []byte) string {
	for i, b := range buf {
		if b == 0 {
			return string(buf[:i])
		}
	}
	return string(buf)
}

// stringCall reads one of the introspection calls into a Go buffer.
func (s *nativeSession) stringCall(proc *syscall.Proc) string {
	if proc == nil || s.handle == 0 {
		return ""
	}
	var buf [512]byte
	code, _, _ := proc.Call(
		s.handle,
		uintptr(unsafe.Pointer(&buf[0])),
		uintptr(len(buf)),
	)
	if code != 0 {
		return ""
	}
	return trimCString(buf[:])
}

func newNativeSession(cfg nativeConfig) (*nativeSession, error) {
	// The CUDA provider pulls in cudart/cuBLAS/cuDNN by bare name; make sure the
	// discovered directories are on PATH before ORT goes looking for them.
	nativeenv.Prepare()
	path := resolveBridge(cfg.dllPath)
	dll, api, err := loadAPI(path)
	if err != nil {
		return nil, err
	}

	// Resolve the runtime library before the native call, and pass the concrete
	// path down. Doing it here rather than leaving it to the bridge's own
	// discovery is what makes the multi-path search above effective: the bridge
	// only knows about the executable's directory and PATH, while a developer
	// machine usually has the right runtime somewhere else entirely.
	resolvedRuntime, _, err := resolveRuntime(cfg.RuntimePath, cfg.Provider)
	if err != nil {
		dll.Release()
		return nil, err
	}
	cfg.RuntimePath = resolvedRuntime

	// Every string the call needs must stay alive until it returns, because the
	// bridge reads them synchronously and Go's collector does not know that.
	keep := make([]*byte, 0, 8+2*len(cfg.ProviderOptions))
	intern := func(s string) (*byte, error) {
		p, err := syscall.BytePtrFromString(s)
		if err != nil {
			return nil, err
		}
		keep = append(keep, p)
		return p, nil
	}

	runtimePath, err := intern(cfg.RuntimePath)
	if err != nil {
		dll.Release()
		return nil, fmt.Errorf("onnx: runtime path contains a NUL byte")
	}
	modelPath, err := intern(cfg.ModelPath)
	if err != nil {
		dll.Release()
		return nil, fmt.Errorf("onnx: model path contains a NUL byte")
	}
	provider, err := intern(cfg.Provider)
	if err != nil {
		dll.Release()
		return nil, fmt.Errorf("onnx: provider contains a NUL byte")
	}

	keys := make([]*byte, 0, len(cfg.ProviderOptions))
	values := make([]*byte, 0, len(cfg.ProviderOptions))
	for k, v := range cfg.ProviderOptions {
		kp, kErr := intern(k)
		vp, vErr := intern(v)
		if kErr != nil || vErr != nil {
			dll.Release()
			return nil, fmt.Errorf("onnx: provider option contains a NUL byte")
		}
		keys = append(keys, kp)
		values = append(values, vp)
	}

	var handle uintptr
	errBuf := make([]byte, errorBuffer)
	code, _, _ := api.createSession.Call(
		uintptr(unsafe.Pointer(runtimePath)),
		uintptr(unsafe.Pointer(modelPath)),
		uintptr(unsafe.Pointer(provider)),
		uintptr(unsafe.Pointer(unsafe.SliceData(keys))),
		uintptr(unsafe.Pointer(unsafe.SliceData(values))),
		uintptr(len(keys)),
		uintptr(unsafe.Pointer(&handle)),
		uintptr(unsafe.Pointer(&errBuf[0])),
		uintptr(len(errBuf)),
	)
	runtime.KeepAlive(keep)
	if code != 0 {
		err := nativeError("create session", errBuf)
		dll.Release()
		return nil, err
	}
	return &nativeSession{handle: handle, dll: dll, api: api}, nil
}

func destroyNativeSession(s *nativeSession) {
	if s == nil || s.handle == 0 {
		return
	}
	s.api.destroySession.Call(s.handle)
	s.handle = 0
	if s.dll != nil {
		s.dll.Release()
	}
}

// tensorInfo reads one side of the model's IO declaration.
func (s *nativeSession) tensorInfo(isInput bool, count int) ([]backend.TensorInfo, error) {
	out := make([]backend.TensorInfo, 0, count)
	flag := uintptr(0)
	if isInput {
		flag = 1
	}
	for i := 0; i < count; i++ {
		var (
			nameBuf [256]byte
			dtype   int32
			shape   [8]int64
			nbDims  int32
		)
		errBuf := make([]byte, errorBuffer)
		code, _, _ := s.api.getTensorInfo.Call(
			s.handle,
			flag,
			uintptr(i),
			uintptr(unsafe.Pointer(&nameBuf[0])),
			uintptr(len(nameBuf)),
			uintptr(unsafe.Pointer(&dtype)),
			uintptr(unsafe.Pointer(&shape[0])),
			uintptr(unsafe.Pointer(&nbDims)),
			uintptr(len(shape)),
			uintptr(unsafe.Pointer(&errBuf[0])),
			uintptr(len(errBuf)),
		)
		if code != 0 {
			return nil, nativeError("read tensor info", errBuf)
		}
		info := backend.TensorInfo{
			Name:    trimCString(nameBuf[:]),
			DType:   backend.DType(dtype),
			IsInput: isInput,
		}
		for d := 0; d < int(nbDims) && d < len(shape); d++ {
			info.Shape = append(info.Shape, int(shape[d]))
		}
		out = append(out, info)
	}
	return out, nil
}

func (s *nativeSession) inputs() ([]backend.TensorInfo, error) {
	n, err := s.count(true)
	if err != nil {
		return nil, err
	}
	return s.tensorInfo(true, n)
}

func (s *nativeSession) outputs() ([]backend.TensorInfo, error) {
	n, err := s.count(false)
	if err != nil {
		return nil, err
	}
	return s.tensorInfo(false, n)
}

func (s *nativeSession) count(isInput bool) (int, error) {
	var n uintptr
	proc := s.api.getInputCount
	if !isInput {
		proc = s.api.getOutputCount
	}
	errBuf := make([]byte, errorBuffer)
	code, _, _ := proc.Call(
		s.handle,
		uintptr(unsafe.Pointer(&n)),
		uintptr(unsafe.Pointer(&errBuf[0])),
		uintptr(len(errBuf)),
	)
	if code != 0 {
		return 0, nativeError("read tensor count", errBuf)
	}
	return int(n), nil
}

// toNative renders a neutral tensor as the C descriptor the bridge expects.
//
// The descriptor's pointers alias the caller's slices, so both the descriptor
// slice and the underlying buffers must stay alive until the native call
// returns; run does that with runtime.KeepAlive.
func toNative(name string, dtype backend.DType, shape []int, data any, dataBytes int) (nativeTensor, error) {
	cName, err := syscall.BytePtrFromString(name)
	if err != nil {
		return nativeTensor{}, fmt.Errorf("onnx: tensor name contains a NUL byte")
	}
	if len(shape) == 0 || len(shape) > 8 {
		return nativeTensor{}, fmt.Errorf("onnx: tensor %q needs 1..8 dimensions", name)
	}
	ptr, elems, err := slicePointer(data, dtype)
	if err != nil {
		return nativeTensor{}, fmt.Errorf("onnx: tensor %q: %w", name, err)
	}
	want := 1
	for _, d := range shape {
		if d <= 0 {
			return nativeTensor{}, fmt.Errorf("onnx: tensor %q has a non-positive dimension in %v", name, shape)
		}
		want *= d
	}
	if elems < want {
		return nativeTensor{}, fmt.Errorf("onnx: tensor %q needs %d elements for %v, got %d",
			name, want, shape, elems)
	}

	t := nativeTensor{name: cName, dtype: int32(dtype), data: ptr, dataBytes: uintptr(dataBytes)}
	for i, d := range shape {
		t.shape[i] = int64(d)
	}
	t.nbDims = int32(len(shape))
	return t, nil
}

// slicePointer returns the base pointer of a typed slice, checking that its
// element type matches the declared dtype.
func slicePointer(data any, dtype backend.DType) (unsafe.Pointer, int, error) {
	switch v := data.(type) {
	case []float32:
		if dtype != backend.Float32 {
			return nil, 0, dtypeMismatch(dtype, data)
		}
		return unsafe.Pointer(unsafe.SliceData(v)), len(v), nil
	case []float64:
		if dtype != backend.Float64 {
			return nil, 0, dtypeMismatch(dtype, data)
		}
		return unsafe.Pointer(unsafe.SliceData(v)), len(v), nil
	case []int64:
		if dtype != backend.Int64 {
			return nil, 0, dtypeMismatch(dtype, data)
		}
		return unsafe.Pointer(unsafe.SliceData(v)), len(v), nil
	case []int32:
		if dtype != backend.Int32 {
			return nil, 0, dtypeMismatch(dtype, data)
		}
		return unsafe.Pointer(unsafe.SliceData(v)), len(v), nil
	case []int16:
		if dtype != backend.Int16 {
			return nil, 0, dtypeMismatch(dtype, data)
		}
		return unsafe.Pointer(unsafe.SliceData(v)), len(v), nil
	case []uint16:
		if dtype != backend.Uint16 {
			return nil, 0, dtypeMismatch(dtype, data)
		}
		return unsafe.Pointer(unsafe.SliceData(v)), len(v), nil
	case []int8:
		if dtype != backend.Int8 {
			return nil, 0, dtypeMismatch(dtype, data)
		}
		return unsafe.Pointer(unsafe.SliceData(v)), len(v), nil
	case []uint8:
		// ONNX bool is one byte per element, so a byte slice is the natural
		// carrier for the tokenizer's mask pipeline.
		if dtype != backend.Uint8 && dtype != backend.Bool {
			return nil, 0, dtypeMismatch(dtype, data)
		}
		return unsafe.Pointer(unsafe.SliceData(v)), len(v), nil
	case []bool:
		if dtype != backend.Bool {
			return nil, 0, dtypeMismatch(dtype, data)
		}
		return unsafe.Pointer(unsafe.SliceData(v)), len(v), nil
	default:
		return nil, 0, fmt.Errorf("unsupported slice type %T", data)
	}
}

func dtypeMismatch(dtype backend.DType, data any) error {
	return fmt.Errorf("dtype %s does not match %T", dtype, data)
}

// resultInfo is declared in ortbackend.go so the non-Windows stub can name it.

// run binds the caller's input buffers and computes the requested outputs.
//
// ONNX Runtime keeps the results; ctx is honoured before the native call starts,
// because an in-flight call cannot be interrupted through this ABI and tearing
// down a session mid-run would be worse than a late result.
func (s *nativeSession) run(ctx context.Context, inputs []backend.Tensor, outputs []string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.runMu.Lock()
	defer s.runMu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}

	inDescs := make([]nativeTensor, 0, len(inputs))
	for _, t := range inputs {
		desc, err := toNative(t.Name, t.DType, t.Shape, t.Data, t.ByteSize())
		if err != nil {
			return err
		}
		inDescs = append(inDescs, desc)
	}

	// The output names are only read during the call, but the pointers must stay
	// alive across it, so they are kept in a slice that outlives the call.
	namePtrs := make([]*byte, 0, len(outputs))
	for _, name := range outputs {
		p, err := syscall.BytePtrFromString(name)
		if err != nil {
			return fmt.Errorf("onnx: output name contains a NUL byte")
		}
		namePtrs = append(namePtrs, p)
	}

	errBuf := make([]byte, errorBuffer)
	code, _, _ := s.api.run.Call(
		s.handle,
		uintptr(unsafe.Pointer(unsafe.SliceData(inDescs))),
		uintptr(len(inDescs)),
		uintptr(unsafe.Pointer(unsafe.SliceData(namePtrs))),
		uintptr(len(namePtrs)),
		uintptr(unsafe.Pointer(&errBuf[0])),
		uintptr(len(errBuf)),
	)
	// The descriptors alias the caller's slices and the names are Go-owned byte
	// buffers, so all of them must outlive the synchronous foreign call.
	runtime.KeepAlive(inDescs)
	runtime.KeepAlive(namePtrs)
	runtime.KeepAlive(inputs)
	if code != 0 {
		return nativeError("run", errBuf)
	}
	return nil
}

// results describes every output of the most recent run.
//
// A dimension is never negative here: the graph has run, so what the model
// declared symbolically is now a concrete number.
func (s *nativeSession) results() ([]resultInfo, error) {
	var count uintptr
	errBuf := make([]byte, errorBuffer)
	code, _, _ := s.api.getResultCount.Call(
		s.handle,
		uintptr(unsafe.Pointer(&count)),
		uintptr(unsafe.Pointer(&errBuf[0])),
		uintptr(len(errBuf)),
	)
	if code != 0 {
		return nil, nativeError("read result count", errBuf)
	}

	out := make([]resultInfo, 0, int(count))
	for i := 0; i < int(count); i++ {
		var (
			nameBuf [256]byte
			dtype   int32
			shape   [8]int64
			nbDims  int32
		)
		errBuf := make([]byte, errorBuffer)
		code, _, _ := s.api.getResultInfo.Call(
			s.handle,
			uintptr(i),
			uintptr(unsafe.Pointer(&nameBuf[0])),
			uintptr(len(nameBuf)),
			uintptr(unsafe.Pointer(&dtype)),
			uintptr(unsafe.Pointer(&shape[0])),
			uintptr(unsafe.Pointer(&nbDims)),
			uintptr(len(shape)),
			uintptr(unsafe.Pointer(&errBuf[0])),
			uintptr(len(errBuf)),
		)
		if code != 0 {
			return nil, nativeError("read result info", errBuf)
		}
		info := resultInfo{
			index: i,
			name:  trimCString(nameBuf[:]),
			dtype: backend.DType(dtype),
		}
		elements := 1
		for d := 0; d < int(nbDims) && d < len(shape); d++ {
			info.shape = append(info.shape, int(shape[d]))
			elements *= int(shape[d])
		}
		info.elements = elements
		out = append(out, info)
	}
	return out, nil
}

// copyResult copies one result into dst.
func (s *nativeSession) copyResult(index int, dst []byte) error {
	if len(dst) == 0 {
		return fmt.Errorf("onnx: result %d has no bytes to copy", index)
	}
	var (
		written uintptr
		errText = make([]byte, errorBuffer)
	)
	code, _, _ := s.api.copyResult.Call(
		s.handle,
		uintptr(index),
		uintptr(unsafe.Pointer(&dst[0])),
		uintptr(len(dst)),
		uintptr(unsafe.Pointer(&written)),
		uintptr(unsafe.Pointer(&errText[0])),
		uintptr(len(errText)),
	)
	runtime.KeepAlive(dst)
	if code != 0 {
		return nativeError("copy result", errText)
	}
	return nil
}

func (s *nativeSession) runtimeVersion() string { return s.stringCall(s.api.runtimeVersion) }
func (s *nativeSession) providerName() string   { return s.stringCall(s.api.providerName) }
func (s *nativeSession) providerNote() string   { return s.stringCall(s.api.providerNote) }

// Probe reports whether the ONNX path could run here, and which runtime version
// is installed, without loading a model.
//
// It checks the three things that fail first and most confusingly: the bridge
// library, the ONNX Runtime library, and whether that runtime actually matches
// the requested provider. Resolving them here by the same rules a load uses means
// the answer is "the runtime is missing" rather than "session creation failed",
// and a stale or mismatched DLL is visible in diagnostics instead of only
// showing up as an inference failure.
func Probe(dllPath, runtimePath string) (string, error) {
	return ProbeFor(dllPath, runtimePath, "")
}

// ProbeFor is Probe with the execution provider taken into account, so the
// reported runtime is the one a load with that provider would actually use.
func ProbeFor(dllPath, runtimePath, provider string) (string, error) {
	nativeenv.Prepare()
	path := resolveBridge(dllPath)
	dll, api, err := loadAPI(path)
	if err != nil {
		return "", err
	}
	defer dll.Release()

	resolved, kind, err := resolveRuntime(runtimePath, provider)
	if err != nil {
		return "", err
	}

	// The bridge reads the version, so the DLL handle and the OrtApiBase lookup
	// stay on the C++ side of the boundary.
	if api.runtimeVersionAt == nil {
		return "", nil
	}
	var (
		errBuf   = make([]byte, errorBuffer)
		version  [64]byte
		cRuntime *byte
	)
	cRuntime, err = syscall.BytePtrFromString(resolved)
	if err != nil {
		return "", fmt.Errorf("onnx: runtime path contains a NUL byte")
	}
	code, _, _ := api.runtimeVersionAt.Call(
		uintptr(unsafe.Pointer(cRuntime)),
		uintptr(unsafe.Pointer(&version[0])),
		uintptr(len(version)),
		uintptr(unsafe.Pointer(&errBuf[0])),
		uintptr(len(errBuf)),
	)
	runtime.KeepAlive(cRuntime)
	if code != 0 {
		return "", nativeError("read runtime version", errBuf)
	}
	got := trimCString(version[:])
	if kind != RuntimeAny && kind != RuntimeCPU {
		// Name the distribution, so a CUDA request that resolved a CUDA package
		// is distinguishable from one that silently landed on a CPU build.
		got = fmt.Sprintf("%s [%s]", got, kind)
	}
	return got, nil
}
