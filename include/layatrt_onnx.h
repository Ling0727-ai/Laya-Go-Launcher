#ifndef LAYA_ONNX_NATIVE_H
#define LAYA_ONNX_NATIVE_H

// Native ONNX Runtime bridge for laya-trt.
//
// This is the laya-trt port of QualityScaler-go's pkg/func/ortcuda: the same
// approach — dlopen onnxruntime.dll, fetch OrtApi through OrtGetApiBase, drive
// an I/O binding — but with the tensor contract generalised. QS's kernel was
// written for one image model (a single float32 [1,3,H,W] input and one output);
// laya's decision model takes five named inputs with mixed dtypes
// (int64/int64/int64/bool/int64) and returns two float32 outputs.
//
// So the ABI here is *generic over tensors* rather than over image geometry:
// a run is described by a flat list of tensor descriptors, each with a name,
// dtype, rank, shape and a host pointer. The caller owns every buffer, exactly
// as in the TensorRT kernel's run builder.
//
// Two consequences of that ownership rule are deliberate:
//
//   - No allocation crosses the boundary. Every string the bridge produces is
//     copied into a caller-supplied buffer, and every error is a return code
//     plus a message in that buffer. Nothing here returns a pointer the caller
//     would have to free or convert.
//   - The bridge does NOT link against onnxruntime.lib. It resolves every
//     symbol at runtime, so a machine without ONNX Runtime can still start the
//     GUI, the server and the doctor command and report the missing runtime
//     instead of dying before main() with 0xC0000279.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#  define LAYA_ONNX_API __declspec(dllexport)
#else
#  define LAYA_ONNX_API
#endif

// Element types. These mirror ONNX_TENSOR_ELEMENT_DATA_TYPE_* so the values can
// be passed straight through to the OrtApi, and are the same numbering the Go
// side uses in internal/backend.DType.
typedef enum LayaOnnxDType {
    LAYA_ONNX_FLOAT32  = 1,
    LAYA_ONNX_UINT8    = 2,
    LAYA_ONNX_INT8     = 3,
    LAYA_ONNX_UINT16   = 4,
    LAYA_ONNX_INT16    = 5,
    LAYA_ONNX_INT32    = 6,
    LAYA_ONNX_INT64    = 7,
    LAYA_ONNX_BOOL     = 9,
    LAYA_ONNX_FLOAT16  = 10,
    LAYA_ONNX_FLOAT64  = 11,
    LAYA_ONNX_BFLOAT16 = 16,
} LayaOnnxDType;

// LayaOnnxTensor describes one input or output of a run.
//
// `data` is host memory owned by the caller and must stay valid until the run
// call returns. For an output the bridge fills it with the model's result; the
// caller sizes it from the shape the model declares, which
// LayaOnnx_GetTensorInfo reports.
typedef struct LayaOnnxTensor {
    const char* name;
    int dtype;              // LayaOnnxDType
    void* data;             // host memory, caller-owned
    size_t data_bytes;      // capacity of `data` in bytes
    int64_t shape[8];       // dimensions
    int nb_dims;            // number of dimensions (1..8)
} LayaOnnxTensor;

// Every entry point follows the same convention: return 0 on success and 1 on
// failure, and on failure write a NUL-terminated message into `error` (a caller
// buffer of `error_bytes`). A NULL or zero-sized buffer is allowed and simply
// discards the message.

// ── Lifecycle ───────────────────────────────────────────────────────────────

// Create a session for `model_path`, loading `runtime_path` as the ONNX Runtime
// library. Pass NULL/"" for runtime_path to search the usual locations.
//
// `provider` is "cuda", "cpu", "directml" or "" (the default: CUDA when a CUDA
// runtime is present, else CPU). `option_keys`/`option_values` carry provider
// options such as device_id, in the OrtApi's own key/value form.
//
// Returns 0 on success and writes a session handle to *out.
LAYA_ONNX_API int LayaOnnx_CreateSession(
    const char* runtime_path, const char* model_path, const char* provider,
    const char* const* option_keys, const char* const* option_values,
    size_t option_count, void** out, char* error, size_t error_bytes);

// Release a session and everything it owns. NULL is a no-op.
LAYA_ONNX_API void LayaOnnx_DestroySession(void* session);

// Number of inputs and outputs the model declares.
LAYA_ONNX_API int LayaOnnx_GetInputCount(void* session, size_t* out, char* error, size_t error_bytes);
LAYA_ONNX_API int LayaOnnx_GetOutputCount(void* session, size_t* out, char* error, size_t error_bytes);

// Describe one input (is_input != 0) or output by index.
//
// Copies the name into `name` (a caller buffer of `name_bytes`), and writes the
// element type, the rank and the dimensions. A dynamic dimension is reported as
// -1. `shape` must hold `max_dims` entries.
LAYA_ONNX_API int LayaOnnx_GetTensorInfo(
    void* session, int is_input, size_t index, char* name, size_t name_bytes,
    int* dtype, int64_t* shape, int* nb_dims, int max_dims, char* error, size_t error_bytes);

// ── Execution ───────────────────────────────────────────────────────────────

// Run one forward pass.
//
// `inputs` are the model's inputs in declaration order. `output_names` are the
// tensors to compute, in the order the caller wants to read them back; pass
// NULL/0 to compute every output the model declares.
//
// Output memory is owned by ONNX Runtime, not the caller. That is deliberate: a
// graph exported with dynamic axes declares its outputs as symbolic dimensions,
// so there is no shape to size a buffer from until the graph has actually run.
// Read the results back with the LayaOnnx_Result* calls below, which copy into
// caller-supplied memory.
LAYA_ONNX_API int LayaOnnx_Run(
    void* session, const LayaOnnxTensor* inputs, size_t input_count,
    const char* const* output_names, size_t output_count, char* error, size_t error_bytes);

// Number of results produced by the most recent run.
LAYA_ONNX_API int LayaOnnx_GetResultCount(void* session, size_t* out, char* error, size_t error_bytes);

// Describe one result of the most recent run.
//
// Copies the name into `name` and writes the dtype, rank and the concrete
// dimensions the run produced. Unlike LayaOnnx_GetTensorInfo, a dimension here
// is never negative: the graph has run, so the shapes are known.
LAYA_ONNX_API int LayaOnnx_GetResultInfo(
    void* session, size_t index, char* name, size_t name_bytes,
    int* dtype, int64_t* shape, int* nb_dims, int max_dims, char* error, size_t error_bytes);

// Copy one result's bytes into `dst`.
//
// `dst_bytes` must be at least the result's size, which LayaOnnx_GetResultInfo
// reports. `written` receives the byte count actually copied.
LAYA_ONNX_API int LayaOnnx_CopyResult(
    void* session, size_t index, void* dst, size_t dst_bytes, size_t* written,
    char* error, size_t error_bytes);

// ── Introspection ───────────────────────────────────────────────────────────

// Copy the runtime library's version string into `buffer`.
LAYA_ONNX_API int LayaOnnx_RuntimeVersion(void* session, char* buffer, size_t buffer_bytes);

// Read the version of an ONNX Runtime library without creating a session.
//
// Loads `runtime_path` (or searches the usual locations when it is NULL/""),
// reads OrtGetApiBase()->GetVersionString(), and releases the library again.
// This is what lets a diagnostics panel report which runtime is installed
// before any model is chosen, and spot a stale DLL.
//
// Returns 0 on success, 1 when the library cannot be loaded or does not export
// the expected entry point. An empty buffer is not an error.
LAYA_ONNX_API int LayaOnnx_RuntimeVersionAt(
    const char* runtime_path, char* buffer, size_t buffer_bytes, char* error, size_t error_bytes);

// Copy the name of the execution provider actually in use into `buffer`.
LAYA_ONNX_API int LayaOnnx_ProviderName(void* session, char* buffer, size_t buffer_bytes);

// Copy a note explaining why the requested provider could not be used, or "" if
// it was honoured. Reported rather than treated as an error, so a fall back to
// CPU is never silent.
LAYA_ONNX_API int LayaOnnx_ProviderNote(void* session, char* buffer, size_t buffer_bytes);

#ifdef __cplusplus
}
#endif

#endif // LAYA_ONNX_NATIVE_H
