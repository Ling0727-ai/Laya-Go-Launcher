#ifndef AI_TENSORRT_CPP_H
#define AI_TENSORRT_CPP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// Opaque handles
typedef void* TensorRTEngineHandle;
typedef void* TensorRTContextHandle;
typedef void* TensorRTBufferHandle;
typedef void* TensorRTEventHandle;

// Error callback function type
typedef void (*ErrorCallback)(const char* error);

// ── Multi-IO tensor descriptor ──────────────────────────────────────────────

// Tensor element types understood by the kernel. These mirror
// nvinfer1::DataType for the subset a transformer uses; every buffer is sized
// from this value instead of assuming sizeof(float), which is what makes int64
// token ids and bool masks expressible.
typedef enum TensorIOType {
    TENSOR_IO_FLOAT32  = 0,
    TENSOR_IO_FLOAT16  = 1,
    TENSOR_IO_BFLOAT16 = 2,
    TENSOR_IO_INT32    = 3,
    TENSOR_IO_INT64    = 4,
    TENSOR_IO_BOOL     = 5,
    TENSOR_IO_INT8     = 6,
    TENSOR_IO_UINT8    = 7,
} TensorIOType;

// TensorIOTensor describes a named input or output tensor for the engine.
// Used by TensorRT_RunInferenceMultiIO to support models with arbitrary
// numbers of inputs/outputs.
//
// `dtype` selects the element size used for every buffer calculation. A
// language model passes TENSOR_IO_INT64 for token ids and TENSOR_IO_BOOL for
// its attention mask; the image models this kernel grew up on leave it at
// TENSOR_IO_FLOAT32. When `dtype` is 0 but the engine reports a different type
// for that tensor, the engine's type wins, so float-only callers are unaffected.
typedef struct {
    const char* name;      // tensor name in the engine (e.g. "input_ids")
    int dtype;             // TensorIOType; 0 = float32
    void* data;            // host memory; NULL for outputs = GPU-allocated only
    int64_t shape[8];      // tensor shape (up to nvinfer1::Dims::MAX_DIMS)
    int nbDims;            // number of dimensions
} TensorIOTensor;

#ifdef _WIN32
#  define TENSORRT_API __declspec(dllexport)
#else
#  define TENSORRT_API
#endif

// ── Lifecycle ───────────────────────────────────────────────────────────────

// Initialize TensorRT runtime. Returns 0 on success, -1 on failure.
TENSORRT_API int TensorRT_Initialize();

// Cleanup TensorRT runtime.
TENSORRT_API void TensorRT_Cleanup();

// TensorRT version the runtime is linked against, as a static string.
TENSORRT_API const char* TensorRT_Version();

// ── Engine management ───────────────────────────────────────────────────────

// Load a TensorRT engine from file. Returns engine handle on success, NULL on failure.
TENSORRT_API TensorRTEngineHandle TensorRT_LoadEngine(const char* enginePath);

// Unload an engine and free resources.
TENSORRT_API void TensorRT_UnloadEngine(TensorRTEngineHandle engine);

// Get engine input dimensions (NCHW). Returns 0 on success, -1 on failure.
TENSORRT_API int TensorRT_GetInputDims(TensorRTEngineHandle engine, int* dims);

// Get engine output dimensions (NCHW). Returns 0 on success, -1 on failure.
TENSORRT_API int TensorRT_GetOutputDims(TensorRTEngineHandle engine, int* dims);

// Get the engine's worst-case activation memory per execution context, in bytes.
TENSORRT_API uint64_t TensorRT_GetDeviceMemorySize(TensorRTEngineHandle engine);

// ── Context management (each context = dedicated IExecutionContext) ─────────

// Create a dedicated execution context for an engine.
// Each context has its own IExecutionContext, CUDA stream, GPU buffers,
// and pinned host buffers — enabling true multi-context parallelism.
// Returns context handle on success, NULL on failure.
TENSORRT_API TensorRTContextHandle TensorRT_CreateContext(TensorRTEngineHandle engine);

// Destroy execution context and all associated resources.
TENSORRT_API void TensorRT_DestroyContext(TensorRTContextHandle context);

// Set the fidelity weight parameter for models that require it (e.g. CodeFormer).
// Must be called before TensorRT_RunInference / Async.
TENSORRT_API void TensorRT_SetWeight(TensorRTContextHandle context, float weight);

// ── Synchronous inference (backward-compatible) ─────────────────────────────

// Run inference on a single image (blocking).
// input:  float32 NCHW tensor [1, 3, height, width]
// output: float32 NCHW tensor [1, 3, outHeight, outWidth] (caller-allocated)
// Returns 0 on success, -1 on failure.
TENSORRT_API int TensorRT_RunInference(TensorRTContextHandle context,
                          int width, int height,
                          const float* input, size_t inputSize,
                          float* output, size_t outputSize);

// Use caller-owned page-locked host buffers for synchronous inference.
// The caller owns the buffers and must keep them valid until the call returns.
TENSORRT_API void TensorRT_SetHostBuffers(TensorRTContextHandle context,
                                  float* input, size_t inputSize,
                                  float* output, size_t outputSize);

// ── CodeFormer-specific inference ───────────────────────────────────────────
//
// CodeFormer has a fixed 512×512 input shape and requires a separate weight
// tensor ("w") alongside the image tensor ("x"). This dedicated function uses
// cached tensor addresses and input shape — avoiding the unconditional
// rebind of the generic RunInference path which can produce green-stripe
// artifacts with multi-input models.
//
//   input:  float32 NCHW tensor [1, 3, 512, 512], normalized [-1, 1]
//   output: float32 NCHW tensor [1, 3, 512, 512], normalized [-1, 1] (caller-allocated)
//   weight: fidelity weight [0.0, 1.0] — 0.0 = strongest restoration
//
// Returns 0 on success, -1 on failure.
TENSORRT_API int TensorRT_RunCodeFormerInference(TensorRTContextHandle context,
    const float* input, size_t inputSize,
    float* output, size_t outputSize,
    float weight);

// ── Multi-IO inference (supports arbitrary inputs/outputs) ──────────────────

// Run inference with multiple named inputs and outputs.
// Each input/output is described by a TensorIOTensor struct.
// Input data is copied from host → GPU; output data is copied GPU → host
// into the caller-provided data buffers (outputs[i].data must be pre-allocated).
//
// Example (a decision model: 5 inputs, 2 outputs, mixed dtypes)
//   TensorIOTensor inputs[5] = {
//     {"input_ids",      TENSOR_IO_INT64,   ids,  {1, L}, 2},
//     {"attention_mask", TENSOR_IO_INT64,   mask, {1, L}, 2},
//     {"marker_pos",     TENSOR_IO_INT64,   pos,  {1, K}, 2},
//     {"marker_mask",    TENSOR_IO_BOOL,    mm,   {1, K}, 2},
//     {"qtype",          TENSOR_IO_INT64,   qt,   {1},    1},
//   };
//   TensorIOTensor outputs[2] = {
//     {"logits",     TENSOR_IO_FLOAT32, logitsBuf, {1, K}, 2},
//     {"act_logits", TENSOR_IO_FLOAT32, actBuf,    {1, 2}, 2},
//   };
//   TensorRT_RunInferenceMultiIO(ctx, 5, inputs, 2, outputs);
//
// Returns 0 on success, -1 on failure.
TENSORRT_API int TensorRT_RunInferenceMultiIO(TensorRTContextHandle context,
    int numInputs,  const TensorIOTensor* inputs,
    int numOutputs, TensorIOTensor* outputs);

// Resolve the concrete shape and dtype of each named output for the given
// input shapes, without transferring any tensor data. Use it to size caller
// output buffers before TensorRT_RunInferenceMultiIO.
//
//   outShapes   : numOutputs * 8 int64 slots, row-major per output
//   outNbDims   : numOutputs entries
//   outDtypes   : numOutputs entries (TensorIOType), may be NULL
//
// Returns 0 on success, -1 on failure.
TENSORRT_API int TensorRT_ResolveOutputShapes(TensorRTContextHandle context,
    int numInputs, const TensorIOTensor* inputs,
    int numOutputs, const char* const* outputNames,
    int64_t* outShapes, int* outNbDims, int* outDtypes);

// Number of bytes one element of the given TensorIOType occupies.
TENSORRT_API size_t TensorRT_DtypeSize(int dtype);

// ── Native run builder ──────────────────────────────────────────────────────
//
// A run is a native-owned description of one inference. Callers add inputs and
// outputs by name through these functions and never construct a TensorIOTensor
// themselves, so no foreign-language struct layout, allocation, or lifetime
// crosses the boundary.
//
// Input data is COPIED into run-owned storage by TensorRT_RunAddInput, so the
// caller's buffer is free as soon as that call returns. Outputs are allocated
// and owned by the run; read them back with TensorRT_RunReadOutput.

typedef void* TensorRunHandle;

// Create a run bound to a context. Returns NULL on failure.
TENSORRT_API TensorRunHandle TensorRT_RunCreate(TensorRTContextHandle context);

// Destroy a run and free everything it owns. NULL is ignored.
TENSORRT_API void TensorRT_RunDestroy(TensorRunHandle run);

// Add an input. `data` must hold product(shape) elements of `dtype`; it is
// copied immediately. Returns 0 on success, -1 on failure.
TENSORRT_API int TensorRT_RunAddInput(TensorRunHandle run, const char* name,
    int dtype, const void* data, const int64_t* shape, int nbDims);

// Declare an output by name. Its shape and dtype are read from the engine once
// the inputs are known. Returns 0 on success, -1 on failure.
TENSORRT_API int TensorRT_RunAddOutput(TensorRunHandle run, const char* name);

// Number of declared outputs.
TENSORRT_API int TensorRT_RunOutputCount(TensorRunHandle run);

// Optimisation-profile bounds for a dynamic input tensor.
//
// `which` selects the profile point: 0 = min, 1 = opt, 2 = max. Writes nbDims
// values into `shape`. Returns 1 when a profile exists for that tensor, 0 when
// the dimension is static (no profile), -1 on error.
//
// This is how a caller learns the range of sequence lengths and marker counts an
// engine actually accepts, instead of guessing from the build shape.
TENSORRT_API int TensorRT_GetProfileShape(TensorRTEngineHandle engine,
    const char* name, int which, int64_t* shape, int* nbDims);

// Resolve every declared output against the current inputs. Fills the caller's
// arrays, each with room for `count` entries (`shape` holds count * 8 int64).
// Returns 0 on success, -1 on failure.
TENSORRT_API int TensorRT_RunResolveOutputs(TensorRunHandle run,
    char* names, int nameStride, int* dtypes, int64_t* shapes, int* nbDims,
    int count);

// Execute the run, blocking until outputs are in run-owned host memory.
// Returns 0 on success, -1 on failure.
TENSORRT_API int TensorRT_RunExecute(TensorRunHandle run);

// Byte size of a resolved output, or 0 if unknown.
TENSORRT_API size_t TensorRT_RunOutputBytes(TensorRunHandle run, int index);

// Copy a resolved output into the caller's buffer. `bytes` must be at least
// TensorRT_RunOutputBytes(run, index). Returns 0 on success, -1 on failure.
TENSORRT_API int TensorRT_RunReadOutput(TensorRunHandle run, int index,
    void* dst, size_t bytes);

// Get the number of IO tensors in the engine.
TENSORRT_API int TensorRT_GetNumIOTensors(TensorRTEngineHandle engine);

// Get IO tensor info by index. Returns 0 on success, -1 on failure.
// name: output buffer (caller allocates at least 256 bytes).
// isInput: 1 if input tensor, 0 if output tensor.
// shape: output shape array (4 ints).
// nbDims: output dimension count.
TENSORRT_API int TensorRT_GetIOTensorInfo(TensorRTEngineHandle engine,
    int index, char* name, int* isInput, int64_t* shape, int* nbDims);

// Like TensorRT_GetIOTensorInfo but also reports the engine's element type as
// a TensorIOType. dtype may be NULL if the caller does not need it.
TENSORRT_API int TensorRT_GetIOTensorInfoEx(TensorRTEngineHandle engine,
    int index, char* name, int* isInput, int* dtype, int64_t* shape, int* nbDims);

// ── Asynchronous inference (new — enables CPU/GPU pipelining) ───────────────

// Launch inference asynchronously. Returns immediately; the caller synchronizes
// via the returned CUDA event. After the event completes, call
// TensorRT_CopyOutputFromPinned to retrieve results.
//
// outEvent: receives a CUDA event handle. Caller must destroy it via
//           TensorRT_EventDestroy after use.
//
// Returns 0 on success, -1 on failure.
TENSORRT_API int TensorRT_RunInferenceAsync(TensorRTContextHandle context,
                               int width, int height,
                               const float* input, size_t inputSize,
                               float* output, size_t outputSize,
                               TensorRTEventHandle* outEvent);

// Slot-based async API. Each slot owns independent staging/device buffers so
// two submissions can remain in flight on the context stream.
TENSORRT_API int TensorRT_SubmitInferenceSlot(TensorRTContextHandle context,
                               int slot, int width, int height,
                               const float* input, size_t inputSize,
                               size_t outputSize,
                               TensorRTEventHandle* outEvent);
TENSORRT_API int TensorRT_WaitInferenceSlot(TensorRTEventHandle event);
TENSORRT_API int TensorRT_CopyOutputSlot(TensorRTContextHandle context,
                               int slot, float* output, size_t outputSize);
TENSORRT_API void TensorRT_ReleaseInferenceSlot(TensorRTContextHandle context, int slot,
                               TensorRTEventHandle event);

// ── Event helpers ───────────────────────────────────────────────────────────

// Block until the event completes. Returns 0 on success, -1 on error.
TENSORRT_API int TensorRT_EventSynchronize(TensorRTEventHandle event);

// Query event status. Returns 1 if complete, 0 if still running, -1 on error.
TENSORRT_API int TensorRT_EventQuery(TensorRTEventHandle event);

// Destroy a CUDA event.
TENSORRT_API void TensorRT_EventDestroy(TensorRTEventHandle event);

// ── Post-async output retrieval ─────────────────────────────────────────────

// After TensorRT_EventSynchronize completes, copy the pinned output buffer
// to the caller-provided output buffer. Must be called after the event signals.
// Returns 0 on success, -1 on failure.
TENSORRT_API int TensorRT_CopyOutputFromPinned(TensorRTContextHandle context,
                                  float* output, size_t outputSize);

// ── VRAM & device query ──────────────────────────────────────────────────────

// Query free and total VRAM (MiB) from CUDA. Returns 0 on success.
TENSORRT_API int TensorRT_GetVRAMInfo(int* freeMB, int* totalMB);

// Query device name and compute capability. nameBuf must be at least 256 bytes.
// Returns 0 on success, -1 on failure.
TENSORRT_API int TensorRT_GetDeviceInfo(char* nameBuf, int* computeMajor, int* computeMinor);

// ── Pinned (page-locked) host memory ─────────────────────────────────────────
// These enable Go to allocate pinned memory directly, avoiding the intermediate
// pageable→pinned copy inside the inference path.

// Allocate pinned host memory. Returns NULL on failure.
TENSORRT_API float* TensorRT_AllocPinned(size_t bytes);

// Free pinned host memory.
TENSORRT_API void TensorRT_FreePinned(float* ptr);

// ── Debug control ────────────────────────────────────────────────────────────

// Enable or disable verbose diagnostic logging in the inference hot path.
// Default: disabled (0). Set to 1 to enable TRT-diag fprintf output.
TENSORRT_API void TensorRT_SetDiagnostics(int enabled);

// ── Error reporting ─────────────────────────────────────────────────────────

// Get last error message.
TENSORRT_API const char* TensorRT_GetLastError();

#ifdef __cplusplus
}
#endif

#endif // AI_TENSORRT_CPP_H
