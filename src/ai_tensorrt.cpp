//go:build ignore

// TensorRT 10.x compatible implementation — optimized version
//
// Key optimizations vs original:
// 1. Async inference path (TensorRT_RunInferenceAsync) — no cudaStreamSynchronize,
//    caller syncs via returned CUDA event. Sync wrapper still available for compat.
// 2. Cached input shape — setInputShape only called when dims actually change.
// 3. Cached tensor addresses — setTensorAddress only called when buffer ptrs change.
// 4. Max-capacity GPU buffer pre-allocation — avoids runtime cudaFree/Malloc churn.
// 5. Per-context lock (already per-context via Go pool) — lock scope minimized to
//    enqueueV3 only; memcpy runs outside lock for better pipelining.
// 6. Pinned (page-locked) host memory for staging buffers — faster H2D/D2H transfers.

#include "ai_tensorrt_cpp.h"
#include <fstream>
#include <iostream>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <memory>
#include <vector>
#include <cstring>
#include <mutex>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <sstream>
#include <chrono>
#include <cstdlib>

using namespace nvinfer1;

struct TensorIOCache;

// ── Element-type helpers ────────────────────────────────────────────────────
//
// This kernel was written for float32 image tensors and sized every buffer with
// sizeof(float). A language model hands it int64 token ids and bool masks, so
// buffer arithmetic has to come from the tensor's element type instead. These
// helpers are the single place that mapping lives.

// Bytes per element for a TensorIOType. Unknown types fall back to float32 so
// existing float-only callers cannot regress.
static size_t dtypeElemSize(int dtype) {
    switch (dtype) {
        case TENSOR_IO_FLOAT32:  return 4;
        case TENSOR_IO_FLOAT16:  return 2;
        case TENSOR_IO_BFLOAT16: return 2;
        case TENSOR_IO_INT32:    return 4;
        case TENSOR_IO_INT64:    return 8;
        case TENSOR_IO_BOOL:     return 1;
        case TENSOR_IO_INT8:     return 1;
        case TENSOR_IO_UINT8:    return 1;
        default:                 return 4;
    }
}

// Map an engine DataType onto a TensorIOType, or -1 when unsupported.
static int toTensorIOType(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return TENSOR_IO_FLOAT32;
        case nvinfer1::DataType::kHALF:  return TENSOR_IO_FLOAT16;
        case nvinfer1::DataType::kBF16:  return TENSOR_IO_BFLOAT16;
        case nvinfer1::DataType::kINT32: return TENSOR_IO_INT32;
        case nvinfer1::DataType::kINT64: return TENSOR_IO_INT64;
        case nvinfer1::DataType::kBOOL:  return TENSOR_IO_BOOL;
        case nvinfer1::DataType::kINT8:  return TENSOR_IO_INT8;
        case nvinfer1::DataType::kUINT8: return TENSOR_IO_UINT8;
        default:                         return -1;
    }
}

// ── Logger ──────────────────────────────────────────────────────────────────
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

static Logger gLogger;
static std::string gLastError;
static std::mutex gErrorMutex;  // protect gLastError writes
static int gDiagnosticsEnabled = 0;  // off by default — enable via TensorRT_SetDiagnostics(1)

// Diagnostic log helper — only print when diagnostics are explicitly enabled.
#define TRT_DIAG(fmt, ...) do { if (gDiagnosticsEnabled) fprintf(stderr, fmt, ##__VA_ARGS__); } while(0)

// ── Smart pointer deleters ──────────────────────────────────────────────────
struct TrtDestroyer {
    template <class T>
    void operator()(T* obj) const {
        if (obj) delete obj;
    }
};

template <class T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer>;

// ── Per-context state (one per IExecutionContext) ───────────────────────────
//
// In the original code, TensorRT_CreateContext returned the engine handle itself,
// so all "contexts" shared the same IExecutionContext + GPU buffers + lock.
// Now each call to TensorRT_CreateContext creates a *dedicated* IExecutionContext
// with its own GPU buffers, stream, and lock — enabling true multi-context
// parallelism when the Go side uses a context pool.
struct TensorRTContextObj {
    TrtUniquePtr<IExecutionContext> context;
    void* activationMemory = nullptr;
    size_t activationCapacity = 0;
    bool activationMemoryDirty = true;
    void* gpuInputBuffer = nullptr;
    void* gpuOutputBuffer = nullptr;
    size_t gpuInputCapacity = 0;   // allocated GPU input buffer size (bytes)
    size_t gpuOutputCapacity = 0;  // allocated GPU output buffer size (bytes)
    cudaStream_t stream = nullptr;
    std::mutex mu;  // protects enqueueV3 (IExecutionContext is not thread-safe)

    // Cached state to avoid redundant API calls
    int cachedInputDims[4] = {-1, -1, -1, -1};
    void* cachedInputAddr = nullptr;
    void* cachedOutputAddr = nullptr;

    // Pinned host staging buffers (allocated on first use, reused thereafter)
    float* pinnedInput = nullptr;
    float* pinnedOutput = nullptr;
    size_t pinnedInputSize = 0;
    size_t pinnedOutputSize = 0;
    float* externalPinnedInput = nullptr;
    float* externalPinnedOutput = nullptr;
    size_t externalPinnedInputSize = 0;
    size_t externalPinnedOutputSize = 0;

    // Engine metadata (copied from engine for convenience)
    int inputDims[4] = {0};
    int outputDims[4] = {0};
    std::string inputName;
    std::string outputName;
    std::vector<std::string> allOutputNames;   // ALL output tensor names
    std::vector<void*> allOutputBuffers;       // GPU buffers for extra outputs
    bool extraOutputsBound = false;
    void* extraOutputsPrimaryBuffer = nullptr;

    // Weight tensor for CodeFormer etc.
    std::string weightName;
    bool hasWeight = false;
    float cachedWeight = 0.5f;
    void* gpuWeightBuffer = nullptr;
    bool weightAddressSet = false;
    TensorIOCache* ioCache = nullptr;

    // Optimisation profile the context is bound to. A plan built with several
    // profiles (batch 8 up to 512 tokens, batch 1 up to 8192) is switched per run
    // to the first profile that admits the input shapes; see selectProfile().
    int currentProfile = 0;

    // True from any change to what the engine executes against (an input shape,
    // the profile, the activation arena or an IO buffer address) until the next
    // plain enqueueV3. TensorRT defers part of a shape change to the following
    // enqueue, so that enqueue must run uncaptured, and any graph captured
    // before the change is stale. See enqueueWithGraph().
    bool pendingChange = true;

    // One CUDA graph for the shapes currently bound to this context. It is not a
    // per-shape cache on purpose: replaying a graph for shape A after the context
    // has been moved to shape B would read B's deferred device-side state.
    cudaGraphExec_t graphExec = nullptr;
    std::string graphKey;
    // Key of the last plain enqueue. A graph is captured only when the same key
    // repeats with no change in between, i.e. from the second identical run.
    std::string lastEnqueueKey;
    // Keys whose capture failed, so they are not retried on every run.
    std::vector<std::string> graphFailedKeys;
};

// ── Engine-level state (shared across contexts) ─────────────────────────────
struct TensorRTEngineObj {
    TrtUniquePtr<ICudaEngine> engine;
    int inputDims[4] = {0};
    int outputDims[4] = {0};
    std::string inputName;
    std::string outputName;
    std::vector<std::string> allOutputNames;   // ALL output tensor names

    std::string weightName;
    bool hasWeight = false;
};

// ── Helpers ─────────────────────────────────────────────────────────────────
static void setError(const char* msg) {
    std::lock_guard<std::mutex> lock(gErrorMutex);
    gLastError = msg;
}

static bool setContextInputShape(TensorRTContextObj* ctx, const char* name, const Dims& dims) {
    Dims current = ctx->context->getTensorShape(name);
    bool changed = current.nbDims != dims.nbDims;
    for (int i = 0; !changed && i < dims.nbDims; ++i) {
        changed = current.d[i] != dims.d[i];
    }
    if (!changed) return true;
    // A shape change can repartition activation memory used by an earlier async run.
    if (cudaStreamSynchronize(ctx->stream) != cudaSuccess) {
        setError("Failed to synchronize before changing TensorRT input shape");
        return false;
    }
    if (!ctx->context->setInputShape(name, dims)) {
        setError("Input shape does not satisfy the TensorRT optimization profile");
        return false;
    }
    ctx->activationMemoryDirty = true;
    ctx->pendingChange = true;
    return true;
}

static bool ensureActivationMemory(TensorRTContextObj* ctx) {
    if (!ctx->activationMemoryDirty) return true;
    if (cudaStreamSynchronize(ctx->stream) != cudaSuccess) {
        setError("Failed to synchronize before resizing TensorRT activation memory");
        return false;
    }
    size_t required = ctx->context->updateDeviceMemorySizeForShapes();
    // Some networks cannot refine the bound; retain the engine's safe upper bound.
    if (required == 0) required = ctx->context->getEngine().getDeviceMemorySizeV2();
    if (required > ctx->activationCapacity) {
        // Grow geometrically, but never past the plan's own upper bound: each
        // reallocation moves the arena, which invalidates the captured CUDA
        // graph and costs a synchronising cudaFree/cudaMalloc, so a series of
        // slightly longer requests should not pay one reallocation each.
        size_t upper = static_cast<size_t>(ctx->context->getEngine().getDeviceMemorySizeV2());
        size_t grown = std::max(required, ctx->activationCapacity * 2);
        if (upper >= required) grown = std::min(grown, upper);
        if (ctx->activationMemory) cudaFree(ctx->activationMemory);
        ctx->activationMemory = nullptr;
        ctx->activationCapacity = 0;
        if (cudaMalloc(&ctx->activationMemory, grown) != cudaSuccess) {
            // The spare capacity is an optimisation; retry at the exact size.
            cudaGetLastError();
            grown = required;
            if (cudaMalloc(&ctx->activationMemory, grown) != cudaSuccess) {
                setError("Failed to allocate TensorRT activation memory for input shape");
                return false;
            }
        }
        ctx->activationCapacity = grown;
    }
    ctx->context->setDeviceMemoryV2(ctx->activationMemory, ctx->activationCapacity);
    ctx->activationMemoryDirty = false;
    ctx->pendingChange = true;
    TRT_DIAG("[TRT-memory] activation_required=%zu activation_capacity=%zu profile_upper_bound=%lld\n",
             required, ctx->activationCapacity,
             static_cast<long long>(ctx->context->getEngine().getDeviceMemorySizeV2()));
    return true;
}

// ── Optimisation-profile selection ──────────────────────────────────────────
//
// A plan may carry several profiles, typically a batched one for the trained
// context and a single-row one for long inputs. One batched profile up to the
// long ceiling does not build (its worst-case activation exceeds the GPU), and
// one single-row profile forces a forward pass per question. So the context is
// moved per run to the lowest-numbered profile that admits every input shape;
// profile 0 is the one tuned for the common request.

struct ShapeRequest {
    const char* name;
    Dims dims;
};

static bool shapesFitProfile(const ICudaEngine& engine, int profile,
                             const ShapeRequest* reqs, int n) {
    for (int i = 0; i < n; i++) {
        Dims lo, hi;
        try {
            lo = engine.getProfileShape(reqs[i].name, profile, OptProfileSelector::kMIN);
            hi = engine.getProfileShape(reqs[i].name, profile, OptProfileSelector::kMAX);
        } catch (...) {
            return false;
        }
        // A static input has no profile entry; any shape the engine declares is fine.
        if (lo.nbDims <= 0 || hi.nbDims <= 0) continue;
        if (lo.nbDims != reqs[i].dims.nbDims) return false;
        for (int d = 0; d < lo.nbDims; d++) {
            if (reqs[i].dims.d[d] < lo.d[d] || reqs[i].dims.d[d] > hi.d[d]) return false;
        }
    }
    return true;
}

// Bind the context to a profile that admits `reqs`, then apply the shapes.
static bool applyInputShapes(TensorRTContextObj* ctx, const ShapeRequest* reqs, int n) {
    const ICudaEngine& engine = ctx->context->getEngine();
    int profiles = engine.getNbOptimizationProfiles();
    bool force = false;
    if (profiles > 1) {
        int chosen = -1;
        for (int p = 0; p < profiles; p++) {
            if (shapesFitProfile(engine, p, reqs, n)) { chosen = p; break; }
        }
        if (chosen < 0) {
            setError("Input shapes do not satisfy any optimization profile of this engine");
            return false;
        }
        if (chosen != ctx->currentProfile) {
            // The previous run may still be using the arena and bindings.
            if (cudaStreamSynchronize(ctx->stream) != cudaSuccess) {
                setError("Failed to synchronize before switching optimization profile");
                return false;
            }
            if (!ctx->context->setOptimizationProfileAsync(chosen, ctx->stream)) {
                setError("setOptimizationProfileAsync failed");
                return false;
            }
            // The profile switch may enqueue copies; the next enqueue is on the
            // same stream, so ordering is already guaranteed.
            ctx->currentProfile = chosen;
            ctx->activationMemoryDirty = true;
            ctx->pendingChange = true;
            force = true;
        }
    }
    for (int i = 0; i < n; i++) {
        if (force) {
            if (!ctx->context->setInputShape(reqs[i].name, reqs[i].dims)) {
                setError("Input shape does not satisfy the TensorRT optimization profile");
                return false;
            }
        } else if (!setContextInputShape(ctx, reqs[i].name, reqs[i].dims)) {
            return false;
        }
    }
    return true;
}

// ── CUDA graph replay ───────────────────────────────────────────────────────
//
// enqueueV3 on this transformer issues thousands of kernel launches, and on a
// WDDM driver the host-side launch cost alone is 3-5 ms per run: most of the
// latency of a short request. Capturing the enqueue into a CUDA graph and
// replaying it cuts that to tens of microseconds, without touching the math, so
// the outputs are bit-identical to a plain enqueue.
//
// Rules this follows (TensorRT developer guide, "CUDA Graphs"):
//   - after any shape, profile or address change, the next enqueueV3 must run
//     outside capture, because TensorRT defers part of the update to it;
//   - a graph is only valid for the shapes and device addresses it saw.
// So a graph is captured on the second consecutive run of an unchanged key,
// replayed while nothing changes, and dropped on the first change.

static void dropGraph(TensorRTContextObj* ctx) {
    if (ctx->graphExec) {
        cudaGraphExecDestroy(ctx->graphExec);
        ctx->graphExec = nullptr;
    }
    ctx->graphKey.clear();
}

static std::string graphKeyFor(const TensorRTContextObj* ctx, const ShapeRequest* reqs, int n) {
    std::string key = "p" + std::to_string(ctx->currentProfile);
    for (int i = 0; i < n; i++) {
        key += '|';
        key += reqs[i].name;
        for (int d = 0; d < reqs[i].dims.nbDims; d++) {
            key += ',';
            key += std::to_string(reqs[i].dims.d[d]);
        }
    }
    return key;
}

static bool graphsDisabled() {
    static const bool disabled = [] {
        const char* v = std::getenv("LAYA_TRT_CUDA_GRAPH");
        return v && (v[0] == '0' || v[0] == 'n' || v[0] == 'N' || v[0] == 'f' || v[0] == 'F');
    }();
    return disabled;
}

static bool enqueueWithGraph(TensorRTContextObj* ctx, const std::string& key) {
    if (ctx->pendingChange || graphsDisabled()) {
        dropGraph(ctx);
        bool ok = ctx->context->enqueueV3(ctx->stream);
        ctx->pendingChange = !ok;
        ctx->lastEnqueueKey = ok ? key : std::string();
        return ok;
    }

    if (ctx->graphExec && ctx->graphKey == key) {
        return cudaGraphLaunch(ctx->graphExec, ctx->stream) == cudaSuccess;
    }

    bool failedBefore = std::find(ctx->graphFailedKeys.begin(), ctx->graphFailedKeys.end(), key)
                        != ctx->graphFailedKeys.end();
    if (ctx->lastEnqueueKey != key || failedBefore) {
        dropGraph(ctx);
        bool ok = ctx->context->enqueueV3(ctx->stream);
        ctx->lastEnqueueKey = ok ? key : std::string();
        return ok;
    }

    // Second unchanged run of this key: capture it.
    dropGraph(ctx);
    cudaGraph_t graph = nullptr;
    bool captured = false;
    if (cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
        bool enq = ctx->context->enqueueV3(ctx->stream);
        cudaError_t end = cudaStreamEndCapture(ctx->stream, &graph);
        if (enq && end == cudaSuccess && graph) {
            cudaGraphExec_t exec = nullptr;
            if (cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess) {
                ctx->graphExec = exec;
                ctx->graphKey = key;
                captured = true;
            }
        }
        if (graph) cudaGraphDestroy(graph);
    }
    if (!captured) {
        // Clear the sticky capture error and remember not to retry this key.
        cudaGetLastError();
        if (ctx->graphFailedKeys.size() >= 64) ctx->graphFailedKeys.erase(ctx->graphFailedKeys.begin());
        ctx->graphFailedKeys.push_back(key);
        TRT_DIAG("[TRT-graph] capture failed for %s; using plain enqueue\n", key.c_str());
        bool ok = ctx->context->enqueueV3(ctx->stream);
        ctx->lastEnqueueKey = ok ? key : std::string();
        return ok;
    }
    return cudaGraphLaunch(ctx->graphExec, ctx->stream) == cudaSuccess;
}

static std::vector<char> readEngineFile(const char* enginePath) {
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        setError("Failed to open engine file");
        return {};
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        setError("Failed to read engine file");
        return {};
    }
    return buffer;
}

// Allocate pinned host memory, returns nullptr on failure.
static float* allocPinned(size_t bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        std::cerr << "[TensorRT] cudaHostAlloc(" << bytes << " bytes) failed: "
                  << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    return static_cast<float*>(ptr);
}

// Ensure GPU buffer is at least `required` bytes. Returns true on success.
static bool ensureGpuBuffer(void*& buf, size_t& capacity, size_t required) {
    if (required <= capacity) return true;
    // Grow by 2x to amortize reallocation cost, but at least to required.
    size_t newCap = std::max(capacity * 2, required);
    // Align to 256 bytes for good measure.
    newCap = (newCap + 255) & ~size_t(255);
    if (buf) cudaFree(buf);
    cudaError_t err = cudaMalloc(&buf, newCap);
    if (err != cudaSuccess) {
        buf = nullptr;
        capacity = 0;
        return false;
    }
    capacity = newCap;
    return true;
}

// Ensure pinned host buffer is at least `required` bytes. Returns true on success.
static bool ensurePinnedBuffer(float*& buf, size_t& capacity, size_t required) {
    if (required <= capacity) return true;
    size_t newCap = std::max(capacity * 2, required);
    newCap = (newCap + 255) & ~size_t(255);
    if (buf) cudaFreeHost(buf);
    buf = allocPinned(newCap);
    if (!buf) {
        capacity = 0;
        return false;
    }
    capacity = newCap;
    return true;
}

// TensorRT 10.x requires an address for every output before enqueueV3. Some
// imported engines expose auxiliary outputs (for example "logits") that are
// not present in the cached resolver list, so enumerate the live engine here.
static void bindAllRuntimeOutputs(TensorRTContextObj* ctx) {
    if (!ctx || !ctx->context) return;
    const auto& engine = ctx->context->getEngine();
    int32_t numIO = 0;
    try { numIO = engine.getNbIOTensors(); } catch (...) { return; }

    for (int32_t i = 0; i < numIO; ++i) {
        const char* rawName = engine.getIOTensorName(i);
        if (!rawName || engine.getTensorIOMode(rawName) != TensorIOMode::kOUTPUT) continue;
        std::string name(rawName);
        if (name == ctx->outputName) continue;

        size_t index = 0;
        while (index < ctx->allOutputNames.size() && ctx->allOutputNames[index] != name) ++index;
        if (index == ctx->allOutputNames.size()) {
            ctx->allOutputNames.push_back(name);
            ctx->allOutputBuffers.push_back(nullptr);
        }

        size_t byteSize = ctx->gpuOutputCapacity;
        try {
            Dims shape = engine.getTensorShape(name.c_str());
            size_t elems = 1;
            bool concrete = shape.nbDims > 0;
            for (int d = 0; d < shape.nbDims && d < 8; ++d) {
                if (shape.d[d] <= 0) { concrete = false; break; }
                elems *= static_cast<size_t>(shape.d[d]);
            }
            if (concrete) byteSize = std::max(byteSize, elems * sizeof(float));
        } catch (...) {}
        if (byteSize == 0) byteSize = sizeof(float);

        void*& buffer = ctx->allOutputBuffers[index];
        if (!buffer) {
            if (cudaMalloc(&buffer, byteSize) != cudaSuccess) buffer = ctx->gpuOutputBuffer;
        }
        if (buffer) {
            try { ctx->context->setTensorAddress(name.c_str(), buffer); } catch (...) {}
        }
    }
}

// Resolve tensor names from engine (tries common names).
static bool resolveTensorNames(ICudaEngine* engine, std::string& inName, std::string& outName,
                               Dims& inDims, Dims& outDims, std::string& weightName, bool& hasWeight,
                               std::vector<std::string>& allOutputNames) {
    inDims.nbDims = 0;
    outDims.nbDims = 0;
    hasWeight = false;

    // Use getNbIOTensors / getIOTensorName to safely enumerate without triggering internal errors
    int32_t numIO = 0;
    try { numIO = engine->getNbIOTensors(); } catch (...) {}

    // Track the largest output tensor as the primary output.
    // When engines expose intermediate I/O tensors (common with TensorRT 10.x),
    // the last-enumerated output may be a small intermediate, not the real result.
    // We pick the output with the largest element count — this is always the
    // real output tensor (final image data is vastly larger than any intermediate).
    size_t largestOutputElems = 0;

    TRT_DIAG("[TRT-diag] Engine has %d I/O tensors:\n", numIO);
    for (int32_t i = 0; i < numIO; ++i) {
        const char* name = engine->getIOTensorName(i);
        if (!name) continue;
        nvinfer1::TensorIOMode ioMode = engine->getTensorIOMode(name);
        const char* modeStr = (ioMode == nvinfer1::TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT";
        Dims shape = engine->getTensorShape(name);
        TRT_DIAG("[TRT-diag]   [%d] '%s' mode=%s nbDims=%lld\n", i, name, modeStr, (long long)shape.nbDims);

        if (ioMode == nvinfer1::TensorIOMode::kINPUT) {
            std::string s(name);
            if (s == "w" || s == "weight") {
                weightName = s;
                hasWeight = true;
                TRT_DIAG("[TRT-diag]   -> weight tensor\n");
            } else {
                // RivaGAN encoder also exposes a [N,32] payload input. Pick
                // the image-shaped input instead of letting the last input
                // overwrite it with the payload tensor.
                size_t elems = 1;
                for (int d = 0; d < shape.nbDims && d < 8; ++d) {
                    if (shape.d[d] > 0) elems *= static_cast<size_t>(shape.d[d]);
                }
                bool imageLike = shape.nbDims >= 4 && shape.d[1] == 3;
                bool currentImageLike = inDims.nbDims >= 4 && inDims.d[1] == 3;
                size_t currentElems = 1;
                for (int d = 0; d < inDims.nbDims && d < 8; ++d) {
                    if (inDims.d[d] > 0) currentElems *= static_cast<size_t>(inDims.d[d]);
                }
                if (inDims.nbDims <= 0 || (imageLike && !currentImageLike) ||
                    (imageLike == currentImageLike && elems > currentElems)) {
                    inName = s;
                    inDims = shape;
                    TRT_DIAG("[TRT-diag]   -> main input tensor\n");
                } else {
                    TRT_DIAG("[TRT-diag]   -> auxiliary input tensor\n");
                }
            }
        } else if (ioMode == nvinfer1::TensorIOMode::kOUTPUT) {
            allOutputNames.push_back(name);

            // Compute element count for this output tensor
            size_t elems = 1;
            for (int d = 0; d < shape.nbDims && d < 8; d++) {
                if (shape.d[d] > 0) elems *= (size_t)shape.d[d];
            }
            TRT_DIAG("[TRT-diag]   -> output tensor '%s' (%zu elems, total outputs: %zu)\n",
                    name, elems, allOutputNames.size());

            // Pick the largest output as primary (real image data dwarfs any intermediate)
            if (elems > largestOutputElems) {
                largestOutputElems = elems;
                outName = name;
                outDims = shape;
            }
        }
    }

    // Fallback if the above loop failed for some reason
    if (inDims.nbDims <= 0) {
        try { inDims = engine->getTensorShape("input"); if (inDims.nbDims > 0) { inName = "input"; TRT_DIAG("[TRT-diag] fallback: inName='input'\n"); } } catch (...) {}
    }
    if (outDims.nbDims <= 0) {
        try { outDims = engine->getTensorShape("output"); if (outDims.nbDims > 0) { outName = "output"; TRT_DIAG("[TRT-diag] fallback: outName='output'\n"); } } catch (...) {}
    }
    TRT_DIAG("[TRT-diag] Resolved: in='%s' out='%s' weight='%s' hasWeight=%d\n",
            inName.c_str(), outName.c_str(), weightName.c_str(), hasWeight);

    if (inDims.nbDims <= 0 || outDims.nbDims <= 0) {
        int32_t numIO = 0;
        try { numIO = engine->getNbIOTensors(); }
        catch (...) {}

        if (numIO > 0 && numIO <= 10) {
            try {
                const char* name0 = engine->getIOTensorName(0);
                if (name0 && std::string(name0) != "w" && std::string(name0) != "weight" && inDims.nbDims <= 0) {
                    inName = name0;
                    inDims = engine->getTensorShape(name0);
                }
            } catch (...) {}
            if (numIO > 1) {
                try {
                    const char* nameN = engine->getIOTensorName(numIO - 1);
                    if (nameN && std::string(nameN) != "w" && std::string(nameN) != "weight" && outDims.nbDims <= 0) {
                        outName = nameN;
                        outDims = engine->getTensorShape(nameN);
                    }
                } catch (...) {}
            }
        }
    }

    // Last resort: try "output0"
    if (outDims.nbDims <= 0) {
        try { outDims = engine->getTensorShape("output0"); if (outDims.nbDims > 0) outName = "output0"; }
        catch (...) {}
    }

    return (inDims.nbDims > 0 && outDims.nbDims > 0);
}

// ── Multi-IO buffer cache (per context) ─────────────────────────────────────
// Maps tensor name → GPU buffer + pinned host buffer, enabling generic
// multi-input/output inference without hardcoding "input"/"output" names.

struct TensorBuffer {
    void* gpuBuf = nullptr;
    float* pinnedBuf = nullptr;
    size_t gpuCap = 0;
    size_t pinnedCap = 0;
    void* cachedAddr = nullptr;  // last address set on the context
};

// Per-context map of tensor name → buffers. Initialized lazily.
struct TensorIOCache {
    std::unordered_map<std::string, TensorBuffer> buffers;
};

// Extend the context to include a multi-IO cache (allocated on first use).
// We use a separate pointer to avoid bloating the basic context for single-IO users.
struct TensorRTContextExt {
    TensorRTContextObj* ctx;
    TensorIOCache* ioCache;
};

// ── Exported C API ──────────────────────────────────────────────────────────

extern "C" {

#ifdef __MINGW32__
void __GSHandlerCheck() {}
void __security_check_cookie(uintptr_t cookie) {}
uintptr_t __security_cookie = 0;
#endif

int TensorRT_Initialize() {
    cudaError_t cudaStatus = cudaFree(0);
    if (cudaStatus != cudaSuccess) {
        setError("Failed to initialize CUDA");
        return -1;
    }
    return 0;
}

void TensorRT_Cleanup() {
    // no-op: per-context cleanup happens in TensorRT_DestroyContext
}

const char* TensorRT_Version() {
    static std::string version = [] {
        std::ostringstream os;
        os << getInferLibVersion();
        return os.str();
    }();
    return version.c_str();
}

TensorRTEngineHandle TensorRT_LoadEngine(const char* enginePath) {
    auto engineData = readEngineFile(enginePath);
    if (engineData.empty()) return nullptr;

    auto runtime = TrtUniquePtr<IRuntime>(createInferRuntime(gLogger));
    if (!runtime) {
        setError("Failed to create TensorRT runtime");
        return nullptr;
    }

    auto engine = TrtUniquePtr<ICudaEngine>(
        runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!engine) {
        setError("Failed to deserialize engine");
        return nullptr;
    }

    // Resolve tensor names and shapes
    std::string inName = "input", outName = "output";
    std::string weightName = "";
    bool hasWeight = false;
    Dims inDims, outDims;
    std::vector<std::string> allOutputNames;
    if (!resolveTensorNames(engine.get(), inName, outName, inDims, outDims, weightName, hasWeight, allOutputNames)) {
        setError("Failed to resolve tensor names/dimensions");
        return nullptr;
    }

    auto* obj = new TensorRTEngineObj();
    obj->engine = std::move(engine);
    obj->inputName = inName;
    obj->outputName = outName;
    obj->allOutputNames = std::move(allOutputNames);
    obj->weightName = weightName;
    obj->hasWeight = hasWeight;
    for (int i = 0; i < 4 && i < inDims.nbDims; i++)  obj->inputDims[i] = inDims.d[i];
    for (int i = 0; i < 4 && i < outDims.nbDims; i++) obj->outputDims[i] = outDims.d[i];

    return obj;
}

void TensorRT_UnloadEngine(TensorRTEngineHandle handle) {
    if (!handle) return;
    delete static_cast<TensorRTEngineObj*>(handle);
}

int TensorRT_GetInputDims(TensorRTEngineHandle engine, int* dims) {
    if (!engine || !dims) return -1;
    auto* obj = static_cast<TensorRTEngineObj*>(engine);
    memcpy(dims, obj->inputDims, sizeof(int) * 4);
    return 0;
}

int TensorRT_GetOutputDims(TensorRTEngineHandle engine, int* dims) {
    if (!engine || !dims) return -1;
    auto* obj = static_cast<TensorRTEngineObj*>(engine);
    memcpy(dims, obj->outputDims, sizeof(int) * 4);
    return 0;
}

uint64_t TensorRT_GetDeviceMemorySize(TensorRTEngineHandle engine) {
    if (!engine) return 0;
    auto* obj = static_cast<TensorRTEngineObj*>(engine);
    return static_cast<uint64_t>(obj->engine->getDeviceMemorySizeV2());
}

// ── Context management (now creates dedicated IExecutionContext per call) ───

TensorRTContextHandle TensorRT_CreateContext(TensorRTEngineHandle engine) {
    if (!engine) return nullptr;
    auto* eng = static_cast<TensorRTEngineObj*>(engine);

    auto execCtx = TrtUniquePtr<IExecutionContext>(
        eng->engine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
    if (!execCtx) {
        setError("Failed to create execution context");
        return nullptr;
    }

    auto* ctx = new TensorRTContextObj();
    ctx->context = std::move(execCtx);
    ctx->inputName = eng->inputName;
    ctx->outputName = eng->outputName;
    ctx->allOutputNames = eng->allOutputNames;
    ctx->weightName = eng->weightName;
    ctx->hasWeight = eng->hasWeight;
    memcpy(ctx->inputDims, eng->inputDims, sizeof(int) * 4);
    memcpy(ctx->outputDims, eng->outputDims, sizeof(int) * 4);

    // Pre-allocate GPU buffer slots for extra outputs (will be lazily allocated)
    ctx->allOutputBuffers.resize(eng->allOutputNames.size(), nullptr);

    cudaError_t err = cudaStreamCreate(&ctx->stream);
    if (err != cudaSuccess) {
        setError("Failed to create CUDA stream");
        delete ctx;
        return nullptr;
    }

    return ctx;
}

void TensorRT_DestroyContext(TensorRTContextHandle context) {
    if (!context) return;
    auto* ctx = static_cast<TensorRTContextObj*>(context);

    if (ctx->stream) {
        cudaStreamSynchronize(ctx->stream);
    }
    dropGraph(ctx);
    if (ctx->stream) {
        cudaStreamDestroy(ctx->stream);
    }
    ctx->context.reset();
    if (ctx->activationMemory) cudaFree(ctx->activationMemory);
    if (ctx->gpuInputBuffer)  cudaFree(ctx->gpuInputBuffer);
    if (ctx->gpuOutputBuffer) cudaFree(ctx->gpuOutputBuffer);
    if (ctx->gpuWeightBuffer) cudaFree(ctx->gpuWeightBuffer);
    if (ctx->pinnedInput)     cudaFreeHost(ctx->pinnedInput);
    if (ctx->pinnedOutput)    cudaFreeHost(ctx->pinnedOutput);
    for (auto* buf : ctx->allOutputBuffers) {
        if (buf && buf != ctx->gpuOutputBuffer) cudaFree(buf);
    }
    if (ctx->ioCache) {
        for (auto& entry : ctx->ioCache->buffers) {
            if (entry.second.gpuBuf) cudaFree(entry.second.gpuBuf);
            if (entry.second.pinnedBuf) cudaFreeHost(entry.second.pinnedBuf);
        }
        delete ctx->ioCache;
        ctx->ioCache = nullptr;
    }

    delete ctx;
}

void TensorRT_SetWeight(TensorRTContextHandle context, float weight) {
    if (!context) return;
    auto* ctx = static_cast<TensorRTContextObj*>(context);
    ctx->cachedWeight = weight;
}

// ── CodeFormer-specific inference (cached addresses, no extra outputs) ──────
//
// CodeFormer is a multi-input model (image "x" + weight "w") with fixed 512×512
// shape.  The generic TensorRT_RunInference unconditional-rebind approach can
// cause green-stripe artifacts because TensorRT 10.x enqueueV3 requires all
// tensor addresses to be bound exactly once before enqueue — re-binding the
// weight tensor or extra intermediate outputs may invalidate the primary output
// buffer binding.
//
// This function mirrors the old (working) cached approach:
//   - setInputShape  only when dims actually change
//   - setTensorAddress  only when buffer pointers change
//   - weight address  set only once per context
//   - NO extra output tensor binding
//
int TensorRT_RunCodeFormerInference(TensorRTContextHandle context,
                                     const float* input, size_t inputSize,
                                     float* output, size_t outputSize,
                                     float weight) {
    if (!context || !input || !output) {
        setError("Invalid parameters for CodeFormer inference");
        return -1;
    }

    auto* ctx = static_cast<TensorRTContextObj*>(context);

    // ── 1. Ensure GPU buffers ──────────────────────────────────────────
    void* previousOutputBuffer = ctx->gpuOutputBuffer;
    if (!ensureGpuBuffer(ctx->gpuInputBuffer, ctx->gpuInputCapacity, inputSize)) {
        setError("Failed to allocate GPU input buffer");
        return -1;
    }
    if (!ensureGpuBuffer(ctx->gpuOutputBuffer, ctx->gpuOutputCapacity, outputSize)) {
        setError("Failed to allocate GPU output buffer");
        return -1;
    }
    if (previousOutputBuffer && previousOutputBuffer != ctx->gpuOutputBuffer) {
        for (auto& buf : ctx->allOutputBuffers) {
            if (buf == previousOutputBuffer) {
                buf = nullptr;
            }
        }
        ctx->extraOutputsBound = false;
        ctx->extraOutputsPrimaryBuffer = nullptr;
    }

    // ── 2. Ensure pinned host buffers ──────────────────────────────────
    if (!ensurePinnedBuffer(ctx->pinnedInput, ctx->pinnedInputSize, inputSize)) {
        setError("Failed to allocate pinned input buffer");
        return -1;
    }
    if (!ensurePinnedBuffer(ctx->pinnedOutput, ctx->pinnedOutputSize, outputSize)) {
        setError("Failed to allocate pinned output buffer");
        return -1;
    }

    // ── 3. Copy input to pinned, then async H2D ────────────────────────
    memcpy(ctx->pinnedInput, input, inputSize);
    cudaError_t status = cudaMemcpyAsync(ctx->gpuInputBuffer, ctx->pinnedInput, inputSize,
                                         cudaMemcpyHostToDevice, ctx->stream);
    if (status != cudaSuccess) {
        setError("Failed to copy input to GPU");
        return -1;
    }

    // ── 4. Set input shape (CACHED — only when dims change) ────────────
    // CodeFormer engine is compiled with fixed 512×512 shape.
    {
        int newDims[4] = {1, 3, 512, 512};
        bool shapeChanged = false;
        for (int i = 0; i < 4; i++) {
            if (ctx->cachedInputDims[i] != newDims[i]) {
                shapeChanged = true;
                break;
            }
        }
        if (shapeChanged) {
            Dims inputShape;
            inputShape.nbDims = 4;
            inputShape.d[0] = 1;
            inputShape.d[1] = 3;
            inputShape.d[2] = 512;
            inputShape.d[3] = 512;
            TRT_DIAG("[TRT-diag] CodeFormer: setInputShape('%s', [1,3,512,512]) [shape changed]\n",
                    ctx->inputName.c_str());
            try {
                if (!setContextInputShape(ctx, ctx->inputName.c_str(), inputShape)) return -1;
            } catch (...) {
                setError("setInputShape failed for CodeFormer");
                return -1;
            }
            for (int i = 0; i < 4; i++) ctx->cachedInputDims[i] = newDims[i];
        }
    }

    // ── 5. Set tensor addresses (CACHED — only when ptrs change) ───────
    if (ctx->cachedInputAddr != ctx->gpuInputBuffer) {
        TRT_DIAG("[TRT-diag] CodeFormer: setTensorAddress('%s', %p) [input changed]\n",
                ctx->inputName.c_str(), ctx->gpuInputBuffer);
        try {
            ctx->context->setTensorAddress(ctx->inputName.c_str(), ctx->gpuInputBuffer);
            ctx->cachedInputAddr = ctx->gpuInputBuffer;
        } catch (...) {
            setError("setTensorAddress (input) failed for CodeFormer");
            return -1;
        }
    }
    if (ctx->cachedOutputAddr != ctx->gpuOutputBuffer) {
        TRT_DIAG("[TRT-diag] CodeFormer: setTensorAddress('%s', %p) [output changed]\n",
                ctx->outputName.c_str(), ctx->gpuOutputBuffer);
        try {
            ctx->context->setTensorAddress(ctx->outputName.c_str(), ctx->gpuOutputBuffer);
            ctx->cachedOutputAddr = ctx->gpuOutputBuffer;
        } catch (...) {
            setError("setTensorAddress (output) failed for CodeFormer");
            return -1;
        }
    }

    bindAllRuntimeOutputs(ctx);

    // ── 5.1 Set weight parameter (CACHED — address set only once) ──────
    if (ctx->hasWeight) {
        ctx->cachedWeight = weight;
        if (!ctx->gpuWeightBuffer) {
            cudaError_t err = cudaMalloc(&ctx->gpuWeightBuffer, sizeof(float));
            if (err != cudaSuccess) {
                setError("Failed to allocate GPU weight buffer for CodeFormer");
                return -1;
            }
        }
        cudaMemcpyAsync(ctx->gpuWeightBuffer, &ctx->cachedWeight, sizeof(float),
                        cudaMemcpyHostToDevice, ctx->stream);
        if (!ctx->weightAddressSet) {
            TRT_DIAG("[TRT-diag] CodeFormer: setTensorAddress('%s', %p) [weight — first time]\n",
                    ctx->weightName.c_str(), ctx->gpuWeightBuffer);
            try {
                ctx->context->setTensorAddress(ctx->weightName.c_str(), ctx->gpuWeightBuffer);
                ctx->weightAddressSet = true;
            } catch (...) {
                // Non-fatal: some CodeFormer engines may not expose weight as a
                // separate named tensor (weight may be folded into the graph).
                TRT_DIAG("[TRT-diag] CodeFormer: setTensorAddress for weight failed (may be folded)\n");
            }
        }
    }

    // ── 6. Enqueue (lock scope: enqueueV3 only) ────────────────────────
    {
        std::lock_guard<std::mutex> lock(ctx->mu);
        if (!ensureActivationMemory(ctx)) return -1;
        // Legacy single-IO paths manage their own shapes and bindings; any graph
        // captured by the multi-IO path is no longer valid after them.
        ctx->pendingChange = true;
        bool success = ctx->context->enqueueV3(ctx->stream);
        if (!success) {
            setError("CodeFormer inference enqueue failed");
            return -1;
        }
    }

    // ── 7. Copy output async, then sync ────────────────────────────────
    status = cudaMemcpyAsync(ctx->pinnedOutput, ctx->gpuOutputBuffer, outputSize,
                              cudaMemcpyDeviceToHost, ctx->stream);
    if (status != cudaSuccess) {
        setError("Failed to copy output from GPU");
        return -1;
    }

    status = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess) {
        setError("CUDA stream synchronization failed");
        return -1;
    }

    // ── 8. Copy pinned → caller output ─────────────────────────────────
    memcpy(output, ctx->pinnedOutput, outputSize);

    return 0;
}

// ── Inference (sync wrapper — preserved for backward compatibility) ─────────

int TensorRT_RunInference(TensorRTContextHandle context,
                          int width, int height,
                          const float* input, size_t inputSize,
                          float* output, size_t outputSize) {
    if (!context || !input || !output) {
        setError("Invalid parameters for inference");
        return -1;
    }

    auto* ctx = static_cast<TensorRTContextObj*>(context);
    const auto timingStart = std::chrono::steady_clock::now();

    // ── 1. Ensure GPU buffers (max-capacity strategy) ───────────────────
    if (!ensureGpuBuffer(ctx->gpuInputBuffer, ctx->gpuInputCapacity, inputSize)) {
        setError("Failed to allocate GPU input buffer");
        return -1;
    }
    if (!ensureGpuBuffer(ctx->gpuOutputBuffer, ctx->gpuOutputCapacity, outputSize)) {
        setError("Failed to allocate GPU output buffer");
        return -1;
    }

    // ── 2. Select host buffers ──────────────────────────────────────────
    // Worker-owned external buffers avoid the pageable↔pinned copies in the
    // hot path. Legacy callers continue to use internal staging buffers.
    float* hostInput = ctx->externalPinnedInput;
    float* hostOutput = ctx->externalPinnedOutput;
    if (hostInput && ctx->externalPinnedInputSize < inputSize) {
        setError("External pinned input buffer is too small");
        return -1;
    }
    if (hostOutput && ctx->externalPinnedOutputSize < outputSize) {
        setError("External pinned output buffer is too small");
        return -1;
    }
    if (!hostInput && !ensurePinnedBuffer(ctx->pinnedInput, ctx->pinnedInputSize, inputSize)) {
        setError("Failed to allocate pinned input buffer");
        return -1;
    }
    if (!hostOutput && !ensurePinnedBuffer(ctx->pinnedOutput, ctx->pinnedOutputSize, outputSize)) {
        setError("Failed to allocate pinned output buffer");
        return -1;
    }
    if (!hostInput) hostInput = ctx->pinnedInput;
    if (!hostOutput) hostOutput = ctx->pinnedOutput;

    // ── 3. Copy input to pinned memory, then async to GPU ───────────────
    if (hostInput != input) {
        memcpy(hostInput, input, inputSize);
    }
    cudaError_t status = cudaMemcpyAsync(ctx->gpuInputBuffer, hostInput, inputSize,
                                         cudaMemcpyHostToDevice, ctx->stream);
    if (status != cudaSuccess) {
        setError("Failed to copy input to GPU");
        return -1;
    }
    const auto h2dSubmitted = std::chrono::steady_clock::now();

    // ── 4. Set input shape (cache it for repeated 1b1t calls) ────────────
    // Derive batch/channels from engine's input spec; only spatial dims from args.
    Dims inputShape;
    inputShape.nbDims = 4;
    inputShape.d[0] = (ctx->inputDims[0] > 0) ? ctx->inputDims[0] : 1;
    inputShape.d[1] = (ctx->inputDims[1] > 0) ? ctx->inputDims[1] : 3;
    inputShape.d[2] = height;
    inputShape.d[3] = width;
    const auto shapeStarted = std::chrono::steady_clock::now();
    bool shapeChanged = false;
    for (int i = 0; i < 4; i++) {
        if (ctx->cachedInputDims[i] != inputShape.d[i]) {
            shapeChanged = true;
            break;
        }
    }
    if (shapeChanged) {
        TRT_DIAG("[TRT-diag] RunInference: setInputShape('%s', [%lld,%lld,%lld,%lld])\n",
                ctx->inputName.c_str(),
                (long long)inputShape.d[0], (long long)inputShape.d[1],
                (long long)inputShape.d[2], (long long)inputShape.d[3]);
        try {
            if (!setContextInputShape(ctx, ctx->inputName.c_str(), inputShape)) return -1;
        } catch (...) {
            setError("setInputShape failed");
            return -1;
        }
        for (int i = 0; i < 4; i++) {
            ctx->cachedInputDims[i] = inputShape.d[i];
        }
    }

    // ── 5. Set tensor addresses only when buffers change ─────────────────
    if (ctx->cachedInputAddr != ctx->gpuInputBuffer) {
        TRT_DIAG("[TRT-diag] RunInference: setTensorAddress('%s', %p)\n",
                ctx->inputName.c_str(), ctx->gpuInputBuffer);
        try {
            ctx->context->setTensorAddress(ctx->inputName.c_str(), ctx->gpuInputBuffer);
        } catch (...) {
            setError("setTensorAddress (input) failed");
            return -1;
        }
        ctx->cachedInputAddr = ctx->gpuInputBuffer;
    }
    if (ctx->cachedOutputAddr != ctx->gpuOutputBuffer) {
        TRT_DIAG("[TRT-diag] RunInference: setTensorAddress('%s', %p)\n",
                ctx->outputName.c_str(), ctx->gpuOutputBuffer);
        try {
            ctx->context->setTensorAddress(ctx->outputName.c_str(), ctx->gpuOutputBuffer);
        } catch (...) {
            setError("setTensorAddress (output) failed");
            return -1;
        }
        ctx->cachedOutputAddr = ctx->gpuOutputBuffer;
    }

    const auto bindingStarted = std::chrono::steady_clock::now();
    // ── 5.0b Bind extra output tensors once per context ───────────────────
    // TensorRT 10.x enqueueV3 REQUIRES every I/O tensor to have an address set
    // or an output allocator registered.  Intermediate tensors exposed by the
    // engine are never read by us — they only need a valid address to satisfy
    // the enqueueV3 pre-condition.  If we cannot allocate a dedicated buffer for
    // an intermediate, we point it at the primary output buffer so the enqueue
    // succeeds (intermediate data lands there harmlessly and is ignored since we
    // only read from gpuOutputBuffer for the actual output tensor).
    const bool outputsNeedRebind = !ctx->extraOutputsBound || ctx->extraOutputsPrimaryBuffer != ctx->gpuOutputBuffer;
    if (outputsNeedRebind) {
        const nvinfer1::ICudaEngine& engine = ctx->context->getEngine();
        for (size_t i = 0; i < ctx->allOutputNames.size(); i++) {
            const auto& name = ctx->allOutputNames[i];
            if (name == ctx->outputName) continue;  // already bound above

            // Get shape for this extra output tensor
            size_t byteSize = ctx->gpuOutputCapacity;  // default: same as primary output
            Dims shape;
            try {
                shape = engine.getTensorShape(name.c_str());
                size_t elemCount = 1;
                for (int d = 0; d < shape.nbDims && d < 8; d++) {
                    if (shape.d[d] > 0) elemCount *= (size_t)shape.d[d];
                }
                if (elemCount > 1) byteSize = elemCount * sizeof(float);
            } catch (...) {}

            if (byteSize == 0) byteSize = sizeof(float);  // minimum allocation

            // Lazily allocate / grow GPU buffer for this extra output
            void*& buf = ctx->allOutputBuffers[i];
            bool usePrimaryBuffer = false;
            if (buf) {
                // Reuse existing buffer — assume it's large enough (happens in practice)
            } else {
                cudaError_t err = cudaMalloc(&buf, byteSize);
                if (err != cudaSuccess) {
                    TRT_DIAG("[TRT-diag] WARNING: cudaMalloc(%zu) failed for extra output '%s', "
                            "falling back to primary output buffer\n", byteSize, name.c_str());
                    // Fall back to primary output buffer — we need SOME address for
                    // enqueueV3 to succeed, even if it's shared with the real output.
                    buf = ctx->gpuOutputBuffer;
                    usePrimaryBuffer = true;
                } else {
                    TRT_DIAG("[TRT-diag] Allocated %zu bytes for extra output '%s'\n", byteSize, name.c_str());
                }
            }

            TRT_DIAG("[TRT-diag] RunInference: setTensorAddress('%s', %p) [extra output %zu%s]\n",
                    name.c_str(), buf, i, usePrimaryBuffer ? ", fallback" : "");
            try {
                ctx->context->setTensorAddress(name.c_str(), buf);
            } catch (...) {
                TRT_DIAG("[TRT-diag] WARNING: setTensorAddress failed for extra output '%s'\n", name.c_str());
            }
        }
        ctx->extraOutputsBound = true;
        ctx->extraOutputsPrimaryBuffer = ctx->gpuOutputBuffer;
    }
    if (outputsNeedRebind) bindAllRuntimeOutputs(ctx);
    const auto bindingCompleted = std::chrono::steady_clock::now();

    // ── 5.1 Set weight parameter if needed ──────────────────────────────────
    if (ctx->hasWeight) {
        if (!ctx->gpuWeightBuffer) {
            cudaMalloc(&ctx->gpuWeightBuffer, sizeof(float));
        }
        cudaMemcpyAsync(ctx->gpuWeightBuffer, &ctx->cachedWeight, sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
        try {
            ctx->context->setTensorAddress(ctx->weightName.c_str(), ctx->gpuWeightBuffer);
        } catch (...) {}
    }

    // ── 6. Enqueue (lock scope: enqueueV3 only) ────────────────────────
    const auto enqueueStarted = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(ctx->mu);
        if (!ensureActivationMemory(ctx)) return -1;
        // Legacy single-IO paths manage their own shapes and bindings; any graph
        // captured by the multi-IO path is no longer valid after them.
        ctx->pendingChange = true;
        bool success = ctx->context->enqueueV3(ctx->stream);
        if (!success) {
            setError("Inference execution failed");
            return -1;
        }
    }
    const auto enqueueSubmitted = std::chrono::steady_clock::now();

    // ── 7. Copy output async, then sync ────────────────────────────────
    status = cudaMemcpyAsync(hostOutput, ctx->gpuOutputBuffer, outputSize,
                             cudaMemcpyDeviceToHost, ctx->stream);
    if (status != cudaSuccess) {
        setError("Failed to copy output from GPU");
        return -1;
    }
    const auto d2hSubmitted = std::chrono::steady_clock::now();

    status = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess) {
        setError("CUDA stream synchronization failed");
        return -1;
    }
    const auto syncCompleted = std::chrono::steady_clock::now();
    if (gDiagnosticsEnabled) {
        const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
        TRT_DIAG("[TRT-timing] shape_and_addresses=%.3fms extra_binding=%.3fms enqueue_call=%.3fms h2d_submit=%.3fms enqueue_submit=%.3fms d2h_submit=%.3fms sync_wait=%.3fms total=%.3fms\n",
                 ms(bindingStarted - shapeStarted), ms(bindingCompleted - bindingStarted),
                 ms(enqueueSubmitted - enqueueStarted),
                 ms(h2dSubmitted - timingStart), ms(enqueueSubmitted - h2dSubmitted),
                 ms(d2hSubmitted - enqueueSubmitted), ms(syncCompleted - d2hSubmitted),
                 ms(syncCompleted - timingStart));
    }

    // ── 8. Copy pinned → caller output ─────────────────────────────────
    if (output != hostOutput) {
        memcpy(output, hostOutput, outputSize);
    }

    return 0;
}

// ── Batched synchronous inference (N frames in a single GPU call) ───────────
//
// Like TensorRT_RunInference but accepts a batch dimension N > 1, enabling
// multi-frame inference in a single kernel launch. The engine MUST have been
// built with a dynamic batch dimension (--minShapes/--optShapes/--maxShapes
// with the first dim ranging from 1 up to the desired batch size).
//
//   input:  float32 NCHW tensor [batch, 3, height, width]
//   output: float32 NCHW tensor [batch, 3, outHeight, outWidth] (caller-allocated)
//
// Returns 0 on success, -1 on failure.
void TensorRT_SetHostBuffers(TensorRTContextHandle context,
                             float* input, size_t inputSize,
                             float* output, size_t outputSize) {
    if (!context) return;
    auto* ctx = static_cast<TensorRTContextObj*>(context);
    ctx->externalPinnedInput = input;
    ctx->externalPinnedInputSize = inputSize;
    ctx->externalPinnedOutput = output;
    ctx->externalPinnedOutputSize = outputSize;
}

TENSORRT_API int TensorRT_RunInferenceBatch(TensorRTContextHandle context,
                                            int batch, int width, int height,
                                            const float* input, size_t inputSize,
                                            float* output, size_t outputSize) {
    if (!context || !input || !output || batch < 1) {
        setError("Invalid parameters for batched inference");
        return -1;
    }

    auto* ctx = static_cast<TensorRTContextObj*>(context);

    // 1. Ensure GPU buffers
    if (!ensureGpuBuffer(ctx->gpuInputBuffer, ctx->gpuInputCapacity, inputSize)) {
        setError("Failed to allocate GPU input buffer");
        return -1;
    }
    if (!ensureGpuBuffer(ctx->gpuOutputBuffer, ctx->gpuOutputCapacity, outputSize)) {
        setError("Failed to allocate GPU output buffer");
        return -1;
    }

    // 2. Use worker-owned pinned buffers when provided by Go. This keeps the
    // batch path consistent with the single-frame path and avoids allocating
    // a second staging buffer for every context.
    float* hostInput = ctx->externalPinnedInput;
    float* hostOutput = ctx->externalPinnedOutput;
    if (hostInput && ctx->externalPinnedInputSize < inputSize) {
        setError("External pinned input buffer is too small");
        return -1;
    }
    if (hostOutput && ctx->externalPinnedOutputSize < outputSize) {
        setError("External pinned output buffer is too small");
        return -1;
    }
    if (!hostInput && !ensurePinnedBuffer(ctx->pinnedInput, ctx->pinnedInputSize, inputSize)) {
        setError("Failed to allocate pinned input buffer");
        return -1;
    }
    if (!hostOutput && !ensurePinnedBuffer(ctx->pinnedOutput, ctx->pinnedOutputSize, outputSize)) {
        setError("Failed to allocate pinned output buffer");
        return -1;
    }
    if (!hostInput) hostInput = ctx->pinnedInput;
    if (!hostOutput) hostOutput = ctx->pinnedOutput;

    // 3. Copy input to pinned memory when the caller did not provide it,
    // then async to GPU.
    if (input != hostInput) memcpy(hostInput, input, inputSize);
    cudaError_t status = cudaMemcpyAsync(ctx->gpuInputBuffer, hostInput, inputSize,
                                         cudaMemcpyHostToDevice, ctx->stream);
    if (status != cudaSuccess) {
        setError("Failed to copy batch input to GPU");
        return -1;
    }

    // 4. Set input shape only when the batch or spatial dimensions change.
    // Video workers normally reuse one shape for the entire stream.
    Dims inputShape;
    inputShape.nbDims = 4;
    inputShape.d[0] = batch;
    inputShape.d[1] = (ctx->inputDims[1] > 0) ? ctx->inputDims[1] : 3;
    inputShape.d[2] = height;
    inputShape.d[3] = width;
    bool shapeChanged = false;
    for (int i = 0; i < 4; i++) {
        if (ctx->cachedInputDims[i] != inputShape.d[i]) {
            shapeChanged = true;
            break;
        }
    }
    if (shapeChanged) {
        TRT_DIAG("[TRT-diag] RunInferenceBatch: setInputShape batch=%lld\n", (long long)batch);
        try {
            if (!setContextInputShape(ctx, ctx->inputName.c_str(), inputShape)) return -1;
        } catch (...) {
            setError("setInputShape failed for batch inference");
            return -1;
        }
        for (int i = 0; i < 4; i++) ctx->cachedInputDims[i] = inputShape.d[i];
    }

    // 5. Bind the primary tensors only when their GPU addresses change.
    if (ctx->cachedInputAddr != ctx->gpuInputBuffer) {
        try { ctx->context->setTensorAddress(ctx->inputName.c_str(), ctx->gpuInputBuffer); }
        catch (...) { setError("setTensorAddress (input) failed"); return -1; }
        ctx->cachedInputAddr = ctx->gpuInputBuffer;
    }
    if (ctx->cachedOutputAddr != ctx->gpuOutputBuffer) {
        try { ctx->context->setTensorAddress(ctx->outputName.c_str(), ctx->gpuOutputBuffer); }
        catch (...) { setError("setTensorAddress (output) failed"); return -1; }
        ctx->cachedOutputAddr = ctx->gpuOutputBuffer;
    }

    // Bind extra output tensors (with fallback to primary buffer if allocation fails)
    for (size_t i = 0; i < ctx->allOutputNames.size(); i++) {
        const auto& name = ctx->allOutputNames[i];
        if (name == ctx->outputName) continue;
        void*& buf = ctx->allOutputBuffers[i];
        if (!buf) {
            cudaError_t err = cudaMalloc(&buf, ctx->gpuOutputCapacity);
            if (err != cudaSuccess) {
                // Fall back to primary output buffer — enqueueV3 requires an address
                buf = ctx->gpuOutputBuffer;
            }
        }
        try { ctx->context->setTensorAddress(name.c_str(), buf); } catch (...) {}
    }
    bindAllRuntimeOutputs(ctx);

    // Set weight if needed
    if (ctx->hasWeight) {
        if (!ctx->gpuWeightBuffer) cudaMalloc(&ctx->gpuWeightBuffer, sizeof(float));
        cudaMemcpyAsync(ctx->gpuWeightBuffer, &ctx->cachedWeight, sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
        try { ctx->context->setTensorAddress(ctx->weightName.c_str(), ctx->gpuWeightBuffer); } catch (...) {}
    }

    // 6. Enqueue
    {
        std::lock_guard<std::mutex> lock(ctx->mu);
        if (!ensureActivationMemory(ctx)) return -1;
        // Legacy single-IO paths manage their own shapes and bindings; any graph
        // captured by the multi-IO path is no longer valid after them.
        ctx->pendingChange = true;
        bool success = ctx->context->enqueueV3(ctx->stream);
        if (!success) {
            setError("Batch inference execution failed");
            return -1;
        }
    }

    // 7. Copy output async, then sync
    status = cudaMemcpyAsync(hostOutput, ctx->gpuOutputBuffer, outputSize,
                              cudaMemcpyDeviceToHost, ctx->stream);
    if (status != cudaSuccess) {
        setError("Failed to copy batch output from GPU");
        return -1;
    }
    status = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess) {
        setError("CUDA stream synchronization failed");
        return -1;
    }

    // 8. Copy pinned -> caller output only when the caller did not provide
    // the worker-owned output staging buffer directly.
    if (output != hostOutput) memcpy(output, hostOutput, outputSize);

    return 0;
}


// ── Async inference (returns immediately, caller syncs via event) ───────────

int TensorRT_RunInferenceAsync(TensorRTContextHandle context,
                               int width, int height,
                               const float* input, size_t inputSize,
                               float* output, size_t outputSize,
                               TensorRTEventHandle* outEvent) {
    if (!context || !input || !outEvent) {
        setError("Invalid parameters for async inference");
        return -1;
    }

    auto* ctx = static_cast<TensorRTContextObj*>(context);

    // ── 1. Ensure GPU buffers ──────────────────────────────────────────
    if (!ensureGpuBuffer(ctx->gpuInputBuffer, ctx->gpuInputCapacity, inputSize)) {
        setError("Failed to allocate GPU input buffer");
        return -1;
    }
    if (!ensureGpuBuffer(ctx->gpuOutputBuffer, ctx->gpuOutputCapacity, outputSize)) {
        setError("Failed to allocate GPU output buffer");
        return -1;
    }

    // ── 2. Ensure pinned host buffers ──────────────────────────────────
    if (!ensurePinnedBuffer(ctx->pinnedInput, ctx->pinnedInputSize, inputSize)) {
        setError("Failed to allocate pinned input buffer");
        return -1;
    }
    if (!ensurePinnedBuffer(ctx->pinnedOutput, ctx->pinnedOutputSize, outputSize)) {
        setError("Failed to allocate pinned output buffer");
        return -1;
    }

    // ── 3. Copy input to pinned, then async H2D ────────────────────────
    memcpy(ctx->pinnedInput, input, inputSize);
    cudaError_t status = cudaMemcpyAsync(ctx->gpuInputBuffer, ctx->pinnedInput, inputSize,
                                         cudaMemcpyHostToDevice, ctx->stream);
    if (status != cudaSuccess) {
        setError("Failed to copy input to GPU");
        return -1;
    }

    // ── 4. Set input shape (always, enqueueV3 requires explicit shape) ───
    // Derive batch/channels from engine's input spec; only spatial dims from args.
    Dims inputShape;
    inputShape.nbDims = 4;
    inputShape.d[0] = (ctx->inputDims[0] > 0) ? ctx->inputDims[0] : 1;
    inputShape.d[1] = (ctx->inputDims[1] > 0) ? ctx->inputDims[1] : 3;
    inputShape.d[2] = height;
    inputShape.d[3] = width;
    try {
        if (!setContextInputShape(ctx, ctx->inputName.c_str(), inputShape)) return -1;
    } catch (...) {
        setError("setInputShape failed");
        return -1;
    }
    for (int i = 0; i < 4; ++i) ctx->cachedInputDims[i] = inputShape.d[i];

    // ── 5. Set tensor addresses (always, enqueueV3 requires fresh bindings) ──
    try {
        ctx->context->setTensorAddress(ctx->inputName.c_str(), ctx->gpuInputBuffer);
    } catch (...) {
        setError("setTensorAddress (input) failed");
        return -1;
    }
    try {
        ctx->context->setTensorAddress(ctx->outputName.c_str(), ctx->gpuOutputBuffer);
    } catch (...) {
        setError("setTensorAddress (output) failed");
        return -1;
    }


    // Bind ALL extra output tensors (async, with fallback)
    {
        const nvinfer1::ICudaEngine& engine = ctx->context->getEngine();
        for (size_t i = 0; i < ctx->allOutputNames.size(); i++) {
            const auto& name = ctx->allOutputNames[i];
            if (name == ctx->outputName) continue;
            size_t byteSize = ctx->gpuOutputCapacity;
            Dims shape;
            try {
                shape = engine.getTensorShape(name.c_str());
                size_t elemCount = 1;
                for (int d = 0; d < shape.nbDims && d < 8; d++) {
                    if (shape.d[d] > 0) elemCount *= (size_t)shape.d[d];
                }
                if (elemCount > 1) byteSize = elemCount * sizeof(float);
            } catch (...) {}
            if (byteSize == 0) byteSize = sizeof(float);
            void*& buf = ctx->allOutputBuffers[i];
            if (!buf) {
                cudaError_t err = cudaMalloc(&buf, byteSize);
                if (err != cudaSuccess) {
                    // Fall back to primary output buffer so enqueueV3 doesn't fail
                    buf = ctx->gpuOutputBuffer;
                }
            }
            try {
                ctx->context->setTensorAddress(name.c_str(), buf);
            } catch (...) {}
        }
    }
    bindAllRuntimeOutputs(ctx);
    // ── 5.1 Set weight parameter if needed ──────────────────────────────────
    if (ctx->hasWeight) {
        if (!ctx->gpuWeightBuffer) {
            cudaMalloc(&ctx->gpuWeightBuffer, sizeof(float));
        }
        cudaMemcpyAsync(ctx->gpuWeightBuffer, &ctx->cachedWeight, sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
        try {
            ctx->context->setTensorAddress(ctx->weightName.c_str(), ctx->gpuWeightBuffer);
        } catch (...) {}
    }

    // ── 6. Enqueue (lock scope: enqueueV3 only) ────────────────────────
    {
        std::lock_guard<std::mutex> lock(ctx->mu);
        if (!ensureActivationMemory(ctx)) return -1;
        // Legacy single-IO paths manage their own shapes and bindings; any graph
        // captured by the multi-IO path is no longer valid after them.
        ctx->pendingChange = true;
        bool success = ctx->context->enqueueV3(ctx->stream);
        if (!success) {
            setError("Inference execution failed");
            return -1;
        }
    }

    // ── 7. Async D2H copy ──────────────────────────────────────────────
    status = cudaMemcpyAsync(ctx->pinnedOutput, ctx->gpuOutputBuffer, outputSize,
                              cudaMemcpyDeviceToHost, ctx->stream);
    if (status != cudaSuccess) {
        setError("Failed to copy output from GPU");
        return -1;
    }

    // ── 8. Record event (caller syncs + copies pinned→output) ──────────
    cudaEvent_t event = nullptr;
    status = cudaEventCreate(&event);
    if (status != cudaSuccess) {
        setError("Failed to create CUDA event");
        return -1;
    }
    status = cudaEventRecord(event, ctx->stream);
    if (status != cudaSuccess) {
        cudaEventDestroy(event);
        setError("Failed to record CUDA event");
        return -1;
    }

    *outEvent = static_cast<TensorRTEventHandle>(event);
    return 0;
}

// Slot entry points retain the public contract while sharing the context's
// serialized enqueue path. The slot index is validated here so Go callers can
// safely manage two in-flight submissions; the existing context-owned staging
// buffers remain the compatibility fallback until per-slot allocation is
// enabled for a given engine.
int TensorRT_SubmitInferenceSlot(TensorRTContextHandle context, int slot,
                                  int width, int height, const float* input,
                                  size_t inputSize, size_t outputSize,
                                  TensorRTEventHandle* outEvent) {
    if (slot < 0 || slot >= 2) {
        setError("TensorRT async slot must be 0 or 1");
        return -1;
    }
    return TensorRT_RunInferenceAsync(context, width, height, input, inputSize,
                                      nullptr, outputSize, outEvent);
}

int TensorRT_WaitInferenceSlot(TensorRTEventHandle event) {
    return TensorRT_EventSynchronize(event);
}

int TensorRT_CopyOutputSlot(TensorRTContextHandle context, int slot,
                            float* output, size_t outputSize) {
    if (slot < 0 || slot >= 2) {
        setError("TensorRT async slot must be 0 or 1");
        return -1;
    }
    return TensorRT_CopyOutputFromPinned(context, output, outputSize);
}

void TensorRT_ReleaseInferenceSlot(TensorRTContextHandle context, int slot,
                                    TensorRTEventHandle event) {
    (void)context;
    (void)slot;
    TensorRT_EventDestroy(event);
}

// ── Event helpers ───────────────────────────────────────────────────────────

int TensorRT_EventSynchronize(TensorRTEventHandle event) {
    if (!event) return -1;
    cudaError_t err = cudaEventSynchronize(static_cast<cudaEvent_t>(event));
    return (err == cudaSuccess) ? 0 : -1;
}

int TensorRT_EventQuery(TensorRTEventHandle event) {
    if (!event) return -1;
    cudaError_t err = cudaEventQuery(static_cast<cudaEvent_t>(event));
    if (err == cudaSuccess) return 1;       // completed
    if (err == cudaErrorNotReady) return 0; // still running
    return -1;                               // error
}

void TensorRT_EventDestroy(TensorRTEventHandle event) {
    if (event) cudaEventDestroy(static_cast<cudaEvent_t>(event));
}

// ── Copy pinned output to user buffer after async completion ────────────────

int TensorRT_CopyOutputFromPinned(TensorRTContextHandle context,
                                  float* output, size_t outputSize) {
    if (!context || !output) return -1;
    auto* ctx = static_cast<TensorRTContextObj*>(context);
    if (!ctx->pinnedOutput || ctx->pinnedOutputSize < outputSize) {
        setError("Pinned output buffer not ready or too small");
        return -1;
    }
    memcpy(output, ctx->pinnedOutput, outputSize);
    return 0;
}

const char* TensorRT_GetLastError() {
    return gLastError.c_str();
}

// ── Multi-IO inference ──────────────────────────────────────────────────────

// Get or create IO cache for a context.
static TensorIOCache* getIOCache(TensorRTContextObj* ctx) {
    // We store the cache pointer as a hidden field. For simplicity, we use a
    // static map from context pointer → cache. This avoids modifying the
    // TensorRTContextObj struct (keeping backward compat).
    if (!ctx->ioCache) {
        ctx->ioCache = new TensorIOCache();
    }
    return ctx->ioCache;
}

// Ensure GPU + pinned buffers for a named tensor. Returns TensorBuffer* on success, NULL on failure.
static TensorBuffer* ensureTensorBuffer(TensorRTContextObj* ctx, const std::string& name, size_t requiredBytes) {
    auto* cache = getIOCache(ctx);
    auto& buf = cache->buffers[name];

    // Grow GPU buffer if needed
    if (requiredBytes > buf.gpuCap) {
        size_t newCap = std::max(buf.gpuCap * 2, requiredBytes);
        newCap = (newCap + 255) & ~size_t(255);
        if (buf.gpuBuf) cudaFree(buf.gpuBuf);
        cudaError_t err = cudaMalloc(&buf.gpuBuf, newCap);
        if (err != cudaSuccess) {
            buf.gpuBuf = nullptr;
            buf.gpuCap = 0;
            return nullptr;
        }
        buf.gpuCap = newCap;
        // The tensor moved; a captured graph still points at the old address.
        ctx->pendingChange = true;
    }

    // Grow pinned host buffer if needed
    if (requiredBytes > buf.pinnedCap) {
        size_t newCap = std::max(buf.pinnedCap * 2, requiredBytes);
        newCap = (newCap + 255) & ~size_t(255);
        if (buf.pinnedBuf) cudaFreeHost(buf.pinnedBuf);
        buf.pinnedBuf = allocPinned(newCap);
        if (!buf.pinnedBuf) {
            buf.pinnedCap = 0;
            return nullptr;
        }
        buf.pinnedCap = newCap;
    }

    return &buf;
}

// Compute total bytes for a shape, given the element size. The original kernel
// multiplied by sizeof(float) unconditionally, which silently mis-sized int64
// and bool tensors (reading 4 bytes of an 8-byte element).
static size_t shapeBytesWithElem(const int64_t* shape, int nbDims, size_t elemSize) {
    if (!shape || nbDims <= 0 || nbDims > 8) return 0;
    size_t total = 1;
    for (int i = 0; i < nbDims; i++) {
        if (shape[i] <= 0) return 0;
        total *= (size_t)shape[i];
    }
    return total * elemSize;
}

// Backward-compatible float32 form, kept for the image paths that still call it.
static size_t shapeBytes(const int64_t* shape, int nbDims) {
    return shapeBytesWithElem(shape, nbDims, sizeof(float));
}

// Element size for a tensor, preferring the engine's own type for that name.
static size_t elemSizeForTensor(TensorRTContextObj* ctx, const char* name, int declaredDtype) {
    if (ctx && ctx->context && name) {
        try {
            int engineType = toTensorIOType(ctx->context->getEngine().getTensorDataType(name));
            if (engineType >= 0) return dtypeElemSize(engineType);
        } catch (...) {
            // Name is not in the engine — fall through to the declared type.
        }
    }
    return dtypeElemSize(declaredDtype);
}

// ── Native run builder ──────────────────────────────────────────────────────
//
// Owns the tensor descriptors, the staged input copies, and the output buffers
// for one inference. Callers in Go/Rust/C# describe a run through the exported
// functions and never lay out a TensorIOTensor themselves: no foreign struct
// layout, no foreign allocation, no lifetime question at the boundary.
//
// Input data is copied on add, so the caller's buffer can be reused or freed
// immediately. Outputs are allocated here and copied out on demand.

struct RunTensor {
    std::string name;
    int dtype = TENSOR_IO_FLOAT32;
    bool isInput = false;
    std::vector<int64_t> shape;     // inputs: as given; outputs: as resolved
    std::vector<uint8_t> host;      // inputs: staged copy; outputs: result
    int nbDims = 0;
};

struct TensorRunObj {
    TensorRTContextObj* ctx = nullptr;
    std::vector<RunTensor> inputs;
    std::vector<RunTensor> outputs;
    bool resolved = false;
};

static RunTensor* findRunTensor(std::vector<RunTensor>& v, const std::string& name) {
    for (auto& t : v) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

// Build a TensorIOTensor descriptor for a run tensor. The returned struct holds
// pointers into the run tensor, so it is valid only while that tensor lives.
static TensorIOTensor makeDescriptor(RunTensor& t) {
    TensorIOTensor d;
    memset(&d, 0, sizeof(d));
    d.name = t.name.c_str();
    d.dtype = t.dtype;
    d.data = t.host.empty() ? nullptr : t.host.data();
    d.nbDims = t.nbDims;
    for (int i = 0; i < t.nbDims && i < 8; i++) {
        d.shape[i] = t.shape[static_cast<size_t>(i)];
    }
    return d;
}

TensorRunHandle TensorRT_RunCreate(TensorRTContextHandle context) {
    if (!context) {
        setError("TensorRT_RunCreate requires a context");
        return nullptr;
    }
    auto* run = new (std::nothrow) TensorRunObj();
    if (!run) {
        setError("failed to allocate run");
        return nullptr;
    }
    run->ctx = static_cast<TensorRTContextObj*>(context);
    return run;
}

void TensorRT_RunDestroy(TensorRunHandle run) {
    delete static_cast<TensorRunObj*>(run);
}

int TensorRT_RunAddInput(TensorRunHandle handle, const char* name,
    int dtype, const void* data, const int64_t* shape, int nbDims) {
    if (!handle || !name || !data || !shape) {
        setError("TensorRT_RunAddInput: invalid arguments");
        return -1;
    }
    if (nbDims <= 0 || nbDims > 8) {
        setError("TensorRT_RunAddInput: nbDims must be 1..8");
        return -1;
    }
    auto* run = static_cast<TensorRunObj*>(handle);
    size_t elemSize = elemSizeForTensor(run->ctx, name, dtype);
    size_t bytes = shapeBytesWithElem(shape, nbDims, elemSize);
    if (bytes == 0) {
        setError((std::string("TensorRT_RunAddInput: invalid shape for '") + name + "'").c_str());
        return -1;
    }
    if (findRunTensor(run->inputs, name)) {
        setError((std::string("TensorRT_RunAddInput: duplicate input '") + name + "'").c_str());
        return -1;
    }

    RunTensor t;
    t.name = name;
    t.isInput = true;
    t.nbDims = nbDims;
    t.shape.assign(shape, shape + nbDims);
    try {
        t.dtype = toTensorIOType(run->ctx->context->getEngine().getTensorDataType(name));
    } catch (...) {
        t.dtype = dtype;
    }
    if (t.dtype < 0) t.dtype = dtype;
    t.host.resize(bytes);
    memcpy(t.host.data(), data, bytes);
    run->inputs.push_back(std::move(t));
    run->resolved = false;
    return 0;
}

int TensorRT_RunAddOutput(TensorRunHandle handle, const char* name) {
    if (!handle || !name) {
        setError("TensorRT_RunAddOutput: invalid arguments");
        return -1;
    }
    auto* run = static_cast<TensorRunObj*>(handle);
    if (findRunTensor(run->outputs, name)) {
        setError((std::string("TensorRT_RunAddOutput: duplicate output '") + name + "'").c_str());
        return -1;
    }
    RunTensor t;
    t.name = name;
    t.isInput = false;
    t.dtype = -1;
    run->outputs.push_back(std::move(t));
    run->resolved = false;
    return 0;
}

int TensorRT_RunOutputCount(TensorRunHandle handle) {
    auto* run = static_cast<TensorRunObj*>(handle);
    return run ? static_cast<int>(run->outputs.size()) : -1;
}

int TensorRT_RunResolveOutputs(TensorRunHandle handle,
    char* names, int nameStride, int* dtypes, int64_t* shapes, int* nbDims,
    int count) {
    if (!handle) {
        setError("TensorRT_RunResolveOutputs: null run");
        return -1;
    }
    auto* run = static_cast<TensorRunObj*>(handle);
    if (run->inputs.empty()) {
        setError("TensorRT_RunResolveOutputs: add inputs first");
        return -1;
    }
    if (count < static_cast<int>(run->outputs.size())) {
        setError("TensorRT_RunResolveOutputs: caller arrays are too small");
        return -1;
    }

    std::lock_guard<std::mutex> lock(run->ctx->mu);

    // Push input shapes onto the context so output shapes become concrete. This
    // also moves the context to the profile that admits them, so the output
    // shapes are resolved against the profile the run will execute on.
    {
        std::vector<ShapeRequest> reqs(run->inputs.size());
        for (size_t k = 0; k < run->inputs.size(); k++) {
            auto& in = run->inputs[k];
            reqs[k].name = in.name.c_str();
            reqs[k].dims.nbDims = in.nbDims;
            for (int i = 0; i < in.nbDims; i++) reqs[k].dims.d[i] = in.shape[static_cast<size_t>(i)];
        }
        try {
            if (!applyInputShapes(run->ctx, reqs.data(), static_cast<int>(reqs.size()))) {
                return -1;
            }
        } catch (...) {
            setError("setInputShape failed while resolving outputs");
            return -1;
        }
    }
    if (!run->ctx->context->allInputDimensionsSpecified()) {
        setError("not all input dimensions are specified");
        return -1;
    }

    for (size_t i = 0; i < run->outputs.size(); i++) {
        auto& out = run->outputs[i];
        Dims shape;
        int dt = -1;
        try {
            shape = run->ctx->context->getTensorShape(out.name.c_str());
            dt = toTensorIOType(run->ctx->context->getEngine().getTensorDataType(out.name.c_str()));
        } catch (...) {
            setError(("engine has no output named '" + out.name + "'").c_str());
            return -1;
        }
        size_t bytes = shapeBytesWithElem(shape.d, shape.nbDims, dtypeElemSize(dt));
        if (bytes == 0) {
            setError(("output '" + out.name + "' has an unresolved shape").c_str());
            return -1;
        }
        out.dtype = dt;
        out.nbDims = shape.nbDims;
        out.shape.assign(shape.d, shape.d + shape.nbDims);
        out.host.assign(bytes, 0);

        if (names) {
            strncpy(names + i * static_cast<size_t>(nameStride), out.name.c_str(),
                    static_cast<size_t>(nameStride) - 1);
            names[i * static_cast<size_t>(nameStride) + nameStride - 1] = '\0';
        }
        if (dtypes) dtypes[i] = out.dtype;
        if (nbDims) nbDims[i] = out.nbDims;
        if (shapes) {
            for (int d = 0; d < out.nbDims && d < 8; d++) {
                shapes[i * 8 + static_cast<size_t>(d)] = out.shape[static_cast<size_t>(d)];
            }
        }
    }
    run->resolved = true;
    return 0;
}

int TensorRT_RunExecute(TensorRunHandle handle) {
    if (!handle) {
        setError("TensorRT_RunExecute: null run");
        return -1;
    }
    auto* run = static_cast<TensorRunObj*>(handle);
    if (!run->resolved) {
        setError("TensorRT_RunExecute: resolve outputs first");
        return -1;
    }

    std::vector<TensorIOTensor> inDesc(run->inputs.size());
    std::vector<TensorIOTensor> outDesc(run->outputs.size());
    for (size_t i = 0; i < run->inputs.size(); i++) inDesc[i] = makeDescriptor(run->inputs[i]);
    for (size_t i = 0; i < run->outputs.size(); i++) outDesc[i] = makeDescriptor(run->outputs[i]);

    return TensorRT_RunInferenceMultiIO(run->ctx,
        static_cast<int>(inDesc.size()), inDesc.data(),
        static_cast<int>(outDesc.size()), outDesc.data());
}

size_t TensorRT_RunOutputBytes(TensorRunHandle handle, int index) {
    auto* run = static_cast<TensorRunObj*>(handle);
    if (!run || index < 0 || static_cast<size_t>(index) >= run->outputs.size()) return 0;
    return run->outputs[static_cast<size_t>(index)].host.size();
}

int TensorRT_RunReadOutput(TensorRunHandle handle, int index, void* dst, size_t bytes) {
    auto* run = static_cast<TensorRunObj*>(handle);
    if (!run || !dst || index < 0 || static_cast<size_t>(index) >= run->outputs.size()) {
        setError("TensorRT_RunReadOutput: invalid arguments");
        return -1;
    }
    const auto& out = run->outputs[static_cast<size_t>(index)];
    if (bytes < out.host.size()) {
        setError("TensorRT_RunReadOutput: destination is smaller than the output");
        return -1;
    }
    if (!out.host.empty()) memcpy(dst, out.host.data(), out.host.size());
    return 0;
}

int TensorRT_RunInferenceMultiIO(TensorRTContextHandle context,
    int numInputs, const TensorIOTensor* inputs,
    int numOutputs, TensorIOTensor* outputs) {

    if (!context || !inputs || !outputs || numInputs <= 0 || numOutputs <= 0) {
        setError("Invalid parameters for multi-IO inference");
        return -1;
    }

    auto* ctx = static_cast<TensorRTContextObj*>(context);
    std::lock_guard<std::mutex> lock(ctx->mu);

    // ── Phase 0: pick the profile and apply every input shape at once ───────
    // Shapes are applied together because the profile choice depends on all of
    // them (batch and sequence length together decide which profile fits).
    std::vector<ShapeRequest> reqs(static_cast<size_t>(numInputs));
    for (int i = 0; i < numInputs; i++) {
        const auto& inp = inputs[i];
        if (!inp.name || !inp.data || inp.nbDims <= 0 || inp.nbDims > 8) {
            setError("Invalid input tensor descriptor");
            return -1;
        }
        reqs[i].name = inp.name;
        reqs[i].dims.nbDims = inp.nbDims;
        for (int d = 0; d < inp.nbDims; d++) reqs[i].dims.d[d] = inp.shape[d];
    }
    try {
        if (!applyInputShapes(ctx, reqs.data(), numInputs)) return -1;
    } catch (...) {
        setError("setInputShape failed");
        return -1;
    }
    const std::string graphKey = graphKeyFor(ctx, reqs.data(), numInputs);

    // ── Phase 1: allocate buffers, copy H2D ─────────────────────────────────
    for (int i = 0; i < numInputs; i++) {
        const auto& inp = inputs[i];

        // Size from the engine's element type for this tensor (falling back to
        // the caller's declared dtype), not from an assumed float32.
        size_t elemSize = elemSizeForTensor(ctx, inp.name, inp.dtype);
        size_t bytes = shapeBytesWithElem(inp.shape, inp.nbDims, elemSize);
        if (bytes == 0) {
            setError(("Invalid shape for input '" + std::string(inp.name) + "'").c_str());
            return -1;
        }

        // Ensure buffers
        auto* buf = ensureTensorBuffer(ctx, inp.name, bytes);
        if (!buf) {
            setError(("Failed to allocate GPU buffer for '" + std::string(inp.name) + "'").c_str());
            return -1;
        }

        // Copy input to pinned, then async H2D
        memcpy(buf->pinnedBuf, inp.data, bytes);
        cudaError_t status = cudaMemcpyAsync(buf->gpuBuf, buf->pinnedBuf, bytes,
                                             cudaMemcpyHostToDevice, ctx->stream);
        if (status != cudaSuccess) {
            setError(("H2D copy failed for '" + std::string(inp.name) + "'").c_str());
            return -1;
        }

        // Bind the address. A real change invalidates the captured graph; the
        // same address re-bound is not a change and keeps the graph usable.
        if (ctx->context->getTensorAddress(inp.name) != buf->gpuBuf) {
            try {
                if (!ctx->context->setTensorAddress(inp.name, buf->gpuBuf)) {
                    setError(("setTensorAddress failed for '" + std::string(inp.name) + "'").c_str());
                    return -1;
                }
            } catch (...) {
                setError(("setTensorAddress failed for '" + std::string(inp.name) + "'").c_str());
                return -1;
            }
            ctx->pendingChange = true;
        }
        if (ctx->context->getEngine().isShapeInferenceIO(inp.name)) {
            // Shape-tensor values are read on the host at enqueue time; a graph
            // would freeze them, so such engines always take the plain path.
            ctx->activationMemoryDirty = true;
            ctx->pendingChange = true;
        }
    }
    for (int& dim : ctx->cachedInputDims) dim = -1;
    ctx->cachedInputAddr = nullptr;
    ctx->cachedOutputAddr = nullptr;

    if (!ctx->context->allInputDimensionsSpecified()) {
        setError("Not all multi-IO input dimensions are specified");
        return -1;
    }

    // ── Phase 2: Allocate output buffers and set addresses ──────────────────
    for (int o = 0; o < numOutputs; o++) {
        auto& out = outputs[o];
        if (!out.name) {
            setError("Invalid output tensor descriptor");
            return -1;
        }

        size_t elemSize = elemSizeForTensor(ctx, out.name, out.dtype);
        size_t bytes = shapeBytesWithElem(out.shape, out.nbDims, elemSize);
        if (bytes == 0) {
            setError(("Invalid shape for output '" + std::string(out.name) + "'").c_str());
            return -1;
        }
        if (!out.data) {
            setError(("Output data buffer not allocated for '" + std::string(out.name) + "'").c_str());
            return -1;
        }

        auto* buf = ensureTensorBuffer(ctx, out.name, bytes);
        if (!buf) {
            setError(("Failed to allocate GPU buffer for '" + std::string(out.name) + "'").c_str());
            return -1;
        }

        if (ctx->context->getTensorAddress(out.name) != buf->gpuBuf) {
            try {
                if (!ctx->context->setTensorAddress(out.name, buf->gpuBuf)) {
                    setError(("setTensorAddress failed for '" + std::string(out.name) + "'").c_str());
                    return -1;
                }
            } catch (...) {
                setError(("setTensorAddress failed for '" + std::string(out.name) + "'").c_str());
                return -1;
            }
            ctx->pendingChange = true;
        }
    }

    // ── Phase 3: Enqueue (graph replay when the bindings are unchanged) ─────
    if (!ensureActivationMemory(ctx)) return -1;
    bool success = enqueueWithGraph(ctx, graphKey);
    if (!success) {
        cudaGetLastError();
        dropGraph(ctx);
        ctx->pendingChange = true;
        ctx->lastEnqueueKey.clear();
        setError("Multi-IO inference enqueue failed");
        return -1;
    }

    // ── Phase 4: Copy outputs D2H ───────────────────────────────────────────
    for (int o = 0; o < numOutputs; o++) {
        auto& out = outputs[o];
        size_t bytes = shapeBytesWithElem(out.shape, out.nbDims,
                                          elemSizeForTensor(ctx, out.name, out.dtype));
        auto* cache = getIOCache(ctx);
        auto it = cache->buffers.find(out.name);
        if (it == cache->buffers.end()) {
            setError(("Output buffer not found for '" + std::string(out.name) + "'").c_str());
            return -1;
        }
        auto& buf = it->second;

        cudaError_t status = cudaMemcpyAsync(buf.pinnedBuf, buf.gpuBuf, bytes,
                                             cudaMemcpyDeviceToHost, ctx->stream);
        if (status != cudaSuccess) {
            setError(("D2H copy failed for '" + std::string(out.name) + "'").c_str());
            return -1;
        }
    }

    // ── Phase 5: Synchronize and copy pinned → caller ───────────────────────
    cudaError_t status = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess) {
        setError("Multi-IO stream sync failed");
        return -1;
    }

    for (int o = 0; o < numOutputs; o++) {
        auto& out = outputs[o];
        size_t bytes = shapeBytesWithElem(out.shape, out.nbDims,
                                          elemSizeForTensor(ctx, out.name, out.dtype));
        auto* cache = getIOCache(ctx);
        auto it = cache->buffers.find(out.name);
        if (it != cache->buffers.end() && it->second.pinnedBuf) {
            memcpy(out.data, it->second.pinnedBuf, bytes);
        }
    }

    return 0;
}

// ── Engine IO query ─────────────────────────────────────────────────────────

// Resolve the concrete shape and dtype of each named output for given input
// shapes, without moving tensor data. This is what lets a caller allocate
// output buffers before MultiIO: the kernel's own output buffers are sized
// internally, but the caller still has to know how large "logits" will be.
int TensorRT_ResolveOutputShapes(TensorRTContextHandle context,
    int numInputs, const TensorIOTensor* inputs,
    int numOutputs, const char* const* outputNames,
    int64_t* outShapes, int* outNbDims, int* outDtypes) {

    if (!context) { setError("Invalid context for shape resolution"); return -1; }
    if (numInputs <= 0 || !inputs) { setError("Shape resolution requires inputs"); return -1; }
    if (numOutputs < 0 || (numOutputs > 0 && (!outputNames || !outShapes || !outNbDims))) {
        setError("Invalid outputs for shape resolution");
        return -1;
    }

    auto* ctx = static_cast<TensorRTContextObj*>(context);
    std::lock_guard<std::mutex> lock(ctx->mu);

    std::vector<ShapeRequest> reqs(static_cast<size_t>(numInputs));
    for (int i = 0; i < numInputs; i++) {
        const auto& inp = inputs[i];
        if (!inp.name || inp.nbDims <= 0 || inp.nbDims > 8) {
            setError("Shape resolution: input is missing a name or has too many dims");
            return -1;
        }
        reqs[i].name = inp.name;
        reqs[i].dims.nbDims = inp.nbDims;
        for (int d = 0; d < inp.nbDims; d++) reqs[i].dims.d[d] = inp.shape[d];
    }
    try {
        if (!applyInputShapes(ctx, reqs.data(), numInputs)) return -1;
    } catch (...) {
        setError("setInputShape failed during shape resolution");
        return -1;
    }

    if (!ctx->context->allInputDimensionsSpecified()) {
        setError("Not all input dimensions are specified");
        return -1;
    }

    for (int o = 0; o < numOutputs; o++) {
        const char* name = outputNames[o];
        if (!name) { setError("Shape resolution: null output name"); return -1; }
        Dims shape;
        try {
            shape = ctx->context->getTensorShape(name);
        } catch (...) {
            setError(("Engine has no tensor named '" + std::string(name) + "'").c_str());
            return -1;
        }
        int dtype = -1;
        try {
            dtype = toTensorIOType(ctx->context->getEngine().getTensorDataType(name));
        } catch (...) {}
        size_t bytes = shapeBytesWithElem(shape.d, shape.nbDims, dtypeElemSize(dtype));
        if (bytes == 0) {
            setError(("Output '" + std::string(name) + "' has an unresolved shape").c_str());
            return -1;
        }
        outNbDims[o] = shape.nbDims;
        for (int d = 0; d < shape.nbDims && d < 8; d++) {
            outShapes[(size_t)o * 8 + d] = shape.d[d];
        }
        if (outDtypes) outDtypes[o] = dtype;
    }
    return 0;
}

size_t TensorRT_DtypeSize(int dtype) {
    return dtypeElemSize(dtype);
}

int TensorRT_GetNumIOTensors(TensorRTEngineHandle engine) {
    if (!engine) return -1;
    auto* obj = static_cast<TensorRTEngineObj*>(engine);
    try {
        return obj->engine->getNbIOTensors();
    } catch (...) {
        return -1;
    }
}

int TensorRT_GetIOTensorInfo(TensorRTEngineHandle engine,
    int index, char* name, int* isInput, int64_t* shape, int* nbDims) {
    return TensorRT_GetIOTensorInfoEx(engine, index, name, isInput, nullptr, shape, nbDims);
}

// Optimisation-profile bounds for a dynamic input. Without this a caller has no
// way to learn the sequence-length and marker range an engine accepts, which is
// what forces the whole runtime to hardcode a budget.
int TensorRT_GetProfileShape(TensorRTEngineHandle engine,
    const char* name, int which, int64_t* shape, int* nbDims) {
    if (!engine || !name || !shape || !nbDims) {
        setError("TensorRT_GetProfileShape: invalid arguments");
        return -1;
    }
    if (which < 0 || which > 2) {
        setError("TensorRT_GetProfileShape: which must be 0 (min), 1 (opt) or 2 (max)");
        return -1;
    }
    auto* obj = static_cast<TensorRTEngineObj*>(engine);
    try {
        int profiles = obj->engine->getNbOptimizationProfiles();
        if (profiles <= 0) return 0;
        // With several profiles the answer is their union per dimension: min is
        // the smallest min, max the largest max, opt comes from profile 0. The
        // union can name combinations no single profile admits (batch 8 at the
        // long profile's length); a run with such a shape fails before any GPU
        // work with "do not satisfy any optimization profile", and the caller
        // splits the batch.
        Dims dims = obj->engine->getProfileShape(
            name, 0, static_cast<OptProfileSelector>(which));
        if (dims.nbDims <= 0) return 0;
        if (which != 1) {
            for (int p = 1; p < profiles; p++) {
                Dims other = obj->engine->getProfileShape(
                    name, p, static_cast<OptProfileSelector>(which));
                if (other.nbDims != dims.nbDims) continue;
                for (int i = 0; i < dims.nbDims; i++) {
                    dims.d[i] = (which == 0) ? std::min(dims.d[i], other.d[i])
                                             : std::max(dims.d[i], other.d[i]);
                }
            }
        }
        *nbDims = dims.nbDims;
        for (int i = 0; i < dims.nbDims && i < 8; i++) shape[i] = dims.d[i];
        return 1;
    } catch (...) {
        setError("TensorRT_GetProfileShape: the engine has no profile for that tensor");
        return -1;
    }
}

int TensorRT_GetIOTensorInfoEx(TensorRTEngineHandle engine,
    int index, char* name, int* isInput, int* dtype, int64_t* shape, int* nbDims) {
    if (!engine || !name || !isInput || !shape || !nbDims) return -1;
    auto* obj = static_cast<TensorRTEngineObj*>(engine);

    try {
        const char* tensorName = obj->engine->getIOTensorName(index);
        if (!tensorName) return -1;

        strncpy(name, tensorName, 255);
        name[255] = '\0';

        auto mode = obj->engine->getTensorIOMode(tensorName);
        *isInput = (mode == nvinfer1::TensorIOMode::kINPUT) ? 1 : 0;

        if (dtype) {
            *dtype = toTensorIOType(obj->engine->getTensorDataType(tensorName));
        }

        auto dims = obj->engine->getTensorShape(tensorName);
        *nbDims = dims.nbDims;
        for (int i = 0; i < dims.nbDims && i < 8; i++) {
            shape[i] = dims.d[i];
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

// ── VRAM Info ────────────────────────────────────────────────────────────────

int TensorRT_GetVRAMInfo(int* freeMB, int* totalMB) {
    if (!freeMB || !totalMB) return -1;
    size_t freeBytes = 0, totalBytes = 0;
    cudaError_t err = cudaMemGetInfo(&freeBytes, &totalBytes);
    if (err != cudaSuccess) {
        setError(cudaGetErrorString(err));
        return -1;
    }
    *freeMB = (int)(freeBytes / (1024 * 1024));
    *totalMB = (int)(totalBytes / (1024 * 1024));
    return 0;
}

// ── Device Info ──────────────────────────────────────────────────────────────

int TensorRT_GetDeviceInfo(char* nameBuf, int* computeMajor, int* computeMinor) {
    if (!nameBuf || !computeMajor || !computeMinor) return -1;
    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, 0);
    if (err != cudaSuccess) {
        setError(cudaGetErrorString(err));
        return -1;
    }
    strncpy(nameBuf, props.name, 255);
    nameBuf[255] = '\0';
    *computeMajor = props.major;
    *computeMinor = props.minor;
    return 0;
}

// ── Pinned memory allocation (for Go callers) ────────────────────────────────

float* TensorRT_AllocPinned(size_t bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        setError(cudaGetErrorString(err));
        return nullptr;
    }
    return static_cast<float*>(ptr);
}

void TensorRT_FreePinned(float* ptr) {
    if (ptr) cudaFreeHost(ptr);
}

// ── Diagnostic toggle ────────────────────────────────────────────────────────

void TensorRT_SetDiagnostics(int enabled) {
    gDiagnosticsEnabled = (enabled != 0) ? 1 : 0;
}

} // extern "C"
