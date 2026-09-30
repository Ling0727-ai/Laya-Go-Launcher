// Native ONNX Runtime bridge for laya-trt.
//
// Ported from QualityScaler-go's pkg/func/ortcuda. The loading strategy is the
// same and deliberate: onnxruntime.dll is opened with LoadLibraryW and every
// entry point is fetched through OrtGetApiBase, so nothing here is a link-time
// dependency. A machine without ONNX Runtime can still start layatrt-gui,
// layatrt-server and layatrt-doctor and get a diagnosis instead of an exit code.
//
// What differs from QS is the tensor contract. QS binds one float32 image input
// and one float32 image output, sized from width/height/scale. laya's decision
// model takes five named inputs of mixed dtype (int64, int64, int64, bool,
// int64) and returns two float32 outputs, so this bridge is generic over a flat
// descriptor list: name, dtype, rank, shape, host pointer. Buffers stay
// caller-owned on both sides, so no allocation crosses the ABI.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "layatrt_onnx.h"
#include "onnxruntime_c_api.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace {

// ── error plumbing ──────────────────────────────────────────────────────────
//
// Every entry point reports failure through a caller-supplied buffer rather than
// a malloc'd string. That keeps the ABI free of ownership questions and means the
// Go side never converts a returned uintptr back into a pointer, which the
// unsafeptr vet check rightly refuses to bless.

void write_error(char* error, size_t error_bytes, const std::string& text) {
    if (!error || error_bytes == 0) return;
    size_t n = text.size() < error_bytes - 1 ? text.size() : error_bytes - 1;
    std::memcpy(error, text.data(), n);
    error[n] = '\0';
}

int fail(char* error, size_t error_bytes, const std::string& text) {
    write_error(error, error_bytes, text);
    return 1;
}

// copy_out writes a string into a caller buffer and always NUL-terminates.
void copy_out(char* buffer, size_t buffer_bytes, const std::string& text) {
    if (!buffer || buffer_bytes == 0) return;
    size_t n = text.size() < buffer_bytes - 1 ? text.size() : buffer_bytes - 1;
    std::memcpy(buffer, text.data(), n);
    buffer[n] = '\0';
}

std::wstring utf8_to_wide(const char* value) {
    if (!value) return std::wstring();
    int count = MultiByteToWideChar(CP_UTF8, 0, value, -1, nullptr, 0);
    if (count <= 0) return std::wstring();
    std::wstring out(static_cast<size_t>(count), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, value, -1, out.data(), count);
    out.resize(static_cast<size_t>(count - 1));
    return out;
}

template <typename T>
bool load_symbol(HMODULE module, const char* name, T* out) {
    *out = reinterpret_cast<T>(GetProcAddress(module, name));
    return *out != nullptr;
}

// ── CUDA runtime, loaded lazily ─────────────────────────────────────────────
//
// Only the three symbols the session needs to place work on its own stream. The
// CUDA runtime is optional: without it the session still runs on the CPU
// provider, which is what makes "onnx" usable on a machine with no NVIDIA GPU.

typedef int cudaError_t;
typedef void* cudaStream_t;

typedef cudaError_t(__cdecl* cudaStreamCreateWithFlagsFn)(cudaStream_t*, unsigned int);
typedef cudaError_t(__cdecl* cudaStreamDestroyFn)(cudaStream_t);
typedef const char*(__cdecl* cudaGetErrorStringFn)(cudaError_t);

struct CudaApi {
    HMODULE module = nullptr;
    cudaStreamCreateWithFlagsFn stream_create = nullptr;
    cudaStreamDestroyFn stream_destroy = nullptr;
    cudaGetErrorStringFn error_string = nullptr;

    void reset() {
        if (module) FreeLibrary(module);
        module = nullptr;
        stream_create = nullptr;
        stream_destroy = nullptr;
        error_string = nullptr;
    }
};

bool load_cuda(CudaApi* cuda) {
    // Newest first: CUDA 13 ships with the TensorRT 10.16 install this project
    // targets, and the older names are kept so a CUDA 12 box still works.
    static const wchar_t* candidates[] = {
        L"cudart64_13.dll", L"cudart64_130.dll", L"cudart64_12.dll",
        L"cudart64_120.dll", L"cudart64_110.dll"};
    for (const wchar_t* name : candidates) {
        cuda->module = LoadLibraryW(name);
        if (cuda->module) break;
    }
    if (!cuda->module) return false;
    bool ok = load_symbol(cuda->module, "cudaStreamCreateWithFlags", &cuda->stream_create) &&
              load_symbol(cuda->module, "cudaStreamDestroy", &cuda->stream_destroy) &&
              load_symbol(cuda->module, "cudaGetErrorString", &cuda->error_string);
    if (!ok) {
        cuda->reset();
        return false;
    }
    return true;
}

// ── session ─────────────────────────────────────────────────────────────────

struct Session {
    HMODULE ort_module = nullptr;
    const OrtApi* ort = nullptr;
    OrtEnv* env = nullptr;
    OrtSession* session = nullptr;
    OrtMemoryInfo* cpu_memory = nullptr;
    CudaApi cuda;
    cudaStream_t stream = nullptr;
    std::string provider_name;
    std::string provider_note;
    std::string runtime_version;

    // Descriptions captured at load time, so the Go side can size its buffers
    // without a second call into the runtime per run.
    struct TensorDesc {
        std::string name;
        int dtype = 0;
        std::vector<int64_t> shape;
    };
    std::vector<TensorDesc> inputs;
    std::vector<TensorDesc> outputs;

    // Results of the most recent run. ONNX Runtime owns the values; the names
    // are kept alongside so the copy-out calls can report them.
    std::vector<OrtValue*> results;
    std::vector<std::string> result_names;
};

// Byte width of one element, or 0 when the type is not one this bridge handles.
size_t element_size_of(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return 2;
        default: return 0;
    }
}

// release_results frees the previous run's values. Safe on an empty session.
void release_results(Session* s) {
    if (!s || !s->ort) return;
    for (OrtValue* v : s->results) {
        if (v) s->ort->ReleaseValue(v);
    }
    s->results.clear();
    s->result_names.clear();
}

int ort_check(Session* s, OrtStatus* status, char* error, size_t error_bytes, const char* operation) {
    if (!status) return 0;
    std::string message(operation);
    message += ": ";
    message += s->ort->GetErrorMessage(status);
    s->ort->ReleaseStatus(status);
    return fail(error, error_bytes, message);
}

void destroy_session(Session* s);

// Search the usual places for onnxruntime.dll. Order matters: an explicit path
// wins, then a copy shipped next to the executable, then whatever is on PATH.
std::wstring discover_runtime(const char* explicit_path) {
    if (explicit_path && *explicit_path) return utf8_to_wide(explicit_path);

    wchar_t exe[MAX_PATH] = {0};
    DWORD n = GetModuleFileNameW(nullptr, exe, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
        std::wstring dir(exe, n);
        size_t slash = dir.find_last_of(L"\\/");
        if (slash != std::wstring::npos) {
            std::wstring base = dir.substr(0, slash + 1);
            for (const wchar_t* rel : {L"onnxruntime.dll",
                                       L"runtime\\onnxruntime.dll",
                                       L"Assets\\onnxruntime.dll"}) {
                std::wstring candidate = base + rel;
                if (GetFileAttributesW(candidate.c_str()) != INVALID_FILE_ATTRIBUTES) {
                    return candidate;
                }
            }
        }
    }
    // Bare name: Windows resolves it against the DLL search path.
    return L"onnxruntime.dll";
}

// Read every input/output name, dtype and shape into the session's descriptors.
int capture_io(Session* s, char* error, size_t error_bytes) {
    size_t input_count = 0, output_count = 0;
    if (ort_check(s, s->ort->SessionGetInputCount(s->session, &input_count), error, error_bytes, "SessionGetInputCount") ||
        ort_check(s, s->ort->SessionGetOutputCount(s->session, &output_count), error, error_bytes, "SessionGetOutputCount")) {
        return 1;
    }

    OrtAllocator* allocator = nullptr;
    if (ort_check(s, s->ort->GetAllocatorWithDefaultOptions(&allocator), error, error_bytes, "GetAllocatorWithDefaultOptions")) {
        return 1;
    }

    struct Side {
        bool is_input;
        size_t count;
    };
    for (const Side side : {Side{true, input_count}, Side{false, output_count}}) {
        std::vector<Session::TensorDesc>* dst = side.is_input ? &s->inputs : &s->outputs;
        dst->reserve(side.count);
        for (size_t i = 0; i < side.count; ++i) {
            char* name = nullptr;
            OrtStatus* status = side.is_input
                ? s->ort->SessionGetInputName(s->session, i, allocator, &name)
                : s->ort->SessionGetOutputName(s->session, i, allocator, &name);
            if (ort_check(s, status, error, error_bytes, "SessionGetName")) return 1;

            Session::TensorDesc desc;
            desc.name = name ? name : "";
            if (name) allocator->Free(allocator, name);

            OrtTypeInfo* type_info = nullptr;
            status = side.is_input
                ? s->ort->SessionGetInputTypeInfo(s->session, i, &type_info)
                : s->ort->SessionGetOutputTypeInfo(s->session, i, &type_info);
            if (ort_check(s, status, error, error_bytes, "SessionGetTypeInfo")) return 1;

            const OrtTensorTypeAndShapeInfo* shape_info = nullptr;
            if (ort_check(s, s->ort->CastTypeInfoToTensorInfo(type_info, &shape_info), error, error_bytes, "CastTypeInfoToTensorInfo")) {
                s->ort->ReleaseTypeInfo(type_info);
                return 1;
            }

            ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            size_t rank = 0;
            if (ort_check(s, s->ort->GetTensorElementType(shape_info, &element_type), error, error_bytes, "GetTensorElementType") ||
                ort_check(s, s->ort->GetDimensionsCount(shape_info, &rank), error, error_bytes, "GetDimensionsCount")) {
                s->ort->ReleaseTypeInfo(type_info);
                return 1;
            }
            desc.dtype = static_cast<int>(element_type);

            if (rank > 0) {
                std::vector<int64_t> dims(rank, 0);
                if (ort_check(s, s->ort->GetDimensions(shape_info, dims.data(), rank), error, error_bytes, "GetDimensions")) {
                    s->ort->ReleaseTypeInfo(type_info);
                    return 1;
                }
                // A symbolic dimension comes back as -1, which is the convention
                // the rest of the program already uses for "dynamic".
                desc.shape = dims;
            }
            s->ort->ReleaseTypeInfo(type_info);
            dst->push_back(desc);
        }
    }
    return 0;
}

// Choose and append an execution provider.
//
// The bridge records a provider append failure and can create a CPU session;
// the Go backend rejects that session for an explicitly requested provider.
// Only an unspecified provider is allowed to keep the CPU fallback.
int append_provider(Session* s, OrtSessionOptions* options, const char* provider,
                    const char* const* keys, const char* const* values, size_t count) {
    std::string want = provider ? provider : "";
    for (char& c : want) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
    if (want.empty()) want = "cuda"; // default: GPU when available, else CPU

    if (want == "cuda") {
        OrtCUDAProviderOptionsV2* cuda_options = nullptr;
        if (s->ort->CreateCUDAProviderOptions(&cuda_options) == nullptr) {
            bool ok = true;
            if (count > 0) {
                ok = s->ort->UpdateCUDAProviderOptions(cuda_options, keys, values, count) == nullptr;
            }
            // The session's own stream keeps inference ordered without touching
            // the legacy default stream, as in the QS kernel.
            if (ok && s->stream) {
                ok = s->ort->UpdateCUDAProviderOptionsWithValue(
                         cuda_options, "user_compute_stream", s->stream) == nullptr;
            }
            if (ok) {
                OrtStatus* status =
                    s->ort->SessionOptionsAppendExecutionProvider_CUDA_V2(options, cuda_options);
                if (status == nullptr) {
                    s->provider_name = "cuda";
                    s->ort->ReleaseCUDAProviderOptions(cuda_options);
                    return 0;
                }
                s->provider_note = std::string("CUDA provider unavailable, using CPU: ") +
                                   s->ort->GetErrorMessage(status);
                s->ort->ReleaseStatus(status);
            } else {
                s->provider_note = "CUDA provider options could not be set, using CPU";
            }
            s->ort->ReleaseCUDAProviderOptions(cuda_options);
        } else {
            s->provider_note = "CUDA provider is not present in this onnxruntime build, using CPU";
        }
        s->provider_name = "cpu";
        return 0;
    }

    // DirectML and any other named provider go through the generic entry point,
    // which is also what keeps this file building against older ORT headers
    // where a dedicated DML function may not exist.
    if (want != "cpu") {
        std::string upper = (want == "directml" || want == "dml") ? "DML" : want;
        for (char& c : upper) c = static_cast<char>(::toupper(static_cast<unsigned char>(c)));
        OrtStatus* status = s->ort->SessionOptionsAppendExecutionProvider(
            options, upper.c_str(), count > 0 ? keys : nullptr,
            count > 0 ? values : nullptr, count);
        if (status == nullptr) {
            s->provider_name = want;
            return 0;
        }
        s->provider_note = want + " provider unavailable, using CPU: " + s->ort->GetErrorMessage(status);
        s->ort->ReleaseStatus(status);
    }

    // CPU is a fresh session's default, so nothing has to be appended.
    s->provider_name = "cpu";
    return 0;
}

} // namespace

// ── exported ABI ────────────────────────────────────────────────────────────

extern "C" int LayaOnnx_CreateSession(
    const char* runtime_path, const char* model_path, const char* provider,
    const char* const* option_keys, const char* const* option_values,
    size_t option_count, void** out, char* error, size_t error_bytes) {
    if (out) *out = nullptr;
    if (!model_path || !*model_path || !out) return fail(error, error_bytes, "invalid session arguments");

    Session* s = new (std::nothrow) Session();
    if (!s) return fail(error, error_bytes, "session allocation failed");

    std::wstring runtime_w = discover_runtime(runtime_path);
    s->ort_module = LoadLibraryW(runtime_w.c_str());
    if (!s->ort_module) {
        delete s;
        return fail(error, error_bytes, "unable to load onnxruntime.dll (pass its path in RuntimePath)");
    }
    typedef const OrtApiBase*(ORT_API_CALL* GetApiBaseFn)(void);
    GetApiBaseFn get_api_base =
        reinterpret_cast<GetApiBaseFn>(GetProcAddress(s->ort_module, "OrtGetApiBase"));
    if (!get_api_base) {
        destroy_session(s);
        return fail(error, error_bytes, "onnxruntime.dll does not export OrtGetApiBase");
    }
    const OrtApiBase* base = get_api_base();
    s->ort = base ? base->GetApi(ORT_API_VERSION) : nullptr;
    if (!s->ort) {
        destroy_session(s);
        return fail(error, error_bytes, "ONNX Runtime C API version is incompatible");
    }
    // GetVersionString lives on OrtApiBase, not OrtApi: it is available before
    // (and independently of) any API-version negotiation.
    if (base->GetVersionString) {
        const char* version = base->GetVersionString();
        s->runtime_version = version ? version : "";
    }

    // The CUDA stream is created before the provider is appended, because the
    // provider option points at it.
    if (load_cuda(&s->cuda)) {
        if (s->cuda.stream_create(&s->stream, 1 /* cudaStreamNonBlocking */) != 0) {
            s->stream = nullptr;
        }
    }

    // Diagnostics, off by default:
    //   LAYA_TRT_ORT_LOG_LEVEL=0..4  ORT log severity (0 = verbose: node placement)
    //   LAYA_TRT_ORT_PROFILE=<prefix> write an ORT profile JSON when the session closes
    OrtLoggingLevel log_level = ORT_LOGGING_LEVEL_WARNING;
    if (const char* v = std::getenv("LAYA_TRT_ORT_LOG_LEVEL")) {
        int n = std::atoi(v);
        if (n >= 0 && n <= 4) log_level = static_cast<OrtLoggingLevel>(n);
    }
    if (ort_check(s, s->ort->CreateEnv(log_level, "laya-trt-onnx", &s->env), error, error_bytes, "CreateEnv")) {
        destroy_session(s);
        return 1;
    }
    if (ort_check(s, s->ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &s->cpu_memory),
                  error, error_bytes, "CreateCpuMemoryInfo")) {
        destroy_session(s);
        return 1;
    }

    OrtSessionOptions* options = nullptr;
    if (ort_check(s, s->ort->CreateSessionOptions(&options), error, error_bytes, "CreateSessionOptions")) {
        destroy_session(s);
        return 1;
    }
    // Single-threaded per session: concurrency is the caller's context pool, and
    // ORT's own spinning threads would fight it for the GPU.
    ort_check(s, s->ort->SetIntraOpNumThreads(options, 1), error, error_bytes, "SetIntraOpNumThreads");
    ort_check(s, s->ort->SetInterOpNumThreads(options, 1), error, error_bytes, "SetInterOpNumThreads");
    ort_check(s, s->ort->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL), error, error_bytes,
              "SetSessionGraphOptimizationLevel");
    if (const char* prefix = std::getenv("LAYA_TRT_ORT_PROFILE")) {
        if (*prefix) {
            std::wstring prefix_w = utf8_to_wide(prefix);
            ort_check(s, s->ort->EnableProfiling(options, prefix_w.c_str()), error, error_bytes,
                      "EnableProfiling");
        }
    }

    // DirectML registers as "DML" in the ORT C API, not "DIRECTML".
    // It requires sequential execution and memory patterns disabled.
    std::string provider_mode = provider ? provider : "";
    for (char& c : provider_mode) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
    if (provider_mode == "directml" || provider_mode == "dml") {
        if (ort_check(s, s->ort->SetSessionExecutionMode(options, ORT_SEQUENTIAL), error, error_bytes,
                      "SetSessionExecutionMode") ||
            ort_check(s, s->ort->DisableMemPattern(options), error, error_bytes, "DisableMemPattern")) {
            s->ort->ReleaseSessionOptions(options);
            destroy_session(s);
            return 1;
        }
    }

    if (append_provider(s, options, provider, option_keys, option_values, option_count) != 0) {
        s->ort->ReleaseSessionOptions(options);
        destroy_session(s);
        return 1;
    }

    std::wstring model_w = utf8_to_wide(model_path);
    OrtStatus* create_status = s->ort->CreateSession(s->env, model_w.c_str(), options, &s->session);
    s->ort->ReleaseSessionOptions(options);
    if (ort_check(s, create_status, error, error_bytes, "CreateSession")) {
        destroy_session(s);
        return 1;
    }

    if (capture_io(s, error, error_bytes)) {
        destroy_session(s);
        return 1;
    }

    *out = s;
    return 0;
}

extern "C" void LayaOnnx_DestroySession(void* handle) {
    if (!handle) return;
    destroy_session(static_cast<Session*>(handle));
}

extern "C" int LayaOnnx_GetInputCount(void* handle, size_t* out, char* error, size_t error_bytes) {
    Session* s = static_cast<Session*>(handle);
    if (!s || !out) return fail(error, error_bytes, "invalid session");
    *out = s->inputs.size();
    return 0;
}

extern "C" int LayaOnnx_GetOutputCount(void* handle, size_t* out, char* error, size_t error_bytes) {
    Session* s = static_cast<Session*>(handle);
    if (!s || !out) return fail(error, error_bytes, "invalid session");
    *out = s->outputs.size();
    return 0;
}

extern "C" int LayaOnnx_GetTensorInfo(
    void* handle, int is_input, size_t index, char* name, size_t name_bytes,
    int* dtype, int64_t* shape, int* nb_dims, int max_dims, char* error, size_t error_bytes) {
    Session* s = static_cast<Session*>(handle);
    if (!s) return fail(error, error_bytes, "invalid session");

    const std::vector<Session::TensorDesc>& list = is_input ? s->inputs : s->outputs;
    if (index >= list.size()) return fail(error, error_bytes, "tensor index out of range");
    const Session::TensorDesc& desc = list[index];

    if (name && name_bytes > 0) {
        size_t n = desc.name.size() < name_bytes - 1 ? desc.name.size() : name_bytes - 1;
        std::memcpy(name, desc.name.data(), n);
        name[n] = '\0';
    }
    if (dtype) *dtype = desc.dtype;
    if (nb_dims) *nb_dims = static_cast<int>(desc.shape.size());
    if (shape) {
        int n = static_cast<int>(desc.shape.size());
        if (n > max_dims) n = max_dims;
        for (int i = 0; i < n; ++i) shape[i] = desc.shape[static_cast<size_t>(i)];
    }
    return 0;
}

// Run computes the requested outputs and keeps the OrtValues.
//
// ONNX Runtime allocates the results, because a graph exported with dynamic
// axes declares its outputs symbolically and there is no shape to pre-size a
// buffer from until the graph has run. The previous run's results are released
// here, so the session holds at most one result set at a time — which is also
// what makes the copy-out calls below unambiguous.
extern "C" int LayaOnnx_Run(
    void* handle, const LayaOnnxTensor* inputs, size_t input_count,
    const char* const* requested_outputs, size_t requested_count,
    char* error, size_t error_bytes) {
    Session* s = static_cast<Session*>(handle);
    if (!s || !s->session) return fail(error, error_bytes, "invalid session");

    // Drop the previous results before anything can fail, so a failed run never
    // leaves stale outputs readable as if they were fresh.
    release_results(s);

    std::vector<const char*> input_names;
    std::vector<OrtValue*> input_values;

    input_names.reserve(input_count);
    input_values.reserve(input_count);

    int rc = 0;
    // Every input OrtValue created here is released on every path, so a failed
    // run cannot leak a tensor.
    auto cleanup_inputs = [&]() {
        for (OrtValue* v : input_values) s->ort->ReleaseValue(v);
    };

    for (size_t i = 0; i < input_count && rc == 0; ++i) {
        const LayaOnnxTensor& t = inputs[i];
        if (!t.name || !t.data) {
            rc = fail(error, error_bytes, "input descriptor is incomplete");
            break;
        }
        OrtValue* value = nullptr;
        rc = ort_check(s, s->ort->CreateTensorWithDataAsOrtValue(
                              s->cpu_memory, t.data, t.data_bytes, t.shape,
                              static_cast<size_t>(t.nb_dims),
                              static_cast<ONNXTensorElementDataType>(t.dtype), &value),
                       error, error_bytes, "Create input tensor");
        if (rc == 0) {
            input_names.push_back(t.name);
            input_values.push_back(value);
        }
    }
    if (rc != 0) {
        cleanup_inputs();
        return rc;
    }

    // Which outputs to compute: the caller's list, or the model's own order.
    std::vector<const char*> output_names;
    if (requested_outputs && requested_count > 0) {
        output_names.assign(requested_outputs, requested_outputs + requested_count);
    } else {
        for (const Session::TensorDesc& desc : s->outputs) output_names.push_back(desc.name.c_str());
    }
    if (output_names.empty()) {
        cleanup_inputs();
        return fail(error, error_bytes, "model declares no outputs");
    }

    std::vector<OrtValue*> results(output_names.size(), nullptr);
    OrtStatus* status = s->ort->Run(s->session, nullptr, input_names.data(),
                                    input_values.data(), input_names.size(),
                                    output_names.data(), output_names.size(),
                                    results.data());
    cleanup_inputs();
    if (ort_check(s, status, error, error_bytes, "Run")) {
        // Run may have populated some entries before failing; release whatever
        // it produced so nothing is left half-owned.
        for (OrtValue* v : results) {
            if (v) s->ort->ReleaseValue(v);
        }
        return 1;
    }

    s->results = std::move(results);
    // Store the names as owned strings: the pointers above point at the model's
    // own descriptors, which outlive the run, but owning them keeps the result
    // set self-contained.
    s->result_names.clear();
    s->result_names.reserve(output_names.size());
    for (const char* n : output_names) s->result_names.emplace_back(n ? n : "");
    return 0;
}

extern "C" int LayaOnnx_GetResultCount(void* handle, size_t* out, char* error, size_t error_bytes) {
    Session* s = static_cast<Session*>(handle);
    if (!s || !out) return fail(error, error_bytes, "invalid session");
    *out = s->results.size();
    return 0;
}

// LayaOnnx_GetResultInfo reports the concrete shape a completed run produced.
extern "C" int LayaOnnx_GetResultInfo(
    void* handle, size_t index, char* name, size_t name_bytes,
    int* dtype, int64_t* shape, int* nb_dims, int max_dims, char* error, size_t error_bytes) {
    Session* s = static_cast<Session*>(handle);
    if (!s) return fail(error, error_bytes, "invalid session");
    if (index >= s->results.size()) return fail(error, error_bytes, "result index out of range");

    OrtValue* value = s->results[index];
    OrtTensorTypeAndShapeInfo* shape_info = nullptr;
    if (ort_check(s, s->ort->GetTensorTypeAndShape(value, &shape_info), error, error_bytes,
                  "GetTensorTypeAndShape")) {
        return 1;
    }
    // One guard for every early return past this point.
    struct ShapeGuard {
        const OrtApi* ort;
        OrtTensorTypeAndShapeInfo* info;
        ~ShapeGuard() { if (info) ort->ReleaseTensorTypeAndShapeInfo(info); }
    } guard{s->ort, shape_info};

    ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    size_t rank = 0;
    if (ort_check(s, s->ort->GetTensorElementType(shape_info, &element_type), error, error_bytes,
                  "GetTensorElementType") ||
        ort_check(s, s->ort->GetDimensionsCount(shape_info, &rank), error, error_bytes,
                  "GetDimensionsCount")) {
        return 1;
    }

    std::vector<int64_t> dims(rank, 0);
    if (rank > 0) {
        if (ort_check(s, s->ort->GetDimensions(shape_info, dims.data(), rank), error, error_bytes,
                      "GetDimensions")) {
            return 1;
        }
    }

    if (name && name_bytes > 0) {
        const std::string& text = s->result_names[index];
        size_t n = text.size() < name_bytes - 1 ? text.size() : name_bytes - 1;
        std::memcpy(name, text.data(), n);
        name[n] = '\0';
    }
    if (dtype) *dtype = static_cast<int>(element_type);
    if (nb_dims) *nb_dims = static_cast<int>(dims.size());
    if (shape) {
        int n = static_cast<int>(dims.size());
        if (n > max_dims) n = max_dims;
        for (int i = 0; i < n; ++i) shape[i] = dims[static_cast<size_t>(i)];
    }
    return 0;
}

// LayaOnnx_CopyResult copies one result's bytes into caller memory.
//
// The copy is explicit rather than a borrowed pointer so the Go side never holds
// a pointer into memory ONNX Runtime owns, which could be freed by the next run.
extern "C" int LayaOnnx_CopyResult(
    void* handle, size_t index, void* dst, size_t dst_bytes, size_t* written,
    char* error, size_t error_bytes) {
    Session* s = static_cast<Session*>(handle);
    if (!s) return fail(error, error_bytes, "invalid session");
    if (index >= s->results.size()) return fail(error, error_bytes, "result index out of range");
    if (!dst) return fail(error, error_bytes, "destination buffer is required");

    void* src = nullptr;
    if (ort_check(s, s->ort->GetTensorMutableData(s->results[index], &src), error, error_bytes,
                  "GetTensorMutableData")) {
        return 1;
    }

    OrtTensorTypeAndShapeInfo* shape_info = nullptr;
    if (ort_check(s, s->ort->GetTensorTypeAndShape(s->results[index], &shape_info), error,
                  error_bytes, "GetTensorTypeAndShape")) {
        return 1;
    }
    size_t elements = 0;
    OrtStatus* count_status = s->ort->GetTensorShapeElementCount(shape_info, &elements);
    s->ort->ReleaseTensorTypeAndShapeInfo(shape_info);
    if (ort_check(s, count_status, error, error_bytes, "GetTensorShapeElementCount")) return 1;

    ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    if (ort_check(s, s->ort->GetTensorTypeAndShape(s->results[index], &shape_info), error,
                  error_bytes, "GetTensorTypeAndShape")) {
        return 1;
    }
    OrtStatus* type_status = s->ort->GetTensorElementType(shape_info, &element_type);
    s->ort->ReleaseTensorTypeAndShapeInfo(shape_info);
    if (ort_check(s, type_status, error, error_bytes, "GetTensorElementType")) return 1;

    size_t element_size = element_size_of(element_type);
    if (element_size == 0) return fail(error, error_bytes, "result has an unsupported element type");
    size_t bytes = elements * element_size;
    if (bytes > dst_bytes) {
        return fail(error, error_bytes, "destination buffer is too small for the result");
    }
    std::memcpy(dst, src, bytes);
    if (written) *written = bytes;
    return 0;
}

// The introspection calls copy into a caller buffer. Returning a `const char*`
// would mean the Go side converting a uintptr back into a pointer, which the
// unsafeptr vet check refuses — and rightly, since the pointer's lifetime would
// be a contract nothing enforces.

extern "C" int LayaOnnx_RuntimeVersion(void* handle, char* buffer, size_t buffer_bytes) {
    Session* s = static_cast<Session*>(handle);
    copy_out(buffer, buffer_bytes, s ? s->runtime_version : std::string());
    return 0;
}

// LayaOnnx_RuntimeVersionAt reports a runtime library's version without creating
// a session.
//
// The library is loaded and released here so a caller can ask "which ONNX
// Runtime is installed?" before committing to a model. OrtGetApiBase is resolved
// rather than OrtApi, because the version string is available before (and
// independently of) any API-version negotiation.
extern "C" int LayaOnnx_RuntimeVersionAt(
    const char* runtime_path, char* buffer, size_t buffer_bytes,
    char* error, size_t error_bytes) {
    copy_out(buffer, buffer_bytes, std::string());

    std::wstring runtime_w = discover_runtime(runtime_path);
    HMODULE module = LoadLibraryW(runtime_w.c_str());
    if (!module) {
        return fail(error, error_bytes,
                    "unable to load onnxruntime.dll (pass its path in RuntimePath)");
    }
    typedef const OrtApiBase*(ORT_API_CALL* GetApiBaseFn)(void);
    GetApiBaseFn get_api_base =
        reinterpret_cast<GetApiBaseFn>(GetProcAddress(module, "OrtGetApiBase"));
    if (!get_api_base) {
        FreeLibrary(module);
        return fail(error, error_bytes, "onnxruntime.dll does not export OrtGetApiBase");
    }
    const OrtApiBase* base = get_api_base();
    if (!base || !base->GetVersionString) {
        FreeLibrary(module);
        return fail(error, error_bytes, "onnxruntime.dll reports no version string");
    }
    copy_out(buffer, buffer_bytes, std::string(base->GetVersionString()));
    FreeLibrary(module);
    return 0;
}

extern "C" int LayaOnnx_ProviderName(void* handle, char* buffer, size_t buffer_bytes) {
    Session* s = static_cast<Session*>(handle);
    copy_out(buffer, buffer_bytes, s ? s->provider_name : std::string());
    return 0;
}

// A note explaining why the requested provider could not be used, or "" when the
// request was honoured. The caller reports this rather than failing, so a silent
// fall back to CPU is impossible.
extern "C" int LayaOnnx_ProviderNote(void* handle, char* buffer, size_t buffer_bytes) {
    Session* s = static_cast<Session*>(handle);
    copy_out(buffer, buffer_bytes, s ? s->provider_note : std::string());
    return 0;
}

namespace {

void destroy_session(Session* s) {
    if (!s) return;
    release_results(s);
    if (s->ort) {
        if (s->session) s->ort->ReleaseSession(s->session);
        if (s->cpu_memory) s->ort->ReleaseMemoryInfo(s->cpu_memory);
        if (s->env) s->ort->ReleaseEnv(s->env);
    }
    if (s->stream && s->cuda.stream_destroy) s->cuda.stream_destroy(s->stream);
    s->cuda.reset();
    if (s->ort_module) FreeLibrary(s->ort_module);
    delete s;
}

} // namespace
