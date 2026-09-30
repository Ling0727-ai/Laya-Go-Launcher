// Run-time binding of qualityscaler_tensorrt.dll.
//
// kernel.go used to link the kernel at load time (-lqualityscaler_tensorrt), so
// a PATH without TensorRT killed the process before main() with 0xC0000279 and
// no message. Here every TensorRT_* function kernel.go calls is defined as a
// forwarder to a pointer resolved by LayaKernel_Load, which Go calls after
// internal/nativeenv has found the DLL directories. Until then — or when the
// load fails — each forwarder returns the kernel's own failure value.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <string.h>

#include "ai_tensorrt_cpp.h"
#include "kernel_loader.h"

// X(return type, name, parameter list, argument list, failure value)
#define LAYA_KERNEL_FUNCS(X)                                                              \
    X(int, TensorRT_Initialize, (void), (), -1)                                           \
    X(const char*, TensorRT_Version, (void), (), "")                                      \
    X(const char*, TensorRT_GetLastError, (void), (), "tensorrt kernel is not loaded")    \
    X(void*, TensorRT_LoadEngine, (const char* p), (p), NULL)                             \
    X(uint64_t, TensorRT_GetDeviceMemorySize, (TensorRTEngineHandle e), (e), 0)           \
    X(TensorRTContextHandle, TensorRT_CreateContext, (TensorRTEngineHandle e), (e), NULL) \
    X(size_t, TensorRT_DtypeSize, (int d), (d), 0)                                        \
    X(TensorRunHandle, TensorRT_RunCreate, (TensorRTContextHandle c), (c), NULL)          \
    X(int, TensorRT_RunAddInput,                                                          \
      (TensorRunHandle r, const char* n, int d, const void* p, const int64_t* s, int nb), \
      (r, n, d, p, s, nb), -1)                                                            \
    X(int, TensorRT_RunAddOutput, (TensorRunHandle r, const char* n), (r, n), -1)         \
    X(int, TensorRT_RunOutputCount, (TensorRunHandle r), (r), -1)                         \
    X(int, TensorRT_GetProfileShape,                                                      \
      (TensorRTEngineHandle e, const char* n, int w, int64_t* s, int* nb),                \
      (e, n, w, s, nb), -1)                                                               \
    X(int, TensorRT_RunResolveOutputs,                                                    \
      (TensorRunHandle r, char* n, int st, int* d, int64_t* s, int* nb, int c),           \
      (r, n, st, d, s, nb, c), -1)                                                        \
    X(int, TensorRT_RunExecute, (TensorRunHandle r), (r), -1)                             \
    X(size_t, TensorRT_RunOutputBytes, (TensorRunHandle r, int i), (r, i), 0)             \
    X(int, TensorRT_RunReadOutput, (TensorRunHandle r, int i, void* p, size_t b),         \
      (r, i, p, b), -1)                                                                   \
    X(int, TensorRT_GetNumIOTensors, (TensorRTEngineHandle e), (e), -1)                   \
    X(int, TensorRT_GetIOTensorInfoEx,                                                    \
      (TensorRTEngineHandle e, int i, char* n, int* in, int* d, int64_t* s, int* nb),     \
      (e, i, n, in, d, s, nb), -1)                                                        \
    X(int, TensorRT_GetVRAMInfo, (int* f, int* t), (f, t), -1)                            \
    X(int, TensorRT_GetDeviceInfo, (char* n, int* a, int* b), (n, a, b), -1)

#define LAYA_KERNEL_VOIDS(X)                                                 \
    X(TensorRT_UnloadEngine, (TensorRTEngineHandle e), (e))                  \
    X(TensorRT_DestroyContext, (TensorRTContextHandle c), (c))               \
    X(TensorRT_RunDestroy, (TensorRunHandle r), (r))                         \
    X(TensorRT_SetDiagnostics, (int v), (v))

#define DECLARE_PTR(ret, name, params, args, fail) static ret(__cdecl* p_##name) params;
#define DECLARE_VPTR(name, params, args) static void(__cdecl* p_##name) params;
LAYA_KERNEL_FUNCS(DECLARE_PTR)
LAYA_KERNEL_VOIDS(DECLARE_VPTR)

#define DEFINE_FWD(ret, name, params, args, fail) \
    ret name params { return p_##name ? p_##name args : (ret)(fail); }
#define DEFINE_VFWD(name, params, args) \
    void name params { if (p_##name) p_##name args; }
LAYA_KERNEL_FUNCS(DEFINE_FWD)
LAYA_KERNEL_VOIDS(DEFINE_VFWD)

static HMODULE g_module;

static void format_error(char* err, int errBytes, const char* what, const char* path, DWORD code) {
    if (!err || errBytes <= 0) return;
    char sys[256] = {0};
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, code,
                   MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US), sys, sizeof(sys), NULL);
    size_t n = strlen(sys);
    while (n > 0 && (sys[n - 1] == '\r' || sys[n - 1] == '\n' || sys[n - 1] == ' ')) sys[--n] = 0;
    snprintf(err, (size_t)errBytes, "%s %s: %s (code %lu)", what, path, sys, (unsigned long)code);
}

int LayaKernel_Load(const char* path, char* err, int errBytes) {
    if (g_module) return 0;
    if (!path || !*path) path = "qualityscaler_tensorrt.dll";

    int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
    if (wlen <= 0 || wlen > 32768) {
        format_error(err, errBytes, "invalid path", path, ERROR_INVALID_NAME);
        return 1;
    }
    wchar_t wpath[32768];
    MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, wlen);

    // A full path is loaded with its own directory searched first for its
    // dependencies; PATH (already prepared by nativeenv) supplies nvinfer.
    DWORD flags = (strchr(path, '\\') || strchr(path, '/')) ? LOAD_WITH_ALTERED_SEARCH_PATH : 0;
    HMODULE m = LoadLibraryExW(wpath, NULL, flags);
    if (!m) {
        format_error(err, errBytes, "cannot load", path, GetLastError());
        return 1;
    }

#define RESOLVE(ret, name, params, args, fail)                                   \
    p_##name = (ret(__cdecl*) params)(void*)GetProcAddress(m, #name);            \
    if (!p_##name) { missing = #name; goto fail_missing; }
#define VRESOLVE(name, params, args)                                             \
    p_##name = (void(__cdecl*) params)(void*)GetProcAddress(m, #name);           \
    if (!p_##name) { missing = #name; goto fail_missing; }

    const char* missing = NULL;
    LAYA_KERNEL_FUNCS(RESOLVE)
    LAYA_KERNEL_VOIDS(VRESOLVE)
    g_module = m;
    return 0;

fail_missing:
#define CLEAR(ret, name, params, args, fail) p_##name = NULL;
#define VCLEAR(name, params, args) p_##name = NULL;
    LAYA_KERNEL_FUNCS(CLEAR)
    LAYA_KERNEL_VOIDS(VCLEAR)
    FreeLibrary(m);
    if (err && errBytes > 0) snprintf(err, (size_t)errBytes, "%s does not export %s", path, missing);
    return 1;
}
