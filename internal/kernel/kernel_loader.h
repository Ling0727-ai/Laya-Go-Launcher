#ifndef LAYA_KERNEL_LOADER_H
#define LAYA_KERNEL_LOADER_H

#ifdef __cplusplus
extern "C" {
#endif

// LayaKernel_Load opens the TensorRT kernel DLL at run time and resolves every
// entry point kernel.go calls. path is UTF-8; a full path is loaded with its own
// directory on the dependency search path. Returns 0 on success; on failure err
// receives a message and every TensorRT_* call keeps returning its failure
// value instead of crashing.
int LayaKernel_Load(const char* path, char* err, int errBytes);

#ifdef __cplusplus
}
#endif

#endif
