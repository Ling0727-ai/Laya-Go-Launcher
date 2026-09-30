// Direct C++ exercise of the ported kernel, bypassing Go entirely.
// Isolates whether a failure is in the kernel or in the cgo layer.

#include "ai_tensorrt_cpp.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const char* kEngine =
    "C:\\Users\\lingxin\\Documents\\laya-ort-probe\\laya_e2e.engine";

int main() {
    printf("[1] Initialize\n"); fflush(stdout);
    if (TensorRT_Initialize() != 0) {
        printf("    FAIL: %s\n", TensorRT_GetLastError());
        return 1;
    }
    printf("    ok\n");

    printf("[2] LoadEngine\n"); fflush(stdout);
    TensorRTEngineHandle engine = TensorRT_LoadEngine(kEngine);
    if (!engine) {
        printf("    FAIL: %s\n", TensorRT_GetLastError());
        return 1;
    }
    printf("    ok\n");

    int n = TensorRT_GetNumIOTensors(engine);
    printf("[3] IO tensors: %d\n", n); fflush(stdout);
    for (int i = 0; i < n; i++) {
        char name[256] = {0};
        int isInput = 0, dtype = -1, nbDims = 0;
        int64_t shape[8] = {0};
        if (TensorRT_GetIOTensorInfoEx(engine, i, name, &isInput, &dtype, shape, &nbDims) != 0) {
            printf("    FAIL reading tensor %d\n", i);
            return 1;
        }
        printf("    %s %-16s dtype=%d dims=%d shape=", isInput ? "in " : "out", name, dtype, nbDims);
        for (int d = 0; d < nbDims; d++) printf("%lld ", (long long)shape[d]);
        printf("\n");
    }

    printf("[4] CreateContext\n"); fflush(stdout);
    TensorRTContextHandle ctx = TensorRT_CreateContext(engine);
    if (!ctx) {
        printf("    FAIL: %s\n", TensorRT_GetLastError());
        return 1;
    }
    printf("    ok\n");

    // ── inputs, matching the engine's reported dtypes ─────────────────────
    const int L = 64, K = 3;
    std::vector<int64_t> ids(L), att(L), pos(K), qt(1, 0);
    for (int i = 0; i < L; i++) { ids[i] = 100 + (i * 7919) % 19000; att[i] = 1; }
    ids[0] = 50281;
    pos[0] = 5; pos[1] = 10; pos[2] = 15;
    std::vector<uint8_t> mm(K, 1);

    TensorIOTensor inputs[5];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].name = "input_ids";      inputs[0].dtype = TENSOR_IO_INT64;   inputs[0].data = ids.data();  inputs[0].nbDims = 2; inputs[0].shape[0] = 1; inputs[0].shape[1] = L;
    inputs[1].name = "attention_mask"; inputs[1].dtype = TENSOR_IO_INT64;   inputs[1].data = att.data();  inputs[1].nbDims = 2; inputs[1].shape[0] = 1; inputs[1].shape[1] = L;
    inputs[2].name = "marker_pos";     inputs[2].dtype = TENSOR_IO_INT64;   inputs[2].data = pos.data();  inputs[2].nbDims = 2; inputs[2].shape[0] = 1; inputs[2].shape[1] = K;
    inputs[3].name = "marker_mask";    inputs[3].dtype = TENSOR_IO_BOOL;    inputs[3].data = mm.data();   inputs[3].nbDims = 2; inputs[3].shape[0] = 1; inputs[3].shape[1] = K;
    inputs[4].name = "qtype";          inputs[4].dtype = TENSOR_IO_INT64;   inputs[4].data = qt.data();   inputs[4].nbDims = 1; inputs[4].shape[0] = 1;

    printf("[5] ResolveOutputShapes\n"); fflush(stdout);
    const char* outNames[2] = {"logits", "act_logits"};
    int64_t shapes[16] = {0};
    int nbDims[2] = {0}, dtypes[2] = {0};
    if (TensorRT_ResolveOutputShapes(ctx, 5, inputs, 2, outNames, shapes, nbDims, dtypes) != 0) {
        printf("    FAIL: %s\n", TensorRT_GetLastError());
        return 1;
    }
    for (int i = 0; i < 2; i++) {
        printf("    %s dtype=%d dims=%d shape=", outNames[i], dtypes[i], nbDims[i]);
        for (int d = 0; d < nbDims[i]; d++) printf("%lld ", (long long)shapes[i * 8 + d]);
        printf("\n");
    }

    std::vector<float> logits(3, 0.f), actLogits(2, 0.f);
    TensorIOTensor outputs[2];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].name = "logits";     outputs[0].dtype = TENSOR_IO_FLOAT32; outputs[0].data = logits.data();    outputs[0].nbDims = 2; outputs[0].shape[0] = 1; outputs[0].shape[1] = 3;
    outputs[1].name = "act_logits"; outputs[1].dtype = TENSOR_IO_FLOAT32; outputs[1].data = actLogits.data(); outputs[1].nbDims = 2; outputs[1].shape[0] = 1; outputs[1].shape[1] = 2;

    printf("[6] RunInferenceMultiIO\n"); fflush(stdout);
    if (TensorRT_RunInferenceMultiIO(ctx, 5, inputs, 2, outputs) != 0) {
        printf("    FAIL: %s\n", TensorRT_GetLastError());
        return 1;
    }
    printf("    logits     = %.6f %.6f %.6f\n", logits[0], logits[1], logits[2]);
    printf("    act_logits = %.6f %.6f\n", actLogits[0], actLogits[1]);

    printf("[7] second run (cache path)\n"); fflush(stdout);
    logits.assign(3, 0.f);
    if (TensorRT_RunInferenceMultiIO(ctx, 5, inputs, 2, outputs) != 0) {
        printf("    FAIL: %s\n", TensorRT_GetLastError());
        return 1;
    }
    printf("    logits     = %.6f %.6f %.6f\n", logits[0], logits[1], logits[2]);

    printf("[8] teardown\n"); fflush(stdout);
    TensorRT_DestroyContext(ctx);
    TensorRT_UnloadEngine(engine);
    printf("ALL OK\n");
    return 0;
}
