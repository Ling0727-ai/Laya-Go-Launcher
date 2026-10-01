# ONNX Runtime 本地目录

此目录只提交本说明和各 provider 目录的 `.gitkeep`，不提交运行时二进制。
按需要安装一种 provider 即可，不必把四种包都下载下来。

获取来源、版本、模型准备和完整操作步骤见 [本地依赖与模型准备](../../docs/local-assets.md)。

解压后保留包内结构，程序支持以下布局：

```text
Assets/onnx/
  cuda12/<包目录>/lib/onnxruntime.dll
  cuda13/<包目录>/lib/onnxruntime.dll
  directml/<包目录>/runtimes/win-x64/native/onnxruntime.dll
  cpu/<包目录>/lib/onnxruntime.dll
```

CUDA 包还需要同目录的 `onnxruntime_providers_shared.dll` 和
`onnxruntime_providers_cuda.dll`，以及匹配的 CUDA/cuDNN 环境。
DirectML 包需要保留同目录的 `DirectML.dll`。请解压完整包，不要只复制主 DLL，
并保留供应商的许可证和第三方声明。
