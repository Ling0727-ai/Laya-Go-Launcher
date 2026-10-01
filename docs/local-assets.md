# 本地依赖与模型准备

Git 仓库不包含模型权重、ONNX Runtime 发布包、TensorRT engine、本地配置或构建产物。
`.gitkeep` 只用于保留目录，不代表依赖已经安装。以下命令默认在仓库根目录执行。

## 需要准备什么

| 位置 | 内容 | 何时需要 | 获取方式 |
|---|---|---|---|
| `Assets/onnx/cuda12/` 或 `cuda13/` | 对应 CUDA 版本的 ONNX Runtime 完整包 | ONNX CUDA 推理 | 项目 Release 或 Microsoft 官方包 |
| `Assets/onnx/directml/` | ONNX Runtime DirectML 完整包 | Windows DirectML 推理 | 项目 Release 或 NuGet |
| `Assets/onnx/cpu/` | ONNX Runtime CPU 完整包 | 仅 CPU 推理 | Microsoft 官方包 |
| `onnx/` | `.onnx` 及其引用的外部权重文件（如 `.data`） | ONNX 推理，或构建 TensorRT engine | 项目 Release 或自行导出 |
| `engines/`、`engines-fp32/` | `.engine`、构建清单和日志 | TensorRT 推理；FP32 目录仅用于对比 | 在目标机器上生成 |
| Hugging Face 缓存，或自行配置的目录 | `tokenizer/tokenizer.json`、匹配的 `rl_agent_config.json` | 模型推理 | 从同一 checkpoint 下载 |
| `layatrt.config.json` | 当前机器的模型、分词器、运行时路径 | 需要覆盖自动发现时 | 复制配置模板 |
| `build/bin/Release/` | `layatrt_onnx.dll`、`qualityscaler_tensorrt.dll` | 对应原生后端 | `build.ps1` 构建 |
| GUI 的 `frontend/node_modules/`、`frontend/wailsjs/`、`frontend/dist/` 和 `build/bin/` | npm 依赖、Wails 绑定、前端及桌面产物 | 开发或构建 GUI | npm 和 Wails 生成 |

无需恢复 `.idea/`、`.vscode/`、`_scratch_verify/`、`release-staging/`、临时日志或
`bench/*.json`。这些是个人设置、实验结果或打包输出，不是启动依赖。

## 1. ONNX Runtime

优先查看 [项目 Releases](https://github.com/Ling0727-ai/Laya-Go-Launcher/releases)。
如果对应版本发布了 `onnx-runtime-windows-x64.zip`，先对照该 Release 的
`SHA256SUMS.txt` 校验，再按包内结构解压到仓库根目录：

```powershell
Get-FileHash .\onnx-runtime-windows-x64.zip -Algorithm SHA256
Expand-Archive .\onnx-runtime-windows-x64.zip -DestinationPath .
```

校验命令只打印哈希，仍需与发布的校验值比对。未发布该资产时，可从供应商获取：

- CUDA / CPU：[Microsoft ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)。选择 Windows x64 包；CUDA 包必须与本机 CUDA/cuDNN 版本相匹配，不能仅把 CUDA 12 包改名放进 CUDA 13 目录。
- DirectML：[Microsoft.ML.OnnxRuntime.DirectML](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML)。下载 `.nupkg`，将副本改为 `.zip` 后完整解压。
- CUDA 依赖和版本矩阵：[CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。

项目已验证 CUDA ONNX Runtime `1.28.0`、DirectML `1.24.4`，见
[依赖版本记录](dependencies.md)。版本记录不保证供应商仍提供相同文件名；按官方说明选择匹配构建。

推荐布局：

```text
Assets/onnx/cuda12/<包目录>/lib/onnxruntime.dll
Assets/onnx/cuda13/<包目录>/lib/onnxruntime.dll
Assets/onnx/directml/<包目录>/runtimes/win-x64/native/onnxruntime.dll
Assets/onnx/cpu/<包目录>/lib/onnxruntime.dll
```

保留完整包及许可证。CUDA 包的 provider DLL、DirectML 包的 `DirectML.dll`
不能丢失。CPU 推理不需要 CUDA/TensorRT。
包放在仓库外时，`LAYA_TRT_ONNX_ROOT` 应指向**包含 `Assets/onnx/` 的父目录**；
也可在本地配置中用 `onnx_runtime_path` 指定主 DLL 的完整路径。

## 2. 模型、分词器和校准配置

模型来源为 [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)。
仅运行现有 ONNX/engine 时，不需要下载原始 PyTorch 权重，也不需要安装 Python laya SDK；
但仍需要分词器和与导出 checkpoint 匹配的校准配置。

```powershell
# 安装 Hugging Face CLI，然后下载到默认缓存，启动器会自动发现
python -m pip install huggingface_hub
hf download convaiinnovations/laya --include "tokenizer/*" "rl_agent_config.json"
```

以上下载根目录 checkpoint 的配置。若模型来自其他 subfolder/revision，应获取相同
checkpoint 的配置，并显式设置 `model_config_path`。不要混用 English 和 multilingual
checkpoint 的配置，否则校准温度、训练长度等参数可能不匹配。
若使用非默认 Hugging Face 缓存位置，也应显式配置 `tokenizer_path` 和 `model_config_path`。

从项目 Release 下载 ONNX 压缩包（如 `laya-models-onnx.zip`）或单独的
`laya_ctx8192*.onnx`，放入 `onnx/`。若 ONNX 引用了外部权重，必须一起下载并保持
相对路径，仅有 `.onnx` 文件不一定足够。可用图包括：

- `laya_ctx8192.onnx`：普通 FP32 图，适合 CPU，亦可作为 TensorRT 构建源。
- `laya_ctx8192.opt.onnx`：ONNX Runtime 优化图。
- `laya_ctx8192.opt.fp16.onnx`：ONNX Runtime FP16 优化图，主要用于 GPU。

若 Release 尚未提供模型，先按上游 laya SDK 的安装说明准备 Python 环境
（须能 `import laya`，且安装其匹配的 PyTorch/Transformers 依赖），再执行：

```powershell
python -m pip install onnx "onnxruntime>=1.28" numpy sympy
python tools\export_onnx.py --model convaiinnovations/laya --out onnx\laya_ctx8192.onnx --seq-len 512 --markers 32
# 可选：生成 ONNX Runtime 优化版本
python tools\optimize_onnx.py --src onnx\laya_ctx8192.onnx --out onnx\laya_ctx8192.opt.onnx
python tools\optimize_onnx.py --src onnx\laya_ctx8192.onnx --out onnx\laya_ctx8192.opt.fp16.onnx --fp16
```

导出图使用动态长度，`--seq-len` 是追踪长度，并不表示权重训练到了 8192 token。
文件名中的上下文长度也不是验证结果；使用时遵守模型训练范围和内存限制。
导出脚本默认验证跨长度一致性，优化脚本默认验证原图与优化图，不应跳过验证。

## 3. TensorRT engine（可选）

从 [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt/downloads) 获取 SDK，
从 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 获取匹配的 CUDA 环境。
本项目验证版本为 TensorRT `10.16`。engine 与 GPU、TensorRT 版本有关，优先在目标机器构建：

```powershell
pwsh -File bench\build-engines.ps1 -Onnx onnx\laya_ctx8192.onnx -OutDir engines -Presets '512,1024' -Precision fp16
# 可选：精度对比用的 FP32 engine
pwsh -File bench\build-engines.ps1 -Onnx onnx\laya_ctx8192.onnx -OutDir engines-fp32 -Presets '512' -Precision fp32
```

必须显式传 `-Onnx`，脚本默认路径是作者本机的实验路径。
TensorRT 构建源使用普通 ONNX，不能用含 ORT 融合算子的 `*.opt.onnx` 或 `*.opt.fp16.onnx`。
也可以让服务在依赖齐全、发现普通 ONNX 且开启自动转换时后台生成 engine。

## 4. 本地配置与生成文件

```powershell
Copy-Item .\layatrt.config.example.json .\layatrt.config.json
```

在本地配置中按需设置 `engine_path`、`tokenizer_path`、`model_config_path` 和运行时路径。
本地配置、令牌及私有路径不提交 Git。

原生桥接库由 `build.ps1` 生成；该脚本构建两个后端，要求 TensorRT/CUDA SDK、
Visual Studio C++ Build Tools 和 CMake。ONNX C API 头文件已在
`third_party/onnxruntime/` 入库，无需另行下载。当前 CMake 配置和 `build.ps1`
都会先检查 TensorRT/CUDA SDK，不能通过现有脚本在缺少这些 SDK 时仅构建 ONNX 桥。
这是构建要求，不代表 CPU/DirectML 推理运行时也需要这些 SDK。

GUI 依赖由前端目录的 `npm ci` 恢复；Wails 在 `wails dev` / `wails build` 时生成
`wailsjs/` 绑定并执行前端构建。根目录 `dev.ps1` 封装开发入口。
空 `.gitkeep` 不能代替绑定、`dist/` 文件或 DLL，因此这些构建目录不添加占位。
完成准备后，可先构建并运行不依赖 GPU DLL 的诊断工具：

```powershell
go build -o doctor.exe ./cmd/layatrt-doctor
.\doctor.exe
```

第三方包和模型分别受各自许可证约束，参见 [依赖与许可证](dependencies.md)。
