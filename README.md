# Laya Go Launcher

**Laya Go Launcher** 是 laya 模型的加载与推理启动层：负责发现模型、选择 TensorRT/ONNX Runtime 后端，并通过桌面 GUI 或 HTTP API 启动推理服务。

它不复刻 laya 模型本身，而是为模型提供统一的本地运行入口。

> 研究发布说明：本项目源码采用 `Laya Go Launcher Research and Non-Commercial License v1.0`。未经作者书面许可，不得将本项目或其衍生代码用于商业用途、重新打包分发或移除署名。模型文件、ONNX Runtime、CUDA、TensorRT 和其他第三方组件不自动继承本项目许可证，详见 [LICENSE](LICENSE)、[NOTICE.md](NOTICE.md) 和 [docs/dependencies.md](docs/dependencies.md)。

## 发布内容

源码仓库只包含程序源码、构建脚本、测试和配置模板。由于 GitHub 普通 Git 提交不适合数 GB 模型和运行时 DLL，预构建依赖与模型通过 GitHub Releases 提供：

- `onnx-runtime-windows-x64.zip`：`Assets/onnx` 下的 CUDA 12、CUDA 13 和 DirectML 运行时 DLL
- `laya-models-onnx.zip`：作者导出的 ONNX 模型
- `laya-engines-windows-x64.zip`：可选的 TensorRT engine；engine 与 GPU、TensorRT 版本相关，优先自行构建

下载 Release 资产后，按其压缩包内的目录结构解压到仓库根目录，或使用 `LAYA_TRT_ONNX_ROOT` 指向包含 `Assets/onnx` 的目录。每个 Release 都附带 SHA-256 校验文件；下载后请先校验再运行。

## 性能对比：本项目 vs 原版 laya（PyTorch）

同一台机器、同一套权重、同一组用例、同一组问题，只换推理引擎。

**测试环境**

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU（12227 MiB） |
| 驱动 / CUDA | `616.92` / CUDA 13.x（`torch 2.14.0+cu130`） |
| 原版 | PyTorch `2.14.0` eager 前向 + AMP fp16，`transformers 5.17.0` |
| 本项目 | Go `1.27.0`、TensorRT `10.16`、ONNX Runtime `1.28.0`（CUDA）/ `1.24.4`（DirectML） |
| 测量时间 | 2026-09-29 ～ 09-30 |

用例是 `bench/autobench` 与 `laya/research/scripts/bench_auto.py` **共用**的 4 条英文工单（`en_billing`、`en_technical`、`en_sales`、`en_hr`，输入约 121–131 token），问题集相同（`dept` 四选一 + `refund` noul）。Go 侧每个用例 `-reps 100`，Python 侧 `--reps 50`，各自先预热 5 次。

### 短输入（约 120–130 token）

| 推理后端 | p50 | p95 | p99 | 相对原版 |
|---|---:|---:|---:|---:|
| 原版 laya · PyTorch eager + AMP fp16 | 23.6 ms | 26.4 ms | 27.5 ms | 1.0× |
| 本项目 · TensorRT 10.16 fp16 | **4.1 ms** | 4.8 ms | 5.0 ms | **5.8× 快** |
| 本项目 · ONNX Runtime CUDA fp16 | **8.1 ms** | 13.2 ms | 14.4 ms | **2.9× 快** |
| 本项目 · ONNX Runtime DirectML | 96.0 ms | 110.9 ms | 122.1 ms | 4.1× 慢 |
| 本项目 · ONNX Runtime CPU | 1032 ms | 1180 ms | 1667 ms | 44× 慢 |

表内是 4 条用例各自分位数的平均。TensorRT 用的是 `engines\laya_s8192_fp16_p2.engine`，ONNX 用的是 `onnx\laya_ctx8192.opt.fp16.onnx`（CPU 档用未融合的 `laya_ctx8192.onnx`，fp16 在 CPU provider 上更慢）。

### 长输入（1024 token）

| 推理后端 | p50 | 说明 |
|---|---:|---|
| 原版 laya · PyTorch | 36.7 ms | ⚠️ 只能跑 multilingual checkpoint，不是同一套权重 |
| 本项目 · TensorRT fp16 | **17.8 ms** | english 权重，超出其训练值 512 |
| 本项目 · ONNX Runtime CUDA fp16 | 28.7 ms | 同上 |
| 本项目 · ONNX Runtime DirectML | 208.9 ms | 同上 |
| 本项目 · ONNX Runtime CPU | 7802.8 ms | 同上 |

原版 english checkpoint 的 `max_len=512`，PyTorch 侧对 1024 token 直接 `SKIP above max_len 512`；这一行的原版数字来自 multilingual checkpoint，所以**长输入这组不是严格的同权重对比**，只说明量级。本项目按[上下文与选项上限](#上下文与选项上限)刻意允许外推到训练值之外，这是原版路径做不到的。

### 结论

- **差距主要来自推理引擎，不是模型**：TensorRT 比 PyTorch eager 快约 5.8×，ONNX Runtime CUDA 快约 2.9×。
- **长度越长，优势越小**：1024 token 时计算本身占主导，TensorRT 只快约 2.1×（PyTorch 那侧本来也只从 23.6 ms 涨到 36.7 ms）。
- **DirectML 和 CPU 档比原版慢**，它们不是性能选项，而是没有 TensorRT/CUDA 时的可用性回退。
- **决策一致**：四种后端在 5 条用例上全部 PASS，原版在 4 条用例上全部 PASS，选中的部门和退款判断相同；fp16 下概率有 ±0.01 量级漂移（见 [bench/REPORT.md](bench/REPORT.md) §5）。
- **加载更快**：engine 反序列化 1.5–3.7 s，而原版 preload 一个 checkpoint 用了 16.2 s（含 HuggingFace 缓存校验，两者不是完全等价的操作）。

### 复现

```powershell
# 原版（在 laya 仓库内）
uv run python research/scripts/bench_auto.py --device cuda --amp fp16 --models english --reps 50
uv run python research/scripts/bench_auto.py --device cuda --amp fp16 --tokens 128,512,1024 --token-only

# 本项目（仓库根目录，四种后端分别跑）
go run ./bench/autobench -reps 100 -log bench\autobench.log -engines onnx-cpu
go run ./bench/autobench -reps 100 -log bench\autobench.log -engines onnx-directml
go run ./bench/autobench -reps 100 -log bench\autobench.log -engines onnx
go run ./bench/autobench -reps 100 -log bench\autobench.log -engines trt
```

读数时注意：

- 两侧都是单次测量（Go 每用例 100 次、Python 每用例 50 次呼叫），没有跨会话重复。笔记本 GPU 受功耗和温度影响，**绝对毫秒数会变，倍数关系比绝对值稳**。
- Python 侧是 eager 前向，没有 `torch.compile`、没有 TensorRT 执行后端。这组数字说明的是「launcher 自带的编译/融合路径」与「原版默认路径」的差距，不是「laya 模型慢」。
- Python 侧 `--reps 50` 的 p99 不稳定，脚本自己也会提示 `P99 needs >= 100`；用 p50 比较更可靠。
- 两侧 `usage.input_tokens` 在同一条用例上相差 2（本项目把 dict state 序列化成紧凑 JSON，原版 `serialize_state` 保留 `": "` 里的空格，每条问题行差 1 token）。这没有改变任何用例的判定结果，但确实是与原版参考实现的差异：分词器和序列构造的 parity 测试只覆盖已序列化好的字符串，不覆盖这一步。

## 许可证与署名

本项目不是 MIT/Apache 等允许任意再分发的开源许可证。源代码仅授权用于个人学习、学术研究和内部非商业评估；发表论文、报告或基于本项目的研究成果时，必须明确引用本项目并保留作者与仓库链接。任何商业部署、SaaS、产品集成、镜像分发或衍生发行版，都需要事先取得作者书面许可。

请勿将模型权重、导出的 ONNX 文件或第三方 DLL 视为本项目源码的一部分。它们分别受模型来源、ONNX Runtime、NVIDIA CUDA/TensorRT、Microsoft DirectML 等各自许可证约束。

## 依赖与环境

Windows x64 是当前支持平台。完整 TensorRT 后端需要：

- Go `1.27` 或更高版本
- Node.js/npm，以及 Wails v2（构建桌面 GUI）
- Visual Studio 2022 C++ Build Tools、CMake `4.4+`
- NVIDIA 驱动、CUDA Toolkit 和 TensorRT `10.16`（TensorRT 后端）
- ONNX Runtime `1.28.0` CUDA 12/13 包，或 DirectML `1.24.4` 包（ONNX 后端）
- laya 模型的 tokenizer、配置和 ONNX/engine 文件

仅运行 `layatrt-doctor` 或编译不依赖本机 CUDA 的 Go 包时，不需要安装完整 GPU 环境。第三方依赖的来源、版本和许可证见 [docs/dependencies.md](docs/dependencies.md)。

## 配置

复制配置模板并按机器路径调整：

```powershell
Copy-Item .\layatrt.config.example.json .\layatrt.config.json
```

常用配置项：

- `engine_path`：`.engine` 或 `.onnx` 模型路径
- `backend`：`auto`、`tensorrt` 或 `onnx`
- `provider`：`cuda`、`directml` 或 `cpu`
- `onnx_runtime_path`：固定 ONNX Runtime DLL；需要切换 provider 时建议留空
- `onnx_bridge_path`：`layatrt_onnx.dll` 路径
- `tokenizer_path`、`model_config_path`：tokenizer 和模型配置路径

对应环境变量为 `LAYA_TRT_*`。这些变量名属于兼容配置接口，项目展示名变更后仍保持不变。不要把包含真实路径、令牌或私有模型位置的配置文件提交到仓库。

## 运行

准备依赖后，先执行环境诊断：

```powershell
. .\env.ps1
.\cmd\layatrt-doctor.exe
```

启动 HTTP 服务：

```powershell
.\run.ps1 -Server -Engine .\onnx\laya_ctx8192.opt.fp16.onnx
```

启动桌面 GUI：

```powershell
.\run.ps1 -Gui
```

开发模式：

```powershell
. .\dev.ps1
```

服务默认监听 `http://127.0.0.1:8420`，API 文档见 [docs/api.md](docs/api.md)。

## 从源码构建

```powershell
go test ./internal/... ./cmd/...
cd cmd\layatrt-gui\frontend
npm ci
npm test
npm run build
cd ..\..
.\build.ps1
```

构建 TensorRT/ONNX 原生桥接前，请确认 `CMakeLists.txt` 能找到对应 SDK。不要将生成的 `.dll`、`.engine`、`.onnx` 和本地配置提交到源码仓库。

## 论文引用

推荐使用仓库中的 [CITATION.cff](CITATION.cff)。正式论文发布后，请补充 DOI、版本号和实验数据对应的 Release 标签。研究复现实验应记录：GPU、驱动、CUDA、TensorRT、ONNX Runtime、模型 SHA-256 和本项目 Release 版本。

## 贡献与安全

未经作者确认，不接受将第三方权重、私有数据、受限 SDK 或未核验二进制直接提交到仓库的 Pull Request。安全问题和潜在许可证问题请私下联系作者，不要公开发布可复现漏洞细节。

## 原始运行说明

先编译一次内核（只需一次）：

```powershell
.\build.ps1
```

然后直接跑命令，不需要设 `$env:PATH`。程序启动时自己找内核 DLL（程序旁边或仓库的 `build\bin\Release`）、TensorRT（`C:\TensorRT-*`）、CUDA（`C:\CUDA`、`CUDA_PATH`、`Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*`）和 cuDNN（`Program Files\NVIDIA\CUDNN\v*\bin\<CUDA 版本>\x64`），并把它们放到本进程 PATH 的最前面。启动日志的 `native:` 一行会写出找到的目录。装在别处时才需要 `LAYA_TRT_KERNEL_DIR` / `TENSORRT_ROOT` / `CUDA_ROOT` / `CUDNN_PATH`。

### 纯 server 版本

```powershell
go build -o layatrt-server.exe ./cmd/layatrt-server   # 不链接 Wails
.\layatrt-server.exe                                  # 控制台 http://127.0.0.1:8420/
```

不带参数启动时，它会找能容纳 8192 token 的模型，按 `tensorrt → onnx-cuda → onnx-directml → onnx-cpu` 的顺序逐级加载。如果没有匹配的 TensorRT plan，就在后台用 trtexec 从 ONNX 编译一个，编译完成后自动切换过去。基础配置可以在控制台里改（保存到 `layatrt.config.json`），也可以用 `--seq`、`--fallback`、`--backend`、`--no-convert` 等参数或 `LAYA_TRT_*` 环境变量覆盖。给其他 AI 调用的说明在 [`skills/laya-server/SKILL.md`](skills/laya-server/SKILL.md)。

### 开发模式（wails dev）

```pwsh
$env:CGO_ENABLED = "1"
cd cmd\layatrt-gui
wails dev
```

打开 `http://localhost:5188`（前端）和 `http://localhost:34116`（在浏览器里调试绑定）。

端口可以改，避免和同机其他项目撞：

```pwsh
$env:LAYATRT_DEV_PORT = "5189"      # 前端
wails dev -devserver localhost:34117
```

### 生产模式（桌面端）

```pwsh
.\cmd\layatrt-gui\build\bin\layatrt-gui.exe
```

### 无头服务

```pwsh
$env:CGO_ENABLED = "1"
go run ./cmd/layatrt-server --engine C:\models\laya_1k.engine --addr 127.0.0.1:8420
```

### 只想跑前端

```pwsh
cd cmd\layatrt-gui\frontend
npm install
npm run dev          # http://localhost:5188
```

---

### 端口约定

刻意避开常见默认值，因为一台开发机上通常已经有别的项目占着：

| 用途 | 端口 | 为什么不用默认 |
|---|---|---|
| 前端 dev server | **5188** | Vite 默认 5173，几乎总被别的 Vite 占 |
| Wails dev server | **34116** | Wails 默认 34115，同类项目会撞 |
| HTTP API | **8420** | 与前端/DLL 无关，被占时自动往后找 |

`5188` 和 `34116` 都是 `strictPort`：**被占用时直接报错，不会偷偷换端口**——否则文档里写的地址就是错的。要换就用上面的环境变量和参数。

---

### 这些环境变量为什么必需

| 变量 | 为什么 |
|---|---|
| `PATH` += 内核 DLL 目录 | `qualityscaler_tensorrt.dll` 是**加载期**依赖，缺了进程在 `main()` 之前就退出；单纯找不到 DLL 通常是 `0xC0000135`，`0xC0000279` 还需检查 PATH 顺序（见故障排查） |
| `PATH` += TensorRT `bin` | `nvinfer_10.dll` 同上 |
| `PATH` += CUDA `bin\x64` | `cudart64_*.dll` 同上；注意在 `bin\x64`，不是 `bin` |
| `CGO_ENABLED=1` | `internal/kernel` 是 cgo 包；关掉时 `go build` 只报 `build constraints exclude all Go files`，看不出原因 |

一次性写进当前会话后，这个终端里后面所有命令都能跑。想固化到系统：

```powershell
.\setup.ps1
```

它做的事就是上面的 PATH 加上编译，外加一次自检。

### 不想手打就用脚本

`run.ps1` / `dev.ps1` 把上面这些包起来了，效果一样：

```powershell
.\run.ps1              # = 设 PATH + 启动桌面端
.\run.ps1 -Server      # = 设 PATH + 启动 HTTP API
.\dev.ps1              # = 设 PATH + wails dev
```

---

> **先跑 `.\doctor.exe`。** 它不链接内核，所以即使 DLL 缺失也能起来，逐项告诉你缺什么、装什么。

---

## 目录

- [启动命令](#启动命令)
- [内核切换](#内核切换)
- [准备模型](#准备模型)
- [使用](#使用)
- [配置](#配置)
- [上下文与选项上限](#上下文与选项上限)
- [从源码构建](#从源码构建)
- [故障排查](#故障排查)
- [架构](#架构)
- [测试](#测试)

---

## 内核切换

推理内核有两种，可以切换：

| | TensorRT | ONNX Runtime |
|---|---|---|
| 模型文件 | `.engine` plan | `.onnx` 图 |
| 依赖 | TensorRT 10.x + CUDA + N 卡 | `onnxruntime.dll` + 桥接库 |
| 速度 | 快（GPU，融合算子） | 慢一些，但能在没有 TensorRT 的机器上跑 |

**默认是 `auto`，含义就是 TensorRT**——和加这个选项之前的行为一致，所以老的
`layatrt.config.json` 不用改。选哪个内核由**文件格式**决定：`.engine` 只能由
TensorRT 读，`.onnx` 只能由 ONNX Runtime 读。

GUI 的「模型」面板可手动选择 **TensorRT / ONNX CUDA / ONNX DirectML / ONNX CPU**，也可保持「自动」。同一 `.onnx` 图在 ONNX 模式之间切换会重新加载 session；TensorRT 需要另选从该图编译的 `.engine` 文件。加载成功后以「当前运行」中的实际设备为准。

ONNX CUDA 和 DirectML 使用各自的 `onnxruntime.dll` 发布包，不是给同一个 CPU 版 DLL 换个选项：分别把 CUDA 版放在 `Assets/onnx/cuda12` 或 `Assets/onnx/cuda13`、DirectML 版放在 `Assets/onnx/directml`（连同各自的依赖），或设置 `LAYA_TRT_ONNX_ROOT` 指向已有的、包含这些 `Assets/onnx/...` 子目录的目录，例如 `$env:LAYA_TRT_ONNX_ROOT = 'C:\Users\lingxin\Downloads\QualityScaler-go'`。配置中的单个 `onnx_runtime_path` 适合固定使用一种 ONNX 发布包；需要在 CUDA / DirectML 间切换时，留空并使用上述目录发现。显式选 GPU provider 后若不可用，加载报错，不再悄悄退到 CPU；未指定 provider 的旧自动路径仍可回退。本机已用 `laya_dyn.onnx` 实测 `ONNX CUDA (ORT 1.28.0) → ONNX DirectML (ORT 1.24.4) → TensorRT`，同一请求三个引擎均选中 `billing`。

四种指定方式，后者被前者覆盖：配置文件 → 环境变量 → 命令行 → 单次请求。

```powershell
# 命令行
.\layatrt-server.exe -engine laya_dyn.onnx -backend onnx -provider cpu

# 运行时切换：同一个进程里从 ONNX 换成 TensorRT，不用重启
curl -X POST http://127.0.0.1:8420/api/v1/engine/load `
  -H 'Content-Type: application/json' `
  -d '{ "path": "laya_1k.engine", "backend": "tensorrt" }'
```

```json
// layatrt.config.json
{
  "backend": "auto",
  "provider": "cuda",
  "onnx_runtime_path": "",
  "onnx_bridge_path": ""
}
```

环境变量：`LAYA_TRT_BACKEND`、`LAYA_TRT_PROVIDER`、`LAYA_TRT_ONNX_RUNTIME`、
`LAYA_TRT_ONNX_BRIDGE`。

**显式指定就不会被替换**：写了 `-backend tensorrt` 却加载不了，会直接报错，而不是
悄悄跑到 CPU 上——那样看起来像成功了，实际只是变慢，最难查。只有 `auto` 会回退，
并且会在 `backend_note` 里说明。

诊断入口：`GET /api/v1/backends` 报告两个内核是否可用、各自版本；`layatrt-doctor`
也会检查。完整说明见 [docs/kernels.md](docs/kernels.md)。

---

## 启动命令

所有命令都在**仓库根目录**执行。全部实测可用。

### 从零到跑起来

```powershell
cd C:\Users\lingxin\Documents\laya-trt

# 1. 自检 —— 缺什么会直接告诉你装什么
.\doctor.exe

# 2. 准备 —— 编译内核 DLL + Go 二进制
.\setup.ps1

# 3. 启动
.\run.ps1
```

### 日常启动

```powershell
# 桌面端（推荐：会自检 → 设 PATH → 开窗口）
.\run.ps1

# 只启动 HTTP API，不开窗口
.\run.ps1 -Server

# 指定 engine
.\run.ps1 -Server -Engine C:\models\laya_1k.engine

# 局域网可访问
.\run.ps1 -Server -Addr 0.0.0.0:8420

# 确认环境没问题时跳过自检，启动更快
.\run.ps1 -NoCheck
```

### 环境自检

```powershell
.\doctor.exe              # 完整报告：每项缺什么、装什么
.\doctor.exe --json       # 机器可读，给脚本用
.\doctor.exe --quiet      # 只看退出码：0 = 就绪，1 = 未就绪
```

`--quiet` 适合放进 CI 或启动脚本。注意用 `$LASTEXITCODE` 判断，不要用 `if (& ...)`——`--quiet` 没有输出，PowerShell 会把它当空值：

```powershell
.\doctor.exe --quiet
if ($LASTEXITCODE -ne 0) { Write-Error "环境未就绪"; exit 1 }
```

要结构化结果就用 `--json`：

```powershell
$report = .\doctor.exe --json | ConvertFrom-Json
if (-not $report.ready) {
    $report.checks | Where-Object { -not $_.ok -and $_.required } |
        ForEach-Object { Write-Host "$($_.name): $($_.fix)" }
}
```

### 编译

```powershell
.\build.ps1                # 只编译内核 DLL
.\setup.ps1                # 内核 + Go 二进制
.\setup.ps1 -SkipKernel    # 只编译 Go 部分
.\setup.ps1 -Check         # 只自检，不编译
```

手工编译 Go 部分时**必须开 CGO**（内核是 cgo 包）：

```powershell
$env:CGO_ENABLED = '1'
go build ./...
```

### 前端 / 开发模式

```powershell
# 桌面端开发模式：Vite 热重载 + Wails 窗口
.\dev.ps1

# 前端结构不变时可跳过绑定生成，启动更快
.\dev.ps1 -SkipBindings

# 只构建前端
cd cmd\layatrt-gui\frontend
npm install
npm run build
npm run lint:struct      # 结构规范检查
```

### 手工启动（不用脚本）

```powershell
# 关键是先设 PATH，否则进程会在 main() 之前崩掉
. .\env.ps1

.\cmd\layatrt-gui\build\bin\layatrt-gui.exe     # 桌面端

.\layatrt-server.exe --engine C:\models\laya_1k.engine --addr 127.0.0.1:8420
```

`. .\env.ps1` 前面的**点不能省**——那是 PowerShell 的点源语法，不加只影响子进程、当前会话无效。

### 编译 engine

```powershell
# 1. 导出 ONNX（脚本会验证跨长度一致性）
python tools\export_onnx.py --out laya_dyn.onnx --seq-len 512 --markers 32

# 2. 按预设编译（推荐：一次编出四档，含动态 batch）
pwsh -File bench\build-engines.ps1 -Onnx laya_dyn.onnx

# 只要一档
pwsh -File bench\build-engines.ps1 -Presets '512' -Onnx laya_dyn.onnx
```

预设的选择依据见[该选哪一档 engine](#该选哪一档-engine)。手工编译时必须让 batch 维是动态的
（`--minShapes`/`--optShapes`/`--maxShapes` 的第一维不要都写 1），否则多个问题会退化成每个问题一次前向：

```powershell
$TRT = "C:\TensorRT-10.16.0.72\bin"
& "$TRT\trtexec.exe" --onnx=laya_dyn.onnx --saveEngine=laya_s512.engine --fp16 `
  --minShapes=input_ids:1x64,attention_mask:1x64,marker_pos:1x2,marker_mask:1x2,qtype:1 `
  --optShapes=input_ids:2x256,attention_mask:2x256,marker_pos:2x8,marker_mask:2x8,qtype:2 `
  --maxShapes=input_ids:8x512,attention_mask:8x512,marker_pos:8x64,marker_mask:8x64,qtype:8 `
  --builderOptimizationLevel=1 --memPoolSize=workspace:6000
```

### 测试

```powershell
# 与 Python 官方实现逐字节对齐（需要 laya SDK 及其 venv）
go test ./internal/tokenizer/ ./internal/sequence/ -v

# 推理链路（需要 engine）
go test ./internal/inference/ -v

# 全部
go test ./internal/...

# 内核直连，绕开 Go
cmd /c tests\build_and_run.bat

# 前端结构
cd cmd\layatrt-gui\frontend; npm run lint:struct
```

### 验证服务在跑

```powershell
curl http://127.0.0.1:8420/api/v1/health
curl http://127.0.0.1:8420/api/v1/limits      # 引擎能接受的范围
curl http://127.0.0.1:8420/api/v1/metrics     # 计数与延迟分位
```

### 基准与对比

`bench\` 下的工具都是只读的：它们只调用 API 和读取 engine，不启动、不停止、不重配任何服务。

```powershell
# engine 声明的 IO、profile 范围、激活内存、能放几个 context
go run ./bench\engprobe -engine engines\laya_s512_fp16_b8.engine

# 延迟扫描：序列长度与问题数两个轴，并拟合出固定开销与每 token 成本
go run ./bench\apibench -base http://127.0.0.1:8471/api/v1 -reps 12

# 两个 engine 的裸输出差异与速度（判断精度是否可接受）
go run ./bench\enginecmp -a fp16.engine -b fp32.engine -questions 8

# 两个运行中服务的答案差异（用户可见层面）
pwsh -File bench\answer-diff.ps1 -A http://127.0.0.1:8471/api/v1 -B http://127.0.0.1:8472/api/v1

# 在私有端口起一个服务用于基准（不会碰 8420 上的实例）
pwsh -File bench\serve.ps1 -Engine engines\laya_s512_fp16_b8.engine -Port 8471
```

`apibench` 的两个拟合值最有用：**固定开销/次前向**（约 7 ms）和**每 token 边际成本**（约 0.012 ms）。
前者远大于后者，说明这个模型的瓶颈是前向次数而不是序列长度，合批比缩短输入更有效。

`bench/autobench` 是 `laya/research/scripts/bench_auto.py` 的 Go 对应物，两者共用同一组用例和问题集，所以可以直接和原版 PyTorch 比。四档后端的对比数字见上文[性能对比](#性能对比本项目-vs-原版-layapytorch)。

### 端口被占用

`run.ps1` 和服务会自动往后找端口，日志里会写：

```
http api: 127.0.0.1:8420 is in use, using 127.0.0.1:8421 instead
```

要固定端口就先腾出来：

```powershell
Get-NetTCPConnection -LocalPort 8420 -State Listen |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

---

## 准备模型

需要两样东西：**tokenizer**（自动找）和 **engine**（必须自己编译）。

### tokenizer

启动时自动在 HuggingFace 缓存里找：

```
%USERPROFILE%\.cache\huggingface\hub\models--convaiinnovations--laya\snapshots\*\tokenizer\tokenizer.json
```

没有就下载：

```powershell
huggingface-cli download convaiinnovations/laya --include "tokenizer/*" "rl_agent_config.json"
```

`rl_agent_config.json` 也要——它带着**校准温度**。少了它概率就不准（见[故障排查](#概率看起来不对)）。

### engine

两步：导出 ONNX，再编译成 engine。

```powershell
# 1. 导出 ONNX（仓库自带脚本，会自动验证跨长度一致性）
python tools\export_onnx.py --out laya_dyn.onnx --seq-len 512 --markers 32

# 2. 编译 engine
$TRT = "C:\TensorRT-10.16.0.72\bin"
& "$TRT\trtexec.exe" --onnx=laya_dyn.onnx --saveEngine=laya_1k.engine --fp16 `
  --minShapes=input_ids:1x64,attention_mask:1x64,marker_pos:1x2,marker_mask:1x2,qtype:1 `
  --optShapes=input_ids:1x256,attention_mask:1x256,marker_pos:1x8,marker_mask:1x8,qtype:1 `
  --maxShapes=input_ids:1x1024,attention_mask:1x1024,marker_pos:1x32,marker_mask:1x32,qtype:1 `
  --builderOptimizationLevel=1 --memPoolSize=workspace:6000
```

编译约 60 秒。**推荐 `laya_1k`**（1024 上下文 / 32 选项 / 88 MiB 激活 / 22 ms）。

#### 该编多大的 engine

| 场景 | 上下文 | 选项上限 | 激活显存 | 推理 |
|---|---|---|---|---|
| **日常文本**（推荐） | 1024 | 32 | 88 MiB | 22 ms |
| 长文档、多选项 | 8192 | 64 | 4563 MiB | 93 ms |

上下文越大，TensorRT 的激活内存按 profile 上限分配，所以代价不是线性的。按实际需要选，别默认拉满。

> `tools\export_onnx.py` 不是可有可无的辅助脚本。laya 的 head 用 `nn.TransformerEncoder`，PyTorch 导出时会把 trace 长度**固化进 reshape 常量**，导致 engine 只接受一个长度。脚本手写了那部分注意力（用层的原始权重）让形状保持动态，并强制在不同长度上验证。直接 `torch.onnx.export` 出来的图建不出动态 engine。

---

## 使用

### 桌面端

```powershell
.\run.ps1
```

1. 左侧 **模型**：点「浏览并加载…」选 engine（或从扫描到的列表里点一个）
2. 右侧 **决策问题**：写下你的问题，选 `choice` / `score` / `noul`
3. 点「打开文件…」或直接粘贴待分析内容
4. 点「运行」

选项只填内容，标签（`choice` 为 A/B/C…，`score` 为 0/1/2…）按行位置自动生成，增删选项后自动重排，不用手输。结果里每个选项都带标签和完整内容，选中的那项同时显示两者。

界面和 HTTP API 共用同一个引擎和计数器。

### HTTP API

```powershell
.\run.ps1 -Server -Engine C:\models\laya_1k.engine
```

```bash
curl -X POST http://127.0.0.1:8420/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "我们 3 月被重复扣款，请今天内退款，否则取消订阅。",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "应该由哪个部门处理？",
        "criteria": {
          "billing": "发票、付款、退款",
          "technical": "bug、故障",
          "sales": "报价、合同"
        }
      },
      "urgent": {
        "type": "noul",
        "instructions": "是否表达了时间压力或截止日期？"
      }
    }
  }'
```

```json
{
  "model": "laya-rl-agent",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": { "billing": 0.8936, "sales": 0.0675, "technical": 0.0389 },
      "confidence": 0.6279
    },
    "urgent": { "type": "noul", "noul": 0.8916, "confidence": 0.8916 }
  },
  "usage": { "input_tokens": 170, "output_tokens": 0 },
  "timing": { "total_ms": 114.9, "tokenize_ms": 0.5, "inference_ms": 114.4 }
}
```

### 三种问题类型

| type | `criteria` | 返回 | 用途 |
|---|---|---|---|
| `choice` | 对象 `{标签: 描述}` | `choice`、`probabilities` | 分类、路由 |
| `score` | 数组，从低到高 | `score`（期望等级）、`legend` | 紧急度、严重度 |
| `noul` | 省略 | `noul` = P(true) | 是否、有无、风险 |

`confidence` 是归一化熵 `1 - H(p)/log(k)`：单一选项为 `1.0`，均匀分布接近 `0`。用它做自动/人工分流的门槛。

完整端点参考：[`docs/api.md`](docs/api.md)。机器可读：`GET /api/v1/openapi.json`。

> **首次请求会慢。** CUDA 上下文和内核首次加载有一次性开销，之后进入稳态。实测同一请求连续 5 次：68 → 56 → 55 → 27 → **8.5 ms**。要避免用户感知到这次抖动，加载后先打一次预热请求。

---

## 配置

`layatrt.config.json`（模板见 `layatrt.config.example.json`），环境变量可覆盖。

```json
{
  "engine_path": "C:\\models\\laya_1k.engine",
  "model_config_path": "",
  "tokenizer_path": "",
  "http_addr": "127.0.0.1:8420",
  "contexts": 0,
  "max_len": 0,
  "head_max_len": 0,
  "pad_len": 0,
  "diagnostics": false
}
```

| 字段 | 环境变量 | 说明 |
|---|---|---|
| `engine_path` | `LAYA_TRT_ENGINE` | 启动时加载的 engine |
| `model_config_path` | `LAYA_TRT_MODEL_CONFIG` | `rl_agent_config.json`；空则自动找 |
| `tokenizer_path` | `LAYA_TRT_TOKENIZER` | `tokenizer.json`；空则自动找 |
| `http_addr` | `LAYA_TRT_HTTP_ADDR` | 监听地址 |
| `contexts` | | 执行上下文数，0 = 按空闲显存自动 |
| `max_len` / `head_max_len` | | token 预算，**0 = 由 engine 和 checkpoint 决定**（推荐） |
| `pad_len` | | 固定序列长度，0 = 动态 |
| `diagnostics` | | TensorRT 诊断日志 |

**预算留 0 最好。** 引擎和 checkpoint 才是权威，手填容易和它们冲突（这正是最初「跑不了」的原因）。

---

## 上下文与选项上限

三个数含义不同，别混：

| | 值 | 含义 |
|---|---|---|
| **编码器硬上限** | **8192** | `max_position_embeddings`，ModernBERT + RoPE；实测 512→8192 全部可跑 |
| checkpoint 训练值 | 512 / 192 | `rl_agent_config.json` 的 `max_len` / `head_max_len` |
| engine 能力 | 由 profile 决定 | 编译时 `--maxShapes` 指定 |

**默认用训练值**（512），超训练上下文是刻意选择，不该首次运行就静默发生。

`GET /api/v1/limits` 会同时报出三者：

```json
{
  "sequence_min": 64, "sequence_max": 8192,
  "markers_max": 64, "fixed_length": false,
  "trained_max_len": 512, "trained_head_max_len": 192,
  "max_len": 512, "head_max_len": 192, "pad_len": 0
}
```

---

## 从源码构建

需要：Visual Studio 2022 (MSVC)、CMake 3.18+、CUDA Toolkit 12.x/13.x、TensorRT 10.x、Go 1.25+、Node 20+、Wails CLI v2。

```powershell
# 一次性装齐
.\setup.ps1

# 或分步
.\build.ps1                              # 内核 DLL
$env:CGO_ENABLED = '1'; go build ./...   # Go 二进制（内核是 cgo 包，必须开 CGO）
cd cmd\layatrt-gui; wails build          # 桌面端（含前端）
```

Wails CLI：

```powershell
go install github.com/wailsapp/wails/v2/cmd/wails@latest
```

前端结构由 `npm run lint:struct` 强制检查：

```
src/components/App/
  App.tsx  App.data.ts  App.api.ts  App.ts
  AppHeader/       AppHeader.{tsx,data.ts,api.ts,ts}
  EnginePanel/     EnginePanel.{tsx,data.ts,api.ts,ts}
  MetricsPanel/    MetricsPanel.{tsx,data.ts,api.ts,ts}
  PredictPanel/
    PredictPanel.{tsx,data.ts,api.ts,ts}
    QuestionEditor/  QuestionEditor.{tsx,data.ts,api.ts,ts}
src/utils/
  backend/  client.ts  index.ts
  format/   duration.ts  number.ts  tensor.ts  index.ts
```

新建组件用生成器：

```powershell
cd cmd\layatrt-gui\frontend
npm run gen:comp Card/CardHeader
npm run lint:struct
```

---

## 故障排查

先跑 `.\doctor.exe`，它会指出大部分问题。以下是它覆盖不到或需要解释的。

### 进程无输出直接退出（`0xC0000279`）

内核是**加载期**依赖，不是运行期，所以连 `--help` 都会崩，没有任何输出。

**但退出码不能直接读成“DLL 缺失”，两种情况的码不同。** 在本机（Windows 11 + PowerShell 7）实测：

| 退出码 | 含义 | 复现条件 |
|---|---|---|
| `0xC0000135` | `STATUS_DLL_NOT_FOUND`——某个 DLL 找不到 | 内核目录或 TensorRT `bin` **完全不在** PATH 上 |
| `0xC0000279` | `STATUS_ENTRYPOINT_NOT_FOUND`——DLL 找到了，但缺所需导出 | DLL 全都在 PATH 上时**也会**出现（见下） |

关键反例：本机 `PATH` 上本来就有 `nvinfer_10.dll`（TensorRT `bin`）、`cudart64_13.dll`（CUDA `v13.3\bin\x64`），内核目录也在，却仍报 `0xC0000279`。也就是说**“DLL 缺失”这个解释对本机不成立**。

实测到的触发条件是**某个 App Execution Alias 出现在内核目录和 TensorRT `bin` 之前**（`%LOCALAPPDATA%\Microsoft\WindowsApps\<pkg>\*.exe`，0 字节的 reparse point）。把 alias 从 PATH 上删掉后，即使不做任何重排也能正常启动：

```
System32;...                                            → 0xC0000135（真缺 DLL）
TensorRT_bin; <alias>; System32                         → 0xC0000279
<alias>; 内核目录; TensorRT_bin                          → 0xC0000279
内核目录; TensorRT_bin; <alias>                          → 正常
内核目录; TensorRT_bin; CUDA; <alias>; ...               → 正常
```

即：**内核目录和 TensorRT `bin` 都要排在 alias 之前**，删掉 alias 或把它们提前到 alias 前面都可以。连跑 6 次结果一致。**这里只陈述观测到的规律，不声称已定位到加载器内部的确切原因。**

所以修法不是手动补某个 DLL，而是用 `env.ps1` 的固定顺序：

```powershell
. .\env.ps1
```

注意 `.\doctor.exe` 只报各目录/文件是否就位，**不检查 PATH 顺序、也不看 alias**，因此“doctor 全绿却仍崩”是可能的——这时按上表看退出码：是 `0xC0000135` 才去补 DLL，是 `0xC0000279` 就先跑 `env.ps1`。

### `wails dev` 报 `exit status 0xc0000279`

同一个原因，但表现更隐蔽——`wails dev` 生成绑定的方式是**编译并运行**你的程序，所以它在绑定阶段就崩了，看起来像 Wails 配置问题。

```powershell
.\dev.ps1
```

### 概率看起来不对

检查 `rl_agent_config.json` 是否被加载了。启动日志会打印：

```
model config: C:\Users\...\rl_agent_config.json
warning: checkpoint ships temperatures outside the usable range; clamped [choice:11+=0.1006].
```

没有第一行就是没加载到——概率会用温度 1.0 算，与训练校准不符。用 `model_config_path` 指定。

那条 clamp 警告是正常的：checkpoint 里 `choice:11+=0.1006` 会把 logits 放大 10 倍，把 0.24 的概率报成 0.99，所以被限制到 `[0.5, 5.0]`。

### `cgo: C compiler "gcc" not found`

编译时需要 CGO 和 gcc。装了 Visual Studio 不代表能用——`cl.exe` 默认不在 PATH 上。

```powershell
$env:CGO_ENABLED = '1'
# 要么装 mingw-w64 并把 bin 加进 PATH，要么先跑：
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
```

### `build constraints exclude all Go files in internal/kernel`

`CGO_ENABLED=0`。这个包是 cgo，必须开：

```powershell
$env:CGO_ENABLED = '1'
```

### `act_probability` 是 0

fp16 engine 的已知现象。act head 内部算 `p*log(p)`，fp16 下概率下溢到 0 会得到 `0 * -inf = NaN`。**只影响辅助输出 `act_logits`，主决策 `logits` 正常。** 需要这个值就用 fp32 engine。

实测触发条件（本机，`bench/enginecmp`）：**约 50 token 以上的输入就会触发**，
50 token 以下正常。与 batch 大小、问题类型无关：

| 输入长度 | act_logits |
|---|---|
| 49 token | 正常 |
| 75 token 及以上 | NaN |

所以任何真实长度的请求都会命中。要可靠的 `act_probability` 就用 fp32 engine
（`bench\build-engines.ps1 -Presets '512' -Precision fp32`，延迟约为 fp16 的 2.2 倍）。

用 `bench\enginecmp` 可以直接看到哪一侧产生了 NaN：

```powershell
go run ./bench/enginecmp -a engines\laya_s512_fp16_b8.engine `
  -b engines-fp32\laya_s512_fp32_b8.engine -label-a fp16 -label-b fp32
```

### 该选哪一档 engine

TensorRT 按 profile 的 `--maxShapes` 预留**最坏情况**激活内存，所以序列上限直接决定显存占用，而不是实际用了多长。本机实测（RTX 5070 Ti Laptop，12199 MiB）：

| 预设 | seq 上限 | batch 上限 | 激活内存/context | 适用 |
|---|---|---|---|---|
| `laya_s512_fp16_b8` | 512 | 8 | **196 MiB** | 默认。与 checkpoint 的 `max_len=512` 一致 |
| `laya_s1024_fp16_b8` | 1024 | 8 | 672 MiB | 需要一倍余量 |
| `laya_s2048_fp16_b4` | 2048 | 4 | 1664 MiB | 长文档 |
| `laya_s8192_fp16_b1` | 8192 | 1 | 4563 MiB | 极长输入，只能单行 |

一档全编：

```powershell
pwsh -File bench\build-engines.ps1                      # 四档 fp16
pwsh -File bench\build-engines.ps1 -Presets '512' -Precision fp32
pwsh -File bench\build-engines.ps1 -DryRun               # 只打印命令
```

**batch 上限随序列上限递减是必须的**：激活内存同时受两者影响，`8192 × batch 8` 需要约 36 GB，编不出来。

**为什么不要沿用 8192 档**：它每个 context 预留 4563 MiB，而 `AdaptToEngine` 实际只用 512 token，多出的容量任何请求都用不到——两个 context 就要 9.1 GB 显存。序列上限只该覆盖真实负载。

### 换了 engine 之后延迟没变好

先确认 batch 维是动态的：

```powershell
go run ./bench/engprobe -engine engines\laya_s512_fp16_b8.engine
```

如果 `input_ids dim 0` 显示 `1 .. 1`，这个 engine 只接受单行，多个问题会退化成**每个问题一次前向**。本模型单次前向的固定开销约 7 ms，远大于 512 token 的计算量（约 1.2 ms），所以合批与否差别很大：实测 16 个问题从 138 ms 降到 24 ms。

### 输入太长 / 选项太多

```
这段输入需要 112 个 token，但这个 engine 固定只接受 64 个。
问题有 20 个选项，但这个 engine 一次只能评分 3 个。
```

engine 编译时就定死了上限。要么缩短输入，要么按更大的 profile 重新编译。

### 动态 engine 建不出来

```
reshape dims{512 16 64}
```

用 `tools\export_onnx.py` 导出，不要直接 `torch.onnx.export`——原因见[准备模型](#准备模型)。

### engine 反序列化失败

engine 与当前 TensorRT 版本或 GPU 架构不匹配。在本机重新编译。

---

## 架构

```
src/ai_tensorrt.cpp         内核（移植自 QS-go，去图像管线）
include/ai_tensorrt_cpp.h   C ABI
internal/kernel/            cgo 绑定，只调导出函数
internal/engine/            引擎注册表、context 池、内存分配器
internal/tokenizer/         ByteLevel BPE（纯 Go，与 Python 对齐）
internal/sequence/          [CLS] head [MASK] opts [SEP] state [SEP]
internal/inference/         predict 用例、temperature 校准
internal/transport/httpapi/ REST
internal/app/               配置、监听、Wails 绑定
cmd/layatrt-gui/            Wails 桌面端（+ frontend/）
cmd/layatrt-server/         无头服务
cmd/layatrt-doctor/         环境自检（无 cgo，永远能跑）
tools/export_onnx.py        ONNX 导出（含跨长度验证）
tests/                      内核直连测试
```

依赖方向单向：入口 → transport → 用例 → 领域 → kernel。内核不知道 laya 存在。

内存策略移植自 QS-go：context 池在加载时预分配、以 borrow 形式借出；pinned 缓冲每 context 一份、只注册一次；按 size 分桶的 scratch 池和按 shape 分桶的 pair 池都有上限。这些让每次请求的路径上没有分配。

细节见 [`docs/architecture.md`](docs/architecture.md)。

---

## 测试

```powershell
# 与 Python 官方实现逐字节对齐（需要 laya SDK 及其 venv）
go test ./internal/tokenizer/ ./internal/sequence/ -v

# 推理链路（需要 engine）
go test ./internal/inference/ -v

# 内核直连，绕开 Go
cmd /c tests\build_and_run.bat

# 前端结构规范
cd cmd\layatrt-gui\frontend; npm run lint:struct
```

分词器和序列构造的两个 parity 测试最关键：token id 差一个就会给出**自信的错误答案**，所以它们是对照 Python 参考实现验证的，不是自证。

---

## 许可

移植的内核源自 QualityScaler-go。laya 本身为 Apache 2.0。
