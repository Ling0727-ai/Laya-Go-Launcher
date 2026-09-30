# Laya Go Launcher 优化报告

本报告记录本轮性能与显存优化的**实测数据**、**改动清单**和**结论**。所有数字都在本机
（RTX 5070 Ti Laptop，12199 MiB，TensorRT 10.16）测得，工具与原始 JSON 都在 `bench/` 下。

---

## 1. 结论摘要

| 维度 | 改动前 | 改动后 | 说明 |
|---|---|---|---|
| 长文本 tokenize（10 问题） | 26.9 ms | **0.6 ms** | 45× |
| 单个 state 的 tokenize（120 重复句） | 2.67 ms | **0.13 ms** | 21× |
| 16 个问题的推理 | 137.6 ms | **24.1 ms** | 5.7× |
| 每增加一个问题的边际成本 | 8.78 ms | **1.14 ms** | 7.7× |
| 激活内存 / context | 4563 MiB | **196 MiB** | 23× |
| 默认 context 数（大 plan） | 2（9.1 GB） | **1（4.6 GB）** | 策略修复 |
| FP16 vs FP32 延迟 | 51.4 ms | **23.7 ms** | FP16 快 2.2× |
| 决策一致性（fp16 vs fp32） | — | **7/7 场景全一致** | 见 §5 |

**最重要的发现**：这个模型的推理耗时**主要不是计算，而是每次前向的固定开销**。

```
inference_ms ≈ 7.45 + 0.011 · tokens
```

512 token 的纯计算约 5.6 ms，而固定开销 7.45 ms。所以：
- **合批比缩短输入有效得多**（多问题共享一次前向）
- 单问题场景优化空间有限（除非减少前向次数）
- 跨语言拷贝**不是瓶颈**（tokenize 降到 0.6 ms 后，推理仍是 13 ms）

---

## 2. 修复的问题

### 2.1 tokenizer：`splitOnAdded` 占 90% 编码时间

**症状**：长文本 tokenize 达 113 ms/请求。

**根因**：`Parse` 把 checkpoint 里全部 116 个 added token 拼成一个巨型正则
（`[unused0..82]`、`|||EMAIL_ADDRESS|||`、2..24 个连续空格），每次编码都对全文跑
`Split` + `FindAllString`。而普通文本几乎不含这些。

**改动**：一次字节扫描先判定"不可能匹配"（首字节触发集 + 最小空格连串），命中才走正则。

**效果**：`splitOnAdded` 2406 µs → **3.9 µs（625×）**；完整 `Encode` 2.67 ms → 0.13 ms。

**等价性保证**：1400 个输入逐一对照旧正则实现，输出**逐字节一致**；
另有"保守性"测试（返回 false 时正则必须确实无匹配）和端到端 `Encode` 对照。

### 2.2 每个问题重复 tokenize 整个 state

**症状**：10 个问题共享同一 state，却把 state 编码了 10 次。

**改动**：`EncodeState` 在请求开始时编码一次，`BuildWithState` 复用。
同时按 `maxLen` 有界编码（超长 state 会被截断，编码超出部分纯属浪费）。

**效果**：10 问题 26.9 ms → 3.5 ms（7.4×）。

**等价性保证**：7 种 state × 5 种问题 × 4 组预算的矩阵测试；
另有"有界编码 == 完整编码的前缀"和"有界下 Build 结果不变"两个专项测试。

### 2.3 合批是死代码（最重要的功能修复）

**症状**：16 个问题耗时 137.6 ms，每加一个问题 +8.78 ms——即每个问题一次独立前向。

**根因**（两处叠加）：
1. 所有现有 engine 的 batch 维都是 `1..1`（固定为 1），TRT 直接拒绝 batch>1：
   `Set dimensions are [3,166]. Expected dimensions are [1,-1]`
2. `maxBatch()` 只读静态 shape：`if shape[0] > 1 { return shape[0] }`，动态 batch 报告
   `-1`，于是**永远返回 1**，已写好的合批循环从不执行。

**改动**：
- 重编 engine 使 batch 动态（`--maxShapes` 首维 > 1）
- `maxBatch()` 改为读取 profile 的 batch 上界，并抽出纯函数 `chooseMaxBatch` 加测试
- `PadInto` 让批次填充直接写入目标切片，去掉每行两次临时分配

**效果**：16 问题 137.6 → 24.1 ms（5.7×）；边际成本 8.78 → 1.14 ms（7.7×）。

### 2.4 context 策略让大 plan 吃掉整块 GPU

**症状**：`ctx8192` engine 的 2 个 context 占 9.1 GB。

**根因**：TRT 按 profile 的 `--maxShapes` 预留最坏情况激活内存（seq 8192 → 4563 MiB/context），
而 `AdaptToEngine` 实际只用 512 token。旧策略"空闲显存 ÷ 单 context 成本"，能塞几个塞几个。

**改动**：`chooseContexts` 增加**总显存占比上限（50%）**，与"空闲显存减保留"取较小值；
抽成纯函数并加测试。

**效果**：大 plan 自动从 2 contexts 降到 **1**（4.6 GB）；小 plan（88 MiB）仍可用满 8 个。

---

## 3. Engine 预设

TensorRT 按 `--maxShapes` 预留最坏情况激活内存，所以**序列上限直接决定显存**：

| 预设 | seq 上限 | batch 上限 | 激活内存/context | 适用 |
|---|---|---|---|---|
| `laya_s512_fp16_b8` | 512 | 8 | **196 MiB** | 默认，与 checkpoint `max_len=512` 一致 |
| `laya_s1024_fp16_b8` | 1024 | 8 | 672 MiB | 一倍余量 |
| `laya_s2048_fp16_b4` | 2048 | 4 | 1664 MiB | 长文档 |
| `laya_s8192_fp16_b1` | 8192 | 1 | 4563 MiB | 极长输入，只能单行 |
| `laya_s512_fp32_b8`（对照） | 512 | 8 | 392 MiB | 需要可靠的 `act_probability` |

**batch 上限随序列上限递减是必须的**：`8192 × batch 8` 约需 36 GB 激活内存，编不出来。

`optShapes` 的 batch 固定为 1：实测 batch=2 时单问题更慢（8.3 vs 7.1 ms），而 8 问题无差别
（11.2 ms），所以最优工作点取最常见的交互式单问题请求；合批仍通过 `--maxShapes` 生效。

---

## 4. 未采纳的优化方向（有数据支持）

**减少 Go↔C++ 拷贝**：原计划是热点，实测**不是**。

依据：tokenize 从 26.7 ms 降到 0.6 ms 后，推理仍是 13 ms；且批量扫描显示每行成本
只从 6.73 ms（batch 1）降到 4.07 ms（batch 8），远非线性——说明是 GPU 计算/带宽主导。
每请求的输入是约 2 KB 量级（int64 token id），H2D 拷贝在微秒级。

**结论**：不动 C++ 拷贝路径。真正的杠杆是减少前向次数（合批）和选对 profile。

---

## 5. FP16 精度验证

用两个运行中的服务（同代码、同请求）对比，比较**用户可见**的 API 输出：

| 场景 | 决策差异 | 最大概率偏差 |
|---|---|---|
| short-1q | 0 | 0.0000 |
| short-3q | 0 | 0.0024 |
| short-9q | 0 | 0.0024 |
| medium-3q | 0 | 0.0010 |
| long-3q | 0 | 0.0007 |
| json-2q | 0 | 0.0084 |
| choice11 | 0 | 0.0000 |

**7/7 场景决策完全一致**（选中的选项、score、noul 的 argmax 都不变）。
概率/置信度在 4 位小数上有个别位漂移，最大 0.0084。

**结论**：FP16 可用于决策；若下游对概率数值本身敏感（而非只取 argmax），
应评估这个漂移是否可接受，或改用 FP32（慢 2.2×）。

### `act_probability` 的 fp16 问题（已记录，非本轮引入）

README 已记录 fp16 下 act head 的 `p*log(p)` 会 `0 * -inf = NaN`。本轮补充实测边界：

| 输入长度 | act_logits |
|---|---|
| 49 token | 正常（max\|Δ\| = 0.298 vs fp32） |
| 75 token 及以上 | **NaN** |

- 触发条件是**输入长度**，与 batch 大小、问题类型、engine profile 都无关。
  验证方式：同一 engine 单行/多行、三种问题类型交叉，NaN 只在 token 数跨过阈值时出现。
- 服务把 NaN 映射为 `act_probability = 0`（见 `inference.actProbability`），不使请求失败。
- 影响：**任何真实长度的请求都会命中**。需要可靠的 `act_probability` 就必须用 FP32 engine
  （延迟约为 fp16 的 2.2 倍）；主决策 `logits` 不受影响。

这意味着 `act_probability` 在 fp16 下**实际上不可用**，而不是"偶尔为 0"。

---

## 6. 工具

`bench/` 下都是只读工具（只调 API、只读 engine，不启动/停止/重配服务）：

| 工具 | 用途 |
|---|---|
| `bench/engprobe` | engine 的 IO、profile 范围、激活内存、能放几个 context |
| `bench/apibench` | 延迟扫描 + 拟合固定开销与每 token 成本 |
| `bench/enginecmp` | 两个 engine 的裸输出差异与速度 |
| `bench/answer-diff.ps1` | 两个服务的 API 答案差异（用户可见层面） |
| `bench/build-engines.ps1` | 按预设批量编译 engine |
| `bench/serve.ps1` | 在私有端口起服务用于基准（自动检测二进制过期并重建） |

---

## 7. 验证清单

- [x] `go build ./...`、`go vet ./...` 通过
- [x] `go test ./internal/...` 全部通过
- [x] tokenizer 快速路径：1400 输入对照旧实现逐字节一致
- [x] sequence 重构：矩阵等价性测试（7 state × 5 问题 × 4 预算）
- [x] 有界编码：== 完整编码前缀；Build 结果不变
- [x] `chooseMaxBatch`：12 个 shape/profile 组合
- [x] `chooseContexts`：8 个场景 + 全区间不超占比的不变量测试
- [x] `PadInto`：8 个边界场景 + 复用缓冲不越界写
- [x] 端到端：新旧 engine 决策一致（0 差异）
- [x] FP16 vs FP32：7 场景决策一致
