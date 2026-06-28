# 实验计划 2026-06-25 — Pattern 描述符可控性(自主循环)

> 用户直觉:窗口是**结构化 pattern**,单个标量(max)太薄、控不动;应给"对 pattern 的描述"(频率/幅度)。
> **核心问题**:在 label 之外,加 **频率(period)/ 幅度(std)** 这种 pattern 描述符,能不能带来**可测的、label 给不了的**控制(steer 出"慢一点的走路 / 幅度小一点的跑步")?
> 证据标准:Internal Direction Check（report-draft 级）。一次一个变量。adherence 噪声大 → 多 seed。

## 假设 / 证伪 / 决策

- **H**:某个描述符(std 或 period)的 adherence **明显 > max 的 0.36**,说明"更对的描述符 → 更强控制"。
- **证伪**:std/period adherence ≤ max(0.36)且 CI 重叠 → 描述符种类不改变控制强度(瓶颈在别处)。
- **决策**:谁最高 → Round 2 做它的 compound(label+该描述符+另一个)+ 多 seed;都不行 → 记边界结论(单标量控制是天花板)。

## 指标

- **adherence = Spearman(requested level, realized stat)**,pooled + per-label,**rank-based**(对 period 的 FFT 量化稳健)。
- 同时记 accept_rate（生成有效性）。
- realized stat 用 `window_stat`(period=FFT 主周期、std、range)在解码窗口上算。

## 对照 / 控制

- **主对照**:max 单标量(已知单 subject 0.36)。Round 1 **重跑 max**(同配置)做 apples-to-apples,排除 config 漂移。
- 固定:Qwen2.5-1.5B、**单 subject 101**(可控性成立的地方)、train_length 5000、FICA、epochs 100、encoding=values(真实数值)、HF 后端、temp 1.0。
- 变量:**只变 `--stat`**(max / std / period)。

## Run Matrix — Round 1

| Run | stat | encoding | subject | seeds | 期望 | 输出 |
|---|---|---|---|---|---|---|
| R1-max | max | values | 101 | 42,7 | ~0.36(对照) | qwen_s101_max_seed{} |
| R1-std | std | values | 101 | 42,7 | ? > max? | qwen_s101_std_seed{} |
| R1-period | period | values | 101 | 42,7 | ? steer 节奏? | qwen_s101_period_seed{} |

## Stop / 后续

- R1 出现某 stat adherence > 0.5 且稳 → Round 2:compound(label+std+period)+ 3 seed,看叠加是否更强。
- 都 ≈ max → Round 2:换"范围/边界约束"型 prompt(min+max box),或上 3B 测容量。
- compute 上限:本轮 ≤ ~12 个 Qwen GPU job。

## Reviewer 风险

- period 的 FFT 量化 → 用 rank(Spearman)而非绝对值,已规避。
- adherence run-variable → ≥2 seed,promising 的扩到 3+。
- 单 subject/单通道 → claim 收窄到"单 subject 上 descriptor 种类对可控性的影响"。
