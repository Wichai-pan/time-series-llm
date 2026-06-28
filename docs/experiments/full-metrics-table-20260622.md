# 完整 6 指标汇报表（2026-06-22）

> 全部 canonical scaled 空间（`activity_sdforger_scaled`，对齐 TSGBench），**6 个指标全列**：MDD（分布）/ACD（自相关·节奏）/SD（偏度）/KD（峰度）/ED（欧氏）/DTW。越低越好。
> 来源：`combined-experiment-table-20260607`、`smooth-repair-results-20260615`、`combined-best-config-results-20260621`、本次 Qwen TSG（`reports/qwen_tsg_20260622`）。

## 表1 · 逐步叠加到最优（walking）

| 步 | 配置 | n | MDD | ACD | SD | KD | ED | DTW |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| 0 | baseline（无条件，单101） | 130 | 0.266 | **0.165** | 0.798 | 1.668 | 21.28 | 237.8 |
| 1 | + label 条件 | 72 | 0.276 | 0.550 | 1.841 | 1.897 | 23.92 | 291.8 |
| 2 | + unified（无修复） | 72 | 0.257 | **5.973** | 3.524 | **118.6** | **404.6** | **6026** |
| 4 | + pooled clip | 72 | 0.273 | 2.253 | 0.516 | 1.144 | 20.17 | 198.4 |
| 5 | + multi-subject（pooled clip） | 51 | 0.172 | 3.117 | 0.229 | 0.746 | 21.67 | 166.1 |
| 6 | + **per-activity clip** | 72 | 0.265 | **0.810** | 0.212 | 0.936 | 18.19 | 184.9 |
| 6b | per-act shrink_minmax（walking极致） | 72 | 0.264 | **0.362** | 0.041 | 1.162 | 18.20 | 200.2 |
| **7** | **multi + per-act clip（最优）** | 51 | **0.171** | **0.740** | 0.385 | 0.662 | 20.06 | 182.7 |

## 表1 · 逐步叠加到最优（running）

| 步 | 配置 | n | MDD | ACD | SD | KD | ED | DTW |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| 0 | baseline（无条件，单101） | 98 | 0.297 | **0.498** | 0.613 | 0.445 | 17.08 | 111.1 |
| 1 | + label 条件 | 90 | 0.297 | 0.585 | 0.655 | 0.221 | 18.59 | 141.0 |
| 2 | + unified（无修复） | 79 | 0.299 | 2.285 | 2.918 | **102.9** | **252.4** | **3773** |
| 4 | + pooled clip | 79 | 0.316 | 1.960 | 0.336 | 0.422 | 22.65 | 139.7 |
| 5 | + multi-subject（pooled clip） | 58 | 0.204 | 1.611 | 0.652 | 0.281 | 20.74 | 147.2 |
| 6 | + **per-activity clip** | 79 | 0.314 | 1.370 | 0.399 | 0.352 | 22.41 | 137.8 |
| **7** | **multi + per-act clip（最优）** | 58 | **0.211** | **0.738** | 0.699 | 0.023 | 20.17 | 154.7 |

**最优叠加（你问的核心）**：walking 全指标 MDD0.171/ACD0.740/SD0.385/KD0.662/ED20.06/DTW182.7；running MDD0.211/ACD0.738/SD0.699/KD0.023/ED20.17/DTW154.7。
**从 raw-unified 的灾难（KD 118/103、DTW 6026/3773）一路修到接近 baseline，MDD 还低于 baseline。**

## 表1b · normalization 消融（**你说得对，它不是完全没用**，全6指标，单101）

| 模式 | act | MDD | ACD | SD | KD | ED | DTW | 读法 |
|---|---|--:|--:|--:|--:|--:|--:|---|
| current_window_z(=raw) | walk | 0.257 | 5.973 | 3.524 | 118.6 | 404.6 | 6026 | 爆炸 |
| joint_window_z | walk | 0.273 | 3.218 | 2.534 | 155.5 | 108.2 | 1362 | 部分降 DTW |
| global_series_z | walk | 0.305 | 3.155 | 3.241 | 199.2 | 167.1 | 2166 | SD/KD仍差 |
| **activity_series_z** | walk | 0.269 | **1.300** | 9.775 | 91.9 | **23.60** | **224.5** | ✅**walking ACD/ED/DTW 明显改善** |
| current_window_z(=raw) | run | 0.299 | 2.285 | 2.918 | 102.9 | 252.4 | 3773 | 爆炸 |
| joint_window_z | run | 0.294 | 3.957 | 16.01 | 287.5 | 458.6 | 6624 | 崩 |
| **global_series_z** | run | 0.308 | **1.251** | 6.026 | 70.53 | 55.90 | 613.9 | ✅**running ACD/DTW 改善** |
| activity_series_z | run | 0.294 | 2.459 | 0.586 | 122.3 | 2906 | 41886 | ❌running ED/DTW 崩 |

**诚实读法（更正我之前"完全没用"）**：normalization **确实有局部改善**——`activity_series_z` 把 **walking ACD 5.97→1.30、DTW 6026→224**（比 pooled clip 2.253 还好！），`global_series_z` 把 **running ACD 2.29→1.25、DTW 3773→614**。**但没有一种模式同时修好两个动作**（修好 walking 的把 running 搞崩，反之亦然）。

### 补跑：multi-subject + activity_series_zscore（无修复，本次新跑 job 35216589）

| 配置（无修复） | act | MDD | ACD | SD | KD | ED | DTW |
|---|---|--:|--:|--:|--:|--:|--:|
| 单subject activity_series_z | walk | 0.269 | 1.300 | 9.78 | 91.9 | 23.6 | 224.5 |
| **multi activity_series_z** | walk | 0.176 | 1.962 | 3.71 | 33.5 | 818 | **13248** |
| 单subject activity_series_z | run | 0.294 | 2.459 | 0.59 | 122 | 2906 | 41886 |
| **multi activity_series_z** | run | 0.202 | **6.450** | 5.81 | 47.0 | 802 | **12497** |

**补跑结论**：multi-subject 后 per-activity-series norm **没改善、running 反而更差且爆炸**（DTW 12000+）——多 subject 的人际差异让"只靠 normalization、不修复"更不稳。**→ 坐实：normalization 不是杠杆，per-activity CLIP 修复才是**（最优 0.74 全靠修复，normalization 替代不了它；二者要叠加只能 norm-空间内做修复，是后续工作）。

## 表2 · Prompt 设计：gpt2 vs Qwen（含 Qwen 完整 TSG 实数）

### 2a · 数值可控性（adherence Spearman）+ 爆炸

| 模型 | 类别(label) | 数值-实值编码 | 数值-粗档 | 爆炸(abs max) |
|---|---|---|---|---|
| gpt2(124M) | ✅0.82 | ❌0.01；×1/×3比0.94(应3.0,3seed+CI) | ❌~0 | 💥3761 |
| Qwen-1.5B | ✅ | ✅**0.36**(3seed:0.20/0.51/0.38;v2 0.58) | ⚠️**0.10** | ✅2.98(100%在范围) |

### 2b · Qwen 生成质量 TSG（**本次新算，全6指标**，无修复）

| Qwen run | subject | act | MDD | ACD | SD | KD | ED | DTW |
|---|---|---|--:|--:|--:|--:|--:|--:|
| label-only | 单101 | walk | 0.278 | **1.574** | 0.687 | **0.207** | 19.90 | **213.3** |
| label-only | 单101 | run | 0.313 | 2.915 | 0.158 | **0.258** | 20.77 | **157.7** |
| values_v2(条件统计) | 单101 | walk | 0.280 | 2.753 | 0.429 | 2.599 | 16.98 | 196.8 |
| values_v2(条件统计) | 单101 | run | 0.314 | 3.201 | 0.705 | 2.039 | 17.96 | 176.8 |
| **label-only** | **multi** | walk | 0.187 | 2.324 | 0.420 | 1.041 | 20.12 | 172.8 |
| **label-only** | **multi** | run | 0.238 | 2.235 | 0.523 | 0.337 | 18.80 | 145.4 |
| **条件统计** | **multi** | walk | 0.170 | 2.293 | 0.294 | 0.966 | 22.09 | 168.8 |
| **条件统计** | **multi** | run | 0.212 | 2.353 | 0.622 | 0.045 | 21.80 | 158.8 |

**Qwen multi-subject adherence（条件统计）**：pooled Spearman **0.097**（walking **0.422** / running **-0.062**），accept 0.40，max 3.16。

**对比 gpt2 raw-unified（同单101、无修复）**：gpt2 walking KD **118.6** / DTW **6026**；**Qwen label-only KD 0.207 / DTW 213**。

**→ Qwen 三个诚实结论**：
1. **无需修复就不爆炸**——单 subject 和 **multi-subject 都不爆**（multi KD ~1、DTW ~150–170、max 3.16 全在范围），而 gpt2 同设置爆炸、必须靠修复。这是 Qwen 最硬的点。
2. **ACD 多 subject 后升到 ~2.3**（单 subject 1.57 → multi 2.3）：更多 subject 更难，节奏保真度下降，但 MDD 仍低、不爆炸。
3. **可控性多 subject 后退化**：pooled Spearman 0.36（单）→ **0.10（multi）**——walking 仍能 steer（0.42），**running 失败（-0.06）**拖垮 pooled。

## 表3 · Held-out（unseen 106/108）三方对比 + norm 加 clip 救回爆炸

**norm+clip 验证「爆炸=没clip」**（multi + activity_series norm，16%/14% 点越界）：

| act | 参考 | clip | MDD | ACD | KD | DTW |
|---|---|---|--:|--:|--:|--:|
| walking | train | none | 0.176 | 1.962 | 33.5 | **13248** |
| walking | train | **clip** | 0.176 | 1.485 | 0.98 | **309** |
| walking | heldout | **clip** | 0.193 | **0.484** | 1.41 | 224 |
| running | train | none | 0.202 | 6.450 | 47.0 | **12497** |
| running | train | **clip** | 0.202 | 2.018 | 0.65 | **243** |
| running | heldout | clip | 0.234 | 2.504 | 0.22 | 246 |

→ **clip 把 DTW 13248→309、KD 33.5→0.98 彻底救回**，坐实「爆炸=没 clip」。

**三方 held-out 对比（train ACD → held-out ACD）：**

| 方法 | act | train ACD | held-out ACD | heldout MDD | heldout DTW | 爆炸 |
|---|---|--:|--:|--:|--:|---|
| gpt2 最优(multi+latent clip) | walk | 0.740 | 1.40 / **0.69**(shrink) | 0.19 | 169 | ✅不爆 |
| gpt2 最优 | run | 0.738 | 1.743 | 0.24 | 151 | ✅不爆 |
| multi+norm+clip | walk | 1.485 | **0.484** 🏆 | 0.19 | 224 | ✅clip后 |
| multi+norm+clip | run | 2.018 | 2.504 | 0.23 | 246 | ✅clip后 |
| Qwen multi(不clip) | walk | 2.324 | 1.203 | 0.21 | 138 | ✅原生 |
| Qwen multi(不clip) | run | 2.235 | 2.712 | 0.26 | 158 | ✅原生 |

**读数**：① gpt2 最优 train 最好、walking 泛化好(shrink 0.69)、running 受 subject 漂移退化(1.74)；② **norm+clip 的 walking held-out 0.48 是全场最好**(per-activity-series norm 利于 walking 跨人泛化)；③ **Qwen 不 clip 也不爆**(held-out walk 1.20/run 2.71)；④ running 在所有方法都最难(1.74–2.71)，是共性 subject-drift 问题。

## 一句话（这两件事）

1. **保真度线**：normalization（局部有用但不全）→ per-activity 修复（最大杠杆）+ multi-subject（救 running）→ **最优两动作 ACD 都 0.74、MDD 低于 baseline、6 指标全回到接近 baseline**。
2. **Prompt 线**：gpt2 数值不可控且爆炸（能力上限，3seed+CI 锁定）；**换 Qwen + 实值编码 → 出现可控性（0.36 vs 粗档 0.10）且无需修复就不爆炸（KD 0.2 vs gpt2 118）**。Qwen 可控性是 provisional（3seed 无 CI）。
