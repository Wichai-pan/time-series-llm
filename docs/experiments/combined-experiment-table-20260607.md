# Combined Experiment Table - Canonical TSG Evaluation

日期：2026-06-07

## 统一评价协议

这张表是当前应该使用的主 TSG 表。

所有 setting 都被转换到同一个 canonical evaluation space：

> `activity_sdforger_scaled`

含义：

- 真实数据：每个 activity 用 SDForger baseline preprocessing 得到 standardized windows。
- 旧实验：原本就是这个 scaled space，直接评估。
- normalization 实验：先用训练集统计量从各自 model space 转回 raw-like，再统一转成同一个 activity-level SDForger scaled space。

因此，这张表里的 `MDD / ACD / SD / KD / ED / DTW` 可以放在一起比较。

输出文件：

- `outputs/canonical-scaled-tsgbench-20260607/canonical_scaled_tsgbench_summary.csv`
- 远程：`/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/canonical_scaled_tsgbench_20260607/`
- 脚本：`scripts/evaluate_canonical_scaled_tsgbench.py`

## 主表

| Group | Setting | Activity | n | MDD ↓ | ACD ↓ | SD ↓ | KD ↓ | ED ↓ | DTW ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| main | baseline | walking | 130 | 0.266 | 0.165 | 0.798 | 1.668 | 21.279 | 237.761 |
| main | baseline | running | 98 | 0.297 | 0.498 | 0.613 | 0.445 | 17.079 | 111.088 |
| main | label-v1 | walking | 72 | 0.276 | 0.550 | 1.841 | 1.897 | 23.915 | 291.820 |
| main | label-v1 | running | 90 | 0.297 | 0.585 | 0.655 | 0.221 | 18.585 | 141.005 |
| main | raw-unified | walking | 72 | 0.257 | 5.973 | 3.524 | 118.648 | 404.584 | 6026.479 |
| main | raw-unified | running | 79 | 0.299 | 2.285 | 2.918 | 102.878 | 252.417 | 3772.691 |
| main | clip_p05_p95 | walking | 72 | 0.273 | 2.253 | 0.516 | 1.144 | 20.166 | 198.446 |
| main | clip_p05_p95 | running | 79 | 0.316 | 1.960 | 0.336 | 0.422 | 22.645 | 139.695 |
| normalization | current_activity_window_zscore | walking | 72 | 0.257 | 5.973 | 3.524 | 118.648 | 404.584 | 6026.479 |
| normalization | current_activity_window_zscore | running | 79 | 0.299 | 2.285 | 2.918 | 102.878 | 252.417 | 3772.691 |
| normalization | joint_window_zscore | walking | 55 | 0.273 | 3.218 | 2.534 | 155.533 | 108.188 | 1362.414 |
| normalization | joint_window_zscore | running | 99 | 0.294 | 3.957 | 16.013 | 287.490 | 458.578 | 6624.401 |
| normalization | global_series_zscore | walking | 61 | 0.305 | 3.155 | 3.241 | 199.238 | 167.121 | 2166.044 |
| normalization | global_series_zscore | running | 62 | 0.308 | 1.251 | 6.026 | 70.526 | 55.901 | 613.962 |
| normalization | activity_series_zscore | walking | 57 | 0.269 | 1.300 | 9.775 | 91.919 | 23.596 | 224.500 |
| normalization | activity_series_zscore | running | 59 | 0.294 | 2.459 | 0.586 | 122.339 | 2906.060 | 41885.922 |
| multi-subject diagnostic | multi_raw_unified | walking | 51 | 0.174 | 5.344 | 0.016 | 40.233 | 700.291 | 9920.940 |
| multi-subject diagnostic | multi_raw_unified | running | 58 | 0.194 | 1.623 | 1.349 | 92.912 | 569.687 | 7943.930 |
| multi-subject diagnostic | multi_clip_p05_p95 | walking | 51 | 0.172 | 3.117 | 0.229 | 0.746 | 21.672 | 166.065 |
| multi-subject diagnostic | multi_clip_p05_p95 | running | 58 | 0.204 | 1.611 | 0.652 | 0.281 | 20.737 | 147.235 |
| held-out subject diagnostic | unseen_raw_unified | walking | 51 | 0.188 | 4.212 | 0.072 | 39.805 | 787.558 | 11160.400 |
| held-out subject diagnostic | unseen_raw_unified | running | 58 | 0.232 | 1.296 | 1.684 | 92.044 | 689.475 | 9567.380 |
| held-out subject diagnostic | unseen_clip_p05_p95 | walking | 51 | 0.189 | 1.992 | 0.173 | 1.175 | 21.496 | 151.102 |
| held-out subject diagnostic | unseen_clip_p05_p95 | running | 58 | 0.236 | 2.228 | 0.987 | 1.149 | 20.685 | 146.689 |
| validity diagnostic | strict_reject_p05_p95_trainref | walking | 14 | 0.242 | 2.715 | 0.250 | 0.336 | 23.934 | 232.255 |
| validity diagnostic | strict_reject_p05_p95_trainref | running | 10 | 0.324 | 1.415 | 0.687 | 0.492 | 16.029 | 125.246 |
| validity diagnostic | soft_repair_p05_p95_minmax_trainref | walking | 27 | 0.200 | 3.087 | 0.414 | 0.250 | 23.195 | 205.601 |
| validity diagnostic | soft_repair_p05_p95_minmax_trainref | running | 34 | 0.232 | 2.016 | 0.646 | 0.349 | 19.771 | 141.353 |
| validity diagnostic | strict_reject_p05_p95_heldout | walking | 14 | 0.259 | 1.648 | 0.194 | 0.765 | 16.100 | 117.918 |
| validity diagnostic | strict_reject_p05_p95_heldout | running | 10 | 0.325 | 1.562 | 1.022 | 1.360 | 19.009 | 157.000 |
| validity diagnostic | soft_repair_p05_p95_minmax_heldout | walking | 27 | 0.218 | 1.974 | 0.358 | 0.679 | 18.915 | 140.000 |
| validity diagnostic | soft_repair_p05_p95_minmax_heldout | running | 34 | 0.256 | 2.615 | 0.981 | 1.218 | 20.320 | 157.195 |

## 结果判断

### 1. Baseline 仍是最稳参考

`baseline` 的 ACD 最好，ED/DTW 也稳定。尤其 running baseline 的 DTW 只有 111.088，是全表最强之一。

### 2. Raw-unified 明确失败

`raw-unified` 和 `current_activity_window_zscore` 数值一致，因为后者就是复现旧 raw-unified preprocessing。

它们的 ED/DTW/KD 都爆炸，说明 unified label conditioning 直接生成不稳定。

### 3. clip_p05_p95 是目前最有用的修复信号，但不是最终方法

相对 raw-unified：

- walking DTW：6026.479 -> 198.446
- running DTW：3772.691 -> 139.695
- KD 也从百级降回接近 baseline

但 ACD 仍明显差于 baseline：

- walking baseline ACD 0.165，clip 2.253
- running baseline ACD 0.498，clip 1.960

所以 clip 修复了形状距离和分布爆炸，但没有恢复 autocorrelation / rhythm structure。

### 4. 今天 normalization ablation 没有形成整体进步

局部看：

- `activity_series_zscore` walking 的 ED/DTW 接近 clip/baseline。
- `global_series_zscore` running 的 DTW 比 raw-unified 明显好。

但整体看：

- `joint_window_zscore` running 很差。
- `global_series_zscore` 的 SD/KD 仍很差。
- `activity_series_zscore` running 的 ED/DTW 直接崩。

因此 normalization 不是当前主要解法。

### 5. Multi-subject 增加数据量没有单独修好 raw-unified

multi-subject 使用 PAMAP2 subject101/102/105，每个 subject/activity 取 5000 rows。执行前发现 subject103 没有 running，因此改用 105。

结果显示：

- `multi_raw_unified` 仍然失败：walking DTW 9920.940，running DTW 7943.930。
- `multi_clip_p05_p95` 仍然有效：walking DTW 166.065，running DTW 147.235，KD 降到小于 1。
- 但 ACD 仍偏高，walking 尤其明显。

因此“数据量不够”不是主要解释；核心问题仍是 unified conditional generation 的 latent validity / decoding stability。

### 6. Held-out subject 结果支持 clip 的初步跨 subject 稳定性

held-out subject 评估使用 subject106/108 的真实 walking/running 作为 reference，synthetic 仍来自 101/102/105 训练出的 generator。

结果显示：

- `unseen_raw_unified` 仍然失败：walking DTW 11160.400，running DTW 9567.380。
- `unseen_clip_p05_p95` 仍保持接近 train-reference clip 的 ED/DTW：walking DTW 151.102，running DTW 146.689。
- ACD 仍然是主要风险，尤其 running unseen ACD 2.228。

因此可以说：`clip_p05_p95` 不是只对训练 subjects 有效；但还不能说它已完成严格 HAR 泛化验证。

### 7. Validity/rejection 诊断显示 strict reject 太苛刻，soft repair 可作为方法候选

新测试了两个 generation-time-style 规则：

- `strict_reject_p05_p95`：只接受所有 latent 维度都在 p05-p95 内的样本。
- `soft_repair_p05_p95_minmax`：超出 train min-max 就拒绝；在 min-max 内但超出 p05-p95 就 clip 回 p05-p95。

结果：

- strict reject 样本数太少：walking 14、running 10，不适合作为主方法。
- soft repair 保留 walking 27、running 34，ED/DTW 接近 post-hoc clip，但没有明显全面优于 post-hoc clip。
- soft repair 的优势主要是方法解释更干净：能报告 clean / repaired / rejected counts。

因此当前推荐：`clip_p05_p95` 作为 simple strong diagnostic baseline，`soft_repair` 作为 generation-time validity control 的初版候选。

## 下一步实验决策

不要继续堆 normalization variant。

下一步最合理的是：

1. 以 `baseline` 和 `clip_p05_p95` 为主要参考。
2. 做 generation-time latent validity control / rejection sampling。
3. 统一用本表的 canonical TSG protocol 评估。
4. 同时保留 raw-like amplitude ratio 作为 sanity check。

目标不是让所有指标都赢 baseline，而是先证明：

> 在 unified label-conditioned setting 下，generation-time validity control 能在保留 label controllability 的同时，把 ED/DTW/KD 从 raw-unified failure 拉回接近 baseline，并尽量改善 ACD。
