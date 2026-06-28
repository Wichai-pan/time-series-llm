# Baseline Verification：PAMAP2 Subject101 周期性与旧 SDForger Baseline

日期：2026-05-22

目标：整理旧 SDForger/PAMAP2 代码与结果，并完成 baseline verification 的前四步：

1. 选数据
2. 选通道
3. 做周期性检查
4. 建立 baseline

本次没有提交新 Slurm 训练任务；只做了原始数据读取、周期性分析和旧 baseline artifact 核验。

## 旧代码整理

远程项目 root：

```text
/scratch/project_2016517/panh/time-series-llm/fms-dgt
```

旧代码/结果最相关路径：

| 用途 | 路径 | 当前判断 |
|---|---|---|
| PAMAP2 原始数据 | `/scratch/project_2016517/panh/datasets/pamap2/PAMAP2_Dataset/Protocol/subject101.dat` | 可用于重新按 activity 选数据 |
| 旧 PAMAP2 预处理 | `scripts/prepare_pamap2_subject101.py` | 可复用，但默认丢弃 `activity_id` |
| SDForger time-series builder | `fms_dgt/public/databuilders/time_series/` | baseline pipeline 主体 |
| 单变量 paper-aligned task | `tasks/public/time_series/pamap2_subject101_univariate_paper/task.yaml` | 旧 baseline 配置 |
| 单变量旧输出 | `output/time_series/pamap2_subject101_univariate_paper/final_data.jsonl` | output 存在，provisional |
| 单变量旧日志 | `logs/sdforger-pamap2-u-paper_33833773.out` | 显示 run completed |
| 旧指标报告 | `reports/pamap2_subject101_univariate_paper_tsgbench_metrics.md` | 可作 baseline 线索，不作 final evidence |

本地输出包：

```text
outputs/baseline-verification-20260522/
```

本地分析脚本：

```text
scripts/pamap2_periodicity_check.py
```

## Step 1：选数据

选择：

- Dataset：PAMAP2
- Subject：101
- 原始文件：`Protocol/subject101.dat`
- 候选 activity：
  - `4 = walking`
  - `5 = running`
  - `6 = cycling`

原因：

- walking / running / cycling 都是较可能具有周期性的人体运动；
- SDForger 原文依赖 periodicity-aware segmentation，因此先验证周期性动作更符合方法前提；
- 旧 processed parquet 不含 `activity_id`，所以 activity-level 检查必须回到原始 `.dat`。

结论：

- **优先 activity：walking、running**
- **cycling 暂时作为 secondary candidate**，因为不同 channel 的主频/ACF 表现不一致。

## Step 2：选通道

选择：

- 主通道：`hand_acc16_x`
- 备选通道：`ankle_acc16_x`

原因：

- `hand_acc16_x` 与旧 SDForger univariate paper-aligned baseline 一致；
- `ankle_acc16_x` 可作为对照，因为 walking/running 的腿部运动通常更周期；
- 先做单变量，避免 multivariate channel alignment 和旧 `embedding_dim: auto` 风险。

结论：

- **当前最优先 baseline：walking / running + hand_acc16_x**
- **可补充 sanity check：walking / running + ankle_acc16_x**

## Step 3：周期性检查

执行脚本：

```text
scripts/pamap2_periodicity_check.py
```

远程输出：

```text
/scratch/project_2016517/panh/time-series-llm/fms-dgt/reports/baseline_verification_20260522
```

本地输出：

```text
outputs/baseline-verification-20260522/
```

检查内容：

- raw signal segment
- ACF
- PSD / FFT-style frequency view

参数：

- segment length：5000 samples
- sampling rate assumption：100 Hz
- max ACF lag：1000

结果汇总：

| activity | channel | rows | ACF peak lag | ACF score | dominant freq Hz | period samples | label |
|---|---|---:|---:|---:|---:|---:|---|
| walking | hand_acc16_x | 22253 | 58 | 0.773 | 1.758 | 56.9 | strong |
| walking | ankle_acc16_x | 22253 | 116 | 0.797 | 1.758 | 56.9 | strong |
| running | hand_acc16_x | 21265 | 82 | 0.929 | 2.441 | 41.0 | strong |
| running | ankle_acc16_x | 21265 | 82 | 0.631 | 4.883 | 20.5 | strong |
| cycling | hand_acc16_x | 23575 | 7 | 0.402 | 11.816 | 8.5 | strong but questionable |
| cycling | ankle_acc16_x | 23575 | 3 | 0.815 | 0.781 | 128.0 | strong but channel-mismatched |

图：

- `outputs/baseline-verification-20260522/subject101_walking_hand_acc16_x_periodicity.png`
- `outputs/baseline-verification-20260522/subject101_walking_ankle_acc16_x_periodicity.png`
- `outputs/baseline-verification-20260522/subject101_running_hand_acc16_x_periodicity.png`
- `outputs/baseline-verification-20260522/subject101_running_ankle_acc16_x_periodicity.png`
- `outputs/baseline-verification-20260522/subject101_cycling_hand_acc16_x_periodicity.png`
- `outputs/baseline-verification-20260522/subject101_cycling_ankle_acc16_x_periodicity.png`

解释：

- walking 和 running 的 ACF peak score 高，PSD 主频也合理，更适合作为 SDForger baseline verification 的第一批 activity。
- cycling 虽然也被标为 strong，但 `hand_acc16_x` 和 `ankle_acc16_x` 的主频/周期差异大，可能涉及姿态、设备位置或动作结构差异；不建议作为第一优先 baseline。

## Step 4：建立 baseline

旧 SDForger univariate baseline 配置：

```text
tasks/public/time_series/pamap2_subject101_univariate_paper/task.yaml
```

核心设置：

```yaml
data:
  data_path: ${DGT_DATA_DIR}/public/time_series/pamap2_subject101_univariate_hand_acc16_x.parquet

data_params:
  train_length: 5000
  train_samples: 1
  augmentation_strategy: univariate
  train_channels:
    - hand_acc16_x

sdforger_params:
  min_outputs_to_generate: 50
  max_outputs_to_generate: 100
  embedding_type: "fica"
  embedding_dim: auto
  variance_explained: 0.7
  min_windows_number: 30
  min_windows_length: 300
```

旧 run artifact：

| 项 | 路径 / 值 |
|---|---|
| output dir | `output/time_series/pamap2_subject101_univariate_paper/` |
| generated file | `final_data.jsonl` |
| generated count | 85 |
| real training windows | 31 |
| sequence length | 300 |
| Slurm log | `logs/sdforger-pamap2-u-paper_33833773.out` |
| task result | `status=completed`, `Number of data produced=85` |
| local copied manifest | `outputs/baseline-verification-20260522/pamap2_subject101_univariate_paper_task_card.jsonl` |
| local copied metrics | `outputs/baseline-verification-20260522/pamap2_subject101_univariate_paper_tsgbench_metrics.md` |

旧 metric snapshot：

| Metric | Value |
|---|---:|
| MDD | 0.395802 |
| ACD | 1.197348 |
| SD | 0.833090 |
| KD | 0.951272 |
| ED | 19.039300 |
| DTW | 304.437101 |
| SHR | N/A |

Baseline 判断：

- 旧 baseline **工程上成立**：配置、输出、task result 和 log 能对应起来。
- 旧 baseline **只能 provisional 使用**：
  - 远程目录没有 Git provenance；
  - 旧 parquet 是 mixed non-zero activities，不是 activity-specific；
  - 旧 `final_data.jsonl` 是 standardized SDForger window space；
  - 旧 report 来自旧 evaluation snapshot，需要在固定脚本下重算。

因此周一汇报建议用这个表述：

> We recovered a provisional SDForger univariate baseline on PAMAP2 subject101 hand_acc16_x. The run completed and generated 85 synthetic windows. However, the old baseline uses a mixed non-zero-activity parquet without activity labels, so it is useful as a recovered baseline candidate but not yet as final evidence. Our periodicity check suggests that walking and running are better controlled activity-specific settings for the next clean baseline.

## 当前最优先的下一步

1. 用原始 `subject101.dat` 重建 activity-specific univariate parquet：
   - walking + `hand_acc16_x`
   - running + `hand_acc16_x`
2. 沿用旧 SDForger univariate setting：
   - FICA
   - train_length 5000
   - min_windows_length around 300
   - generated 50-100 samples
3. 对比：
   - recovered mixed-activity old baseline
   - clean walking-only baseline
   - clean running-only baseline
4. 之后才测试：
   - `+ activity label`
   - `ankle_acc16_x`
   - multivariate hand/ankle channels

## 汇报结论

目前最可靠的结论不是“SDForger 已经在 PAMAP2 上有效”，而是：

> SDForger 的单变量 baseline 可以在 PAMAP2 上被恢复；但由于 PAMAP2 不同 activity 的周期性不同，baseline verification 应先聚焦于 walking/running 这类周期性强的 activity，再扩展到 activity-conditioned prompting 和 multivariate generation。
