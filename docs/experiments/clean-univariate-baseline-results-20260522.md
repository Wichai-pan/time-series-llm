# Clean Univariate Baseline Results - PAMAP2 subject101

日期：2026-05-22

## 目的

用原始 PAMAP2 `subject101.dat` 重新建立两个干净的 activity-specific SDForger univariate baseline：

- walking-only + `hand_acc16_x`
- running-only + `hand_acc16_x`

这次实验的目标是 baseline verification，不是证明最终方法优越。旧 mixed-activity baseline 只能作为参考；本实验用原始 `.dat` 重新过滤 activity。

## 数据

原始数据：

`/scratch/project_2016517/panh/datasets/pamap2/PAMAP2_Dataset/Protocol/subject101.dat`

生成的新 parquet：

- walking: `/scratch/project_2016517/panh/time-series-llm/fms-dgt/data/public/time_series/pamap2_subject101_walking_hand_acc16_x.parquet`
- running: `/scratch/project_2016517/panh/time-series-llm/fms-dgt/data/public/time_series/pamap2_subject101_running_hand_acc16_x.parquet`

过滤结果：

| Activity | PAMAP2 activity_id | Channel | Rows |
|---|---:|---|---:|
| walking | 4 | `hand_acc16_x` | 22,253 |
| running | 5 | `hand_acc16_x` | 21,265 |

注意：这两个 parquet 只保留单变量传感器通道，不保存 `activity_id`。这是为了复现 SDForger univariate baseline；activity provenance 记录在 metadata JSON 中。

## Baseline 配置

共同设置：

- SDForger univariate setting
- embedding: FICA
- channel: `hand_acc16_x`
- `train_length: 5000`
- `min_windows_length: 300`
- `min_windows_number: 30`
- `embedding_dim: auto`
- `variance_explained: 0.7`
- target generated samples: 50-100 minimum, stopping by norm diversity

Task configs:

- `/scratch/project_2016517/panh/time-series-llm/fms-dgt/tasks/public/time_series/pamap2_subject101_walking_hand_acc16_x_univariate_baseline/task.yaml`
- `/scratch/project_2016517/panh/time-series-llm/fms-dgt/tasks/public/time_series/pamap2_subject101_running_hand_acc16_x_univariate_baseline/task.yaml`

本地生成脚本和 Slurm 文件：

- `scripts/prepare_pamap2_activity_subject.py`
- `puhti-generated/slurm/run_sdforger_pamap2_clean_univariate_baselines_gpu.sh`
- `puhti-generated/slurm/run_sdforger_pamap2_running_univariate_baseline_gpu.sh`
- `puhti-generated/slurm/evaluate_sdforger_pamap2_walking_univariate_baseline_cpu.sh`

## 运行状态

第一次合并 job：

- Job: `34519769`
- 状态：failed after walking completed
- 原因：walking 生成完成后，第二个 Python invocation 前后出现环境/container 相关 `context deadline exceeded`，exit code `127`。
- 判断：不影响 walking 生成结果本身，但说明多个 SDForger/vLLM run 串在一个 Slurm script 中不够稳。

拆分后：

- walking eval job: `34520119`, completed, 44s
- running generation + eval job: `34520120`, completed, 5m43s

## SDForger 内部预处理观察

| Activity | Selected period | Windows | Window length | FICA dim | Variance retained | Generated samples |
|---|---:|---:|---:|---:|---:|---:|
| walking | 58 | 31 | 300 | 4 | 0.8306 | 130 |
| running | 82 | 31 | 300 | 2 | 0.7010 | 98 |

这些周期和先前 ACF/PSD 检查一致，支持 walking/running + `hand_acc16_x` 作为当前 clean univariate baseline 的合理起点。

## Metrics

评估脚本：

`/scratch/project_2016517/panh/time-series-llm/fms-dgt/scripts/evaluate_sdforger_paper_metrics.py`

Synthetic value space: `scaled`

| Activity | MDD | ACD | SD | KD | ED | DTW | SHR |
|---|---:|---:|---:|---:|---:|---:|---|
| walking | 0.266287 | 0.165201 | 0.797960 | 1.667799 | 21.279075 | 237.760946 | N/A |
| running | 0.296503 | 0.497915 | 0.612561 | 0.444503 | 17.078956 | 111.087924 | N/A |

Interpretation:

- 这些指标可以作为 baseline verification evidence。
- 不能单独证明生成样本对 HAR 有用。
- `ED/DTW` 使用 deterministic subset 到 31 个真实窗口做配对比较。
- `final_data.jsonl` 是 SDForger standardized window space，不能直接当 raw sensor units 用于 classifier。

## Artifacts

本地结果目录：

`outputs/clean-univariate-baseline-20260522/`

包含：

- metadata JSON
- TSGBench-style metric JSON/MD
- overlay PDF
- Slurm logs

远程输出：

- walking output: `/scratch/project_2016517/panh/time-series-llm/fms-dgt/output/time_series/pamap2_subject101_walking_hand_acc16_x_univariate_baseline/`
- running output: `/scratch/project_2016517/panh/time-series-llm/fms-dgt/output/time_series/pamap2_subject101_running_hand_acc16_x_univariate_baseline/`

## 当前判断

Baseline 状态：supported as clean baseline run / provisional for research claim。

可以说：

- 已用原始 subject101.dat 重建 walking-only 和 running-only 的 clean univariate SDForger baseline。
- walking/running 都能被 SDForger 周期性切窗逻辑处理，且生成样本数达到 baseline 目标。
- 该 baseline 可以作为周一汇报中的“已验证 baseline 起点”。

暂时不要说：

- 生成样本已经足够真实。
- 生成数据已经能提升 HAR 分类。
- SDForger 在 PAMAP2 上优于其他方法。

## 下一步

1. 人工查看 overlay PDF，标记 walking/running 是否保留明显周期形状。
2. 补一页 activity label 的意义：当前 baseline 是 activity-specific filtering，不是 activity-conditioned generation。
3. 做最小 HAR utility smoke：real-only vs synthetic-only vs real+synthetic，但必须先明确 scaled/raw value-space。
4. 若要继续方法改进，优先考虑 label/covariate conditioning 或 nonlinear embedding，而不是先做多数据集大扫。
