# Puhti 旧项目只读盘点：time-series-llm

日期：2026-05-21

远程路径：`/scratch/project_2016517/panh/time-series-llm`

检查方式：只读 SSH 文件系统检查；未提交 Slurm job，未修改远程代码，未删除文件。

## 结论概览

- 目标路径存在，Puhti login node 为 `puhti-login15.bullx`。
- 远程目录不是 Git repo；`/scratch/project_2016517/panh/time-series-llm` 及 4 层内未发现 `.git`。因此不能从远程目录本身确定 branch、commit 或 dirty state。
- 真正代码主体在 `fms-dgt/`，是 IBM `fms-dgt` 框架的一份工作区拷贝，新增/包含 `time_series` data builder、PAMAP2 task、Slurm scripts、reports、logs 和 output artifacts。
- 旧结果文件、日志、plots、checkpoints 都存在，但由于缺少 Git provenance 和未重跑验证，研究结论仍保持 `provisional / needs-verification`。

## 顶层结构

`/scratch/project_2016517/panh/time-series-llm/`

- `fms-dgt/`：主要代码与实验工作区。
- `pylibs/`、`pylibs_dtw/`：看起来是临时/本地安装的 Python 依赖，包含 `numpy`、`dtaidistance`。
- `pamap2_compare.py`、`pamap2_stats.py`：顶层临时分析脚本，用于比较 PAMAP2 real/generated 的均值、方差、范围等。
- `eval2.py`、`eval3.py`、`eval_final.py`、`eval_quality.py` 及对应 txt：目前大小为 0，不能作为有效代码依据。

`fms-dgt/` 关键目录：

- `fms_dgt/public/databuilders/time_series/`：SDForger-style time-series generation 实现。
- `tasks/public/time_series/`：bikesharing、PAMAP2 subject101/102 的 task yaml。
- `scripts/`：PAMAP2 预处理、结果评估、run summary/model comparison 脚本。
- `slurm/`：运行 SDForger/PAMAP2/bikesharing 的 Slurm 脚本。
- `data/public/time_series/`：已准备好的 parquet 数据。
- `output/time_series/`：生成输出、模型 checkpoint、plots、jsonl。
- `reports/`：metric reports、summary reports、overlay plots。
- `logs/`：Slurm `.out` 日志。
- `mlruns/`：MLflow run 目录，存在多个 run id。

## Git 状态

观察结果：

- `git rev-parse --is-inside-work-tree` 在远程 root 没有返回 repo。
- `find . -maxdepth 4 -name .git -type d` 没找到 `.git`。

判断：

- 远程代码不能被直接锁定到 Git commit。
- 后续如果要把旧结果变成 evidence，必须先找到来源 repo/commit，或重新建立干净本地 repo + rerun。

## 主要脚本与配置

### PAMAP2 数据准备

- `scripts/prepare_pamap2_subject101.py`
  - 读取 PAMAP2 `Protocol/subjectXXX.dat`。
  - 默认保留 `hand_acc16_x` 单通道。
  - 过滤 `activity_id != 0`，线性插值缺失值，输出 `float32` parquet。
  - 支持通过 `--columns` 输出多通道。
- `scripts/prepare_pamap2_subject.py`
  - 兼容入口，调用 `prepare_pamap2_subject101.py`。

### SDForger 实现

- `fms_dgt/public/databuilders/time_series/generate.py`
  - pipeline：读取 seed data -> preprocessing/windowing/scaling -> FPC/FastICA embedding -> embedding-to-text -> LLM fine-tuning -> vLLM generation -> decode/filter -> generated time series -> plot。
- `fms_dgt/public/databuilders/time_series/trainer.py`
  - 使用 HuggingFace `Trainer` 做 causal LM fine-tuning。
  - 将 embedding row 转成文本，80/20 train-validation split。
- `fms_dgt/public/databuilders/time_series/utils.py`
  - preprocessing、periodicity/window construction、scaling、FPC/FastICA encode/decode、distribution filtering 等。

### PAMAP2 task yaml

- `tasks/public/time_series/pamap2_subject101_multivariate/task.yaml`
  - 数据：`data/public/time_series/pamap2_subject101_multivariate.parquet`
  - 通道：hand/ankle 的 6 个 `acc16` channels。
  - `train_length: 60000`
  - `augmentation_strategy: multivariate`
  - `train_splitting: no-periodicity`
  - `embedding_type: fica`
  - `variance_explained: 0.7`
  - `min_windows_length: 800`
- `tasks/public/time_series/pamap2_subject101_univariate_paper/task.yaml`
  - 数据：`data/public/time_series/pamap2_subject101_univariate_hand_acc16_x.parquet`
  - 通道：`hand_acc16_x`
  - `train_length: 5000`
  - `augmentation_strategy: univariate`
  - `embedding_type: fica`
  - `min_windows_number: 30`
  - `min_windows_length: 300`

### Slurm scripts

典型入口：

- `slurm/run_sdforger_pamap2_formal_gpu.sh`
- `slurm/run_sdforger_pamap2_subject101_univariate_paper_gpu.sh`
- `slurm/run_sdforger_pamap2_subject101_univariate_*`
- `slurm/run_sdforger_pamap2_subject102_*`

共同模式：

- `#SBATCH --account=project_2016517`
- `#SBATCH --partition=gpu`
- `#SBATCH --gres=gpu:v100:1`
- `module load pytorch/2.6`
- env：`/projappl/project_2016517/panh/time-series-llm/envs/sdforger-py312`
- code root：`/scratch/project_2016517/panh/time-series-llm/fms-dgt`
- `DGT_DATA_DIR=$BASE/data`
- HF cache：`/scratch/project_2016517/panh/hf_cache`
- command：`python -m fms_dgt.public --task-paths <task.yaml> --restart-generation`

## 数据、输出、日志、checkpoint

### 数据

`fms-dgt/data/public/time_series/` 中观察到：

- `pamap2_subject101_multivariate.parquet`，约 7.5 MB。
- `pamap2_subject101_univariate_hand_acc16_x.parquet`，约 1.3 MB。
- `pamap2_subject102_multivariate.parquet`，约 7.7 MB。
- `pamap2_subject102_univariate_hand_acc16_x.parquet`，约 1.4 MB。
- 还有 `bikesharing_full.parquet` 和 `monash_nn5_full.parquet`。

### 输出目录

`fms-dgt/output/time_series/` 中存在大量 PAMAP2 run：

- `pamap2_subject101_multivariate`
- `pamap2_subject101_multivariate_smoke`
- `pamap2_subject101_multivariate_thr095`
- `pamap2_subject101_multivariate_train120k`
- `pamap2_subject101_multivariate_train120k_win400`
- `pamap2_subject101_multivariate_var90`
- `pamap2_subject101_univariate_formal`
- `pamap2_subject101_univariate_paper`
- `pamap2_subject101_univariate_paper_gpt2`
- `pamap2_subject101_univariate_paper_phi2`
- `pamap2_subject101_univariate_paper_qwen1p5b`
- `pamap2_subject101_univariate_paper_smoke`
- `pamap2_subject101_univariate_smoke`
- `pamap2_subject101_univariate_thr095`
- `pamap2_subject101_univariate_train120k`
- `pamap2_subject101_univariate_train120k_win400`
- `pamap2_subject101_univariate_var90`
- `pamap2_subject102_multivariate_train120k`
- `pamap2_subject102_univariate_train120k_win400`

典型 run 内部包含：

- `data.jsonl`
- `dataloader_state.jsonl`
- `final_data.jsonl`
- `task_card.jsonl`
- `task_results.jsonl`
- `plot_generated_data.pdf`
- `logs/*.log`
- `model_iter-1/best/` 下的 `model.safetensors`、`config.json`、`tokenizer.json`、`training_args.bin` 等。

这些说明旧 generated outputs 和 checkpoints 存在，但还不能说明结果可复现或指标正确。

### Reports

`fms-dgt/reports/` 中观察到：

- `pamap2_subject101_run_summary.md/json`
- `pamap2_subject101_univariate_run_summary.md/json`
- `pamap2_subject101_univariate_paper_tsgbench_metrics.md/json`
- `pamap2_subject101_univariate_paper_*_tsgbench_metrics_scaled_cap100_shr.md/json`
- `pamap2_model_comparison.md/json`
- `pamap2_subject102_*_summary.md/json`
- overlay plots under `pamap2_subject101_univariate_plots/` 和 `pamap2_subject102_univariate_plots/`

报告里能看到旧 summary numbers。例如 univariate run summary 排序了 `formal`、`var90`、`thr095`、`train120k`、`train120k_win400`；paper metric report 包含 `MDD/ACD/SD/KD/ED/DTW`。

注意：这些数字目前只记录为旧结果存在，不能直接作为论文 claim。

### Logs

`fms-dgt/logs/` 中存在多个 PAMAP2 Slurm out：

- `sdforger-pamap2-formal_32666175.out`
- `sdforger-pamap2-smoke_32666126.out` 等 smoke logs
- `sdforger-pamap2-120k_32735124.out`
- `sdforger-pamap2-120k-w400_32735240.out`
- `sdforger-pamap2-u-paper_33833773.out`
- `sdforger-pamap2-u-gpt2_34286407.out`
- `sdforger-pamap2-u-qwen_34286406.out`
- `sdforger-pamap2-u-phi2_34285316.out`
- `eval_pamap2_*_3428*.out`

抽查 `sdforger-pamap2-u-paper_33833773.out` 显示该 job 完成 fine-tuning、vLLM generation、filtering，并保存了 `final_data.jsonl`。该日志支持“run 曾经完成并写出文件”的存在性判断，不支持“结果质量可信”的最终判断。

## 初步可信度判断

可以相对可信的事实：

- 远程路径存在。
- `fms-dgt/` 是主要工作区。
- 没有可见 Git metadata。
- PAMAP2 parquet、task yaml、Slurm scripts、outputs、logs、reports、model checkpoints 都存在。
- pipeline 形态确实是 SDForger-style：FPC/FastICA embedding -> text -> LLM fine-tuning/generation -> decode synthetic time series。

需要验证的部分：

- 远程 `fms-dgt/` 对应的原始 Git commit 和本地 legacy code 的关系。
- 旧 reports 的 metric 实现是否完全对应 SDForger/TSGBench 定义，尤其不同时间点的 report 可能来自不同版本脚本。
- `final_data.jsonl` 与对应 task yaml、log、checkpoint 是否一一匹配。
- PAMAP2 preprocessing 是否符合现在想做的 HAR task，尤其当前默认 parquet 丢弃了 `activity_id`，对 classification/label-conditioned generation 不够。
- 旧 output 位于 `/scratch`，长期可用性和完整性不能假设。
- 多个模型比较结果是否来自同一数据、同一 metric protocol、同一 synthetic-space 设置。

## 后续建议

1. 先把这次远程目录视为旧 artifact archive，不要当作 clean code repo。
2. 建立一个本地或远程 Git-controlled clean pipeline，再从旧目录迁移必要脚本。
3. 为每个可用旧 run 建一个 manifest：task yaml、Slurm log、output dir、final_data.jsonl、checkpoint、report、eval script version。
4. 如果要做 HAR utility，重新准备保留 `activity_id` 的 PAMAP2 windows，避免只生成无标签单通道序列。
5. 旧结果可以用于“下一步该检查什么”，不能直接用于最终 claim。
