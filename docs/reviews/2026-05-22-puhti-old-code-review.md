# Puhti 旧 SDForger/PAMAP2 代码审阅

日期：2026-05-22

远程路径：`/scratch/project_2016517/panh/time-series-llm/fms-dgt`

检查方式：只读 SSH 审阅；未提交 job，未修改远程代码。

## Verdict

结论：`request changes before using as final evidence`

旧代码**有用**，可以作为：

- SDForger/PAMAP2 baseline 的恢复起点。
- 周一汇报里的 recovered baseline candidate。
- 下一步 `+ activity label` 实验的对照基础。

但旧代码/旧结果目前**不能直接作为最终证据**，原因是：

- 远程目录没有 Git provenance。
- 旧 PAMAP2 parquet 不含 `activity_id`。
- 生成 JSONL 保存的是标准化窗口空间，不是原始 sensor 单位。
- multivariate `embedding_dim: auto` 有明显实现风险。
- time-series builder 没有专门测试覆盖。

## High Findings

### High: 旧 PAMAP2 数据丢掉了 activity label，不能直接支持 HAR / activity-aware generation

- 远程文件：`scripts/prepare_pamap2_subject101.py:69,121-140`
- 远程数据：
  - `data/public/time_series/pamap2_subject101_univariate_hand_acc16_x.parquet` shape `(249957, 1)`，columns `['hand_acc16_x']`
  - `data/public/time_series/pamap2_subject101_multivariate.parquet` shape `(249957, 6)`，columns 只有 hand/ankle acceleration channels
- 问题：预处理脚本默认只保留 sensor columns，并且输出 parquet 不保留 `activity_id`。
- 影响：旧 pipeline 可以评估 waveform generation，但不能直接评估“按活动类别生成”、HAR utility、activity-conditioned prompting。
- 需要修正：新建 HAR-aware preprocessing，保留 `activity_id`，并按 activity / subject / split 生成 windows。
- 建议测试：读 parquet schema，断言 `activity_id` 存在；检查每个 window 的 label 来源一致或定义好 mixed-label 处理。

### High: `final_data.jsonl` 是标准化窗口空间，不是原始传感器单位

- 远程文件：
  - `fms_dgt/public/databuilders/time_series/generate.py:210-245`
  - `fms_dgt/public/databuilders/time_series/generate.py:617-673`
  - `scripts/evaluate_sdforger_paper_metrics.py:16-18,83-91,705-708`
- 问题：FICA/FPC decode 后得到的是 standardized SDForger window space。plot 函数会 inverse-transform 用于画图，但返回并保存到 `final_data.jsonl` 的仍是 `generated_data`，即标准化空间。
- 影响：如果后续 classifier 期望 raw PAMAP2 scale，直接用 `final_data.jsonl` 会错。旧 evaluation 用 `--synthetic-space=scaled` 时可以自洽，但必须明确说明。
- 需要修正：输出中显式记录 `value_space: scaled`，或另存 raw-scale synthetic JSONL；classification pipeline 必须使用一致的 normalization。
- 建议测试：对一个 generated window inverse-transform 后检查统计范围是否接近 raw PAMAP2；评估脚本应拒绝未声明 value space 的输入。

### High: multivariate `embedding_dim: auto` 只会按第一个通道确定维度

- 远程文件：
  - `fms_dgt/public/databuilders/time_series/utils.py:414-484`
  - `fms_dgt/public/databuilders/time_series/utils.py:487-561`
- 问题：`embedding_dim` 在 channel loop 内被从 `"auto"` 改成整数；后续 channel 不再单独 auto-select，而是复用第一个 channel 的维度。
- 影响：multivariate PAMAP2 baseline 的 channel-wise representation 可能不可靠，尤其不同 sensor/channel 的复杂度不同。旧 multivariate reports 不宜作为强证据。
- 需要修正：在每个 channel 内使用局部变量，例如 `embedding_dim_var`，不要覆盖全局配置。
- 建议测试：构造两个方差结构不同的 synthetic channels，断言 auto embedding dims 可以不同。

## Medium Findings

### Medium: generation error handling 可能吞掉异常并返回部分结果

- 远程文件：`fms_dgt/public/databuilders/time_series/generate.py:431-615`
- 问题：generation 主循环被 broad `except Exception` 包住，只记录 error；如果已有部分 `dfs`，函数后续仍可能返回部分数据。
- 影响：旧 logs/outputs 需要逐个核对，不能只看 `final_data.jsonl` 存在。
- 需要修正：生成失败时应 fail fast，或在 output metadata 中写 `generation_status` / `error`。
- 建议测试：mock malformed LLM output，断言任务失败或明确标记 incomplete，而不是静默产出。

### Medium: 当前评估指标可以作为 similarity baseline，但不能替代 HAR utility

- 远程文件：`scripts/evaluate_sdforger_paper_metrics.py:551-603,684-769`
- 观察：脚本实现了 MDD、ACD、SD、KD、ED、DTW 和可选 SHR，并明确记录 `synthetic_space`。
- 限制：这些指标主要衡量分布/自相关/距离，不判断生成样本是否保留 activity semantics。
- 需要补充：real-trained classifier on generated、synthetic-trained classifier on real、class-conditional nearest-neighbor 等 HAR utility 指标。

### Medium: 缺少 time-series builder 的专门测试

- 远程观察：`tests/` 下只有 base/core dataloader/datastore tests，没有 `time_series` builder、PAMAP2 preprocessing、FICA/FPC encode-decode、JSONL parse 的测试。
- 影响：小改 prompt/label/embedding 时容易破坏 shape 或 value-space contract。
- 需要补充：最小 unit tests + one tiny end-to-end smoke with fake data。

## Low Findings

### Low: 顶层临时脚本和空文件容易混淆 provenance

- 远程路径：`/scratch/project_2016517/panh/time-series-llm/`
- 观察：`eval2.py`、`eval3.py`、`eval_final.py`、`eval_quality.py` 为空；`pamap2_compare.py`、`pamap2_stats.py` 是 ad hoc analysis。
- 建议：正式使用时迁移到 `scripts/analysis/`，并给每个 report 记录生成命令。

## 是否正确

工程逻辑上，旧 SDForger pipeline 的大方向是正确的：

```text
PAMAP2 parquet
  -> windowing / scaling
  -> FICA/FPC embedding
  -> embedding text
  -> LLM fine-tuning
  -> generated embedding text
  -> parse / filter
  -> inverse FICA/FPC
  -> synthetic windows
```

日志和 `task_results.jsonl` 显示至少 `pamap2_subject101_univariate_paper` 这类 run 曾经完成并生成 `final_data.jsonl`。

但它现在只能算“技术上跑通”，不能算“研究上已证明有效”。

## 是否有用

有用，分三层：

1. **可立即用于汇报**：作为 recovered SDForger/PAMAP2 baseline candidate，说明已有 pipeline、outputs、logs、reports。
2. **可作为下一步 baseline**：在修正 provenance/value-space/label 之后，作为 `without activity label` 的 baseline。
3. **不可直接作为 final evidence**：旧结果缺 Git provenance，且当前数据不支持 activity-conditioned/HAR utility claim。

## 周一前最实际的使用方式

建议汇报时这样表述：

> We recovered an SDForger-style PAMAP2 baseline from Puhti. The pipeline appears executable and generated outputs exist, but the old artifacts are treated as provisional because the remote directory lacks Git provenance and the prepared PAMAP2 files do not include activity labels. Therefore, our next controlled experiment is to rebuild a label-aware PAMAP2 preprocessing pipeline and compare original SDForger prompts against activity-conditioned prompts.

## 下一步

1. 固定一个 baseline run manifest：`task.yaml`、Slurm log、`final_data.jsonl`、checkpoint、report、eval script。
2. 新建 label-aware PAMAP2 parquet/window dataset，保留 `activity_id`。
3. 修正 output value-space contract：明确 scaled/raw，并保证 classifier 使用一致空间。
4. 修正 `embedding_dim: auto` 的 channel-wise bug。
5. 做最小对照：original SDForger prompt vs `+ activity label` prompt。
