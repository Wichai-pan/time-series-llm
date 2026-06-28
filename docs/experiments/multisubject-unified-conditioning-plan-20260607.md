# Multi-subject Unified Conditioning Experiment Plan

日期：2026-06-07

## 使用的 skills 和带来的效果

| Skill | 用途 | 带来的效果 |
|---|---|---|
| `experiment-design-planner` | 把“下一步做多 subject 还是继续归一化”变成可检验实验 | 明确 hypothesis、controls、run matrix、stop condition，避免随便堆实验 |
| `baseline-selection-audit` | 检查新旧 setting 是否公平比较 | 固定 baseline、metric space、数据/训练预算，避免把不同评估口径混在一起 |
| `research-project-memory` | 把实验计划写入项目长期状态 | 后续 session 可以从 action/evidence/risk 继续，不重新迷路 |

## 当前背景

已经完成：

1. `baseline`：subject101 walking/running 分开训练，clean SDForger univariate baseline。
2. `raw-unified`：subject101 walking/running 合并训练，prompt 加 activity label，但生成不稳定。
3. `clip_p05_p95`：post-hoc latent clipping，显著修复 raw-unified 的 ED/DTW/KD，但 ACD 仍差。
4. normalization ablation：`joint/global/activity/current zscore`，没有形成稳定整体改进。
5. canonical TSG protocol：所有 setting 统一到 `activity_sdforger_scaled` 后计算 `MDD/ACD/SD/KD/ED/DTW`。

因此下一步不应继续堆 normalization variant，而应测试一个更基础的问题：

> unified conditional generation 是否主要受限于 subject101 数据太少？

执行前数据检查发现：subject103 没有 running(activity 5)，因此原计划的 101-103 不平衡。实际采用的最小平衡组合改为 subject101/102/105。

## 实验问题

在 PAMAP2 walking/running `hand_acc16_x` 上，将训练数据从 subject101 扩展到 subject101/102/105，是否能稳定 unified activity-conditioned SDForger generation？

## Primary hypothesis

如果 failure 主要来自 subject101 数据量太少，那么 subject101/102/105 多 subject 训练应当：

- 降低 raw-unified 的 ED/DTW/KD 爆炸程度；
- 提高或保持 label controllability；
- 让 clip / validity control 后的 ACD 更接近 baseline；
- raw-like amplitude ratio 不再出现几百倍或几千倍 outlier。

## Alternative explanations

| Explanation | 如果为真，会看到什么 |
|---|---|
| 数据量不足是主因 | 多 subject raw-unified 明显比 subject101 raw-unified 稳定 |
| embedding / decoding 机制是主因 | 多 subject 后仍然 value outlier / ACD 崩 |
| subject 间差异引入新噪声 | 多 subject 让 MDD/ACD/DTW 更差，尤其 walking |
| prompt/LLM parsing 是主因 | label controllability 或 accepted rows 仍不稳定 |

## 数据设计

### Stage 1：within-train-subject smoke

| Field | Value |
|---|---|
| Dataset | PAMAP2 |
| Subjects | 101, 102, 105 |
| Activities | walking, running |
| Channel | `hand_acc16_x` |
| Train data | 每个 subject/activity 取前 `train_length=5000` raw points |
| Window length | 300 |
| Windowing | 每个 subject/activity 独立按 SDForger periodicity preprocessing 切 windows，然后 activity-level 合并 |
| Evaluation real reference | 同一 subjects 的 real training windows, canonical scaled |
| Purpose | 判断增加 subject/data 是否改善 generation stability |

### Stage 2：unseen-subject check

等 Stage 1 显示有希望后再做：

| Field | Value |
|---|---|
| Train subjects | 101, 102, 105 |
| Test subject | 104 |
| Metric | HAR utility / classifier transfer, plus raw-like sanity |
| Purpose | 回答导师关心的 unseen subject 问题 |

当前先不要直接做 Stage 2，因为如果 Stage 1 的 generator 仍不稳定，unseen subject utility 没有解释价值。

## Run matrix

| Run ID | Purpose | Change | Fixed controls | Dataset/split | Primary metrics | Expected result |
|---|---|---|---|---|---|---|
| R0 | 已有参考 | subject101 `baseline` | channel/window/model/eval fixed | subject101 | canonical TSG | Clean reference |
| R1 | 已有失败参考 | subject101 `raw-unified` | same | subject101 | canonical TSG + label acc | Failure baseline |
| R2 | 已有修复参考 | subject101 `clip_p05_p95` | same | subject101 | canonical TSG + label acc | Post-hoc repair |
| R3 | 新实验核心 | multi-subject raw-unified | same model/prompt/eval | subjects101/102/105 | canonical TSG + label acc + amplitude | Test data-size effect |
| R4 | 新实验修复 | multi-subject clip_p05_p95 | same clipping rule | subjects101/102/105 | canonical TSG + label acc + amplitude | Test if clipping still needed |
| R5 | 方法升级，可选 | multi-subject generation-time validity control | same train data | subjects101/102/105 | canonical TSG + accepted/rejected counts | Replace post-hoc clip |

先执行 R3 和 R4。R5 等 R3/R4 结果出来后再决定。

## 必须固定的 controls

| Control | Fixed value |
|---|---|
| LLM backbone | `gpt2` |
| SDForger embedding | FICA |
| Prompt format | `Condition: data is walking/running [sep] Input ... Target:` |
| Window length | 300 |
| Channel | `hand_acc16_x` |
| Activities | walking/running only |
| Generated samples | 50-100 per label if parser allows |
| Evaluation | canonical `activity_sdforger_scaled` TSG |
| Random seed | 42 first; later add seeds only if result looks promising |

## Main metrics

主指标必须是 canonical TSG：

| Metric | Direction | Why |
|---|---|---|
| MDD | lower | value distribution |
| ACD | lower | autocorrelation / rhythm |
| SD | lower | skewness |
| KD | lower | kurtosis / outlier shape |
| ED | lower | paired shape distance |
| DTW | lower | time-warped shape distance |

辅助指标：

| Metric | Direction | Why |
|---|---|---|
| amplitude ratio | close to 1 | 防止 raw-like value explosion |
| label controllability | higher | 检查 prompt label 是否被模型使用 |
| accepted/generated count | higher accepted, transparent rejected | 检查 parser 和 validity control 是否稳定 |
| HAR utility | higher | Stage 1 后再看，避免过早解释 |

## Fairness ledger

| Comparison | Data | Model | Compute | Protocol | Metric | Verdict |
|---|---|---|---|---|---|---|
| subject101 raw-unified vs multi-subject raw-unified | only subject count changes | same | same target budget | same prompt/window/eval | same canonical TSG | fair |
| subject101 clip vs multi-subject clip | subject count changes only | same | same post-hoc rule | same eval | same canonical TSG | fair |
| baseline vs multi-subject raw-unified | different training objective | same backbone but different task | similar | baseline is reference, not direct method-win claim | same canonical TSG | fair-with-caveat |
| normalization variants vs multi-subject | different intervention | same backbone | similar | compare as ablation, not final ranking | same canonical TSG after conversion | fair-with-caveat |

## Stop conditions

### Continue multi-subject direction if

At least one of R3/R4 shows:

- ED/DTW/KD clearly better than subject101 raw-unified;
- no severe raw-like amplitude explosion;
- label controllability remains above 0.75;
- ACD does not get worse than subject101 clip.

### Move to generation-time validity control if

R3 still has outliers but R4 is stable:

- This means data helps only after constraints.
- Next method should be validity control / rejection sampling, not more normalization.

### Park multi-subject expansion if

R3 and R4 both fail:

- ED/DTW/KD remain bad;
- label controllability collapses;
- amplitude ratio remains extreme.

Then likely issue is FICA/joint embedding design or prompt decoding, not data quantity.

## Execution plan

### Step 1：prepare multi-subject data

Create:

- `pamap2_subject101_102_105_walking_hand_acc16_x.parquet`
- `pamap2_subject101_102_105_running_hand_acc16_x.parquet`

Implementation detail:

- Read `subject101.dat`, `subject102.dat`, `subject103.dat`.
- Filter activity_id 4 and 5.
- Interpolate missing values per subject/activity.
- Keep one column: `hand_acc16_x`.
- Concatenate within activity.
- Save metadata with subject list and per-subject row counts.

### Step 2：run multi-subject raw-unified

Use adapted `run_unified_label_conditioning.py`:

- walking parquet = multi-subject walking
- running parquet = multi-subject running
- output = `pamap2_subject101_102_105_unified_label_conditioned_hand_acc16_x`

### Step 3：apply same clip rule

Use `apply_unified_latent_constraints.py` or adapted version:

- variant = `clip_p05_p95`
- output = `pamap2_subject101_102_105_unified_label_conditioned_hand_acc16_x_constraints/clip_p05_p95`

### Step 4：evaluate

Use canonical TSG evaluator:

- include old subject101 rows
- include new multi-subject raw-unified
- include new multi-subject clip

Also run:

- raw-like amplitude sanity
- label controllability

## Expected table for reporting

| Setting | Subject scope | Activity | MDD | ACD | SD | KD | ED | DTW | Label acc | Amp ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 101 | walking | ... | ... | ... | ... | ... | ... | n/a | ... |
| raw-unified | 101 | walking | ... | ... | ... | ... | ... | ... | ... | ... |
| clip_p05_p95 | 101 | walking | ... | ... | ... | ... | ... | ... | ... | ... |
| raw-unified | 101/102/105 | walking | ... | ... | ... | ... | ... | ... | ... | ... |
| clip_p05_p95 | 101/102/105 | walking | ... | ... | ... | ... | ... | ... | ... | ... |

Same for running.

## Recommended immediate next action

Execute Step 1-4 for R3/R4 only.

Do not yet:

- add more channels;
- add more activities;
- add subject104 unseen test;
- add new normalization variants.

This keeps the next experiment scientifically interpretable:

> The only major new variable is more subjects / more windows.
