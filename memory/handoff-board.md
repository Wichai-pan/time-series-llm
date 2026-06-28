# Handoff Board

最后更新：2026-05-18

## HND-001：从 literature sprint 到 idea revision

- 状态：proposed
- Producer：literature-review-sprint
- Consumer：research-idea-validator / project planning
- From component：reference
- To component：project
- Payload：ranked literature map、closest-work risks、baseline implications 和 revised claim boundary。
- Source paths：
  - `reference/cards/`
  - `docs/literature/`
- Expected outputs：
  - 更新 `memory/claim-board.md`
  - 更新 `memory/risk-board.md`
  - 更新 `docs/designs/idea-reset.md`
- Acceptance check：至少得到一个 defensible minimal claim 和 must-check baselines。
- Linked claims：CLM-001、CLM-002、CLM-003
- Linked actions：ACT-005

## HND-002：从远程 artifacts 核验到 evidence board

- 状态：blocked
- Producer：future remote verification
- Consumer：evidence/provenance boards
- From component：code
- To component：project
- Payload：verified logs、configs、output files、report files、plots、commit hashes 和 rerun instructions。
- Expected outputs：
  - 升级或替换 EVD-001
  - 升级或替换 PRV-001
  - 修订 minimal experiment plan
- Acceptance check：每个 result number 都能追溯到 file、config、command 和 commit 或清楚的 source state。
- Linked actions：ACT-006、ACT-007

## HND-003：从 T2 实验设计到 run-experiment

- 状态：ready
- Producer：experiment-design-planner
- Consumer：run-experiment
- From component：experiment design
- To component：code / experiment
- Payload：可执行的 T2 stat-prompt 实验计划（假设、变量、对照、run matrix、stop conditions、reviewer risks）。
- Source paths：
  - `docs/experiments/experiment_plan_2026-06-15_stat-prompt.md`
  - 改动点：`puhti-generated/code_patch/fms_dgt/public/databuilders/time_series/generate.py`（179-183 条件注入、~390-424 生成 prompt）
  - run 模板：`slurm/run_sdforger_pamap2_unified_label_conditioned_gpu.sh`
  - 评估：`scripts/apply_smooth_latent_repair.py`、`scripts/rerun_smooth_repair_tsgbench_table.py`（+ 新增 stat-adherence 脚本）
- Expected outputs：R2 GPU SFT run、R2a(raw)/R2b(轻修复) TSG 表、stat-adherence 结果、`docs/experiments/` 报告、`memory/evidence-board.md` EVD-022。
- Acceptance check：stat-prompt-raw 的 abs-max/ACD 能与 raw-unified 和 T1 修复直接可比；统计量遵守度有量化结果（support 或 falsify 都可）。
- Linked actions：ACT-036
- 约束：GPU job，提交 sbatch 前向用户出示 diff + 命令（"操作前检查"）。
