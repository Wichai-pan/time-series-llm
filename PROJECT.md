# Time-Series LLM Synthetic Sensor Data

状态：离线项目重置，2026-05-18。

这个目录是项目的 control root，用来统一管理项目记忆、文献、设计文档、代码指针，以及之后可能出现的论文或报告工作。已有本地代码和旧实验记录都先视为不完整、`needs-verification`，只有在远程输出、commit、日志和报告重新核验后，才能作为证据使用。

## 当前研究主题

使用 LLM-enabled 或 LLM-adjacent 的生成方法，为 human activity recognition (HAR) 生成合成传感器时间序列数据。

## 当前候选方向

围绕 `SDForger` 在 `PAMAP2/HAR` wearable sensor 数据上的适配与评估重新收窄题目：重点不是立刻跑更多实验，而是判断生成数据是否保留了统计结构、时间结构和下游 HAR utility。

## 当前目录结构

- `memory/`：跨 session 的项目状态、claim、risk、action 和 provenance。
- `reference/`：文献/source 库、processing status、source cards、project-use notes。
- `docs/`：项目级设计、文献计划、实验计划、更新、审计和时间线。
- `code/`：未来干净代码组件的占位和指针。
- `legacy/old-project-files/`：旧文件集中存放处，包括旧笔记、PDF、PPT、临时提取文本和残留代码 repo。
- `paper/`：未来论文或报告组件，占位。
- `code-worktrees/`、`paper-worktrees/`：未来 sibling worktree 根目录。

## 离线约束

当前不要尝试 SSH、SLURM、RunAI、远程同步或提交实验。远程核验等集群维护结束后再做。
