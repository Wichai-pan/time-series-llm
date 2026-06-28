# Agent 工作说明

默认从本 control root 工作，除非任务明确指定某个组件目录。

集群已恢复，用户于 2026-06-15 授权远程操作（SSH / SLURM / rsync / sbatch）：

- 旧代码、旧远程路径、旧 run summaries 和旧 plots 仍视为 `needs-verification`。
- 远程操作已授权，但每次写操作（同步 / 提交 / 修改 / 删除）前必须先只读检查，确认安全、不破坏集群上的其他文件或他人数据。
- 不要把旧 result counts 直接写成论文 claim。
- 长期状态写入 `memory/`；source library 状态写入 `reference/.agent/`。
- 原始论文和协作者文件放在 `reference/` 或 `legacy/old-project-files/` 中；memory 中只保留摘要、指针和判断，不复制长篇原文。

## 下一次 session 入口

1. 先读 `memory/current-status.md`。
2. 再读 `docs/designs/idea-reset.md`。
3. 向用户补齐 `memory/action-board.md` 中列出的缺失材料。
4. 只有在远程访问恢复后，才核验 `memory/risk-board.md` 和 `memory/action-board.md` 中列出的远程路径和旧输出。
