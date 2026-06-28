# Claude / Sidecar Agent 说明

集群维护已结束。用户于 2026-06-15 明确授权远程操作（含 `ssh puhti` 只读检查、`rsync` 同步代码、`sbatch` 提交 job）。

**用户要求的标准约束（每次远程操作都适用）**：在执行任何远程写操作（同步、提交、修改、删除）之前，必须先做一次只读检查，确认该操作安全、范围正确，不会破坏集群上其他文件、其他人的数据或已有 run 输出。不做未经确认的删除、覆盖或全局性改动。

项目上下文以 `AGENTS.md` 和 `memory/current-status.md` 为准。旧代码 / 旧 run summaries / 旧 plots 仍按 `needs-verification` 对待，进入论文 claim 前需重跑或重算。
