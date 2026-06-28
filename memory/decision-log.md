# Decision Log

## DEC-001：先离线 reset，再考虑新实验

- 日期：2026-05-18
- 状态：active
- 决策：先重建项目 memory、reference structure 和 research direction，不运行或提交新实验。
- 原因：用户说明集群不可用、本地代码不完整，旧结果都应视为 `needs-verification`。
- 结果：本阶段不做 SSH、SLURM、RunAI、远程同步或实验提交。
- 确定性：user-stated

## DEC-002：把 `SDForger + PAMAP2/HAR` 作为候选方向，而不是最终 claim

- 日期：2026-05-18
- 状态：active
- 决策：将恢复出的 SDForger/PAMAP2 线作为 leading candidate，但保持 revised idea 状态，先做文献和证据检查。
- 原因：旧笔记显示曾做过 SDForger、PAMAP2、subject101、subject102 和多组参数变体，但 artifacts 当前未核验。
- 结果：相关 claims 保持 `evidence-needed` 或 `needs-verification`。
- 确定性：inferred
