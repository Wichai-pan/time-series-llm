# 离线 Literature Sprint Plan

最后更新：2026-05-18

这是一个 provisional offline plan。之后允许联网或有新材料时，需要用 primary sources 更新。

## Sprint 问题

判断把 SDForger-style LLM-based synthetic time-series generation 适配到 HAR wearable sensor data 是否有 novelty、utility 和可评估性；比较对象包括 time-series generation、LLM-time-series 和 HAR synthetic-data 相关工作。

## Search log

| 日期 | Source | Query / path | Useful hits | Notes |
|---|---|---|---|---|
| 2026-05-18 | local files | `legacy/old-project-files/Forging Time Series Synthetic Data.pdf` / `legacy/old-project-files/tmp_forging.txt` | SDForger | 本地 seed |
| 2026-05-18 | local files | `legacy/old-project-files/ChatTS Time Series Alignment.pdf` / `legacy/old-project-files/tmp_chatts.txt` | ChatTS | 本地 seed |
| 2026-05-18 | local files | `legacy/old-project-files/AgentSense Virtual Sensor Data.pdf` / `legacy/old-project-files/tmp_agentsense.txt` | AgentSense | 本地 seed |
| 2026-05-18 | local notes | `legacy/old-project-files/TS LLM.md`, `legacy/old-project-files/SESSION_HANDOFF.md` | old project state | needs verification |

## Read-now queue

| Paper/source | 为什么现在读 | 要回答的问题 | 影响的决策 |
|---|---|---|---|
| SDForger | closest method seed | HAR/PAMAP2 适配是否真有新东西？ | claim boundary |
| TS generation benchmark source | metric standard | quality 和 utility 的必备 metrics 是什么？ | evaluation plan |
| AgentSense | HAR synthetic-data adjacent | 本项目是在和 HAR simulation 竞争，还是 wearable generation？ | positioning |
| TimeGAN/TimeVAE/TimeVQVAE/diffusion family | baseline family | 哪个 baseline 公平且可行？ | baseline policy |
| PAMAP2/HAR classifier papers | downstream protocol | 什么 split 和 metrics 可接受？ | utility claim |

## 输出目标

在 `docs/literature/` 下写 literature map，并把稳定的风险和 action 提升到：

- `memory/risk-board.md`
- `memory/action-board.md`
