# 初始 Project-Use Synthesis

最后更新：2026-05-18

## 综合判断

三个 seed sources 指向三种不同项目框架：

- SDForger：LLM-enabled synthetic time-series generation。
- ChatTS：time-series understanding/reasoning with MLLMs and synthetic alignment data。
- AgentSense：LLM-guided simulated sensor data for HAR。

当前 reset 下，最干净的框架不是“所有 LLMs for time series”，而是更窄的经验问题：

> SDForger-style generation pipeline 能否作为 wearable HAR synthetic-data baseline 被适配、评估和诊断？

## 对项目的影响

- 保持 SDForger 为 primary method seed。
- 用 AgentSense 作为 application-area related work 和 reviewer-risk boundary。
- 用 ChatTS 作为 adjacent LLM-time-series context，而不是主方法线。
- 在 labels、splits 和 classifier protocol 明确之前，不要声称 downstream HAR utility。
- 在 recent HAR synthetic-data 和 time-series generation baselines 梳理完成前，不要声称 novelty。

## Memory links

- Claims：CLM-001、CLM-002、CLM-003
- Risks：RSK-003、RSK-004、RSK-005、RSK-006
- Actions：ACT-003、ACT-005、ACT-008
