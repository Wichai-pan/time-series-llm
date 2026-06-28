# Source Card：AgentSense

## Metadata

- Title: AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments
- Authors: Zikang Leng, Megha Thukral, Yaqi Liu, Hrudhai Rajasekhar, Shruthi K. Hiremath, Jiaman He, Thomas Ploetz
- Year: 2025 / 2026 copyright notice in PDF
- Venue/status: arXiv preprint / conference status 需要后续核验
- DOI/arXiv/URL: arXiv:2506.11773
- Source PDF: `reference/papers/AgentSense Virtual Sensor Data.pdf`
- Source type: paper-or-pdf
- Citation key: `leng2025agentsense`

## Reading Setup

- Reading mode: extract-method + extract-benchmark + extract-baseline + extract-risk
- Model tier: main model
- Reader: Codex
- Date: 2026-05-18
- Confidence: medium
- Source run: `reference/.agent/runs/2026-05-18-source-cards/agentsense.txt`

## Role Labels

- method-source: adjacent
- benchmark-source: yes, for HAR downstream utility
- baseline: yes, for real-only vs real+virtual training
- citation-support: yes
- closest-work: application-level closest work
- reviewer-risk: high

## Summary

### Problem

Smart-home HAR 缺少 large、diverse、annotated sensor datasets。真实采集成本高，且不同 home layouts、sensor configurations 和 resident routines 会影响模型泛化。

### Main idea

AgentSense 用 LLM-guided embodied agents 在 simulated smart homes 中生成 ambient sensor data。它不是直接生成连续 IMU/wearable sensor waveform，而是通过 persona、daily routine、VirtualHome actions 和 virtual ambient sensors 得到 smart-home sensor event streams。

Pipeline：

1. LLM 生成 diverse synthetic personas。
2. LLM 基于 persona、day of week 和 home environment 生成 daily schedules。
3. 高层活动被分解成 VirtualHome 可执行的 low-level actions。
4. 用 X-VirtualHome 执行动作。
5. 扩展 simulator，加入 motion、door、device activation sensors。
6. 记录 sensor trigger events，映射到目标 HAR dataset labels。
7. 用 virtual data pretrain HAR classifier，再用 real data finetune。

### Main contribution

- 提出 LLM-guided embodied simulation pipeline 来生成 smart-home ambient sensor data。
- 在多个真实 HAR datasets 上证明 virtual pretraining 可以提高 downstream HAR performance，尤其是低数据场景。
- 通过 ablation 展示 environment diversity、weekly routine coverage、persona variation 都有帮助。

## Method Details

### Persona and routine generation

LLM 生成包含 age、occupation、health、lifestyle 的 persona。再结合 day of week 和 home layout 生成日程。作者强调不使用过于整齐的时间点，以更像真实生活。

### Action grounding

高层 routine 被分解为 simulator-compatible actions。由于 LLM 会 hallucinate 或生成 simulator 不支持的 token，作者用 embedding + nearest-neighbor retrieval 将动作和 object grounding 到 VirtualHome vocabulary，并设置 threshold 过滤/重试。

### Virtual sensors

AgentSense 扩展 VirtualHome，模拟：

- motion sensors
- door sensors
- device activation sensors

这类数据是 ambient event stream，不是 PAMAP2 那种 continuous wearable accelerometer/gyro signal。

## Baselines

主要 baseline 不是 generative model baseline，而是 training setting baseline：

- Real: 只用真实数据训练/测试 HAR classifier。
- Real + Virtual: 先用 virtual data pretrain，再用真实数据 finetune，最后在真实 test split 上评估。

模型框架：

- TDOST-Basic
- TDOST-Temporal
- Bi-LSTM classifier over sentence-transformer embeddings of sensor-event descriptions

对当前项目的启发：

- 如果你要做 HAR augmentation utility，最自然的 comparison 也是 real-only vs real+synthetic。
- 低数据比例实验很有价值，例如 5%、10%、20% real data。

## Datasets

真实 datasets：

- Aruba
- Milan
- Kyoto7
- Cairo
- Orange4Home

这些主要是 smart-home ambient sensors，不是 wearable PAMAP2。

Virtual dataset：

- 18 personas
- 22 simulated home environments
- 250 days of activity data
- 3266 activity windows
- 每个 window 3 到 393 sensor triggers，平均 36

重要区别：

- AgentSense 是 ambient sensor HAR。
- 当前项目是 wearable HAR / continuous multivariate sensor windows。

## Evaluation

Metrics：

- Accuracy
- Macro F1
- Weighted F1

Training/evaluation settings：

- Real-only fully supervised baseline
- Real+Virtual pretrain then finetune
- final evaluation on real test split

核心结果：

- Real+Virtual 在五个 datasets 上总体优于 Real-only。
- Macro F1 在低资源 datasets 上改善明显。
- 低数据 ablation 显示 virtual pretraining 在 5%-10% real data 时仍有帮助。
- diversity ablation 显示 home environments、weekly routines、personas 都贡献性能提升。

## Limitations / Risks

- 它解决的是 smart-home ambient sensor data，不是 wearable continuous IMU/HAR。
- 它依赖 simulator 和 LLM-generated routines，不适合直接迁移到 PAMAP2 waveform generation。
- 它的强 claim 是 downstream HAR utility，而不是 waveform similarity。
- 当前项目如果只做 SDForger/PAMAP2 的 waveform generation，需要解释和 AgentSense 的区别。

## 对当前项目选题的启发

AgentSense 是当前项目的 application-level closest work。它给出一个很清楚的 HAR synthetic-data 证据标准：

- 不只生成数据，还要证明对 downstream HAR classifier 有帮助。
- 使用 real-only vs real+synthetic / real+virtual 对照。
- 使用 Accuracy、Macro F1、Weighted F1。
- 低数据实验能更好体现 synthetic data 的价值。

但当前项目不应该直接模仿 AgentSense pipeline，因为 sensor modality 不同。更合理的定位是：

> AgentSense 证明 LLM-guided synthetic sensor data 对 smart-home ambient HAR 有价值；本项目诊断 SDForger-style sequence generation 是否能在 wearable continuous HAR sensors 上达到类似目标。

## Claims To Avoid

- 不要说 AgentSense 和 SDForger 是同类方法。
- 不要把 AgentSense 的 downstream gains 直接类比到 PAMAP2。
- 不要在没有 classifier experiment 前声称本项目能改善 HAR。

## Provenance

- Pages/sections inspected: Abstract, Introduction, Related Work, Methodology, Virtual sensor implementation, Datasets, Classifier training, Results, Ablations, Conclusion.
- Extraction method: local `pdftotext -layout` + targeted section reading.
- Reviewed by: agent only.
