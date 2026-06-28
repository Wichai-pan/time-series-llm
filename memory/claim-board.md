# Claim Board

最后更新：2026-05-18

## CLM-001：SDForger 可以适配到 wearable HAR sensor time series

- 生命周期状态：evidence-needed
- Claim 类型：method / application
- 内容：SDForger 可能可以从通用多变量时间序列生成，适配到 PAMAP2 这类 wearable HAR sensor 数据，尤其是 acceleration channels。
- 需要的证据：已核验的 preprocessing、task configs、生成样本、可复现 run logs，以及真实/生成数据的统计和时间结构对比。
- 已核验证据：暂无
- 主要风险：RSK-001、RSK-002、RSK-004
- 相关 action：ACT-002、ACT-006、ACT-007
- 确定性：inferred

## CLM-002：生成样本保留了有用的统计和时间结构

- 生命周期状态：planned
- Claim 类型：empirical
- 内容：生成样本应保留足够的 distributional、temporal 和 multivariate structure，至少要优于 naive 或很弱的 synthetic baselines。
- 需要的证据：每通道统计量、ACF、PSD、多样性指标、跨 subject 检查和 baseline 对比。
- 已核验证据：暂无
- 主要风险：RSK-002、RSK-003、RSK-005
- 相关 action：ACT-003、ACT-004、ACT-007
- 确定性：inferred

## CLM-003：合成数据能改善或保持下游 HAR utility

- 生命周期状态：planned
- Claim 类型：empirical / application
- 内容：候选方法生成的数据，在 low-data 或 augmentation setting 下，可以改善或至少保持 HAR classifier performance。
- 需要的证据：下游分类协议、real-only baseline、synthetic-only 或 real+synthetic training、subject split、Accuracy/F1 和统计报告。
- 已核验证据：暂无
- 主要风险：RSK-003、RSK-005、RSK-006
- 相关 action：ACT-004、ACT-008
- 确定性：inferred

## 目前不要使用的强 claim

- “该方法优于所有 time-series generative baselines。”
- “旧的 `train120k` 结果已经证明是最佳配置。”
- “生成质量已经充分证明。”
- “该方向相对所有 HAR synthetic-data 工作都有明确 novelty。”

这些 claim 都必须等证据和文献覆盖完成后再考虑。
