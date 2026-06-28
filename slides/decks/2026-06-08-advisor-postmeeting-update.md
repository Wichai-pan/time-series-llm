---
theme: default
title: SDForger 在 PAMAP2 上的会后验证实验
info: Two-slide advisor update on normalization, multi-subject, unseen-subject, and latent validity diagnostics.
class: text-left
drawings:
  persist: false
transition: none
---

<style>
.slidev-layout table {
  font-size: 0.78rem;
}
.slidev-layout th,
.slidev-layout td {
  padding: 0.34rem 0.42rem;
}
.note {
  color: #475569;
  font-size: 0.82rem;
}
.takeaway {
  margin-top: 0.75rem;
  padding: 0.55rem 0.75rem;
  border-left: 4px solid #0f766e;
  background: #ecfdf5;
  color: #134e4a;
  font-weight: 650;
}
</style>

# SDForger 在 PAMAP2 上的会后验证实验

Advisor update · 两页进度汇报 · 2026-06-08

::div{class="grid grid-cols-2 gap-6 mt-6"}

::div
### 当前任务

- Activity-conditioned sensor generation
- 数据集：PAMAP2
- 动作：walking / running
- 通道：`hand_acc16_x`
- 基础流程：SDForger + FICA latent + LLM generation
::

::div
### 主要问题

为什么 unified activity-conditioned generation 会不稳定？

哪种约束能让生成结果稳定到可以继续做下一步评估？

::p{class="note"}
当前结果是 diagnostic verification，不是最终 HAR augmentation claim。
::
::

::

| 实验 | 设置 | 目的 | 当前发现 |
|---|---|---|---|
| 归一化 | window / joint / global / activity-level z-score | 检查 scale mismatch 是否导致数值爆炸 | 方向合理，但单靠 normalization 没有稳定 unified generation |
| Multi-subject | train subjects 101 / 102 / 105 | 检查 subject101 数据量太少是不是主因 | 增加 subject 后，raw unified 仍然不稳定 |
| Unseen subject | reference subjects 106 / 108 | 检查 synthetic windows 是否只贴近训练 subject | clip 在 held-out reference 上仍稳定，但还不是 HAR utility 结论 |
| Latent validity | post-hoc clip / strict reject / soft repair | 在 decode 前控制无效 latent 数值 | simple clip 是最强 diagnostic baseline；soft repair 更适合作为方法候选 |

::div{class="takeaway"}
目前看，问题不只是数据量或归一化；unified generation 需要 latent validity control。
::

<!--
Speaker notes:
这页先讲“这次做了什么”，不要一开始陷入指标。

我们这次沿着老师上次反馈做了四组检查：归一化、多 subject、unseen subject、以及不同的 clip/validity control。

重点不是说我们已经有最终方法，而是把问题定位得更清楚了：raw unified activity-conditioned generation 容易产生无效 latent，经过 FICA decode 后会变成极端异常波形。
-->

---

# 主要结果：clip 能明显稳定 unified generation

TSGBench-style diagnostic metrics · `DTW` 越低越好

| 设置 | 参考真实数据 | Walking DTW ↓ | Running DTW ↓ | 解释 |
|---|---|---:|---:|---|
| raw unified | train 101/102/105 | 9920.9 | 7943.9 | 多 subject 后仍然生成不稳定 |
| clip p05-p95 | train 101/102/105 | 166.1 | 147.2 | 简单 latent clip 把 DTW 拉回合理范围 |
| raw unified | held-out 106/108 | 11160.4 | 9567.4 | 面对 unseen-subject reference 时仍失败 |
| clip p05-p95 | held-out 106/108 | 151.1 | 146.7 | clip 在 unseen-subject reference 上仍稳定 |
| soft repair | held-out 106/108 | 140.0 | 157.2 | 更可解释，但目前还没有明显优于 simple clip |

::div{class="grid grid-cols-2 gap-6 mt-6"}

::div
### 可以稳妥汇报的结论

`clip_p05_p95` 是目前稳定 PAMAP2 上 unified activity-conditioned SDForger 的 strong diagnostic baseline。

结果支持把 latent validity control 作为下一步方法方向。
::

::div
### 下一步讨论

1. 从 post-hoc clip 推进到 generation-time validity + resampling。
2. 报告 clean / repaired / rejected / malformed counts。
3. 生成质量稳定后，再补 held-out HAR utility。
::

::

::div{class="takeaway"}
下一步关键决策：是否把 simple clipping 扩展成更正式的 soft validity control，并用 held-out HAR utility 评估。
::

<!--
Speaker notes:
这页只讲 DTW，是为了避免表格太复杂。完整指标还有 MDD、ACD、SD、KD、ED、DTW，但是汇报时先用 DTW 说明核心现象最清楚。

raw unified 的 DTW 非常大，说明生成曲线和真实 walking/running window 差很远。多 subject 后仍然很大，说明不是简单因为 subject101 数据少。

clip p05-p95 的含义是：LLM 生成 FICA latent 后，把每一维限制在训练 latent 的 5% 到 95% 分位数范围内，再 decode 回 sensor window。它不是最终方法，但能明显压住极端 latent，说明 failure mode 很可能发生在 generated latent validity 上。

unseen reference 是用 106/108 的真实数据做参考，不是重新训练。这个结果不能说已经证明泛化，但能说明 clip 后的 synthetic windows 不是只贴近训练 subject。

soft repair 比 clip 更像一个方法：先判断 latent 是否超出合理范围，能修就修，太离谱就拒绝。但现在它没有全面超过 simple clip，所以后续需要配合 resampling，补回样本数，再做 HAR utility。
-->
