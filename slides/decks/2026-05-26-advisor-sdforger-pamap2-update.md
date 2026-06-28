---
theme: default
title: Activity-Conditioned SDForger on PAMAP2
info: Advisor progress update for PAMAP2 activity-conditioned sensor generation.
class: text-left
---

<style>
.slidev-layout {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.kicker {
  color: #64748b;
  font-size: 0.9rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}
.takeaway {
  margin-top: 0.8rem;
  padding: 0.55rem 0.75rem;
  border-left: 4px solid #0f766e;
  background: #ecfdf5;
  color: #134e4a;
  font-weight: 650;
}
.twocol {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.1rem;
  align-items: start;
}
.card {
  border: 1px solid #d8dee9;
  border-radius: 8px;
  padding: 0.8rem 0.9rem;
  background: #fff;
}
.card h3 {
  margin: 0 0 0.45rem 0;
  font-size: 1.05rem;
}
.small {
  font-size: 0.82rem;
  color: #475569;
}
.metric-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.86rem;
}
.metric-table th {
  color: #0f172a;
  background: #f1f5f9;
  font-weight: 700;
}
.metric-table td,
.metric-table th {
  border: 1px solid #d8dee9;
  padding: 0.42rem 0.48rem;
}
.metric-table td.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.figure-wide {
  width: 100%;
  border: 1px solid #d8dee9;
  border-radius: 8px;
}
.figure-label {
  margin-top: 0.25rem;
  font-size: 0.75rem;
  color: #64748b;
}
.pipeline {
  padding: 0.65rem 0.85rem;
  border: 1px solid #d8dee9;
  border-radius: 8px;
  background: #f8fafc;
  font-size: 0.86rem;
  line-height: 1.65;
}
</style>

# Activity-Conditioned SDForger on PAMAP2

<div class="kicker">Advisor update · 3-slide progress summary</div>

<div class="twocol" style="margin-top:1.1rem;">
<div class="card">
<h3>Previous recovered setup</h3>

- PAMAP2 subject101
- Mixed non-zero activities
- Single channel: `hand_acc16_x`
- No activity label in processed parquet
- Unconditioned SDForger-style generation

</div>

<div class="card">
<h3>Current controlled setup</h3>

- PAMAP2 subject101
- Walking / running only
- Single channel: `hand_acc16_x`
- Activity text as condition
- Baseline + conditional generation

</div>
</div>

<img class="figure-wide" style="margin-top:0.8rem; max-height:260px; object-fit:contain;" src="./assets/2026-05-26-advisor-sdforger-pamap2-update/running-periodicity.png" />
<div class="figure-label">Running was selected as a controlled periodic HAR signal; walking is the paired activity in the current setup.</div>

<div class="takeaway">We moved from recovering a mixed-activity baseline to a controlled HAR generation task.</div>

<!--
Speaker notes:
The old Puhti code is useful, but it used a processed PAMAP2 file where all non-zero activities were mixed and the activity labels were not kept. That setup can show that the SDForger pipeline runs, but it cannot directly answer whether we can generate a requested HAR activity.

For the current work, I narrowed the setting to subject101, walking and running, and one interpretable channel, hand_acc16_x. The point is not that this is the final HAR benchmark. The point is to isolate a clean activity-conditioned generation problem before moving to multi-subject or multichannel data.
-->

---

# Activity conditioning works, but the unified generator is unstable

<div class="twocol">
<div>
<div class="pipeline">
Condition: data is walking/running<br/>
+ FICA embedding text<br/>
→ LLM fine-tuning / generation<br/>
→ decoded synthetic sensor window
</div>

<table class="metric-table" style="margin-top:0.8rem;">
<thead>
<tr><th>Diagnostic</th><th>Raw unified</th></tr>
</thead>
<tbody>
<tr><td>Label controllability</td><td class="num">0.8212</td></tr>
<tr><td>Walking abs max</td><td class="num">3761.8</td></tr>
<tr><td>Running abs max</td><td class="num">1572.1</td></tr>
<tr><td>Synthetic-only HAR</td><td class="num">0.4955</td></tr>
</tbody>
</table>
</div>

<div>
<img class="figure-wide" src="./assets/2026-05-26-advisor-sdforger-pamap2-update/running-unified-raw-acf-psd.png" />
<div class="figure-label">Unified label-conditioned generation before latent constraint.</div>
</div>
</div>

<div class="takeaway">The model learns activity-label signal, but unconstrained generated latents can leave the valid sensor range.</div>

<!--
Speaker notes:
This follows the language-shaping idea from SDForger Section 6. In the paper, the condition is channel identity, such as temp, count, or humidity. Here, I adapt the same idea to activity identity: walking or running.

I tested two versions. First, activity-specific models: walking-only and running-only with activity text. Second, a unified walking/running model, where one model is asked to generate the requested activity. The unified version is the more interesting one because it tests true conditional generation.

The classifier-based controllability score is not a TSGBench metric. It asks: if I request walking, does the generated window look walking-like to a classifier trained on real data? The raw unified model gets 0.8212, so the label signal is there.

But the generated values can explode after decoding. Since these windows are in standardized SDForger space, abs max values in the thousands are clearly invalid. This is a sanity-check failure, not just a small metric degradation.
-->

---

# Latent validity constraint improves stability

<div class="twocol">
<div>
<table class="metric-table">
<thead>
<tr><th>Metric</th><th>Raw unified</th><th>`clip_p05_p95`</th></tr>
</thead>
<tbody>
<tr><td>Walking abs max</td><td class="num">3761.8</td><td class="num">3.24</td></tr>
<tr><td>Running abs max</td><td class="num">1572.1</td><td class="num">2.65</td></tr>
<tr><td>Label controllability</td><td class="num">0.8212</td><td class="num">0.9868</td></tr>
<tr><td>Synthetic-only HAR</td><td class="num">0.4955</td><td class="num">0.6937</td></tr>
</tbody>
</table>

<div class="card" style="margin-top:0.9rem;">
<h3>Next method step</h3>
<div class="small">
Generate candidate latent → check validity → accept/reject → decode.
</div>
</div>
</div>

<div>
<img class="figure-wide" src="./assets/2026-05-26-advisor-sdforger-pamap2-update/running-constrained-acf-psd.png" />
<div class="figure-label">After percentile latent constraint, running recovers reasonable value scale and rhythm diagnostics.</div>
</div>
</div>

<div class="takeaway">The current direction is not just more data; it is controlled conditional generation with latent validity checks.</div>

<!--
Speaker notes:
To diagnose the failure, I constrained generated latent values to the training latent distribution before decoding. The best diagnostic variant clips each latent dimension to the 5th-95th percentile range observed in training.

This is still post-hoc, so I would not present it as a final method. But it strongly supports the diagnosis: the main failure mode is invalid generated latent values. After constraint, value explosion disappears, controllability improves, and synthetic-only HAR utility recovers.

Evaluation explanation:
1. abs max/std are sanity checks for value explosion.
2. ACF/PSD check motion rhythm and frequency structure.
3. label controllability checks whether the requested activity affects generation.
4. synthetic-only HAR trains on synthetic windows and tests on real held-out windows, following the downstream utility idea from the advisor feedback.

The next step is generation-time latent validity control or rejection sampling, where invalid candidates are rejected before decoding. After that is stable, we can extend to multi-subject training to increase data and reduce subject-specific bias, and then to multichannel data such as hand plus ankle acceleration.
-->
