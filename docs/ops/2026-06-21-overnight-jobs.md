# 2026-06-21 过夜作业 manifest（明早分析用）

目标：明天下午 3pm 小组会前,产出"方法有效(带误差棒)+ 应用有用(带 CI)+ 模型是瓶颈(自己做了对照)"的图与故事。

## 今晚提交的 job

### 1. gpt2 5-seed 严谨批（保证有结果）
- job `35209871`,脚本 `slurm/run_overnight_gpt2_rigor_gpu.sh`
- 每个 seed(1-5):multi-subject unified 生成 → per-activity 修复 → TSG(train+heldout) → held-out HAR
- 输出:`output/time_series/overnight_rigor_20260621/{gen,repair}_seed*`;`reports/overnight_rigor_20260621/{tsg_seed*, har_seed*}`
- **明早聚合**:
  - 组合最优 TSG:对 5 个 seed 的 `tsg_seed*/combined_best_tsgbench_summary.csv`,取 clip_p05_p95、train ref,算 walking/running 的 ACD/MDD **mean±95%CI**
  - held-out HAR:5 个 `har_seed*/har_utility_heldout_results.csv`,算 real-only / synthetic-only / real+synthetic 的 accuracy/F1 **mean±CI**(确认 real+synthetic > real-only 显著)

### 2. 强模型对照（条件触发,头牌）
- vLLM + Qwen 在本集群崩(LLVM bug,旧 run 也撞过)→ 改用 **transformers 生成**(`scripts/run_synthetic_amplitude_control_hf.py`)
- Qwen HF 烟雾 `35209886` → 若成功,监控自动提交:
  - **Qwen2.5-1.5B ×1/×3**(3 seed):`output/time_series/qwen_amp_hf_seed{42,7,123}_20260621/amplitude_adherence.csv`
  - **gpt2 HF 对照**(3 seed,同生成后端,apples-to-apples):`gpt2_amp_hf_seed{...}_20260621/`
- **明早聚合**:每 seed 算 a1/a0 的 latent_l2 中位比 + Mann-Whitney;**Qwen 比值是否 > gpt2**(≈3=完全遵守,≈1=不遵守)。
  - 若 Qwen 明显 > gpt2 → **"数值条件化随模型变强而出现"**(和队友 gemma 互证)
  - 若 Qwen 也 ≈1 → 限制更深

## 明早 morning session(2-3h)
1. 收集上面所有 CSV,聚合(带 CI)。
2. 做 4-5 张 deck 图:① 组合最优 TSG(误差棒) ② held-out HAR(CI) ③ Qwen vs gpt2 ×1/×3 ④ 诊断流程(温度/模型/prompt→模型)。
3. 快速温度消融(现有 checkpoint,temp {0.5,0.7,1.0,1.3} 生成,画温度 vs 爆炸率)。
4. 串成会上故事。

## 第二批（审计后补充,2026-06-21 深夜）

强模型/编码对照(全 HF 生成,因 vLLM+Qwen 崩):
- `35209978` Qwen ×1/×3 @100ep seed42；`35209995` seeds 7/123（去 epoch confound）
- `35209996` phi-2(2.7B) ×1/×3 seed42（规模天花板）
- `35210000` **Qwen stat-prompt 真实数值编码**(队友式,干净无泄漏,平衡请求,秩 adherence)seed42 → `output/time_series/qwen_statprompt_values_seed42_20260621/`
- `35210001` Qwen stat-prompt 粗档编码(我们 T2 式)seed42 → `qwen_statprompt_bins_seed42_...`
- 脚本:`run_synthetic_amplitude_control_hf.py`、`run_stat_prompt_hf.py`、`run_amp_hf_param_gpu.sh`、`run_statprompt_hf_param_gpu.sh`

明早聚合:
- ×1/×3:gpt2(0.94)/Qwen-30ep(0.99)/Qwen-100ep/phi-2 的 a1/a0 比 → "数值条件化是否随规模出现"
- stat-prompt values vs bins(Qwen):读 `run_metadata.json` 的 `adherence.spearman_req_vs_realmax` → **真实数值编码在强模型上是否产生 per-sample 遵守**(结案队友:若 values 有 ρ>0.3 而 bins 没有 → 我们之前的负结果是编码 artifact;若都没有 → 队友的效果是类别先验+泄漏)

## 审计发现的高优先缺口（明早/会后补,多为 CPU）
1. **naive augmentation baseline**(jitter/time-warp)—— HAR/TSG 都缺;synthetic 必须打赢它才算有价值
2. **real-vs-real 指标地板**—— ACD/DTW "near baseline" 无意义除非有真实-真实下限
3. **seeds/CI 不对称**—— 正面 claim 缺误差棒(已补 gpt2 5-seed)
4. **multivariate/多通道**—— 全程单通道,SDForger 的招牌能力 + HAR 真正难点没测(还有 embedding_dim auto bug)
5. **teammate 泄漏 reconciliation**(prompt_split=test)—— 正在 settling 中

## 状态检查命令
`ssh puhti "squeue -u panh; ls output/time_series/overnight_rigor_20260621/ output/time_series/qwen_amp_hf_seed*_20260621/ 2>/dev/null"`
