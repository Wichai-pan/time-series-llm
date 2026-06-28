> 配套：已标注的官方代码在 `reference/sdforger-official/`（标注行以 `# paper-note-added:` 开头）。
> 论文：Forging Time Series with Language (arXiv:2505.17103v2)。本文件 = 代码↔论文对照导读。

# SDForger 代码到论文对照速览（15-20 分钟讲解用）

> 适用范围：仅基于已标注的官方代码（`utils.py`、`generate.py`、`trainer.py`、`data_objects.py`、`task.py`、`time_series.yaml`）。所有标注行均以 `# paper-note-added:` 为前缀。

---

## 1. 代码—论文主对照表（按 6 步流程分组）

### Step 1 周期感知分割（Periodicity-aware Segmentation）— 论文 A.1
| 文件 | 函数/类 | 论文章节 | 一句话作用 |
|------|---------|----------|------------|
| utils.py | `estimate_period_acf` | A.1 | ACF 自相关估周期：nlags=L/2 强制 P<L/2，排除 lag0 后按自相关强度排候选周期 |
| utils.py | `compute_period` | A.1 | 定主周期 P（ACF 为主，FFT 兜底），由 min_windows/min_points 推 max_period，窗长 L=N·P |
| utils.py | `construct_windows` | A.1 | 把长序列 L0 切成 I 个等长窗（实例），步长 s 取 P 的整数倍；minimize-overlap 实现论文 s=floor((L0-L)/(I-1)) |
| utils.py | `preprocess_uni_multi_variate_data` | A.1 | 多通道分别估周期，取最频繁周期统一，每通道切约 I=15 个对齐窗 |
| utils.py | `standardscale_train` | Sec3.1 | 逐通道 StandardScaler，使嵌入在零均值单位方差窗上进行；保留 scaler 供解码逆标准化 |

### Step 2 嵌入（FPC / FastICA Embedding）— 论文 Sec3.1 + A.2
| 文件 | 函数/类 | 论文章节 | 一句话作用 |
|------|---------|----------|------------|
| utils.py | `fpc_embed_data` | Sec3.1+A.2 | FPC：协方差 X^T X 特征分解得有序基 b_j，系数 e_ij=<X_i,b_j>=X@basis；auto-k 取达到目标累计方差的最小维；拼成 I×K 嵌入表 E |
| utils.py | `fica_embed_data` | Sec3.1+A.2 | FastICA：fit_transform 得统计独立（无序）系数；auto-k 增长至重构方差达标（上限 30，呼应 K>25 不稳）；保留 mixing_/mean_ 供解码 |

### Step 3 文本编码（Textual Encoding）— 论文 Sec3.2.1
| 文件 | 函数/类 | 论文章节 | 一句话作用 |
|------|---------|----------|------------|
| utils.py | `embeddings_to_text` | Sec3.2.1 | Fill-in-the-middle 模板：`random.shuffle(columns)` 实现每实例置换 π 去序偏；输出 `Input: col is [blank]...[sep] Target: value [answer]...` |
| trainer.py | `train: embeddings_to_text 调用` | Sec3.2.1 | 训练侧调用 utils 把每实例系数 e_ij 转 FIM 文本（模板与置换 π 实现在 utils） |
| trainer.py | `train: dataset.sample(frac=1)` | Sec3.2.1 | 打乱实例行序去除实例间排列偏置（作用于行，思路同 π） |
| generate.py | `init_tokens` (内嵌函数) | Sec3.2.1 | 推理侧文本编码：构造 FIM 提示，每实例随机置换 π，blank 待 LLM 填系数 |
| generate.py | `random.shuffle(col_list)` | Sec3.2.1 | 推理侧排列 π：每批样本独立打乱特征列序 |
| generate.py | `input_prompts = 'Input:...[sep] Target:'` | Sec3.2.1 | FIM 模板组装，模型自回归填 [answer] 系数 |
| generate.py | `fim_template_textual_encoding 分支` | Sec3.2.1 | 条件 FIM 变体：加 Condition: 段注入分类特征作上下文，数值特征仍 blank 且仍置换 |

### Step 4 LLM 微调与推理（GPT-2 Finetune + Inference）— 论文 Sec3.2.2
| 文件 | 函数/类 | 论文章节 | 一句话作用 |
|------|---------|----------|------------|
| trainer.py | `class SDForgerTuningBlock` | Sec3.2.2 | LLM 微调步骤入口：在系数文本上微调 GPT-2 并早停取最优 |
| trainer.py | `get_model` | Sec3.2.2 | 加载论文使用的 GPT-2（AutoModelForCausalLM）+ tokenizer |
| trainer.py | `train` 主循环 | Sec3.2.2 | E(I×K)→文本→分词→80/20 划分→Adam 微调→早停加载最优 |
| trainer.py | `train_args` (lr/epochs/batch) | Sec3.2.2 | 超参对应 lr=8e-5、batch=32、最多 200 epoch |
| trainer.py | `eval_strategy='steps', eval_steps=5` | Sec3.2.2 | 每 5 步评估验证损失，对应论文早停监控频率 |
| trainer.py | `EvalLossEarlyStopping(patience=5)` | Sec3.2.2 | 早停回调 patience=5，与论文一致 |
| generate.py | `self.trainer(...)` | Sec3.2.1+3.2.2 | FIM 编码 + GPT-2 微调（Adam lr8e-5, batch32, 早停）触发块 |
| generate.py | `llm([{input, gen_kwargs}...])` | Sec3.2.2 | 推理采样调用：blank 提示喂微调 LLM，多项式/温度采样自回归填系数 |
| time_series.yaml | trainer 块 / `model_id_or_path: gpt2` | Sec3.2.2 | 微调块；默认骨干 GPT-2（granite 为可选） |
| time_series.yaml | `learning_rate / batch_size / epochs` | Sec3.2.2 | lr=8e-5、batch=32、epoch 上限 100（配置值）+ 早停 |
| time_series.yaml | target 块(vllm) / `temperature: 1.3` | Sec3.2.2 | 推理块载微调检查点；温度 1.3 实现多样采样 |

### Step 5 文本解码 + 生成过滤（Textual Decoding + Filtering）— 论文 Sec3.2.1 解码 + A.3/A.4
| 文件 | 函数/类 | 论文章节 | 一句话作用 |
|------|---------|----------|------------|
| generate.py | `self.generate_data` | Sec3.2.2+A.3+A.4 | 核心生成循环：推理采样→文本解码→生成内过滤→停止判据 |
| utils.py | `convert_texts_to_tabular_data` | Sec3.2.1解码+A.3 | 按 [sep]/[answer] 解析 LLM 文本，按置换序回对齐 Target 值，转回数值/分类系数；残缺实例丢弃 |
| generate.py | `generated_df[~(...'NaN'...)] / dropna` | A.3(1) | 过滤(1)：丢弃含缺失值/生成不完整实例 |
| generate.py | `pd.concat(dfs).round(3).drop_duplicates()` | A.3(2) | 过滤(2)：丢弃重复实例 |
| generate.py | `check_distribution` 块 | A.3(3) | 过滤(3)：L2 范数落在 [q1-3·IQR, q3+3·IQR] 外的发散实例丢弃，逐通道检查，全通过才接受 |
| generate.py | `old_l2_norms_splits = np.linalg.norm(...)` | A.3 | 原始嵌入向量逐通道 L2 范数，作为发散检测参考分布 |
| generate.py | `split_indices / original_data_splits` | A.3 | 按通道(各 k_c)切分嵌入向量，使范数过滤逐通道进行 |
| generate.py | `diversity_score = unique_norms/len(...)` | A.4 | 多样性分 D=唯一舍入范数数/有效实例数 |
| generate.py | `max_element < norms_diversity_threshold` | A.4 | 停止判据：各通道 D 的最大值 < λ_stop（范数饱和），或样本数超 I_max |
| generate.py | `min/max_outputs_to_generate, inference_batch` | A.4 | 停止判据旋钮：I_max 上界、最小样本数、每轮采样批大小 |
| generate.py | `return pd.concat(dfs).to_numpy()` | A.3→Step6 | 返回过滤后有效系数表，喂入逆嵌入 |

### Step 6 逆嵌入解码（Inverse Embedding）— 论文 Sec3.3
| 文件 | 函数/类 | 论文章节 | 一句话作用 |
|------|---------|----------|------------|
| utils.py | `fpc_transform_to_original_feature_space` | Sec3.3 | 线性重构 x_i=Σ_j e_ij·b_j（coeff@basis^T）再逆标准化；fpc-filled 变体用真实高频细节覆盖低频分量 |
| utils.py | `fica_transform_to_original_feature_space` | Sec3.3 | ICA 逆变换 x_i=e_i@mixing^T+mean，先按通道拆分拼接系数 |
| data_objects.py | `TimeSeriesOutputData.generated_time_series` | Sec3.3/Step6 | 封装最终重构的合成多通道序列 |

### 编排与框架管线（顶层 / 无直接论文映射）
| 文件 | 函数/类 | 论文章节 | 一句话作用 |
|------|---------|----------|------------|
| generate.py | `TimeSeriesDataBuilder` / `__call__` | Sec3 全流程 | 顶层编排：分割→嵌入→编码/微调→推理→解码→逆嵌入 |
| generate.py | `original_dataset = embedded_data.copy()` | A.3/A.4 | E 基线副本，作 IQR 过滤与多样性打分参考 |
| data_objects.py | `TimeSeriesInputData.observations` | 输入 X | 封装待分割/嵌入的原始种子序列 |
| task.py | `TimeSeriesTask` / `transform_batch_size=train_length(5000)` | A.1 输入长度 L0 | 接入 fms-dgt 框架；train_length 即原始序列长度 L0 |

---

## 2. 推荐阅读顺序（端到端跟流程）

1. **`data_objects.py`** — 先看输入 `TimeSeriesInputData` / 输出 `TimeSeriesOutputData`，明确管线两端的数据形态。
2. **`generate.py: TimeSeriesDataBuilder.__call__`** — 抓住 6 步骨架（分割→嵌入→微调→生成→逆嵌入），它是导航地图。
3. **`utils.py` 按 Step 顺序**：
   - `compute_period` / `estimate_period_acf` / `construct_windows` / `standardscale_train`（Step 1）
   - `fpc_embed_data` 与 `fica_embed_data`（Step 2，重点看 e_ij 内积/特征分解）
   - `embeddings_to_text`（Step 3，FIM 模板 + 置换 π）
4. **`trainer.py: SDForgerTuningBlock.train` + `EvalLossEarlyStopping`**（Step 4 微调侧）。
5. **回到 `generate.py: generate_data`**（Step 4 推理 + Step 5）：`init_tokens` 文本编码 → `llm(...)` 采样 → `convert_texts_to_tabular_data` 解码 → A.3 三重过滤 → A.4 停止判据。
6. **`utils.py: fpc_/fica_transform_to_original_feature_space`**（Step 6 逆嵌入收尾）。
7. **`time_series.yaml`** 最后看，把超参（gpt2、lr8e-5、batch32、temperature1.3）对回论文 Sec3.2.2。

> 一句话主线：长序列 → 切窗 → 标准化 → FPC/ICA 系数 → 置换后 FIM 文本 → GPT-2 微调/采样 → 解码+过滤系数 → 线性逆变换回序列。

---

## 3. 演讲/Q&A 值得强调的 3-5 个实现细节（代码可见、能落地论文）

1. **嵌入即一行内积/特征分解。** FPC 是协方差 `X^T X` 特征分解得有序基 b_j，系数 `e_ij=<X_i,b_j>` 直接 `X@basis`；FastICA 用 `fit_transform` 得无序独立系数。两者最终都拼成同一张 **I×K 嵌入表 E**（K=Σ k_c），且 ICA auto-k 上限被硬编为 30，正对应论文 "K>25 不稳定"。讲解时可指出：基有序(FPC)/无序(ICA)的差别决定了后面 fpc-filled 高频回填只对 FPC 有意义。

2. **置换 π 就是 `random.shuffle`。** 论文的去序偏置换 π 在代码里是字面的 `random.shuffle(columns/col_list)`，**每个实例/每批样本独立打乱**，配合 FIM 模板 `Input: ... is [blank] ... [sep] Target: ... [answer] ...`。解码端 `convert_texts_to_tabular_data` 必须按置换后的特征序回对齐 Target —— 这是 FIM 能工作的关键闭环，适合做一张图。

3. **A.3 三重过滤的确切阈值。** (1) `dropna` 丢缺失；(2) `round(3).drop_duplicates()` 丢重复；(3) 发散过滤用 **`[q1 - 3*IQR, q3 + 3*IQR]`** 的 L2 范数区间，**逐通道**判定，必须所有通道都通过才接受。这是论文 A.3 最具体、最可被问到的数字。

4. **A.4 停止判据是多样性分 D。** `D = 唯一舍入范数数 / 有效实例数`，**取各通道最大值**与 λ_stop（`norms_diversity_threshold`）比较；低于阈值即认为范数饱和/重复而停止，或样本数达 I_max。

5. **采样在推理块、温度=1.3。** 论文的多项式/温度采样不在 trainer 里，而是 `generate.py` 的 `llm([{input, gen_kwargs}...])` 调用 + YAML 中 `temperature: 1.3`（>1 增加随机性）。微调与采样分属两个文件，问到"采样在哪"时不要指向 trainer.py。

---

## 4. 代码与论文的偏差 / 可忽略的框架噪声

**需要主动说明的偏差：**
- **多样性分舍入位数不一致**：代码 `rounding_factor=3`（保留 3 位小数），**论文 A.4 写的是 4 位**。已在 generate.py 行内标注。
- **epoch 上限不一致**：`time_series.yaml` 写 `num_train_epochs: 100`（trainer 默认更保守），**论文允许最多 200 epoch**，均靠早停实际终止。
- **`if attempt > 9: break` 是工程兜底**：10 轮无新有效样本即停，**论文无此判据**，属防卡死安全阀，不要当成 A.4 内容讲。

**可忽略的框架噪声（无论文映射）：**
- trainer.py 的 `compute_device`(mps/cuda/cpu 选择)、`get_quantization_config`(4/8bit 量化)、`set_seed`、`preprocess_function`/分词 labels —— 纯训练管线。
- utils.py 的 `preprocess_train_data` / `preprocess_multisample_data` / `get_data` / `interpolate_na` / `get_feature_distribution` —— 调度/采窗/取数/插值/列画像等 plumbing。其中 `random.sample` 仅是 multisample 取窗，**不是论文采样**。
- task.py / data_objects.py 整体是 fms-dgt 接入层；`seed: 42`、`dtype: float32`、`metadata.version` 论文未指定。

**模板分支提醒（易讲错）：** `generate_data` 内有三个模板：`base_template`(非 FIM 旧路径)、`fim_template`(论文方法)、`fim_template_textual_encoding`(分类条件扩展)。**论文方法对应后两者，base_template 是 legacy，不要拿来讲 Sec3.2.1。**

**实现位置提醒：** trainer.py 只覆盖 Step 4 的"微调侧"；Step 3 的模板与置换 π 实际委托给 `utils.embeddings_to_text`；推理采样、A.3 过滤、A.4 判据都在 generate.py；ACF 分割与 ICA/FPC 嵌入解码在 utils.py。task.py/data_objects.py/yaml 不含核心算法。
```

文件路径供查阅（均为绝对路径）：
- `/Users/huataipan/Wichai/CODAS/Aalto/ELEC-E7633 - Project Course/reference/sdforger-official/utils.py`
- `/Users/huataipan/Wichai/CODAS/Aalto/ELEC-E7633 - Project Course/reference/sdforger-official/generate.py`
- `/Users/huataipan/Wichai/CODAS/Aalto/ELEC-E7633 - Project Course/reference/sdforger-official/trainer.py`
- `/Users/huataipan/Wichai/CODAS/Aalto/ELEC-E7633 - Project Course/reference/sdforger-official/data_objects.py`
- `/Users/huataipan/Wichai/CODAS/Aalto/ELEC-E7633 - Project Course/reference/sdforger-official/task.py`
- `/Users/huataipan/Wichai/CODAS/Aalto/ELEC-E7633 - Project Course/reference/sdforger-official/time_series.yaml