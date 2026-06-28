# 2026-05-18 Teammate Review

## Scope

Reviewed repositories:

- `IkMangMok/mmfit-inference`
- `SJYANG555/sdforger-pamap2`

Baseline branch requested: `main`.

Safety constraints followed: no merge, no push, no SSH, no SLURM/RunAI, no experiment execution, no overwrite of local work.

## Git And Access Status

The project control root is not itself a Git repository. The old local code archive at `legacy/old-project-files/puhti-time-series-llm/` is a dirty Git repo with many modified and untracked files, so it was treated as read-only and not used as a merge target.

`IkMangMok/mmfit-inference` was not reviewable from the current environment. GitHub API requests for repository metadata, branches, and PRs returned `404`. This usually means private repository, renamed/mistyped path, or missing access.

`SJYANG555/sdforger-pamap2` is public and was reviewed from the downloaded `main` snapshot:

- Default branch: `main`
- Reviewed commit: `90884c109e48bb7a83b4ba722455a59ffa9983fe`
- Branches visible: only `main`
- PRs visible: none
- Local review copy: `reviews/teammate-repos/SJYANG555-sdforger-pamap2-90884c1/`

Because only `main` was visible and no PR/teammate branch existed, there was no meaningful branch-vs-main diff to review. This report reviews the uploaded `main` snapshot and its recent commit history instead.

## What Was Added

Feature / code:

- PAMAP2 preprocessing and SDForger-style dataset construction.
- Text template, embedding, reconstruction, generation parsing, filtering, and visualization modules under `pamap2_forger/`.
- Training, generation, evaluation, KNN consistency, and plotting scripts under `scripts/`.
- Experimental support for GPT-2, Gemma 2 2B, Llama 3.2 3B, full-body and sensor-subset variants.

Experiments / results:

- Archived result summaries for GPT-2, Gemma, Llama, five-class v2, compact checkpoint, hand/chest variants, and KNN condition consistency.
- README reports formal metrics for retention, duplicate ratio, DTW, MDD, synthetic-only accuracy, and real+synthetic accuracy.
- Result artifacts include CSVs, logs, config snapshots, and plots, but not raw PAMAP2, generated window tensors, train/val/test tensors, or model checkpoints.

Docs:

- README with project purpose, metric tables, and reproduction entry points.
- Pipeline, training, evaluation, smoke-test, and Triton documentation.
- `CONTEXT.md` records experiment history, failures, parser fixes, storage failures, and recovered outputs.

Config / infra:

- YAML configs for model and dataset variants.
- Slurm scripts for preprocessing, training, generation, evaluation, KNN checks, and comparison plots.
- `environment.yml` for environment reconstruction.

## Findings

### High: Utility Evaluation Appears Test-Conditioned

All reviewed generation configs use `generation.prompt_split: test`, for example `config/pamap2_sdforger_gpt2_5class_v2.yaml:49-63`. The generated sample metadata is seeded from the prompt metadata, including `window_id`, `subject_id`, `activity_id`, and `activity_name` in `pamap2_forger/generate.py:150-163`. The utility evaluator then trains on synthetic labels from `synthetic_metadata["activity_id"]` and tests on real test labels in `scripts/evaluate_utility.py:42-52`.

This does not necessarily mean the classifier sees real test windows as features, but it does mean the synthetic augmentation set is generated from test-split prompts and carries test-split labels/conditions. Therefore the reported `real_plus_synthetic` HAR utility is not a clean train-only augmentation protocol. It is closer to a test-conditioned generation diagnostic.

Impact: claims like "synthetic data improves HAR utility" are not currently well supported. A reviewer or advisor can fairly object that the protocol uses test distribution/metadata during synthetic-data creation.

Required change: rerun utility with synthetic samples generated only from train-side prompts or a clearly separated unlabeled/held-out prompt protocol. Keep test-conditioned generation as a diagnostic, not as the main utility evidence.

### High: Reported MDD / SD Metrics Do Not Match Their Usual Meaning

The README says formal similarity values come from `similarity_metrics.csv` and reports `MDD` as a main metric (`README.md:74-101`). In the implementation, `MDD` is computed as only the absolute difference between one global mean of real samples and one global mean of synthetic samples: `pamap2_forger/metrics.py:98`. That is much weaker than a marginal distribution difference over values/channels/features.

The implementation also labels `SD` as a spectral magnitude difference over prototype windows (`pamap2_forger/metrics.py:112-124`), then exports it as `SD` alongside `MDD`, `ACD`, `KD`, `ED`, and `DTW` (`pamap2_forger/metrics.py:140-150`). If `SD` is intended to mean the SDForger/TSGBench-style skewness difference, this is a metric-definition mismatch.

Impact: the current metric table may be internally reproducible but is not safe to describe as official SDForger/TSGBench similarity evidence.

Required change: either rename these as local proxy metrics, or implement the intended definitions and rerun the similarity tables.

### Medium: Results Are Not Reproducible From The GitHub Archive Alone

The README explicitly excludes model checkpoints, optimizer states, generated windows, train/val/test windows, raw PAMAP2, and environments (`README.md:20-31`). Reproduction commands assume a Triton path and environment under `/scratch/work/yangs9/ChatTS` (`README.md:116-127`).

Impact: the uploaded repo is useful as a code/result archive, but a third party cannot currently reproduce the reported metrics from the public checkout alone.

Required change: add an artifact manifest with exact filenames, hashes, sizes, and retrieval locations; or provide a small reproducible smoke artifact bundle that can verify the end-to-end path.

### Medium: Some Reported Results Depend On Recovery After Failed Runs

`CONTEXT.md` records several important failures and repairs:

- GPT-2 v2 generation produced `0/857` valid embeddings due context truncation.
- Compact GPT-2 training failed while saving optimizer because `/scratch` was full; checkpoint `1300` was reused.
- Compact GPT-2 generation initially failed norm filtering; parser was fixed and existing raw generations were reparsed.
- Gemma v2 failed when saving `generated_windows.npy`; windows were reconstructed later from embeddings/reducer.

The README notes that compact GPT-2 initially generated zero embeddings and the reparsed archive recovered 835 usable embeddings (`README.md:84-87`, `README.md:101`).

Impact: these are plausible engineering recoveries, but the affected results should remain provisional until a clean rerun confirms them.

Required change: rerun the formal comparison end-to-end from pinned configs after storage and parser fixes, then regenerate the metric tables.

### Medium: No Automated Test Suite Was Found

The archive contains a smoke-test script and smoke-test docs, but no unit/integration test suite was found under `tests/`, and no `pytest.ini` or similar test runner configuration was present.

Impact: parser behavior, metric definitions, split integrity, filter behavior, and reconstruction round-trip can regress silently.

Required change: add small tests for parsing generated outputs, split selection, metric sanity checks, synthetic empty-set handling, and reconstruction shape/value invariants.

### Low: `mmfit-inference` Is Blocked By Access

The requested `IkMangMok/mmfit-inference` repository could not be inspected. No branch, PR, code, docs, or results can be reviewed until access or the URL is corrected.

## Result Status

Supported:

- `SJYANG555/sdforger-pamap2` public repo metadata, branch visibility, and reviewed commit SHA.
- The archive contains code, configs, docs, logs, CSV summaries, and plots for the reported SDForger/PAMAP2 workflow.
- The dataset split intent is visible in configs, e.g. subject split `101-106/107/108` in `config/pamap2_sdforger_gpt2_5class_v2.yaml:1-8`.

Provisional:

- README metric tables for GPT-2, Gemma, Llama, sensor subsets, and KNN consistency.
- Recovered compact GPT-2 and Gemma v2 outputs.
- Any comparison between full-body and hand/chest variants.

Needs verification:

- Any claim that synthetic data improves HAR utility, because generation appears test-conditioned.
- Any claim using `MDD` or `SD` as standard SDForger/TSGBench metrics, because the implementation uses local proxy definitions.
- Any result requiring omitted tensors/checkpoints/raw data.
- `IkMangMok/mmfit-inference`, because the repository was inaccessible.

## Recommendation

For `SJYANG555/sdforger-pamap2`: request changes and rerun smoke before treating the results as merge-ready evidence. The direction is still worth continuing, but the current result claims should not be used as final project evidence until the utility protocol and metric definitions are fixed or clearly renamed.

For `IkMangMok/mmfit-inference`: ask clarification or request access. There is no basis yet for merge, rejection, or technical review.

Immediate asks for the teammate:

1. Confirm whether `prompt_split: test` is intentional for all utility results.
2. Provide a train-conditioned synthetic generation run or revise the utility claim.
3. Clarify whether `MDD` and `SD` are intended as official SDForger metrics or local proxies.
4. Provide artifact hashes or a small reproducible smoke bundle.
5. Add tests for parser, metric definitions, split integrity, and empty synthetic cases.

## Local Follow-Up: `teamate/mmfit-inference`

After the repository was placed locally under `teamate/mmfit-inference`, it became reviewable.

Git status:

- Local path: `teamate/mmfit-inference`
- Current branch: `main`
- Remote: `https://github.com/IkMangMok/mmfit-inference.git`
- Reviewed commit: `009302f Fix workflow dependencies and classifier test metrics`
- Local status: clean
- Visible branch relation: `main`, `origin/main`, and `origin/HEAD` point to the same commit

There is still no separate teammate branch or PR diff against `main` in the local checkout. This review therefore covers the local `main` snapshot.

### What Was Added In `mmfit-inference`

Feature / code:

- MM-Fit latent-generation pipeline around a learned conditional autoencoder.
- Transformer/MLP VAE training, latent extraction, frozen-decoder decoding, and generated sample evaluation in `mmfit_inference/class_autoencoder.py`.
- Latent SFT dataset construction, LoRA SFT, and latent JSON generation in `mmfit_inference/latent_sft.py`.
- Raw MM-Fit time-series classifier for evaluating decoded synthetic samples in `mmfit_inference/raw_classifier.py`.
- Helper modules for raw ChatTS-style prompts, completion utilities, transforms, and shared IO.

Experiments / logs:

- Slurm logs for autoencoder training, latent SFT, and generated-data attempts are checked in under `slurm_logs/`.
- Autoencoder training appears to have completed and saved artifacts, with training loss decreasing to a reported best train loss.
- Latent SFT logs show completed training/eval-loss reporting.
- Generated-data logs show attempts to generate latent JSON, but archived generation runs include failures from invalid latent vector lengths and missing decoded JSONL files.

Docs:

- README explains the full intended workflow: MM-Fit data -> encoder/decoder -> latent SFT data -> LoRA SFT -> generated latents -> decoder -> held-out evaluation.
- README documents train subjects `w00-w05`, held-out subjects `w06-w07`, output paths, evaluation metrics, and a local smoke-test recipe.

Config / scripts:

- `run_full_workflow.sh` submits a Slurm dependency DAG.
- Separate scripts cover autoencoder training, latent SFT data building, LoRA SFT, raw classifier training, LLM generation/evaluation, and evaluation-only reruns.
- `pyproject.toml` and `requirements.txt` describe package dependencies.

### Additional Findings For `mmfit-inference`

#### High: Archived Generation/Evaluation Logs Do Not Show A Clean Completed Result

The checked-in logs show generation failures caused by malformed latent vectors, for example `Sample 0 z must have length 32, got 30`, `z must have shape (32,), got (33,)`, and downstream `FileNotFoundError` for `synthetic_mmfit.jsonl`. These are visible in `slurm_logs/mmfit_generate_llm_data_*.err`.

Impact: the repository contains a plausible pipeline, but no checked-in `eval_report.json`, generated JSONL, decoded samples, or artifact manifest proving that the full generate -> decode -> evaluate path completed successfully.

Status: generated-data results are `needs-verification`, not supported.

Required change: rerun a clean smoke path and a clean full path after the latest parsing/retry changes, then archive the final `eval_report.json`, `raw_generations.jsonl`, generated latents, decoded JSONL metadata, and exact commit/config.

#### Medium: Shell Workflow Can Continue After Failed Steps

`run_generate_llm_data.sh` does not enable `set -euo pipefail`. It runs latent generation, decoding, and evaluation sequentially (`run_generate_llm_data.sh:44-75`). If generation fails or does not produce the expected file, later commands can still run and produce secondary errors or misleading final echo messages.

Impact: Slurm logs can look like the workflow reached later stages even when the first required artifact was missing.

Required change: add strict shell failure handling, explicit file-existence checks after each stage, and clear exit messages.

#### Medium: Autoencoder Selection Uses Training Loss Only

The autoencoder training loop keeps `best_state` based on training loss only (`mmfit_inference/class_autoencoder.py:756-807`). The README correctly says final evaluation uses held-out subjects, but the decoder itself is still selected without a validation split.

Impact: the latent space and decoder may be overfit to `w00-w05`, and generated-sample quality depends heavily on that decoder.

Required change: add an autoencoder validation split or held-out train-side subject for checkpoint selection; report reconstruction quality separately from generation quality.

#### Medium: Default Evaluation Does Not Include Raw-Classifier Real-Test Baseline

The evaluator can include `raw_classifier_real_test_accuracy` only when `--classifier-evaluate-real-test` is set (`mmfit_inference/class_autoencoder.py:2082-2086`). The default generation script passes a classifier checkpoint but does not set that flag (`run_generate_llm_data.sh:65-75`).

Impact: `raw_classifier_generated_accuracy` is hard to interpret without the same classifier's held-out real-test accuracy in the same report.

Required change: include real-test classifier baseline in report artifacts, or always link the corresponding raw classifier report with `test_metrics`.

#### Medium: No Automated Test Suite Was Found

The repo has no visible `tests/` directory or test runner config. Given the amount of parsing, tensor shaping, latent dimension validation, and metric code, this is a reproducibility risk.

Required change: add tests for latent JSON parsing, latent dimension validation, decode shape invariants, split isolation, empty/malformed JSONL handling, and workflow script failure behavior.

#### Low: `--parse-retries` Is Exposed But Not Used

`latent_sft.py` exposes `--parse-retries` in argument parsing, but current inference control uses `--max-attempts-per-exercise` instead. The unused option can mislead future runs.

Required change: remove `--parse-retries` or wire it into the retry logic.

### Result Status For `mmfit-inference`

Supported:

- The local repo exists, is clean, and is on `main` at commit `009302f`.
- The code implements a coherent MM-Fit latent autoencoder + latent SFT + decoder + evaluation pipeline.
- Autoencoder and latent SFT logs indicate those stages ran to completion at least once.

Provisional:

- Autoencoder reconstruction quality, because only train-loss evidence is archived.
- Latent SFT quality, because logs show train/eval loss but not downstream valid generation quality.
- Raw classifier quality, unless the corresponding `classifier_report.json` is provided.

Needs verification:

- Any claim that LLM-generated latent samples successfully decode into high-quality MM-Fit synthetic time series.
- Any claim based on generated-data evaluation metrics, because final reports/artifacts are not present in the local repo and archived generation logs include failures.

### Updated Recommendation For `mmfit-inference`

Recommendation: request changes + rerun smoke. Do not merge/use as project evidence yet.

This direction is interesting as a method-extension branch, but it is not currently a clean evidence source for the main SDForger/PAMAP2 project. Treat it as an exploratory teammate branch until a small end-to-end run produces archived, inspectable artifacts.
