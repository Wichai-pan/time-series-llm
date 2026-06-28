# SDForger Benchmark Metrics (hand_acc16_x)

- Synthetic file: `output/time_series/pamap2_subject101_102_105_unified_label_conditioned_hand_acc16_x_repair_peractivity_20260621/clip_p05_p95/running_final_data.jsonl`
- Real training windows: `44`
- Synthetic windows generated: `58`
- Synthetic windows evaluated: `58`
- Distance-metric subset mode: `first`
- Distance-metric paired sample count: `44`
- Synthetic value space: `scaled`
- SHR computed: `False`

| Metric | Value |
|---|---:|
| `MDD` | `0.242515` |
| `ACD` | `1.743177` |
| `SD` | `1.033801` |
| `KD` | `0.845185` |
| `ED` | `20.117143` |
| `DTW` | `151.330394` |
| `SHR` | `N/A` |

## Notes

- `MDD/ACD/SD/KD/ED/DTW` follow the TSGBench code path.
- `SHR` / `SHAP-RE` uses the SIDL-based shapelet reconstruction path from the supplementary implementation when `--compute-shr` is enabled.
- `ED/DTW` require equal sample counts, so synthetic windows are deterministically reduced to the real training-window count before pairing.
- `--max-synthetic-samples` applies a hard cap before all metrics when a run generates more samples than the paper protocol intends to evaluate.
- fms-dgt generated JSONL values are usually already standardized; use `--synthetic-space=original` only for raw-scale synthetic JSONL.
