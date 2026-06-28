# SDForger Benchmark Metrics (hand_acc16_x)

- Synthetic file: `/scratch/project_2016517/panh/time-series-llm/fms-dgt/output/time_series/pamap2_subject101_unified_label_conditioned_hand_acc16_x_smooth_repair_20260615/clip_p05_p95/running_final_data.jsonl`
- Real training windows: `31`
- Synthetic windows generated: `79`
- Synthetic windows evaluated: `79`
- Distance-metric subset mode: `first`
- Distance-metric paired sample count: `31`
- Synthetic value space: `scaled`
- SHR computed: `False`

| Metric | Value |
|---|---:|
| `MDD` | `0.315895` |
| `ACD` | `1.959807` |
| `SD` | `0.336292` |
| `KD` | `0.422136` |
| `ED` | `22.644881` |
| `DTW` | `139.694653` |
| `SHR` | `N/A` |

## Notes

- `MDD/ACD/SD/KD/ED/DTW` follow the TSGBench code path.
- `SHR` / `SHAP-RE` uses the SIDL-based shapelet reconstruction path from the supplementary implementation when `--compute-shr` is enabled.
- `ED/DTW` require equal sample counts, so synthetic windows are deterministically reduced to the real training-window count before pairing.
- `--max-synthetic-samples` applies a hard cap before all metrics when a run generates more samples than the paper protocol intends to evaluate.
- fms-dgt generated JSONL values are usually already standardized; use `--synthetic-space=original` only for raw-scale synthetic JSONL.
