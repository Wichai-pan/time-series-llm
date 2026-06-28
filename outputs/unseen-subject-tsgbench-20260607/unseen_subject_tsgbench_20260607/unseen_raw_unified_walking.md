# SDForger Benchmark Metrics (hand_acc16_x)

- Synthetic file: `/scratch/project_2016517/panh/time-series-llm/fms-dgt/output/time_series/pamap2_subject101_102_105_unified_label_conditioned_hand_acc16_x/walking_final_data.jsonl`
- Real training windows: `45`
- Synthetic windows generated: `51`
- Synthetic windows evaluated: `51`
- Distance-metric subset mode: `first`
- Distance-metric paired sample count: `45`
- Synthetic value space: `scaled`
- SHR computed: `False`

| Metric | Value |
|---|---:|
| `MDD` | `0.187568` |
| `ACD` | `4.211875` |
| `SD` | `0.072371` |
| `KD` | `39.804451` |
| `ED` | `787.557755` |
| `DTW` | `11160.440820` |
| `SHR` | `N/A` |

## Notes

- `MDD/ACD/SD/KD/ED/DTW` follow the TSGBench code path.
- `SHR` / `SHAP-RE` uses the SIDL-based shapelet reconstruction path from the supplementary implementation when `--compute-shr` is enabled.
- `ED/DTW` require equal sample counts, so synthetic windows are deterministically reduced to the real training-window count before pairing.
- `--max-synthetic-samples` applies a hard cap before all metrics when a run generates more samples than the paper protocol intends to evaluate.
- fms-dgt generated JSONL values are usually already standardized; use `--synthetic-space=original` only for raw-scale synthetic JSONL.
