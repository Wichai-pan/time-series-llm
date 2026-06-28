# SDForger Benchmark Metrics (hand_acc16_x)

- Synthetic file: `output/time_series/pamap2_subject101_102_105_walking_running_cycling_repair_20260621/shrink_minmax/cycling_final_data.jsonl`
- Real training windows: `51`
- Synthetic windows generated: `50`
- Synthetic windows evaluated: `50`
- Distance-metric subset mode: `first`
- Distance-metric paired sample count: `50`
- Synthetic value space: `scaled`
- SHR computed: `False`

| Metric | Value |
|---|---:|
| `MDD` | `0.194847` |
| `ACD` | `1.821918` |
| `SD` | `0.350321` |
| `KD` | `1.523769` |
| `ED` | `19.340754` |
| `DTW` | `207.067108` |
| `SHR` | `N/A` |

## Notes

- `MDD/ACD/SD/KD/ED/DTW` follow the TSGBench code path.
- `SHR` / `SHAP-RE` uses the SIDL-based shapelet reconstruction path from the supplementary implementation when `--compute-shr` is enabled.
- `ED/DTW` require equal sample counts, so synthetic windows are deterministically reduced to the real training-window count before pairing.
- `--max-synthetic-samples` applies a hard cap before all metrics when a run generates more samples than the paper protocol intends to evaluate.
- fms-dgt generated JSONL values are usually already standardized; use `--synthetic-space=original` only for raw-scale synthetic JSONL.
