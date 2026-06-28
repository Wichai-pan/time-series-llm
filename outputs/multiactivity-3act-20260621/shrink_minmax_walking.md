# SDForger Benchmark Metrics (hand_acc16_x)

- Synthetic file: `output/time_series/pamap2_subject101_102_105_walking_running_cycling_repair_20260621/shrink_minmax/walking_final_data.jsonl`
- Real training windows: `54`
- Synthetic windows generated: `52`
- Synthetic windows evaluated: `52`
- Distance-metric subset mode: `first`
- Distance-metric paired sample count: `52`
- Synthetic value space: `scaled`
- SHR computed: `False`

| Metric | Value |
|---|---:|
| `MDD` | `0.201976` |
| `ACD` | `1.938689` |
| `SD` | `1.365339` |
| `KD` | `4.799065` |
| `ED` | `18.411094` |
| `DTW` | `188.146178` |
| `SHR` | `N/A` |

## Notes

- `MDD/ACD/SD/KD/ED/DTW` follow the TSGBench code path.
- `SHR` / `SHAP-RE` uses the SIDL-based shapelet reconstruction path from the supplementary implementation when `--compute-shr` is enabled.
- `ED/DTW` require equal sample counts, so synthetic windows are deterministically reduced to the real training-window count before pairing.
- `--max-synthetic-samples` applies a hard cap before all metrics when a run generates more samples than the paper protocol intends to evaluate.
- fms-dgt generated JSONL values are usually already standardized; use `--synthetic-space=original` only for raw-scale synthetic JSONL.
