# SDForger Benchmark Metrics (hand_acc16_x)

- Synthetic file: `output/time_series/pamap2_subject101_unified_stat_prompt_hand_acc16_x/running_final_data.jsonl`
- Real training windows: `31`
- Synthetic windows generated: `74`
- Synthetic windows evaluated: `74`
- Distance-metric subset mode: `first`
- Distance-metric paired sample count: `31`
- Synthetic value space: `scaled`
- SHR computed: `False`

| Metric | Value |
|---|---:|
| `MDD` | `0.295776` |
| `ACD` | `1.798189` |
| `SD` | `5.462542` |
| `KD` | `50.704079` |
| `ED` | `2112.422106` |
| `DTW` | `32972.986805` |
| `SHR` | `N/A` |

## Notes

- `MDD/ACD/SD/KD/ED/DTW` follow the TSGBench code path.
- `SHR` / `SHAP-RE` uses the SIDL-based shapelet reconstruction path from the supplementary implementation when `--compute-shr` is enabled.
- `ED/DTW` require equal sample counts, so synthetic windows are deterministically reduced to the real training-window count before pairing.
- `--max-synthetic-samples` applies a hard cap before all metrics when a run generates more samples than the paper protocol intends to evaluate.
- fms-dgt generated JSONL values are usually already standardized; use `--synthetic-space=original` only for raw-scale synthetic JSONL.
