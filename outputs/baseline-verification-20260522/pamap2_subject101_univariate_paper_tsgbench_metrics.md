# SDForger Benchmark Metrics (hand_acc16_x)

- Synthetic file: `output/time_series/pamap2_subject101_univariate_paper/final_data.jsonl`
- Real training windows: `31`
- Synthetic windows generated: `85`
- Distance-metric subset mode: `first`
- Distance-metric paired sample count: `31`

| Metric | Value |
|---|---:|
| `MDD` | `0.395802` |
| `ACD` | `1.197348` |
| `SD` | `0.833090` |
| `KD` | `0.951272` |
| `ED` | `19.039300` |
| `DTW` | `304.437101` |
| `SHR` | `N/A` |

## Notes

- `MDD/ACD/SD/KD/ED/DTW` follow the TSGBench code path.
- `SHR` is left as `N/A` because the paper uses SIDL-based shapelet reconstruction and that implementation is not available in this repo.
- `ED/DTW` require equal sample counts, so synthetic windows are deterministically reduced to the real training-window count before pairing.
