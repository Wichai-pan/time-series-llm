# PAMAP2 Subject101 Periodicity Check

- Input: `/scratch/project_2016517/panh/datasets/pamap2/PAMAP2_Dataset/Protocol/subject101.dat`
- Channels: `hand_acc16_x, ankle_acc16_x`
- Activities: `4, 5, 6`
- Segment length: `5000` samples
- Sampling rate assumption: `100.0` Hz

| activity | channel | rows | ACF peak lag | ACF score | dominant freq Hz | period samples | label | plot |
|---|---|---:|---:|---:|---:|---:|---|---|
| walking | hand_acc16_x | 22253 | 58 | 0.773 | 1.758 | 56.9 | strong | `subject101_walking_hand_acc16_x_periodicity.png` |
| walking | ankle_acc16_x | 22253 | 116 | 0.797 | 1.758 | 56.9 | strong | `subject101_walking_ankle_acc16_x_periodicity.png` |
| running | hand_acc16_x | 21265 | 82 | 0.929 | 2.441 | 41.0 | strong | `subject101_running_hand_acc16_x_periodicity.png` |
| running | ankle_acc16_x | 21265 | 82 | 0.631 | 4.883 | 20.5 | strong | `subject101_running_ankle_acc16_x_periodicity.png` |
| cycling | hand_acc16_x | 23575 | 7 | 0.402 | 11.816 | 8.5 | strong | `subject101_cycling_hand_acc16_x_periodicity.png` |
| cycling | ankle_acc16_x | 23575 | 3 | 0.815 | 0.781 | 128.0 | strong | `subject101_cycling_ankle_acc16_x_periodicity.png` |
