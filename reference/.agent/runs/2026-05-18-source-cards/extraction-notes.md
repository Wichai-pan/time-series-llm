# Extraction Notes

Run ID: 2026-05-18-source-cards

## Method

Used `pdftotext -layout` to extract each PDF into local run text files:

- `sdforger.txt`
- `chatts.txt`
- `agentsense.txt`

Then inspected targeted sections relevant to method, baselines, datasets, evaluation, and project relevance.

## Notes

- SDForger is the closest method source for the current project.
- ChatTS is adjacent; it is about time-series MLLM understanding/reasoning, not HAR training-data generation.
- AgentSense is application-level closest work for HAR synthetic sensor data, but it uses ambient smart-home event sensors rather than wearable continuous signals.

## Output Cards

- `reference/cards/sdforger-2025-forging-time-series.md`
- `reference/cards/chatts-2025-aligning-time-series.md`
- `reference/cards/agentsense-2025-virtual-sensor-data.md`
