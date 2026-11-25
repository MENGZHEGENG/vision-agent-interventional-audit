# Recorded five-seed selection extension

This directory provides a lightweight replay of the five-seed selection extension. It contains task identifiers, 36 fixed selection sets, held-out evaluation labels, predictions without labels, per-task summaries, and a validator. It contains no model weights or source images, and it does not run models.

## Contents

- `configs/tasks.json`: 110 identity records (2 base and 108 update tasks).
- `configs/selection_sets.json`: 36 frozen selection sets across the two task families.
- `data/labels/`: 288 held-out labels per family, without scene payloads.
- `data/tasks/`: 110 files with 288 label-free predictions each.
- `data/results.jsonl`: compact evaluation rows bound to each task.
- `data/analysis_summary.json`: current descriptive analysis summary.
- `scripts/replay_selection_extension.py`: deterministic recorded-output validation.

Run:

```bash
python3 scripts/replay_selection_extension.py --root .
```

The validator checks task counts, sample-ID joins, prediction counts, and every recorded accuracy. The package validates the recorded outputs; it does not reproduce model inference. Labels were joined after the label-free selection matrix and selected IDs were frozen. The recorded protocol did not independently audit a physical boundary that would prevent labels from entering model selection, so the paper keeps the corresponding claim descriptive and scoped.
