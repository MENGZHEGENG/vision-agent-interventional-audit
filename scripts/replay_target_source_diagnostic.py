#!/usr/bin/env python3
"""Recompute the original-seed target-source diagnostic from saved predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


FAMILIES = ("shape_count", "color_count")
ARMS = ("random", "same_model")
FIT_SEEDS = {
    "shape_count": (20261005, 20261007, 20261009),
    "color_count": (20261006, 20261008, 20261010),
}
TARGET_SOURCES = ("proposer_answer", "program_answer")
SELECTION_SEED = 20261004
SAMPLES_PER_RUN = 288


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    predictions = read_jsonl(root / "data/target_source_diagnostic/heldout_predictions.jsonl")
    labels_by_family = {}
    support = {}
    for family in FAMILIES:
        labels = read_jsonl(root / f"selection_seed_extension_v1/data/labels/{family}.jsonl")
        if len(labels) != SAMPLES_PER_RUN or len({row["sample_id"] for row in labels}) != SAMPLES_PER_RUN:
            raise ValueError(f"{family}: expected 288 unique held-out labels")
        labels_by_family[family] = {row["sample_id"]: str(row["answer"]) for row in labels}
        support[family] = dict(sorted(Counter(str(row["answer"]) for row in labels).items()))

    runs: dict[tuple, dict[str, dict[str, int]]] = {}
    for row in predictions:
        key = (
            row["task_family"], row["arm"], int(row["fit_seed"]),
            int(row["selection_seed"]),
        )
        if row["task_family"] not in FAMILIES or row["arm"] not in ARMS:
            raise ValueError("unexpected diagnostic task")
        if key[3] != SELECTION_SEED or row["target_source"] not in TARGET_SOURCES:
            raise ValueError("unexpected selection seed or target source")
        prediction = row["prediction"]
        if not isinstance(prediction, int) or prediction < 0:
            raise ValueError("held-out predictions must be nonnegative integers")
        source_rows = runs.setdefault(key, {}).setdefault(row["target_source"], {})
        if row["sample_id"] in source_rows:
            raise ValueError("duplicate prediction for a task and sample")
        source_rows[row["sample_id"]] = prediction

    expected_keys = {
        (family, arm, fit_seed, SELECTION_SEED)
        for family in FAMILIES for arm in ARMS for fit_seed in FIT_SEEDS[family]
    }
    if set(runs) != expected_keys:
        raise ValueError("diagnostic task roster is incomplete")

    per_task = []
    outside_range = 0
    for family in FAMILIES:
        labels = labels_by_family[family]
        for arm in ARMS:
            for fit_seed in FIT_SEEDS[family]:
                key = (family, arm, fit_seed, SELECTION_SEED)
                pair = runs[key]
                if set(pair) != set(TARGET_SOURCES):
                    raise ValueError("both target sources are required")
                proposer, program = pair["proposer_answer"], pair["program_answer"]
                if set(proposer) != set(labels) or set(program) != set(labels):
                    raise ValueError("held-out prediction IDs differ from the label set")
                outside_range += sum(value > 4 for values in pair.values() for value in values.values())
                proposer_correct = sum(str(value) == labels[sample_id] for sample_id, value in proposer.items())
                program_correct = sum(str(value) == labels[sample_id] for sample_id, value in program.items())
                proposer_accuracy = proposer_correct / SAMPLES_PER_RUN
                program_accuracy = program_correct / SAMPLES_PER_RUN
                per_task.append({
                    "task_family": family,
                    "arm": arm,
                    "fit_seed": fit_seed,
                    "proposer_accuracy": proposer_accuracy,
                    "program_accuracy": program_accuracy,
                    "program_minus_proposer_percent": 100 * (program_accuracy - proposer_accuracy),
                })

    aggregate = {}
    for family in FAMILIES:
        for arm in ARMS:
            rows = [row for row in per_task if row["task_family"] == family and row["arm"] == arm]
            effects = [row["program_minus_proposer_percent"] for row in rows]
            aggregate[f"{family}/{arm}"] = {
                "fit_seeds": [row["fit_seed"] for row in rows],
                "proposer_accuracy_mean": mean(row["proposer_accuracy"] for row in rows),
                "program_accuracy_mean": mean(row["program_accuracy"] for row in rows),
                "program_minus_proposer_mean_percent": mean(effects),
                "fit_seed_effects_percent": effects,
                "fit_seed_effect_range_percent": [min(effects), max(effects)],
            }

    result = {
        "schema": "rsi_acl_target_source_diagnostic_replay_v1",
        "status": "SUCCEEDED",
        "selection_seed": SELECTION_SEED,
        "task_count": len(per_task),
        "predictions_per_target_source": len(predictions) // 2,
        "out_of_range_predictions": outside_range,
        "answer_support_unique_examples": support,
        "per_task": per_task,
        "aggregate": aggregate,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
