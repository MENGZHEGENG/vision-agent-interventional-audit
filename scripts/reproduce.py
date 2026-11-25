#!/usr/bin/env python3
"""Portable entry points for benchmark construction and the model study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("base", "random", "same_model", "independent", "consensus", "qwen_8b", "independent_8b")
FAMILIES = ("shape_count", "color_count")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def executed_sum(values, start=0):
    """Preserve the executed Python 3.11 summation order on newer interpreters."""
    for value in values:
        start += value
    return start


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        for value in values: handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def revision():
    value = (ROOT / "SOURCE_REVISION").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", value): raise ValueError("invalid bound source revision")
    return value


def config(path):
    result = load(path)
    if result["schema"] != "visual_verifier_experiment_v1" or result["arms"] != list(ARMS) or result["selection_budget"] != 48 or result["selection_seed"] != 20261004 or result["bootstrap_draws"] != 20000:
        raise ValueError("configuration differs from the registered study")
    from development_core import MODEL_PANEL
    if result["checkpoints"] != MODEL_PANEL: raise ValueError("checkpoint identities differ")
    if result["recipe"] != {"batch_size": 1, "epochs": 1, "optimizer_steps": 48, "learning_rate": 0.0001, "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05, "gradient_clip_norm": 1.0, "optimizer": "AdamW defaults", "development_max_new_tokens": 16, "heldout_max_new_tokens": 64}:
        raise ValueError("training or decoding recipe differs")
    return result


def chosen_families(args):
    return [args.family] if args.family else FAMILIES


def benchmark(args, cfg):
    for family in chosen_families(args):
        specification = cfg["families"][family]
        for stage, seed, splits in (("development", specification["development_seed"], ["evolution", "selection"]), ("heldout", specification["heldout_seed"], ["heldout"])):
            directory = args.workspace / "datasets" / family / stage
            common = ["--source-commit", revision(), "--seed", str(seed), "--task-family", family, "--items-per-split", "96", "--splits", ",".join(splits)]
            subprocess.run([sys.executable, str(ROOT / "scripts/visual_verifier_dataset.py"), "generate", "--output-root", str(directory), *common], check=True, capture_output=True)
            subprocess.run([sys.executable, str(ROOT / "scripts/visual_verifier_dataset.py"), "validate", "--dataset-root", str(directory), *common, "--output", str(directory / "validation_receipt.json")], check=True, capture_output=True)
    print("Benchmark generated and validated: two families, fixed development and held-out seeds.")


def identity_map(family, cfg):
    specification = cfg["families"][family]
    path = ROOT / specification["identity_map"]
    if not path.resolve().is_relative_to(ROOT) or digest(path) != specification["identity_map_sha256"]:
        raise ValueError("fixed identifier mapping differs")
    result = rows(path)
    if len({row["model_sample_id"] for row in result}) != len(result) or len({row["source_sample_id"] for row in result}) != len(result):
        raise ValueError("identifier mapping is not one-to-one")
    return result


def prepare(args, cfg):
    from condition_core import _build_derangement
    for family in chosen_families(args):
        directory = args.workspace / "datasets" / family / "development"
        policies = {}
        labels = {}
        for split in ("evolution", "selection"):
            policies.update({row["sample_id"]: {"split": split, **row} for row in rows(directory / "policy_inputs" / (split + ".jsonl"))})
            labels.update({row["sample_id"]: row for row in rows(directory / "evaluator_labels" / (split + ".jsonl"))})
        mapping = identity_map(family, cfg)
        if {row["source_sample_id"] for row in mapping} != set(policies) or set(policies) != set(labels):
            raise ValueError("fixed identifiers do not cover the generated development data")
        target_field = "target_shape" if family == "shape_count" else "target_color"
        records = {}
        by_source = {}
        output_rows = []
        groups = []
        for entry in mapping:
            source = entry["source_sample_id"]
            policy = policies[source]; label = labels[source]
            image = (directory / policy["image_path"]).resolve(strict=True)
            if not image.is_relative_to(directory.resolve()) or digest(image) != entry["image_sha256"] or policy["question"] != entry["question"] or label["source_id"] != entry["source_scene_id"] or policy["split"] != entry["split"]:
                raise ValueError("fixed identifiers or image bytes disagree with the benchmark")
            records[source] = {"split": policy["split"], "target": label[target_field], "label": label, "policy": policy, "image_sha256": digest(image)}
            relative = image.relative_to(args.workspace.resolve()).as_posix()
            by_source[source] = {"sample_id": entry["model_sample_id"], "image_path": relative, "question": policy["question"]}
            output_rows.append(by_source[source])
            groups.append({"sample_id": entry["model_sample_id"], "split": entry["split"], "source_scene_id": entry["source_scene_id"]})
        shuffled = _build_derangement({"records": records, "target_field": target_field}, cfg["families"][family]["shuffle_seed"])
        shuffled_rows = []
        for entry in mapping:
            donor = shuffled[entry["source_sample_id"]]["donor_sample_id"]
            shuffled_rows.append({"sample_id": entry["model_sample_id"], "image_path": by_source[donor]["image_path"], "question": entry["question"]})
        root = args.workspace / "inputs" / family
        write_rows(root / "original.jsonl", output_rows)
        write_rows(root / "shuffled.jsonl", shuffled_rows)
        write_rows(root / "no_image.jsonl", [{**row, "image_path": None} for row in output_rows])
        write_rows(root / "groups.jsonl", groups)
        write(root / "input_index.json", {"identity_map_sha256": cfg["families"][family]["identity_map_sha256"], "files": {name: digest(root / name) for name in ("original.jsonl", "shuffled.jsonl", "no_image.jsonl", "groups.jsonl")}})
    print("Prepared label-free model inputs using the fixed public identifier mappings.")


def inputs(args, family, condition):
    root = args.workspace / "inputs" / family
    path = root / (condition + ".jsonl")
    if digest(path) != load(root / "input_index.json")["files"][path.name]:
        raise ValueError("model input bytes changed")
    values = rows(path)
    if any(set(row) != {"sample_id", "image_path", "question"} for row in values):
        raise ValueError("model input exposes non-policy fields")
    return values


def models(args, cfg):
    from huggingface_hub import snapshot_download
    for key, item in cfg["checkpoints"].items():
        directory = args.model_cache / key
        snapshot_download(repo_id=item["repo_id"], revision=item["revision"], local_dir=directory)
        files = []
        for path in sorted(directory.rglob("*")):
            if ".cache" in path.relative_to(directory).parts or not path.is_file(): continue
            if path.is_symlink(): raise ValueError("model snapshot contains symbolic links")
            files.append({"path": path.relative_to(directory).as_posix(), "sha256": digest(path), "bytes": path.stat().st_size})
        write(args.model_cache / (key + "_index.json"), {"repo_id": item["repo_id"], "revision": item["revision"], "files": files})
    print("Pinned model snapshots downloaded and indexed.")


def model_directory(args, cfg, key):
    directory = (args.model_cache / key).resolve(strict=True)
    index = load(args.model_cache / (key + "_index.json"))
    if any(index[field] != cfg["checkpoints"][key][field] for field in ("repo_id", "revision")):
        raise ValueError("model snapshot identity differs")
    if not index["files"]: raise ValueError("empty model snapshot index")
    for row in index["files"]:
        path = (directory / row["path"]).resolve(strict=True)
        if not path.is_relative_to(directory) or path.is_symlink() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("model snapshot file differs")
    return directory


def infer(torch, key, model, processor, transform, image, text):
    from development_core import MODEL_PANEL, qwen_call, intern_call, validate_model_input_metadata
    family = MODEL_PANEL[key]["family"]
    response, metadata = qwen_call(torch, model, processor, image, text) if family == "qwen3_vl" else intern_call(torch, model, processor, transform, image, text)
    validate_model_input_metadata(family, image is not None, metadata)
    return response


def propose(args, cfg):
    import torch
    from PIL import Image
    from development_core import load_model, build_text, parse_answer
    key = "qwen3_vl_2b_instruct"
    model, processor, transform = load_model(key, model_directory(args, cfg, key))
    for family in chosen_families(args):
        values = []
        for row in inputs(args, family, "original"):
            image_path = args.workspace / row["image_path"]
            with Image.open(image_path) as opened:
                response = infer(torch, key, model, processor, transform, opened.convert("RGB"), build_text("candidate", row["question"], None))
            answer, status = parse_answer(response)
            values.append({"sample_id": row["sample_id"], "question": row["question"], "candidate_answer": answer, "candidate_parse_status": status, "candidate_raw_response": response, "candidate_image_sha256": digest(image_path)})
        path = args.workspace / "candidates" / (family + ".jsonl")
        write_rows(path, values)
        write(path.with_suffix(".json"), {"sha256": digest(path), "count": len(values)})


def candidates(args, family):
    path = args.workspace / "candidates" / (family + ".jsonl")
    index = load(path.with_suffix(".json"))
    values = rows(path)
    if digest(path) != index["sha256"] or len(values) != index["count"] or len({row["sample_id"] for row in values}) != len(values):
        raise ValueError("frozen candidate bytes or coverage changed")
    return {row["sample_id"]: row for row in values}


def check(args, cfg):
    import torch
    from PIL import Image
    from development_core import load_model, build_text, parse_verdict
    for key in ([args.model] if args.model else cfg["checkpoints"]):
        model, processor, transform = load_model(key, model_directory(args, cfg, key))
        for family in chosen_families(args):
            frozen = candidates(args, family)
            for condition in ([args.condition] if args.condition else ("original", "no_image", "shuffled")):
                values = []
                for row in inputs(args, family, condition):
                    candidate = frozen[row["sample_id"]]
                    image = None
                    if row["image_path"] is not None:
                        with Image.open(args.workspace / row["image_path"]) as opened: image = opened.convert("RGB")
                    if candidate["candidate_parse_status"] == "PARSEABLE":
                        response = infer(torch, key, model, processor, transform, image, build_text("checker", row["question"], candidate["candidate_answer"]))
                        verdict, status = parse_verdict(response)
                    else:
                        response, verdict, status = "", None, "CANDIDATE_INVALID_REJECTION"
                    values.append({**candidate, "model_key": key, "condition": condition, "raw_verdict_response": response, "parsed_verdict": verdict, "parse_status": status})
                path = args.workspace / "checkers" / family / (key + "_" + condition + ".jsonl")
                write_rows(path, values)
                write(path.with_suffix(".json"), {"sha256": digest(path), "candidate_sha256": digest(args.workspace / "candidates" / (family + ".jsonl"))})
        del model, processor, transform
        torch.cuda.empty_cache()


def select(args, cfg):
    from selection_core import _candidate_maps, _select
    for family in chosen_families(args):
        input_rows = {row["sample_id"]: row for row in inputs(args, family, "original")}
        groups = {row["sample_id"]: row for row in rows(args.workspace / "inputs" / family / "groups.jsonl")}
        input_root = args.workspace / "inputs" / family
        if digest(input_root / "groups.jsonl") != load(input_root / "input_index.json")["files"]["groups.jsonl"]:
            raise ValueError("grouping input bytes changed")
        joined = []
        for key in cfg["checkpoints"]:
            path = args.workspace / "checkers" / family / (key + "_original.jsonl")
            index = load(path.with_suffix(".json"))
            if index["sha256"] != digest(path) or index["candidate_sha256"] != digest(args.workspace / "candidates" / (family + ".jsonl")):
                raise ValueError("checker or candidate bytes changed")
            frozen = candidates(args, family)
            checked = rows(path)
            if len(checked) != len(frozen) or {row["sample_id"] for row in checked} != set(frozen):
                raise ValueError("checker coverage differs from frozen candidates")
            for row in checked:
                original = frozen[row["sample_id"]]
                if any(row[field] != original[field] for field in original):
                    raise ValueError("checker metadata differs from the frozen candidate")
            joined.extend({**row, **groups[row["sample_id"]]} for row in checked)
        for arm in ARMS[1:]:
            checker = {"qwen_8b": "qwen3_vl_8b_instruct", "independent_8b": "internvl3_5_8b"}.get(arm, "internvl3_5_2b_instruct")
            self_rows, peer_rows = _candidate_maps(joined, "qwen3_vl_2b_instruct", checker)
            eligible = []
            for sample_id, row in self_rows.items():
                if row["candidate_parse_status"] != "PARSEABLE": continue
                self_yes = row["parsed_verdict"] == "YES"; peer_yes = peer_rows[sample_id]["parsed_verdict"] == "YES"
                if arm == "random" or (arm == "same_model" and self_yes) or (arm == "consensus" and self_yes and peer_yes) or (arm in {"independent", "qwen_8b", "independent_8b"} and peer_yes): eligible.append(row)
            chosen = _select(eligible, arm, 48, cfg["selection_seed"])
            examples = [{"sample_id": row["sample_id"], "source_scene_id": row["source_scene_id"], "question": row["question"], "claimed_answer": row["candidate_answer"], "image_path": input_rows[row["sample_id"]]["image_path"], "image_sha256": row["candidate_image_sha256"]} for row in chosen]
            directory = args.workspace / "updates" / family / arm
            path = directory / "update_examples.jsonl"
            write_rows(path, examples)
            write(directory / "update_manifest.json", {"schema": "rsi_visual_verifier_update_set_v2", "status": "ACCEPTED", "admission_rule": arm, "builder_source_commit": revision(), "task_family": family, "image_root": str(args.workspace), "same_model_key": "qwen3_vl_2b_instruct", "independent_model_key": checker, "fixed_budget": 48, "selection_seed": cfg["selection_seed"], "selection_order": "shared_sha256_seed_sample_id_across_arms", "selected_count": 48, "selected_scene_count": 48, "evaluator_answers_in_update_set": False, "selected_sample_ids_sha256": hashlib.sha256("\n".join(row["sample_id"] for row in examples).encode()).hexdigest(), "selected_source_scene_ids_sha256": hashlib.sha256("\n".join(row["source_scene_id"] for row in examples).encode()).hexdigest(), "files": [{"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}]})


def train(args, cfg):
    import torch
    from training_core import load_update_examples, training_order, load_qwen, train_adapter, evaluate
    key = "qwen3_vl_2b_instruct"
    model_dir = model_directory(args, cfg, key)
    for family in chosen_families(args):
        if args.seed is not None and args.seed not in cfg["families"][family]["training_seeds"]:
            raise ValueError("training seed is outside the registered family")
        for seed in ([args.seed] if args.seed is not None else cfg["families"][family]["training_seeds"]):
            for arm in ([args.arm] if args.arm else ARMS):
                dataset = args.workspace / "datasets" / family / "heldout"
                output = args.workspace / "results" / family / ("seed" + str(seed)) / arm
                output.mkdir(parents=True, exist_ok=False)
                random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                model, processor = load_qwen(model_dir)
                training = None; update = None
                if arm != "base":
                    directory = args.workspace / "updates" / family / arm
                    examples = load_update_examples(directory, args.workspace)
                    training_order(examples, seed, 48, 1, True)
                    options = argparse.Namespace(seed=seed, max_steps=48, epochs=1, require_complete_epoch=True, learning_rate=1e-4, lora_rank=8, lora_alpha=16)
                    training, model = train_adapter(model, processor, examples, args.workspace, output / "adapter", options)
                    model = model.merge_and_unload().eval(); model.config.use_cache = True
                    training["merged_for_evaluation"] = True
                    update = load(directory / "update_manifest.json")
                predictions, evaluation = evaluate(model, processor, dataset, 96, output / "predictions_unscored.jsonl")
                write(output / "result.json", {"status": "SUCCEEDED", "arm": arm, "seed": seed, "source_commit": revision(), "dataset_manifest_sha256": digest(dataset / "dataset_manifest.json"), "asset_receipt_sha256": digest(args.model_cache / (key + "_index.json")), "model": cfg["checkpoints"][key], "update_manifest": update, "training": training, "predictions": predictions, "evaluation": evaluation})
                del model, processor
                torch.cuda.empty_cache()


def aggregate(args, cfg):
    for family in chosen_families(args):
        output = args.workspace / "analysis" / (family + ".json")
        subprocess.run([sys.executable, str(ROOT / "scripts/aggregate_updates.py"), "--results-root", str(args.workspace / "results" / family), "--dataset-root", str(args.workspace / "datasets" / family / "heldout"), "--update-root", str(args.workspace / "updates" / family), "--output", str(output), "--expected-seeds", *map(str, cfg["families"][family]["training_seeds"]), "--require-complete-epoch", "--expected-pairs", "96", "--draws", "20000", "--seed", "20261024"], check=True)


def replay(args, cfg):
    """Recompute measurements from recorded responses; perform no model calls."""
    from development_core import parse_answer, parse_verdict
    from evaluation_core import parse_integer
    from metric_core import gate_metrics, paired_flip_metrics
    import aggregate_updates
    # Python 3.12 changed float sum; the accepted analysis used Python 3.11.
    # Preserve that execution's arithmetic without changing the bootstrap AST.
    aggregate_updates.sum = executed_sum
    from aggregate_updates import bootstrap, validate_consumed_examples
    for family in chosen_families(args):
        specification = cfg["families"][family]
        path = ROOT / specification["recorded_development_rows"]
        if digest(path) != specification["recorded_development_sha256"]:
            raise ValueError("recorded development bytes changed")
        recorded = rows(path)
        original = {row["sample_id"]: row for row in inputs(args, family, "original")}
        frozen = {}
        for row in recorded:
            if row["model_key"] == "qwen3_vl_2b_instruct" and row["condition"] == "original":
                sample = row["sample_id"]
                if sample in frozen: raise ValueError("duplicate recorded candidate")
                if parse_answer(row["candidate_raw_response"]) != (row["candidate_answer"], row["candidate_parse_status"]):
                    raise ValueError("candidate text and parsing disagree")
                if row["question"] != original[sample]["question"] or row["candidate_image_sha256"] != digest(args.workspace / original[sample]["image_path"]):
                    raise ValueError("recorded candidate and generated input disagree")
                frozen[sample] = {key: row[key] for key in ("sample_id", "question", "candidate_answer", "candidate_parse_status", "candidate_raw_response", "candidate_image_sha256")}
        if set(frozen) != set(original): raise ValueError("recorded candidate coverage differs")
        candidate_path = args.workspace / "candidates" / (family + ".jsonl")
        write_rows(candidate_path, list(frozen.values()))
        write(candidate_path.with_suffix(".json"), {"sha256": digest(candidate_path), "count": len(frozen), "execution_mode": "RECORDED_RESPONSE_REPLAY"})
        gates = {}; sensitivity = {}
        for key in cfg["checkpoints"]:
            by_condition = {}
            for condition in ("original", "no_image", "shuffled"):
                selected = [row for row in recorded if row["model_key"] == key and row["condition"] == condition]
                conditioned = {row["sample_id"]: row for row in inputs(args, family, condition)}
                if len(selected) != len(frozen) or {row["sample_id"] for row in selected} != set(frozen):
                    raise ValueError("recorded checker coverage differs")
                checked = []
                for row in selected:
                    sample = row["sample_id"]
                    if any(row[field] != frozen[sample][field] for field in frozen[sample]): raise ValueError("recorded checker uses a different candidate")
                    if parse_verdict(row["raw_verdict_response"]) != (row["parsed_verdict"], row["parse_status"]):
                        raise ValueError("verdict text and parsing disagree")
                    image_path = conditioned[sample]["image_path"]
                    image_sha = digest(args.workspace / image_path) if image_path else None
                    if image_sha != row["checker_image_sha256"]: raise ValueError("recorded checker image differs from the generated condition")
                    checked.append({**frozen[sample], "model_key": key, "condition": condition, "raw_verdict_response": row["raw_verdict_response"], "parsed_verdict": row["parsed_verdict"], "parse_status": row["parse_status"]})
                checker_path = args.workspace / "checkers" / family / (key + "_" + condition + ".jsonl")
                write_rows(checker_path, checked)
                write(checker_path.with_suffix(".json"), {"sha256": digest(checker_path), "candidate_sha256": digest(candidate_path), "execution_mode": "RECORDED_RESPONSE_REPLAY"})
                by_condition[condition] = [row for row in selected if row["split"] == "evolution"]
                if condition == "original": gates[key] = gate_metrics(selected)
            sensitivity[key] = {condition: paired_flip_metrics(by_condition["original"], by_condition[condition]) for condition in ("no_image", "shuffled")}
        # Selection itself consumes only the label-free replayed responses.
        scoped = argparse.Namespace(**vars(args)); scoped.family = family
        select(scoped, cfg)
        labels = {}
        development = args.workspace / "datasets" / family / "development"
        for split in ("evolution", "selection"):
            labels.update({row["sample_id"]: row for row in rows(development / "evaluator_labels" / (split + ".jsonl"))})
        mapped = {row["model_sample_id"]: labels[row["source_sample_id"]] for row in identity_map(family, cfg)}
        for row in recorded:
            label = mapped[row["sample_id"]]
            correct = row["candidate_answer"] is not None and int(row["candidate_answer"]) == int(label["answer"])
            if int(row["gold_answer"]) != int(label["answer"]) or type(row["candidate_correct"]) is not bool or row["candidate_correct"] != correct or row["edit_kind"] != label["edit_kind"]:
                raise ValueError("recorded development target disagrees with generated oracle")
        path = ROOT / specification["recorded_updates"]
        if digest(path) != specification["recorded_updates_sha256"]:
            raise ValueError("recorded held-out response bytes changed")
        updates = load(path)
        heldout_labels = {row["sample_id"]: row for row in rows(args.workspace / "datasets" / family / "heldout/evaluator_labels/heldout.jsonl")}
        pairs = {}
        for sample, label in heldout_labels.items(): pairs.setdefault(label["pair_id"], []).append(sample)
        if len(pairs) != 96 or any(len(samples) != 3 for samples in pairs.values()): raise ValueError("held-out pair support differs")
        seeds = specification["training_seeds"]
        if updates.get("schema") != "visual_verifier_recorded_updates_v1" or updates.get("source_revision") != revision() or set(updates.get("seeds", {})) != set(map(str, seeds)):
            raise ValueError("recorded held-out revision or seed matrix differs")
        correct = {}; selected_precision = {}; selected_composition = {}; subgroups = {}; selected_ids = {}; selected_scenes = {}
        for arm in ARMS:
            if any(set(updates["seeds"][str(seed)]) != set(ARMS) for seed in seeds): raise ValueError("recorded arm matrix differs")
            correct[arm] = {}
            subgroups[arm] = {}
            for seed in seeds:
                record = updates["seeds"][str(seed)][arm]
                if record["arm"] != arm or record["seed"] != seed: raise ValueError("recorded arm or seed identity differs")
                raw = record["predictions"]
                if len(raw) != 288 or len({row["sample_id"] for row in raw}) != 288 or {row["sample_id"] for row in raw} != set(heldout_labels): raise ValueError("recorded held-out coverage differs")
                correct[arm][seed] = {}
                for row in raw:
                    if parse_integer(row["prediction_text"]) != row["prediction"] or (row["prediction"] is not None and type(row["prediction"]) is not int): raise ValueError("recorded held-out parsing disagrees")
                    correct[arm][seed][row["sample_id"]] = row["prediction"] is not None and str(row["prediction"]) == str(heldout_labels[row["sample_id"]]["answer"])
                by_edit = {}; by_answer = {}
                for kind in ("clean", "answer_changing", "answer_preserving"):
                    support = [sample for sample, label in heldout_labels.items() if label["edit_kind"] == kind]
                    if len(support) != 96: raise ValueError("held-out variant support differs")
                    count = sum(correct[arm][seed][sample] for sample in support)
                    by_edit[kind] = {"count": len(support), "correct": count, "accuracy": count / len(support)}
                for answer in range(5):
                    support = [sample for sample, label in heldout_labels.items() if int(label["answer"]) == answer]
                    count = sum(correct[arm][seed][sample] for sample in support)
                    by_answer[str(answer)] = {"count": len(support), "correct": count, "accuracy": count / len(support) if support else None}
                subgroups[arm][str(seed)] = {"by_edit_kind": by_edit, "by_answer": by_answer}
                if arm != "base":
                    directory = args.workspace / "updates" / family / arm
                    examples = rows(directory / "update_examples.jsonl")
                    validate_consumed_examples({"training": record["training"], "seed": seed, "update_manifest": load(directory / "update_manifest.json")}, examples)
                    selected_precision[arm] = sum(int(row["claimed_answer"]) == int(mapped[row["sample_id"]]["answer"]) for row in examples) / 48
                    selected_ids[arm] = {row["sample_id"] for row in examples}
                    selected_scenes[arm] = {row["source_scene_id"] for row in examples}
                    selected_composition[arm] = {"selected_count": 48, "correct_targets": round(selected_precision[arm] * 48), "incorrect_targets": 48 - round(selected_precision[arm] * 48), "selected_target_precision": selected_precision[arm], "by_edit_kind": {kind: sum(mapped[row["sample_id"]]["edit_kind"] == kind for row in examples) for kind in ("clean", "answer_changing", "answer_preserving")}, "training_target_distribution": {str(answer): sum(str(row["claimed_answer"]) == str(answer) for row in examples) for answer in range(5)}, "oracle_answer_distribution": {str(answer): sum(int(mapped[row["sample_id"]]["answer"]) == answer for row in examples) for answer in range(5)}}
                elif record["training"] is not None: raise ValueError("recorded no-update arm contains training")
        means = {arm: {pair: sum(int(correct[arm][seed][sample]) for seed in seeds for sample in samples) / 9 for pair, samples in sorted(pairs.items())} for arm in ARMS}
        comparisons = {}
        for reference, compared, offset in (("base", ARMS[1:], 0), ("same_model", ("independent", "consensus", "qwen_8b", "independent_8b"), 100)):
            comparisons[reference] = {arm: bootstrap([means[arm][pair] - means[reference][pair] for pair in sorted(pairs)], 20000, 20261024 + offset + len(arm)) for arm in compared}
        overlaps = {left + "__" + right: {"shared_samples": len(selected_ids[left] & selected_ids[right]), "shared_source_scenes": len(selected_scenes[left] & selected_scenes[right])} for index, left in enumerate(ARMS[1:]) for right in ARMS[index + 2:]}
        write(args.workspace / "replay_analysis" / (family + ".json"), {"execution_mode": "RECORDED_RESPONSE_REPLAY_NO_MODEL_CALLS", "pair_count": 96, "unique_items": 288, "training_seeds": seeds, "arms": {arm: {"accuracy": sum(sum(correct[arm][seed].values()) for seed in seeds) / 864, "seed_accuracies": {str(seed): sum(correct[arm][seed].values()) / 288 for seed in seeds}} for arm in ARMS}, "selected_target_precision": selected_precision, "selected_composition": selected_composition, "selected_overlap": overlaps, "conditional_subgroups": subgroups, "comparisons": comparisons, "development_gate_metrics": gates, "image_sensitivity": sensitivity})
    print("Recorded responses, generated oracle answers, selection and conditional paired statistics verified without model calls.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("benchmark", "prepare", "models", "propose", "check", "select", "train", "aggregate", "replay"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.json")
    parser.add_argument("--workspace", type=Path, default=Path("output"))
    parser.add_argument("--model-cache", type=Path, default=Path("model_cache"))
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--model", choices=("qwen3_vl_2b_instruct", "internvl3_5_2b_instruct", "qwen3_vl_8b_instruct", "internvl3_5_8b"))
    parser.add_argument("--condition", choices=("original", "no_image", "shuffled"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    args.workspace = args.workspace.resolve(); args.model_cache = args.model_cache.resolve()
    cfg = config(args.config)
    globals()[args.stage](args, cfg)


if __name__ == "__main__":
    main()
