from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any
ARMS = ('base', 'random', 'same_model', 'independent', 'consensus', 'qwen_8b', 'independent_8b')
EDIT_KINDS = ('clean', 'answer_changing', 'answer_preserving')

def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a JSON object')
    return value

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f'{path}: row is not an object')
            rows.append(value)
    return rows

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def result_path(seed_dir: Path, arm: str) -> Path:
    nested = seed_dir / arm / 'result.json'
    return nested if nested.is_file() else seed_dir / f'{arm}.json'

def validate_frozen_predictions(record: dict[str, Any], frozen: Path) -> None:
    evaluation = record.get('evaluation') or {}
    if not frozen.is_file() or sha256_file(frozen) != evaluation.get('label_free_predictions_sha256'):
        raise ValueError('frozen prediction digest missing or mismatched')
    if evaluation.get('labels_opened_after_prediction_freeze') is not True:
        raise ValueError('prediction freeze chronology is not recorded')
    raw_rows = load_jsonl(frozen)
    raw = {}
    for row in raw_rows:
        if set(row) != {'sample_id', 'prediction_text', 'prediction'} or not isinstance(row['sample_id'], str) or row['sample_id'] in raw:
            raise ValueError('invalid or duplicate frozen prediction identity')
        if not isinstance(row['prediction_text'], str):
            raise ValueError('frozen prediction text is not a string')
        match = re.search('(?<!\\d)(\\d+)(?!\\d)', row['prediction_text'])
        parsed = int(match.group(1)) if match else None
        if row['prediction'] != parsed or (parsed is not None and type(row['prediction']) is not int):
            raise ValueError('frozen parsed integer disagrees with the canonical text parser')
        raw[row['sample_id']] = row
    scored = record.get('predictions', [])
    if len(scored) != len(raw) or {r.get('sample_id') for r in scored} != set(raw):
        raise ValueError('frozen and scored prediction coverage differs')
    for row in scored:
        frozen_row = raw[row['sample_id']]
        if any((row.get(key) != frozen_row[key] for key in ('prediction_text', 'prediction'))):
            raise ValueError('frozen and scored predictions disagree')

def validate_consumed_examples(record: dict[str, Any], examples: list[dict[str, Any]]) -> None:
    training = record.get('training') or {}
    budget = (record.get('update_manifest') or {}).get('fixed_budget')
    consumed = training.get('consumed_sample_ids', [])
    if len(consumed) != budget or training.get('example_presentations') != budget or training.get('steps') != budget:
        raise ValueError('executed presentation count differs from the full-epoch budget')
    selected = [row.get('sample_id') for row in examples]
    if len(selected) != budget or any((not isinstance(s, str) for s in selected)) or len(set(selected)) != budget or (set(consumed) != set(selected)):
        raise ValueError('selected and consumed example identities disagree')
    expected = list(selected)
    random.Random(record['seed']).shuffle(expected)
    if consumed != expected or training.get('order_seed') != record['seed']:
        raise ValueError('consumed order differs from the declared shuffle seed')
    if training.get('examples_seen') != budget or training.get('epochs_completed') != 1 or training.get('complete_selected_epoch') is not True:
        raise ValueError('executed full-epoch metadata disagrees with selected examples')

def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[int(q * (len(values) - 1))]

def bootstrap(values: list[float], draws: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(values)
    sampled = [sum((values[rng.randrange(n)] for _ in range(n))) / n for _ in range(draws)]
    return {'mean': sum(values) / n, 'ci95': [percentile(sampled, 0.025), percentile(sampled, 0.975)], 'clusters': n, 'draws': draws, 'seed': seed}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--expected-seeds', type=int, nargs='+', required=True)
    parser.add_argument('--expected-pairs', type=int, default=96)
    parser.add_argument('--require-complete-epoch', action='store_true')
    parser.add_argument('--update-root', type=Path, help="Sealed parent containing each admission arm's update artifacts")
    args = parser.parse_args()
    if args.draws < 1000:
        raise SystemExit('--draws must be at least 1000')
    if args.require_complete_epoch and args.update_root is None:
        parser.error('--update-root is required to verify selected/consumed identities')
    labels_path = args.dataset_root / 'evaluator_labels' / 'heldout.jsonl'
    if args.require_complete_epoch:
        dataset_manifest = load_json(args.dataset_root / 'dataset_manifest.json')
        label_entries = [entry for entry in dataset_manifest.get('files', []) if entry.get('path') == 'evaluator_labels/heldout.jsonl']
        if len(label_entries) != 1 or label_entries[0].get('sha256') != sha256_file(labels_path) or label_entries[0].get('bytes') != labels_path.stat().st_size:
            raise ValueError('held-out oracle bytes differ from the sealed dataset inventory')
    labels = load_jsonl(labels_path)
    by_sample: dict[str, dict[str, Any]] = {}
    by_pair: dict[str, list[str]] = {}
    for row in labels:
        sample_id = row.get('sample_id')
        pair_id = row.get('pair_id')
        if not isinstance(sample_id, str) or not isinstance(pair_id, str):
            raise ValueError(f'{labels_path}: missing sample_id or pair_id')
        if sample_id in by_sample:
            raise ValueError(f'{labels_path}: duplicate sample_id {sample_id}')
        by_sample[sample_id] = row
        by_pair.setdefault(pair_id, []).append(sample_id)
    if not by_pair or any((len(ids) != 3 for ids in by_pair.values())):
        raise ValueError('held-out labels must contain exactly three variants per pair')
    if len(by_pair) != args.expected_pairs:
        raise ValueError('held-out pair count differs from the declared protocol')
    dataset_sha = sha256_file(args.dataset_root / 'dataset_manifest.json')
    records: dict[int, dict[str, dict[str, Any]]] = {}
    rows: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
    seed_dirs = sorted((path for path in args.results_root.iterdir() if path.is_dir() and path.name.startswith('seed')))
    for seed_dir in seed_dirs:
        loaded: dict[str, tuple[dict[str, Any], Path]] = {}
        for arm in ARMS:
            path = result_path(seed_dir, arm)
            if not path.is_file():
                loaded = {}
                break
            loaded[arm] = (load_json(path), path)
        if len(loaded) != len(ARMS):
            raise ValueError(f'{seed_dir}: incomplete arm matrix')
        seeds = {int(record['seed']) for record, _ in loaded.values()}
        commits = {record['source_commit'] for record, _ in loaded.values()}
        manifests = {record['dataset_manifest_sha256'] for record, _ in loaded.values()}
        if len(seeds) != 1 or len(commits) != 1 or len(manifests) != 1:
            raise ValueError(f'{seed_dir}: arm provenance mismatch')
        seed = seeds.pop()
        if seed in records or manifests != {dataset_sha}:
            raise ValueError(f'{seed_dir}: duplicate seed or dataset manifest mismatch')
        arm_rows: dict[str, dict[str, dict[str, Any]]] = {}
        for arm, (record, path) in loaded.items():
            if record.get('status') != 'SUCCEEDED':
                raise ValueError(f'{path}: status is not SUCCEEDED')
            if record.get('arm') != arm:
                raise ValueError(f'{path}: result arm identity mismatch')
            if args.require_complete_epoch:
                frozen = path.parent / 'predictions_unscored.jsonl'
                validate_frozen_predictions(record, frozen)
                if arm != 'base':
                    t = record.get('training') or {}
                    u = record.get('update_manifest') or {}
                    expected_checker = {'qwen_8b': 'qwen3_vl_8b_instruct', 'independent_8b': 'internvl3_5_8b'}.get(arm, 'internvl3_5_2b_instruct')
                    if u.get('independent_model_key') != expected_checker or u.get('same_model_key') != 'qwen3_vl_2b_instruct' or u.get('admission_rule') != arm:
                        raise ValueError(f'{path}: executed checker identity differs from the declared arm')
                    ids = t.get('consumed_sample_ids', [])
                    if t.get('steps') != u.get('fixed_budget') or t.get('examples_seen') != u.get('fixed_budget') or len(set(ids)) != u.get('fixed_budget') or (t.get('complete_selected_epoch') is not True):
                        raise ValueError(f'{path}: incomplete selected-example exposure')
                    update_dir = args.update_root / arm
                    if load_json(update_dir / 'update_manifest.json') != u:
                        raise ValueError(f'{path}: embedded update manifest differs from sealed artifact')
                    examples_path = update_dir / 'update_examples.jsonl'
                    inventories = [row for row in u.get('files', []) if row.get('path') == 'update_examples.jsonl']
                    if len(inventories) != 1 or inventories[0].get('sha256') != sha256_file(examples_path) or inventories[0].get('bytes') != examples_path.stat().st_size:
                        raise ValueError(f'{path}: selected examples differ from sealed file identity')
                    examples = load_jsonl(examples_path)
                    for field, digest_field in (('sample_id', 'selected_sample_ids_sha256'), ('source_scene_id', 'selected_source_scene_ids_sha256')):
                        digest = hashlib.sha256('\n'.join((str(row[field]) for row in examples)).encode()).hexdigest()
                        if digest != u.get(digest_field):
                            raise ValueError(f'{path}: selected identity digest disagrees with sealed examples')
                    validate_consumed_examples(record, examples)
            current = {}
            for row in record.get('predictions', []):
                sample_id = row.get('sample_id')
                if not isinstance(sample_id, str) or sample_id in current:
                    raise ValueError(f'{path}: duplicate or invalid sample_id')
                if sample_id not in by_sample or not isinstance(row.get('correct'), bool):
                    raise ValueError(f'{path}: prediction is not covered by held-out labels')
                answer = str(by_sample[sample_id]['answer'])
                correct = row.get('prediction') is not None and str(row['prediction']) == answer
                if row['correct'] != correct or str(row.get('ground_truth_answer')) != answer:
                    raise ValueError(f'{path}: stored correctness or target disagrees with held-out oracle')
                if args.require_complete_epoch and row.get('edit_kind') != by_sample[sample_id].get('edit_kind'):
                    raise ValueError(f'{path}: stored edit kind disagrees with held-out oracle')
                current[sample_id] = row
            if set(current) != set(by_sample):
                raise ValueError(f'{path}: prediction coverage differs from held-out labels')
            arm_rows[arm] = current
        records[seed] = {arm: record for arm, (record, _) in loaded.items()}
        rows[seed] = arm_rows
    if len(records) < 2:
        raise SystemExit('at least two complete seed directories are required')
    seeds = sorted(records)
    if seeds != sorted(args.expected_seeds) or len(set(args.expected_seeds)) != len(args.expected_seeds):
        raise ValueError('effective seed matrix differs from the declared protocol')
    all_records = [record for by_arm in records.values() for record in by_arm.values()]
    for field in ('source_commit', 'dataset_manifest_sha256', 'asset_receipt_sha256'):
        if len({record.get(field) for record in all_records}) != 1:
            raise ValueError(f'cross-seed provenance mismatch: {field}')
    for arm in ARMS[1:]:
        signatures = []
        for seed in seeds:
            r = records[seed][arm]
            t = r.get('training') or {}
            u = r.get('update_manifest') or {}
            signatures.append(json.dumps({'training': {k: t.get(k) for k in ('steps', 'examples_available', 'examples_seen', 'lora_rank', 'lora_alpha', 'learning_rate', 'target_modules')}, 'update_files': u.get('files')}, sort_keys=True))
        if len(set(signatures)) != 1:
            raise ValueError(f'cross-seed scientific settings or selected-set mismatch: {arm}')
    pair_ids = sorted(by_pair)
    pair_accuracy: dict[str, dict[str, list[float]]] = {arm: {} for arm in ARMS}
    pair_delta: dict[str, list[float]] = {arm: [] for arm in ARMS[1:]}
    for pair_id in pair_ids:
        ids = by_pair[pair_id]
        for arm in ARMS:
            values = [int(rows[seed][arm][sample_id]['correct']) for seed in seeds for sample_id in ids]
            pair_accuracy[arm][pair_id] = values
        base_mean = sum(pair_accuracy['base'][pair_id]) / len(pair_accuracy['base'][pair_id])
        for arm in ARMS[1:]:
            arm_mean = sum(pair_accuracy[arm][pair_id]) / len(pair_accuracy[arm][pair_id])
            pair_delta[arm].append(arm_mean - base_mean)
    arms_summary: dict[str, Any] = {}
    for arm in ARMS:
        all_values = [value for pair_values in pair_accuracy[arm].values() for value in pair_values]
        arms_summary[arm] = {'accuracy': sum(all_values) / len(all_values), 'items': len(all_values), 'pairs': len(pair_ids), 'seeds': seeds, 'seed_accuracies': {str(seed): sum((int(row['correct']) for row in rows[seed][arm].values())) / len(by_sample) for seed in seeds}}
    paired: dict[str, Any] = {}
    for arm in ARMS[1:]:
        paired[arm] = {'delta': sum(pair_delta[arm]) / len(pair_delta[arm]), 'cluster_bootstrap': bootstrap(pair_delta[arm], args.draws, args.seed + len(arm)), 'seed_deltas': {str(seed): (sum((int(row['correct']) for row in rows[seed][arm].values())) - sum((int(row['correct']) for row in rows[seed]['base'].values()))) / len(by_sample) for seed in seeds}}
    first = records[seeds[0]]['base']
    versus_self = {}
    for arm in ('independent', 'consensus', 'qwen_8b', 'independent_8b'):
        deltas = [sum(pair_accuracy[arm][p]) / len(pair_accuracy[arm][p]) - sum(pair_accuracy['same_model'][p]) / len(pair_accuracy['same_model'][p]) for p in pair_ids]
        versus_self[arm] = {'delta': sum(deltas) / len(deltas), 'cluster_bootstrap': bootstrap(deltas, args.draws, args.seed + 100 + len(arm)), 'seed_deltas': {str(seed): (sum((int(row['correct']) for row in rows[seed][arm].values())) - sum((int(row['correct']) for row in rows[seed]['same_model'].values()))) / len(by_sample) for seed in seeds}}
    summary = {'schema': 'rsi_visual_verifier_cluster_aggregate_v1', 'results_root': str(args.results_root), 'dataset_root': str(args.dataset_root), 'dataset_manifest_sha256': first['dataset_manifest_sha256'], 'source_commit': first['source_commit'], 'seeds': seeds, 'pair_count': len(pair_ids), 'variants_per_pair': 3, 'uncertainty_unit': 'pair_id, averaging three variants and repeated training seeds before bootstrap', 'uncertainty_scope': 'Conditional on the selected sets and observed fitted seeds; excludes new optimizer-seed and selection-set uncertainty', 'arms': arms_summary, 'paired_vs_base': paired, 'paired_vs_self_check': versus_self}
    payload = json.dumps(summary, sort_keys=True, indent=2) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'sha256': hashlib.sha256(payload.encode()).hexdigest(), 'seeds': seeds, 'pairs': len(pair_ids)}))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
