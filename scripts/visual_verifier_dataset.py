from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any
from render_scenes import COLORS, HEIGHT, WIDTH, render_scene
DEFAULT_SEED = 20261003
DEFAULT_ITEMS_PER_SPLIT = 96
SPLITS = ('evolution', 'selection', 'heldout')
SPLIT_FAMILY = {'evolution': 'grid_a', 'selection': 'grid_b', 'heldout': 'grid_c'}
SPLIT_OFFSET = {'evolution': 11, 'selection': 23, 'heldout': 37}
SHAPES = ('square', 'circle', 'triangle')
TASK_FAMILIES = ('shape_count', 'color_count')
DEFAULT_TASK_FAMILY = 'shape_count'
POLICY_FIELDS = {'sample_id', 'image_path', 'question'}
COMMON_LABEL_FIELDS = {'sample_id', 'split', 'pair_id', 'edit_kind', 'answer_delta', 'source_id', 'image_sha256', 'question', 'answer', 'scene'}

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write('\n')

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(',', ':')))
            handle.write('\n')

def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'expected JSON object: {path}')
    return value

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f'blank JSONL row at {path}:{line_number}')
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f'expected JSON object: {path}:{line_number}')
            rows.append(value)
    return rows

def inventory(root: Path, excluded: set[str] | None=None) -> list[dict[str, Any]]:
    excluded = excluded or set()
    rows = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': sha256_file(path)})
    return rows

def _positions(family: str) -> list[tuple[int, int]]:
    if family == 'grid_a':
        return [(18, 18), (96, 18), (174, 18), (18, 96), (96, 96), (174, 96)]
    if family == 'grid_b':
        return [(24, 24), (102, 24), (180, 24), (24, 94), (102, 94), (180, 94)]
    if family == 'grid_c':
        return [(20, 30), (98, 30), (176, 30), (20, 100), (98, 100), (176, 100)]
    raise ValueError(f'unknown scene family: {family}')

def make_scene(seed: int, split: str, index: int, task_family: str, target: str, target_count: int) -> dict[str, Any]:
    if task_family not in TASK_FAMILIES:
        raise ValueError(f'unknown task family: {task_family}')
    if not 0 <= target_count <= 4:
        raise ValueError(f'target_count must be in [0, 4], got {target_count}')
    family = SPLIT_FAMILY[split]
    rng = random.Random(seed * 100003 + index * 7919 + SPLIT_OFFSET[split])
    if task_family == 'shape_count':
        if target not in SHAPES:
            raise ValueError(f'invalid shape target: {target}')
        alternatives = [shape for shape in SHAPES if shape != target]
        rng.shuffle(alternatives)
        remaining = 6 - target_count
        first_count = remaining // 2
        shape_order = [target] * target_count + [alternatives[0]] * first_count + [alternatives[1]] * (remaining - first_count)
        colors = [rng.choice(COLORS) for _ in range(6)]
    else:
        if target not in COLORS:
            raise ValueError(f'invalid color target: {target}')
        alternatives = [color for color in COLORS if color != target]
        rng.shuffle(alternatives)
        remaining = 6 - target_count
        quotient, remainder = divmod(remaining, len(alternatives))
        colors = [target] * target_count + [color for color in alternatives for _ in range(quotient)] + alternatives[:remainder]
        shape_order = [rng.choice(SHAPES) for _ in range(6)]
    rng.shuffle(shape_order)
    rng.shuffle(colors)
    positions = _positions(family)
    rng.shuffle(positions)
    objects = []
    for object_index, shape in enumerate(shape_order):
        x, y = positions[object_index]
        objects.append({'id': f'obj_{object_index}', 'color': colors[object_index], 'shape': shape, 'x': x + rng.randrange(0, 5), 'y': y + rng.randrange(0, 5), 'width': 36, 'height': 34})
    return {'width': WIDTH, 'height': HEIGHT, 'objects': objects, 'generator_family': family}

def _copy_scene(scene: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(scene)

def variant_scene(scene: dict[str, Any], edit_kind: str, target: str, task_family: str, answer_delta: int=0, seed: int=0) -> dict[str, Any]:
    changed = _copy_scene(scene)
    objects = changed['objects']
    if edit_kind == 'clean':
        return changed
    if edit_kind == 'answer_changing':
        if answer_delta not in {-1, 1}:
            raise ValueError('answer-changing edit requires answer_delta -1 or +1')
        if task_family == 'shape_count':
            field = 'shape'
            alternatives = [shape for shape in SHAPES if shape != target]
        else:
            field = 'color'
            alternatives = [color for color in COLORS if color != target]
        if answer_delta == 1:
            candidates = [obj for obj in objects if obj[field] != target]
            if not candidates:
                raise ValueError('cannot increase a target count in this scene')
            candidates[seed % len(candidates)][field] = target
        else:
            candidates = [obj for obj in objects if obj[field] == target]
            if not candidates:
                raise ValueError('cannot decrease a zero target count')
            candidates[seed % len(candidates)][field] = alternatives[seed % len(alternatives)]
        return changed
    if edit_kind == 'answer_preserving':
        if task_family == 'shape_count':
            target_object = next((obj for obj in objects if obj['shape'] != target))
            current = target_object['color']
            target_object['color'] = next((color for color in COLORS if color != current))
        else:
            target_object = next((obj for obj in objects if obj['color'] != target))
            current = target_object['shape']
            target_object['shape'] = next((shape for shape in SHAPES if shape != current))
        return changed
    raise ValueError(f'unknown edit kind: {edit_kind}')

def _answer(scene: dict[str, Any], target: str, task_family: str) -> str:
    key = 'shape' if task_family == 'shape_count' else 'color'
    return str(sum((obj[key] == target for obj in scene['objects'])))

def _targets(task_family: str) -> tuple[str, ...]:
    return SHAPES if task_family == 'shape_count' else tuple(COLORS)

def _balanced_design(items_per_split: int, task_family: str, seed: int, split: str) -> list[tuple[str, int, int]]:
    """Balance answer marginals and changing-edit directions within each target."""
    targets = _targets(task_family)
    if items_per_split < 96 or items_per_split % 96:
        raise ValueError('items_per_split must be a positive multiple of 96 for exact balance')
    rows: list[tuple[str, int, int]] = []
    for target_index, target in enumerate(targets):
        for answer_count, total_for_count in ((0, items_per_split // 8), (1, items_per_split // 4), (2, items_per_split // 4), (3, items_per_split // 4), (4, items_per_split // 8)):
            per_target = total_for_count // len(targets)
            if answer_count == 0:
                deltas = [1] * per_target
            elif answer_count == 4:
                deltas = [-1] * per_target
            else:
                if per_target % 2:
                    raise ValueError('per-target transition counts must be even')
                deltas = [-1] * (per_target // 2) + [1] * (per_target // 2)
            local_rng = random.Random(seed * 1000003 + SPLIT_OFFSET[split] * 1009 + target_index * 101 + answer_count)
            local_rng.shuffle(deltas)
            rows.extend(((target, answer_count, delta) for delta in deltas))
    random.Random(seed * 900001 + SPLIT_OFFSET[split]).shuffle(rows)
    if len(rows) != items_per_split:
        raise AssertionError('balanced design has the wrong number of rows')
    return rows

def _parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple((part.strip() for part in value.split(',') if part.strip()))
    if not splits or len(set(splits)) != len(splits) or any((split not in SPLITS for split in splits)):
        raise ValueError(f'splits must be a unique comma-separated subset of {SPLITS}')
    return splits

def generate(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed < 1:
        raise ValueError('seed must be positive')
    if args.task_family not in TASK_FAMILIES:
        raise ValueError(f'unknown task family: {args.task_family}')
    if not re.fullmatch('[0-9a-f]{40}', args.source_commit):
        raise ValueError('source_commit must be a full lowercase Git commit SHA')
    if args.items_per_split < 96 or args.items_per_split % 96:
        raise ValueError('items_per_split must be a positive multiple of 96')
    selected_splits = _parse_splits(args.splits)
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'refusing to overwrite non-empty output root: {root}')
    root.mkdir(parents=True, exist_ok=True)
    generator_path = Path(__file__).resolve()
    renderer_path = generator_path.with_name('render_scenes.py')
    if not renderer_path.is_file():
        raise FileNotFoundError(f'renderer dependency is missing: {renderer_path}')
    policy_hashes: dict[str, str] = {}
    label_hashes: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    all_image_hashes: set[str] = set()
    all_source_ids: set[str] = set()
    scene_counts: dict[str, int] = {}
    for split in selected_splits:
        policy_rows: list[dict[str, Any]] = []
        label_rows: list[dict[str, Any]] = []
        design = _balanced_design(args.items_per_split, args.task_family, args.seed, split)
        target_field = 'target_shape' if args.task_family == 'shape_count' else 'target_color'
        for index, (target, clean_answer, changing_delta) in enumerate(design):
            pair_id = hashlib.sha256(f'source:v2:{args.task_family}:{args.seed}:{split}:{index}'.encode()).hexdigest()[:24]
            if pair_id in all_source_ids:
                raise ValueError('global source scene ID collision')
            all_source_ids.add(pair_id)
            base_scene = make_scene(args.seed, split, index, args.task_family, target, clean_answer)
            question = f'How many {target} objects are visible in the image? Answer with one integer.'
            for edit_kind in ('clean', 'answer_changing', 'answer_preserving'):
                sample_id = hashlib.sha256(f'sample:v2:{args.task_family}:{args.seed}:{split}:{index}:{edit_kind}'.encode()).hexdigest()[:24]
                delta = changing_delta if edit_kind == 'answer_changing' else 0
                scene = variant_scene(base_scene, edit_kind, target, args.task_family, delta, index)
                image_bytes = render_scene(scene)
                relative = f'images/{sample_id}.png'
                image_path = root / relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(image_bytes)
                image_hash = sha256_bytes(image_bytes)
                if image_hash in all_image_hashes:
                    raise ValueError('global image hash collision')
                all_image_hashes.add(image_hash)
                policy_rows.append({'sample_id': sample_id, 'image_path': relative, 'question': question})
                label_rows.append({'sample_id': sample_id, 'split': split, 'pair_id': pair_id, 'edit_kind': edit_kind, 'source_id': pair_id, 'image_sha256': image_hash, 'question': question, 'answer_delta': delta, target_field: target, 'answer': _answer(scene, target, args.task_family), 'scene': scene})
        policy_path = root / 'policy_inputs' / f'{split}.jsonl'
        label_path = root / 'evaluator_labels' / f'{split}.jsonl'
        write_jsonl(policy_path, policy_rows)
        write_jsonl(label_path, label_rows)
        policy_hashes[split] = sha256_file(policy_path)
        label_hashes[split] = sha256_file(label_path)
        split_counts[split] = len(policy_rows)
        scene_counts[split] = len(design)
    manifest = {'schema': 'rsi_visual_verifier_dataset_v2', 'source_commit': args.source_commit, 'generator_sha256': sha256_file(generator_path), 'renderer_sha256': sha256_file(renderer_path), 'seed': args.seed, 'task_family': args.task_family, 'items_per_split': args.items_per_split, 'variants_per_pair': 3, 'selected_splits': list(selected_splits), 'splits': {split: {'generator_family': SPLIT_FAMILY[split], 'count': split_counts[split], 'base_scenes': scene_counts[split]} for split in selected_splits}, 'policy_fields': sorted(POLICY_FIELDS), 'evaluator_fields': sorted(COMMON_LABEL_FIELDS | {'target_shape' if args.task_family == 'shape_count' else 'target_color'}), 'external_assets': [], 'license_status': 'Program-generated data; no third-party images; software and data reuse terms unspecified', 'geometry': {'width': WIDTH, 'height': HEIGHT, 'objects_per_scene': 6, 'shapes': list(SHAPES), 'colors': list(COLORS)}, 'policy_sha256': policy_hashes, 'evaluator_sha256': label_hashes, 'total_items': sum(split_counts.values()), 'unique_image_count': len(all_image_hashes), 'unique_source_count': len(all_source_ids), 'answer_distribution_per_edit': {'0': args.items_per_split // 8, '1': args.items_per_split // 4, '2': args.items_per_split // 4, '3': args.items_per_split // 4, '4': args.items_per_split // 8}, 'files': inventory(root)}
    write_json(root / 'dataset_manifest.json', manifest)
    receipt = {'schema': 'rsi_visual_verifier_dataset_build_v2', 'status': 'SUCCEEDED', 'source_commit': args.source_commit, 'seed': args.seed, 'task_family': args.task_family, 'items_per_split': args.items_per_split, 'dataset_root': str(root), 'manifest_sha256': sha256_file(root / 'dataset_manifest.json'), 'total_items': manifest['total_items'], 'unique_image_count': manifest['unique_image_count'], 'unique_source_count': manifest['unique_source_count'], 'selected_splits': list(selected_splits)}
    write_json(root / 'build_receipt.json', receipt)
    return receipt

def validate(args: argparse.Namespace) -> dict[str, Any]:
    root = args.dataset_root.resolve(strict=True)
    manifest = read_json(root / 'dataset_manifest.json')
    if manifest.get('schema') != 'rsi_visual_verifier_dataset_v2':
        raise ValueError('unexpected dataset schema')
    if manifest.get('source_commit') != args.source_commit or manifest.get('seed') != args.seed:
        raise ValueError('source commit or seed mismatch')
    if manifest.get('task_family') != args.task_family:
        raise ValueError('task family mismatch')
    generator_path = Path(__file__).resolve()
    renderer_path = generator_path.with_name('render_scenes.py')
    if manifest.get('generator_sha256') != sha256_file(generator_path) or manifest.get('renderer_sha256') != sha256_file(renderer_path):
        raise ValueError('dataset generator or renderer hash does not match the staged source files')
    if manifest.get('items_per_split') != args.items_per_split:
        raise ValueError('items_per_split mismatch')
    selected_splits = _parse_splits(args.splits)
    if manifest.get('selected_splits') != list(selected_splits) or set(manifest.get('splits', {})) != set(selected_splits):
        raise ValueError('selected split mismatch')
    if manifest.get('external_assets') != []:
        raise ValueError('external assets are not permitted')
    expected_files = manifest.get('files')
    if not isinstance(expected_files, list):
        raise ValueError('file inventory missing')
    for entry in expected_files:
        if not isinstance(entry, dict):
            raise ValueError('invalid inventory entry')
        path = (root / str(entry['path'])).resolve(strict=True)
        if root not in path.parents or not path.is_file() or path.is_symlink():
            raise ValueError(f"invalid dataset path: {entry['path']}")
        if path.stat().st_size != entry['bytes'] or sha256_file(path) != entry['sha256']:
            raise ValueError(f"dataset file changed: {entry['path']}")
    actual = inventory(root, excluded={'dataset_manifest.json', 'build_receipt.json', 'validation_receipt.json'})
    listed = [entry for entry in expected_files if entry['path'] not in {'dataset_manifest.json', 'build_receipt.json', 'validation_receipt.json'}]
    if actual != listed:
        raise ValueError('actual dataset inventory differs from manifest')
    all_images: set[str] = set()
    all_sources: set[str] = set()
    source_owner: dict[str, str] = {}
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    answer_distributions: dict[str, dict[str, dict[str, int]]] = {}
    target_field = 'target_shape' if args.task_family == 'shape_count' else 'target_color'
    expected_label_fields = COMMON_LABEL_FIELDS | {target_field}
    question_pattern = 'How many (square|circle|triangle) objects are visible in the image\\? Answer with one integer\\.' if args.task_family == 'shape_count' else 'How many (red|blue|green|yellow|purple|orange) objects are visible in the image\\? Answer with one integer\\.'
    for split in selected_splits:
        policy_path = root / 'policy_inputs' / f'{split}.jsonl'
        label_path = root / 'evaluator_labels' / f'{split}.jsonl'
        if sha256_file(policy_path) != manifest['policy_sha256'][split] or sha256_file(label_path) != manifest['evaluator_sha256'][split]:
            raise ValueError(f'split digest mismatch: {split}')
        policy = read_jsonl(policy_path)
        labels = read_jsonl(label_path)
        if len(policy) != args.items_per_split * 3 or len(labels) != len(policy):
            raise ValueError(f'split count mismatch: {split}')
        if len({row.get('sample_id') for row in policy}) != len(policy):
            raise ValueError(f'duplicate policy sample ID: {split}')
        if any((set(row) != POLICY_FIELDS for row in policy)):
            raise ValueError(f'policy fields leak evaluator data: {split}')
        by_id = {row['sample_id']: row for row in labels}
        if len(by_id) != len(labels):
            raise ValueError(f'duplicate sample ID: {split}')
        for row in policy:
            if set(by_id[row['sample_id']]) != expected_label_fields:
                raise ValueError(f"label fields mismatch: {split}/{row['sample_id']}")
            label = by_id[row['sample_id']]
            if label['split'] != split or label['question'] != row['question']:
                raise ValueError(f"policy/evaluator join mismatch: {split}/{row['sample_id']}")
            image_path = (root / row['image_path']).resolve(strict=True)
            if root not in image_path.parents or not image_path.is_file():
                raise ValueError(f"image path escapes dataset root: {row['image_path']}")
            image_hash = sha256_file(image_path)
            if image_hash != label['image_sha256'] or image_hash in all_images:
                raise ValueError(f"image hash mismatch or cross-split collision: {row['sample_id']}")
            all_images.add(image_hash)
            source_id = label['source_id']
            if not isinstance(source_id, str) or not re.fullmatch('[0-9a-f]{24}', source_id):
                raise ValueError(f"invalid source ID: {row['sample_id']}")
            owner = source_owner.setdefault(source_id, label['pair_id'])
            if owner != label['pair_id']:
                raise ValueError(f"source ID is shared across pairs: {row['sample_id']}")
            all_sources.add(source_id)
            if not re.fullmatch('[0-9a-f]{24}', row['sample_id']):
                raise ValueError(f"sample ID is not opaque: {row['sample_id']}")
            if row['image_path'] != f"images/{row['sample_id']}.png":
                raise ValueError(f"image path encodes non-opaque metadata: {row['sample_id']}")
            scene = label['scene']
            if scene.get('width') != WIDTH or scene.get('height') != HEIGHT or scene.get('generator_family') != SPLIT_FAMILY[split]:
                raise ValueError(f"scene geometry or generator mismatch: {row['sample_id']}")
            objects = scene.get('objects')
            if not isinstance(objects, list) or len(objects) != 6:
                raise ValueError(f"scene object count mismatch: {row['sample_id']}")
            if any((obj.get('shape') not in SHAPES or obj.get('color') not in COLORS for obj in objects)):
                raise ValueError(f"invalid scene object: {row['sample_id']}")
            target = label[target_field]
            expected_question = f'How many {target} objects are visible in the image? Answer with one integer.'
            expected_answer = _answer(scene, target, args.task_family)
            if label['answer'] != expected_answer or row['question'] != expected_question or (not re.fullmatch(question_pattern, row['question'])):
                raise ValueError(f"answer/question mismatch: {row['sample_id']}")
            if render_scene(scene) != image_path.read_bytes():
                raise ValueError(f"rendered image does not match the sealed scene program: {row['sample_id']}")
            pair_id = label['pair_id']
            variants = pairs.setdefault(pair_id, {})
            if label['edit_kind'] in variants:
                raise ValueError(f"duplicate edit variant: {pair_id}/{label['edit_kind']}")
            variants[label['edit_kind']] = label
        counts[split] = len(policy)
        answer_distributions[split] = {}
        expected_distribution = {'0': args.items_per_split // 8, '1': args.items_per_split // 4, '2': args.items_per_split // 4, '3': args.items_per_split // 4, '4': args.items_per_split // 8}
        for edit_kind in ('clean', 'answer_changing', 'answer_preserving'):
            distribution: dict[str, int] = {}
            for label in labels:
                if label['edit_kind'] == edit_kind:
                    key = str(label['answer'])
                    distribution[key] = distribution.get(key, 0) + 1
            if distribution != expected_distribution:
                raise ValueError(f'answer distribution is not balanced for {split}/{edit_kind}: {distribution}')
            answer_distributions[split][edit_kind] = distribution
    if len(all_images) != manifest.get('unique_image_count') or len(all_sources) != manifest.get('unique_source_count'):
        raise ValueError('global uniqueness counts mismatch')
    if len(pairs) != args.items_per_split * len(selected_splits):
        raise ValueError('pair count mismatch')
    for pair_id, variants in pairs.items():
        if set(variants) != {'clean', 'answer_changing', 'answer_preserving'}:
            raise ValueError(f'missing paired edit: {pair_id}')
        clean = variants['clean']
        changing = variants['answer_changing']
        preserving = variants['answer_preserving']
        if clean['source_id'] != pair_id or changing['source_id'] != pair_id or preserving['source_id'] != pair_id:
            raise ValueError(f'base-scene identity mismatch: {pair_id}')
        if changing['answer_delta'] not in {-1, 1}:
            raise ValueError(f'invalid changing-edit delta: {pair_id}')
        if int(changing['answer']) - int(clean['answer']) != changing['answer_delta']:
            raise ValueError(f'changing-edit delta does not match answers: {pair_id}')
        if changing['answer'] == clean['answer'] or preserving['answer'] != clean['answer']:
            raise ValueError(f'paired edit semantics fail: {pair_id}')
        if clean['answer_delta'] != 0 or preserving['answer_delta'] != 0:
            raise ValueError(f'non-changing edit has a nonzero answer delta: {pair_id}')
        target_key = 'shape' if args.task_family == 'shape_count' else 'color'
        other_key = 'color' if args.task_family == 'shape_count' else 'shape'
        base_objects = {obj['id']: obj for obj in clean['scene']['objects']}
        changed_objects = {obj['id']: obj for obj in changing['scene']['objects']}
        preserved_objects = {obj['id']: obj for obj in preserving['scene']['objects']}
        changed_ids = [object_id for object_id in base_objects if base_objects[object_id][target_key] != changed_objects[object_id][target_key]]
        preserved_ids = [object_id for object_id in base_objects if base_objects[object_id][other_key] != preserved_objects[object_id][other_key]]
        if len(changed_ids) != 1 or len(preserved_ids) != 1:
            raise ValueError(f'paired edit must change exactly one relevant object: {pair_id}')
        for object_id, base_object in base_objects.items():
            changed_keys = {key for key, value in base_object.items() if changed_objects[object_id].get(key) != value}
            preserved_keys = {key for key, value in base_object.items() if preserved_objects[object_id].get(key) != value}
            expected_changed = {target_key} if object_id in changed_ids else set()
            expected_preserved = {other_key} if object_id in preserved_ids else set()
            if changed_keys != expected_changed or preserved_keys != expected_preserved:
                raise ValueError(f'paired edit changed an unintended attribute: {pair_id}')
    for split in selected_splits:
        for target in _targets(args.task_family):
            for answer_count in range(5):
                matching = [variants for variants in pairs.values() if variants['clean']['split'] == split and variants['clean']['answer'] == str(answer_count) and (variants['clean'][target_field] == target)]
                deltas = [int(variants['answer_changing']['answer_delta']) for variants in matching]
                expected_per_target = args.items_per_split // (8 * len(_targets(args.task_family))) if answer_count in {0, 4} else args.items_per_split // (4 * len(_targets(args.task_family)))
                if len(matching) != expected_per_target:
                    raise ValueError(f'target/count stratum is unbalanced: {split}/{target}/{answer_count}')
                if answer_count in {0, 4}:
                    expected_delta = 1 if answer_count == 0 else -1
                    if not deltas or any((delta != expected_delta for delta in deltas)):
                        raise ValueError(f'boundary transition direction mismatch: {split}/{target}/{answer_count}')
                elif deltas.count(-1) != deltas.count(1):
                    raise ValueError(f'transition directions are not balanced: {split}/{target}/{answer_count}')
    build = read_json(root / 'build_receipt.json')
    if build.get('status') != 'SUCCEEDED' or build.get('manifest_sha256') != sha256_file(root / 'dataset_manifest.json') or build.get('selected_splits') != list(selected_splits):
        raise ValueError('build receipt does not bind current manifest')
    validation = {'schema': 'rsi_visual_verifier_dataset_validation_v1', 'status': 'ACCEPTED', 'source_commit': args.source_commit, 'task_family': args.task_family, 'dataset_root': str(root), 'dataset_manifest_sha256': sha256_file(root / 'dataset_manifest.json'), 'counts': counts, 'total_items': sum(counts.values()), 'unique_images': len(all_images), 'unique_sources': len(all_sources), 'answer_distribution': answer_distributions, 'policy_fields': sorted(POLICY_FIELDS), 'evaluator_join': 'passed', 'rendered_program_match': 'passed', 'paired_edit_semantics': 'passed', 'external_assets': []}
    if args.output:
        write_json(args.output, validation)
    return validation

def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    generate_parser = sub.add_parser('generate')
    generate_parser.add_argument('--output-root', type=Path, required=True)
    generate_parser.add_argument('--source-commit', required=True)
    generate_parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    generate_parser.add_argument('--task-family', choices=TASK_FAMILIES, default=DEFAULT_TASK_FAMILY)
    generate_parser.add_argument('--items-per-split', type=int, default=DEFAULT_ITEMS_PER_SPLIT)
    generate_parser.add_argument('--splits', default=','.join(SPLITS))
    validate_parser = sub.add_parser('validate')
    validate_parser.add_argument('--dataset-root', type=Path, required=True)
    validate_parser.add_argument('--source-commit', required=True)
    validate_parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    validate_parser.add_argument('--task-family', choices=TASK_FAMILIES, default=DEFAULT_TASK_FAMILY)
    validate_parser.add_argument('--items-per-split', type=int, default=DEFAULT_ITEMS_PER_SPLIT)
    validate_parser.add_argument('--splits', default=','.join(SPLITS))
    validate_parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'generate':
        result = generate(args)
    else:
        result = validate(args)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
