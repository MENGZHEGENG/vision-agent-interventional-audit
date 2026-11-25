from __future__ import annotations
import argparse, hashlib, json, os, random, re, struct, zlib
from pathlib import Path
from typing import Any

SPLITS = ('evolution', 'selection', 'heldout')

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

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
                raise ValueError(f'blank JSONL row: {path}:{line_number}')
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f'expected JSON object: {path}:{line_number}')
            rows.append(value)
    return rows

def parse_integer(text: str) -> int | None:
    match = re.search('(?<!\\d)(\\d+)(?!\\d)', text or '')
    return int(match.group(1)) if match else None

def qwen_call(model: Any, processor: Any, image_path: Path, prompt: str) -> str:
    import torch
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors='pt', padding=True)
    inputs = {key: value.to('cuda') if hasattr(value, 'to') else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    prompt_length = inputs['input_ids'].shape[1]
    response = processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)[0].strip()
    del image, messages, inputs, generated
    return response

def load_policy(dataset_root: Path, expected_items_per_split: int, selected_splits: tuple[str, ...]=SPLITS) -> list[dict[str, Any]]:
    if not selected_splits or len(set(selected_splits)) != len(selected_splits) or any((split not in SPLITS for split in selected_splits)):
        raise ValueError('invalid requested policy split')
    rows: list[dict[str, Any]] = []
    for split in selected_splits:
        split_rows = read_jsonl(dataset_root / 'policy_inputs' / f'{split}.jsonl')
        if len(split_rows) != expected_items_per_split * 3:
            raise ValueError(f'unexpected policy count for {split}: {len(split_rows)}')
        for row in split_rows:
            if set(row) != {'sample_id', 'image_path', 'question'}:
                raise ValueError(f"policy input contains evaluator fields: {split}/{row.get('sample_id')}")
            image_path = (dataset_root / str(row['image_path'])).resolve(strict=True)
            if dataset_root.resolve(strict=True) not in image_path.parents or not image_path.is_file():
                raise ValueError(f"invalid policy image path: {row['image_path']}")
            rows.append({'split': split, **row, 'image_path': str(image_path), 'image_sha256': sha256_file(image_path)})
    return rows
