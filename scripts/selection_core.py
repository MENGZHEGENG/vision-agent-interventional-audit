from __future__ import annotations
import argparse, hashlib, json, os, random, re, struct, zlib
from pathlib import Path
from typing import Any

ANSWER_RE = re.compile('[0-4]')

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def _candidate_maps(rows: list[dict[str, Any]], same_model_key: str, independent_model_key: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    selected = [row for row in rows if row.get('split') == 'evolution' and row.get('condition') == 'original' and (row.get('model_key') in {same_model_key, independent_model_key})]
    maps: dict[str, dict[str, Any]] = {same_model_key: {}, independent_model_key: {}}
    for row in selected:
        sample_id = row.get('sample_id')
        model_key = row.get('model_key')
        if not isinstance(sample_id, str) or not isinstance(model_key, str) or sample_id in maps[model_key]:
            raise ValueError('evaluator rows have duplicate or invalid update candidate ID')
        for field in ('question', 'source_scene_id', 'candidate_parse_status', 'candidate_answer', 'candidate_image_sha256', 'parsed_verdict'):
            if field not in row:
                raise ValueError(f'evaluator row is missing {field}: {sample_id}')
        if row['candidate_parse_status'] == 'PARSEABLE':
            if not isinstance(row['candidate_answer'], str) or not ANSWER_RE.fullmatch(row['candidate_answer']):
                raise ValueError(f'candidate answer is not a canonical integer: {sample_id}')
        elif row['candidate_parse_status'] != 'INVALID_FORMAT' or row.get('candidate_answer') is not None:
            raise ValueError(f'invalid candidate parse status: {sample_id}')
        maps[model_key][sample_id] = row
    if set(maps[same_model_key]) != set(maps[independent_model_key]) or not maps[same_model_key]:
        raise ValueError('same-model and independent lanes do not cover the same evolution IDs')
    for sample_id, same in maps[same_model_key].items():
        independent = maps[independent_model_key][sample_id]
        for field in ('question', 'source_scene_id', 'candidate_answer', 'candidate_parse_status', 'candidate_image_sha256'):
            if same.get(field) != independent.get(field):
                raise ValueError(f'checker lanes disagree about frozen candidate: {sample_id}/{field}')
    return (maps[same_model_key], maps[independent_model_key])

def _select(candidates: list[dict[str, Any]], rule: str, budget: int, seed: int) -> list[dict[str, Any]]:
    if budget < 1:
        raise ValueError('fixed budget must be positive')
    order = sorted(candidates, key=lambda row: sha256_bytes(f"{seed}:{row['sample_id']}".encode()))
    selected: list[dict[str, Any]] = []
    scenes: set[str] = set()
    for row in order:
        scene = row.get('source_scene_id')
        if not isinstance(scene, str) or scene in scenes:
            continue
        selected.append(row)
        scenes.add(scene)
        if len(selected) == budget:
            break
    if len(selected) != budget:
        raise ValueError(f'{rule} has only {len(selected)} source-scene-unique candidates for budget {budget}')
    return selected
