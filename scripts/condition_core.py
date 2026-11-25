from __future__ import annotations
import argparse, hashlib, json, os, random, re, struct, zlib
from pathlib import Path
from typing import Any

def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def _derived_seed(seed: int, purpose: str) -> int:
    payload = f'rsi_visual_verifier_bundle_v1\x00{seed}\x00{purpose}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')

def _build_derangement(dataset: dict[str, Any], seed: int) -> dict[str, dict[str, Any]]:
    records = dataset['records']
    strata: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for sample_id, record in records.items():
        label = record['label']
        question = record['policy']['question']
        key = (record['split'], record['target'], label['edit_kind'], question)
        scenes = strata.setdefault(key, {})
        source_id = label['source_id']
        if source_id in scenes:
            raise ValueError(f'duplicate scene in shuffle stratum: {key}/{source_id}')
        scenes[source_id] = sample_id
    recipient_to_donor: dict[str, dict[str, Any]] = {}
    for (split, target, edit_kind, question), scene_to_sample in sorted(strata.items()):
        scene_ids = sorted(scene_to_sample)
        if len(scene_ids) < 2:
            raise ValueError(f'cannot derange a singleton stratum: {split}/{target}/{edit_kind}')
        stratum_digest = _sha256_bytes(question.encode('utf-8'))
        rng = random.Random(_derived_seed(seed, f'derangement:{split}:{target}:{edit_kind}:{stratum_digest}'))
        rng.shuffle(scene_ids)
        offset = rng.randrange(1, len(scene_ids))
        donors = {recipient_scene: scene_ids[(index + offset) % len(scene_ids)] for index, recipient_scene in enumerate(scene_ids)}
        if len(set(donors.values())) != len(scene_ids) or any((recipient == donor for recipient, donor in donors.items())):
            raise AssertionError('internal derangement construction failed')
        for recipient_scene, donor_scene in donors.items():
            recipient_id = scene_to_sample[recipient_scene]
            donor_id = scene_to_sample[donor_scene]
            recipient = records[recipient_id]
            donor = records[donor_id]
            if recipient['split'] != donor['split'] or recipient['target'] != donor['target'] or recipient['label']['edit_kind'] != donor['label']['edit_kind'] or (recipient['label']['source_id'] == donor['label']['source_id']):
                raise AssertionError('internal derangement crossed a required stratum')
            recipient_to_donor[recipient_id] = {'recipient_sample_id': recipient_id, 'recipient_source_id': recipient['label']['source_id'], 'split': split, dataset['target_field']: target, 'edit_kind': edit_kind, 'question': question, 'recipient_image_sha256': recipient['image_sha256'], 'donor_sample_id': donor_id, 'donor_source_id': donor['label']['source_id'], 'donor_image_sha256': donor['image_sha256']}
    if set(recipient_to_donor) != set(records):
        raise ValueError('derangement does not cover every recipient row')
    return recipient_to_donor
