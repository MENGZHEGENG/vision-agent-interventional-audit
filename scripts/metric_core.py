from __future__ import annotations
import argparse, hashlib, json, os, random, re, struct, zlib
from pathlib import Path
from typing import Any

def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None

def parse_gold_answer(value: Any) -> int:
    """Normalize serialized evaluator answers without widening their range."""
    if isinstance(value, bool):
        raise ValueError('gold answer must be an integer or canonical integer string')
    if isinstance(value, int):
        answer = value
    elif isinstance(value, str) and re.fullmatch('[0-4]', value):
        answer = int(value)
    else:
        raise ValueError('gold answer must be an integer or canonical integer string')
    if not 0 <= answer <= 4:
        raise ValueError('gold answer outside accepted 0-4 task definition')
    return answer

def gate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row['candidate_parse_status'] == 'PARSEABLE']
    correct = [row for row in valid if row['candidate_correct']]
    incorrect = [row for row in valid if not row['candidate_correct']]
    admitted = [row for row in valid if row['parsed_verdict'] == 'YES']
    admitted_correct = [row for row in admitted if row['candidate_correct']]
    admitted_incorrect = [row for row in admitted if not row['candidate_correct']]
    rejected_correct = [row for row in correct if row['parsed_verdict'] != 'YES']
    unparsable = [row for row in valid if row['parse_status'] == 'UNPARSEABLE_REJECTION']
    return {'n_all': len(rows), 'n_candidate_parseable': len(valid), 'n_candidate_invalid': len(rows) - len(valid), 'n_candidate_correct_parseable': len(correct), 'n_candidate_incorrect_parseable': len(incorrect), 'n_admitted': len(admitted), 'n_admitted_correct': len(admitted_correct), 'n_admitted_incorrect': len(admitted_incorrect), 'n_unparseable_verdict': len(unparsable), 'n_rejected_correct': len(rejected_correct), 'candidate_parseable_rate': safe_ratio(len(valid), len(rows)), 'candidate_accuracy_all_items': safe_ratio(len(correct), len(rows)), 'candidate_accuracy_parseable': safe_ratio(len(correct), len(valid)), 'admission_precision_observed_coverage': safe_ratio(len(admitted_correct), len(admitted)), 'false_admission_rate_among_incorrect_candidates': safe_ratio(len(admitted_incorrect), len(incorrect)), 'false_rejection_rate_among_correct_candidates': safe_ratio(len(rejected_correct), len(correct)), 'gate_coverage_among_parseable_candidates': safe_ratio(len(admitted), len(valid)), 'effective_coverage_all_items': safe_ratio(len(admitted), len(rows)), 'unparseable_verdict_rate_among_parseable_candidates': safe_ratio(len(unparsable), len(valid)), 'admitted_correct_per_all_items': safe_ratio(len(admitted_correct), len(rows))}

def paired_flip_metrics(original: list[dict[str, Any]], comparison: list[dict[str, Any]]) -> dict[str, Any]:
    right = {row['sample_id']: row for row in comparison}
    if len(right) != len(comparison) or {row['sample_id'] for row in original} != set(right):
        raise ValueError('paired condition outputs do not cover exactly the same IDs')
    pairs = [(row, right[row['sample_id']]) for row in original if row['sample_id'] in right and row['candidate_parse_status'] == 'PARSEABLE']
    if not pairs:
        return {'n_paired_parseable_candidates': 0, 'verdict_change_rate': None, 'admission_change_rate': None}
    return {'n_paired_parseable_candidates': len(pairs), 'verdict_change_count': sum((left['parsed_verdict'] != right_row['parsed_verdict'] for left, right_row in pairs)), 'admission_change_count': sum(((left['parsed_verdict'] == 'YES') != (right_row['parsed_verdict'] == 'YES') for left, right_row in pairs)), 'verdict_change_rate': sum((left['parsed_verdict'] != right_row['parsed_verdict'] for left, right_row in pairs)) / len(pairs), 'admission_change_rate': sum(((left['parsed_verdict'] == 'YES') != (right_row['parsed_verdict'] == 'YES') for left, right_row in pairs)) / len(pairs)}
