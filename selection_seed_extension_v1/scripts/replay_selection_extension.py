#!/usr/bin/env python3
"""Validate the recorded five-seed selection-extension package."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    root=ap.parse_args().root
    tasks=json.loads((root/'configs/tasks.json').read_text())['tasks']
    assert len(tasks)==110 and len({t['task_index'] for t in tasks})==110
    labels={}
    for fam in ('shape_count','color_count'):
        rows=read_jsonl(root/f'data/labels/{fam}.jsonl')
        assert len(rows)==288 and len({r['sample_id'] for r in rows})==288
        labels[fam]={r['sample_id']:r for r in rows}
    results=read_jsonl(root/'data/results.jsonl')
    assert len(results)==110 and {r['task_index'] for r in results}==set(range(110))
    for t in tasks:
        idx=t['task_index']; fam=t['task_family']
        rows=read_jsonl(root/f'data/tasks/task_{idx:03d}/predictions.jsonl')
        assert len(rows)==288 and len({r['sample_id'] for r in rows})==288
        assert {r['sample_id'] for r in rows}==set(labels[fam])
        correct=0
        for r in rows:
            lab=labels[fam][r['sample_id']]['answer']
            try: pred=int(r['prediction'])
            except (TypeError,ValueError): pred=None
            correct += int(pred is not None and str(pred)==str(lab))
        got=next(x for x in results if x['task_index']==idx)
        assert got['prediction_count']==288
        assert math.isclose(got['accuracy'], correct/288, rel_tol=0, abs_tol=1e-12), (idx,got['accuracy'],correct/288)
        assert got['task_family']==fam and got['arm']==t['arm']
    print('PASS: 110 tasks, 31,680 predictions, two 288-row label files, and recorded accuracies validated.')

if __name__=='__main__': main()
