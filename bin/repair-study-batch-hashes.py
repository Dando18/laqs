#!/usr/bin/env python3
"""One-time audited repair for the post-measurement study batch-hash bug.

Pass platform directories (ROOT/tuolumne or ROOT/matrix). Defaults to a dry run.
Only the exact hash-timing source patch is allowed; kernel and timing changes
cannot be migrated. Original metadata is backed up before any replacement.
"""
from copy import deepcopy
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'triton/packet_layout'))
import _bootstrap
from debug_suite import OUTPUTS, stage_complete
from experiment_support import digest
from layout_contract import RuntimeLayout
from manual_layout import split_tile
from packet_workflow import identity, write_json


def source_transition():
    current = identity()
    name = 'triton/packet_layout/layout_study.py'
    fixed = (ROOT/name).read_text()
    insertion = '    batch_hash = digest(batch)\n'
    assignment = "    record['batch_hash'] = batch_hash\n"
    if fixed.count(insertion) != 1 or fixed.count(assignment) != 1:
        raise ValueError('repair requires the exact reviewed hash-timing patch')
    original = fixed.replace(insertion, '').replace(assignment, "    record['batch_hash'] = digest(batch)\n")
    previous = deepcopy(current)
    previous['sources'][name] = hashlib.sha256(original.encode()).hexdigest()
    previous['source_hash'] = digest(previous['sources'])
    return previous, current


def signed(value, field):
    if value[field] != digest({k: v for k, v in value.items() if k != field}):
        raise ValueError(f'{field} integrity check failed')


def plan(directory):
    previous, current = source_transition()
    originals, changes = {}, {}
    def read(relative):
        relative = Path(relative)
        raw = (directory/relative).read_text()
        originals[relative] = raw
        return json.loads(raw)
    config = read('debug-suite.json')
    if config['source'] != previous or config['stages'] != ['capture', 'study_search', 'study']:
        raise ValueError('source or workflow differs from the exact affected study')
    old_config_hash = digest(config)
    config['source'] = current
    changes[Path('debug-suite.json')] = config
    count = 0
    for case in config['cases']:
        relative = Path(case)/'expert'
        case_dir = directory/relative
        if (case_dir/'study.json').exists() or (case_dir/'study-partial.json').exists():
            raise ValueError('repair is restricted to studies without final/partial reports')
        for stage in ('capture', 'study_search'):
            if not stage_complete(case_dir, stage, old_config_hash):
                raise ValueError(f'{case}: {stage} not complete')
        capture = read(relative/'capture.json')
        search = read(relative/'search.json')
        signed(search, 'selection_hash')
        if capture['source_identity'] != previous or search['source_identity'] != previous:
            raise ValueError('capture/search source mismatch')
        old_selection = search['selection_hash']
        capture['source_identity'] = search['source_identity'] = current
        search['selection_hash'] = digest({k: v for k, v in search.items() if k != 'selection_hash'})
        changes[relative/'capture.json'] = capture
        changes[relative/'search.json'] = search
        old_choice = new_choice = None
        if (case_dir/'study-choice.json').exists():
            choice = read(relative/'study-choice.json')
            signed(choice, 'choice_hash')
            if choice['selection_hash'] != old_selection:
                raise ValueError('choice belongs to another selection')
            old_choice = choice['choice_hash']
            choice['selection_hash'] = search['selection_hash']
            new_choice = choice['choice_hash'] = digest({k: v for k, v in choice.items() if k != 'choice_hash'})
            changes[relative/'study-choice.json'] = choice
        for path in sorted(case_dir.glob('study-*/batch-*/batch.json')):
            batch_path = path.relative_to(directory)
            batch = read(batch_path)
            if batch['selection_hash'] != old_selection or batch['choice_hash'] not in (None, old_choice):
                raise ValueError('batch selection/choice mismatch')
            annotated = deepcopy(batch)
            for storage in annotated['storages']:
                storage['tiles'] = {str(item['argument']): list(split_tile(RuntimeLayout.from_dict(item)))
                                    for item in storage['runtime_layouts']}
            updated = deepcopy(batch)
            updated['selection_hash'] = search['selection_hash']
            if updated['choice_hash'] is not None:
                updated['choice_hash'] = new_choice
            changes[batch_path] = updated
            for worker_path in sorted(path.parent.glob('process-*.json')):
                worker_relative = worker_path.relative_to(directory)
                worker = read(worker_relative)
                if 'record_hash' not in worker:
                    continue  # An interrupted, unsigned measurement must run again.
                signed(worker, 'record_hash')
                if (worker['batch_hash'] != digest(annotated) or worker['storages'] != annotated['storages']
                        or worker['selection_hash'] != old_selection):
                    raise ValueError('worker differs by more than deterministic tile diagnostics')
                worker['selection_hash'] = search['selection_hash']
                worker['batch_hash'] = digest(updated)
                worker['record_hash'] = digest({k: v for k, v in worker.items() if k != 'record_hash'})
                changes[worker_relative] = worker
                count += 1
        # Rebind completed stage receipts without rerunning capture or scoring.
        previous_receipt = None
        for stage in ('capture', 'study_search'):
            receipt_path = relative/'.debug-stages'/f'{stage}.json'
            read(receipt_path)
            receipt = {'binding': digest({'config': digest(config), 'stage': stage,
                                          'previous': previous_receipt}),
                       'outputs': {name: hashlib.sha256(encode(changes[relative/name]).encode()).hexdigest()
                                   for name in OUTPUTS[stage]}}
            changes[receipt_path] = receipt
            previous_receipt = hashlib.sha256(encode(receipt).encode()).hexdigest()
        read(relative/'status.json')
        changes[relative/'status.json'] = {'stage': 'study', 'status': 'paused',
                                          'reason': 'batch-hash bug repaired; completed timings preserved'}
    state = read('debug-state.json')
    for record in state['cases'].values():
        record.update(status='paused', reason='batch-hash bug repaired; resume the same command')
    changes[Path('debug-state.json')] = state
    return originals, changes, count


def encode(value):
    return json.dumps(value, indent=2, sort_keys=True)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    for directory in args.directories:
        with (directory/'debug-suite.lock').open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            originals, changes, count = plan(directory)
            print(f'{directory}: verified {count} completed workers, {len(changes)} metadata replacements')
            if not args.apply:
                continue
            backup = directory/'checkpoint-hash-repair'
            if backup.exists():
                raise ValueError(f'{backup} exists; inspect the prior repair before proceeding')
            for path, raw in originals.items():
                target = backup/'original'/path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(raw)
            write_json(backup/'audit.json', {'reason': 'hash immutable batch before measurement annotates storages',
                'completed_workers_preserved': count,
                'before': {str(p): hashlib.sha256(raw.encode()).hexdigest() for p, raw in originals.items()},
                'after': {str(p): hashlib.sha256(encode(value).encode()).hexdigest() for p, value in changes.items()}})
            for path, value in changes.items():
                write_json(directory/path, value)
            print('Repaired. Rerun the original study command; no fresh root or retry flag needed.')


if __name__ == '__main__':
    main()
