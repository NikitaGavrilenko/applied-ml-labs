"""Train and evaluate discrete MARL policies on fixed MIRACL search results."""
from __future__ import annotations
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from algorithms import MARL
from environment import SearchEnvironment

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
METHODS = ['reinforce', 'ia2c', 'ippo', 'iql', 'vdn', 'qmix', 'mappo', 'coma', 'single_ppo']
SEEDS = [11, 23, 37]


def read_jsonl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def load(split):
    rows = read_jsonl(DATA / 'retrieval_full' / f'{split}.jsonl')
    labels = defaultdict(dict)
    for q in read_jsonl(DATA / f'{split}_qrels.jsonl'):
        labels[q['qid']][q['docid']] = q['rel']
    return rows, labels


def episode(model, row, qrels, training=False, communication=True, stress=False, baseline=False):
    n = 1 if model is not None and model.n_agents == 1 else 2
    env = SearchEnvironment(row, qrels, n_agents=n, communication=communication, stress=stress)
    batch = {key: [] for key in ('obs', 'mask', 'actions', 'rewards', 'next_obs', 'next_mask', 'dones')}
    rewards = 0.0
    while True:
        obs, mask = env.observe()
        if baseline:
            # Equal allocation: alternate fixed search variants; select the highest available rank.
            actions = np.array([1 + ((env.t // 2 * n + i) % 3) if env.t % 2 == 0 else 1 for i in range(n)])
            actions = np.array([a if mask[i, a] else 0 for i, a in enumerate(actions)])
        else:
            actions = model.act(obs, mask, explore=training)
        reward, done = env.step(actions)
        rewards += reward
        nxt, nm = env.observe() if not done else (np.zeros_like(obs), np.zeros_like(mask))
        for key, value in zip(batch, (obs, mask, actions, reward, nxt, nm, done)):
            batch[key].append(value)
        if done:
            break
    if training:
        model.update(batch)
    return {**env.metrics(), 'reward': rewards, 'evidence_ids': '|'.join(env.selected)}


def evaluate(model, rows, labels, **kwargs):
    out = []
    for row in rows:
        out.append({'qid': row['qid'], **episode(model, row, labels.get(row['qid'], {}), **kwargs)})
    return out


def average(rows, key):
    return float(np.mean([r[key] for r in rows]))


def main():
    torch.set_num_threads(1)
    RESULTS.mkdir(exist_ok=True)
    splits = {s: load(s) for s in ('train', 'validation', 'test', 'application')}
    history, metrics, predictions, validation = [], [], [], []
    trained = {}
    for method in METHODS:
        for seed in SEEDS:
            started = time.perf_counter()
            n = 1 if method == 'single_ppo' else 2
            model = MARL('ippo' if n == 1 else method, n, SearchEnvironment.OBS, SearchEnvironment.ACTIONS, seed, hidden=32)
            rng = random.Random(seed)
            rows, labels = splits['train']
            order = list(rows); rng.shuffle(order)
            for index, row in enumerate(order, 1):
                result = episode(model, row, labels.get(row['qid'], {}), training=True)
                history.append({'method': method, 'seed': seed, 'episode': index, 'reward': result['reward']})
            val = evaluate(model, *splits['validation'])
            score = average(val, 'recall') - .02 * average(val, 'searches')
            elapsed = time.perf_counter() - started
            validation.append({'method': method, 'seed': seed, 'recall': average(val, 'recall'), 'searches': average(val, 'searches'), 'selection_score': score, 'training_seconds': elapsed})
            trained[method, seed] = model
            torch.save(model, RESULTS / f'{method}_{seed}.pt')
            write_csv(RESULTS / 'learning.csv', history)
            write_csv(RESULTS / 'validation.csv', validation)
            print(f'{method} seed={seed} val={score:.4f} elapsed={elapsed:.1f}s', flush=True)
    candidates = METHODS[:-1]
    winner = max(candidates, key=lambda m: np.mean([r['selection_score'] for r in validation if r['method'] == m]))
    (RESULTS / 'selection.json').write_text(json.dumps({'rule': 'mean validation recall - 0.02 * mean search calls, across seeds; only MARL methods', 'winner': winner, 'seeds': SEEDS}, ensure_ascii=False, indent=2), encoding='utf-8')
    for method in METHODS + ['equal_allocation']:
        seeds = [0] if method == 'equal_allocation' else SEEDS
        for seed in seeds:
            model = None if method == 'equal_allocation' else trained[method, seed]
            cases = [('test', True, False), ('application', True, False)]
            if method == winner:
                cases += [('test_no_communication', False, False), ('test_stress', True, True)]
            for case, comm, stress in cases:
                split = 'test' if case.startswith('test') else case
                started = time.perf_counter()
                values = evaluate(model, *splits[split], communication=comm, stress=stress, baseline=model is None)
                elapsed = time.perf_counter() - started
                for row in values:
                    predictions.append({'method': method, 'seed': seed, 'case': case, **row})
                metrics.append({'method': method, 'seed': seed, 'case': case, **{k: average(values, k) for k in ('recall','hit','precision','searches','selected','duplicate_searches','reward')}, 'mask_checks': sum(v['mask_checks'] for v in values), 'invalid_actions': sum(v['invalid_actions'] for v in values), 'evaluation_seconds': elapsed, 'queries': len(values)})
    write_csv(RESULTS / 'learning.csv', history)
    write_csv(RESULTS / 'validation.csv', validation)
    write_csv(RESULTS / 'metrics.csv', metrics)
    write_csv(RESULTS / 'predictions.csv', predictions)
    print('Selected on validation:', winner, flush=True)


if __name__ == '__main__':
    main()
