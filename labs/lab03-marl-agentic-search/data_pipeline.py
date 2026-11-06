"""MIRACL-RU preparation and lexical search; qrels are isolated from observations."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAB01 = ROOT.parent / 'lab01-llm-miracl' / 'data'
SPLITS = ('train', 'validation', 'test', 'application')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def prepare(source=LAB01, target=ROOT / 'data', seed='42'):
    """Split topics by query hash; copy corpus; write qrels separately, with no label filtering."""
    source, target = Path(source), Path(target)
    raw, prepared = source / 'raw', source / 'prepared'
    source_files = [prepared / 'corpus.jsonl'] + [raw / f'{kind}.miracl-v1.0-ru-{split}.tsv' for split in ('train', 'dev') for kind in ('topics', 'qrels')]
    if not all(p.is_file() for p in source_files):
        raise FileNotFoundError('Не найдены корпус или исходные topics/qrels lab01: ' + str(source))
    target.mkdir(parents=True, exist_ok=True)
    inputs = {p.name: sha256(p) for p in source_files}
    seen, outputs, counts, cross_split_texts = set(), {}, {}, {}
    corpus_ids = set()
    for doc in jsonl(source_files[0]):
        if set(doc) != {'id', 'title', 'text'} or doc['id'] in corpus_ids:
            raise ValueError('Некорректная запись или повтор docid в корпусе')
        corpus_ids.add(doc['id'])
    all_topics, all_qrels = {}, {}
    for source_split in ('train', 'dev'):
        topics, qrels = {}, {}
        with (raw / f'topics.miracl-v1.0-ru-{source_split}.tsv').open(encoding='utf-8') as f:
            for line in f:
                qid, query = line.rstrip('\n\r').split('\t', 1)
                if qid in topics:
                    raise ValueError(f'Повтор qid {source_split}:{qid}')
                topics[qid] = query
        with (raw / f'qrels.miracl-v1.0-ru-{source_split}.tsv').open(encoding='utf-8') as f:
            for line in f:
                qid, _, docid, rel = line.split()
                if (qid, docid) in qrels:
                    raise ValueError(f'Повтор qrel {source_split}:{qid}/{docid}')
                qrels[qid, docid] = int(rel)
        orphan_qids = {q for q, _ in qrels} - topics.keys()
        if orphan_qids:
            raise ValueError(f'qrels с неизвестными qid: {len(orphan_qids)}')
        all_topics[source_split], all_qrels[source_split] = topics, qrels
    # Duplicate text across official splits: retain dev only, without consulting qrels.
    normalize = lambda q: ' '.join(q.casefold().split())
    dev_texts = {normalize(q) for q in all_topics['dev'].values()}
    dropped = [{'qid': 'train:' + qid, 'query': query, 'reason': 'normalized_query_present_in_official_dev'}
               for qid, query in all_topics['train'].items() if normalize(query) in dev_texts]
    for row in dropped:
        del all_topics['train'][row['qid'].split(':', 1)[1]]
    (target / 'dropped_duplicates.json').write_text(json.dumps(dropped, ensure_ascii=False, indent=2), encoding='utf-8')
    limits = {'train': 400, 'validation': 100, 'test': 150, 'application': 100}
    selected = {name: [] for name in SPLITS}
    for official_split in ('train', 'dev'):
        for qid, query in all_topics[official_split].items():
            key = normalize(query)
            bucket = int(hashlib.sha256((seed + '\0' + key).encode('utf-8')).hexdigest()[:16], 16) % 100
            split = ('train' if bucket < 80 else 'validation') if official_split == 'train' else ('test' if bucket < 80 else 'application')
            rank = hashlib.sha256((seed + '\0select\0' + official_split + ':' + qid).encode('utf-8')).hexdigest()
            selected[split].append((rank, official_split, qid, query))
    for split in SPLITS:
        selected[split] = sorted(selected[split])[:limits[split]]
        if len(selected[split]) != limits[split]:
            raise ValueError(f'Недостаточно запросов для {split}')
    handles = {}
    try:
        for split in SPLITS:
            handles[split] = ((target / f'{split}.jsonl').open('w', encoding='utf-8', newline='\n'),
                              (target / f'{split}_qrels.jsonl').open('w', encoding='utf-8', newline='\n'))
            counts[split] = {'queries': 0, 'judgments': 0, 'missing_doc_judgments': 0, 'positive_missing_docs': 0}
        destinations = {}
        for split in SPLITS:
            for _, official_split, qid, query in selected[split]:
                key = normalize(query)
                destinations[official_split, qid] = split
                cross_split_texts.setdefault(key, set()).add(split)
                row = {'qid': official_split + ':' + qid, 'query': query, 'official_split': official_split}
                handles[split][0].write(json.dumps(row, ensure_ascii=False) + '\n')
                counts[split]['queries'] += 1
        for official_split in ('train', 'dev'):
            for (qid, docid), rel in sorted(all_qrels[official_split].items()):
                if (official_split, qid) not in destinations:
                    continue
                split = destinations[official_split, qid]
                row = {'qid': official_split + ':' + qid, 'docid': docid, 'rel': rel}
                handles[split][1].write(json.dumps(row, ensure_ascii=False) + '\n')
                counts[split]['judgments'] += 1
                if docid not in corpus_ids:
                    counts[split]['missing_doc_judgments'] += 1
                    counts[split]['positive_missing_docs'] += int(rel > 0)
    finally:
        for pair in handles.values():
            for stream in pair:
                stream.close()
    for path in sorted(target.glob('*.jsonl')):
        outputs[path.name] = sha256(path)
    manifest = {'version': 'miracl-ru-full-v1', 'seed': seed, 'split_rule': 'Remove normalized query overlaps across official splits giving dev priority. SHA256(seed + NUL + normalized query) mod 100, 0..79 first group, 80..99 second; official train -> train/validation, official dev -> test/application. Within each group select lowest SHA256(seed + NUL + select + NUL + official_split:qid).', 'dropped_cross_source_duplicates': len(dropped), 'input_sha256': inputs, 'output_sha256': outputs, 'coverage_reference_docs': len(corpus_ids), 'counts': counts, 'normalized_query_texts_in_multiple_splits': sum(len(v) > 1 for v in cross_split_texts.values()), 'coverage_reference_only': 'lab01 judged-pool corpus; never used for retrieval', 'qrels_policy': 'kept separate, never included in queries, corpus or search index'}
    (target / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


FULL_INDEX = Path(r'D:\Projects\University_Current\vkr-agentic-search\.local\full-miracl-ru-v1\index\miracl_ru_fts.sqlite')

class BM25:
    """Read-only SQLite FTS5/BM25 against the full 9,543,918-passage index."""
    def __init__(self, index_path=FULL_INDEX):
        import sqlite3
        self.connection = sqlite3.connect(Path(index_path).resolve().as_uri() + '?mode=ro', uri=True)
        shards = list(self.connection.execute('SELECT number, documents FROM completed_shards ORDER BY number'))
        if [row[0] for row in shards] != list(range(20)) or sum(row[1] for row in shards) != 9543918:
            raise ValueError('Индекс неполон: ожидается 20 частей и 9543918 документов')

    def search(self, query, k=5, operator='AND'):
        import sqlite3
        if k < 1:
            raise ValueError('k должно быть положительным')
        started = time.perf_counter()
        tokens = re.findall(r'\w+', query, flags=re.UNICODE)[:32]
        if not tokens:
            return {'hits': [], 'latency_s': time.perf_counter() - started, 'status': 'empty_query'}
        if operator not in ('AND', 'OR'):
            raise ValueError('Недопустимый оператор')
        expression = f' {operator} '.join('"' + token + '"' for token in tokens)
        self.connection.set_progress_handler(lambda: int(time.perf_counter() - started > 3), 5000)
        status = 'ok'
        try:
            rows = self.connection.execute('SELECT docid,title,text,bm25(passages) AS score FROM passages WHERE passages MATCH ? ORDER BY score, docid LIMIT ?', (expression, k)).fetchall()
        except sqlite3.OperationalError as exc:
            if 'interrupted' not in str(exc).lower():
                raise
            rows, status = [], 'timeout'
        finally:
            self.connection.set_progress_handler(None, 0)
        return {'hits': [{'id': str(docid), 'title': title, 'text': text, 'score': float(-score)} for docid, title, text, score in rows], 'latency_s': time.perf_counter() - started, 'status': status}


def precompute(target=ROOT / 'data', k=5, index_path=FULL_INDEX):
    """Materialize three question-only queries and BM25 results; never open qrels."""
    target = Path(target)
    start = time.perf_counter()
    index = BM25(index_path)
    timing = {'full_corpus': True, 'index_open_s': time.perf_counter() - start, 'index_path': str(index_path), 'index_sha256': sha256(index_path), 'index_passages': 9543918, 'top_k': k, 'query_timeout_s': 3, 'fts': 'unicode61 remove_diacritics 2; SQLite FTS5 bm25 default weights; all variants AND', 'selection_caveat': 'Before the final search rules were frozen, aggregate empty-result counts for an earlier AND-only version were inspected on test and application. No relevance judgments or test quality metrics were used. Therefore the search protocol is not strictly blinded to these aggregate diagnostics.', 'splits': {}}
    stopwords = set('когда кто что как где почему какой какая какое какие сколько был была были было это этот эта эти году год года который которая которых для или при из по на в с и а не о об у к от до за во со же ли чем его ее их каком таком такое'.split())
    directory = target / 'retrieval_full'
    directory.mkdir(exist_ok=True)
    for split in SPLITS:
        split_start = time.perf_counter()
        with (directory / f'{split}.jsonl').open('w', encoding='utf-8', newline='\n') as out:
            for number, row in enumerate(jsonl(target / f'{split}.jsonl'), 1):
                words = re.findall(r'[a-zа-яё0-9]{2,}', row['query'].casefold())
                content = [word for word in words if word not in stopwords]
                variants = [(' '.join(content[:2]) or row['query'], 'AND'), (' '.join(content[-2:]) or row['query'], 'AND'), (' '.join(sorted(content, key=lambda word: -len(word))[:2]) or row['query'], 'AND')]
                results = []
                for query, operator in variants:
                    result = index.search(query, k=k, operator=operator)
                    results.append({'query': query, 'operator': operator, **result})
                out.write(json.dumps({'qid': row['qid'], 'query': row['query'], 'variants': results}, ensure_ascii=False) + '\n')
                if number % 25 == 0:
                    out.flush()
                    print(f'{split}: {number}', flush=True)
        print(f'{split}: ready', flush=True)
        timing['splits'][split] = {'seconds': time.perf_counter() - split_start, 'sha256': sha256(directory / f'{split}.jsonl')}
    (directory / 'timing.json').write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=LAB01)
    parser.add_argument('--target', type=Path, default=ROOT / 'data')
    parser.add_argument('--seed', default='42')
    parser.add_argument('--precompute', action='store_true')
    parser.add_argument('--index', type=Path, default=FULL_INDEX)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.target, args.seed), ensure_ascii=False, indent=2))
    if args.precompute:
        precompute(args.target, index_path=args.index)
