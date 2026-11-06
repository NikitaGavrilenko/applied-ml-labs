"""Run the reproducible experiment using local MIRACL data and the full FTS index."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script, *args):
    cmd = [sys.executable, str(ROOT / script), *args]
    print('Run:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh-data', action='store_true', help='recreate query splits and search cache')
    parser.add_argument('--skip-report', action='store_true', help='skip Word report generation')
    args = parser.parse_args()
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    retrieval = ROOT / 'data' / 'retrieval_full'
    expected = [retrieval / f'{x}.jsonl' for x in ('train', 'validation', 'test', 'application')]
    if args.refresh_data or not all(p.is_file() for p in [*expected, retrieval / 'timing.json']):
        run('data_pipeline.py', '--precompute')
    meta = json.loads((retrieval / 'timing.json').read_text(encoding='utf-8'))
    if not meta.get('full_corpus') or meta.get('index_passages') != 9543918:
        raise RuntimeError('Кэш не подтверждает использование полного индекса MIRACL-RU')
    for path in expected:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != meta['splits'][path.stem]['sha256']:
            raise RuntimeError(f'Контрольная сумма выдачи не совпадает: {path}')
    run('test_environment.py')
    run('experiment.py')
    run('analysis_results.py')
    if not args.skip_report:
        runtime = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies'
        node = runtime / 'node/bin/node.exe'
        doc_python = runtime / 'python/python.exe'
        marker = Path.home() / '.codex/plugins/cache/openai-primary-runtime/documents/26.1026.10807/skills/documents/container_tools/mark_artifact_operation_started.mjs'
        if not all(p.is_file() for p in (node, doc_python, marker)):
            raise FileNotFoundError('Результаты готовы, но для сборки Word необходим bundled runtime Codex; см. README')
        subprocess.run([str(node), str(marker), '--operation-kind', 'create', '--expected-output-count', '1', '--output-format', 'docx'], check=True, cwd=ROOT)
        subprocess.run([str(doc_python), str(ROOT / 'build_report.py')], check=True, cwd=ROOT)
    print('Расчёты завершены; для нового Word требуется визуальная проверка страниц.', flush=True)


if __name__ == '__main__':
    main()
