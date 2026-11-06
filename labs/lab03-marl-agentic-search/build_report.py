"""Сборка отчёта по уже полученным артефактам; эксперимент здесь не запускается.

Запуск: bundled Python build_report.py. Перед созданием DOCX требуется marker
documents skill; после создания обязательны рендер и визуальная проверка.
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
OUT = ROOT / 'report_lab3.docx'
NAMES = {'reinforce': 'Independent REINFORCE', 'ia2c': 'Independent A2C',
         'ippo': 'IPPO', 'iql': 'Independent Q-learning', 'vdn': 'VDN',
         'qmix': 'QMIX', 'mappo': 'MAPPO', 'coma': 'COMA',
         'single_ppo': 'Одноагентный PPO', 'equal_allocation': 'Равное распределение'}
ORDER = list(NAMES)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'Пустой файл: {path}')
    return rows


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def plain_md(text):
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', text)
    return text.replace('**', '').replace('*', '').replace('`', '')


def parse_literature():
    text = (ROOT / 'literature.md').read_text(encoding='utf-8')
    rows = []
    for line in text.splitlines():
        if line.startswith('|') and re.match(r'^\|\s*\d+\s*\|', line):
            cells = [plain_md(c.strip()) for c in line.strip('|').split('|')]
            if len(cells) == 5:
                rows.append(cells)
    section = text.split('## Проверенные первоисточники', 1)[1]
    sources = [plain_md(line) for line in section.splitlines() if re.match(r'^\d+\.\s', line)]
    if not (10 <= len(rows) <= 15 and len(rows) == len(sources)):
        raise ValueError('Обзор должен содержать 10–15 сопоставленных источников')
    return rows, sources


def shade(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = tcpr.find(qn('w:shd'))
    if shd is None:
        shd = OxmlElement('w:shd'); tcpr.append(shd)
    shd.set(qn('w:fill'), fill)


def format_table(table, widths=None):
    table.autofit = False
    pr = table._tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        item = OxmlElement('w:' + edge)
        for key, val in [('val', 'single'), ('sz', '4'), ('color', 'D9D9D9')]:
            item.set(qn('w:' + key), val)
        borders.append(item)
    pr.append(borders)
    margins = OxmlElement('w:tblCellMar')
    for edge in ('top', 'bottom', 'left', 'right'):
        item = OxmlElement('w:' + edge); item.set(qn('w:w'), '85'); item.set(qn('w:type'), 'dxa'); margins.append(item)
    pr.append(margins)
    header = table.rows[0]._tr.get_or_add_trPr()
    repeat = OxmlElement('w:tblHeader'); repeat.set(qn('w:val'), 'true'); header.append(repeat)
    for r, row in enumerate(table.rows):
        row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
        for c, cell in enumerate(row.cells):
            if widths:
                cell.width = Cm(widths[c])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if r == 0:
                shade(cell, 'DCE8F1')
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.line_spacing = 1.0
                for run in p.runs:
                    run.font.name = 'Times New Roman'; run.font.size = Pt(9)
                    if r == 0:
                        run.bold = True


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    for i, value in enumerate(headers):
        table.rows[0].cells[i].text = str(value)
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = str(value)
    format_table(table, widths)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_p(doc, text, style=None):
    p = doc.add_paragraph(text, style=style)
    if text.startswith('Таблица '):
        p.paragraph_format.keep_with_next = True
    return p


def ci(values):
    vals = [float(v) for v in values]
    m = statistics.mean(vals)
    if len(vals) == 1:
        return f'{m:.3f}'
    # Для небольшого числа запусков используем двусторонние t-квантили 95%.
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(len(vals))
    if critical is None:
        raise ValueError('Добавьте точный t-квантиль для данного числа seed')
    half = critical * statistics.stdev(vals) / math.sqrt(len(vals))
    return f'{m:.3f} [{m-half:.3f}; {m+half:.3f}]'


def aggregate(metrics, case):
    groups = defaultdict(list)
    for row in metrics:
        if row['case'] == case:
            groups[row['method']].append(row)
    return groups


def main():
    required = [RESULTS / x for x in ('metrics.csv', 'validation.csv', 'learning.csv', 'predictions.csv', 'selection.json')]
    required += [ROOT / 'data' / 'manifest.json', ROOT / 'literature.md', ROOT / 'environment.py', ROOT / 'experiment.py']
    absent = [str(p) for p in required if not p.is_file()]
    if absent:
        raise SystemExit('Отчёт не создан: отсутствуют фактические артефакты:\n' + '\n'.join(absent))
    metrics, valid, learning, pred = (read_csv(RESULTS / x) for x in ('metrics.csv', 'validation.csv', 'learning.csv', 'predictions.csv'))
    selection, manifest = read_json(RESULTS / 'selection.json'), read_json(ROOT / 'data' / 'manifest.json')
    matrix, sources = parse_literature()
    winner, seeds = selection['winner'], selection['seeds']
    test, app = aggregate(metrics, 'test'), aggregate(metrics, 'application')
    if any(m not in test or m not in app for m in ORDER) or winner not in NAMES:
        raise SystemExit('Метрики неполны или выбран неизвестный метод; отчёт не создан')
    if any(len(test[m]) != len(seeds) for m in ORDER if m != 'equal_allocation'):
        raise SystemExit('Не все seed присутствуют в test; отчёт не создан')
    if not any(r['method'] == winner and r['case'] == 'test_stress' for r in metrics):
        raise SystemExit('Нет стресс-теста выбранного метода; отчёт не создан')

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Cm(2)
    sec.left_margin, sec.right_margin = Cm(2.4), Cm(1.7)
    for name in ('Normal', 'Title', 'Heading 1', 'Heading 2'):
        style = doc.styles[name]
        style.font.name = 'Times New Roman'
        style.font.color.rgb = RGBColor(0, 0, 0)
    normal = doc.styles['Normal']
    normal.font.size = Pt(12)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.15
    doc.styles['Title'].font.size = Pt(20)
    title_ppr = doc.styles['Title'].element.get_or_add_pPr()
    for border in title_ppr.findall(qn('w:pBdr')):
        title_ppr.remove(border)
    doc.styles['Heading 1'].font.size = Pt(15)
    doc.styles['Heading 2'].font.size = Pt(13)

    title = doc.add_paragraph(style='Title')
    title.add_run('Мультиагентное обучение для поиска документов')
    add_p(doc, 'Лабораторная работа № 3\nДисциплина «Прикладные задачи машинного обучения»\nГавриленко Никита, группа ПМ23-2')
    add_p(doc, 'Цель — сравнить кооперативные методы обучения с подкреплением для последовательного выбора поисковых действий и свидетельств при ограниченном числе обращений. В качестве данных использованы русские запросы и оценки релевантности MIRACL. Проверяется извлечение документов; генерация и оценка текстового ответа в эксперимент не входят.')
    add_p(doc, 'Тема связана с ВКР «Разработка языковой модели для многошагового взаимодействия с поисковыми инструментами агентной системы». Эксперимент является ограниченной моделью последовательного поиска, а не проверкой готовой языковой модели или способности к многошаговому ответу.')
    add_p(doc, 'По согласованию с преподавателем финансовая предметная область заменена поисковой: вместо OHLCV, активов и портфеля используются запросы, документы и бюджет обращений; вместо доходности и риска — релевантность, цена поиска и устойчивость. Равное распределение поисковых действий служит альтернативным контрольным методом, но не является равновзвешенным портфелем. Временной финансовый тест заменён отложенными запросами.')

    doc.add_heading('1 Исследовательский обзор', level=1)
    add_p(doc, 'Многоагентная кооперация требует согласования действий и распределения общей награды. При централизованном обучении с децентрализованным исполнением (CTDE) центральный критик или функция смешивания использует совместную информацию только при обучении; отдельная политика при исполнении опирается на собственное наблюдение и разрешённый обмен. Конкурентная постановка включает противоположные интересы и отличается от выбранной общей награды [1].')
    add_p(doc, 'VDN предполагает аддитивность совместной ценности, QMIX — монотонность, COMA использует контрфактический baseline, а MAPPO и IPPO различаются доступной критику информацией [2–5]. Исследования агентного поиска мотивируют выбор шагов и учёт стоимости [11–15], но не доказывают преимущество MARL в данной задаче.')
    add_p(doc, 'Таблица 1 — Сопоставление научных источников')
    compact = [(r[0], r[1], r[3], r[4]) for r in matrix]
    add_table(doc, ['№', 'Источник', 'Метод', 'Вывод и ограничение'], compact, [0.7, 3.5, 5.2, 7.0])

    doc.add_heading('2 Данные и экспериментальный протокол', level=1)
    counts = manifest['counts']
    add_p(doc, 'Используются запросы MIRACL-RU и полнотекстовый индекс полного русского корпуса из 9 543 918 фрагментов, подготовленный для ВКР. Индекс открыт только для чтения. Источники, контрольные суммы и параметры выдачи сохранены в data/manifest.json и data/retrieval_full/timing.json. Индекс строился по корпусу, а не по меткам релевантности отдельных тестовых запросов.')
    data_rows = [(s, counts[s]['queries'], counts[s]['judgments']) for s in ('train', 'validation', 'test', 'application')]
    add_p(doc, 'Таблица 2 — Паспорт разбиений')
    add_table(doc, ['Часть', 'Запросы', 'Оценки релевантности'], data_rows, [5, 4, 7.4])
    add_p(doc, 'Официальный train разделён по хешу нормализованного текста на обучение и validation; официальный dev разделён на test и application. Повторяющиеся тексты между исходными частями удаляются по правилу, заданному до анализа меток; точный алгоритм и контрольные суммы приведены в паспорте. Application — отдельная подвыборка dev, а не новый тематический набор или независимо собранный benchmark.')
    add_p(doc, 'Метки train используются в общей награде; validation — для выбора алгоритма. Метки test и application используются только для вычисления награды среды при оценке и метрик, но не входят в наблюдения политики или признаки поиска. Тестовые метрики не должны использоваться для последующей перенастройки и повторного выбора модели.')

    doc.add_heading('3 Многоагентная среда', level=1)
    add_p(doc, 'Два агента проходят чередующиеся фазы: выбирают одну из трёх заранее заданных поисковых формулировок или останавливаются, затем выбирают один из первых пяти документов соответствующей выдачи или останавливаются. После этого возможен второй цикл. При двух агентах верхняя граница — четыре поисковых обращения и четыре выбора документа; это разные типы действий. Функция разрешённых действий задаётся маской. Поиск и результаты заранее вычислены лексическим BM25 из текста запроса и корпуса.')
    add_p(doc, 'Локальное наблюдение включает фазу, номер агента, длину вопроса, число общих выбранных документов и поисков, собственное состояние, выбранную формулировку, нормированные оценки до пяти документов и признаки их повторного выбора. При включённом обмене добавляются сведения о последнем действии второго агента и совпадении формулировки. Содержание документов и семантическое представление текста вопроса в наблюдение не входят. Глобальное состояние критика или миксера — конкатенация наблюдений; политики при выполнении получают только локальные наблюдения.')
    add_p(doc, 'За найденный ранее не выбранный документ с положительной оценкой начисляется +1; каждое поисковое обращение имеет стоимость 0,02. Повторное свидетельство не даёт второй награды. Окончание определяется горизонтом, остановившийся агент впоследствии выполняет только остановку. Непомеченный документ в вычисляемой награде не получает положительного балла; это не доказывает его нерелевантность. Штраф за стоимость не является универсальной гарантией безопасности.')
    add_p(doc, 'Выдача всех трёх фиксированных формулировок готовится заранее. Реальные сетевые вызовы, языковая модель, синтез нового запроса по содержимому найденного документа и проверка ответа отсутствуют. Поэтому измеряется последовательная процедура отбора, а не доказанная способность к содержательному multi-hop QA. Стоимость полной подготовки выдачи следует учитывать отдельно от времени принятия решений по кэшу.')

    doc.add_heading('4 Алгоритмы и контрольные методы', level=1)
    add_p(doc, 'Сравниваются восемь различных учебных реализаций: Independent REINFORCE, Independent A2C, Independent PPO (IPPO), Independent Q-learning, VDN, QMIX, MAPPO и COMA. Набор отличается от иллюстративного набора методички; переименование методов не используется. Independent REINFORCE обновляет локальные политики по общей дисконтированной награде, Independent A2C добавляет локальный критик, IPPO применяет ограниченное обновление PPO. IQL обучает локальные Q-функции, VDN суммирует их, QMIX применяет монотонный миксер, MAPPO использует общий критик, COMA — централизованный критик с контрфактическим baseline.')
    add_p(doc, 'Совместный вход QMIX, MAPPO и COMA — конкатенация наблюдений, а не доказанно полное марковское состояние; VDN суммирует локальные ценности и использует общую награду без отдельного глобального входа. Это упрощённые реализации: COMA применяет одношаговую sampled SARSA-цель вместо TD(λ), VDN и QMIX обучаются без replay и рекуррентной истории, PPO использует полную Monte Carlo-награду без GAE и минибатчей. Равный бюджет эпизодов не означает одинакового числа обновлений: PPO делает четыре шага оптимизатора на эпизод.')
    add_p(doc, f"Использованы seed {', '.join(map(str, seeds))}. Все методы получают один проход по обучающим запросам и одинаковый бюджет среды; одноагентный PPO и фиксированное равное распределение действий используются как два контрольных метода. Компактные реализации не включают все инженерные детали авторских библиотек и не равнозначны их полному воспроизведению. Правило выбора зафиксировано по validation: {selection['rule']}.")
    add_p(doc, 'Таблица 3 — Выбор по validation')
    grouped_val = defaultdict(list)
    for r in valid:
        grouped_val[r['method']].append(r)
    val_rows = []
    for method in ORDER[:-1]:
        if method in grouped_val:
            v = grouped_val[method]
            val_rows.append([NAMES[method], ci([r['recall'] for r in v]), ci([r['searches'] for r in v]), f"{statistics.mean(float(r['selection_score']) for r in v):.3f}"])
    add_table(doc, ['Метод', 'Recall и 95% ДИ', 'Обращения и 95% ДИ', 'Балл выбора'], val_rows, [5.1, 4.3, 4.4, 2.6])
    add_p(doc, f"По правилу validation выбран {NAMES[winner]}. Выбор не означает автоматически превосходство над контрольными методами на отложенных данных.")

    doc.add_heading('5 Отложенная оценка и устойчивость', level=1)
    add_p(doc, 'Recall — доля известных положительных документов, попавших в свидетельства; Hit — доля запросов, где найден хотя бы один такой документ; precision — доля выбранных документов с положительной оценкой. Оценки неполны, поэтому precision относительно разметки не равна истинной полноте релевантности. Интервалы для обучаемых методов — двусторонние 95% t-интервалы по средним отдельных seed; при трёх seed они особенно нестабильны и не заменяют интервалы по запросам. Для детерминированного фиксированного контроля показано одно значение.')
    for caption, groups in [('Таблица 4 — Подвыборка test', test), ('Таблица 5 — Отдельная подвыборка application', app)]:
        add_p(doc, caption)
        tab = []
        for method in ORDER:
            rows = groups[method]
            tab.append([NAMES[method], ci([r['recall'] for r in rows]), ci([r['hit'] for r in rows]), ci([r['searches'] for r in rows])])
        add_table(doc, ['Метод', 'Recall [95% ДИ]', 'Hit [95% ДИ]', 'Вызовы [95% ДИ]'], tab, [5.1, 4.0, 3.8, 3.5])

    cases = [('test', 'обычный режим'), ('test_no_communication', 'без явного сообщения'), ('test_stress', 'стресс выдачи')]
    ablation = []
    for case, label in cases:
        vals = [r for r in metrics if r['method'] == winner and r['case'] == case]
        if vals:
            ablation.append([label, ci([r['recall'] for r in vals]), ci([r['searches'] for r in vals]), ci([r['duplicate_searches'] for r in vals])])
    add_p(doc, f'Таблица 6 — Проверки выбранного метода {NAMES[winner]}')
    add_table(doc, ['Режим', 'Recall [95% ДИ]', 'Вызовы [95% ДИ]', 'Дубли вызовов'], ablation, [5.0, 4.2, 3.8, 3.4])
    add_p(doc, 'Отключение сообщения удаляет явные признаки последнего действия и совпадения формулировки, однако общее состояние среды и признаки уже выбранных документов остаются доступными. Поэтому это частичная абляция связи, а не полное удаление координации. Стресс здесь задаётся удалением чередующихся позиций выдачи; это контролируемая порча выдачи, а не оценка реальной частоты сбоев. Оба воздействия применяются к уже обученной политике и создают сдвиг наблюдений.')

    for image in sorted(RESULTS.glob('*.png')):
        doc.add_picture(str(image), width=Cm(15))
        p = add_p(doc, image.stem.replace('_', ' ')); p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading('6 Ошибки, ограничения и вычислительные затраты', level=1)
    winner_test = [p for p in pred if p['method'] == winner and p['case'] == 'test']
    if winner_test:
        misses = sum(float(p['hit']) == 0 for p in winner_test)
        add_p(doc, f'В выбранном методе на test зафиксировано {misses} траекторий без положительно размеченного свидетельства из {len(winner_test)} траекторий по всем seed. Это число учитывает повтор одних и тех же запросов для разных seed; конкретные примеры и выбранные идентификаторы сохранены в results/predictions.csv.')
    test_qrels = defaultdict(dict)
    with (ROOT / 'data' / 'test_qrels.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line); test_qrels[item['qid']][item['docid']] = item['rel']
    with (ROOT / 'data' / 'retrieval_full' / 'test.jsonl').open(encoding='utf-8') as stream:
        retrieval_rows = [json.loads(line) for line in stream]
    coverage = sum(any(test_qrels[row['qid']].get(hit['id'], 0) > 0 for variant in row['variants'] for hit in variant['hits']) for row in retrieval_rows)
    add_p(doc, f'После фиксации модели установлено, что объединение трёх поисковых выдач содержит хотя бы один положительно размеченный документ для {coverage} из {len(retrieval_rows)} test-запросов. Это ограничивает максимально достижимый Hit относительно qrels на уровне {coverage / len(retrieval_rows):.3f}. Примеры ошибок и выбранных документов сохранены в results/error_analysis.md. Сокращение запросов до двух слов может терять важные различия. Неизвестная qrel сама по себе не означает нерелевантности.')
    add_p(doc, 'Риски включают неполноту разметки, переобучение на малой выборке, случайность обучения, ограничение заранее заданными вариантами запросов и чрезмерное доверие к метрикам отбора свидетельств. Неравные семантические трудности запросов и различия в распределении train/dev также ограничивают перенос. В примерах с отсутствующей положительной оценкой нельзя сделать вывод о содержательной ошибке без ручной проверки документа. Официальный dev ранее использовался в исследованиях ВКР; до фиксации поисковых правил здесь также были просмотрены агрегированные числа пустых выдач старой версии для test и application без оценок релевантности. Поэтому оценку следует считать отложенной для выбора политики внутри данного эксперимента, но не полностью независимой от предыдущего анализа.')
    time_rows = []
    for method in ORDER[:-1]:
        rs = grouped_val.get(method, [])
        if rs:
            time_rows.append([NAMES[method], f"{statistics.mean(float(x['training_seconds']) for x in rs):.2f}"])
    add_p(doc, 'Таблица 7 — Среднее время обучения и validation на seed, секунды')
    add_table(doc, ['Метод', 'Время, с'], time_rows, [10, 6.4])
    checks = sum(int(r['mask_checks']) for r in metrics)
    violations = sum(int(r['invalid_actions']) for r in metrics)
    add_p(doc, f'Время относится к обучению политики и её validation при уже подготовленной выдаче. Оно не включает построение корпуса, индекса и всех поисковых вариантов. В сохранённых оценочных эпизодах выполнено {checks} проверок маски, зарегистрировано {violations} недопустимых действий. При обнаружении недопустимого действия реализация прекращает выполнение с ошибкой; результат относится к завершённым эпизодам, а не является гарантией для новых условий.')

    doc.add_heading('7 Вывод для ВКР', level=1)
    wr, br = test[winner], test['single_ppo']
    wm = statistics.mean(float(r['recall']) for r in wr)
    bm = statistics.mean(float(r['recall']) for r in br)
    em = statistics.mean(float(r['recall']) for r in test['equal_allocation'])
    add_p(doc, f'В проведённом эксперименте validation выбрал {NAMES[winner]}; средний recall на test составил {wm:.3f}, у одноагентного PPO — {bm:.3f}, у фиксированного равного распределения — {em:.3f}. Фиксированная стратегия нашла больше размеченных документов при большем числе поисковых вызовов. Интервалы и издержки приведены в таблицах; по трём seed нельзя утверждать надёжное причинное преимущество кооперации.')
    add_p(doc, 'Для ВКР полезны явный бюджет инструмента, разделение информации при обучении и исполнении, а также проверка стоимости и сбоев. Для следующего исследования нужны новый независимый набор запросов, содержательные наблюдения, формирование последующих запросов по результатам, языковая модель ответа и разметка многошагового обоснования. Текущий эксперимент не устанавливает качество такой системы.')

    doc.add_heading('Список источников', level=1)
    for source in sources:
        add_p(doc, source)
    doc.save(OUT)
    print(f'Создан {OUT}; обязательны рендер и визуальная проверка всех страниц.')


if __name__ == '__main__':
    main()
