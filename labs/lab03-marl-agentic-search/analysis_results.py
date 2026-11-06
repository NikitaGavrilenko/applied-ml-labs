"""Summarise completed runs without selecting models on the test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parent
METRICS = ("recall", "hit", "precision", "searches", "selected",
           "duplicate_searches", "reward", "evaluation_seconds")


def read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = required.difference(frame.columns)
    if frame.empty or missing:
        raise ValueError(f"{path}: empty or missing columns: {sorted(missing)}")
    return frame


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Student t intervals over independent seed-level means, never query-level samples."""
    records = []
    for (case, method), group in frame.groupby(["case", "method"], sort=True):
        if group["seed"].duplicated().any():
            raise ValueError(f"Duplicate seed in {case}/{method}")
        for metric in METRICS:
            x = pd.to_numeric(group[metric], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(x).all():
                raise ValueError(f"Non-finite {metric} in {case}/{method}")
            n = len(x)
            avg = float(np.mean(x))
            sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
            half = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
            records.append({"case": case, "method": method, "metric": metric,
                            "seed_count": n, "seeds": "|".join(str(s) for s in sorted(group["seed"])),
                            "queries_min": int(group["queries"].min()),
                            "queries_max": int(group["queries"].max()),
                            "mean": avg, "seed_sd": sd,
                            "ci95_low": avg - half, "ci95_high": avg + half})
    return pd.DataFrame(records)


def plot_learning(frame: pd.DataFrame, path: Path) -> None:
    methods = sorted(frame["method"].unique())
    cols = min(3, len(methods))
    rows = (len(methods) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.6 * rows), squeeze=False)
    for ax, method in zip(axes.flat, methods):
        sub = frame[frame.method == method]
        for seed, run in sub.groupby("seed", sort=True):
            run = run.sort_values("episode")
            # This is a descriptive running mean, not validation or test quality.
            smooth = run.reward.rolling(window=25, min_periods=1).mean()
            ax.plot(run.episode, smooth, alpha=.75, linewidth=1, label=str(seed))
        ax.set_title(method)
        ax.set_xlabel("Эпизод обучения")
        ax.set_ylabel("Награда, среднее за 25")
        ax.grid(alpha=.2)
    for ax in axes.flat[len(methods):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="seed", loc="upper right")
    fig.suptitle("Обучение по seed")
    fig.tight_layout(rect=(0, 0, .96, .97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_test(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, title in zip(axes, ("recall", "searches"),
                                 ("Recall по размеченным документам", "Число поисковых вызовов")):
        s = summary[(summary.case == "test") & (summary.metric == metric)].sort_values("mean")
        y = np.arange(len(s))
        for i, (_, r) in enumerate(s.iterrows()):
            # Baseline with one deterministic run has no seed CI.
            if r.seed_count > 1:
                ax.errorbar(r["mean"], i, xerr=[[r["mean"] - r.ci95_low], [r.ci95_high - r["mean"]]],
                            fmt="o", capsize=3, color="#285b89", markersize=4)
            else:
                ax.plot(r["mean"], i, "s", color="#9a592c", markersize=5)
        ax.set_yticks(y, s.method)
        ax.set_title(title)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Отложенный test: среднее по seed и 95% t-интервал")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def error_analysis(predictions: pd.DataFrame, winner: str, output: Path) -> tuple[int, int]:
    test = predictions[(predictions.case == "test") & (predictions.method == winner)].copy()
    if test.empty:
        raise ValueError("Selected method has no test predictions")
    if test.duplicated(["seed", "qid"]).any():
        raise ValueError("Duplicate test prediction for selected method and seed")
    failures = test[test.hit == 0].copy()
    failures["error_type"] = np.where(failures.selected == 0, "no_evidence_selected", "no_judged_relevant_evidence")
    columns = ["qid", "seed", "error_type", "recall", "hit", "precision", "searches",
               "selected", "duplicate_searches", "reward", "evidence_ids"]
    failures.sort_values(["qid", "seed"])[columns].to_csv(output, index=False, encoding="utf-8-sig")
    return len(failures), len(test)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    out = parser.parse_args().results_dir
    metrics = read_csv(out / "metrics.csv", {"case", "method", "seed", "queries", *METRICS})
    learning = read_csv(out / "learning.csv", {"method", "seed", "episode", "reward"})
    val = read_csv(out / "validation.csv", {"method", "seed", "recall", "searches", "selection_score"})
    predictions = read_csv(out / "predictions.csv", {"method", "seed", "case", "qid", "hit", "recall", "precision", "searches", "selected", "duplicate_searches", "reward", "evidence_ids"})
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    winner = selection["winner"]
    if winner not in set(val.method) or "test" not in set(metrics.case):
        raise ValueError("Selection is inconsistent with validation or test results are missing")
    summary = summarise(metrics)
    summary.to_csv(out / "summary_ci.csv", index=False, encoding="utf-8-sig")
    plot_learning(learning, out / "learning_curves.png")
    plot_test(summary, out / "test_comparison.png")
    failures, total = error_analysis(predictions, winner, out / "error_cases.csv")
    counts = predictions[(predictions.case == "test") & (predictions.method == winner)]
    zero = int(((counts.hit == 0) & (counts.selected == 0)).sum())
    (out / "analysis_notes.md").write_text(
        "# Анализ результатов\n\n"
        f"Выбранный до теста по validation метод: **{winner}**. Правило: {selection['rule']}. "
        "Тест здесь используется только для описания результатов, а не для выбора.\n\n"
        f"Для выбранного метода на test: {failures} из {total} пар запрос–seed не нашли "
        f"ни одного размеченного релевантного документа; в {zero} из них не выбрано ни одного документа. "
        "Это не число различных запросов: один запрос может повторяться для нескольких seed. "
        "Строки и ID выбранных документов записаны в error_cases.csv. Неизвестная оценка документа "
        "не доказывает его нерелевантность; разбор не устанавливает причину ошибки без чтения текстов.\n\n"
        "summary_ci.csv содержит среднее, стандартное отклонение и двусторонний 95% t-интервал "
        "по средним независимых запусков: mean ± t(0,975; n−1)·s/√n. Число seed и число запросов "
        "указаны отдельно. При n=3 интервал крайне неустойчив, нормальность распределения средних "
        "по seed проверить практически невозможно; он не охватывает неопределённость выборки запросов "
        "и не подтверждает статистической значимости различий между моделями. Для детерминированного "
        "baseline с одним запуском интервал по seed не вычисляется. Интервалы могут выходить за "
        "границы допустимых значений метрики. Общее число документов и запросов зависит от сохранённого "
        "набора и должно проверяться по паспорту данных.\n\n"
        "Графики обучения показывают скользящую среднюю тренировочной награды за 25 эпизодов, "
        "а не качество на отложенной выборке. Затраты evaluation_seconds зависят от конкретного "
        "оборудования и числа запросов. Сравнение средств и интервалов не заменяет парной проверки "
        "различий на общих запросах и дополнительных независимых запусках.\n", encoding="utf-8")
    print(f"Saved analysis in {out}; selected on validation: {winner}")


if __name__ == "__main__":
    main()
