"""Analyze scored translation runs and produce summary tables + plots.

Reads every `results/*.scored.jsonl` (one per forward model) and produces:

  plots/<metric>_by_type.png     — grouped bars: metric × sentence-type × model,
                                    with subplots for backwards vs comparator.
  plots/<metric>_overall.png     — bar chart of overall mean/median per model.
  plots/gap_by_model.png         — backwards − comparator gap (how much the
                                    model cheats via English placeholders).
  plots/placeholder_rate.png     — fraction of sentences containing placeholders.
  summary.csv                    — one row per (model, sentence_type, arm, metric).

Dashed horizontal guide lines show the dataset baseline (mean & mean+3σ of
unrelated-pair scores) when `--baseline` is provided.

Usage:
    uv run --project yaduha-ovp python yaduha-ovp/experiments/analyze.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = HERE / "plots"
ARMS = ("comparator", "backwards")
METRICS = ("bleu", "chrf", "chrfpp", "comet")
TYPE_ORDER = [
    "subject-verb",
    "subject-verb-object",
    "two-verb",
    "two-clause",
    "complex",
    "nominalization",
]


def model_from_filename(name: str) -> str:
    # e.g. "llama3.2_1b__gpt-4o-mini.scored.jsonl" -> "llama3.2:1b"
    stem = name.replace(".scored.jsonl", "").replace(".jsonl", "")
    forward = stem.split("__", 1)[0]
    return forward.replace("_", ":", 1)


def model_sort_key(m: str) -> tuple[int, int, str]:
    """Order: self-hosted (by size asc), then API models (gpt-*) last.

    Size extracted from the trailing '<N>b' in the Ollama tag.
    """
    import re

    is_api = 1 if m.startswith("gpt") or m.startswith("claude") else 0
    match = re.search(r"(\d+(?:\.\d+)?)b", m)
    size = float(match.group(1)) if match else 0.0
    return (is_api, int(size * 1000), m)


def load_all() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(RESULTS.glob("*.scored.jsonl")):
        forward = model_from_filename(path.name)
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("error"):
                    continue
                base = {
                    "model": forward,
                    "source": r["source"],
                    "type": r["type"],
                    "has_placeholders": bool(r.get("has_placeholders")),
                }
                rows.append({**base})
                # one tidy row per (model, source, metric, arm)
                # handled below through melt
        # (walk again to extract metrics — simpler to do at the end)

    # Re-walk building tidy rows
    long_rows: list[dict[str, Any]] = []
    for path in sorted(RESULTS.glob("*.scored.jsonl")):
        forward = model_from_filename(path.name)
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("error"):
                    continue
                for metric in METRICS:
                    for arm in ARMS:
                        key = f"{metric}_{arm}"
                        if key in r:
                            long_rows.append(
                                {
                                    "model": forward,
                                    "source": r["source"],
                                    "type": r["type"],
                                    "metric": metric,
                                    "arm": arm,
                                    "score": r[key],
                                    "has_placeholders": bool(r.get("has_placeholders")),
                                }
                            )
    return pd.DataFrame(long_rows)


def load_baseline(path: Path | None) -> dict[str, tuple[float, float]]:
    """Return {metric: (mean, std)} if path provided, else {}."""
    if not path or not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def plot_by_type(df: pd.DataFrame, metric: str, baseline: dict, out_dir: Path) -> None:
    sub = df[df["metric"] == metric].copy()
    if sub.empty:
        return

    # Aggregate to median + IQR per (model, type, arm)
    agg = (
        sub.groupby(["arm", "type", "model"])["score"]
        .agg(median="median", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75))
        .reset_index()
    )
    agg["err_lo"] = agg["median"] - agg["q1"]
    agg["err_hi"] = agg["q3"] - agg["median"]

    models = sorted(df["model"].unique(), key=model_sort_key)
    palette = sns.color_palette("viridis", n_colors=len(models))
    color_map = dict(zip(models, palette))

    fig, axes = plt.subplots(
        len(ARMS), 1, figsize=(11, 3.0 * len(ARMS)), sharex=True, sharey=True
    )
    if len(ARMS) == 1:
        axes = [axes]

    for ax, arm in zip(axes, ARMS):
        asub = agg[agg["arm"] == arm]
        types_present = [t for t in TYPE_ORDER if t in asub["type"].unique()]
        x = list(range(len(types_present)))
        width = 0.8 / max(len(models), 1)

        for i, m in enumerate(models):
            msub = asub[asub["model"] == m].set_index("type").reindex(types_present)
            xs = [xi + (i - (len(models) - 1) / 2) * width for xi in x]
            ax.bar(
                xs,
                msub["median"].fillna(0).values,
                width=width,
                yerr=[msub["err_lo"].fillna(0).values, msub["err_hi"].fillna(0).values],
                color=color_map[m],
                label=m,
                capsize=2,
                edgecolor="black",
                linewidth=0.5,
            )

        if metric in baseline:
            mu, sigma = baseline[metric]
            ax.axhline(mu, color="gray", linestyle="--", linewidth=1, alpha=0.7)
            ax.axhline(mu + 3 * sigma, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        ax.set_title(f"arm: {arm}")
        ax.set_xticks(x)
        ax.set_xticklabels(types_present, rotation=20, ha="right")
        ax.set_ylabel(f"median {metric}")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].legend(
        title="Forward model",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(f"{metric.upper()}: median across sentence types (25/75 percentile bars)",
                 fontsize=12)
    fig.tight_layout()
    out = out_dir / f"{metric}_by_type.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_overall(df: pd.DataFrame, out_dir: Path) -> None:
    """One bar chart per metric: overall median per (model, arm)."""
    models = sorted(df["model"].unique(), key=model_sort_key)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.5 * len(METRICS), 3.6), sharey=False)
    if len(METRICS) == 1:
        axes = [axes]
    palette = dict(zip(ARMS, sns.color_palette("Set2", n_colors=len(ARMS))))

    for ax, metric in zip(axes, METRICS):
        sub = df[df["metric"] == metric]
        if sub.empty:
            ax.set_visible(False)
            continue
        agg = sub.groupby(["model", "arm"])["score"].median().reset_index()
        x = list(range(len(models)))
        width = 0.35
        for i, arm in enumerate(ARMS):
            vals = [
                agg[(agg["model"] == m) & (agg["arm"] == arm)]["score"].mean() for m in models
            ]
            offset = (i - 0.5) * width
            ax.bar(
                [xi + offset for xi in x],
                vals,
                width=width,
                label=arm,
                color=palette[arm],
                edgecolor="black",
                linewidth=0.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric.upper())
        ax.set_ylabel("median score")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].legend(title="arm", loc="best", fontsize=8)
    fig.suptitle("Overall median by model and arm", fontsize=12)
    fig.tight_layout()
    out = out_dir / "overall.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_gap(df: pd.DataFrame, out_dir: Path) -> None:
    """backwards − comparator gap: how much does the strong LLM 'cheat' per model."""
    models = sorted(df["model"].unique(), key=model_sort_key)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.5 * len(METRICS), 3.6), sharey=False)
    if len(METRICS) == 1:
        axes = [axes]
    for ax, metric in zip(axes, METRICS):
        sub = df[df["metric"] == metric]
        if sub.empty:
            ax.set_visible(False)
            continue
        # pivot to per-sentence gap, then mean per model
        pivot = sub.pivot_table(
            index=["model", "source"], columns="arm", values="score"
        ).reset_index()
        if "backwards" not in pivot or "comparator" not in pivot:
            ax.set_visible(False)
            continue
        pivot["gap"] = pivot["backwards"] - pivot["comparator"]
        agg = pivot.groupby("model")["gap"].agg(["mean", "std"]).reindex(models)
        x = list(range(len(models)))
        ax.bar(
            x,
            agg["mean"].values,
            yerr=agg["std"].values,
            color="#c44",
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
        )
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric.upper())
        ax.set_ylabel("backwards − comparator")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle("Placeholder-cheating gap (positive ⇒ surface form leaks English)", fontsize=12)
    fig.tight_layout()
    out = out_dir / "gap_by_model.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_placeholder_rate(df: pd.DataFrame, out_dir: Path) -> None:
    """How often does each model produce OOV placeholders?"""
    # Take unique (model, source) so we don't double-count via the long-format.
    dedup = df[["model", "source", "type", "has_placeholders"]].drop_duplicates()
    models = sorted(dedup["model"].unique(), key=model_sort_key)
    rate = (
        dedup.groupby(["model", "type"])["has_placeholders"]
        .mean()
        .reset_index()
        .rename(columns={"has_placeholders": "placeholder_rate"})
    )
    types_present = [t for t in TYPE_ORDER if t in rate["type"].unique()]

    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = list(range(len(types_present)))
    width = 0.8 / max(len(models), 1)
    palette = sns.color_palette("viridis", n_colors=len(models))
    for i, m in enumerate(models):
        vals = [
            rate[(rate["model"] == m) & (rate["type"] == t)]["placeholder_rate"].mean()
            for t in types_present
        ]
        xs = [xi + (i - (len(models) - 1) / 2) * width for xi in x]
        ax.bar(xs, vals, width=width, color=palette[i], label=m, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(types_present, rotation=20, ha="right")
    ax.set_ylabel("fraction of sentences with [lemma] placeholders")
    ax.set_title("OOV placeholder rate by sentence type")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(title="Forward model", bbox_to_anchor=(1.01, 1.0), loc="upper left", fontsize=8,
              frameon=False)
    fig.tight_layout()
    out = out_dir / "placeholder_rate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def write_summary(df: pd.DataFrame, out: Path) -> None:
    summary = (
        df.groupby(["model", "type", "arm", "metric"])["score"]
        .agg(n="count", mean="mean", median="median",
             q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
        .reset_index()
    )
    summary.to_csv(out, index=False)
    print(f"  wrote {out}")

    # Console highlight
    print("\n=== MEDIAN by (model, arm, metric) — averaged across sentence types ===")
    pivot = (
        df.groupby(["model", "arm", "metric"])["score"]
        .median()
        .reset_index()
        .pivot_table(index="model", columns=["metric", "arm"], values="score")
    )
    # Order models by size
    models_ordered = sorted(pivot.index, key=model_sort_key)
    pivot = pivot.reindex(models_ordered)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(pivot)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default=None,
                   help="JSON file mapping metric -> [mean, std] for dashed guide lines. "
                        "Defaults to results/baseline.json if it exists.")
    p.add_argument("--out", default=str(PLOTS))
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all()
    if df.empty:
        print("No scored results found in", RESULTS)
        return 1

    if args.baseline:
        baseline_path = Path(args.baseline)
    else:
        default_baseline = RESULTS / "baseline.json"
        baseline_path = default_baseline if default_baseline.exists() else None
    baseline = load_baseline(baseline_path)
    if baseline_path:
        print(f"using baseline from {baseline_path}")

    print(f"Loaded {df['source'].nunique()} sentences × {df['model'].nunique()} models "
          f"× {len(df['arm'].unique())} arms × {len(df['metric'].unique())} metrics "
          f"→ {len(df)} rows")

    print("\n== plots ==")
    for metric in METRICS:
        plot_by_type(df, metric, baseline, out_dir)
    plot_overall(df, out_dir)
    plot_gap(df, out_dir)
    plot_placeholder_rate(df, out_dir)

    print("\n== summary ==")
    write_summary(df, out_dir.parent / "summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
