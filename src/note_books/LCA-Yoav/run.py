"""Run every model over every text in texts/, save per-(model,text) CSVs, then build the
summary table and plots (per model, and per text with all models + their average) for
each metric:
    lca_height    Eran's definition: max climb from any top-k completion to their LCA
    lca_weighted  (only with --weighted) expected LCA height of two completions drawn by the
                  model's probabilities - set aside, it didn't change factual vs creative

    python run.py                          # -> results/k5_clean  (whole-word completions only)
    python run.py --raw-candidates         # -> results/k5        (old behaviour, word pieces allowed)
    python run.py --adj                    # also measure adjectives
    python run.py --replot                 # rebuild summary + plots from the saved CSVs
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
DEFAULT_MODELS = ["gpt2-medium", "EleutherAI/pythia-410m", "Qwen/Qwen2.5-1.5B"]
COLORS = {"factual": "#2a6fdb", "creative": "#e0662f"}
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # dataviz categorical slots 1-3 (all-pairs CVD-safe)
AVG_COLOR = "#1f1f1f"
METRICS = {"lca_height": "LCA height (max)", "lca_weighted": "LCA height (prob-weighted)"}


def kind(text_name):
    return "factual" if text_name.startswith("factual") else "creative"


def slug(model):
    return model.replace("/", "_")


def summarize(df, col="lca_height"):
    h = df[col]
    return dict(n=len(h), mean=h.mean(), var=h.var(), std=h.std(),
                mean_abs_jump=h.diff().abs().mean(), p90=h.quantile(0.9))


def load(out_dir, models):
    """{model: {text: df}} from the saved CSVs."""
    return {m: {f.stem.split("__")[1]: pd.read_csv(f) for f in sorted(out_dir.glob(f"{slug(m)}__*.csv"))}
            for m in models}


def plot_model(model, results, out, smooth, col):
    """Factual texts in the left column, creative texts in the right one."""
    cols = [[n for n in results if kind(n) == k] for k in ("factual", "creative")]
    rows = max(len(c) for c in cols)
    fig, axes = plt.subplots(rows, 2, figsize=(20, 2.1 * rows), sharey=True, squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for c, names in enumerate(cols):
        for r, name in enumerate(names):
            ax, df = axes[r, c], results[name]
            color = COLORS[kind(name)]
            ax.set_visible(True)
            ax.plot(df[col].values, color=color, alpha=0.35, lw=0.9)
            ax.plot(df[col].rolling(smooth, center=True, min_periods=1).mean().values, color=color, lw=2)
            s = summarize(df, col)
            ax.set_title(f"{name}   mean={s['mean']:.2f}  var={s['var']:.2f}  |jump|={s['mean_abs_jump']:.2f}",
                         fontsize=9, loc="left")
            ax.set_ylabel(METRICS[col])
            ax.grid(alpha=0.3)
        if names:
            axes[len(names) - 1, c].set_xlabel("content word # along the text")
    fig.suptitle(f"{model} — {METRICS[col]} of top-k completions (thin: raw, thick: rolling mean {smooth})")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_per_text(data, out_dir, smooth, col):
    """One figure per text: every model's curve + the cross-model average.
    Curves are aligned on word_idx (the same content words for every model)."""
    texts = sorted({t for per_model in data.values() for t in per_model})
    out_dir.mkdir(exist_ok=True)
    for name in texts:
        table = pd.DataFrame({m: d[name].set_index("word_idx")[col] for m, d in data.items() if name in d})
        table = table.sort_index()  # rows: content word, cols: model
        roll = lambda s: s.rolling(smooth, center=True, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(14, 4.2))
        for (m, s), color in zip(table.items(), MODEL_COLORS):
            ax.plot(table.index, s, color=color, lw=0.7, alpha=0.18)
            ax.plot(table.index, roll(s), color=color, lw=1.6, alpha=0.9, label=f"{m}  (mean {s.mean():.2f})")
        avg = table.mean(axis=1)
        ax.plot(table.index, roll(avg), color=AVG_COLOR, lw=3,
                label=f"average of models  (mean {avg.mean():.2f}, var {avg.var():.2f})")
        ax.set_title(f"{name}  [{kind(name)}]  — {METRICS[col]} along the text "
                     f"(thin: raw, thick: rolling mean {smooth})", loc="left", fontsize=11)
        ax.set_xlabel("content word # along the text")
        ax.set_ylabel(METRICS[col])
        ax.set_ylim(0, 16)
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=110)
        plt.close(fig)


def report(out_dir, models, smooth, weighted=False):
    """summary.csv + all plots, for every metric present in the CSVs."""
    data = load(out_dir, models)
    some_df = next(df for d in data.values() for df in d.values())
    metrics = [c for c in METRICS if c in some_df and (weighted or c != "lca_weighted")]
    rows = []
    for m, per_text in data.items():
        for name, df in per_text.items():
            row = dict(model=m, text=name, kind=kind(name))
            for col in metrics:
                sfx = "" if col == "lca_height" else "_w"
                s, nouns = summarize(df, col), summarize(df[df["pos"] == "n"], col)
                row.update({f"{k}{sfx}": v for k, v in s.items() if k != "n"})
                row.update({f"noun_mean{sfx}": nouns["mean"], f"noun_var{sfx}": nouns["var"]})
            row["n"], row["verb_frac"] = len(df), (df["pos"] == "v").mean()
            rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv(out_dir / "summary.csv", index=False)

    for col in metrics:
        sfx = "" if col == "lca_height" else "_w"
        for m in models:
            plot_model(m, data[m], out_dir / f"{slug(m)}{sfx}.png", smooth, col)
        plot_per_text(data, out_dir / ("per_text" if col == "lca_height" else "per_text_weighted"), smooth, col)
        cols = [f"{c}{sfx}" for c in ("mean", "var", "mean_abs_jump", "noun_mean", "noun_var")]
        print(f"\n== {METRICS[col]}\n", s.groupby(["model", "kind"])[cols].mean().round(2))
    print("\nverb fraction:", s.groupby("kind")["verb_frac"].mean().round(2).to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--texts", nargs="*", help="text names (default: all of texts/*.txt)")
    ap.add_argument("--k", type=int, default=5, help="completions per position")
    ap.add_argument("--adj", action="store_true", help="also measure adjectives")
    ap.add_argument("--raw-candidates", action="store_true",
                    help="don't require completions to be whole words (old behaviour)")
    ap.add_argument("--weighted", action="store_true", help="also compute/plot the prob-weighted metric")
    ap.add_argument("--smooth", type=int, default=10)
    ap.add_argument("--replot", action="store_true", help="rebuild summary + plots from saved CSVs")
    args = ap.parse_args()

    tag = f"k{args.k}" + ("_adj" if args.adj else "") + ("" if args.raw_candidates else "_clean")
    out_dir = ROOT / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.replot:
        from measure import LCAMeter  # heavy import (torch) only when running models
        texts = {p.stem: p.read_text(encoding="utf-8") for p in sorted((ROOT / "texts").glob("*.txt"))
                 if not args.texts or p.stem in args.texts}
        for model in args.models:
            meter = LCAMeter(model)
            for name, text in texts.items():
                df = meter.run(text, k=args.k, use_adj=args.adj, complete_words=not args.raw_candidates,
                               weighted=args.weighted)
                df.to_csv(out_dir / f"{slug(model)}__{name}.csv", index=False)
                print(f"{model:28s} {name:40s} n={len(df)}  mean={df['lca_height'].mean():.2f}", flush=True)
            meter.free()
    report(out_dir, args.models, args.smooth, args.weighted)
    print("\nresults ->", out_dir)


if __name__ == "__main__":
    main()
