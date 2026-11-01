"""Run the LLMs only on texts in texts/ that don't have a result CSV yet (same settings as LCA_Creativity.ipynb).

    python run_missing.py            # k=5, whole words only -> results/k5_clean
"""
import sys
import time
from pathlib import Path

ROOT = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
            if (p / "lca_core.py").exists())
sys.path.insert(0, str(ROOT))
from lca_core import LCAMeter, load_texts, slug  # noqa: E402

HERE = Path(__file__).parent
MODELS = ["gpt2-medium", "EleutherAI/pythia-410m", "Qwen/Qwen2.5-1.5B"]
K, USE_ADJ, COMPLETE_WORDS = 5, False, True
OUT_DIR = HERE / "results" / "k5_clean"

if __name__ == "__main__":
    texts = load_texts(HERE / "texts")
    for model in MODELS:
        todo = {n: t for n, t in texts.items() if not (OUT_DIR / f"{slug(model)}__{n}.csv").exists()}
        if not todo:
            print(f"{model}: nothing to do", flush=True)
            continue
        meter = LCAMeter(model)
        for name, text in todo.items():
            t0 = time.time()
            df = meter.run(text, k=K, use_adj=USE_ADJ, complete_words=COMPLETE_WORDS)
            df.to_csv(OUT_DIR / f"{slug(model)}__{name}.csv", index=False)
            print(f"{model:28s} {name:32s} n={len(df):4d} mean={df['lca_height'].mean():.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        meter.free()
