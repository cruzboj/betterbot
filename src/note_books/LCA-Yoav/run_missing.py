"""Run the LLMs only on texts in texts/ that don't have a result CSV yet (same settings as LCA_Creativity.ipynb).

    python run_missing.py            # k=5, whole words only -> results/k5_clean
"""
import sys
import time
from pathlib import Path

# walk up from this script's folder until we find the repo root (the folder holding lca_core.py)
ROOT = Path(__file__).resolve().parent
while not (ROOT / "lca_core.py").exists():
    if ROOT == ROOT.parent:
        raise FileNotFoundError("lca_core.py not found in this folder or any parent folder")
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from lca_core import LCAMeter, load_texts, slug  # noqa: E402

HERE = Path(__file__).parent
MODELS = ["gpt2-medium", "EleutherAI/pythia-410m", "Qwen/Qwen2.5-1.5B"]
K = 5                    # top-k guesses per measured word
USE_ADJ = False          # also measure adjectives
COMPLETE_WORDS = True    # only whole-word guesses
OUT_DIR = HERE / "results" / "k5_clean"


def result_path(model, text_name):
    return OUT_DIR / f"{slug(model)}__{text_name}.csv"


if __name__ == "__main__":
    texts = load_texts(HERE / "texts")
    for model in MODELS:
        # texts this model hasn't measured yet
        todo = {}
        for name, text in texts.items():
            if not result_path(model, name).exists():
                todo[name] = text
        if not todo:
            print(f"{model}: nothing to do", flush=True)
            continue
        meter = LCAMeter(model)
        for name, text in todo.items():
            started = time.time()
            df = meter.run(text, k=K, use_adj=USE_ADJ, complete_words=COMPLETE_WORDS)
            df.to_csv(result_path(model, name), index=False)
            seconds = time.time() - started
            average_height = df["lca_height"].mean()
            print(f"{model:28s} {name:32s} n={len(df):4d} mean={average_height:.2f} ({seconds:.0f}s)", flush=True)
        meter.free()
