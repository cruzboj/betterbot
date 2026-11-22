# make analysis/ (and through it config/qwen_config.py) importable when pytest runs from Qwen7B
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
