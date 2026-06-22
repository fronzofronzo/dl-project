"""Single source of truth for filesystem paths.

Replaces the brittle `Path(__file__).resolve().parent.parent` anchors that
broke once modules moved into sub-packages. Import from here instead.
"""
from pathlib import Path

# src/common/paths.py -> parents[2] == repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data"
RESULTS = PROJECT_ROOT / "results"
FEATURES = PROJECT_ROOT / "features"

EVAL_JSON = DATA / "celeba_evaluation.json"
DB_TEST = DATA / "clip_features_test.pt"
DB_TRAIN = DATA / "clip_features_train.pt"  # extracted offline for Solution B training
