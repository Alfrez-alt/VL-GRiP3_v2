"""Centralized, portable path resolution for VL-GRiP3.

Every path is derived from the repository root, which is detected from this
file's location, so the project runs from any clone directory without editing
source files. Each value can be overridden with an environment variable to
support different machines / data layouts.

Environment overrides:
    VLGRIP3_ROOT         repository root (defaults to the detected clone dir)
    VLGRIP3_SCENE_DIR    active scene/working dir (captures + per-run artifacts)
    VLGRIP3_CHECKPOINTS  PaliGemma PEFT checkpoints dir
    VLGRIP3_PREDATOR     OverlapPredator install dir
    VLGRIP3_ROBOT_IP     UR controller IP address (UR3/UR5e/...)
"""
from __future__ import annotations

import os
from pathlib import Path

# utils/ -> GRiP3_Pipeline/ -> <repo root>
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(os.environ.get("VLGRIP3_ROOT", _DEFAULT_ROOT))

PIPELINE_DIR = REPO_ROOT / "GRiP3_Pipeline"
UTILS_DIR = PIPELINE_DIR / "utils"

# Active scene / working directory. Captures are written here and the
# Predator + M2T2 stages read & write their artifacts here. Defaults to the
# bundled sample so the pipeline can be exercised without a camera/robot.
SCENE_DIR = Path(
    os.environ.get("VLGRIP3_SCENE_DIR", PIPELINE_DIR / "sample_data" / "real_world" / "MX")
)

CHECKPOINTS_DIR = Path(os.environ.get("VLGRIP3_CHECKPOINTS", PIPELINE_DIR / "checkpoints"))
SEGM_CHECKPOINT = CHECKPOINTS_DIR / "paligemma-segm-module"
ACTION_CHECKPOINT = CHECKPOINTS_DIR / "paligemma-action-module"

# Datasets used by the fine-tuning scripts (see ft_*_module.py).
ACTION_DATASET_DIR = UTILS_DIR / "dataset" / "train_basic_cmd"
SEGM_DATASET_DIR = PIPELINE_DIR / "dataset" / "training_dataset_segmentation"
# Image fed to the action head at inference time.
ACTION_INFER_IMAGE = ACTION_DATASET_DIR / "1.jpg"

WHISPER_DIR = Path(os.environ.get("VLGRIP3_WHISPER", PIPELINE_DIR / "whisper"))

# OverlapPredator integration.
OVERLAP_PREDATOR_DIR = Path(os.environ.get("VLGRIP3_PREDATOR", REPO_ROOT / "OverlapPredator"))
PREDATOR_SCRIPT = OVERLAP_PREDATOR_DIR / "scripts" / "demo_save.py"
PREDATOR_CFG_DIR = OVERLAP_PREDATOR_DIR / "configs" / "test"

# UR controller IP. NOTE: the bundled robot poses/workspace were calibrated for
# the original UR3 cell; re-measure them for your own cell (e.g. a UR5e).
ROBOT_IP = os.environ.get("VLGRIP3_ROBOT_IP", "192.168.1.254")


def scene_file(name: str, scene_dir: os.PathLike | str | None = None) -> Path:
    """Return the path of an artifact ``name`` inside the active scene dir."""
    return Path(scene_dir) / name if scene_dir is not None else SCENE_DIR / name
