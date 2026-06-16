"""Central, location-independent path resolution for VL-GRiP3.

Every path used by the pipeline is derived from this file's location, so the
project runs from any working directory or machine without editing the code.

Optional environment-variable overrides:
  VLGRIP3_ROOT        project root          (default: auto-detected from this file)
  VLGRIP3_SAMPLE_DIR  active scene directory (default: <root>/GRiP3_Pipeline/sample_data/real_world/MX)
"""
import os
from pathlib import Path

# This file lives at <root>/GRiP3_Pipeline/utils/paths.py -> parents[2] == <root>.
PROJECT_ROOT = Path(os.environ.get("VLGRIP3_ROOT", Path(__file__).resolve().parents[2]))

GRIP3_PIPELINE_DIR = PROJECT_ROOT / "GRiP3_Pipeline"
UTILS_DIR = GRIP3_PIPELINE_DIR / "utils"
CHECKPOINTS_DIR = GRIP3_PIPELINE_DIR / "checkpoints"
SAMPLE_DATA_ROOT = GRIP3_PIPELINE_DIR / "sample_data" / "real_world"

# Active scene/sample directory shared by capture, segmentation, registration and grasping.
SAMPLE_DIR = Path(os.environ.get("VLGRIP3_SAMPLE_DIR", SAMPLE_DATA_ROOT / "MX"))

OVERLAP_PREDATOR_DIR = PROJECT_ROOT / "OverlapPredator"

__all__ = [
    "PROJECT_ROOT",
    "GRIP3_PIPELINE_DIR",
    "UTILS_DIR",
    "CHECKPOINTS_DIR",
    "SAMPLE_DATA_ROOT",
    "SAMPLE_DIR",
    "OVERLAP_PREDATOR_DIR",
]
