"""Fine-tune the PaliGemma *action* (robotic command) head.

Run as a script:
    python ft_action_module.py                 # train, save locally (no Hub push)
    python ft_action_module.py --push-to-hub   # also push to the HF Hub
"""
from pathlib import Path

try:
    import ft_common
except ImportError:  # when imported as part of the package
    from . import ft_common

# Repo-relative dataset / output paths (no hardcoded absolute paths).
HERE = Path(__file__).resolve().parent          # GRiP3_Pipeline/
DATASET_DIR = HERE / "utils" / "dataset" / "train_basic_cmd"
OUTPUT_DIR = HERE / "checkpoints" / "paligemma-action-module"


def build_config() -> "ft_common.FineTuneConfig":
    return ft_common.FineTuneConfig(
        train_images_dir=str(DATASET_DIR),
        train_annotations=str(DATASET_DIR / "annotations_train.jsonl"),
        valid_images_dir=str(DATASET_DIR),
        valid_annotations=str(DATASET_DIR / "annotations_valid.jsonl"),
        output_dir=str(OUTPUT_DIR),
        # Action head: QLoRA, freeze vision tower AND projector.
        use_lora=False,
        use_qlora=True,
        freeze_vision=True,
        freeze_projector=True,
        num_train_epochs=10,
        gradient_accumulation_steps=8,
        warmup_steps=20,
        learning_rate=3e-5,
        push_to_hub=False,
    )


if __name__ == "__main__":
    cfg = ft_common.apply_cli_overrides(build_config())
    ft_common.train(cfg)
