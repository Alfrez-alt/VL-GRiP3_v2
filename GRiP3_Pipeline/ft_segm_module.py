"""Fine-tune the PaliGemma *segmentation* head.

Run as a script:
    python ft_segm_module.py                 # train, save locally (no Hub push)
    python ft_segm_module.py --push-to-hub   # also push to the HF Hub
"""
from pathlib import Path

try:
    import ft_common
except ImportError:  # when imported as part of the package
    from . import ft_common

# Repo-relative dataset / output paths (no hardcoded absolute paths).
HERE = Path(__file__).resolve().parent          # GRiP3_Pipeline/
DATASET_DIR = HERE / "dataset" / "training_dataset_segmentation"
OUTPUT_DIR = HERE / "checkpoints" / "paligemma-segm-module"


def build_config() -> "ft_common.FineTuneConfig":
    return ft_common.FineTuneConfig(
        train_images_dir=str(DATASET_DIR),
        train_annotations=str(DATASET_DIR / "annotations_train.jsonl"),
        valid_images_dir=str(DATASET_DIR),
        valid_annotations=str(DATASET_DIR / "annotations_valid.jsonl"),
        output_dir=str(OUTPUT_DIR),
        # Segmentation head: LoRA, freeze vision tower but keep the projector
        # trainable so visual features can adapt to the segmentation task.
        use_lora=True,
        use_qlora=False,
        freeze_vision=True,
        freeze_projector=False,
        num_train_epochs=15,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        learning_rate=1e-5,
        push_to_hub=False,
    )


if __name__ == "__main__":
    cfg = ft_common.apply_cli_overrides(build_config())
    ft_common.train(cfg)
