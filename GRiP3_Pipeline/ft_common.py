"""Shared PaliGemma PEFT fine-tuning logic for the segmentation and action heads.

The two thin scripts ``ft_segm_module.py`` and ``ft_action_module.py`` only
define their dataset paths and hyperparameters and then call :func:`train`,
so the dataset class, collation, model setup and training loop live in one place.

Nothing runs on import — call :func:`train` from a ``__main__`` guard.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)

MODEL_ID = "google/paligemma-3b-mix-448"
LORA_TARGET_MODULES = [
    "q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj",
]


class JSONLDataset(Dataset):
    """Reads a JSONL file and loads images from a local folder.

    Each line must contain: ``{"image": "<file>", "prefix": "<input>", "suffix": "<target>"}``.
    """

    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        super().__init__()
        self.records: List[Dict[str, Any]] = []
        self.image_dir = image_directory_path
        with open(jsonl_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        pil_image = Image.open(os.path.join(self.image_dir, item["image"])).convert("RGB")
        return {"image": pil_image, "prefix": item["prefix"], "suffix": item.get("suffix", "")}


@dataclass
class FineTuneConfig:
    """All knobs for one fine-tuning run."""
    train_images_dir: str
    train_annotations: str
    valid_images_dir: str
    valid_annotations: str
    output_dir: str
    # PEFT strategy
    use_lora: bool = False
    use_qlora: bool = True
    freeze_vision: bool = True
    freeze_projector: bool = True
    lora_r: int = 8
    # Optimization
    num_train_epochs: int = 10
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 20
    learning_rate: float = 3e-5
    weight_decay: float = 1e-6
    # Misc
    push_to_hub: bool = False
    model_id: str = MODEL_ID


def _build_collate_fn(processor: PaliGemmaProcessor, device: str):
    def collate_fn(examples: List[Dict[str, Any]]):
        texts = [f"<image>{ex['prefix']}" for ex in examples]
        images = [ex["image"] for ex in examples]
        targets = [ex["suffix"] for ex in examples]
        batch = processor(text=texts, images=images, suffix=targets,
                          return_tensors="pt", padding="longest")
        for key, value in batch.items():
            if value.dtype in (torch.float32, torch.float64):
                batch[key] = value.to(torch.bfloat16)
            batch[key] = batch[key].to(device)
        return batch
    return collate_fn


def _load_model(cfg: FineTuneConfig, device: str):
    if cfg.use_lora or cfg.use_qlora:
        lora_config = LoraConfig(r=cfg.lora_r, target_modules=LORA_TARGET_MODULES,
                                 task_type="CAUSAL_LM")
        if cfg.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base_model = PaliGemmaForConditionalGeneration.from_pretrained(
                cfg.model_id, device_map="auto",
                quantization_config=bnb_config, torch_dtype=torch.bfloat16,
            )
        else:
            base_model = PaliGemmaForConditionalGeneration.from_pretrained(
                cfg.model_id, device_map="auto", torch_dtype=torch.bfloat16,
            )
        model = get_peft_model(base_model, lora_config).to(device)
        model.print_trainable_parameters()
    else:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            cfg.model_id, device_map="auto", torch_dtype=torch.bfloat16,
        ).to(device)

    if cfg.freeze_vision and hasattr(model, "vision_tower"):
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        if cfg.freeze_projector and hasattr(model, "multi_modal_projector"):
            for param in model.multi_modal_projector.parameters():
                param.requires_grad = False
    return model


def train(cfg: FineTuneConfig) -> None:
    """Run a fine-tuning job described by ``cfg``."""
    logging.basicConfig(level=os.environ.get("VLGRIP3_LOGLEVEL", "INFO"),
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = JSONLDataset(cfg.train_annotations, cfg.train_images_dir)
    valid_dataset = JSONLDataset(cfg.valid_annotations, cfg.valid_images_dir)
    processor = PaliGemmaProcessor.from_pretrained(cfg.model_id)
    model = _load_model(cfg, device)

    args = TrainingArguments(
        num_train_epochs=cfg.num_train_epochs,
        remove_unused_columns=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        adam_beta2=0.999,
        logging_steps=10,
        optim="adamw_hf",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        push_to_hub=cfg.push_to_hub,
        output_dir=cfg.output_dir,
        bf16=True,
        report_to=["tensorboard"],
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=_build_collate_fn(processor, device),
        args=args,
    )

    torch.cuda.empty_cache()
    logger.info("Starting training -> %s", cfg.output_dir)
    trainer.train()
    logger.info("Training completed.")


def apply_cli_overrides(cfg: FineTuneConfig, argv=None) -> FineTuneConfig:
    """Let a caller override the most common knobs from the command line."""
    p = argparse.ArgumentParser(description="Fine-tune a PaliGemma PEFT head")
    p.add_argument("--epochs", type=int, default=cfg.num_train_epochs)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--output-dir", default=cfg.output_dir)
    p.add_argument("--push-to-hub", action="store_true", default=cfg.push_to_hub,
                   help="push checkpoints to the Hugging Face Hub (off by default)")
    args = p.parse_args(argv)
    cfg.num_train_epochs = args.epochs
    cfg.learning_rate = args.lr
    cfg.output_dir = args.output_dir
    cfg.push_to_hub = args.push_to_hub
    return cfg
