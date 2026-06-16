import os
import json
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig
from transformers import BitsAndBytesConfig

# ------------------------------------------------------------------------------
# DATASET PATHS
# ------------------------------------------------------------------------------
# Repo-relative dataset location (edit _DATASET_DIR if your data lives elsewhere).
_DATASET_DIR = Path(__file__).resolve().parent / "utils" / "dataset" / "train_basic_cmd"
TRAIN_IMAGES_DIR = str(_DATASET_DIR)
TRAIN_ANNOTATIONS = str(_DATASET_DIR / "annotations_train.jsonl")

VALID_IMAGES_DIR = str(_DATASET_DIR)
VALID_ANNOTATIONS = str(_DATASET_DIR / "annotations_valid.jsonl")

# Repo-relative output directory for the trained adapter.
_OUTPUT_DIR = Path(__file__).resolve().parent / "checkpoints" / "paligemma-action-module"

# ------------------------------------------------------------------------------
# LOCAL JSONL DATASET CLASS
# ------------------------------------------------------------------------------
class JSONLDataset(Dataset):
    """
    Reads a JSONL file and loads images from the local folder.
    Each line must contain:
    {
      "image": "filename.jpg",
      "prefix": "input text",
      "suffix": "expected output (segmentation tokens + robot commands)"
    }
    """
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        super().__init__()
        self.records = []
        self.image_dir = image_directory_path

        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    self.records.append(data)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        image_name = item["image"]
        prefix = item["prefix"]
        suffix = item.get("suffix", "")
        img_path = os.path.join(self.image_dir, image_name)
        pil_image = Image.open(img_path).convert("RGB")
        return {
            "image": pil_image,
            "prefix": prefix,
            "suffix": suffix
        }

# ------------------------------------------------------------------------------
# LOAD LOCAL DATASETS
# ------------------------------------------------------------------------------
train_dataset = JSONLDataset(jsonl_file_path=TRAIN_ANNOTATIONS, image_directory_path=TRAIN_IMAGES_DIR)
valid_dataset = JSONLDataset(jsonl_file_path=VALID_ANNOTATIONS, image_directory_path=VALID_IMAGES_DIR)

# ------------------------------------------------------------------------------
# LOAD PROCESSOR
# ------------------------------------------------------------------------------
model_id = "google/paligemma-3b-mix-448"  # Choose the desired model
processor = PaliGemmaProcessor.from_pretrained(model_id)

# ------------------------------------------------------------------------------
# CUSTOM COLLATE FUNCTION
# ------------------------------------------------------------------------------
def collate_fn(examples: List[Dict[str, Any]]):
    # Prepend the <image> token to signal multi-modal input.
    texts = [f"<image>{ex['prefix']}" for ex in examples]
    images = [ex["image"] for ex in examples]
    targets = [ex["suffix"] for ex in examples]

    batch = processor(
        text=texts,
        images=images,
        suffix=targets,
        return_tensors="pt",
        padding="longest"
    )

    # Cast floating-point tensors to bfloat16 and move everything to the device.
    for key, value in batch.items():
        if value.dtype in [torch.float32, torch.float64]:
            batch[key] = value.to(torch.bfloat16).to(device)
        else:
            batch[key] = value.to(device)
    return batch

# ------------------------------------------------------------------------------
# LORA / QLoRA CONFIGURATION AND DEVICE SETUP
# ------------------------------------------------------------------------------
USE_LORA = False
USE_QLORA = True   # Use QLoRA if desired
# Set FREEZE_VISION to True to freeze the visual backbone (do not train segmentation).
FREEZE_VISION = True
device = "cuda" if torch.cuda.is_available() else "cpu"

if USE_LORA or USE_QLORA:
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )
    else:
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    model = get_peft_model(base_model, lora_config)
    model = model.to(device)
    model.print_trainable_parameters()
else:
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).to(device)

# ---- FREEZING THE VISION MODULE ----
if FREEZE_VISION and hasattr(model, "vision_tower"):
    # Freeze the vision tower: the visual features will not be updated.
    for param in model.vision_tower.parameters():
        param.requires_grad = False
    # If present, freeze the multi_modal_projector as well.
    if hasattr(model, "multi_modal_projector"):
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = False

# ------------------------------------------------------------------------------
# TRAINING ARGUMENTS
# ------------------------------------------------------------------------------
args = TrainingArguments(
    num_train_epochs=10,                        # Small dataset, hence fewer epochs
    remove_unused_columns=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,              # Larger virtual batch for stability
    warmup_steps=20,                            # Longer warmup for a gradual transition
    learning_rate=3e-5,                         # Slightly reduced learning rate
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=10,
    optim="adamw_hf",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    push_to_hub=True,                         # Set to False if you do not want to push to the hub
    output_dir=str(_OUTPUT_DIR),
    bf16=True,
    report_to=["tensorboard"],
    dataloader_pin_memory=False,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

# ------------------------------------------------------------------------------
# CREATE TRAINER
# ------------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=collate_fn,
    args=args
)

# ------------------------------------------------------------------------------
# START TRAINING
# ------------------------------------------------------------------------------
torch.cuda.empty_cache()
print("Starting training...")
trainer.train()
print("Training completed.")


# FINE-TUNING ROBOTIC COMMAND SCRIPT.
