#!/usr/bin/env python3
"""LoRA fine-tune Qwen3.5-0.8B on InfoVQA-style samples (T4-sized config).

- LoRA r=16 on decoder attention+MLP projections; vision encoder frozen.
- fp16 AMP, batch 1 + grad-accum 8, grad checkpointing, max_pixels capped at
  768*28*28 for training memory (eval keeps its own 1280*28*28 cap).
- Prompt matches eval exactly: question + "Answer the question using a single
  word or phrase."; loss only on answer tokens.
- Saves adapter + merged fp16 model (for vLLM serving) to Drive.
"""
import json
import os
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (AutoModelForImageTextToText, AutoProcessor, Trainer,
                          TrainingArguments)

MODEL = "Qwen/Qwen3.5-0.8B"
LORA_DIR = Path("/content/drive/MyDrive/vlm_eval/lora")
SUFFIX = "\nAnswer the question using a single word or phrase."
MAX_PIXELS = 768 * 28 * 28

processor = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28, max_pixels=MAX_PIXELS)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="cuda")
model.config.use_cache = False

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()


class InfoVQATrain(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["image"]).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": r["question"] + SUFFIX}]}]
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        full = prompt + r["answer"] + processor.tokenizer.eos_token
        enc = processor(text=full, images=[img], return_tensors="pt")
        ids = enc["input_ids"][0]
        # loss only on answer tokens: mask everything up to the prompt length
        plen = len(processor(text=prompt, images=[img])["input_ids"][0])
        labels = ids.clone()
        labels[:plen] = -100
        # squeeze the batch dim only on per-token tensors; pixel_values /
        # image_grid_thw are not batch-shaped and must stay intact
        out = {k: (v[0] if k in ("input_ids", "attention_mask", "mm_token_type_ids") else v)
               for k, v in enc.items()}
        out["labels"] = labels
        return out


def collate(batch):  # batch size 1: per-token tensors get a batch dim,
    b = batch[0]      # pixel_values / image_grid_thw stay unbatched (qwen-vl convention)
    return {k: (v.unsqueeze(0) if k in ("input_ids", "attention_mask", "labels",
                                        "mm_token_type_ids") else v)
            for k, v in b.items()}


args = TrainingArguments(
    output_dir=str(LORA_DIR / "ckpt"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="no",
    report_to=[],
    remove_unused_columns=False,
    dataloader_num_workers=2,
    seed=42,
)

trainer = Trainer(model=model, args=args,
                  train_dataset=InfoVQATrain(LORA_DIR / "train.jsonl"),
                  data_collator=collate)
trainer.train()

model.save_pretrained(LORA_DIR / "adapter")
print("adapter saved")
merged = model.merge_and_unload()
merged.save_pretrained(LORA_DIR / "merged", safe_serialization=True)
processor.save_pretrained(LORA_DIR / "merged")
print("merged model saved ->", LORA_DIR / "merged")
