#!/usr/bin/env python3
"""Patch the local VLMEvalKit checkout for this eval. Idempotent: safe to rerun
after a fresh clone (e.g. post-disconnect):
  python /content/drive/MyDrive/vlm_eval/scripts/setup/register_models.py

Patches:
1. Register `Qwen3.5-0.8B` (Qwen/Qwen3.5-0.8B) in vlmeval/config.py: VLMEvalKit
   only ships large Qwen3.5 entries. Uses Qwen3VLChat (same class the 397B entry
   uses), greedy decoding (temperature=0) per the eval protocol in TASK.md.
2. qwen3_vl/model.py: flash_attention_2 -> sdpa. T4 (Turing, SM75) is not
   supported by flash-attn 2; SDPA is the correct fallback.
"""
import re
from pathlib import Path

KIT = Path("/content/VLMEvalKit")

# --- 1. config.py: add Qwen3.5-0.8B to qwen3_5_series ---
cfg = KIT / "vlmeval/config.py"
text = cfg.read_text()
if '"Qwen3.5-0.8B"' not in text:
    anchor = "qwen3_5_series = {"
    entry = '''qwen3_5_series = {
    "Qwen3.5-0.8B": partial(
        vlm.Qwen3VLChat,
        model_path="Qwen/Qwen3.5-0.8B",
        use_custom_prompt=False,
        max_new_tokens=512,
        temperature=0.0,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,  # uncapped tiling OOMs a 15GB T4 on doc images
    ),'''
    assert anchor in text, "qwen3_5_series anchor not found in config.py"
    text = text.replace(anchor, entry, 1)
    cfg.write_text(text)
    print("config.py: added Qwen3.5-0.8B entry")
else:
    print("config.py: Qwen3.5-0.8B already registered")

# --- 2. qwen3_vl/model.py: no flash-attn on T4 ---
m = KIT / "vlmeval/vlm/qwen3_vl/model.py"
mt = m.read_text()
n = mt.count("attn_implementation='flash_attention_2'")
if n:
    mt = mt.replace("attn_implementation='flash_attention_2'", "attn_implementation='sdpa'")
    m.write_text(mt)
    print(f"qwen3_vl/model.py: replaced flash_attention_2 -> sdpa ({n} sites)")
else:
    print("qwen3_vl/model.py: no flash_attention_2 references (already patched)")
