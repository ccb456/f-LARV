"""
Stage2 LoRA Verifier Training Script (4090 OOM-Safe)
===================================================

Key fixes:
- Custom collator pads BOTH inputs and labels.
- Enable gradient checkpointing to reduce activation memory.
- Use bitsandbytes 8-bit optimizer (paged_adamw_8bit) to reduce optimizer memory.
- Reduce per-device batch size; keep effective batch via gradient accumulation.
- group_by_length=True to reduce padding waste.

Environment:
- AutoDL
- Single RTX 4090 (24GB)
- FP16

Input:
- stage3_lora_train.jsonl

Output:
- output_lora_4090/
"""

# ===============================
# 0. Imports
# ===============================
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


# ===============================
# 1. Configuration (EDIT HERE)
# ===============================
MODEL_NAME = "/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct"
DATA_PATH = "stage3_lora_train.jsonl"
OUTPUT_DIR = "output_lora_4090"

assert os.path.isdir(MODEL_NAME), f"MODEL_NAME not found: {MODEL_NAME}"
print(f"Using local model: {MODEL_NAME}")

# ---- 4090 SAFE DEFAULTS ----
MAX_SEQ_LEN = 512

# IMPORTANT: 4090 OOM-safe
PER_DEVICE_BATCH_SIZE = 1

# Keep your original effective batch roughly:
# originally 4 * 8 = 32 effective batch
# now 1 * 32 = 32 effective batch
GRADIENT_ACCUMULATION_STEPS = 32

NUM_EPOCHS = 1
LEARNING_RATE = 1e-4

USE_FP16 = True


# ===============================
# 2. Prompt Template
# ===============================
def build_prompt(example):
    return (
        "Evidence:\n"
        f"{example['evidence_text']}\n\n"
        f"Subject: {example['head']}\n"
        f"Object: {example['tail']}\n"
        f"Relation: {example['relation']}\n\n"
        "Question:\n"
        "Is the relation explicitly stated in the evidence above?\n"
        "Answer:"
    )


# ===============================
# 3. Load Dataset
# ===============================
print("Loading dataset...")
dataset = load_dataset("json", data_files={"train": DATA_PATH}, split="train")
print(f"Total training samples: {len(dataset)}")


# ===============================
# 4. Tokenizer
# ===============================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ===============================
# 5. Load Base Model (FP16)
# ===============================
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Enable memory-saving features BEFORE training
# 1) gradient checkpointing (big memory saver)
model.gradient_checkpointing_enable()
# 2) disable cache to avoid extra memory during training
model.config.use_cache = False

# (Optional) Try enabling SDPA/Flash attention (depends on your torch/transformers)
# If your environment doesn't support it, it will just be ignored / not used.
try:
    model.config.attn_implementation = "sdpa"
    print("Enabled SDPA attention.")
except Exception:
    pass


# ===============================
# 6. Apply LoRA
# ===============================
# print("Applying LoRA...")
# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     lora_dropout=0.05,
#     target_modules=["q_proj", "v_proj"],
#     bias="none",
#     task_type="CAUSAL_LM",
# )

# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

print("Applying LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# ✅ 关键修复：LoRA trainable params -> FP32
for n, p in model.named_parameters():
    if p.requires_grad:
        p.data = p.data.float()

model.print_trainable_parameters()


# ===============================
# 7. Preprocess Function
# ===============================
def preprocess(example):
    """
    Tokenize prompt + answer.
    Only compute loss on the answer tokens ("Yes"/"No").
    """
    prompt = build_prompt(example)
    answer = example["label"]  # "Yes" or "No"

    full_text = prompt + " " + answer

    tokenized_full = tokenizer(
        full_text,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding=False,
        add_special_tokens=False,
    )

    prompt_ids = tokenizer(
        prompt,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding=False,
        add_special_tokens=False,
    )["input_ids"]

    labels = [-100] * len(prompt_ids) + tokenized_full["input_ids"][len(prompt_ids):]
    labels = labels[:MAX_SEQ_LEN]
    tokenized_full["labels"] = labels
    return tokenized_full


print("Tokenizing dataset...")
dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names,
    num_proc=4
)


# ===============================
# 8. Custom Data Collator (Pads labels too!)
# ===============================
@dataclass
class DataCollatorForCausalLMWithLabels:
    tokenizer: Any
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = [f["labels"] for f in features]
        for f in features:
            f.pop("labels")

        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt"
        )

        max_len = batch["input_ids"].size(1)

        padded_labels = []
        for l in labels:
            if len(l) > max_len:
                l = l[:max_len]
            padded_labels.append(l + [self.label_pad_token_id] * (max_len - len(l)))

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


# ===============================
# 9. Training Arguments (4090 optimized)
# ===============================
# Use bitsandbytes optimizer to save a lot of VRAM.
# If you don't have bitsandbytes installed, run:
#   pip install bitsandbytes
#
# (AutoDL usually has it, but if not, install it.)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    fp16=USE_FP16,
    bf16=False,  # 4090: keep False unless you're 100% sure your stack supports BF16 well
    logging_steps=50,
    save_steps=2000,
    save_total_limit=2,
    evaluation_strategy="no",
    report_to="none",

    # BIG VRAM saver:
    optim="paged_adamw_8bit",
    max_grad_norm=0.0,


    dataloader_pin_memory=True,

    # Reduce padding waste:
    group_by_length=True,

    # A little extra stability/perf:
    warmup_ratio=0.03,
)


# ===============================
# 10. Trainer
# ===============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForCausalLMWithLabels(tokenizer=tokenizer),
)


# ===============================
# 11. Train
# ===============================
print("Starting training...")
trainer.train()

print("Saving LoRA adapter...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training finished.")
print(f"LoRA adapter saved to: {OUTPUT_DIR}")
