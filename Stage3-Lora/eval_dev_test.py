import json
import os
import re
from typing import Dict, List, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm



# -----------------------------
# 0. 配置（按你的实际路径改）
# -----------------------------
BASE_MODEL = "/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct"
LORA_DIR = "output_lora_4090"      # 你训练脚本保存的 adapter 目录
DEV_PATH = "/root/autodl-tmp/dev-F1-input-k4-.json"             # 你的 dev 文件
TEST_PATH = "test_revised.json"           # 你的 test 文件
OUT_DEV = "dev_F1_pred.json"
OUT_TEST = "test_F1_pred.json"

MAX_SEQ_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# 1. Prompt（保持与训练一致）
# -----------------------------
def build_prompt(evidence_text: str, head: str, tail: str, relation: str) -> str:
    return (
        "Evidence:\n"
        f"{evidence_text}\n\n"
        f"Subject: {head}\n"
        f"Object: {tail}\n"
        f"Relation: {relation}\n\n"
        "Question:\n"
        "Is the relation explicitly stated in the evidence above?\n"
        "Answer:"
    )


# -----------------------------
# 3. 计算 P(Yes) vs P(No)
# 使用“下一 token logits”而不是生成，快且稳定
# -----------------------------
@torch.inference_mode()
def yes_probability(model, tokenizer, prompt: str) -> float:
    # 注意：Yes/No 在你训练里是 " Yes" / " No"（前面带空格）更常见
    yes_token = tokenizer.encode(" Yes", add_special_tokens=False)
    no_token = tokenizer.encode(" No", add_special_tokens=False)
    if len(yes_token) != 1 or len(no_token) != 1:
        # 保险：如果被分成多个 token，就退化为生成式判断（很少发生）
        return yes_probability_by_generate(model, tokenizer, prompt)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        add_special_tokens=False,
    ).to(model.device)

    out = model(**inputs)
    logits = out.logits[0, -1, :]  # 下一 token 的 logits
    yes_logit = logits[yes_token[0]]
    no_logit = logits[no_token[0]]
    probs = torch.softmax(torch.stack([yes_logit, no_logit]), dim=0)
    return probs[0].item()


@torch.inference_mode()
def batch_yes_probability(model, tokenizer, prompts: List[str], batch_size: int = 32) -> List[float]:
    """
    返回每个 prompt 的 P(Yes)（在 Yes/No 二分类里 softmax 得到的 Yes 概率）
    """
    yes_token = tokenizer.encode(" Yes", add_special_tokens=False)
    no_token  = tokenizer.encode(" No", add_special_tokens=False)

    # 如果 Yes/No 不是单 token（很少见），就退化为逐条 generate（不建议大批量）
    if len(yes_token) != 1 or len(no_token) != 1:
        return [yes_probability_by_generate(model, tokenizer, p) for p in prompts]

    yes_id, no_id = yes_token[0], no_token[0]
    probs_all: List[float] = []

    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i+batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,              # 关键：batch padding
            truncation=True,
            max_length=MAX_SEQ_LEN,
            add_special_tokens=False,
        ).to(model.device)

        out = model(**enc)
        logits = out.logits  # [B, T, V]

        # 取每个样本“最后一个非 padding token”的位置
        # attention_mask: [B, T]，每行 1 的个数就是有效长度
        last_pos = enc["attention_mask"].sum(dim=1) - 1  # [B]

        batch_idx = torch.arange(logits.size(0), device=logits.device)
        last_logits = logits[batch_idx, last_pos, :]     # [B, V]

        yes_logits = last_logits[:, yes_id]
        no_logits  = last_logits[:, no_id]
        p_yes = torch.softmax(torch.stack([yes_logits, no_logits], dim=1), dim=1)[:, 0]  # [B]

        probs_all.extend(p_yes.detach().cpu().tolist())

    return probs_all



@torch.inference_mode()
def yes_probability_by_generate(model, tokenizer, prompt: str) -> float:
    # 极少用到：仅用于 Yes/No 被分成多个 token 的情况
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        add_special_tokens=False,
    ).to(model.device)

    gen = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        temperature=0.0,
    )
    # 取生成的 1 个 token，粗略判别
    new_token_id = gen[0, inputs["input_ids"].shape[1]].item()
    token_str = tokenizer.decode([new_token_id])
    return 1.0 if "Yes" in token_str else 0.0


# -----------------------------
# 4. 对一个样本的一个 pair 做预测
# -----------------------------
def predict_pair_multi(
    model,
    tokenizer,
    pair: Dict[str, Any],
    threshold: float = 0.5,
    keep_scores: bool = True,
    batch_size: int = 16,   # 新增：batch 大小
) -> Dict[str, Any]:
    evidence = pair["evidence_text_marked"]
    head = pair["h"]
    tail = pair["t"]
    cand_rels = pair["candidate_relations"]

    prompts = [build_prompt(evidence, head, tail, rel) for rel in cand_rels]
    p_list = batch_yes_probability(model, tokenizer, prompts, batch_size=batch_size)

    rel_scores = [{"relation": rel, "p_yes": p} for rel, p in zip(cand_rels, p_list)]
    rel_scores.sort(key=lambda x: x["p_yes"], reverse=True)

    selected = [x for x in rel_scores if x["p_yes"] >= threshold]

    if keep_scores:
        return {"relations": selected}
    else:
        return {"relations": [x["relation"] for x in selected]}


# -----------------------------
# 5. 跑 dev/test 并写出你要的格式
# -----------------------------
def run_file(in_path: str, out_path: str, model, tokenizer):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    outputs: List[Dict[str, Any]] = []
    for sample in tqdm(data, desc=f"Evaluating {os.path.basename(in_path)}", unit="sample"):
        out_item = {"title": sample["title"], "relations": []}

        for pair in tqdm(sample["pairs"], desc="Pairs", unit="pair", leave=False):
            # pred = predict_pair(model, tokenizer, sample, pair)
            pred = predict_pair_multi(model, tokenizer, pair)

            # 你期望的 relation 条目结构
            out_item["relations"].append(
                {
                    "h": pair["h"],
                    "t": pair["t"],
                    "h_id": pair["h_id"],
                    "t_id": pair["t_id"],
                    "type_h": pair["type_h"],
                    "type_t": pair["type_t"],
                    "relations": pred["relations"],
                    # "evidence_sent_ids": pair.get("evidence_sent_ids", []),
                }
            )

        outputs.append(out_item)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {out_path}  (items={len(outputs)})")


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    base.config.use_cache = True  # 评估可开 cache

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()

    # dev
    if os.path.exists(DEV_PATH):
        run_file(DEV_PATH, OUT_DEV, model, tokenizer)
    else:
        print(f"[WARN] DEV not found: {DEV_PATH}")

    # test
    # if os.path.exists(TEST_PATH):
    #     run_file(TEST_PATH, OUT_TEST, model, tokenizer)
    # else:
    #     print(f"[WARN] TEST not found: {TEST_PATH}")


if __name__ == "__main__":
    main()
