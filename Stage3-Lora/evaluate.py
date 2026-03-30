import json
import argparse
from collections import defaultdict


def load_gold(data_dir, split="test"):
    """
    Load Re-DocRED gold triples.
    Returns:
        gold_triples: set of (title, h, t, r)
        train_triples: set of (title, h, t, r)  # for Ign F1
    """
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # train_data = load(f"{data_dir}/train_revised.json")
    # eval_data = load(f"{data_dir}/{split}_revised.json")
    train_data = load(f"{data_dir}/train_annotated.json")
    eval_data = load(f"{data_dir}/{split}.json")

    train_triples = set()
    for doc in train_data:
        title = doc["title"]
        for label in doc.get("labels", []):
            train_triples.add((
                title,
                label["h"],
                label["t"],
                label["r"]
            ))

    gold_triples = set()
    for doc in eval_data:
        title = doc["title"]
        for label in doc.get("labels", []):
            gold_triples.add((
                title,
                label["h"],
                label["t"],
                label["r"]
            ))

    return gold_triples, train_triples


def evaluate(preds, gold_triples, train_triples):
    """
    preds: list of dicts with keys {title, h_idx, t_idx, r}
    """

    pred_triples = set()
    for p in preds:
        pred_triples.add((
            p["title"],
            p["h_idx"],
            p["t_idx"],
            p["r"]
        ))

    # ===== Standard Micro P/R/F1 =====
    tp = len(pred_triples & gold_triples)
    fp = len(pred_triples - gold_triples)
    fn = len(gold_triples - pred_triples)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    # ===== Ign F1 (ignore triples seen in training) =====
    ign_gold = gold_triples - train_triples
    ign_pred = pred_triples - train_triples

    ign_tp = len(ign_pred & ign_gold)
    ign_fp = len(ign_pred - ign_gold)
    ign_fn = len(ign_gold - ign_pred)

    ign_precision = ign_tp / (ign_tp + ign_fp) if ign_tp + ign_fp > 0 else 0.0
    ign_recall = ign_tp / (ign_tp + ign_fn) if ign_tp + ign_fn > 0 else 0.0
    ign_f1 = (
        2 * ign_precision * ign_recall / (ign_precision + ign_recall)
        if ign_precision + ign_recall > 0 else 0.0
    )

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Ign_F1": ign_f1
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="../ablation/results/with-doc_dev_pred_docred_official.json", help="official prediction json")
    parser.add_argument("--data_dir", default="../datasets/docred", help="Re-DocRED data dir")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])

    args = parser.parse_args()

    with open(args.pred, "r", encoding="utf-8") as f:
        preds = json.load(f)
    print( f"preds: {len(preds)}")

    gold_triples, train_triples = load_gold(args.data_dir, args.split)

    print(f"gold_triples: {len(gold_triples)}")
    print(f"train_triples: {len(train_triples)}")
    metrics = evaluate(preds, gold_triples, train_triples)

    print("=" * 60)
    print(f"[Re-DocRED Evaluation] split={args.split}")
    for k, v in metrics.items():
        print(f"{k:10s}: {v * 100:.2f}")
    print("=" * 60)

    # print("Pred relations sample:", list({p["r"] for p in preds})[:10])
    # print("Gold relations sample:", list({r for (_, _, _, r) in gold_triples})[:10])


if __name__ == "__main__":
    main()
