# convert_dev_pred_to_official.py
import argparse
import json
from typing import Dict, List, Any, Tuple


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def invert_rel_info(rel_info: Dict[str, str]) -> Dict[str, str]:
    """
    rel_info: { "P17": "country", ... }
    return:   { "country": "P17", ... }
    """
    name2pid = {}
    for pid, name in rel_info.items():
        if name in name2pid and name2pid[name] != pid:
            print(f"[WARN] duplicate relation name: {name} -> {name2pid[name]} / {pid}")
            continue
        name2pid[name] = pid
    return name2pid


def to_official_triples(
    dev_pred: List[Dict[str, Any]],
    name2pid: Dict[str, str],
    threshold: float = None,
    topk: int = None,
) -> List[Dict[str, Any]]:

    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int, int, str]] = set()

    missing_rel = 0

    for doc in dev_pred:
        title = doc["title"]
        for pair in doc.get("relations", []):
            h_id = pair["h_id"]
            t_id = pair["t_id"]
            rel_list = pair.get("relations", [])

            # sort by p_yes if available
            if rel_list and isinstance(rel_list[0], dict) and "p_yes" in rel_list[0]:
                rel_list = sorted(rel_list, key=lambda x: x.get("p_yes", 0.0), reverse=True)
                if topk is not None:
                    rel_list = rel_list[:topk]
                if threshold is not None:
                    rel_list = [x for x in rel_list if x.get("p_yes", 0.0) >= threshold]

            for r_item in rel_list:
                rel_name = r_item["relation"] if isinstance(r_item, dict) else str(r_item)
                pid = name2pid.get(rel_name)
                if pid is None:
                    missing_rel += 1
                    continue

                key = (title, h_id, t_id, pid)
                if key in seen:
                    continue
                seen.add(key)

                out.append({
                    "title": title,
                    "h_idx": h_id,
                    "t_idx": t_id,
                    "r": pid
                })

    if missing_rel > 0:
        print(f"[WARN] {missing_rel} relations not found in rel_info (skipped).")

    print(f"[OK] total triples: {len(out)}")
    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--pred",
        default="../ablation/results/with-doc_dev_pred_docred.json",
        help="LoRA prediction file (default: dev_F1_pred.json)"
    )
    ap.add_argument(
        "--rel_info",
        default="../datasets/docred/rel_info.json",
        help="relation mapping Pxx -> name (default: datasets/redocred/rel_info.json)"
    )
    ap.add_argument(
        "--out",
        default="with-doc_dev_pred_docred_official.json",
        help="output file in official Re-DocRED format"
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="optional p_yes threshold (default: None, keep all)"
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=None,
        help="optional top-k relations per entity pair (default: None)"
    )

    args = ap.parse_args()

    dev_pred = load_json(args.pred)
    rel_info = load_json(args.rel_info)
    name2pid = invert_rel_info(rel_info)

    triples = to_official_triples(
        dev_pred=dev_pred,
        name2pid=name2pid,
        threshold=args.threshold,
        topk=args.topk,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False, indent=2)

    print(f"[DONE] wrote {args.out}")


if __name__ == "__main__":
    main()
