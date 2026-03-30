import json
from collections import defaultdict
from typing import List, Dict, Tuple
import random



def load_redocred(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_rel_info(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        rel_map = json.load(f)
    return rel_map


def build_type_constraint_map(docs: List[Dict]) -> Dict[Tuple[str, str], set]:
    """
    (type_h, type_t) -> set(relations)
    """
    type_map = defaultdict(set)

    for doc in docs:
        vertex_set = doc["vertexSet"]
        for label in doc.get("labels", []):
            h = label["h"]
            t = label["t"]
            r = label["r"]

            type_h = vertex_set[h][0]["type"]
            type_t = vertex_set[t][0]["type"]

            type_map[(type_h, type_t)].add(r)

    return type_map

def collect_entity_spans(vertex_set, ent_id, sent_id):
    spans = []
    for m in vertex_set[ent_id]:
        if m["sent_id"] == sent_id:
            spans.append((m["pos"][0], m["pos"][1]))
    return spans


def mark_sentence(tokens, h_spans, t_spans):
    marks = []
    for s, e in h_spans:
        marks.append((s, e, "H"))
    for s, e in t_spans:
        marks.append((s, e, "T"))

    # 按 start 排序，长 span 优先
    marks.sort(key=lambda x: (x[0], -(x[1]-x[0])))

    result = []
    idx = 0
    for s, e, tag in marks:
        if s < idx:
            continue
        result.extend(tokens[idx:s])
        result.append(f"[[{tag}]]")
        result.extend(tokens[s:e])
        result.append(f"[[/{tag}]]")
        idx = e
    result.extend(tokens[idx:])
    return " ".join(result)


def build_evidence_text(doc, h_id, t_id, evidence_sent_ids):
    sentences = doc["sents"]
    vertex_set = doc["vertexSet"]

    marked_sents = []
    for sid in evidence_sent_ids:
        tokens = sentences[sid]

        h_spans = collect_entity_spans(vertex_set, h_id, sid)
        t_spans = collect_entity_spans(vertex_set, t_id, sid)

        sent = mark_sentence(tokens, h_spans, t_spans)
        marked_sents.append(sent)

    return "\n".join(marked_sents)

def build_stage2_samples(
    docs: List[Dict],
    type_map: Dict[Tuple[str, str], set],
    rel_map: Dict[str, str],          # 【新增】
    max_no_per_yes: int = 3,
    max_pairs_per_doc: int = 30
) -> List[Dict]:

    samples = []

    for doc in docs:
        title = doc["title"]
        vertex_set = doc["vertexSet"]
        sentences = doc["sents"]       # 【新增】
        labels = doc.get("labels", [])

        pair2rels = defaultdict(list)
        pair2evidence = defaultdict(list)

        for lab in labels:
            pair = (lab["h"], lab["t"])
            pair2rels[pair].append(lab["r"])
            pair2evidence[pair].append(lab["evidence"])

        used_pairs = list(pair2rels.keys())
        random.shuffle(used_pairs)
        used_pairs = used_pairs[:max_pairs_per_doc]

        for (h_id, t_id) in used_pairs:
            gold_rels = set(pair2rels[(h_id, t_id)])

            # =====【关键修改 1：空 evidence → 全文档】=====
            evidence_ids = pair2evidence[(h_id, t_id)][0]
            if not evidence_ids:
                evidence_ids = list(range(len(sentences)))
            # =============================================

            evidence_text = build_evidence_text(
                doc, h_id, t_id, evidence_ids
            )

            head_name = vertex_set[h_id][0]["name"]
            tail_name = vertex_set[t_id][0]["name"]

            type_h = vertex_set[h_id][0]["type"]
            type_t = vertex_set[t_id][0]["type"]

            candidates = list(type_map.get((type_h, type_t), []))
            neg_rels = [r for r in candidates if r not in gold_rels]
            random.shuffle(neg_rels)
            neg_rels = neg_rels[:max_no_per_yes * len(gold_rels)]

            # ===== Yes samples =====
            for r in gold_rels:
                if r not in rel_map:
                    continue  # 安全检查
                samples.append({
                    "title": title,
                    "head": head_name,
                    "tail": tail_name,
                    "relation": rel_map[r],   # 【关键修改 2】
                    "evidence_text": evidence_text,
                    "label": "Yes"
                })

            # ===== No samples =====
            for r in neg_rels:
                if r not in rel_map:
                    continue
                samples.append({
                    "title": title,
                    "head": head_name,
                    "tail": tail_name,
                    "relation": rel_map[r],
                    "evidence_text": evidence_text,
                    "label": "No"
                })

    return samples


def save_jsonl(samples: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # train_path = "../datasets/docred/train_annotated.json"
    # rel_info_path = "../datasets/docred/rel_info.json"   # 【新增】
    # out_path = "stage3_lora_train-docred.jsonl"
    #
    # docs = load_redocred(train_path)
    # type_map = build_type_constraint_map(docs)
    # rel_map = load_rel_info(rel_info_path)   # 【新增】
    #
    # samples = build_stage2_samples(
    #     docs,
    #     type_map,
    #     rel_map,
    #     max_no_per_yes=2,
    #     max_pairs_per_doc=20
    # )
    #
    # print(f"Total samples: {len(samples)}")
    # save_jsonl(samples, out_path)
    #
    # import json
    # from collections import Counter
    #
    # path = "stage3_lora_train.jsonl"
    #
    # counter = Counter()
    #
    # with open(path, "r", encoding="utf-8") as f:
    #     for line in f:
    #         item = json.loads(line)
    #         counter[item["label"]] += 1
    #
    # print("Label distribution:")
    # for k, v in counter.items():
    #     print(f"{k}: {v}")
    #
    # total = sum(counter.values())
    # print("\nRatio:")
    # for k, v in counter.items():
    #     print(f"{k}: {v / total:.3f}")

    import json

    path = "../ablation/with-doc.json"

    lengths = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # 用 split() 近似 token 数（训练前足够）
            lengths.append(len(item["evidence_text"].split()))


    lengths.sort()

    n = len(lengths)
    print("Total samples:", n)
    print("Min length:", lengths[0])
    print("Max length:", lengths[-1])
    print("Average length:", sum(lengths) / n)
    print("P90:", lengths[int(0.90 * n)])
    print("P95:", lengths[int(0.95 * n)])
    print("P99:", lengths[int(0.99 * n)])

