import json


def load_redocred_docs(path):
    """Load Re-DocRED json and build title -> doc mapping"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    title2doc = {}
    for doc in data:
        title2doc[doc["title"]] = doc
    return title2doc


def generate_step15_json(
    step1_path,
    redocred_test_path,
    type_constraint_path,
    output_path
):
    """
    Stage 1.5: Entity Type Constraint
    + include sentences for Stage 2 (LLM-friendly)
    """

    # -------- load files --------
    with open(step1_path, "r", encoding="utf-8") as f:
        step1_data = json.load(f)

    title2doc = load_redocred_docs(redocred_test_path)

    with open(type_constraint_path, "r", encoding="utf-8") as f:
        type_constraints = json.load(f)

    step15_results = []

    for item in step1_data:
        title = item["title"]
        doc = title2doc[title]
        vertex_set = doc["vertexSet"]

        # -------- sentences (token list -> string) --------
        sentences = [" ".join(sent) for sent in doc["sents"]]

        pairs_out = []

        for h_id, t_id in item["entity_pairs"]:
            h_entity = vertex_set[h_id][0]
            t_entity = vertex_set[t_id][0]

            h_name = h_entity["name"]
            t_name = t_entity["name"]

            type_h = h_entity["type"]
            type_t = t_entity["type"]

            # key = f"{type_h}_{type_t}"
            key = "all"
            candidate_relations = type_constraints.get(key, [])

            if not candidate_relations:
                continue

            pairs_out.append({
                "h": h_name,
                "t": t_name,
                "h_id": h_id,
                "t_id": t_id,
                "type_h": type_h,
                "type_t": type_t,
                "candidate_relations": candidate_relations
            })

        step15_results.append({
            "title": title,
            "sentences": sentences,
            "pairs": pairs_out
        })

    # -------- save --------
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(step15_results, f, ensure_ascii=False, indent=2)

    print(f"[Stage1.5] step1.5.json saved to {output_path}")
    print(f"[Stage1.5] total pairs:",
          sum(len(x["pairs"]) for x in step15_results))



if __name__ == "__main__":

    # -------- paths (按你的工程结构自行调整) --------
    step1_path = "../Stage1/results/docred/dev-best-F1-0.3-docred.json"
    # step1_path = "../Stage1/results/redocred/dev-best-recall-0.27.json"
    redocred_test_path = "../datasets/docred/dev.json"
    # redocred_test_path = "../datasets/redocred/test_revised.json"
    type_constraint_path = "../datasets/docred/relation/基于count/docred-entity_type_pair_relations_minK10.json"
    # output_path = "redocred/dev-best-recall-0.27-type-to2-k10.json"
    output_path = "docred/all-relations.json"

    # -------- run Stage 1.5 --------
    generate_step15_json(
        step1_path=step1_path,
        redocred_test_path=redocred_test_path,
        type_constraint_path=type_constraint_path,
        output_path=output_path
    )

    # -------- optional sanity check --------
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_docs = len(data)
    total_pairs = sum(len(d["pairs"]) for d in data)
    avg_pairs = total_pairs / total_docs if total_docs > 0 else 0

    print("[Stage1.5] sanity check")
    print(f"  documents      : {total_docs}")
    print(f"  total pairs    : {total_pairs}")
    print(f"  avg pairs/doc  : {avg_pairs:.2f}")


