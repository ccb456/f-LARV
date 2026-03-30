import json
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional


# ----------------------------
# Evidence sentence selection
# ----------------------------

def extract_evidence_sentence_ids(
    raw_doc: Dict[str, Any],
    h_id: int,
    t_id: int,
    K: int = 12,
    window: int = 1,             # fallback window (non-cooccur)
    use_span: bool = True,
    cooccur_window: int = 0,     # cooccur exists: usually 0
    max_span_len: int = 8,       # NEW: cap for span length when cooccur absent
) -> List[int]:
    num_sents = len(raw_doc["sents"])

    def mention_sent_ids(eid: int) -> List[int]:
        sids = []
        for m in raw_doc["vertexSet"][eid]:
            if "sent_id" in m:
                sids.append(int(m["sent_id"]))
        return sids

    h_sids = mention_sent_ids(h_id)
    t_sids = mention_sent_ids(t_id)

    # ✅ safer: if either side missing, no evidence
    if not h_sids or not t_sids:
        return []

    h_set = set(h_sids)
    t_set = set(t_sids)

    # -------------------------
    # co-occur priority branch
    # -------------------------
    cooccur = sorted(h_set & t_set)
    if cooccur:
        candidate_ids = set(cooccur)

        for sid in cooccur:
            for d in range(1, cooccur_window + 1):
                if 0 <= sid - d < num_sents:
                    candidate_ids.add(sid - d)
                if 0 <= sid + d < num_sents:
                    candidate_ids.add(sid + d)

        candidate_ids = sorted(candidate_ids)

        if len(candidate_ids) <= K:
            return candidate_ids

        base = cooccur  # for boundary bonus

    else:
        # -------------------------
        # fallback branch (no cooccur)
        # -------------------------
        base = sorted(h_set | t_set)
        if not base:
            return []

        candidate_ids = None

        if use_span:
            lo, hi = min(base), max(base)
            span_ids = list(range(lo, hi + 1))

            # ✅ NEW: span guardrail (avoid huge spans)
            if len(span_ids) <= max_span_len:
                if len(span_ids) <= K:
                    return span_ids
                candidate_ids = span_ids

        if candidate_ids is None:
            candidate_ids = base

        expanded = set(candidate_ids)
        for sid in base:
            for d in range(1, window + 1):
                if 0 <= sid - d < num_sents:
                    expanded.add(sid - d)
                if 0 <= sid + d < num_sents:
                    expanded.add(sid + d)

        candidate_ids = sorted(expanded)

        if len(candidate_ids) <= K:
            return candidate_ids

    # -------------------------
    # Top-K scoring
    # -------------------------
    h_cnt = Counter(h_sids)
    t_cnt = Counter(t_sids)

    REL_HINT_WORDS = {
        "born", "birth", "died", "death",
        "country", "citizen", "nationality",
        "located", "location", "part", "member",
        "capital", "head", "government", "state",
        "coach", "team", "won", "competed",
        "inception", "start", "end", "time"
    }

    left, right = min(base), max(base)

    def score(sent_id: int) -> float:
        toks = [w.lower() for w in raw_doc["sents"][sent_id]]
        has_h = sent_id in h_set
        has_t = sent_id in t_set
        both = has_h and has_t

        s = 0.0
        if both:
            s += 2.0
        elif has_h or has_t:
            s += 1.0

        s += 0.2 * min(h_cnt.get(sent_id, 0), 3)
        s += 0.2 * min(t_cnt.get(sent_id, 0), 3)

        if any(w in REL_HINT_WORDS for w in toks):
            s += 0.5

        if sent_id == left or sent_id == right:
            s += 0.3

        return s

    ranked = sorted(candidate_ids, key=lambda sid: score(sid), reverse=True)
    chosen = sorted(ranked[:K])
    return chosen


def build_full_document_evidence(
    raw_doc: Dict[str, Any],
    h_id: int,
    t_id: int,
) -> str:
    """
    Build full-document evidence with entity markers, without [S#] prefixes
    """
    marked_sents = []

    for sid, tokens in enumerate(raw_doc["sents"]):
        spans = []
        spans += _collect_spans_for_sentence(raw_doc, h_id, "H", sid)
        spans += _collect_spans_for_sentence(raw_doc, t_id, "T", sid)

        marked_tokens = mark_sentence_tokens(tokens, spans)
        sent_text = " ".join(marked_tokens)
        marked_sents.append(sent_text)

    return "\n".join(marked_sents)




# ----------------------------
# Entity marking in sentences
# ----------------------------

def _collect_spans_for_sentence(raw_doc: Dict[str, Any], entity_id: int, label: str, sent_id: int) -> List[Tuple[int, int, str]]:
    spans = []
    for m in raw_doc["vertexSet"][entity_id]:
        if int(m.get("sent_id", -1)) == sent_id and "pos" in m:
            start, end = m["pos"]
            spans.append((int(start), int(end), label))
    return spans


def _merge_and_filter_overlaps(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Simple overlap policy:
    - Sort by start asc, longer first
    - Greedily keep non-overlapping spans, skip overlaps
    """
    spans = [s for s in spans if s[0] >= 0 and s[1] > s[0]]
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    kept = []
    last_end = -1
    for s, e, lab in spans:
        if s >= last_end:
            kept.append((s, e, lab))
            last_end = e
    return kept


def mark_sentence_tokens(tokens: List[str], spans: List[Tuple[int, int, str]]) -> List[str]:
    spans = _merge_and_filter_overlaps(spans)
    if not spans:
        return tokens[:]

    out = []
    i = 0
    span_i = 0
    while i < len(tokens):
        if span_i < len(spans) and i == spans[span_i][0]:
            s, e, lab = spans[span_i]
            out.append(f"[[{lab}]]")
            out.extend(tokens[s:e])
            out.append(f"[[/{lab}]]")
            i = e
            span_i += 1
        else:
            out.append(tokens[i])
            i += 1
    return out


def build_marked_evidence(
    raw_doc: Dict[str, Any],
    evidence_sent_ids: List[int],
    h_id: int,
    t_id: int,
    include_prefix: bool = True,
) -> Tuple[List[str], str]:
    """
    Return:
    - evidence_sentences_marked: List[str] 每句已标记
    - evidence_text_marked: str 拼接后的文本块
    """
    marked_sents = []
    for sid in evidence_sent_ids:
        tokens = raw_doc["sents"][sid]

        spans = []
        spans += _collect_spans_for_sentence(raw_doc, h_id, "H", sid)
        spans += _collect_spans_for_sentence(raw_doc, t_id, "T", sid)

        marked_tokens = mark_sentence_tokens(tokens, spans)
        sent_text = " ".join(marked_tokens)

        if include_prefix:
            marked_sents.append(f"[S{sid}] {sent_text}")
        else:
            marked_sents.append(sent_text)

    evidence_text = "\n".join(marked_sents)
    return marked_sents, evidence_text


# ----------------------------
# Adapter main
# ----------------------------

def adapt_step15_for_stage2(
    raw_docs: List[Dict[str, Any]],
    step15_docs: List[Dict[str, Any]],
    K: int = 12,
    window: int = 1,
    use_span: bool = True,
    include_prefix: bool = True,
    cooccur_window: int = 0,   # NEW
    max_span_len=8,
) -> List[Dict[str, Any]]:
    """
    raw_docs: 原始 Re-DocRED json list
    step15_docs: 你的 Stage1.5 输出 list (title/sentences/pairs)
    """
    raw_by_title = {d["title"]: d for d in raw_docs}

    out_docs = []
    missing_titles = 0

    for d in step15_docs:
        title = d["title"]
        raw = raw_by_title.get(title)
        if raw is None:
            missing_titles += 1
            # 保留原样，但标记为 missing
            dd = dict(d)
            dd["_adapter_warning"] = "missing_raw_doc_by_title"
            out_docs.append(dd)
            continue

        dd = {
            "title": title,
            "sentences": d.get("sentences", []),  # 你现有的字符串句子（保留）
            "pairs": []
        }

        for p in d.get("pairs", []):
            h_id = int(p["h_id"])
            t_id = int(p["t_id"])

            evidence_ids = extract_evidence_sentence_ids(
                raw_doc=raw,
                h_id=h_id,
                t_id=t_id,
                K=K,
                window=window,
                use_span=use_span,
                cooccur_window=cooccur_window,
                max_span_len=max_span_len,
            )


            marked_sents, marked_text = build_marked_evidence(
                raw_doc=raw,
                evidence_sent_ids=evidence_ids,
                h_id=h_id,
                t_id=t_id,
                include_prefix=include_prefix,
            )

            pp = dict(p)
            pp["evidence_sent_ids"] = evidence_ids
            pp["evidence_sentences_marked"] = marked_sents
            pp["evidence_text_marked"] = marked_text
            dd["pairs"].append(pp)

            # marked_text = build_full_document_evidence(
            #     raw_doc=raw,
            #     h_id=h_id,
            #     t_id=t_id,
            # )

            # pp = dict(p)
            # pp["evidence_text_marked"] = marked_text
            # dd["pairs"].append(pp)

        out_docs.append(dd)

    if missing_titles > 0:
        print(f"[WARN] {missing_titles} titles in step1.5 not found in raw docs by title.")

    return out_docs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../datasets/docred/dev.json", help="path to original Re-DocRED json (with sents/vertexSet)")
    ap.add_argument("--step15", default="docred/all-relations.json", help="path to your step1.5 json")
    ap.add_argument("--out", default="all-relations-input.json", help="output path for adapted json")
    ap.add_argument("--K", type=int, default=4, help="K for evidence sentence extraction")
    ap.add_argument("--window", type=int, default=1, help="window used when span exists (default: 1 => keep only span sentences)")
    ap.add_argument("--cooccur_window", type=int, default=0,
                    help="window used when co-occur exists (default: 0 => keep only co-occur sentences)")
    ap.add_argument("--use_span", action="store_true", default=True)
    ap.add_argument("--no_span", dest="use_span", action="store_false")
    ap.add_argument("--no_prefix", action="store_false", help="do not add [S#] prefix in evidence sentences")
    ap.add_argument("--max_span_len", type=int, default=8)

    args = ap.parse_args()

    with open(args.raw, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)
    with open(args.step15, "r", encoding="utf-8") as f:
        step15_docs = json.load(f)

    adapted = adapt_step15_for_stage2(
        raw_docs=raw_docs,
        step15_docs=step15_docs,
        K=args.K,
        window=args.window,
        use_span=args.use_span,
        include_prefix=not args.no_prefix,
        cooccur_window=args.cooccur_window,  # NEW
        max_span_len=args.max_span_len,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(adapted, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved adapted stage2 input to: {args.out}")


if __name__ == "__main__":
    main()
