#!/usr/bin/env python
"""Score a results file against a reference file with the brief's three thresholds.

    python tools/evaluate_results.py reference.jsonl candidate.jsonl

Both files are the run.py output format (.jsonl or .jsonl.gz). The reference is
what you get from running every frame:  python run.py ... --samples-per-sec 0
Only (video, second) pairs of the reference are scored; a second missing from
the candidate counts as "nothing found, 0 people".
"""
import gzip
import json
import sys
from collections import Counter, defaultdict

NAMES = ["person", "car", "dog", "laptop", "cell phone"]
F1_MIN, EXACT_MIN, WITHIN1_MIN = 0.95, 0.85, 0.95


def load(path):
    rows, errors = {}, {}
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "error" in d:
                errors[d["video"]] = d["error"]
            else:
                rows[(d["video"], d["second"])] = (frozenset(d["objects"]), d["people"])
    return rows, errors


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0


def main(ref_path, cand_path):
    ref, ref_err = load(ref_path)
    cand, cand_err = load(cand_path)
    ref_videos = {v for v, _ in ref}
    tp, fp, fn = Counter(), Counter(), Counter()
    exact = within1 = n = missing = 0
    bad_per_video = defaultdict(int)
    for key, (ro, rp) in ref.items():
        if key[0] in cand_err and key[0] not in ref_err:
            pass  # candidate failed on a video the reference could read: it simply scores as empty
        co, cp = cand.get(key, (frozenset(), 0))
        if key not in cand:
            missing += 1
        for name in NAMES:
            a, b = name in ro, name in co
            if a and b:
                tp[name] += 1
            elif b:
                fp[name] += 1
            elif a:
                fn[name] += 1
        n += 1
        exact += rp == cp
        within1 += abs(rp - cp) <= 1
        if rp != cp or ro != co:
            bad_per_video[key[0]] += 1

    micro = f1(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    per_class = {c: f1(tp[c], fp[c], fn[c]) for c in NAMES}
    support = {c: tp[c] + fn[c] for c in NAMES}
    used = [c for c in NAMES if support[c] or fp[c]]  # classes that exist in the reference or were wrongly reported
    macro = sum(per_class[c] for c in used) / max(1, len(used))
    exact_r, within1_r = exact / max(n, 1), within1 / max(n, 1)

    print(f"reference: {len(ref_videos)} videos, {n} seconds | candidate: {len(cand)} seconds, "
          f"{len(cand_err)} error videos (reference: {len(ref_err)})")
    if missing:
        print(f"  !! {missing} reference seconds are missing from the candidate")
    extra = len(set(cand) - set(ref))
    if extra:
        print(f"  !! {extra} candidate seconds do not exist in the reference (wrong video length?)")
    print("object F1 per class: " + "  ".join(f"{c}={per_class[c]:.3f} (n={support[c]})" for c in NAMES))
    print(f"object F1 micro : {micro:.4f}   (need >= {F1_MIN})  {'OK' if micro >= F1_MIN else 'FAIL'}")
    print(f"object F1 macro : {macro:.4f}")
    print(f"people exact    : {exact_r:.4f}   (need >= {EXACT_MIN})  {'OK' if exact_r >= EXACT_MIN else 'FAIL'}")
    print(f"people within 1 : {within1_r:.4f}   (need >= {WITHIN1_MIN})  {'OK' if within1_r >= WITHIN1_MIN else 'FAIL'}")
    worst = sorted(bad_per_video.items(), key=lambda kv: -kv[1])[:5]
    if worst:
        print("videos with most differing seconds: " + ", ".join(f"{v} ({c})" for v, c in worst))
    ok = min(micro, macro) >= F1_MIN and exact_r >= EXACT_MIN and within1_r >= WITHIN1_MIN
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
