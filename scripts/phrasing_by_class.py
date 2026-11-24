"""Phrasing profile of each OR problem class and its link to wrong-value rate (SFT, GPT-4o).
Class-level Spearman (cue density vs wrong|runnable) and class-stratified Cochran-Mantel-Haenszel test
(>=3 relational+logical cues vs fewer), which holds training coverage fixed within each class.
Output: results/phrasing_by_class.json
"""
import json, re, csv, collections, math, sys
sys.path.insert(0, "scripts")
from lexical_cue_difficulty import CUES, B, TAGS
from error_vs_class_distribution import EN, spearman


def main():
    cls = {}
    for b in csv.DictReader(open("data/problem_or_class.csv")):
        bench, i = b["pid"].split("#"); cls[(bench, int(i))] = b["or_class"]
    rows = {k: {(r["bench"], r["i"]): r["status"] for r in map(json.loads, open(f"results/vllm/{t}_fa_rows.jsonl"))}
            for k, t in TAGS.items()}
    P = []
    for b in B:
        for it in json.load(open(f"data/benchmarks/{b}.json")):
            q = it["question"].lower(); k = (b, it["index"])
            P.append(dict({c: len(re.findall(p, q)) for c, p in CUES.items()}, key=k, cls=cls[k], words=len(q.split())))
    agg = collections.defaultdict(list)
    for p in P:
        agg[p["cls"]].append(p)
    prof = []
    for c, L in sorted(agg.items(), key=lambda x: -len(x[1])):
        if len(L) < 20:
            continue
        m = lambda f: round(sum(p[f] for p in L) / len(L), 2)
        wv = {}
        for mod in TAGS:
            run = [p for p in L if rows[mod].get(p["key"]) in ("correct", "wrong_value")]
            wv[mod] = round(100 * sum(rows[mod][p["key"]] == "wrong_value" for p in run) / len(run), 1)
        prof.append({"class": EN[c], "n": len(L), "words": m("words"), "bound": m("bound"), "relational": m("relational"),
                     "logical": m("logical"), "indexed": m("indexed"),
                     "hi_rel_log_pct": round(100 * sum(p["relational"] + p["logical"] >= 3 for p in L) / len(L), 1),
                     "wrong_sft": wv["sft"], "wrong_gpt4o": wv["gpt4o"]})
    cl = [r for r in prof if r["class"] != "Unclassified"]
    dens = [r["relational"] + r["logical"] for r in cl]
    out = {"profile": prof, "class_spearman": {m: round(spearman(dens, [r[f"wrong_{m}"] for r in cl]), 2) for m in TAGS}}
    for mod in TAGS:
        S = E = V = num = den = 0
        for L in agg.values():
            t = collections.Counter()
            for p in L:
                st = rows[mod].get(p["key"])
                if st in ("correct", "wrong_value"):
                    t[(p["relational"] + p["logical"] >= 3, st == "wrong_value")] += 1
            a, bb, cc, d = t[(True, True)], t[(True, False)], t[(False, True)], t[(False, False)]; n = a + bb + cc + d
            if n < 2:
                continue
            S += a; E += (a + bb) * (a + cc) / n; V += (a + bb) * (cc + d) * (a + cc) * (bb + d) / (n * n * (n - 1))
            num += a * d / n; den += bb * cc / n
        chi = (abs(S - E) - 0.5) ** 2 / V
        out[f"class_stratified_{mod}"] = {"MH_OR": round(num / den, 2), "p": round(math.erfc(math.sqrt(chi / 2)), 4)}
    json.dump(out, open("results/phrasing_by_class.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "profile"}))


if __name__ == "__main__":
    main()
