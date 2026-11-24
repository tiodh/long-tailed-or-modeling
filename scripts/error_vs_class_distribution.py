"""Error rate per OR problem class vs the class's share of the training data.
Inputs: logs/or_class_cross_table.json (training share, same keyword classifier), data/problem_or_class.csv
        (class per benchmark problem), results/vllm/<tag>_fa_rows.jsonl (per-problem status).
Output: results/error_vs_class_distribution.json
Usage : python3 scripts/error_vs_class_distribution.py [tag]   (default sft_instruct_8k)
"""
import json, csv, sys, collections

TAG = sys.argv[1] if len(sys.argv) > 1 else "sft_instruct_8k"
MAP = {"job_shop": "Job-shop / makespan", "workforce_scheduling": "Penjadwalan tenaga kerja",
       "transportation": "Transportasi / distribusi", "facility_location": "Lokasi fasilitas",
       "diet_blending": "Diet / blending", "knapsack_selection": "Knapsack / seleksi",
       "bin_packing": "Bin packing / cutting stock", "portfolio": "Portofolio / investasi",
       "inventory_lotsizing": "Inventori / lot sizing", "assignment": "Penugasan",
       "set_covering": "Cakupan (set covering)", "geometric_design": "Desain geometris",
       "production_planning": "Perencanaan produksi", "resource_allocation": "Alokasi sumber daya",
       "unclassified": "Tak terklasifikasi"}
EN = {"job_shop": "Job-shop scheduling", "workforce_scheduling": "Workforce scheduling",
      "transportation": "Transportation", "facility_location": "Facility location",
      "diet_blending": "Diet / blending", "knapsack_selection": "Knapsack", "bin_packing": "Bin packing",
      "portfolio": "Portfolio", "inventory_lotsizing": "Inventory / lot sizing", "assignment": "Assignment",
      "set_covering": "Set covering", "geometric_design": "Geometric design",
      "production_planning": "Production planning", "resource_allocation": "Resource allocation",
      "unclassified": "Unclassified"}


def rank(x):
    s = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and x[s[j + 1]] == x[s[i]]:
            j += 1
        for k in range(i, j + 1):
            r[s[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b); n = len(a); ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** .5; vb = sum((y - mb) ** 2 for y in rb) ** .5
    return cov / (va * vb)


def main():
    cross = json.load(open("logs/or_class_cross_table.json"))
    train_total = cross["sets"]["_train"]["total"]
    rows = {(r["bench"], r["i"]): r for r in map(json.loads, open(f"results/vllm/{TAG}_fa_rows.jsonl"))}
    agg = collections.defaultdict(collections.Counter)
    for b in csv.DictReader(open("data/problem_or_class.csv")):
        bench, idx = b["pid"].split("#")
        if (bench, int(idx)) not in rows:  # gpt4o_bench has 2021/2023 (two API failures)
            continue
        st = rows[(bench, int(idx))]["status"]
        c = agg[b["or_class"]]
        c["n"] += 1
        c["correct"] += st == "correct"
        c["wrong"] += st == "wrong_value"
        c["fail"] += st not in ("correct", "wrong_value")
    out = []
    for k, c in agg.items():
        tr = cross["classes"][MAP[k]]["_train"]
        out.append({"class": EN[k], "train_n": tr["n"], "train_pct": round(100 * tr["n"] / train_total, 2),
                    "bench_n": c["n"], "acc": round(100 * c["correct"] / c["n"], 1),
                    "wrong": round(100 * c["wrong"] / c["n"], 1), "fail": round(100 * c["fail"] / c["n"], 1)})
    out.sort(key=lambda r: -r["train_pct"])
    print(f"{'class':24s}{'train%':>8s}{'train n':>9s}{'bench n':>9s}{'acc':>7s}{'wrong':>7s}{'fail':>7s}")
    for r in out:
        print(f"{r['class']:24s}{r['train_pct']:8.2f}{r['train_n']:9d}{r['bench_n']:9d}{r['acc']:7.1f}{r['wrong']:7.1f}{r['fail']:7.1f}")
    cl = [r for r in out if r["class"] != "Unclassified" and r["bench_n"] >= 20]
    x = [r["train_pct"] for r in cl]
    corr = {m: round(spearman(x, [r[m] for r in cl]), 2) for m in ("acc", "wrong", "fail")}
    print(f"\nSpearman vs training share (classified, bench n>=20, k={len(cl)}):", corr)
    json.dump({"tag": TAG, "train_total": train_total, "rows": out, "spearman_n20": corr,
               "classes_in_corr": [r["class"] for r in cl]},
              open("results/error_vs_class_distribution.json", "w"), indent=1)


if __name__ == "__main__":
    main()
