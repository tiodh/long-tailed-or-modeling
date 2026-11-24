"""Which supervision (reasoning) data beats ReSocratic-style reasoning? Offline analysis of ablation arms.

Inputs  : results/ablation/abl_<arm>_bench/{bench}.jsonl (generations) + abl_<arm>_bench_fa_rows.jsonl (FA scores)
          results/vllm/sft_instruct_8k_fa_rows.jsonl (full-ReSocratic SFT, reference point)
Outputs : results/ablation/reasoning_data_analysis.json + printed tables
Usage   : python3 scripts/reasoning_data_analysis.py arm0 arm1 arm2 [arm3 armR0 armR1]
"""
import json, os, re, sys, math, collections

BENCH = ["nl4opt", "mamo_easy", "mamo_complex", "optibench", "industryor", "optmath"]
ARMS = sys.argv[1:] or ["arm0", "arm1", "arm2"]
D = "results/ablation"


def rows(path):
    return {(r["bench"], r["i"]): r for r in map(json.loads, open(path))}


def gens(arm):
    g = {}
    for b in BENCH:
        for ln in open(f"{D}/abl_{arm}_bench/{b}.jsonl"):
            r = json.loads(ln)
            if "i" in r:
                g[(b, r["i"])] = r
    return g


def mcnemar(a, b, keys):
    """exact two-sided binomial on discordant pairs (b01: a wrong b right)."""
    n01 = sum(1 for k in keys if not a[k]["correct"] and b[k]["correct"])
    n10 = sum(1 for k in keys if a[k]["correct"] and not b[k]["correct"])
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n01, n10, min(1.0, 2 * p)


def block(text, tag):
    m = re.search(rf"\[{tag}\](.*?)\[/{tag}\]", text, re.S)
    return m.group(1) if m else ""


def feats(text):
    rea, frm = block(text, "REASONING"), block(text, "FORMULA")
    code = text.split("[CODE]", 1)[-1]
    n_cons_code = len(re.findall(r"addCons\(", code))
    n_cons_frm = len(re.findall(r"^\s*Constraint\b", frm, re.M))
    n_cons_rea = len(re.findall(r"^\s*-\s", rea, re.M))
    return {
        "rea_words": len(rea.split()),
        "n_quotes": frm.count('<- "'),
        "n_cons_code": n_cons_code,
        "n_cons_formula": n_cons_frm,
        "cons_match": (n_cons_frm == 0) or (n_cons_code - 1 <= n_cons_frm <= n_cons_code + 1),
        "has_verify": bool(re.search(r"\[VERIF|verify|check:", text, re.I)),
    }


def main():
    R = {a: rows(f"{D}/abl_{a}_bench_fa_rows.jsonl") for a in ARMS}
    R["sft_full"] = rows("results/vllm/sft_instruct_8k_fa_rows.jsonl")
    keys = sorted(set.intersection(*[set(v) for v in R.values()]))
    out = {"n": len(keys), "acc": {}, "mcnemar": {}, "status": {}, "features": {}}

    print(f"n={len(keys)}\n{'arm':10s}" + "".join(f"{b[:9]:>10s}" for b in BENCH) + f"{'MICRO':>9s}")
    for a, rr in R.items():
        per = {}
        for b in BENCH:
            kb = [k for k in keys if k[0] == b]
            per[b] = round(100 * sum(rr[k]["correct"] for k in kb) / len(kb), 2)
        per["micro"] = round(100 * sum(rr[k]["correct"] for k in keys) / len(keys), 2)
        out["acc"][a] = per
        st = collections.Counter(rr[k]["status"] for k in keys)
        out["status"][a] = dict(st)
        print(f"{a:10s}" + "".join(f"{per[b]:10.2f}" for b in BENCH) + f"{per['micro']:9.2f}")

    print("\nMcNemar (a→b: n_gain, n_loss, p)")
    pairs = [(x, y) for i, x in enumerate(list(R)) for y in list(R)[i + 1:]]
    for a, b in pairs:
        g, l, p = mcnemar(R[a], R[b], keys)
        out["mcnemar"][f"{a}->{b}"] = {"gain": g, "loss": l, "p": round(p, 4)}
        print(f"  {a:9s}→{b:9s} +{g:4d} −{l:4d}  p={p:.4f}")

    print("\nstatus (non-correct share of all problems, %)")
    for a in R:
        st = out["status"][a]
        print(f"  {a:9s} " + "  ".join(f"{s}={100 * c / len(keys):.1f}" for s, c in sorted(st.items()) if s != "correct"))

    # generation features vs correctness, per arm that has reasoning
    print("\nfeatures of generated output (correct vs wrong)")
    for a in ARMS:
        G = gens(a)
        F = {k: feats(G[k]["text"]) | {"n_tok": G[k].get("n_tok", 0)} for k in keys if k in G}
        res = {}
        for f in ["rea_words", "n_quotes", "n_cons_code", "n_cons_formula", "n_tok"]:
            c = [F[k][f] for k in F if R[a][k]["correct"]]
            w = [F[k][f] for k in F if not R[a][k]["correct"]]
            res[f] = {"correct_mean": round(sum(c) / max(1, len(c)), 1), "wrong_mean": round(sum(w) / max(1, len(w)), 1)}
        if any(F[k]["n_cons_formula"] for k in F):
            m = [k for k in F if F[k]["cons_match"]]
            mm = [k for k in F if not F[k]["cons_match"]]
            res["acc_if_formula_matches_code"] = round(100 * sum(R[a][k]["correct"] for k in m) / max(1, len(m)), 2)
            res["acc_if_formula_mismatch_code"] = round(100 * sum(R[a][k]["correct"] for k in mm) / max(1, len(mm)), 2)
            res["n_mismatch"] = len(mm)
        out["features"][a] = res
        print(f"  {a}: " + json.dumps(res))

    json.dump(out, open(f"{D}/reasoning_data_analysis.json", "w"), indent=1)
    print(f"\n→ {D}/reasoning_data_analysis.json")


if __name__ == "__main__":
    main()
