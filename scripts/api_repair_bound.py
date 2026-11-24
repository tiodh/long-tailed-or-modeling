"""Upper bound: how many ablation-arm failures are only a PySCIPOpt API slip?

Rewrites solver-API hallucinations (Gurobi/other idioms) into PySCIPOpt calls, re-executes,
and re-scores with the FA rule (rel err < 1e-6). Only the result-reading API is touched,
never the formulation, so a gain = formulation already right, code interface wrong.
Usage: .venv-gpt/bin/python scripts/api_repair_bound.py arm0 arm1 arm2 ...
"""
import json, re, sys, os, collections
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.getcwd())
from or_rlvr.reward.solver import SCIPSandbox
from or_rlvr.evaluation.benchmarks import _extract_ground_truth

SB = SCIPSandbox(timeout=30)
B = ["nl4opt", "mamo_easy", "mamo_complex", "optibench", "industryor", "optmath"]
FIX = [
    (r"(\w+(?:\[[^\]]*\])?)\.getValue\(\)", r"model.getVal(\1)"),
    (r"(\w+(?:\[[^\]]*\])?)\.getVal\(\)", r"model.getVal(\1)"),
    (r"(\w+(?:\[[^\]]*\])?)\.x\b", r"model.getVal(\1)"),
    (r"(\w+(?:\[[^\]]*\])?)\.X\b", r"model.getVal(\1)"),
    (r"model\.getObjValue\(\)", "model.getObjVal()"),
    (r"model\.objVal\b", "model.getObjVal()"),
    (r"model\.ObjVal\b", "model.getObjVal()"),
]


def repair(code):
    for a, b in FIX:
        code = re.sub(a, b, code)
    return code


def one(job):
    text, gt = job
    code = SB.extract_code_block(text)
    if not code or gt is None:
        return False
    res = SB._execute_with_timeout(repair(code))
    ov = res.get("objval") if res else None
    return ov is not None and abs(ov - gt) / max(abs(gt), 1.0) < 1e-6


def main():
    out = {}
    for arm in sys.argv[1:]:
        R = {(r["bench"], r["i"]): r for r in map(json.loads, open(f"results/ablation/abl_{arm}_bench_fa_rows.jsonl"))}
        keys, jobs = [], []
        for b in B:
            items = json.load(open(f"data/benchmarks/{b}.json"))
            for ln in open(f"results/ablation/abl_{arm}_bench/{b}.jsonl"):
                d = json.loads(ln)
                if "_meta" in d or R[(b, d["i"])]["status"] != "no_value":
                    continue
                if repair(d["text"]) == d["text"]:
                    continue
                try:
                    gt = _extract_ground_truth(items[d["i"]])
                except Exception:
                    gt = None
                keys.append((b, d["i"])); jobs.append((d["text"], gt))
        with ProcessPoolExecutor(max_workers=10) as ex:
            ok = list(ex.map(one, jobs, chunksize=4))
        base = sum(r["correct"] for r in R.values())
        gain = sum(ok)
        per = collections.Counter(k[0] for k, o in zip(keys, ok) if o)
        out[arm] = {"base": base, "touched": len(jobs), "recovered": gain,
                    "base_pct": round(100 * base / len(R), 2), "repaired_pct": round(100 * (base + gain) / len(R), 2),
                    "recovered_by_bench": dict(per)}
        print(arm, json.dumps(out[arm]), flush=True)
    json.dump(out, open("results/ablation/api_repair_bound.json", "w"), indent=1)


if __name__ == "__main__":
    main()
