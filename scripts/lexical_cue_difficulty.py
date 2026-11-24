"""Do constraint phrasings that encode relations between quantities make formulation harder?
Counts lexical cues in each benchmark problem statement and relates them to the wrong-value rate
among runnable programs (correct + wrong_value), for SFT and GPT-4o.
Output: results/lexical_cue_difficulty.json
"""
import json, re, collections

B = ["nl4opt", "mamo_easy", "mamo_complex", "optibench", "industryor", "optmath"]
CUES = {
    "bound": r"\b(at least|at most|no more than|no less than|not exceed|exceeds?|up to|minimum of|maximum of|limited to|cannot exceed)\b",
    "relational": r"\b(times as (many|much)|twice|thrice|half (as|of)|ratio|proportion|fraction|percent|%|more than .{1,40}? than|less than .{1,40}? than|at least \w+ times|at most \w+ times|as many as)\b|%",
    "logical": r"\b(if|either|only if|unless|whether|or both|cannot both|mutually exclusive|at least one|exactly one|must be selected|fixed cost|setup cost)\b",
    "indexed": r"\b(each|every|for all|per)\b",
}
TAGS = {"sft": "sft_instruct_8k", "gpt4o": "gpt4o_bench"}


def main():
    rows = {k: {(r["bench"], r["i"]): r["status"] for r in map(json.loads, open(f"results/vllm/{t}_fa_rows.jsonl"))}
            for k, t in TAGS.items()}
    probs = []
    for b in B:
        for it in json.load(open(f"data/benchmarks/{b}.json")):
            q = it["question"].lower()
            f = {c: len(re.findall(p, q)) for c, p in CUES.items()}
            f["key"] = (b, it["index"]); f["words"] = len(q.split())
            probs.append(f)
    out = {}
    for model in TAGS:
        R = rows[model]
        res = {}
        for cue in ("bound", "relational", "logical", "indexed"):
            bins = collections.defaultdict(lambda: [0, 0])
            for p in probs:
                st = R.get(p["key"])
                if st not in ("correct", "wrong_value"):
                    continue
                n = p[cue]; k = "0" if n == 0 else "1" if n == 1 else "2" if n == 2 else "3+"
                bins[k][0] += 1; bins[k][1] += st == "wrong_value"
            res[cue] = {k: {"n": v[0], "wrong_pct": round(100 * v[1] / v[0], 1)} for k, v in sorted(bins.items())}
        # combined: relational + logical cues
        bins = collections.defaultdict(lambda: [0, 0])
        for p in probs:
            st = R.get(p["key"])
            if st not in ("correct", "wrong_value"):
                continue
            n = p["relational"] + p["logical"]; k = "0" if n == 0 else "1-2" if n <= 2 else "3-4" if n <= 4 else "5+"
            bins[k][0] += 1; bins[k][1] += st == "wrong_value"
        res["relational+logical"] = {k: {"n": v[0], "wrong_pct": round(100 * v[1] / v[0], 1)}
                                     for k, v in sorted(bins.items(), key=lambda x: ["0", "1-2", "3-4", "5+"].index(x[0]))}
        out[model] = res
    for model, res in out.items():
        print(f"== {model}: wrong-value rate among runnable programs, by cue count")
        for cue, bins in res.items():
            print(f"  {cue:20s} " + "  ".join(f"{k}: {v['wrong_pct']:5.1f}% (n={v['n']})" for k, v in bins.items()))
    json.dump(out, open("results/lexical_cue_difficulty.json", "w"), indent=1)


if __name__ == "__main__":
    main()
