"""armR0-nc: ReSocratic reference code with ALL comments removed (tokenize-safe), same 2,814 problems as armR0.
Only COMMENT tokens and the comment-only lines they leave are removed; code, prints, and names unchanged.
Each stripped program is re-executed and must reproduce the original program's objval.
Usage: .venv-gpt/bin/python scripts/build_armR0nc.py
"""
import io, json, os, sys, tokenize
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.getcwd())
from or_rlvr.reward.solver import SCIPSandbox

SB = SCIPSandbox(timeout=30)


def strip_comments(code):
    toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    kill = {}  # line -> start col of comment
    for t in toks:
        if t.type == tokenize.COMMENT:
            kill[t.start[0]] = t.start[1]
    out = []
    for i, ln in enumerate(code.splitlines(), 1):
        if i in kill:
            ln = ln[:kill[i]].rstrip()
            if not ln.strip():
                continue
        out.append(ln)
    s = "\n".join(out)
    while "\n\n\n" in s:
        s = s.replace("\n\n\n", "\n\n")
    return s.strip() + "\n"


def check(pair):
    a, b = pair
    ra, rb = SB._execute_with_timeout(a), SB._execute_with_timeout(b)
    oa, ob = (ra or {}).get("objval"), (rb or {}).get("objval")
    return oa is not None and ob is not None and abs(oa - ob) <= 1e-6 * max(1.0, abs(oa))


def main():
    rows = [json.loads(l) for l in open("data/ablation_3k/armR0.jsonl")]
    new = [dict(r, code_block=strip_comments(r["code_block"]), verified="reference-nocomment") for r in rows]
    with ProcessPoolExecutor(max_workers=10) as ex:
        ok = list(ex.map(check, [(r["code_block"], n["code_block"]) for r, n in zip(rows, new)], chunksize=16))
    bad = [r["problem_id"] for r, o in zip(rows, ok) if not o]
    n_hash = sum("#" in n["code_block"] for n in new)
    ratio = sum(len(n["code_block"]) for n in new) / sum(len(r["code_block"]) for r in rows)
    print(f"rows {len(new)} | objval identik {sum(ok)}/{len(ok)} | sisa '#' (dalam string) {n_hash} | panjang {ratio:.2f}x")
    if bad:
        print("objval berbeda/gagal:", bad[:20])
    with open("data/ablation_3k/armR0nc.jsonl", "w") as f:
        for n in new:
            f.write(json.dumps(n) + "\n")


if __name__ == "__main__":
    main()
