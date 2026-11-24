"""Skor per-soal ReSocratic-val (format-agnostic, logika identik score_bench_fa.py) untuk uji McNemar antar-metode.
Usage: python scripts/score_val_fa.py <tag> [<tag> ...]   (membaca results/metrics/<tag>_val_gen.jsonl)
Tulis results/metrics/<tag>_val_fa_rows.jsonl : {i, correct}
"""
import json, sys, os
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "/home/tio/Documents/sirl-pgrpo")
from or_rlvr.reward.solver import SCIPSandbox
SB = SCIPSandbox(timeout=30)
DATA = json.load(open("data/resocratic_29k_final.json"))

def one(row):
    i, text = row
    gt = float(DATA[i]["answer"])
    code = SB.extract_code_block(text)
    res = SB._execute_with_timeout(code) if code else None
    ov = res.get("objval") if res else None
    return {"i": i, "correct": bool(ov is not None and abs(ov - gt) / max(abs(gt), 1.0) < 1e-6)}

for tag in sys.argv[1:]:
    rows = []
    for l in open(f"results/metrics/{tag}_val_gen.jsonl"):
        d = json.loads(l)
        if "_meta" in d: continue
        rows.append((int(d["i"]), d["text"]))
    with ProcessPoolExecutor(max_workers=12) as ex:
        out = list(ex.map(one, rows, chunksize=8))
    with open(f"results/metrics/{tag}_val_fa_rows.jsonl", "w") as f:
        for o in out: f.write(json.dumps(o) + "\n")
    print(f"[val-fa] {tag}: {sum(o['correct'] for o in out)}/{len(out)} = {sum(o['correct'] for o in out)/len(out)*100:.2f}%", flush=True)
