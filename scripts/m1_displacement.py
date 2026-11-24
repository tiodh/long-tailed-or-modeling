"""Uji M1 (likelihood displacement): apakah OTPO bersih menekan token yang BENAR?

Satu model dasar bf16 + dua adapter (SFT, OTPO bersih), log-prob per token dihitung dgn keduanya.
A) 319 pasangan training: chosen & rejected, token dibagi: prose | kode-sama (baris ada di pasangan) |
   kode-beda, plus kategori objective/constraint/variable. Belajar sehat: chosen naik, rejected turun.
   Displacement: chosen (terutama token SAMA) ikut turun.
B) Held-out: jawaban BENAR SFT pada ReSocratic-val. Kalau log-prob-nya turun di OTPO,
   model jadi lebih jarang menulis formulasi yang benar -> menjelaskan akurasi in-dist turun.
Keluaran: results/m1/displacement.json
"""
import ast, json, os, re, sys, collections, statistics as st
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
sys.path.insert(0, os.getcwd())
from or_rlvr.data.dataset import build_prompt

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, "checkpoints/sft_instruct/final", adapter_name="sft")
model.load_adapter("checkpoints/otpo_ot_wnll0_reso/final", adapter_name="otpo")
model.eval()
CODE_RE = re.compile(r"\[CODE\].*?```(?:python)?\s*\n(.*?)```", re.DOTALL)
MAXTOK = 3000

def regions(raw, other):
    m = CODE_RE.search(raw)
    R = {"code": None, "same": [], "diff": [], "obj": [], "con": [], "var": []}
    if not m: return R
    code, off = m.group(1), m.start(1); R["code"] = (off, off + len(code))
    lines = code.split("\n"); st_ = [0]
    for ln in lines[:-1]: st_.append(st_[-1] + len(ln) + 1)
    om = CODE_RE.search(other) if other else None
    oset = set(x.strip() for x in (om.group(1) if om else "").split("\n"))
    for s, l in zip(st_, lines):
        if not l.strip(): continue
        (R["same"] if (other and l.strip() in oset) else R["diff"]).append((off + s, off + s + len(l)))
    try: tree = ast.parse(code)
    except SyntaxError: return R
    ch = lambda l, c: off + st_[l-1] + len(lines[l-1].encode()[:c].decode(errors="ignore"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            sp = (ch(n.lineno, n.col_offset), ch(n.end_lineno, n.end_col_offset))
            if n.func.attr in ("addVar", "addVars"): R["var"].append(sp)
            elif n.func.attr == "setObjective" or (n.func.attr == "addCons" and any(isinstance(x, ast.Name) and x.id == "obj" for x in ast.walk(n))): R["obj"].append(sp)
            elif n.func.attr == "addCons": R["con"].append(sp)
    return R

inside = lambda mid, sp: any(s <= mid < e for s, e in sp)

@torch.no_grad()
def logps(prompt, resp, adapter):
    model.set_adapter(adapter)
    p = tok(prompt, add_special_tokens=False).input_ids
    r = tok(resp, add_special_tokens=False, return_offsets_mapping=True)
    ids = (p + r.input_ids)[:MAXTOK]
    x = torch.tensor([ids], device="cuda")
    lg = model(input_ids=x).logits[0, :-1].float()
    lp = -F.cross_entropy(lg, x[0, 1:], reduction="none")[len(p) - 1:]
    return lp.cpu().tolist(), r.offset_mapping[:len(lp)]

def groups_of(raw, other, offs):
    R = regions(raw, other); out = []
    for a, b in offs:
        mid = (a + b) / 2; g = []
        if R["code"] and R["code"][0] <= mid < R["code"][1]:
            g.append("code-same" if inside(mid, R["same"]) else "code-diff" if inside(mid, R["diff"]) else "code-blank")
            for k in ("obj", "con", "var"):
                if inside(mid, R[k]): g.append(k)
        else:
            g.append("prose")
        out.append(g)
    return out

def accumulate(acc, prompt, raw, other, label):
    a, offs = logps(prompt, raw, "sft"); b, _ = logps(prompt, raw, "otpo")
    for g, x, y in zip(groups_of(raw, other, offs), a, b):
        for k in g + ["ALL"]:
            acc[(label, k)].append(y - x)

data = json.load(open("data/resocratic_29k_final.json"))
acc = collections.defaultdict(list)
pairs = [json.loads(l) for l in open("data/otpo_pairs_resocratic.jsonl")]
for n, p in enumerate(pairs):
    pr = build_prompt(data[int(p["pid"].split("#")[1])]["question"])
    accumulate(acc, pr, p["chosen"], p["rejected"], "train-chosen")
    accumulate(acc, pr, p["rejected"], p["chosen"], "train-rejected")
    if n % 50 == 0: print(f"[m1] pasangan {n}/{len(pairs)}", flush=True)
rows = {json.loads(l)["i"]: json.loads(l)["correct"] for l in open("results/metrics/qwen_sft_val_fa_rows.jsonl")}
held = [json.loads(l) for l in open("results/metrics/qwen_sft_val_gen.jsonl")]
held = [h for h in held if "_meta" not in h and rows.get(int(h["i"]))][:400]
for n, h in enumerate(held):
    accumulate(acc, build_prompt(data[int(h["i"])]["question"]), h["text"], None, "heldout-correct")
    if n % 100 == 0: print(f"[m1] held-out {n}/{len(held)}", flush=True)
res = {f"{l}|{k}": {"n_tok": len(v), "mean_dlogp": st.mean(v), "frac_down": sum(x < 0 for x in v) / len(v)}
       for (l, k), v in acc.items() if v}
os.makedirs("results/m1", exist_ok=True)
json.dump(res, open("results/m1/displacement.json", "w"), indent=1)
print("SELESAI", flush=True)
