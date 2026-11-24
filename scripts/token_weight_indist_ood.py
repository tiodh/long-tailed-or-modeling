"""Bobot token (omega, seperti OTPO arm 2_OT) pada pasangan benar-vs-salah: IN-DIST vs OOD.

IN-DIST : data/otpo_pairs_resocratic.jsonl  (soal ReSocratic-train, sampel model fine-tuned n=8)
OOD     : data/otpo_pairs.jsonl             (soal 6 benchmark yang gagal di greedy, 50 sampel/soal)
Dipakai HANYA untuk analisis; omega bergantung pada teks saja (embedding input statis).

Per pasangan: symptom rejected dgn chosen (benar) sbg reference; token pembawa kesalahan:
  O-*  -> baris objective      V-* -> deklarasi variabel      C-* -> baris constraint yg tak ada di pasangan
  BUILD (exec/syntax/format) -> baris tempat error terjadi (dari traceback eksekusi ulang)
Rasio penargetan = (% massa omega pd token pembawa) / (% token pembawa); 1 = setara bobot seragam.
"""
import ast, glob, json, os, re, subprocess, sys, tempfile, uuid, collections
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
sys.path.insert(0, os.getcwd())
from or_rlvr.reward.structure import capture_structure, _extract_code
from or_rlvr.reward.solver import SCIP_TEMPLATE
from scripts.wrong_value_symptoms import symptoms

BASE = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*"))[0]
wm = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]
with safe_open(f"{BASE}/{wm['model.embed_tokens.weight']}", "pt") as f:
    E = f.get_tensor("model.embed_tokens.weight").float().cuda()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
CODE_RE = re.compile(r"\[CODE\].*?```(?:python)?\s*\n(.*?)```", re.DOTALL)
CODE_RE2 = re.compile(r"\[CODE\]\s*\n(.*?)\[/CODE\]", re.DOTALL)
HDR = SCIP_TEMPLATE.split("{code}")[0].count("\n")

def omega_ot(rc, rr, eps=0.1, iters=50):          # identik train_otpo.py::omega_ot
    Ec = E[torch.tensor(rc, device="cuda")]; Er = E[torch.tensor(rr, device="cuda")]
    Ec = Ec / (Ec.norm(dim=1, keepdim=True) + 1e-8); Er = Er / (Er.norm(dim=1, keepdim=True) + 1e-8)
    C = 1.0 - Ec @ Er.T; K = torch.exp(-C / eps)
    a = torch.full((C.shape[0],), 1.0 / C.shape[0], device="cuda"); b = torch.full((C.shape[1],), 1.0 / C.shape[1], device="cuda")
    u = torch.ones_like(a); v = torch.ones_like(b)
    for _ in range(iters):
        u = a / (K @ v + 1e-8); v = b / (K.T @ u + 1e-8)
    P = u.unsqueeze(1) * K * v.unsqueeze(0)
    nz = lambda w: ((w + 1e-6) * len(w) / (w + 1e-6).sum()).cpu().tolist()
    return nz((P * C).sum(1)), nz((P * C).sum(0))

def code_of(raw):
    m = CODE_RE.search(raw) or CODE_RE2.search(raw)
    return (m.group(1), m.start(1)) if m else (None, None)

def regions(raw, other):
    code, off = code_of(raw)
    R = {"code": None, "obj": [], "var": [], "con": [], "lines": []}
    if code is None: return R
    lines = code.split("\n"); st = [0]
    for ln in lines[:-1]: st.append(st[-1] + len(ln) + 1)
    R["code"] = (off, off + len(code))
    R["lines"] = [(off + s, off + s + len(l)) for s, l in zip(st, lines)]
    ocode, _ = code_of(other); oset = set(l.strip() for l in (ocode or "").split("\n"))
    R["diff"] = [(off + s, off + s + len(l)) for s, l in zip(st, lines) if l.strip() and l.strip() not in oset]
    try: tree = ast.parse(code)
    except SyntaxError: return R
    ch = lambda l, c: off + st[l-1] + len(lines[l-1].encode()[:c].decode(errors="ignore"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            sp = (ch(n.lineno, n.col_offset), ch(n.end_lineno, n.end_col_offset))
            if n.func.attr in ("addVar", "addVars"): R["var"].append(sp)
            elif n.func.attr == "setObjective" or (n.func.attr == "addCons" and any(isinstance(x, ast.Name) and x.id == "obj" for x in ast.walk(n))):
                R["obj"].append(sp)
            elif n.func.attr == "addCons": R["con"].append(sp)
    return R

def error_line(raw):
    code, off = code_of(raw)
    if code is None: return None
    try: ast.parse(code)
    except SyntaxError as e:
        ln = e.lineno or 1
    else:
        path = tempfile.mktemp(suffix=".py"); lp = tempfile.mktemp(suffix=".lp")
        open(path, "w").write(SCIP_TEMPLATE.format(code=code, lp_path=lp))
        try:
            p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return None
        finally:
            for q in (path, lp):
                if os.path.exists(q): os.unlink(q)
        hits = re.findall(r'File "%s", line (\d+)' % re.escape(path), p.stderr)
        if not hits: return None
        ln = int(hits[-1]) - HDR
    lines = code.split("\n")
    if not (1 <= ln <= len(lines)): return None
    s = off + sum(len(l) + 1 for l in lines[:ln-1])
    return (s, s + len(lines[ln-1]))

inside = lambda mid, spans: any(s <= mid < e for s, e in spans)

def analyse(name, path):
    pairs = [json.loads(l) for l in open(path)]
    cat_mass = collections.Counter(); cat_tok = collections.Counter()
    diffw = [0.0, 0, 0.0, 0]
    acc = collections.defaultdict(lambda: [0.0, 0.0, 0, 0]); cnt = collections.Counter(); used = 0
    for p in pairs:
        ec = tok(p["chosen"], add_special_tokens=False, return_offsets_mapping=True)
        er = tok(p["rejected"], add_special_tokens=False, return_offsets_mapping=True)
        if len(ec.input_ids) > 6000 or len(er.input_ids) > 6000: continue
        wc, wr = omega_ot(ec.input_ids, er.input_ids); used += 1
        st_ = p["rejected_status"]
        if st_ in ("wrong_value", "optimal"):
            g = capture_structure(_extract_code(p["rejected"])); r = capture_structure(_extract_code(p["chosen"]))
            S = symptoms(g, r) if g and r else set()
            fam = "structural"
        elif st_ in ("infeasible", "unbounded", "inforunbd"):
            S = {"SOLVE: " + st_}; fam = "solve"
        else:
            S = {"BUILD: error line"}; fam = "build"
        for s in S: cnt[s] += 1
        cnt["_group:" + fam] += 1
        eline = error_line(p["rejected"]) if fam == "build" else None
        for raw, enc, w, other, is_rej in ((p["chosen"], ec, wc, p["rejected"], False), (p["rejected"], er, wr, p["chosen"], True)):
            R = regions(raw, other)
            for (a, b), wi in zip(enc.offset_mapping, w):
                mid = (a + b) / 2
                if R["code"] and R["code"][0] <= mid < R["code"][1]:
                    c = "objective" if inside(mid, R["obj"]) else "variable" if inside(mid, R["var"]) else "constraint" if inside(mid, R["con"]) else "code-other"
                    d = inside(mid, R.get("diff", []))
                    diffw[0 if d else 2] += wi; diffw[1 if d else 3] += 1
                else:
                    c = "prose"
                cat_mass[c] += wi; cat_tok[c] += 1
                car = {"O": inside(mid, R["obj"]), "V": inside(mid, R["var"]),
                       "C": inside(mid, R["con"]) and inside(mid, R.get("diff", []))}
                for s in S:
                    if s.startswith("BUILD"):
                        if not is_rej: continue
                        hit = eline is not None and eline[0] <= mid < eline[1]
                    elif s[0] in "OVC": hit = car[s[0]]
                    else: continue
                    acc[s][0] += wi * hit; acc[s][1] += wi; acc[s][2] += hit; acc[s][3] += 1
    T = sum(cat_tok.values()); M = sum(cat_mass.values())
    out = {"pairs": used,
           "category": {c: {"tok_pct": cat_tok[c] / T * 100, "mass_pct": cat_mass[c] / M * 100, "mean_w": cat_mass[c] / cat_tok[c]} for c in cat_tok},
           "diff_line_mean_w": diffw[0] / max(diffw[1], 1), "same_line_mean_w": diffw[2] / max(diffw[3], 1),
           "groups": {k[7:]: v / used * 100 for k, v in cnt.items() if k.startswith("_group:")},
           "symptoms": {s: {"pair_pct": cnt[s] / used * 100,
                            "carrier_tok_pct": (acc[s][2] / acc[s][3] * 100) if acc[s][3] else None,
                            "carrier_mass_pct": (acc[s][0] / acc[s][1] * 100) if acc[s][1] else None,
                            "ratio": ((acc[s][0] / acc[s][1]) / (acc[s][2] / acc[s][3])) if acc[s][2] else None}
                        for s in cnt if not s.startswith("_group:")}}
    print(f"[{name}] {used} pasangan selesai", flush=True)
    return out

res = {"in-dist": analyse("in-dist", "data/otpo_pairs_resocratic.jsonl"), "ood": analyse("ood", "data/otpo_pairs.jsonl")}
os.makedirs("results/token_weight", exist_ok=True)
json.dump(res, open("results/token_weight/indist_vs_ood.json", "w"), indent=1)
print("SELESAI", flush=True)
