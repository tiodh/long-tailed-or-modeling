"""Audit bobot token OTPO (arm 2_OT) pada pair bersih: ke mana massa omega jatuh, dan
apakah jatuh ke bagian kode yang BENAR-BENAR berbeda antara chosen dan rejected.

omega dihitung PERSIS seperti scripts/train_otpo.py::omega_ot (embedding input statis,
Sinkhorn seimbang eps=0.1 iters=50, bobot = sum_j P_ij*C_ij, dinormalisasi rerata 1).
LoRA SFT tidak menyentuh embed_tokens -> cukup embedding model dasar.

Kategori token: prose | code-other | constraint-body | constraint-relation | objective
(objective = setObjective + addCons yang memuat variabel 'obj', gaya ReSocratic).
diff-code: token di baris kode yang tidak ada di sequence pasangannya (difflib per baris).
"""
import ast, json, os, re, sys, glob, difflib, collections
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
sys.path.insert(0, os.getcwd())
from or_rlvr.data.dataset import build_prompt

BASE = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*"))[0]
idx = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]
with safe_open(f"{BASE}/{idx['model.embed_tokens.weight']}", "pt") as f:
    E = f.get_tensor("model.embed_tokens.weight").float().cuda()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
MAXLEN = 1200
CODE_RE = re.compile(r"\[CODE\].*?```(?:python)?\s*\n(.*?)```", re.DOTALL)
CODE_RE2 = re.compile(r"\[CODE\]\s*\n(.*?)\[/CODE\]", re.DOTALL)
REL_RE = re.compile(r"(<=|>=|==|<|>)")

def omega_ot(rc, rr, eps=0.1, iters=50):              # salinan persis trainer
    Ec = E[torch.tensor(rc, device="cuda")]; Er = E[torch.tensor(rr, device="cuda")]
    Ec = Ec / (Ec.norm(dim=1, keepdim=True) + 1e-8); Er = Er / (Er.norm(dim=1, keepdim=True) + 1e-8)
    C = 1.0 - Ec @ Er.T; K = torch.exp(-C / eps)
    a = torch.full((C.shape[0],), 1.0 / C.shape[0], device="cuda"); b = torch.full((C.shape[1],), 1.0 / C.shape[1], device="cuda")
    u = torch.ones_like(a); v = torch.ones_like(b)
    for _ in range(iters):
        u = a / (K @ v + 1e-8); v = b / (K.T @ u + 1e-8)
    P = u.unsqueeze(1) * K * v.unsqueeze(0)
    wc = (P * C).sum(1); wr = (P * C).sum(0)
    nz = lambda w: ((w + 1e-6) * len(w) / (w + 1e-6).sum()).cpu()
    return nz(wc), nz(wr), bool(torch.isfinite(P).all())

def spans(raw):
    m = CODE_RE.search(raw) or CODE_RE2.search(raw)
    if not m: return None
    code, off = m.group(1), m.start(1)
    lines = code.split("\n"); starts = [0]
    for ln in lines[:-1]: starts.append(starts[-1] + len(ln) + 1)
    out = {"code": (off, off + len(code)), "con": [], "rel": [], "obj": [], "lines": [(off + s, off + s + len(l), l.strip()) for s, l in zip(starts, lines)]}
    try: tree = ast.parse(code)
    except SyntaxError: return out
    bstart = [len(code[:s].encode()) for s in starts]           # col_offset = BYTE
    def ch(lineno, col):                                           # byte col -> char offset
        line = lines[lineno - 1]; return off + starts[lineno - 1] + len(line.encode()[:col].decode(errors="ignore"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("addCons", "setObjective"):
            s, e = ch(n.lineno, n.col_offset), ch(n.end_lineno, n.end_col_offset)
            is_obj = n.func.attr == "setObjective" or any(isinstance(x, ast.Name) and x.id == "obj" for x in ast.walk(n))
            if is_obj: out["obj"].append((s, e))
            else:
                out["con"].append((s, e)); mm = REL_RE.search(raw[s:e])
                if mm: out["rel"].append((s + mm.start(), s + mm.end()))
    return out

def cats(raw, offs, sp, difflines):
    res = []
    for a, b in offs:
        mid = (a + b) / 2; c = "prose"; d = False
        if sp and sp["code"][0] <= mid < sp["code"][1]:
            c = "code-other"
            if any(s <= mid < e for s, e in sp["obj"]): c = "objective"
            elif any(s <= mid < e for s, e in sp["rel"]): c = "constraint-relation"
            elif any(s <= mid < e for s, e in sp["con"]): c = "constraint-body"
            d = any(s <= mid < e for s, e, _ in difflines)
        res.append((c, d))
    return res

def difflines(sp_a, sp_b):
    if not sp_a or not sp_b: return []
    A = [l[2] for l in sp_a["lines"]]; B = set(l[2] for l in sp_b["lines"])
    return [l for l in sp_a["lines"] if l[2] and l[2] not in B]

pairs = [json.loads(l) for l in open("data/otpo_pairs_resocratic.jsonl")]
data = json.load(open("data/resocratic_29k_final.json"))
mass = collections.defaultdict(float); ntok = collections.Counter(); wsum = collections.defaultdict(float)
dm = collections.defaultdict(float); dn = collections.Counter()
top = []; kept = 0; nonfinite = 0; nonascii = 0
for p in pairs:
    q = data[int(p["pid"].split("#")[1])]["question"]
    plen = len(tok(build_prompt(q), add_special_tokens=False).input_ids)
    ec = tok(p["chosen"], add_special_tokens=False, return_offsets_mapping=True)
    er = tok(p["rejected"], add_special_tokens=False, return_offsets_mapping=True)
    if plen + len(ec.input_ids) > MAXLEN + plen and False: pass
    keep = (plen + len(ec.input_ids) <= MAXLEN + plen - 0) and not (len(ec.input_ids) - (MAXLEN - plen) >= plen) and not (len(er.input_ids) - (MAXLEN - plen) >= plen)
    # replikasi encode(): contoh dibuang bila pemotongan prompt tak cukup
    keep = all(len(x.input_ids) + plen - MAXLEN < plen for x in (ec, er))
    if not keep: continue
    kept += 1
    wc, wr, fin = omega_ot(ec.input_ids, er.input_ids); nonfinite += (not fin)
    for raw, enc, w, other in ((p["chosen"], ec, wc, p["rejected"]), (p["rejected"], er, wr, p["chosen"])):
        sp, so = spans(raw), spans(other)
        if sp and any(ord(ch) > 127 for ch in raw[sp["code"][0]:sp["code"][1]]): nonascii += 1
        cl = cats(raw, enc.offset_mapping, sp, difflines(sp, so))
        for (c, d), wi, tid in zip(cl, w.tolist(), enc.input_ids):
            mass[c] += wi; ntok[c] += 1
            if c != "prose": dm[("diff" if d else "same")] += wi; dn[("diff" if d else "same")] += 1
            top.append((wi, tok.decode([tid]), c, d))
T = sum(ntok.values()); M = sum(mass.values())
print(f"pair dipakai (lolos MAXLEN seperti training): {kept}/{len(pairs)} | Sinkhorn non-finite: {nonfinite} | kode ber-karakter non-ASCII: {nonascii}")
print(f"\n{'kategori':22s}{'% token':>9s}{'% massa w':>11s}{'rerata w':>10s}")
for c in ["prose", "code-other", "objective", "constraint-body", "constraint-relation"]:
    print(f"{c:22s}{ntok[c]/T*100:8.1f}%{mass[c]/M*100:10.1f}%{mass[c]/max(ntok[c],1):10.3f}")
D = dn["diff"] + dn["same"]; DM = dm["diff"] + dm["same"]
print(f"\nDi dalam KODE: token di baris yang BERBEDA dgn pasangannya = {dn['diff']/D*100:.1f}% token, {dm['diff']/DM*100:.1f}% massa w "
      f"(rerata w diff {dm['diff']/max(dn['diff'],1):.3f} vs sama {dm['same']/max(dn['same'],1):.3f})")
top.sort(key=lambda x: -x[0])
k = int(len(top) * 0.01)
tc = collections.Counter(c for _, _, c, _ in top[:k])
print(f"\n1% token berbobot tertinggi (n={k}) per kategori: {dict(tc.most_common())}")
print("contoh 25 teratas:", [(round(w, 1), t, c[:6]) for w, t, c, _ in top[:25]])
