"""AST error taxonomy: kelompokkan kegagalan program yang dihasilkan, in-dist vs OOD.

Tiap sampel: ekstrak kode -> ast.parse -> fitur statis AST -> eksekusi (stderr ditangkap,
beda dari SCIPSandbox yang membuangnya) -> kelompok kegagalan.
Keluaran: results/ast_taxonomy/<set>.jsonl (satu baris per soal) + ringkasan di stdout.
"""
import ast, json, os, re, subprocess, sys, tempfile, uuid, collections, glob
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from or_rlvr.reward.solver import SCIPSandbox, SCIP_TEMPLATE

SB = SCIPSandbox(timeout=30)
REL = 1e-6


class VarScan(ast.NodeVisitor):
    """Nama yang di-assign dari addVar (skalar, dict/list comprehension, loop)."""
    def __init__(self):
        self.vars = set()
    def _has_addvar(self, node):
        return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ("addVar", "addVars") for n in ast.walk(node))
    def visit_Assign(self, node):
        if self._has_addvar(node.value):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name): self.vars.add(n.id)
        self.generic_visit(node)


def refs_var(node, vs):
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id in vs: return True
    return False


def features(code):
    tree = ast.parse(code)
    vs_ = VarScan(); vs_.visit(tree); vs = vs_.vars
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    attr = lambda c: c.func.attr if isinstance(c.func, ast.Attribute) else (c.func.id if isinstance(c.func, ast.Name) else "")
    addcons = [c for c in calls if attr(c) == "addCons"]
    f = {
        "n_addCons_calls": len(addcons),
        "n_addVar_calls": sum(attr(c) in ("addVar", "addVars") for c in calls),
        "has_setObjective": any(attr(c) == "setObjective" for c in calls),
        "has_optimize": any(attr(c) == "optimize" for c in calls),
        "cons_in_loop": any(isinstance(p, (ast.For, ast.comprehension)) and any(
            isinstance(n, ast.Call) and attr(n) == "addCons" for n in ast.walk(p))
            for p in ast.walk(tree) if isinstance(p, ast.For)),
        "vtype_binary": bool(re.search(r"vtype\s*=\s*['\"](B|BINARY)", code)),
        "vtype_integer": bool(re.search(r"vtype\s*=\s*['\"](I|INTEGER)", code)),
        "nonlinear_term": False, "builtin_on_var": False, "division_by_var": False,
        "strict_ineq": False, "bigM_literal": bool(re.search(r"\b(M|big_?M)\s*=\s*[0-9e.+]{3,}", code)),
    }
    for n in ast.walk(tree):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult) and refs_var(n.left, vs) and refs_var(n.right, vs):
            f["nonlinear_term"] = True
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod)) and refs_var(n.right, vs):
            f["division_by_var"] = True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("max", "min", "abs") \
                and any(refs_var(a, vs) for a in n.args):
            f["builtin_on_var"] = True
    for c in addcons:
        for n in ast.walk(c):
            if isinstance(n, ast.Compare) and any(isinstance(o, (ast.Lt, ast.Gt)) for o in n.ops):
                f["strict_ineq"] = True
    return f


def execute(code):
    run_id = uuid.uuid4().hex
    lp = os.path.join(tempfile.gettempdir(), f"ast_{run_id}.lp")
    script = SCIP_TEMPLATE.format(code=code, lp_path=lp)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(script); path = fh.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=30)
        res = None
        for line in p.stdout.splitlines():
            if line.startswith("SCIP_RESULT:"):
                res = json.loads(line[len("SCIP_RESULT:"):])
        err = [l for l in p.stderr.strip().splitlines() if l.strip()]
        return res, (err[-1][:200] if err else ""), False
    except subprocess.TimeoutExpired:
        return None, "", True
    finally:
        for q in (path, lp):
            if os.path.exists(q): os.unlink(q)


def exc_group(msg):
    m = msg.lower()
    if "scip_error" in m or "pyscipopt" in m or "scip" in m:
        if any(k in m for k in ("nonlinear", "non-linear", "quadratic", "expr", "not supported", "polynomial")):
            return "run:scip-nonlinear/expr"
        return "run:scip-other"
    cls = msg.split(":")[0].strip().split(".")[-1]
    known = {"NameError", "KeyError", "IndexError", "TypeError", "AttributeError", "ValueError",
             "ZeroDivisionError", "NotImplementedError", "AssertionError", "ModuleNotFoundError", "ImportError"}
    if cls in known:
        if cls == "TypeError" and any(k in m for k in ("expr", "unsupported operand", "not supported between")):
            return "run:TypeError-expr"
        return f"run:{cls}"
    return "run:other"


def classify(args):
    text, gt = args
    code = SB.extract_code_block(text)
    if not code or not SB.check_format(text):
        return {"group": "format:no-code-or-segments", "feat": None}
    try:
        feat = features(code)
    except SyntaxError:
        return {"group": "parse:SyntaxError", "feat": None}
    except Exception:
        feat = None
    res, err, to = execute(code)
    if to:
        g = "run:timeout"
    elif res is None:
        g = exc_group(err) if err else "run:no-result"
    else:
        st = (res.get("status") or "").lower()
        ov = res.get("objval")
        if st in ("infeasible", "inforunbd"): g = "solve:infeasible"
        elif st == "unbounded": g = "solve:unbounded"
        elif ov is None: g = "solve:no-objective-value"
        elif gt is not None and abs(ov - gt) / max(abs(gt), 1.0) < REL: g = "correct"
        else: g = "wrong_value"
    return {"group": g, "feat": feat, "err": err if g.startswith("run:") else ""}


def load_sets():
    sets = {}
    data = json.load(open("data/resocratic_29k_final.json"))
    ind = []
    for l in open("results/metrics/qwen_sft_val_gen.jsonl"):
        d = json.loads(l)
        if "_meta" in d: continue
        i = int(d["i"]); ind.append((d["text"], float(data[i]["answer"])))
    sets["in-dist ReSocratic-val"] = ind
    from or_rlvr.evaluation.benchmarks import _extract_ground_truth
    ood = []
    for b in ["nl4opt", "mamo_easy", "mamo_complex", "optibench", "industryor", "optmath"]:
        items = json.load(open(f"data/benchmarks/{b}.json"))
        for l in open(f"results/vllm/sft_instruct_8k/{b}.jsonl"):
            d = json.loads(l)
            if "_meta" in d: continue
            try: gt = _extract_ground_truth(items[d["i"]])
            except Exception: gt = None
            ood.append((d["text"], gt))
    sets["OOD 6 benchmark"] = ood
    lg = {x["index"]: float(x["answer"]) for x in json.load(open("data/benchmarks/logior.json"))}
    sets["OOD-far LogiOR"] = [(json.loads(l)["text"], lg[json.loads(l)["i"]])
                             for l in open("results/bench/logior_sft_gen.jsonl")]
    return sets


if __name__ == "__main__":
    os.makedirs("results/ast_taxonomy", exist_ok=True)
    summary = {}
    for name, items in load_sets().items():
        with ProcessPoolExecutor(max_workers=12) as ex:
            out = list(ex.map(classify, items, chunksize=8))
        fn = "results/ast_taxonomy/" + re.sub(r"[^a-z0-9]+", "_", name.lower()) + ".jsonl"
        with open(fn, "w") as fh:
            for o in out: fh.write(json.dumps(o) + "\n")
        summary[name] = dict(collections.Counter(o["group"] for o in out))
        print(f"[ast] {name}: n={len(out)} {summary[name]}", flush=True)
    json.dump(summary, open("results/ast_taxonomy/summary.json", "w"), indent=2)
