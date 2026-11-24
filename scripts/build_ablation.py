#!/usr/bin/env python
"""Build a 4-arm supervision-type ablation dataset (PILOT).

Nested ladder — each arm adds exactly ONE factor over the previous:
  arm0_code_only          : [CODE] only (floor)
  arm1_generic_cot        : existing ReSocratic template reasoning (decoupled)   [no teacher]
  arm2_grounded           : causal derivation (quote NL clause -> math relation)  [teacher]
  arm3_grounded_selfverify: arm2 + [self-check: ...] lines                        [teacher]

Efficiency: teacher generates arm3 ONCE, the rest are derived mechanically
  arm2 = arm3 minus every "[self-check: ...]" line
  arm1 = build_completion(item) (existing nl_solution template)
  arm0 = code only

Verifier gate (mandatory): run the teacher's generated [CODE] in SCIP and keep the item
only if  |z_pred - y| / max(|y|, 1) < 1e-6  (y = ground-truth answer). All four arms for a
problem are built ONLY if arm3 passed -> all arms share the identical verified problem set.

[CODE] block is IDENTICAL across all four arms (so a gain is attributable to reasoning
content, not code). --shared_code chooses which code that is:
  teacher   (default) : the teacher's own verified arm3 code (matches the one-shot spec)
  reference           : the ReSocratic reference code_solution (already y-correct)

Usage (pilot, 50 problems):
  DEEPSEEK_API_KEY=... TEACHER_MODEL=<deepseek-v4-flash-id> \
    .venv-rl/bin/python scripts/build_ablation.py --n 50 --out data/ablation

  # dry validation of the non-teacher machinery (no API needed):
  .venv-rl/bin/python scripts/build_ablation.py --self_test
"""
import os, sys, re, json, time, random, argparse
sys.path.insert(0, "/home/tio/Documents/sirl-pgrpo")
from or_rlvr.reward.solver import SCIPSandbox
from or_rlvr.data.dataset import build_completion  # existing template -> arm1

DATA = "data/resocratic_29k_final.json"
REL_TOL = 1e-6

# ---- teacher config (all overridable via env) --------------------------------
API_KEY   = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
BASE_URL  = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# NOTE: confirm the real API model id for "DeepSeek V4 Flash" before a paid run.
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "deepseek-chat")

# ---- block parsing -----------------------------------------------------------
_R = re.compile(r"\[REASONING\](.*?)\[/REASONING\]", re.DOTALL)
_F = re.compile(r"\[FORMULA\](.*?)\[/FORMULA\]", re.DOTALL)


def parse_blocks(text, sb):
    """Return (reasoning, formula, code) strings from a tagged completion. code via SCIP extractor."""
    rm, fm = _R.search(text), _F.search(text)
    reasoning = rm.group(1).strip() if rm else ""
    formula = fm.group(1).strip() if fm else ""
    code = sb.extract_code_block(text) or ""
    return reasoning, formula, code


def strip_selfcheck(formula):
    """arm2 = arm3 formula minus every line containing a [self-check: ...] marker."""
    return "\n".join(l for l in formula.splitlines() if "[self-check" not in l.lower()).strip()


def verify(code, y, sb):
    """Run code in SCIP; keep iff |objval - y|/max(|y|,1) < REL_TOL. Returns (ok, objval)."""
    if not code or y is None:
        return False, None
    try:
        res = sb._execute_with_timeout(code)
    except Exception:
        return False, None
    ov = res.get("objval") if res else None
    if ov is None:
        return False, None
    ok = abs(ov - y) / max(abs(y), 1.0) < REL_TOL
    return ok, ov


# ---- teacher prompt ----------------------------------------------------------
TEACHER_SYS = (
    "You are an operations-research modeling teacher. Given a natural-language optimization "
    "problem and a REFERENCE pyscipopt solution (a correctness hint), write a GROUNDED causal "
    "derivation, then runnable code. Rules:\n"
    "1. Output EXACTLY three tagged blocks: [REASONING]...[/REASONING] [FORMULA]...[/FORMULA] "
    "[CODE]```python ...```[/CODE]. Nothing else.\n"
    "2. In [FORMULA], for EACH objective and constraint: quote the exact NL clause it comes from, "
    "then map it to a math relation and briefly say why (direction of <=/>=, units).\n"
    "3. DERIVE from the NL — do not merely copy the reference code. Use the reference only to reach "
    "the correct final formulation (STaR-with-hint).\n"
    "4. After EACH objective/constraint line add a self-check line of the form "
    "'[self-check: <what you verified> OK]' (direction / units / coverage).\n"
    "5. The [CODE] must be runnable pyscipopt that solves to the same optimum as the reference."
)


def _fewshot_user(nl, ref_code):
    return (f"Input NL:\n{nl}\n\nReference code (correctness hint, DERIVE — do not copy verbatim "
            f"into the reasoning):\n{ref_code}\n\nProduce the grounded + self-verify version in the "
            f"exact tagged format.")


# One-shot demonstration (the garden example from the task spec).
FEWSHOT_NL = ("A rectangular garden is fenced on three sides (the fourth is a river). The length is "
              "twice the width; the area must be at least 50 m2; the width must not exceed half the "
              "length. Minimize the total wire used for fencing.")
FEWSHOT_REF_CODE = (
    "model = pyscipopt.Model()\n"
    "w = model.addVar(vtype=\"CONTINUOUS\", name=\"w\", lb=0)\n"
    "l = model.addVar(vtype=\"CONTINUOUS\", name=\"l\", lb=0)\n"
    "obj = model.addVar(\"obj\"); model.setObjective(obj, \"minimize\")\n"
    "model.addCons(obj == 4*w)\nmodel.addCons(w * l >= 50)\n"
    "model.addCons(w <= l / 2)\nmodel.addCons(l == 2*w)\nmodel.optimize()")
FEWSHOT_ASSISTANT = """[REASONING]
Variables: choose w = width, l = length (continuous, >= 0); the two dimensions to decide.
[/REASONING]
[FORMULA]
Objective <- "minimize the wire on the THREE fenced sides (the 4th is a river)":
  the river side (one length) is unfenced => wire = 2*width + 1*length = 2w + l
  NL: "length is twice the width" => l = 2w => wire = 4w. Minimize 4w.
  [self-check: 3 of 4 sides fenced => coefficient on l is 1, not 2  OK]
Constraint 1 <- "area must be at least 50 m2":
  area = width * length = w*l >= 50   [">=" = "at least"; w*l bilinear = nonlinear]
  [self-check: "at least" => >=  OK]
Constraint 2 <- "width must not exceed half the length":
  w <= l / 2   ["<=" = "not exceed"]
  [self-check: "not exceed" => <=  OK]
[/FORMULA]
[CODE]
```python
model = pyscipopt.Model()
w = model.addVar(vtype="CONTINUOUS", name="w", lb=0)
l = model.addVar(vtype="CONTINUOUS", name="l", lb=0)
obj = model.addVar("obj"); model.setObjective(obj, "minimize")
model.addCons(obj == 4*w)
model.addCons(w * l >= 50)
model.addCons(w <= l / 2)
model.addCons(l == 2*w)
model.optimize()
```
[/CODE]"""


# Second one-shot: a NONLINEAR (Cobb-Douglas) example, verified to solve to its answer
# (ReSocratic pid 20746, objval 14686.33). Teaches power/bilinear pyscipopt formulation.
FEWSHOT2_NL = ("A manufacturing company produces 3 types of electronic devices and allocates labor "
               "and machinery across production lines to maximize profit. Profits are nonlinear: "
               "P1 = 100*L1^0.5*M1^0.5, P2 = 150*L2^0.6*M2^0.4, P3 = 200*L3^0.7*M3^0.3. Total labor "
               "available is 100 units; total machinery 80 units; each line uses at most 40 units of "
               "labor and at most 30 units of machinery. Maximize total profit.")
FEWSHOT2_REF_CODE = ("model=pyscipopt.Model(); L1..L3 ub=40, M1..M3 ub=30 (INTEGER); "
                     "P_i == coef*L^a*M^b; obj==P1+P2+P3 maximize; sum L<=100; sum M<=80")
FEWSHOT2_ASSISTANT = r'''[REASONING]
Variables: L1,L2,L3 = labor on lines 1-3; M1,M2,M3 = machinery on lines 1-3 (integer, >= 0).
[/REASONING]
[FORMULA]
Objective <- "maximize the total profit" (nonlinear Cobb-Douglas per line):
  P1 = 100*L1^0.5*M1^0.5 ; P2 = 150*L2^0.6*M2^0.4 ; P3 = 200*L3^0.7*M3^0.3 ; maximize P1+P2+P3
  [self-check: exponents per line sum to 1; coefficients 100/150/200 match text OK]
Constraint 1 <- "total labor available is 100 units": L1+L2+L3 <= 100  [self-check: "available total" => <= OK]
Constraint 2 <- "total machinery available is 80 units": M1+M2+M3 <= 80  [self-check: <= OK]
Constraint 3 <- "each line at most 40 units of labor": L1,L2,L3 <= 40 (variable ub)  [self-check: per-line cap OK]
Constraint 4 <- "each line at most 30 units of machinery": M1,M2,M3 <= 30 (variable ub)  [self-check: per-line cap OK]
[/FORMULA]
[CODE]
```python
import pyscipopt
model = pyscipopt.Model()
L1 = model.addVar(vtype="INTEGER", name="L1", lb=0, ub=40)
L2 = model.addVar(vtype="INTEGER", name="L2", lb=0, ub=40)
L3 = model.addVar(vtype="INTEGER", name="L3", lb=0, ub=40)
M1 = model.addVar(vtype="INTEGER", name="M1", lb=0, ub=30)
M2 = model.addVar(vtype="INTEGER", name="M2", lb=0, ub=30)
M3 = model.addVar(vtype="INTEGER", name="M3", lb=0, ub=30)
P1 = model.addVar(name="P1"); P2 = model.addVar(name="P2"); P3 = model.addVar(name="P3")
obj = model.addVar(name="obj"); model.setObjective(obj, "maximize")
model.addCons(P1 == 100 * L1**0.5 * M1**0.5)
model.addCons(P2 == 150 * L2**0.6 * M2**0.4)
model.addCons(P3 == 200 * L3**0.7 * M3**0.3)
model.addCons(obj == P1 + P2 + P3)
model.addCons(L1 + L2 + L3 <= 100)
model.addCons(M1 + M2 + M3 <= 80)
model.optimize()
```
[/CODE]'''


def teacher_generate(nl, ref_code):
    """Call the teacher (OpenAI-compatible / DeepSeek) once. Returns (text, (in_tok, out_tok))."""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    msgs = [
        {"role": "system", "content": TEACHER_SYS},
        {"role": "user", "content": _fewshot_user(FEWSHOT_NL, FEWSHOT_REF_CODE)},
        {"role": "assistant", "content": FEWSHOT_ASSISTANT},
        {"role": "user", "content": _fewshot_user(FEWSHOT2_NL, FEWSHOT2_REF_CODE)},
        {"role": "assistant", "content": FEWSHOT2_ASSISTANT},
        {"role": "user", "content": _fewshot_user(nl, ref_code)},
    ]
    r = client.chat.completions.create(model=TEACHER_MODEL, temperature=0.2,
                                       max_tokens=2048, messages=msgs)
    u = getattr(r, "usage", None)
    usage = (getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)
    return r.choices[0].message.content, usage


# USD per 1M tokens (in, out) - ESTIMATE, confirm against current OpenAI pricing.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.0), "gpt-4.1": (2.0, 8.0), "o4-mini": (1.10, 4.40),
}


# ---- arm derivation ----------------------------------------------------------
def arm1_template(item, sb):
    """Existing ReSocratic template (no teacher): parse build_completion output."""
    return parse_blocks(build_completion(item), sb)


def sample_pilot(data, n, seed=0, train_only=True):
    """Sample n problems mixing linear + nonlinear (by `type`), needing a numeric answer.
    train_only: restrict to first 95% (ReSocratic-train) so the ablation never trains on val."""
    n_pool = int(len(data) * 0.95) if train_only else len(data)
    idxs = [i for i in range(n_pool) if data[i].get("answer") is not None]
    nonlin = [i for i in idxs if "nonlinear" in str(data[i].get("type", "")).lower()]
    lin = [i for i in idxs if i not in set(nonlin)]
    random.Random(seed).shuffle(nonlin)
    random.Random(seed + 1).shuffle(lin)
    half = n // 2
    picked = nonlin[:half] + lin[: n - half]
    return sorted(picked)[:n]


def process_one(pid, data, args):
    """Teacher-generate arm3 (with retries) + verify; return arms if kept. Thread-safe (own sandbox)."""
    sb = SCIPSandbox(timeout=15)
    item = data[pid]
    nl, ref_code, y = item["question"], item["code_solution"], item["answer"]
    ti = to = 0; ok = False; ov = None; r3 = f3 = c3 = ""; last = ""
    for _ in range(1, args.retries + 1):
        try:
            raw, usage = teacher_generate(nl, ref_code); ti += usage[0]; to += usage[1]
        except Exception as e:
            last = f"TEACHER-ERROR {e}"; continue
        r3, f3, c3 = parse_blocks(raw, sb)
        if not (r3 and f3 and c3):
            last = "PARSE-FAIL"; continue
        ok, ov = verify(c3, y, sb)
        if ok:
            break
        last = f"VERIFY-FAIL objval={ov} y={y}"
    res = {"pid": pid, "ok": ok, "ov": ov, "last": last, "ti": ti, "to": to}
    if ok:
        shared_code = c3 if args.shared_code == "teacher" else ref_code
        r1, f1, _ = arm1_template(item, sb)
        f2 = strip_selfcheck(f3)
        res["arms"] = {"a3": row(pid, nl, r3, f3, shared_code), "a2": row(pid, nl, r3, f2, shared_code),
                       "a1": row(pid, nl, r1, f1, shared_code), "a0": row(pid, nl, "", "", shared_code)}
    return res


def row(pid, nl, r, f, c):
    return {"problem_id": pid, "nl": nl, "reasoning_block": r,
            "formula_block": f, "code_block": c, "verified": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="data/ablation")
    ap.add_argument("--shared_code", choices=["teacher", "reference"], default="teacher")
    ap.add_argument("--retries", type=int, default=2, help="teacher attempts per problem before discard")
    ap.add_argument("--workers", type=int, default=8, help="concurrent teacher calls")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self_test", action="store_true",
                    help="Validate non-teacher machinery on 3 problems; no API call.")
    args = ap.parse_args()

    data = json.load(open(DATA))
    sb = SCIPSandbox(timeout=15)
    os.makedirs(args.out, exist_ok=True)

    # ---------- self-test: prove arm0/arm1 + verifier work, no teacher ----------
    if args.self_test:
        print("[self_test] validating non-teacher machinery on 3 problems (no API)\n")
        for pid in sample_pilot(data, 3, args.seed):
            item = data[pid]
            r1, f1, c1 = arm1_template(item, sb)
            ok, ov = verify(item["code_solution"], item["answer"], sb)
            print(f"--- problem_id={pid}  type={item.get('type')}  y={item['answer']}")
            print(f"    reference code verifies: {ok}  (objval={ov})")
            print(f"    arm1 reasoning[:90]: {r1[:90]!r}")
            print(f"    arm1 formula[:90]  : {f1[:90]!r}")
            print(f"    arm0 code lines    : {len(item['code_solution'].splitlines())}\n")
        print("[self_test] OK — parsing, verifier, arm0/arm1 derivation all functional.")
        return

    # ---------- real pilot: needs teacher ----------
    if not API_KEY:
        sys.exit("ERROR: DEEPSEEK_API_KEY not set. Set it (and TEACHER_MODEL / DEEPSEEK_BASE_URL) "
                 "before a paid teacher run, or use --self_test to validate the pipeline.")

    pids = sample_pilot(data, args.n, args.seed)
    print(f"[build] pilot n={len(pids)}  teacher={TEACHER_MODEL} @ {BASE_URL}  shared_code={args.shared_code}")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    a0, a1, a2, a3 = [], [], [], []
    kept, tried, teacher_fail, verify_fail = [], 0, 0, 0
    tot_in = tot_out = 0
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, pid, data, args): pid for pid in pids}
        for fut in as_completed(futs):
            r = fut.result()
            tried += 1; done += 1
            tot_in += r["ti"]; tot_out += r["to"]
            if not r["ok"]:
                if "TEACHER-ERROR" in r["last"] or "PARSE" in r["last"]:
                    teacher_fail += 1
                else:
                    verify_fail += 1
                if done % 50 == 0 or "TEACHER-ERROR" in r["last"]:
                    print(f"  [{done}/{len(pids)}] pid={r['pid']} {r['last']}", flush=True)
                continue
            arms = r["arms"]
            a3.append(arms["a3"]); a2.append(arms["a2"]); a1.append(arms["a1"]); a0.append(arms["a0"])
            kept.append(r["pid"])
            if done % 50 == 0:
                print(f"  [{done}/{len(pids)}] kept={len(kept)} tfail={teacher_fail} vfail={verify_fail}", flush=True)

    for name, arm in [("arm0", a0), ("arm1", a1), ("arm2", a2), ("arm3", a3)]:
        with open(os.path.join(args.out, f"{name}.jsonl"), "w") as fo:
            for r in arm:
                fo.write(json.dumps(r) + "\n")

    manifest = {
        "teacher_model": TEACHER_MODEL, "base_url": BASE_URL, "shared_code": args.shared_code,
        "rel_tol": REL_TOL, "seed": args.seed,
        "n_tried": tried, "n_kept": len(kept),
        "teacher_pass_rate": round(len(kept) / tried, 4) if tried else 0.0,
        "teacher_fail": teacher_fail, "verify_fail": verify_fail,
        "per_arm_counts": {"arm0": len(a0), "arm1": len(a1), "arm2": len(a2), "arm3": len(a3)},
        "tokens_in": tot_in, "tokens_out": tot_out,
        "shared_verified_problem_ids": kept,
    }
    # cost projection to full dataset from REAL pilot token usage
    N_FULL = len(data)
    pr = PRICES.get(TEACHER_MODEL)
    if pr and tried:
        cost_pilot = (tot_in / 1e6) * pr[0] + (tot_out / 1e6) * pr[1]
        scale = N_FULL / tried
        manifest["cost_usd_pilot"] = round(cost_pilot, 4)
        manifest["cost_usd_full_projected"] = round(cost_pilot * scale, 2)
        manifest["price_in_out_per_1M"] = pr
        manifest["n_full"] = N_FULL
    json.dump(manifest, open(os.path.join(args.out, "ablation_manifest.json"), "w"), indent=2)

    # ---------- summary ----------
    print("\n===== SUMMARY =====")
    print(f"teacher pass-rate : {manifest['teacher_pass_rate']*100:.1f}%  "
          f"({len(kept)}/{tried})  [teacher_fail={teacher_fail} verify_fail={verify_fail}]")
    print(f"per-arm counts    : {manifest['per_arm_counts']}")
    print(f"tokens            : in={tot_in} out={tot_out}  (model={TEACHER_MODEL})")
    if "cost_usd_full_projected" in manifest:
        print(f"cost (REAL pilot) : ${manifest['cost_usd_pilot']}  ->  projected full "
              f"({manifest['n_full']} items): ${manifest['cost_usd_full_projected']}")
    print(f"written to        : {args.out}/  (+ ablation_manifest.json)")
    print("\n===== 3 SAMPLE PROBLEMS ACROSS ALL 4 ARMS =====")
    for pid in kept[:3]:
        g = {name: next(r for r in arm if r["problem_id"] == pid)
             for name, arm in [("arm0", a0), ("arm1", a1), ("arm2", a2), ("arm3", a3)]}
        print(f"\n########## problem_id={pid} ##########")
        print(f"NL: {g['arm0']['nl'][:200]}")
        for name in ("arm0", "arm1", "arm2", "arm3"):
            r = g[name]
            print(f"\n---- {name} ----")
            print(f"[REASONING] {r['reasoning_block'][:160]}")
            print(f"[FORMULA]   {r['formula_block'][:220]}")


if __name__ == "__main__":
    main()
