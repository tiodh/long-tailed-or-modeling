"""Symptom granular untuk kegagalan: bandingkan struktur model yang dihasilkan vs reference.

Hanya bisa di set yang punya reference: in-dist (ReSocratic-val, data/ref_structures.jsonl)
dan LogiOR (reference gurobipy, ditangkap via mock). Soal yang BENAR dipakai sebagai kontrol
(detektor harus hampir diam di sana).

Symptom per soal (boleh lebih dari satu):
  V-missing / V-extra       jumlah variabel keputusan beda (tanpa var 'obj')
  V-int-relaxed             reference punya integer/binary lebih banyak (gen melemaskan)
  V-int-over                gen punya integer/binary lebih banyak
  O-sense                   arah objective (min/max) salah
  O-coef                    koefisien objective beda (multiset |koef|, invariant nama)
  C-missing / C-extra       jumlah constraint beda
  C-topology                ada constraint ref tanpa pasangan berbentuk sama
  C-coef                    bentuk sama, koefisien beda
  C-rhs                     bentuk + koefisien sama, rhs beda
"""
import json, os, sys, collections
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from or_rlvr.reward.structure import capture_structure, _extract_code

VT = {'C': 'CONTINUOUS', 'B': 'BINARY', 'I': 'INTEGER'}

def norm_terms(terms):
    """terms: {tuple(varnames)|'a*b': coef} -> (tuple sorted (arity,|coef| scaled)), scale)."""
    items = []
    for k, v in terms.items():
        names = k if isinstance(k, tuple) else tuple(k.split('*'))
        items.append((names, float(v)))
    return items

def split_obj(cons):
    obj, rest = None, []
    for c in cons:
        items = norm_terms(c['terms'])
        if any('obj' in n for n, _ in items) and obj is None:
            obj = [(n, v) for n, v in items if 'obj' not in n]
        else:
            rest.append((items, float(c['rhs'])))
    return obj, rest

def sig(items):              # bentuk: multiset derajat monomial
    return tuple(sorted(len(n) for n, _ in items))
def coefkey(items):          # bentuk + |koef| diskalakan
    m = max((abs(v) for _, v in items), default=1.0) or 1.0
    return tuple(sorted((len(n), round(abs(v) / m, 6)) for n, v in items)), m
def full(items, rhs):
    ck, m = coefkey(items)
    return ck, round(abs(rhs) / m, 6)

def symptoms(gen, ref):
    s = set()
    gvt = {k: v for k, v in gen['vtypes'].items() if k != 'obj'}
    rv = ref['vtypes']
    if isinstance(rv, dict): rvl = [v for k, v in rv.items() if k != 'obj']
    else: rvl = [VT.get(v, v) for v in rv]
    ng, nr = len(gvt), len(rvl)
    if ng < nr: s.add('V-missing')
    if ng > nr: s.add('V-extra')
    ig = sum(v in ('INTEGER', 'BINARY') for v in gvt.values()); ir = sum(v in ('INTEGER', 'BINARY') for v in rvl)
    if ir > ig: s.add('V-int-relaxed')
    if ig > ir: s.add('V-int-over')
    if gen['sense'] and ref.get('sense') and gen['sense'] != ref['sense']: s.add('O-sense')
    gobj, gcons = split_obj(gen['cons'])
    if 'obj_terms' in ref:
        robj = ref['obj_terms']; rcons = [(norm_terms(c['terms']), float(c['rhs'])) for c in ref['cons']]
    else:
        robj, rcons = split_obj(ref['cons'])
    if gobj is not None and robj is not None and coefkey(gobj)[0] != coefkey(robj)[0]:
        s.add('O-coef')
    if len(gcons) < len(rcons): s.add('C-missing')
    if len(gcons) > len(rcons): s.add('C-extra')
    pool = [full(i, r) for i, r in gcons]
    unmatched = []
    for i, r in rcons:
        k = full(i, r)
        if k in pool: pool.remove(k)
        else: unmatched.append((i, r))
    gsig = [sig(i) for i, _ in gcons]; gck = [coefkey(i)[0] for i, _ in gcons]
    for i, r in unmatched:
        if coefkey(i)[0] in gck: s.add('C-rhs')
        elif sig(i) in gsig: s.add('C-coef')
        else: s.add('C-topology')
    return s

def job(args):
    text, ref, status = args
    g = capture_structure(_extract_code(text))
    if g is None: return None
    return status, sorted(symptoms(g, ref))

def load():
    sets = {}
    refs = {}
    for l in open('data/ref_structures.jsonl'):
        d = json.loads(l)
        if d.get('ok'): refs[int(d['i'])] = d
    tax = [json.loads(l) for l in open('results/ast_taxonomy/in_dist_resocratic_val.jsonl')]
    gens = [json.loads(l) for l in open('results/metrics/qwen_sft_val_gen.jsonl')]
    gens = [g for g in gens if '_meta' not in g]
    items = []
    for g, t in zip(gens, tax):
        i = int(g['i'])
        if i in refs and t['group'] in ('correct', 'wrong_value'):
            items.append((g['text'], refs[i], t['group']))
    sets['in-dist'] = items
    SC = '/private/tmp/claude-501/-Users-tiodh-Projects-experiments-sirl-pgrpo/d3cede75-3be8-4856-89cc-c57dca1f5cfd/scratchpad'
    lref = json.load(open(f'{SC}/logior_ref_struct.json'))
    for v in lref.values():
        v['obj_terms'] = [(tuple(k.split('*')), val) for k, val in (v.get('obj') or {}).items()] or None
    rows = {json.loads(l)['pid']: json.loads(l)['status'] for l in open('results/bench/logior_sft_rows.jsonl')}
    gen = {json.loads(l)['pid']: json.loads(l)['text'] for l in open('results/bench/logior_sft_gen.jsonl')}
    sets['LogiOR'] = [(gen[p], lref[p], rows[p]) for p in lref if p in gen and rows.get(p) in ('correct', 'wrong_value')]
    return sets

if __name__ == '__main__':
    out = {}
    for name, items in load().items():
        with ProcessPoolExecutor(max_workers=12) as ex:
            res = [r for r in ex.map(job, items, chunksize=8) if r]
        out[name] = res
        print(f'[sym] {name}: {len(res)} ter-capture dari {len(items)}', flush=True)
    json.dump(out, open('results/ast_taxonomy/wrong_value_symptoms.json', 'w'))
