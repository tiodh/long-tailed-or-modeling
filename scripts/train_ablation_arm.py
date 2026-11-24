#!/usr/bin/env python
"""Train ONE ablation arm via SFT (reuse run_sft; only the completion content differs per arm).
Monkeypatches build_completion -> the arm's precomputed [REASONING][FORMULA][CODE] and feeds the
arm's problem set. Same QLoRA r64 recipe for all four arms -> a gain is attributable to the arm.

Usage:
  MODEL_FAMILY=qwen .venv-rl/bin/python scripts/train_ablation_arm.py \
    --arm data/ablation_3k/arm2.jsonl --output_dir checkpoints/abl_arm2 --config config_instruct.yaml
"""
import sys, json, yaml, argparse
sys.path.insert(0, "/home/tio/Documents/sirl-pgrpo")
import or_rlvr.data.dataset as ds
import or_rlvr.training.sft as sm

TURN_END = getattr(ds, "_TURN_END", "")


def assemble(r):
    """Reconstruct the arm's completion. arm0 (empty reasoning+formula) = [CODE] only."""
    code = f"[CODE]\n```python\n{r['code_block']}\n```\n[/CODE]{TURN_END}"
    if not r.get("reasoning_block") and not r.get("formula_block"):
        return code
    return (f"[REASONING]\n{r['reasoning_block']}\n[/REASONING]\n"
            f"[FORMULA]\n{r['formula_block']}\n[/FORMULA]\n" + code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--config", default="config_instruct.yaml")
    ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.arm)]
    items = [{"question": r["nl"], "_ac": assemble(r)} for r in rows]
    print(f"[abl-train] arm={a.arm} n={len(items)} -> {a.output_dir}", flush=True)
    print(f"[abl-train] sample completion head:\n{items[0]['_ac'][:200]}\n...", flush=True)

    # patch: completion = arm's own; data loader = arm items (bypass full ReSocratic)
    ds.build_completion = lambda item: item["_ac"]
    sm.load_resocratic = lambda path: items

    cfg = yaml.safe_load(open(a.config))
    cfg["sft"]["output_dir"] = a.output_dir
    cfg["sft"]["num_epochs"] = a.epochs
    # small dataset -> frequent save/eval so load_best_model_at_end has checkpoints
    cfg["sft"]["save_steps"] = 50
    cfg["sft"]["eval_steps"] = 50
    cfg["sft"]["logging_steps"] = 10
    sm.run_sft(cfg)
    print("[abl-train] DONE", flush=True)


if __name__ == "__main__":
    main()
