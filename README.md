# Long-Tailed Data, Uneven Understanding

**What limits small language models in optimization modeling**
Tio Dharmawan, Irvan Zidny, Taesun Yu, Sungchul Choi — 11th GRN-BRI Conference.

This repository holds the evidence behind that paper and the talk: every experiment in the
order it was run, the per-problem outcomes behind each number, the analysis scripts that
produced them, and the slides.

Training code for the pipeline itself lives in a separate repository; what is here is what a
reader needs to check the numbers in the paper.

- **Task.** A natural-language decision problem goes in; a PySCIPOpt program comes out; SCIP
  solves it. An answer is correct when the returned optimum is within a relative error of
  10⁻⁶ of the benchmark key (greedy decoding, pass@1).
- **Evaluation set.** Six public benchmarks, 2,023 problems: OptiBench 605, NL4OPT 289,
  MAMO-Easy 652, MAMO-Complex 211, IndustryOR 100, OptMATH 166. In-distribution checks use
  the ReSocratic validation split (1,115 problems).
- **Hardware.** One RTX 4090 (24 GB) per run, on two machines.
- **Scorer.** All headline numbers come from one format-agnostic scorer, so every model in
  this pack is comparable with every other.

Folders run from the first step to the last. Inside each, `*_fa_summary.json` is the headline
number and `*_fa_rows.jsonl` is the per-problem outcome (`correct`, `wrong_value`, `no_value`)
used for the paired McNemar tests.

---

## Step 0 — Overview (`00_overview/`)

| File | What it holds |
|---|---|
| `SYMPTOM-CURE-2026-09-13.md` | Failure taxonomy and the symptom → cure table |
| `effectiveness_table.json` | Every post-training method vs SFT, in-distribution and on the six benchmarks, with McNemar p-values |

Three corrections are worth reading before any number here is quoted elsewhere:

1. The early claim that OTPO beat SFT by 11.6 points out of distribution was **contamination**:
   422 of its training pairs came from the evaluation benchmarks. On clean pairs the gain is
   +1.6 points (p = 0.052).
2. "Objective direction is never wrong" was true only of the *direction*; the objective's
   *content* is wrong in 70.3% of wrong-value cases.
3. The 17-point logistics gap reported in September was a benchmark-mix confound.

## Step 1 — The training data and what it covers (`01_training_data/`)

| File | What it holds |
|---|---|
| `or_class_cross_table.json` | Share of each of 15 OR classes in the training corpus and in each benchmark |
| `resocratic_or_class_distribution.csv` | Class distribution of ReSocratic with per-class linearity and model size |
| `problem_or_class.csv` | OR class of every one of the 2,023 benchmark problems |
| `ablation_manifest.json` | The 2,814 verified problems used in the training-data ablation (teacher GPT-4.1-mini, pass rate 93.8%) |
| `logior_family_labels.json` | Manual constraint-family labels for the 92 LogiOR problems |

Headline: the corpus is long-tailed. Production planning is 58.0% of training but 11.9% of the
benchmarks; job-shop scheduling is 0% of training and 2.1% of the benchmarks. Only 3.1% of
training programs declare a binary variable and 0.1% write constraints over indexed sets.
An audit finds 23.6% of items broken — they run, but are infeasible or ill-posed.

## Step 2 — Supervised fine-tuning (`02_sft/`)

Qwen2.5-7B-Instruct, QLoRA 4-bit (r = 64), one epoch over ReSocratic-29k.

| Model | Six-benchmark accuracy |
|---|---|
| Base, no fine-tuning | 38.21% (41.37% in the paper, from a run with the larger 8,192-token budget) |
| **SFT (headline model)** | **53.58%** |
| Qwen2.5-Coder SFT | 48.79% |
| Llama-3-8B SFT | 46.12% |

The gain is concentrated in OptiBench and NL4OPT, where the base model writes code that does
not run; the hard benchmarks barely move. Two backbone checks sit here as well: the Coder
variant is worse than the Instruct one, and Llama-3-8B reproduces the same wrong-value floor.

## Step 3 — Reinforcement learning from solver feedback (`03_rlvr/`)

GRPO, reference-model-free (β = 0), 1,000 steps, 10 rollouts per prompt at temperature 1.2,
on 708 prompts whose rollouts differ in reward.

**54.33% — that is +0.74 points over SFT, not significant (exact McNemar p = 0.23; 77 problems
fixed, 62 broken).** Only MAMO-Easy moves on its own (p = 0.022, uncorrected).

Two RLVR runs are in this folder and they are not the same run:
`checkpoints__rlvr_instruct_graded__final__vllm.json` is the headline model (54.33%, 1,099 of
2,023), while `graded_1000_8k_fa_*` is a second graded run that scores 53.83%. Quote the first.

## Step 4 — Does the reward design explain it? (`04_reward_ablation/`)

Two controlled variants against the graded reward, everything else frozen:

| Run | Reward | Accuracy |
|---|---|---|
| R0 | Graded ladder L0–L6 | 54.33% |
| R1 | Additive 0.5/1/2 | 54.57% |
| R2 | Additive + KL (β = 0.02) | 54.37% |

All three are within 0.24 points of each other.

92.2% of problems end identically in all three runs. The floor is not an artifact of reward
magnitude or KL grounding.

## Step 5 — Does an off-policy objective cross it? (`05_offpolicy/`)

From the same SFT checkpoint, on matched pairs that share no problem with any evaluation set.

| Objective | Validation Δ | p | Six benchmarks Δ | p |
|---|---|---|---|---|
| ORPO-B (one constraint removed) | +1.26 | 0.22 | +0.74 | 0.44 |
| ORPO-C (two dimensions perturbed) | −1.88 | 0.09 | −2.67 | 0.006 |
| ORPO-A (own wrong answer) | −3.68 | 0.002 | −5.24 | <0.001 |
| OTPO (own wrong answer) | −3.05 | 0.003 | −0.84 | 0.35 |

Three probes explain the pattern:

- `token_weight_indist_vs_ood.json` — where OTPO's weight lands. Targeting ratio 1.24–1.80 on
  training pairs, 0.89–1.36 on benchmark pairs (1.0 = uniform), because correct and wrong
  answers there differ in 53.7% of their code rather than 35.8%.
- `m1_displacement.json` — what training changes. The log-probability of correct answers falls
  in the reasoning text (−0.0036 nats per token) while objective and constraint tokens move by
  less than 0.001.
- `m3_*` — how the damage accumulates: +0.27 at step 10 (p = 0.81), −2.33 at step 40
  (p = 0.019). No checkpoint beats SFT, so early stopping avoids harm but produces no gain.

## Step 6 — Is it just model size? (`06_frontier/`)

GPT-4o under the identical protocol: **63.63%**, about nine points above our RLVR model.

| | Ours | GPT-4o |
|---|---|---|
| Six-benchmark accuracy | 54.33 | 63.63 |
| Code fails to run | 16.9% | 16.7% |
| Job-shop scheduling | 0.0% | 9.3% |
| OptMATH | 1.8% | 17.5% |

`frontier_clean_rescore.json` re-scores the contaminated OTPO comparison on the 1,601 problems
that were never trained on. LogiOR files hold the logistics benchmark runs, including the
three GPT-4o replications (40.2–41.3%) that exposed the API's non-determinism.

## Step 7 — Where the errors actually are (`07_error_analysis/`)

Every generated program re-executed with stderr captured, then compared against reference code
where one exists.

- Failure composition (in-distribution / six benchmarks / LogiOR): build failures
  6.9 / 20.8 / 39.8%, solve failures 7.9 / 15.5 / 19.3%, wrong value 85.2 / 63.7 / 41.0%.
- Of 576 wrong-value cases, 96.5% involve mis-specified constraints, evenly split between too
  loose (48.1%) and too tight (48.4%); objective direction is wrong in 3.5%.
- On the validation split, 84.8% of deviating constraints connect the wrong variables, and
  **70.3% of wrong-value cases carry a wrong objective expression** against 7.4% of correct ones.
- `error_vs_class_distribution*.json` relate every class's error rates to its share of the
  corpus: accuracy follows the share (Spearman ρ = 0.77), code that fails to run runs against
  it (ρ = −0.66), wrong formulations do not follow it at all (ρ = 0.42, n.s.).

## Step 8 — How the problem is phrased (`08_phrasing_analysis/`)

Cue phrases counted in every problem statement: bounds ("at least"), relational ("twice as
many", "percent"), logical ("if", "either", "fixed cost"), indexed ("each", "per").

- Classes differ sharply: job-shop statements average 381 words and 2.23 logical cues;
  resource allocation averages 186 words and 0.22.
- Across classes, cue density tracks the wrong-value rate (ρ = 0.69 ours, 0.36 GPT-4o).
- Within a class, where coverage is fixed, three or more relational or logical cues raise the
  odds of a wrong formulation: OR 1.43 for our model (Cochran–Mantel–Haenszel p = 0.074) and
  1.56 for GPT-4o (p = 0.024).

Caveat recorded with the result: the cue counts are regular-expression proxies, not annotated
by hand, so these are associations.

## Step 9 — What the training data should contain (`09_data_ablation/`)

Seven training targets over the same 2,814 verified problems, three epochs each. The program is
held fixed; only the text around it changes.

| Training target | Accuracy | Wrong value | No value |
|---|---|---|---|
| Code only (teacher rewrite) | **62.04** | 22.0 | 15.9 |
| Reference program, with clause comments | 61.34 | 25.7 | 13.0 |
| Grounded reasoning + self-verification | 60.90 | 20.5 | 18.6 |
| Template trace + its own reference program | 59.42 | 24.8 | 15.8 |
| Grounded reasoning | 58.77 | 21.0 | 20.2 |
| Reference program, comments removed | 58.28 | 24.9 | 16.9 |
| Template trace + teacher rewrite | 57.69 | 25.9 | 16.4 |

Readings, all McNemar on the same problems:

- Clause comments **inside** the code are worth +3.06 points (61.34 vs 58.28, p = 0.0003).
- The same kind of reasoning moved to a **block before** the program costs 1.92 points
  (59.42 vs 61.34, p = 0.010).
- The teacher's rewrite is not what makes code-only strong (62.04 vs 61.34, p = 0.37).
- A smaller, verified set trained longer beats the full corpus by 5.84 points (59.42 vs 53.58,
  p < 0.001) — descriptive only: problems, epochs and the linear/nonlinear mix all differ.
- `api_repair_bound.json` — repairing only solver-interface calls lifts grounded reasoning from
  58.77 to 60.31, still below code only.

## Step 10 — The talk (`talk/`)

`grnbri2026_talk.pptx` and its PDF are the conference slides, with the two figures they use.
The paper itself appears in the conference proceedings.

## Scripts (`scripts/`)

The analysis programs that produced the files above: class-distribution join, phrasing counts
and the stratified test, the per-target ablation reader, the API-repair bound, the AST failure
taxonomy, the wrong-value symptom comparison, the OTPO weight audit, the likelihood-displacement
probe, the validation scorer, and the builders for the ablation datasets.

---

## What the whole log says, in four lines

1. Training harder does not help: RLVR adds 0.74 points, and no reward or preference variant does better.
2. A frontier model is better overall but fails in the same classes, so the limit is the task.
3. Coverage explains the code that crashes, not the formulations that run and are wrong.
4. What moves accuracy is the data: verified problems (+5.8), clause comments inside the code
   (+3.1), and voting over several answers (23% of always-wrong problems recovered).

---

## Citing

> Dharmawan, T., Zidny, I., Yu, T., & Choi, S. (2026). *Long-Tailed Data, Uneven Understanding:
> What Limits Small Language Models in Optimization Modeling.* 11th GRN-BRI Conference.

Code and scripts are released under the MIT License. The benchmarks (OptiBench, NL4OPT, MAMO,
IndustryOR, OptMATH) and the ReSocratic corpus remain under their own licenses; this repository
redistributes only derived results, not those datasets.
