# Experiments

## Phase 1 — forgetting vs. data-mixing curve (headline)

**Setup.** Base `Qwen/Qwen2.5-0.5B-Instruct`, full fine-tuning, continual pre-training on
narrow Terminal trajectories (28k scenarios / 252k turns). Mixing ratio = fraction of each
batch drawn from a general-text replay corpus (English Wikipedia, `wikimedia/wikipedia
20231101.en`) instead of terminal data; total token/step budget held constant (lr 5e-5, 500
steps, batch 4 × grad-accum 4, seq 1024). General-ability eval via lm-evaluation-harness
(loglikelihood MC, limit 200): arc_challenge, arc_easy, hellaswag, winogrande. World-model
fidelity via held-out terminal next-observation exact-match (`sim_em`), scored only on
output-producing turns. 3 seeds per mixing ratio (control: 1 seed).

**Result** (mean Δ = after − before over seeds; negative = forgetting):

| mixing | n | Δarc_c | Δarc_e | Δhellaswag | Δwinogrande | **mean Δ** | sim_em (after) |
|---|---|---|---|---|---|---|---|
| 0%   | 3 | −0.070 | −0.133 | −0.065 | −0.032 | **−0.075** | 0.897 |
| 10%  | 3 | −0.037 | −0.008 | −0.018 | +0.013 | **−0.012** | 0.897 |
| 25%  | 3 | −0.028 | −0.003 | +0.007 | +0.008 | **−0.004** | 0.897 |
| 50%  | 3 | −0.038 | +0.013 | +0.003 | +0.005 | **−0.004** | 0.885 |
| 100% (control) | 1 | −0.060 | +0.005 | −0.015 | +0.010 | **−0.015** | 0.259 |

(sim_em baseline before training = 0.356.)

**Findings.**
1. **CPT on narrow trajectories causes measurable forgetting** — mean −0.075, worst on
   arc_easy (−0.133) — while the model learns the world-model task (sim_em 0.356 → 0.897).
2. **A small replay fraction recovers most of it, for free.** 10% mixing collapses mean
   forgetting from −0.075 to −0.012 (~84% recovered) with **no cost to task learning**
   (sim_em unchanged at 0.897). This is exactly the data-mixing benefit Qwen-AgentWorld
   asserts by design but never measures.
3. **Diminishing returns past 10%** — 25%/50% reach ≈0 forgetting; 50% starts to nibble task
   acquisition (sim_em 0.885).
4. **Negative control (100% replay) brackets the design:** no terminal data → no task
   learning (sim_em falls to 0.259, below the 0.356 baseline) and ≈flat general ability.

**Caveats.** Toy scale (0.5B); eval at limit 200 (some noise, esp. arc_challenge / single-seed
control); sim_em is near-saturated (0.90) because the Terminal task is easily learned;
mmlu/triviaqa/ifeval deferred (lm-eval 0.4.4 loading issues). The forgetting axis is
nonetheless consistent across 3 seeds.

**Infra.** AML command jobs on a scale-to-zero T4 cluster (`lwmf-t4cl`, NC4as_T4_v3) in
workspace `model_server` (australiaeast); offline HF cache + local Wikipedia replay JSONL.
Run names recorded in the controller session; raw per-run JSON in each job's `results` output.
