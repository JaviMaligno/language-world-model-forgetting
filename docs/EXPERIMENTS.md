# Experiments

## Phase 1 — forgetting vs. data-mixing curve (headline)

**Setup.** Base `Qwen/Qwen2.5-0.5B-Instruct`, full fine-tuning, continual pre-training on
narrow Terminal trajectories (28k scenarios / 252k turns). Mixing ratio = fraction of each
batch drawn from a general-text replay corpus (English Wikipedia, `wikimedia/wikipedia
20231101.en`) instead of terminal data; total token/step budget held constant (lr 5e-5, 500
steps, batch 4 × grad-accum 4, seq 1024). General-ability eval via lm-evaluation-harness
(loglikelihood MC, **full test sets**): arc_challenge, arc_easy, hellaswag, winogrande.
World-model fidelity via held-out terminal next-observation exact-match (`sim_em`), scored
only on output-producing turns. 3 seeds per mixing ratio (control: 1 seed).

**Result** (mean Δ = after − before over seeds, ± std; negative = forgetting):

| mixing | n | Δarc_c | Δarc_easy | Δhellaswag | Δwinogrande | **mean Δ** | sim_em (after) |
|---|---|---|---|---|---|---|---|
| 0%   | 3 | −0.048 ± .016 | **−0.153 ± .015** | −0.078 ± .002 | −0.031 ± .014 | **−0.078** | 0.897 |
| 10%  | 3 | −0.026 ± .008 | −0.028 ± .009 | −0.025 ± .001 | −0.007 ± .007 | **−0.021** | 0.897 |
| 25%  | 3 | −0.016 ± .010 | −0.004 ± .005 | −0.017 ± .000 | −0.011 ± .004 | **−0.012** | 0.897 |
| 50%  | 3 | −0.015 ± .005 | −0.010 ± .003 | −0.022 ± .002 | −0.014 ± .003 | **−0.015** | 0.879 |
| 100% (control) | 1 | −0.011 | −0.014 | −0.020 | −0.009 | **−0.013** | 0.253 |

(sim_em baseline before training = 0.356.) Eval on full test sets; an earlier limit-200 pass
showed the same trend with looser noise.

**Findings.**
1. **CPT on narrow trajectories causes measurable, robust forgetting** — mean −0.078, worst on
   arc_easy (−0.153 ± .015, i.e. ≫ noise) — while the model learns the world-model task
   (sim_em 0.356 → 0.897).
2. **A small replay fraction recovers most of it, for free.** 10% mixing cuts mean forgetting
   from −0.078 to −0.021 (~73% recovered; ~82% on arc_easy: −0.153 → −0.028) with **no cost to
   task learning** (sim_em unchanged at 0.897). This is exactly the data-mixing benefit
   Qwen-AgentWorld asserts by design but never measures.
3. **Diminishing returns past ~10–25%** — 25% reaches mean −0.012 (arc_easy ≈0); 50% is no
   better and starts to nibble task acquisition (sim_em 0.879).
4. **Negative control (100% replay) brackets the design:** no terminal data → no task learning
   (sim_em falls to 0.253, below the 0.356 baseline) and ≈flat general ability.

**Caveats.** Toy scale (0.5B); sim_em near-saturated (~0.90) because the Terminal task is
easily learned; battery is reasoning+commonsense (arc/hellaswag/winogrande) — mmlu/triviaqa/
ifeval deferred (lm-eval 0.4.4 loading issues on this stack). Forgetting axis is consistent
and tight across 3 seeds.

**Infra.** AML command jobs on a scale-to-zero T4 cluster (`lwmf-t4cl`, NC4as_T4_v3) in
workspace `model_server` (australiaeast); offline HF cache (model + datasets) + local
Wikipedia replay JSONL; checkpoints to local tmp (not persisted). Per-run JSON in each job's
`results` output; aggregates in `curve_full_summary.json`.

## Phase 2 — method / flavor / size axes

Same setup as Phase 1, varying one axis at a time; full test sets, 3 seeds (control 1 seed),
mean Δ over seeds (negative = forgetting).

**Mean forgetting (avg over the 4 tasks) vs mixing:**

| mixing | full-FT Instruct (P1) | full-FT Base | LoRA 0.5B | LoRA 1.5B |
|---|---|---|---|---|
| 0%   | −0.078 | −0.077 | **+0.002** | **−0.000** |
| 10%  | −0.021 | −0.014 | +0.005 | +0.006 |
| 25%  | −0.012 | −0.002 | +0.009 | +0.011 |
| 50%  | −0.015 | −0.010 | +0.009 | +0.015 |
| 100% | −0.013 | −0.006 | +0.009 | — |

sim_em (task learned) at mix0 ≈ 0.90 for every method; unchanged by method/size.

**Findings.**
1. **Base ≈ Instruct (H4).** Full-FT forgetting curves are near-identical for the base and
   instruct 0.5B models (−0.077 vs −0.078 at mix0, same recovery). The "instruct has more to
   lose" intuition does not hold on reasoning/commonsense tasks. (The instruct-specific loss
   would be in instruction-following / IFEval, which is deferred — our battery can't see it.)
2. **LoRA hides the forgetting entirely (H2) — the headline of Phase 2.** LoRA shows ≈0
   forgetting at *every* mixing ratio, including mix0 (+0.002), while learning the task just as
   well (sim_em 0.897). Full fine-tuning at mix0 forgets −0.078; LoRA forgets nothing. The
   cheap/default method makes CPT *look* safe because it barely perturbs the base weights —
   only full-FT (which actually moves the model) reveals the forgetting. Mixing is irrelevant
   under LoRA (no forgetting to recover). This is a "measure with the wrong method → false
   security" result.
3. **Size, within LoRA (H3): also ≈0 at both sizes.** 1.5B-LoRA forgets ≈0 like 0.5B-LoRA, so
   LoRA masks forgetting regardless of size. The sharper "does full-FT 1.5B forget
   more/less than 0.5B" could **not** be tested: 1.5B full fine-tuning OOMs on the 16 GB T4
   (params+grads+optimizer > 16 GB), so the size axis had to be run with LoRA. Honest hardware
   limitation.

**Phase-2 caveats.** 1.5B axis is LoRA-only (full-FT infeasible on T4). LoRA's slightly
*positive* deltas (~+0.01) are within noise / minor adapter effects, not a real gain.
Aggregates in `axes_summary.json`.
