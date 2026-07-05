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
2. **LoRA barely forgets (H2).** LoRA shows ≈0 forgetting at *every* mixing ratio, including
   mix0 (+0.002), while learning the task just as well (sim_em 0.897). Full fine-tuning at mix0
   forgets −0.078; LoRA forgets nothing — not because it hides a loss, but because it freezes
   the base weights and only trains a thin adapter, leaving general knowledge untouched by
   construction. Mixing is irrelevant under LoRA (no forgetting to recover). Phase 3 shows the
   same pattern extends to instruction-following and OOD robustness: LoRA genuinely spares the
   costs full-FT pays, at the usual price of learning the new task less aggressively.
3. **Size (H3): larger forgets modestly less under full-FT; LoRA masks it at both sizes.**
   Two ways in:
   - *Within LoRA*, 1.5B and 0.5B both forget ≈0 — LoRA masks forgetting regardless of size.
   - *Full fine-tuning* was made to fit on the 16 GB T4 (paged 8-bit optimizer offloading
     optimizer state to CPU + `expandable_segments` + batch 1; the earlier OOM overshot by only
     ~200 MB). **1.5B full-FT mix0 forgets a mean −0.060 ± (per-task .01–.03) vs 0.5B's −0.078**
     — the larger model forgets **somewhat less**, weakly supporting "smaller models forget
     more." But the effect is modest (~0.018, comparable to the seed std) and **not uniform**:
     1.5B forgets clearly less on arc_easy (−0.083 vs −0.153) and winogrande (−0.006 vs −0.031),
     the same on hellaswag, and slightly *more* on arc_challenge (−0.069 vs −0.048). Both learn
     the task equally (sim_em 0.897). So the size effect is real but soft — far weaker than the
     mixing (#1) or LoRA (H2) effects.

   Per-seed 1.5B full-FT numbers in `dl_b15full_all/`.

**Phase-2 caveats.** 1.5B mixing-sweep axis was run with LoRA; 1.5B **full-FT** was measured
only at mix0 (the discriminating point for H3), via CPU-offloaded paged optimizer. LoRA's
slightly *positive* deltas (~+0.01) are within noise / minor adapter effects, not a real gain.
Aggregates in `axes_summary.json`; 1.5B full-FT in `dl_b15full_all/`.

## Phase 3 — what full-FT actually costs (method probes at mix0)

Same setup, mix0, 3 seeds. Beyond the reasoning/commonsense battery, two extra probes:
IFEval (instruction-following, generative) and held-out simulation on **out-of-distribution**
terminal commands (`build_scenario_hard`: sed/awk/grep/sort/tr/cut/pipes — never in training).

| cell | mean Δ reasoning | ΔIFEval | sim (in-dist) | sim (OOD) |
|---|---|---|---|---|
| instruct, full-FT | −0.086 | **−0.071** (0.194→0.123) | 0.897 | **0.00** |
| instruct, LoRA    | ≈0 (+0.00) | **+0.033** (0.194→0.227) | 0.897 | **0.15** |
| base, full-FT     | −0.073 | n/a (base) | 0.897 | 0.00 |

**Findings.**
1. **Full-FT degrades instruction-following (IFEval −0.071, ~37% relative); LoRA preserves it**
   (+0.033). The reasoning/commonsense battery could not see this — it took a dedicated
   instruction-following eval. This is the concrete capability "general benchmarks barely
   moved" was hiding on the full-FT side.
2. **The learned world model is more brittle under full-FT than LoRA.** Both score ~0.90 on
   in-distribution held-out commands, but on OOD commands full-FT collapses to 0.00 while LoRA
   still gets 0.15. Full-FT overfits the exact command shapes seen; LoRA, riding the frozen
   base, generalizes a little. (Predicted the opposite — "LoRA learns shallower" — and was
   wrong.)
3. **Base vs instruct, refined:** base and instruct forget reasoning about equally; the
   instruct model's distinctive loss is IFEval itself, which a base model has little of to
   begin with.

**Takeaway.** Full fine-tuning on a narrow task pays three costs — general reasoning,
instruction-following, and OOD robustness of the learned task — that a single benchmark
undersells. LoRA mostly avoids them by perturbing the model far less (frozen base weights),
at the usual cost of learning the new task less aggressively (here immaterial: both hit 0.90).
Per-cell aggregates in `batchA_summary.json`.
