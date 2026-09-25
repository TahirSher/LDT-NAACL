# Critical review for NAACL: what changed in `LDT.py` and why

`LDT.py` is the complete, updated script (one file, about 11.8k lines).
Run the unit tests with `python LDT.py --self-test` (14 tests). By default they also run automatically before every experiment.

---

## 1. Verdict

The design is sound and in places careful: cross-fitted probes, a fixed prompt, left-padding-safe positions and paired McNemar tests. The code as submitted, however, was **not** NAACL-ready, for two reasons.

1. **Several reported numbers were wrong or always missing** (bugs in §2).
2. **The central claim had no control against the most obvious alternative explanation.** A high WORD/NONWORD probe accuracy is uninterpretable until it is compared with:
   - (a) the same probe on a **randomly initialised** model (A2);
   - (b) classifiers that never see a hidden state: character n-grams, the tokenizer's own sub-word IDs, and length / Ortho_N / token count (**A6**, which I added; it is not in your list).

   On the synthetic smoke-test data, a character n-gram baseline beat the "LLM" probe (93% vs 88%). ELP pseudowords are orthographically legal, but they are *not* statistically indistinguishable from words. If A6 comes close to your primary curve on real data, the paper's framing has to change. Run A2 and A6 before anything else.

---

## 2. Bugs found in the original code (all fixed)

| # | Bug | Effect | Fix |
|---|-----|--------|-----|
| 1 | Cross-model transfer reloaded the fast linear probe (`_LinearProbeShim`, keys `_lin.*`) into `LexicalDecisionClassifier` (keys `out.*`) | `load_state_dict` raised, so **every transfer row was NaN** and carried the exception in `skip_reason` | The linear probe is stored as its logit `(w, b)`; the closed-form check reproduces the pipeline's predictions at 100% |
| 2 | `auc_difference = hf['auc'] − lf['auc']` computed on word-only subsets | AUC is undefined for one class, so the column was **always NaN** | AUC(HF vs nonwords) − AUC(LF vs nonwords) on shared test nonwords, DeLong variance |
| 3 | `p = 2*(1 − norm.cdf(z))` | Underflowed to **p = 0.0** | p-values computed from log survival functions; `*_log10` columns everywhere; a Mills-ratio tail fallback |
| 4 | Standardised RT β used the SD of *raw* RT while the regression was on *log* RT | `reg_beta1_std` was on the wrong scale | Uses the SD of the dependent variable actually regressed |
| 5 | Confound matching: the comment said "sort by best match distance", but items were processed in file order; there was no caliper | Order-dependent matches; no rejection rule | Order-independent greedy matching with a per-covariate caliper; rejections counted |
| 6 | Single-token and token-matched subsets never received FDR correction | `freq_p_fdr` was NaN for those tables | `apply_frequency_fdr` runs inside the producer (J3) |
| 7 | Metadata described the primary probe as `nn.Linear` with early stopping | False: it was sklearn `liblinear`, C = 1, no early stopping. liblinear also **penalises the intercept** | Metadata now tells the truth. Default solver is now `lbfgs` (textbook L2 logistic regression); `PRIMARY_PROBE_SOLVER='liblinear'` reproduces the old numbers |
| 8 | RT tables labelled `holdout_linear_probe` | They are the cross-fitted scores | Relabelled `crossfitted_linear_probe` |
| 9 | "sklearn LR vs Primary probe" plot used the hold-out probe as "primary" | Mislabelled figure (J4) | Compared against the cross-fitted curve |
| 10 | Contextual analysis: 200 items, one 15% split | **30 test items** (accuracy SE ≈ 0.09); underpowered | 1000 items, every item scored out-of-fold, Wilson CIs |
| 11 | Hard-coded MIG UUID in `CUDA_VISIBLE_DEVICES` | Not reproducible (J1) | Set only when `LDT_GPU_INDEX` is given |

A note for the Limitations section (not a bug): `hidden_states[-1]` is the output of the **final norm**, while every other layer is the raw residual stream. The last point of every curve is therefore a different kind of state. This is now written into the metadata.

---

## 3. Where your master list is wrong or ill-posed

You asked me not to be generous, so here is each problem stated plainly.

| Item | Problem | What the code does instead |
|------|---------|----------------------------|
| **B4, B5** — regress on / control for `log_freq` over words **and nonwords** | **Not estimable.** Nonwords have no corpus frequency: it is undefined, not zero. | B4: erase length / Ortho_N / token count *from the hidden states* inside each training fold (OLS residualisation, which gives zero cross-covariance, i.e. linear guardedness; LEACE Thm 3.1), then probe. B5(i): words only, `S ~ log_freq + covariates` (HC3 SEs, Bayes factor). B5(ii): all items, likelihood-ratio test of `is_word ~ covariates + S` against covariates only. |
| **E2** — cluster-robust SEs with `groups=freq_group` | **3 clusters.** The cluster sandwich estimator needs dozens (Cameron & Miller, 2015); with 3 it is badly biased. | HC3 SEs. Cluster SEs are reported only when there are ≥ 20 clusters; otherwise NaN with the reason. |
| **E3** — `(1\|freq_group) + (1\|length_bin)` with `log_freq` as a fixed effect | `freq_group` is a deterministic discretisation of `log_freq`, so the random intercept is confounded with the fixed slope. A variance cannot be estimated from 3 levels. ELP gives item *means*, so there is no subject/trial structure. | `RT ~ S + log_freq + Ortho_N + tokens + (1 \| word_length)`, a sensitivity analysis only. |
| **F3** — filler sentence "with no target word" | **Vacuous.** Identical inputs give identical hidden states, so chance accuracy holds by construction. | Derangement control: each carrier holds a *different* sampled word while the labels stay with the original item. |
| **B3** — KS / t-test p-values as balance evidence | The "balance test fallacy" (Imai, King & Stuart, 2008): the p-values depend on the matched sample size. | Reported as asked, but SMD (< 0.1) and variance ratio are the pre-specified criteria. Love plot (I3). |
| **E1** — human accuracy as a covariate | Accuracy and RT are joint outcomes of the same decision, so this is a potential **bad control**. | The ≥ 0.8 accuracy *filter* is the primary sample; the covariate model is sensitivity only and flagged. |
| **G3** — FDR "for Cohen's h" | Cohen's h is an effect size and has no test of its own; its test *is* the two-proportion z-test. | `p_value_cohens_h_corrected` = the corrected two-proportion p (documented). |
| **C1** — magnitudes {0.1 … 5} in z-units | In 4096-dimensional standardised space ‖x‖ ≈ 64, so 0.5 z-units is negligible. Without a null the curve cannot be interpreted. | Magnitudes are in SD of the training projection, with a null of 20 random k-sparse directions. |
| **C3** — zero-ablation | Pushes activations off-distribution (Wang et al., 2023). | Mean-ablation by default (`HEAD_ABLATION_MODE`), with a null of k random heads. |
| **C4 / H3** — shuffled-pair permutation null only | A shuffled map outputs roughly the mean, so this null is almost always rejected and is weak evidence. Real transfer is also expected whenever both models encode frequency linearly. | Adds a **random-probe-direction null**. Both nulls are exact in closed form, so 1000 permutations are cheap. |
| **H1 / M3** — "scaling laws" | 9 models, size confounded with family, data and tokens. A 9-point OLS supports no scaling-law claim. | Reported as descriptive: HC3 slope, bootstrap CI, a within-family slope and an explicit caveat. Do not cite Kaplan / Hoffmann as support. |
| **H2** — correlate accuracy curves | Monotone curves correlate highly by construction. | Also the correlation of first differences (detrended), plus **linear CKA** between models (added). |
| **K6** — "pre-registration" | The plan is written after seeing results, so it is **not** a pre-registration. Calling it one would be a misrepresentation. | `analysis_plan.json` separates confirmatory from exploratory analyses and says explicitly that it is post hoc. |
| **A4** | Was already enforced (`_verify_final_positions`). | Now counted and written to metadata. |
| **K1–K5, M1–M3** | Writing, not code. | K1–K5 are encoded in the metadata (`claim_boundary`, `multiple_comparisons_procedure`). M1–M3 belong in the paper text. |

---

## 4. Implementation map (per-model and global CSVs in `OUTPUT_DIR/naacl_revision/`)

| Item | Where in `LDT.py` | Output |
|------|-------------------|--------|
| A1 MLP probe `[512,256]`, residual, all layers, paired McNemar (Holm) | `FrequencyAnalyzer._fit_predict_mlp`, `_crossfit_table(probe='mlp')` | `A1_mlp_vs_linear_probe.csv` |
| A2 random-init model | `AllLayerExtractor(random_init=True)`, `_random_init_control` | `A2_random_init_vs_trained.csv` |
| A3 attn_out / mlp_out / resid_pre / block_out (additivity verified) | `ModelLevelAnalyses.sublayer_decomposition` | `A3_sublayer_decomposition.csv` |
| A4 identity check | `_verify_final_positions` | `experiment_metadata.json` |
| A5 mean-pool, cross-fitted, same folds, Holm | `_process_model` + `_paired_vs_primary` | `A5_crossfitted_representation_controls.csv` |
| **A6 (added)** surface baselines | `SurfaceFormBaselines` | `A6_surface_form_baselines.csv` |
| B1–B3 caliper, SMD, VR, KS, t | `confound_matched_analysis`, `greedy_caliper_match` | `B1_B3_matching_balance.csv`, I3 |
| B4 / B5 / B6 | `_crossfit_table(covariates=…)`, `frequency_covariate_models` | `B4_*.csv`, `B5_*.csv` |
| C1 dose-response + null | `intervention_analysis` | `C1_*.csv`, I4 |
| C2 steering / C5 INLP mean-ablation | `steering_and_subspace_ablation` | `C2_*.csv`, `C5_*.csv` |
| C3 head attribution + ablation | `head_ablation` | `C3_*.csv` |
| C4 / H3 / H4 transfer nulls + McNemar | `_transfer_nulls`, `cross_model_transfer_analysis` | `C4_H3_H4_transfer_with_nulls.csv`, I5 |
| D1 attention mass (raw and per token) | `attention_mass` + instrumented attention | `D1_*.csv`, I6 |
| D2 attention knockout + non-stimulus control | `attention_knockout` | `D2_attention_knockout.csv` |
| D3 stimulus first / middle / last | `position_sensitivity` | `D3_*.csv` |
| D4 token-count strata | `tokenization_stratified_primary` | `D4_*.csv` |
| E1–E6 | `HumanRTAlignmentAnalysis` | `human_rt_alignment/results/*` (mixed model, diagnostics, overlap, sensitivity), I7 |
| F1–F3 | `contextual_analysis` | `F_contextual_conditions.csv` |
| G1–G6 | module-level statistics utilities | every table |
| G4, H1, H2, I2 | `_cross_model_naacl` | `G4_*.csv`, `H1_*.csv`, `H2_*.csv`, `H2b_*` (CKA) |
| J1–J10 | Config validation, seeds, environment snapshot, self-tests | `experiment_metadata.json` |

---

## 5. How this was verified, and what was not

- **Unit tests (14):** McNemar, bootstrap CI, stitching map, p-value underflow, Cohen's h CI, DeLong vs Mann–Whitney, Bayes-factor direction, caliper matching and SMD, CKA invariances, residualisation guardedness, INLP, torch-vs-sklearn logistic equivalence, direction backends.
- **End-to-end runs:** two tiny random Llama models of different width and depth, with synthetic ELP-format CSVs, on CPU and transformers 5.17. Every analysis A–J ran and wrote output with zero errors, except H1: it needs ≥ 3 models, so it was exercised separately on 4 synthetic model summaries and recovered the planted slope. Checks: the closed-form transfer reproduces the pipeline 100%; sublayer additivity is exact; random models give chance-level transfer and knockout effects; the torch solver matches sklearn to 3 decimal places.
- **Not verified here:** real 1–8B models, a GPU, or your ELP files. The attention instrumentation uses `AttentionInterface` (transformers ≥ 4.48) and was tested on 5.17 only. Run `--self-test`, then one small model with `MAX_SAMPLES=2000`, before launching all 9.

**Compute.** A2 doubles the extraction, and A5 and B4 add full sets of cross-fits. On CUDA, `PRIMARY_PROBE_BACKEND='auto'` moves every logistic fit to a GPU full-batch L-BFGS solver (the same objective as sklearn), which offsets most of this. The knobs are `MODEL_LEVEL_*`, `MLP_PROBE_LAYERS`, `RANDOM_INIT_MAX_ITEMS` and `N_RANDOM_DIRECTIONS_MODEL_LEVEL`.

---

## 6. Weaknesses that code cannot fix (state them in Limitations)

- **One hand-written prompt.** D3 tests reordering, not paraphrase; the prompt is a single point in prompt space.
- **Base English models, one task, one dataset.** ELP pseudowords only; no lexicality-by-morphology analysis.
- **Frequency confounds left uncontrolled:** bigram / letter-sequence probability, age of acquisition, morphological family size.
- **Decodability ≠ use.** Only the C2 / C3 / C5 / D2 interventions address use. In the synthetic run their effects were at the random-direction null, so report nulls honestly if they stay there on real models.
