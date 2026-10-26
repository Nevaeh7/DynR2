# Method And Release Scope

This release extracts the method kernels from the implementation used for the manuscript. The model operations, protected-response loss, training-schedule construction, full-vocabulary beam scoring, belief features, and Ridge heads are retained. Portable command-line entry points replace experiment-specific data manifests and cluster launchers.

## Method Correspondence

| Paper component | Implementation |
| --- | --- |
| Residual k-means semantic identifiers | `prep_kernels._run_rq_kmeans`, `preprocess.construct_sids` |
| Chronological data splits and response input export | `preprocess.prepare_splits` |
| SID vocabulary expansion and initial recommendation training | `initialize`, `run_train.sh` |
| Answer-aligned belief observations | `DynR2.observe`, `DynR2Base._recommendation_belief` |
| Explicit and probability-weighted latent inputs | `DynR2.advance`, `DynR2._answer` |
| Entropy, unnormalized JS revision, Rule decisions | `normalized_entropy`, `js_divergence`, `decide_mode` |
| Protected history and initial Direct predictor | `DynR2.prefill`, `DynR2.adapter_scope` |
| Seven-token response supervision | `DynR2.forward`, `DynR2._target_logits` |
| Four-code fixed-reference KL | `legal_sid_kl` |
| Target-independent training layouts | `schedule` |
| Full-vocabulary, temperature/weight-adjusted catalog decoding | `cached_beam` |
| Exact-item collision expansion | `Catalog.expand` |
| Two signed continuation-gain predictors | `ridge`, `predict`, `choose_value` |
| Complete-request timing | `evaluate.request` |

The release has no baseline, ablation, visualization, queue-management, or report-generation code. It includes the necessary data preparation, semantic-identifier construction, SID vocabulary expansion, alignment, and initial-recommender training stages. Preprocessing preserves the reference quantization, chronology, history formatting, and SID-transition construction; experiment-specific difficulty reports and auxiliary evaluation subsets are omitted. No synthetic result is presented as a measured benchmark result.

## Reproduction Conditions

Keep the initial predictor, tokenizer, catalog, prompt text, history truncation, response adapter, decoder configuration, and collision counts fixed across matched policy comparisons. A response adapter must not be merged into the initial predictor. Base history encoding is adapter-free. The first continuation input is formed from the frozen initial logits; subsequent appended positions use response-adapted model operations.

Use explicit prepared files for checkpoint selection, controller fitting, policy selection, and final evaluation. Controller cohorts are disjoint by user, and all sampled states from one user belong to one cohort. Checkpoint-selection users are excluded from all controller cohorts. The paper's main SID benchmark and its controlled exact-item studies are distinct evaluation protocols; their profiles and cohorts should not be interchanged when comparing reported values.

The released deterministic state sampler provides an executable state-bank construction. Its portable sampling key is not a replacement for archived experiment partitions. Exact numerical reproduction requires the original input artifacts and selected model weights. The domain profiles specify inference settings, not claims that any newly trained checkpoint will match the paper's tables.

For latency comparisons, use the same GPU, warm-up, sample order, and repeat count. Rotate policy order across repeated measurements when comparing policies, and aggregate repeats within users before reporting quantiles. The evaluation command measures one policy per invocation. Counterfactual bank timing is used for development cost calibration, not deployment latency.

## Checks

The tests cover raw-record parsing, actual FAISS quantization, chronological split export, initial-model vocabulary expansion and training, padded vocabulary loading, frozen-Direct preservation after response updates, answer-query isolation, explicit/latent state transitions, top-one equivalence, controller ties, finite-horizon query counts, seven-token supervision, KL support, weighted full-vocabulary decoder scores, collision priors, user-disjoint fitting, and inference-profile validation. These are implementation checks using tiny fixtures, not full model-quality experiments.
