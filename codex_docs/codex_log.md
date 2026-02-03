# Codex Log — SyntheMol

Date: 2026-02-03

## Repo overview

SyntheMol is a Python package for generating synthetically accessible molecules using two search strategies:
- SyntheMol-RL: reinforcement learning over reaction graphs.
- SyntheMol-MCTS: Monte Carlo tree search over reaction graphs.

The core idea is to explore large combinatorial chemical spaces (Enamine REAL Space and WuXi GalaXi) using a property predictor (Chemprop or sklearn models) to guide search. The repo bundles building block CSVs and reaction-to-building-block mappings under `synthemol/resources/` and exposes a CLI entrypoint `synthemol` implemented in `synthemol/generate/generate.py`.

High-level structure I observed:
- `synthemol/`: main package.
  - `generate/`: generation logic, RL models, scoring, and tree search utilities.
  - `models/`: wrappers for Chemprop and sklearn models.
  - `reactions/`: reaction definitions and loaders for REAL/WuXi/custom.
  - `resources/`: building blocks and reaction mappings for REAL and WuXi.
  - `constants.py`: all schema, column names, sizes, and defaults.
- `scripts/`: training, prediction, data processing, filtering, plotting.
- `docs/`: reproducibility instructions for RL and MCTS papers.
- `requirements_rl.txt`, `requirements_mcts.txt`, `setup.py`: pinned deps and package metadata.

## Key workflows and code paths

Generation workflow (CLI):
- `synthemol/generate/generate.py` orchestrates end-to-end generation, wiring together
  - scorers (`synthemol/generate/scorer.py`),
  - RL models (`synthemol/generate/rl_models.py`),
  - reactions and building blocks (`synthemol/reactions/*`, `synthemol/constants.py`),
  - and data persistence (`synthemol/generate/utils.py`).

Scoring:
- `ChempropScorer` (in `synthemol/generate/scorer.py`) loads Chemprop checkpoints and predicts per-molecule scores via helper functions in `synthemol/models/chemprop_models.py`.
- Non-model scorers include QED, cLogP, wavelength filtering, and an sp2 network heuristic.

Model training/prediction wrappers:
- `scripts/models/chemprop_models.py` provides training/prediction wrappers for Chemprop v1 (TrainArgs, MoleculeModel, chemprop.train, chemprop.utils).
- `scripts/models/train.py` and `scripts/models/predict.py` are CLI-like utilities that drive those wrappers and do cross-validation, model persistence, and CSV output.

RL models:
- `RLModelChemprop` in `synthemol/generate/rl_models.py` builds or loads Chemprop models and runs them inside the RL loop using Chemprop’s BatchMolGraph/MolGraph.
- `RLModelMLP` uses RDKit fingerprints and a small MLP as an alternative.

Docs and reproducibility:
- `docs/rl` and `docs/mcts` provide long-form scripts for reproducing paper results. Many examples assume Chemprop v1 commands (`chemprop_train`, `chemprop_predict`) and v1 checkpoint layouts.

## Reflections

- The repo is organized around clear research workflows rather than a minimal library API. The scripts and docs are first-class artifacts.
- The Chemprop integration is deep and appears tightly coupled to Chemprop v1 APIs and CLI. This is both a strength (reproducibility) and a modernization risk.
- The RL and MCTS code paths are cleanly separated, but they share core constants and resource files, so changes to chemical space schemas are centralized.

## Compatibility assessment with the latest Chemprop

The repo currently pins Chemprop v1.x (1.6.1 in `setup.py` and `requirements_rl.txt`, 1.5.2 in `requirements_mcts.txt`). The latest Chemprop release is v2.2.2 (Jan 19, 2026), which is a ground-up rewrite with a new CLI (`chemprop train/predict/convert`) and new Python APIs (MPNN models, Lightning-based training, new save/load helpers). A direct upgrade will break multiple modules and docs.

### Primary breaking points

1. Python API changes
- Current code imports and uses `chemprop.args.TrainArgs`, `chemprop.models.MoleculeModel`, `chemprop.train.train/predict`, `chemprop.utils.load_checkpoint/load_scalers`, `chemprop.data.MoleculeDataLoader/MoleculeDataset/MoleculeDatapoint`, and `chemprop.features.BatchMolGraph/MolGraph`.
- Chemprop v2 exposes an `MPNN` model and new save/load utilities in `chemprop.models.utils`, and a reorganized data pipeline. The old classes/functions are not the primary API surface in v2.

2. CLI changes
- Docs and scripts rely on `chemprop_train` and `chemprop_predict`, which were v1 console scripts. In v2 the CLI is `chemprop train`, `chemprop predict`, and `chemprop convert`.

3. Checkpoint formats and loading
- v1 checkpoint loading uses `load_checkpoint` and `load_scalers` with `.pt` files.
- v2 expects model files that include hyperparameters and state dict, and supports `.ckpt` via Lightning. Loading is done via `MPNN.load_from_file` or `MPNN.load_from_checkpoint`.
- There is a v1 → v2 conversion command (`chemprop convert --conversion v1_to_v2`) for checkpoint migration.

### What is needed to make this repo compatible

Priority 1: update Chemprop integration to v2 APIs
- Replace uses of `TrainArgs`, `MoleculeModel`, and `chemprop.train` in:
  - `synthemol/models/chemprop_models.py`
  - `scripts/models/chemprop_models.py`
  - `synthemol/generate/rl_models.py`
- Decide on a v2 data pipeline for both single-molecule scoring and RL training. Likely path:
  - Use `chemprop.data.MoleculeDatapoint/MoleculeDataset/build_dataloader` to construct batches.
  - Replace direct `model(batch=..., features_batch=...)` calls with the v2 forward signature (e.g., `model(bmg, V_d, X_d)` style or equivalent).
- Replace model loading and scaler handling with v2 `save_model/load_model` or `MPNN.load_from_file` and v2 scaling utilities.

Priority 2: update CLI docs and scripts
- Update README and docs to use the v2 CLI (`chemprop train/predict`) and new flag names.
- Update any references in `docs/rl/*` and `docs/mcts/*` that are tied to v1 argument names, v1 data split flags, and v1 checkpoint layouts.

Priority 3: handle checkpoint conversion and backward compatibility
- Add a migration path for existing v1 models, either by:
  - running `chemprop convert --conversion v1_to_v2` before use, or
  - supporting both v1 and v2 checkpoints with a small compatibility layer.
- Ensure feature generator mode (e.g., RDKit features) is aligned with the conversion guidance (v1 featurizer mode when converting v1 checkpoints).

Priority 4: update dependency pins and validate
- Move `chemprop==1.6.1`/`1.5.2` to a v2 pin (e.g., `chemprop>=2.2.2`), and verify transitive dependency compatibility (Lightning, Torch).
- Validate runtime with small smoke tests for:
  - Chemprop scoring in `ChempropScorer`.
  - RL training loop with `RLModelChemprop`.
  - Prediction scripts in `scripts/models/predict.py`.

### Suggested next steps

1. Decide on whether to fully upgrade to Chemprop v2 or to keep v1 for reproducibility and add a parallel v2 path.
2. If upgrading, start by refactoring the thin wrappers (`synthemol/models/chemprop_models.py` and `scripts/models/chemprop_models.py`), then update RL model integration.
3. Add a minimal test or smoke script to verify the new Chemprop path before updating the docs.


## Detailed migration plan to Chemprop v2.2.2 (with alternatives)

This plan documents the steps and decision points before any code changes. It assumes Chemprop v2.2.2 and its Lightning-based API and CLI.

### Plan A: Full upgrade (v2-only, clean break)

1. Preflight inventory
- Identify every import/use of Chemprop v1 APIs.
- Confirm which code paths must be retained (RL, MCTS, training, prediction, plotting scripts).
- Collect current model artifacts and determine which must be converted or retrained.

2. Dependency and environment update
- Update `setup.py`, `requirements_rl.txt`, and `requirements_mcts.txt` to `chemprop==2.2.2` (or a constrained range).
- Validate compatible versions of `torch`, `lightning`, `rdkit`, `numpy`, `pandas`, `scikit-learn`, and `chemfunc`.
- Decide whether to include `lightning` explicitly or rely on Chemprop’s dependency chain.
- Update Python version constraints if Chemprop v2 requires a higher minimum.

3. Replace core Chemprop wrappers (package-level)
- Rewrite `synthemol/models/chemprop_models.py` to use v2 APIs:
  - Build models using `chemprop.models.MPNN` and `chemprop.nn` components (message passing, aggregation, predictor).
  - Replace `load_checkpoint`/`load_scalers` with `MPNN.load_from_file` (or `load_from_checkpoint`).
  - Replace direct `model(batch=..., features_batch=...)` calls with v2 signature (`model(bmg, V_d, X_d)`).
  - Map existing fingerprints (RDKit/Morgan + optional H2O features) to `X_d` (descriptor features) in v2 datapoints.
  - Remove manual scaling where v2 already handles scaling via output transforms.

4. Update training and prediction wrappers (scripts)
- Refactor `scripts/models/chemprop_models.py` to v2 data pipeline:
  - Use `MoleculeDatapoint` + `MoleculeDataset` and `build_dataloader` to construct batches.
  - Implement training through Lightning `Trainer` (or replace with CLI calls and parse outputs).
  - Rework prediction to use model forward (`model(bmg, V_d, X_d)`) and handle multi-task outputs.
- Update `scripts/models/train.py` and `scripts/models/predict.py` accordingly.
- Remove v1-specific caching calls (`set_cache_graph`, `set_cache_mol`) unless v2 provides equivalents.

5. Update RL integration
- Replace use of `chemprop.features.BatchMolGraph/MolGraph` with v2 data objects.
- Update `RLChempropMoleculeDataset` and `rl_chemprop_collate_fn` to emit v2 batch tuples.
- Adjust caching of molecular graphs for speed (likely via precomputed `MoleculeDatapoint` instances or cached `Mol` objects).
- Ensure reward shaping and predictions are consistent (classification logits vs probabilities).

6. Update docs and CLI examples
- Replace `chemprop_train`/`chemprop_predict` with `chemprop train`/`chemprop predict` and update flags.
- Document the v1 → v2 model conversion workflow (`chemprop convert --conversion v1_to_v2`).
- Add notes about changes in defaults and reproducibility differences.

7. Validation and regression checks
- Create a small smoke test that trains and predicts on a tiny dataset using v2.
- Verify that `ChempropScorer` and `RLModelChemprop` return stable shapes and reasonable values.
- Compare v1 vs v2 outputs on a fixed dataset to quantify drift.

### Plan B: Dual-stack compatibility (v1 + v2)

Goal: preserve reproducibility for existing papers while enabling v2 for new work.

Steps:
- Introduce a version switch in config/CLI (`--chemprop_version v1|v2`).
- Implement two adapter layers inside `synthemol/models/chemprop_models.py`:
  - v1 adapter using existing code.
  - v2 adapter using the new API.
- Update scripts to route to the correct adapter based on version.
- Keep `requirements_mcts.txt` pinned to v1 for MCTS reproducibility while `requirements_rl.txt` moves to v2.
- Provide conversion utilities and clear documentation on when to convert vs retrain.

Pros:
- Low risk to existing results.
- Enables incremental adoption.

Cons:
- Higher maintenance and test burden.
- Adds complexity to packaging and CI.

### Plan C: CLI boundary (decouple Chemprop from runtime)

Goal: remove Python-level Chemprop dependency from runtime generation paths.

Steps:
- Treat Chemprop as an external preprocessing step for training and scoring.
- Store precomputed building block scores and use them as inputs to SyntheMol.
- Deprecate RL-Chemprop path or make it optional behind a plugin.

Pros:
- Minimizes API coupling and breakage risk.

Cons:
- Reduces flexibility and eliminates end-to-end RL-Chemprop training inside the repo.

## Traps and pitfalls to watch for

- Feature pipeline mismatch: v1 `rdkit_2d_normalized`/`morgan` fingerprints are passed as `features_batch`; v2 uses descriptor inputs (`X_d`) and may treat scaling differently.
- Double-scaling risk: v1 used `load_scalers` and manually unscaled predictions; v2 often includes scaling transforms inside the model.
- Output semantics: v2 returns raw logits for classification unless you apply a sigmoid; ensure scoring logic matches prior expectations.
- Ensemble handling: v2 checkpoints may be `.ckpt` from Lightning; file discovery must handle both `.pt` and `.ckpt`.
- Caching: v1 cache toggles are removed; prediction on large datasets may require new memory strategies.
- Defaults drift: v2 defaults (scheduler, batch sizes, split logic, metrics) may shift results vs published benchmarks.
- Determinism: Lightning’s seed handling and multi-worker dataloaders can introduce non-determinism if not controlled.

## Environment and dependency considerations

- Chemprop v2 depends on Lightning; Torch version must be compatible with Lightning.
- Potential conflicts with RDKit, pandas, and numpy versions when updating Torch/Lightning.
- `chemfunc` + `descriptastorus` may also need updates to remain compatible.
- If supporting both v1 and v2, consider two isolated environments or extras (`synthemol[v1]`, `synthemol[v2]`).

## Downstream impacts

- Existing v1 models require conversion or retraining; predictions will not match bit-for-bit.
- Precomputed building block scores and generated molecule rankings may shift.
- Plotting scripts and analysis notebooks that assume v1 output column names may need updates.
- Reproducibility of paper results becomes harder; document this explicitly.

## Recommendation

Start with Plan B if reproducibility is a priority, then consolidate to Plan A once the v2 path is validated and stable. If runtime coupling becomes a maintenance bottleneck, consider Plan C for a clean separation.


## Migration execution notes (2026-02-03)

I implemented Plan B (dual-stack) with v2 as the default path and a best-effort v1 fallback (only if v1 is installed in a separate environment). Summary of changes:

- Added a Chemprop version switch (`chemprop_version`) across generation, scoring, and RL code paths.
- Rewrote `synthemol/models/chemprop_models.py` as a versioned adapter with v1 and v2 implementations.
- Updated `RLModelChemprop` and `ChempropScorer` to pass versioned calls for build/load/predict.
- Rewrote `scripts/models/chemprop_models.py` to support v2 training and prediction with Lightning, with a v1 fallback.
- Updated `scripts/models/train.py`, `scripts/models/predict.py`, and `scripts/models/chemprop_multi_to_single_task.py` to accept `chemprop_version`.
- Bumped Chemprop pins to `2.2.2` in `setup.py`, `requirements_rl.txt`, and `requirements_mcts.txt`.
- Added v2 CLI migration notes to `README.md`, `docs/rl/README.md`, and `docs/mcts/README.md`.

### Known limitations and follow-ups

- v2 training currently uses `val_loss` for checkpoint selection; this may differ from v1’s PRC-AUC/MAE selection.
- v2 predictors are assumed to handle output transforms internally; if probabilities/logits differ, update scoring accordingly.
- v2 model construction assumes a default 300 hidden dim and concatenates descriptor features; confirm against Chemprop defaults.
- Per-molecule v2 scoring constructs a new datapoint/dataloader each call; this is slower than v1 and should be optimized with caching if performance is an issue.
- `chemprop_multi_to_single_task.py` is v1-only and now explicitly requires v1.

Next step if we want to tighten correctness: add a small smoke test that trains/predicts with v2 and validates output ranges, then optimize v2 per-molecule inference caching.


## Smoke test + env status (2026-02-03)

- Smoke test script added: `scripts/tests/smoke_chemprop_v2.py`.
- README updated to reference the `synthemol-v2` conda env and the smoke test command.
- Conda environment creation started with `conda create -y -n synthemol-v2 python=3.11` but timed out during metadata collection. No env confirmed as created yet.

To resume: rerun the conda creation (possibly with a longer timeout or `conda config --set channel_priority strict`), then install deps (`pip install -r requirements_rl.txt`, `pip install -e .`) and run the smoke test.

