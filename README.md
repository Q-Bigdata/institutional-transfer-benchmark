# Institutional transfer benchmark

Code and saved data accompanying **Allocation institutions validated with people have little leverage in simulated societies of language-model strategies**.

Authors: Tianyu Shi, Hanqing Qian, Chengyi Tu. Repository: https://github.com/Q-Bigdata/institutional-transfer-benchmark . Original code is licensed under MIT; original data and figures are licensed under CC BY 4.0, with third-party rights retained.

## Contents

- `07_code/`: revised simulator, simulation pipeline, strategy/probe prompt builders, standalone plotting loader and validation entry point.
- `05_source_data/Source_Data.xlsx`: figure source tables.
- `06_llm_raw/`: original model responses, revision calls, probes and token ledger.
- `08_derived_data/inputs/`: human decision/first-round/outcome tables, planner training tables, elicited strategies and fitting penalty matrix.
- `08_derived_data/results_rev/`: saved simulation and bootstrap outputs.
- `08_derived_data/derived/`: published summary tables.
- `07_code/legacy/local_analysis.py`: original upstream-data preparation and historical elicitation helpers, retained for provenance; uses the original host API wrapper for elicitation and requires adaptation before new API calls. It is not invoked by the default commands.

## Install and run

Validated locally with Python 3.13.5; exact tested package versions are in `requirements.txt`. This environment is a current verification environment, not a reconstruction of the historical server environment described in the manuscript.

```sh
python -m pip install -r requirements.txt
python 07_code/validate_release.py
python 07_code/reproduce_figures.py
```

Both commands work from an arbitrary current directory when the script path is absolute. Validation recomputes point estimates and paired percentile intervals from saved results, checks source-table agreement, audits token usage, replays six selected published cells (6,000 games each), and checks the homogeneity proposition for five analytic rules. Plotting regenerates Figures 1–6 and Extended Data Figures 1–4 as PNG and PDF under `output/figures/`. It uses the original figure functions and released source tables; it does not rerun every analysis from raw observations. Original figure styling helpers were absent; the loader supplies neutral Matplotlib style and panel-label defaults. Numerical inputs are unchanged. Existing figure-label/spacing issues remain for manuscript figure polishing.

The default commands perform no model API requests, model training, full bootstrap, or complete exploitability search. Validation results are saved under `output/validation/`.

## Full simulation pipeline

`07_code/run_rev.py` supports `prep`, `main`, `boot`, `loo`, `valid`, `tune`, `mech`, and `exploit`. Input locations were corrected to `08_derived_data/inputs/`. New outputs go to `output/simulations/`, preserving released results. Set `INSTITUTION_RESULTS_DIR` to override that output directory. `prep` trains planner surrogates and builds human-bootstrap models; it can be expensive. Set `NPROC` to a suitable process count before running (the historical script defaults to 96), and retain the manuscript bootstrap counts for a formal reproduction. For example, in PowerShell:

```powershell
$env:NPROC = '4'
python 07_code/run_rev.py prep
python 07_code/run_rev.py main
```

No complete simulation-pipeline reproduction is claimed by the included validation. Previously saved serialized planner models were not included in the submitted package; `prep` must create them. Exact equivalence of retrained planners across scikit-learn versions is unverified. New LLM elicitation requires a separate API client/host adapter and may depend on model availability; no credentials are included.

## Human-data provenance and licensing

Human inputs are derivatives of Koster et al., *Deep reinforcement learning can promote sustainable human behaviour in a common-pool resource problem* (Nature Communications, 2025), from [google-deepmind/sustainable_behavior](https://github.com/google-deepmind/sustainable_behavior). The upstream README licenses software under Apache-2.0 and other materials under CC BY 4.0. Preserve attribution to DeepMind Technologies Limited and the original authors, link to the original source, and identify these tables as transformed inputs. An upstream Apache license copy is in `licenses/`; see `THIRD_PARTY_NOTICES.md` for scope. The original raw human CSV is not duplicated here.

This repository's original software is licensed under MIT; original model-response data, source tables and generated figures are licensed under CC BY 4.0. See `LICENSE`. Third-party rights and licenses remain applicable.

## Token-accounting correction completed during verification

The SI, manuscript summary and revision response were corrected locally on 2026-10-31 to distinguish the first 224 original records, the complete 266 original calls, and 213 revision calls, as well as cache-inclusive and non-cache token counts. All 266 original ledger entries reconcile with the raw usage records. Original/revision cache-inclusive totals are 1,908,408/1,153,408; non-cache input-plus-output totals are 1,037,314/533,674. The previously reported 894,973 corresponds to the first 224 original records only. SI Table S2 reports cache-inclusive counts, whereas Table S3 reports non-cache input-plus-output counts; both captions specify the scope. Original logs and ledger values are unchanged.

Initial-round human input covers 440 groups; the fitting decision table has 36,156 rows from 438 groups. No records were altered by this packaging work.

## Archived version

Version **v1.0.0** is archived at [10.5281/zenodo.23836527](https://doi.org/10.5281/zenodo.23836527). It fixes source commit `5e4698a0726537de71230eaaf75c8d56afb2039c`. Please cite this specific version DOI. The archive retains the reproduction boundaries and third-party license terms described above.
