# Reproducing the paper

Everything below is the pipeline that produces the reported numbers, figures and tables; nothing else is used.

The records of the article's run are in `records/results_2026-10-29_1457/` (the GPU campaign, with the node measurement under
`hardware/`). To rebuild every figure, table and value from them, without data or a GPU, run step 6 with
`--runs records/results_2026-10-29_1457`; steps 2 to 5 and 7 re-run the experiments themselves.

## 1. Environment (once)
```
bash setup_env.sh && . .venv/bin/activate
```

## 2. Data
Put the datasets under one folder as described in `DATASETS.md` (e.g. `~/data`), then
```
python3 scripts/prepare_data.py --data_root ~/data
```
This loads every dataset once through the real loaders, reports what it found, and builds the frozen ResNet-18 feature caches for
CIFAR-10, CIFAR-100 and Tiny-ImageNet (a few minutes on a GPU). It ends with `ALL DATASETS READY`.

## 3. Checks (about two minutes, CPU is enough)
```
REWIRE_PREFLIGHT_CPU=1 python3 -m rewire.preflight   # the properties the method relies on, the protocol, the energy model, the statistics
                                                     # (on the GPU machine, drop the variable: the preflight then also requires CUDA)
python3 selftest.py                  # every learner end to end on a short synthetic stream, the native protocols, every figure
python3 -m pytest tests -q           # Propositions 1 and 2 as tests, the discriminant against NumPy, recursive vs direct solve
```

## 4. A short end-to-end run on the real data
```
python3 scripts/run_all.py --data_root ~/data --smoke
```
Short streams, one seed, eight learners, every dataset and every stage, the determinism check, the merge and the archive. The result
lands in `results/smoke_<date>_<time>/`. On a machine without the data, `--synthetic` runs the same pipeline on synthetic stand-ins
of the same shapes (a test of the tools only: its rows are marked and the paper scripts refuse them).

## 5. The campaign
```
nohup python3 scripts/run_all.py --data_root ~/data > run_all.log 2>&1 &
tail -f run_all.log            # or: cat results/results_<date>_<time>/status.json
```
The stages run in this order, each as independent units of work (dataset, seed, stage) in parallel processes (`--workers`, default
cores - 1):

| stage | what | units |
|---|---|---|
| `eps_dev` | the discriminant's shrinkage, selected once on a development stream that no evaluation uses (CIFAR-100, seed 100) by the rule in `scripts/run_all.py`; written to `eps_selected.json` and passed to every later unit | 1 |
| `main` | class-incremental learning with repetition: every learner x dataset x seed (0-4) | 30 |
| `recur` | long streams: R in {8, 16, 32, 64, 128} recurrence chunks on the 8- and 16-class streams, {8, 16, 32, 64} on the 80-class streams (the main grid supplies R = 24 and R = 16) | 130 |
| `gate` | the gate's competence threshold 0.75 and 0.95 (0.85 is the main grid), seeds 0-2 | 36 |
| `shift` | every recurrence of a class from a shifted domain (`DATASETS[...]["shift"]` in `rewire/configure.py`) | 30 |
| `native` | task-incremental Split-MNIST and domain-incremental Permuted MNIST (also the baselines' fidelity check) | 5 |
| `eps_sens` | REWIRE at every shrinkage of the grid on every dataset and seed (reported, never used to select) | 30 |

A finished unit is never repeated, and an interrupted run continues with `--resume results/results_<date>_<time>`. At the end the
first seed's MNIST main unit is run a second time and the largest difference is written to `determinism.json`
(`--no-determinism-check` skips it). Plan for about 12 hours on one RTX A6000 with 6 CPU cores (the article's run took 11.9 h; the
gradient baselines are CPU-bound, so more cores shorten it) and about 3 GB of disk for the feature caches (including the rotated images' features) and the run folder.
SSC is optional: `--datasets mnist nmnist shd ssc cifar10 cifar100 tinyimagenet` adds it everywhere, at about 1.3 times the time.

When the queue is empty the run folder holds
```
<dataset>/shift_none/rows.json, figures, summary_table.csv      the main grid
<dataset>/sweep_recur/n<R>/, <dataset>/sweep_gate/t<theta>/      long streams, the gate's thresholds
<dataset>/shift_domain/rows.json                                 domain-shifted recurrences
<dataset>/eps_sens/e<eps>/, eps/dev/, eps_selected.json          the shrinkage: sensitivity and selection
native/native_rows.json, native_summary.csv, fidelity/           the task- and domain-incremental protocols
campaign_leaderboard.csv, paired_*.csv, pooled_*.csv, summary_*.md, determinism.json
ENVIRONMENT.md, MANIFEST.json, logs/, parts/                     the machine, the code hashes, the per-unit outputs
```
and `results/results_<date>_<time>.zip` is the archive to download. `python3 scripts/merge_results.py <run folder> --zip` rebuilds the
merged tree from `parts/` without re-running anything.

## 6. The paper's figures, tables and values
```
python3 scripts/make_paper_figures.py --runs results/results_<date>_<time> --name final
python3 scripts/make_paper_figures.py --runs records/results_2026-10-29_1457 --name check --out paper_results_check   # the article's records
```
Writes `paper_results_final_<time>/`: the figures (EPS and PNG at 300 dpi, plus PDF copies), one `.tex` per table, `macros/numbers.tex`
(every number the text quotes, as `\defval{key}{value}`), `facts.json`, one source-data workbook per figure, a manifest and an
integrity scan of the figures' text. A new folder is written on every call.

## 7. Hardware (optional, with a BrainChip Akida board)
```
python3 hardware/akida/run_akida.py --data_root ~/data --run results/results_<date>_<time> --datasets shd nmnist --backend hw
```
Writes `<run>/hardware/akida_<dataset>.json`, which step 6 then includes (see `hardware/akida/README.md`).

## 8. One dataset in one process (for inspection)
```
python3 run_campaign.py --datasets shd --seeds 0 --arms rewire rewire_gated hlop replay --no-sweeps --data_root ~/data --out results/dev
```
