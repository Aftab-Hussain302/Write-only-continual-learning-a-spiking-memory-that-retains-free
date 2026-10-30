#!/usr/bin/env python3
"""Join the per-unit outputs of a run (<run>/parts/...) into the result tree and produce every diagnostic figure, table and report:

    python3 scripts/merge_results.py results/results_2026-11-02_0830 [--zip]       # (run_all.py calls this itself at the end)

Writes, inside the run folder:
  <dataset>/shift_none/rows.json + figures + leaderboard.csv        class-incremental with repetition: the main grid (every arm x seed)
  <dataset>/shift_domain/rows.json + figures                        domain-shifted recurrences (the shift arms)
  <dataset>/sweep_recur/n<R>/rows.json, sweep_recurrence_density.*  recurrence density and long streams
  <dataset>/sweep_gate/t<theta>/rows.json, gate_sweep.*             the gate's operating curve
  <dataset>/eps_sens/e<eps>/rows.json                               the LDA-shrinkage sensitivity analysis (REWIRE only)
  eps/dev/e<eps>/rows.json, eps_selected.json                       the development stream on which the shrinkage was selected
  native/native_rows.json, native_summary.csv                       task-incremental Split-MNIST and domain-incremental Permuted MNIST
  fidelity/fidelity_gate.csv, fidelity_gate_per_seed.csv            the gradient baselines on their native protocols (the same runs)
  campaign_leaderboard.csv/.tex, paired_*.csv, pooled_*.csv, summary_all_datasets.csv, summary_<dataset>.md
  <run>.zip                                                         everything above, ready to download
Nothing is recomputed from the learners; only the saved rows are read, so this can be re-run at any time. The paper's own figures and
tables come from scripts/make_paper_figures.py, which reads this tree."""
import glob, json, os, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from rewire.configure import config_for, DATASET_NAME                 # noqa: E402
from rewire.campaign import reports                                   # noqa: E402
from rewire.utils.plotting import plot_all, plot_sweep, plot_gate_sweep, chunk_costs, _label   # noqa: E402


def _rows(paths):
    rows, seen = [], set()
    for p in sorted(paths):
        for r in json.load(open(p)):
            k = (r["dataset"], r.get("shift_mode"), r.get("n_recur"), r.get("theta_hi"), (r.get("config") or {}).get("rw2_eps"), r["seed"], r["arm"])
            if k not in seen: seen.add(k); rows.append(r)
    return rows


def _write(rows, out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "rows.json"), "w") as f: json.dump(rows, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def summary_table(rows):
    """One row per arm: the numbers the per-dataset summaries are built from (mean over seeds)."""
    cc = chunk_costs(rows).groupby("arm").mean(numeric_only=True)
    t = pd.DataFrame([dict(arm=r["arm"], seed=r["seed"], ACC=100 * r["final_acc"], BWT=100 * r["bwt"], forgetting=100 * r["forgetting"],
                           AUROC_stream=r.get("recog_auroc_fam_x1", r["recog_auroc_fam"]), AUROC_test=r["fam_auroc_test"], S_AUR=r["S_AUR"], RIF=r["RIF"],
                           mJ=r["lifetime_energy_uJ"] / 1000, mJ_front_end=r.get("frontend_energy_uJ", 0) / 1000,
                           stored_KB=r["standing_bytes"] / 1024, resident_KB=r.get("resident_bytes_total", r.get("resident_bytes", np.nan)) / 1024,
                           d=r.get("feature_dim"), time_s=r.get("train_time_s", np.nan)) for r in rows])
    g = t.groupby("arm"); m = g.mean(numeric_only=True).drop(columns="seed"); s = g.std(numeric_only=True).drop(columns="seed")
    m["ACC_sd"], m["BWT_sd"], m["n_seeds"] = s["ACC"], s["BWT"], g.size()
    m["mJ_admission"] = cc["intro_uJ"] / 1000; m["mJ_readmission"] = cc["recur_uJ"] / 1000; m["skipped"] = cc["skipped"]
    m.insert(0, "label", [_label(a) for a in m.index]); return m.sort_values("ACC", ascending=False)


def merge(run, zip_it=False, figures=True):
    run = Path(run); parts = run / "parts"; t0 = time.time(); main_rows, shift_rows = [], []
    datasets = sorted(p.name for p in parts.iterdir() if p.is_dir() and not p.name.startswith("_")) if parts.exists() else []
    frames = []
    for ds in datasets:
        cfg = config_for(ds)
        mains = _rows(glob.glob(str(parts / ds / "main" / "s*" / "rows.json")))
        if mains:
            out = run / ds / "shift_none"; _write(mains, out); main_rows += mains
            if figures: plot_all(mains, dict(cfg, out=str(out), shift_mode="none"))
            tab = summary_table(mains); tab.to_csv(out / "summary_table.csv"); frames.append(tab.assign(dataset=ds))
            with open(run / f"summary_{ds}.md", "w") as f: f.write(f"# {DATASET_NAME.get(ds, ds)} ({len(mains)} runs)\n\n" + tab.round(3).to_markdown() + "\n")
        shifts = _rows(glob.glob(str(parts / ds / "shift" / "s*" / "rows.json")))
        if shifts:
            out = run / ds / "shift_domain"; _write(shifts, out); shift_rows += shifts
            if figures: plot_all(shifts, dict(cfg, out=str(out), shift_mode="domain"))
        sweep = []
        for d in sorted(glob.glob(str(parts / ds / "recur" / "n*"))):
            n = int(Path(d).name[1:]); rs = _rows(glob.glob(f"{d}/s*/rows.json"))
            if rs: _write(rs, run / ds / "sweep_recur" / f"n{n}"); sweep += rs
        if sweep and figures:
            from rewire.campaign import sweep_arms
            plot_sweep(sweep + [r for r in mains if r["arm"] in sweep_arms(cfg)], dict(cfg, out=str(run / ds / "sweep_recur")))
        gate = []
        for d in sorted(glob.glob(str(parts / ds / "gate" / "t*"))):
            rs = _rows(glob.glob(f"{d}/s*/rows.json"))
            if rs: _write(rs, run / ds / "sweep_gate" / Path(d).name); gate += rs
        if gate and figures:
            from rewire.campaign import GATED
            plot_gate_sweep(gate + [r for r in mains if r["arm"] in GATED and r["seed"] in sorted({x["seed"] for x in gate})], mains, str(run / ds / "sweep_gate"), ds)
        sens = {}
        for d in sorted(glob.glob(str(parts / ds / "eps_sens" / "s*" / "e*"))):
            sens.setdefault(Path(d).name, []).extend(_rows([f"{d}/rows.json"]) if os.path.exists(f"{d}/rows.json") else [])
        for e, rs in sens.items(): _write(rs, run / ds / "eps_sens" / e)
        devs = {}
        for d in sorted(glob.glob(str(parts / ds / "eps_dev" / "s*" / "e*"))):
            if os.path.exists(f"{d}/rows.json"): devs.setdefault(Path(d).name, []).extend(_rows([f"{d}/rows.json"]))
        for e, rs in devs.items(): _write(rs, run / "eps" / "dev" / e)
        nat = sorted(glob.glob(str(parts / ds / "native" / "s*" / "native_rows.json"))) + sorted(glob.glob(str(parts / ds / "fidelity" / "s*" / "native_rows.json")))
        if nat:
            rows = [r for p in nat for r in json.load(open(p))]; os.makedirs(run / "native", exist_ok=True)
            with open(run / "native" / "native_rows.json", "w") as f: json.dump(rows, f)
            from rewire.native import verdict
            df = pd.DataFrame([{k: v for k, v in r.items() if k != "A"} for r in rows])
            g = df.groupby(["protocol", "arm"]).agg(acc=("acc", "mean"), acc_sd=("acc", "std"), acc_se=("acc", lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0),
                                                    bwt=("bwt", "mean"), bwt_sd=("bwt", "std"), bwt_se=("bwt", lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0),
                                                    energy_mJ=("lifetime_energy_uJ", lambda v: float(np.mean(v)) / 1000), n=("seed", "count")).reset_index()
            g["verdict"] = [verdict(p, a, ac, bw) for p, a, ac, bw in zip(g.protocol, g.arm, g.acc, g.bwt)]
            g.to_csv(run / "native" / "native_summary.csv", index=False)
            os.makedirs(run / "fidelity", exist_ok=True); g.to_csv(run / "fidelity" / "fidelity_gate.csv", index=False)
            df.to_csv(run / "fidelity" / "fidelity_gate_per_seed.csv", index=False)
    if (run / "eps_selected.json").exists():
        os.makedirs(run / "eps", exist_ok=True); shutil.copy(run / "eps_selected.json", run / "eps" / "eps_selected.json")
    if frames:
        allt = pd.concat(frames); allt.to_csv(run / "summary_all_datasets.csv")
        piv = allt.reset_index().pivot_table(index="arm", columns="dataset", values="ACC"); piv.round(1).to_csv(run / "acc_by_dataset.csv")
        print("\nACC by dataset (mean over seeds)\n" + piv.round(1).to_string())
    if main_rows: reports(str(run), main_rows)
    if shift_rows:
        from rewire.metrics import paired_table, pooled_table
        os.makedirs(run / "shift", exist_ok=True)
        for m in ("final_acc", "final_acc_shifted", "lifetime_energy_uJ"): paired_table(shift_rows, m, "rewire", str(run / "shift")); pooled_table(shift_rows, m, "rewire", str(run / "shift"))
    (run / "MERGED").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    print(f"merged {len(main_rows)} main-grid runs and {len(shift_rows)} shifted runs over {len(datasets)} datasets in {time.time() - t0:.0f}s -> {run}")
    if zip_it:
        z = shutil.make_archive(str(run), "zip", root_dir=run.parent, base_dir=run.name); print(f"archive: {z} ({os.path.getsize(z) / 1e6:.0f} MB)")
    return run


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="the run folder (it holds parts/)"); ap.add_argument("--zip", action="store_true", help="also write <run>.zip")
    ap.add_argument("--no-figures", action="store_true", help="skip the diagnostic figures (faster)")
    a = ap.parse_args()
    if not (Path(a.run) / "parts").exists(): raise SystemExit(f"{a.run} has no parts/ folder: not a run folder of scripts/run_all.py")
    merge(a.run, zip_it=a.zip, figures=not a.no_figures)
