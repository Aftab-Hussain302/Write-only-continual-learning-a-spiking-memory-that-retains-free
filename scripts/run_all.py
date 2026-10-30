#!/usr/bin/env python3
"""The whole final campaign on one machine, in parallel, resumable, into a new time-stamped folder.

    python3 scripts/run_all.py --data_root ~/data                      # everything (the paper's final experiments)
    python3 scripts/run_all.py --data_root ~/data --smoke              # a short end-to-end check of every dataset and stage
    python3 scripts/run_all.py --data_root ~/data --resume results/results_2026-11-02_0830   # continue an interrupted run

Stages, in order:
  eps_dev   the LDA shrinkage of REWIRE is selected ONCE on a development stream that shares no seed with the evaluation (CIFAR-100,
            seed 100; grid and rule in rewire/configure.py); the selected value is written to <run>/eps_selected.json and passed to
            every later unit. --eps 0.1 skips the selection and fixes the value instead.
  main      class-incremental learning with repetition: every arm x every dataset x every seed (the main grid)
  recur     recurrence density and long streams: R in {8, 16, 32, 64, 128} on the 8- and 16-class streams, {8, 16, 32, 64} on the
            80-class streams (the value of the main grid is reused), every seed, the sweep arms
  gate      the gate's competence threshold theta in {0.75, 0.95} (0.85 is the main grid), seeds 0-2, the gated arms
  shift     domain-shifted recurrences (the per-modality shift of configure.DATASETS), every seed, the shift arms
  native    task-incremental Split-MNIST and domain-incremental Permuted MNIST, every seed, the native arms (also the baselines'
            fidelity gate)
  eps_sens  REWIRE for every shrinkage of the grid on every dataset and seed (the sensitivity analysis; reported, never used to select)

Each unit of work (dataset, seed, stage) runs in its own process through `python -m rewire.worker` and writes its own folder under
<run>/parts/, so nothing is shared between processes and a finished unit is never repeated. The longest units start first. When the
queue is empty the parts are merged into the result tree (scripts/merge_results.py), ENVIRONMENT.md and MANIFEST.json record the
machine and the code, and <run>.zip is written for download. Then scripts/make_paper_figures.py builds the paper's figures and tables.

--synthetic replaces every dataset by a synthetic stand-in of the same shape (and the ResNet-18 front end by random weights) with
small sample counts: a test of the whole pipeline on a machine without the data. Its rows are marked and the figure script refuses them
unless told otherwise. Never use it for results."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from rewire.configure import config_for, PAPER_DATASETS, DEFAULT            # noqa: E402
from rewire.campaign import sweep_arms, GATED                               # noqa: E402

STAGES = ("eps_dev", "main", "recur", "gate", "shift", "native", "eps_sens")
WEIGHT = dict(tinyimagenet=10.0, cifar100=9.0, ssc=7.0, shd=5.0, nmnist=2.5, cifar10=1.5, mnist=1.5)   # rough relative cost of one seed of the main grid
SMOKE = ["n_recur=6", "full_steps=10", "n_train_per_class=150", "n_test_per_class=40", "recovery_budgets=[0,1,2]", "sweep_recur=[4,8]",
         "sweep_theta=[0.85,0.95]", "native_epochs=1", "native_tasks_pmnist=3", "native_n_per_class=150", "native_n_test=40", "eps_grid=[0.1,0.3]",
         'arms=["rewire","rewire_gated","rewire_gated_maha","abl_sketch_ridge","ctrl_slda_input","hlop","replay","naive"]']
SYNTHETIC = ["n_train_per_class=60", "n_test_per_class=12", "full_steps=6", "budget=10", "recovery_budgets=[0,1,2,5]", "stem_resize=64",
             "native_epochs=1", "native_n_per_class=60", "native_n_test=20", "native_tasks_pmnist=4", "eps_grid=[0.03,0.1,0.3]",
             "sweep_recur=[8,16,32,64]"]
SYNTHETIC_DS = dict(cifar100=["n_stream_classes=16", "n_holdout=4", "sweep_recur=[8,16,32]"], tinyimagenet=["n_stream_classes=16", "n_holdout=4", "sweep_recur=[8,16,32]"])


def parse_over(extra): return {kv.split("=", 1)[0]: json.loads(kv.split("=", 1)[1]) for kv in extra}


def plan(datasets, seeds, extra, extra_ds, stages):
    """The list of units: (priority, dataset, seed, stage, extra args), from the same configuration the workers will use."""
    units = []
    for ds in datasets:
        cfg = config_for(ds, dict(parse_over(extra + extra_ds.get(ds, [])), seeds=seeds)); w = WEIGHT.get(ds, 3.0)
        K, nr = cfg["n_stream_classes"], cfg["n_recur"]
        for s in seeds:
            if "main" in stages: units.append((w, ds, s, "main", []))
            if "recur" in stages:
                for n in cfg["sweep_recur"]:
                    if n != nr: units.append((w * 0.4 * (K + n) / (K + nr), ds, s, "recur", ["--n-recur", str(n)]))
            if "gate" in stages and s in seeds[:3]:
                for th in cfg["sweep_theta"]:
                    if abs(th - cfg["theta_hi"]) > 1e-9: units.append((w * 0.45, ds, s, "gate", ["--theta", str(th)]))
            if "shift" in stages: units.append((w * 0.4, ds, s, "shift", []))
            if "eps_sens" in stages: units.append((w * 0.1, ds, s, "eps_sens", []))
    if "native" in stages:
        for s in seeds: units.append((1.5, "mnist", s, "native", []))
    units.sort(key=lambda u: -u[0]); return units


def unit_dir(run, ds, seed, stage, extra):
    tag = {"main": "main", "native": "native", "shift": "shift", "eps_dev": "eps_dev", "eps_sens": "eps_sens",
           "recur": "recur/n" + (extra[1] if extra else ""), "gate": "gate/t" + (extra[1] if extra else "")}[stage]
    return run / "parts" / ds / tag / f"s{seed}"


def environment_md(run, args):
    import platform, torch
    lines = ["# Environment of this run", "", "```"]
    try: lines.append(subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip())
    except Exception: lines.append("gpu: none")
    lines += [f"python {platform.python_version()}  torch {torch.__version__}  cuda {torch.version.cuda}  cudnn {torch.backends.cudnn.version() if torch.cuda.is_available() else 'n/a'}",
              f"cpu: {os.cpu_count()} cores  ({platform.processor() or platform.machine()})", f"workers: {args.workers}  threads per worker: 1",
              f"deterministic algorithms: on (warn only), CUBLAS_WORKSPACE_CONFIG=:4096:8", f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}", "```", ""]
    try: lines += ["<details><summary>pip freeze</summary>", "", "```", subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).strip(), "```", "</details>"]
    except Exception: pass
    (run / "ENVIRONMENT.md").write_text("\n".join(lines) + "\n")


def manifest(run, args, units, eps):
    files = sorted(p for p in ROOT.rglob("*.py") if "results" not in p.parts and ".venv" not in p.parts)
    h = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    code_hash = hashlib.sha256("".join(h[k] for k in sorted(h)).encode()).hexdigest()
    old = json.loads((run / "MANIFEST.json").read_text()) if (run / "MANIFEST.json").exists() else {}
    if old and old.get("code_hash") != code_hash:
        print(f"!! the code changed since this run started (MANIFEST code_hash {old.get('code_hash', '')[:12]} -> {code_hash[:12]}); "
              "resuming would mix code versions. Start a fresh run unless the change is deliberate.", flush=True)
    (run / "MANIFEST.json").write_text(json.dumps(dict(command=" ".join(sys.argv), args=vars(args), code_hash=code_hash, files=h, units=len(units),
                                                      datasets=args.datasets, seeds=args.seeds, stages=args.stages, rw2_eps=eps,
                                                      synthetic=bool(args.synthetic), started=old.get("started", time.strftime("%Y-%m-%dT%H:%M:%S")),
                                                      resumed=time.strftime("%Y-%m-%dT%H:%M:%S") if old else None), indent=1))


def run_units(run, units, args, extra, extra_ds, env, label="units"):
    """Run units in parallel (longest first) with one retry of failures. Returns the list of failures."""
    todo = [u for u in units if not (unit_dir(run, *u[1:]) / "DONE").exists()]; done = len(units) - len(todo)
    print(f"[{label}] {len(units)} units, {done} already finished, {len(todo)} to run, {args.workers} workers", flush=True)
    active, t0, durations, failed = {}, time.time(), [], []
    def cmd_of(u):
        d = unit_dir(run, *u[1:]); x = extra + extra_ds.get(u[1], [])
        return [sys.executable, "-m", "rewire.worker", "--dataset", u[1], "--seed", str(u[2]), "--stage", u[3], "--out", str(d), "--data_root", args.data_root, *u[4]] + (["--set", *x] if x else [])
    def write_status():
        # Time left = the queued units' weights plus what remains of the running ones, at the seconds per unit of weight measured on the
        # finished units, spread over the workers (the merge and the archive, about 10-20 minutes, come after it).
        rem = len(todo) + len(active); avg = (sum(durations) / len(durations)) if durations else None; now = time.time()
        eta_s = ((sum(max(0.2, u[0]) for u in todo) * avg + sum(max(0.0, max(0.2, u[0]) * avg - (now - ts)) for (_, u, ts, _) in active.values())) / args.workers) if avg else None
        json.dump(dict(stage=label, finished=done, total=len(units), running=[f"{u[1]} s{u[2]} {u[3]} {' '.join(u[4])}" for (_, u, _, _) in active.values()], remaining=rem,
                       failed=failed, elapsed_min=round((now - T_START) / 60, 1), eta_min=(round(eta_s / 60, 1) if eta_s is not None else None),
                       eta_finish=(time.strftime("%a %H:%M", time.localtime(now + eta_s)) if eta_s is not None else None), updated=time.strftime("%H:%M:%S")),
                  open(run / "status.json", "w"), indent=1)
    while todo or active:
        for pid, (p, u, ts, log) in list(active.items()):
            code = p.poll()
            if code is None: continue
            log.close(); del active[pid]; dt = time.time() - ts
            if code == 0: done += 1; durations.append(dt / max(0.2, u[0])); print(f"finished {u[1]} s{u[2]} {u[3]} {' '.join(u[4])} in {dt/60:.1f} min ({done}/{len(units)})", flush=True)
            else: failed.append(u); print(f"FAILED {u[1]} s{u[2]} {u[3]} {' '.join(u[4])} (exit {code}); see logs/", flush=True)
            write_status()
        while todo and len(active) < args.workers:
            u = todo.pop(0); d = unit_dir(run, *u[1:]); d.mkdir(parents=True, exist_ok=True)
            log = open(run / "logs" / f"{u[1]}_s{u[2]}_{u[3]}{('_' + u[4][1]) if u[4] else ''}.log", "a"); log.write("\n" + " ".join(cmd_of(u)) + "\n")
            p = subprocess.Popen(cmd_of(u), stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT)); active[p.pid] = (p, u, time.time(), log)
            print(f"started  {u[1]} s{u[2]} {u[3]} {' '.join(u[4])}  ({len(active)} running, {len(todo)} queued)", flush=True); write_status()
        time.sleep(2)
    write_status()
    if failed:                                                                        # one retry of the failures, sequentially
        print(f"retrying {len(failed)} failed units", flush=True); retry, failed = list(failed), []
        for u in retry:
            with open(run / "logs" / f"{u[1]}_s{u[2]}_{u[3]}_retry.log", "a") as log: code = subprocess.call(cmd_of(u), stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
            if code != 0: failed.append(u)
        write_status()
    return failed


def select_eps(run, cfg_dev):
    """The pre-declared rule: the shrinkage with the highest final accuracy on the development stream; values within 0.1 points of
    the best are tied, and a tie goes to the value closest to 0.1 on a log scale (the earlier default)."""
    import math
    accs = {}
    for e in cfg_dev["eps_grid"]:
        p = unit_dir(run, cfg_dev["eps_dev_dataset"], cfg_dev["eps_dev_seed"], "eps_dev", []) / f"e{float(e):g}" / "rows.json"
        rows = json.loads(p.read_text()); accs[float(e)] = float(sum(r["final_acc"] for r in rows) / len(rows))
    best = max(accs.values()); tied = [e for e, a in accs.items() if a >= best - 0.001]
    sel = min(tied, key=lambda e: abs(math.log10(e) - math.log10(0.1)))
    out = dict(grid=sorted(accs), dev_final_acc={str(k): v for k, v in sorted(accs.items())}, selected=sel, dataset=cfg_dev["eps_dev_dataset"],
               seed=cfg_dev["eps_dev_seed"], rule="highest final accuracy on the development stream; within 0.1 points tie -> closest to 0.1 (log scale)")
    (run / "eps_selected.json").write_text(json.dumps(out, indent=1)); return sel


T_START = time.time()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True); ap.add_argument("--out", default="results", help="parent folder; the run goes into <out>/results_<date>_<time>")
    ap.add_argument("--resume", default=None, help="an existing run folder to continue"); ap.add_argument("--workers", type=int, default=0, help="0 = cores - 1 (max 8)")
    ap.add_argument("--datasets", nargs="+", default=PAPER_DATASETS); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    ap.add_argument("--eps", type=float, default=None, help="fix REWIRE's LDA shrinkage instead of selecting it on the development stream")
    ap.add_argument("--smoke", action="store_true", help="short streams, seed 0, eight arms: an end-to-end check of every dataset and stage")
    ap.add_argument("--synthetic", action="store_true", help="synthetic stand-ins for every dataset: a pipeline test only, never results")
    ap.add_argument("--no-determinism-check", action="store_true", help="skip the repeat of the first seed's MNIST main unit (run by default; the paper reports the largest difference)")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides passed to every unit (JSON values)")
    args = ap.parse_args()
    args.workers = args.workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    if args.smoke: args.seeds = [0]
    kind = "synthetic" if args.synthetic else ("smoke" if args.smoke else "results")
    run = Path(args.resume) if args.resume else Path(args.out) / f"{kind}_{time.strftime('%Y-%m-%d_%H%M')}"
    run.mkdir(parents=True, exist_ok=True); (run / "logs").mkdir(exist_ok=True); (run / "parts").mkdir(exist_ok=True)
    extra = list(args.set) + (SMOKE if args.smoke else []) + (SYNTHETIC if args.synthetic else [])
    extra_ds = dict(SYNTHETIC_DS) if args.synthetic else {}
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if args.synthetic: env.update(REWIRE_SYNTHETIC_DATA="1", REWIRE_TEST_RANDOM_STEM="1")
    else: env.pop("REWIRE_SYNTHETIC_DATA", None); env.pop("REWIRE_TEST_RANDOM_STEM", None)
    environment_md(run, args)
    print(f"run folder: {run}\nstages {args.stages}, datasets {args.datasets}, seeds {args.seeds}", flush=True)
    # ---- stage 0: the LDA shrinkage, selected on the development stream before any evaluation run
    cfg0 = config_for(DEFAULT["eps_dev_dataset"], parse_over(extra + extra_ds.get(DEFAULT["eps_dev_dataset"], [])))
    if args.eps is not None:
        eps = float(args.eps); (run / "eps_selected.json").write_text(json.dumps(dict(selected=eps, rule="fixed on the command line (--eps)"), indent=1))
    elif (run / "eps_selected.json").exists():
        eps = float(json.loads((run / "eps_selected.json").read_text())["selected"])
    elif "eps_dev" in args.stages:
        dev = [(1.0, cfg0["eps_dev_dataset"], int(cfg0["eps_dev_seed"]), "eps_dev", [])]
        manifest(run, args, dev, None)
        if run_units(run, dev, args, extra, extra_ds, env, "eps_dev"): raise SystemExit("the development stream failed; see logs/")
        eps = select_eps(run, cfg0); print(f"[eps_dev] selected LDA shrinkage {eps:g} on {cfg0['eps_dev_dataset']} seed {cfg0['eps_dev_seed']}", flush=True)
    else:
        eps = float(DEFAULT["rw2_eps"]); (run / "eps_selected.json").write_text(json.dumps(dict(selected=eps, rule="default (eps_dev stage not run)"), indent=1))
    extra_eps = extra + [f"rw2_eps={eps}"]
    stages = [s for s in args.stages if s != "eps_dev"]
    units = plan(args.datasets, args.seeds, extra_eps, extra_ds, stages)
    manifest(run, args, units, eps)
    failed = run_units(run, units, args, extra_eps, extra_ds, env, "campaign")
    if not args.no_determinism_check and "main" in stages and "mnist" in args.datasets:
        determinism_check(run, args, extra_eps, extra_ds, env)
    print(f"all units finished in {(time.time() - T_START) / 60:.1f} min; failed: {[' '.join(map(str, u[1:4])) for u in failed] or 'none'}", flush=True)
    if not args.no_merge:
        import importlib.util
        spec = importlib.util.spec_from_file_location("merge_results", ROOT / "scripts" / "merge_results.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        mod.merge(run, zip_it=True)


def determinism_check(run, args, extra, extra_ds, env):
    """Run the first seed's MNIST main unit a second time into parts/_determinism and compare the scalar results row by row."""
    ref = unit_dir(run, "mnist", args.seeds[0], "main", []) / "rows.json"
    if not ref.exists(): print("[determinism] reference unit missing; skipped", flush=True); return
    d = run / "parts" / "_determinism" / "mnist" / f"s{args.seeds[0]}"; d.mkdir(parents=True, exist_ok=True)
    if not (d / "DONE").exists():
        x = extra + extra_ds.get("mnist", [])
        cmd = [sys.executable, "-m", "rewire.worker", "--dataset", "mnist", "--seed", str(args.seeds[0]), "--stage", "main", "--out", str(d), "--data_root", args.data_root] + (["--set", *x] if x else [])
        with open(run / "logs" / "determinism.log", "a") as log: subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
    a = {r["arm"]: r for r in json.loads(ref.read_text())}; b = {r["arm"]: r for r in json.loads((d / "rows.json").read_text())}
    keys = ("final_acc", "bwt", "lifetime_energy_uJ", "recog_auroc_fam_x1", "retention_mean")
    diffs = {k: max(abs(float(a[n][k]) - float(b[n][k])) for n in a if n in b and a[n].get(k) is not None and b[n].get(k) is not None) for k in keys}
    out = dict(compared_arms=sorted(set(a) & set(b)), max_abs_difference=diffs, bitwise_identical=all(v == 0 for v in diffs.values()))
    (run / "determinism.json").write_text(json.dumps(out, indent=1)); print(f"[determinism] {out['max_abs_difference']}", flush=True)


if __name__ == "__main__": main()
