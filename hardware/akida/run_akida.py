#!/usr/bin/env python3
"""REWIRE's frozen expansion measured on a BrainChip Akida processor (inference only); see hardware/akida/README.md.

    python3 hardware/akida/run_akida.py --data_root ~/data --run results/results_2026-11-02_0830 --datasets shd nmnist --backend hw
    python3 hardware/akida/run_akida.py --data_root ~/data --run results/results_2026-11-02_0830 --datasets shd --backend sw      # no board
    REWIRE_SYNTHETIC_DATA=1 python3 hardware/akida/run_akida.py --data_root /tmp/x --run /tmp/run --datasets shd --backend numpy  # test

Writes <run>/hardware/akida_<dataset>.json (read by scripts/make_paper_figures.py: Extended Data Table 2 and the named values
akida.*): per seed and as means, the final accuracy of REWIRE with the LIF layer, with the converted network (NumPy emulation, MetaTF
software model) and with the features computed on the chip; the chip's idle power floor, energy per frame (total and above the
floor), latency, the operation-count model of the same network and the ratio of measured to modelled energy."""
import argparse, json, math, os, platform, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import rewire_akida as RA                                                    # noqa: E402


def versions():
    v = dict(python=platform.python_version(), torch=torch.__version__)
    for m in ("akida", "cnn2snn", "quantizeml", "tf_keras"):
        try: v[m] = __import__(m).__version__
        except Exception: v[m] = None
    return v


def one_seed(ds, seed, args, eps, device):
    over = {"rw2_eps": eps} if eps is not None else {}
    cfg, X, y, st = RA.prepare(ds, args.data_root, seed, over)
    arm = RA.reference(cfg, seed, X, st); E = RA.expansion(arm)
    idx_w, y_w = RA.written_multiset(st, cfg); idx_t, S = RA.test_set(st)
    need = np.unique(np.concatenate([idx_w, idx_t, st["chunks"][0]["target"]])); pos = {int(i): k for k, i in enumerate(need)}
    remap = lambda a: np.array([pos[int(i)] for i in a]); calib = remap(st["chunks"][0]["target"])
    Xn = np.asarray(X)[need]; yn = np.asarray(y)[need]
    from rewire.modules import DEV
    with torch.no_grad():
        Hl = np.concatenate([arm.feats(torch.tensor(Xn[i:i + 512], device=DEV)).cpu().numpy() for i in range(0, len(Xn), 512)])
    row = dict(dataset=ds, seed=seed, n_written=int(len(idx_w)), n_test=int(len(idx_t)), eps=eps)
    row["acc_lif"] = 100 * RA.lda_accuracy(Hl, yn, remap(idx_w), y_w, remap(idx_t), S, cfg)
    # Proposition 1 in practice: the joint fit on the written multiset must equal the campaign's streaming REWIRE run of this seed
    camp = Path(args.run) / ds / "shift_none" / "rows.json"
    if camp.exists():
        ref = [r for r in json.loads(camp.read_text()) if r["arm"] == "rewire" and r["seed"] == seed]
        if ref:
            row["acc_lif_campaign"] = 100 * float(ref[0]["final_acc"])
            if abs(row["acc_lif_campaign"] - row["acc_lif"]) > 1e-3:
                print(f"[akida] !! {ds} seed {seed}: joint fit {row['acc_lif']:.3f} differs from the campaign's REWIRE {row['acc_lif_campaign']:.3f}; "
                      "check that the run and this script use the same configuration", flush=True)
    Xf = RA.flat_input(Xn); v = RA.input_view(arm, Xf); Xu8, scale = RA.to_input(Xf)
    r_np, nnz = RA.numpy_chip(E, Xu8, scale)
    row["acc_numpy"] = 100 * RA.lda_accuracy(RA.dual(r_np, v, calib), yn, remap(idx_w), y_w, remap(idx_t), S, cfg)
    H, d_s = E["W"].shape[0], E["S"].shape[1]
    row["modelled_uJ"] = RA.modelled_energy_uJ(nnz, H, d_s); row["nnz_input"] = nnz["input"]; row["nnz_hidden"] = nnz["hidden"]
    if args.backend in ("sw", "hw"):
        model = RA.akida_model(E, Xu8[calib], scale)
        r_sw, m_sw = RA.akida_features(model, Xu8, None)
        row["acc_sim"] = 100 * RA.lda_accuracy(RA.dual(r_sw, v, calib), yn, remap(idx_w), y_w, remap(idx_t), S, cfg); row.update({f"sw_{k}": x for k, x in m_sw.items()})
        if args.backend == "hw":
            r_hw, m = RA.akida_features(model, Xu8, device, repeats=args.repeats)
            row["acc_chip"] = 100 * RA.lda_accuracy(RA.dual(r_hw, v, calib), yn, remap(idx_w), y_w, remap(idx_t), S, cfg)
            row.update({k: x for k, x in m.items() if k != "statistics"}); row["statistics"] = m.get("statistics")
            if math.isfinite(row.get("net_energy_per_frame_uJ", float("nan"))) and row["modelled_uJ"] > 0:
                row["measured_over_modelled"] = row["net_energy_per_frame_uJ"] / row["modelled_uJ"]
    print(f"[akida] {ds} seed {seed}: " + ", ".join(f"{k} {v:.2f}" for k, v in row.items() if isinstance(v, float) and k.startswith(("acc", "net", "energy", "floor", "latency", "measured"))), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True); ap.add_argument("--run", required=True, help="the campaign's run folder (its eps_selected.json is used)")
    ap.add_argument("--datasets", nargs="+", default=["shd", "nmnist"]); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--backend", choices=("hw", "sw", "numpy"), default="hw")
    ap.add_argument("--repeats", type=int, default=3, help="chip runs per seed for the power measurement (each over every frame)")
    a = ap.parse_args()
    run = Path(a.run); (run / "hardware").mkdir(parents=True, exist_ok=True)
    eps = None
    for p in (run / "eps_selected.json", run / "eps" / "eps_selected.json"):
        if p.exists(): eps = float(json.loads(p.read_text())["selected"]); break
    if eps is None: print("[akida] no eps_selected.json in the run folder: the configured default shrinkage is used", flush=True)
    device = None
    if a.backend == "hw":
        import akida
        devs = akida.devices()
        if not devs: raise SystemExit("no Akida device found (akida.devices() is empty); use --backend sw for the software model")
        device = devs[0]; print(f"[akida] device: {device.desc if hasattr(device, 'desc') else device}", flush=True)
    for ds in a.datasets:
        rows = [one_seed(ds, s, a, eps, device) for s in a.seeds]
        keys = ("acc_chip", "acc_sim", "acc_numpy", "acc_lif", "floor_mW", "energy_per_frame_uJ", "net_energy_per_frame_uJ", "latency_ms", "modelled_uJ", "measured_over_modelled")
        summary = {k: float(np.mean([r[k] for r in rows if isinstance(r.get(k), float) and math.isfinite(r[k])])) for k in keys
                   if any(isinstance(r.get(k), float) and math.isfinite(r[k]) for r in rows)}
        out = dict(dataset=ds, backend=a.backend, seeds=a.seeds, eps=eps, versions=versions(), device=(str(getattr(device, "desc", device)) if device else None),
                   date=time.strftime("%Y-%m-%dT%H:%M:%S"), synthetic=os.environ.get("REWIRE_SYNTHETIC_DATA") == "1", summary=summary, per_seed=rows)
        p = run / "hardware" / f"akida_{ds}.json"; p.write_text(json.dumps(out, indent=1, default=str)); print(f"[akida] {ds}: {summary} -> {p}", flush=True)


if __name__ == "__main__": main()
