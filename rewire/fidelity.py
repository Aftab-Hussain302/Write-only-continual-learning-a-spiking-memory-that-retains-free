"""The fidelity gate of the gradient baselines on their native protocols. Kept as an entry point for compatibility; the protocols
themselves (task-incremental Split-MNIST and domain-incremental Permuted MNIST, for gradient and write-only learners alike) live in
rewire/native.py, which also runs REWIRE on them."""
from .native import run_native, verdict, PASS  # noqa: F401


def run_fidelity(cfg=None, arms=("naive", "hlop", "dsd_snn", "ewc"), seeds=(0, 1, 2), epochs=3, n_per_class=300, log=print,
                 protocols=("split", "pmnist"), out=None):
    rows = run_native(cfg, arms=arms, seeds=seeds, epochs=epochs, n_per_class=n_per_class, protocols=protocols, out=out, log=log)
    report = {}
    for p in protocols:
        for a in arms:
            rs = [r for r in rows if r["protocol"] == p and r["arm"] == a]
            if rs:
                import numpy as np
                report[(p, a)] = dict(acc=float(np.mean([r["acc"] for r in rs])), bwt=float(np.mean([r["bwt"] for r in rs])))
    return report


if __name__ == "__main__":
    import argparse
    from .configure import DEFAULT
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2]); ap.add_argument("--data_root", default="data")
    ap.add_argument("--out", default="results"); a = ap.parse_args()
    c = dict(DEFAULT); c["data_root"] = a.data_root; c["out"] = a.out
    run_fidelity(c, seeds=a.seeds, out=a.out)
