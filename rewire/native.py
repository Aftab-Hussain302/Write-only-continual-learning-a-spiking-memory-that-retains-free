"""Task-incremental and domain-incremental learning on the native protocols of the gradient baselines, for every kind of learner.

  split  : task-incremental Split-MNIST (DSD-SNN's protocol): five tasks of two classes; the task is given at test time, so the
           prediction is restricted to the task's two classes (a masked head; for the write-only learners the argmax of the
           discriminant is restricted to the task's classes).
  pmnist : domain-incremental Permuted MNIST (HLOP's protocol): the unpermuted task followed by native_tasks_pmnist - 1 fixed pixel
           permutations, the same ten labels in every task, one ten-way head.

Gradient learners train on each task for native_epochs epochs of class-balanced minibatches (the loop of the class-incremental
campaign); write-only learners write each task's samples once (one write and one solve per task). After each task every task seen so
far is evaluated, which gives the accuracy matrix A[i, j] (accuracy on task j after training on task i), ACC = mean_j A[T, j] and
BWT = mean_{j<T} (A[T, j] - A[j, j]) (Lopez-Paz & Ranzato 2017). The lifetime energy is the cost meter of the class-incremental
campaign. The same runs give the fidelity gate of the gradient baselines: a baseline passes when its protection removes the
forgetting that fine-tuning shows (split: BWT >= -5 points and ACC >= 90; pmnist: BWT >= -10 and ACC >= 80)."""
import json, os, time, numpy as np, pandas as pd, torch
from .configure import DEFAULT
from .dataloader import load_dataset, synthetic_enabled
from .modules import DEV
from .models import build

PASS = {"split": (-0.05, 0.90), "pmnist": (-0.10, 0.80)}       # (BWT >=, ACC >=) on the 0-1 scale


def _tasks(protocol, cfg, rng, n_pix):
    if protocol == "split": return [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], [None] * 5
    n = int(cfg.get("native_tasks_pmnist", 10))
    return [list(range(10))] * n, [None] + [torch.tensor(rng.permutation(n_pix), device=DEV) for _ in range(n - 1)]


def run_one_native(name, cfg, seed, protocol, Xt, yt, tr, te, epochs, log=print):
    """One learner on one native protocol and seed. Returns a row (dict) with the accuracy matrix and the costs."""
    torch.manual_seed(seed); rng = np.random.RandomState(seed + 17)
    tasks, perms = _tasks(protocol, cfg, rng, Xt[0].numel())
    pm = lambda x, p: x if p is None else x.flatten(1)[:, p].reshape(x.shape)
    arm = build(name, cfg, seed, Xt[tr[0][:64]]); write_only = getattr(arm, "gradient", True) is False
    T = len(tasks); A = np.full((T, T), np.nan); t0 = time.time()
    base_loss = arm.loss if not write_only else None
    for i, tk in enumerate(tasks):
        idx = np.concatenate([tr[c] for c in tk]); Xi = pm(Xt[idx], perms[i]); yi = yt[idx]
        if write_only:
            arm.write(Xi, yi)                                               # one write (forward, sums) and one solve per task
        else:
            mask = torch.full((arm.K,), -1e9, device=DEV); mask[tk] = 0
            arm.loss = (lambda out, yb, mask=mask: base_loss(out + mask, yb))
            steps = epochs * len(yi) // cfg["batch"]; arm.before_visit(tk, True); arm.train_steps(Xi, yi, steps); arm.after_visit(Xi, yi, tk)
            arm.seen |= set(tk)
        for j, tj in enumerate(tasks[: i + 1]):
            mj = torch.full((arm.K,), -1e9, device=DEV); mj[tj] = 0
            jdx = np.concatenate([te[c] for c in tj]); Xe = pm(Xt[jdx], perms[j]); ye = yt[jdx]
            with torch.no_grad(): A[i, j] = ((arm.predict(Xe, count=False) + mj).argmax(1) == ye).float().mean().item()
    acc = float(np.mean(A[-1])); bwt = float(np.mean([A[-1, j] - A[j, j] for j in range(T - 1)]))
    row = dict(protocol=protocol, arm=name, seed=seed, n_tasks=T, acc=acc, bwt=bwt, A=A.tolist(), write_only=write_only,
               lifetime_energy_uJ=float(arm.meter.energy_uJ()), standing_bytes=int(arm.standing_bytes()), time_s=time.time() - t0,
               epochs=(None if write_only else epochs), synthetic_data=synthetic_enabled())
    log(f"  native {protocol:6s} {name:16s} seed {seed}: acc {acc:.3f} bwt {bwt:+.3f} energy {row['lifetime_energy_uJ'] / 1000:.1f} mJ "
        f"diag {np.round(np.diag(A), 2).tolist()}")
    return row


def run_native(cfg=None, arms=None, seeds=(0, 1, 2, 3, 4), epochs=None, n_per_class=None, protocols=None, out=None, log=print):
    """Every arm in `arms` on every native protocol and seed; writes <out>/native_rows.json and <out>/fidelity_gate.csv."""
    cfg = dict(DEFAULT if cfg is None else cfg); cfg["dataset"] = "mnist"; cfg["n_classes"] = 10
    arms = list(arms or cfg.get("native_arms")); protocols = list(protocols or cfg.get("native_protocols", ["split", "pmnist"]))
    epochs = int(epochs or cfg.get("native_epochs", 3)); n_per_class = int(n_per_class or cfg.get("native_n_per_class", 300))
    X, y = load_dataset(cfg.get("native_dataset", "mnist"), cfg.get("data", "torchvision"), cfg.get("data_root", "data"),
                        {**cfg, "n_train_per_class": n_per_class, "n_test_per_class": int(cfg.get("native_n_test", 100))})
    Xt = torch.tensor(X, device=DEV); yt = torch.tensor(y, device=DEV)
    n_avail = min(int((y == c).sum()) for c in range(10)); n_te = min(int(cfg.get("native_n_test", 100)), n_avail // 3); n_tr = min(n_per_class, n_avail - n_te)
    rows = []
    for protocol in protocols:
        for name in arms:
            for seed in seeds:
                rng = np.random.RandomState(seed); tr, te = {}, {}
                for c in range(10):
                    idx = np.where(y == c)[0]; rng.shuffle(idx); tr[c] = idx[:n_tr]; te[c] = idx[n_tr:n_tr + n_te]
                rows.append(run_one_native(name, cfg, seed, protocol, Xt, yt, tr, te, epochs, log))
    out = out or cfg.get("out", "results"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "native_rows.json"), "w") as f: json.dump(rows, f)
    df = pd.DataFrame(rows)
    g = df.groupby(["protocol", "arm"]).agg(acc=("acc", "mean"), acc_se=("acc", lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0),
                                            bwt=("bwt", "mean"), bwt_se=("bwt", lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)).reset_index()
    g.to_csv(os.path.join(out, "fidelity_gate.csv"), index=False)
    return rows


def verdict(protocol, arm, acc, bwt):
    """The fidelity gate of a gradient baseline on its native protocol (fine-tuning is the reference; write-only arms are reported)."""
    if arm == "naive": return "reference"
    if arm in ("rewire", "ctrl_slda_input") or arm.startswith("rewire") or arm.startswith("abl_"): return "reported"
    b, a = PASS[protocol]
    return "pass" if (bwt >= b and acc >= a) else "fail"
