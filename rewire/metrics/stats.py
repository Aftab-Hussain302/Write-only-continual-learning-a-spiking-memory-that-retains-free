"""Paired statistics: every arm sees the identical stream for a given seed (same class roles, same chunk contents, same order), so
seeds are PAIRED and a paired test is the correct one. The Wilcoxon signed-rank test is exact (its null distribution is computed by
dynamic programming over the signed ranks, ties given average ranks, zero differences dropped) for up to 60 non-zero pairs, which
covers every test of the campaign; beyond that the normal approximation with tie and continuity corrections is used. The
bootstrap CI is reported on the paired differences, not on the two means separately."""
import json, math, os, numpy as np, pandas as pd

MAX_EXACT = 60


def _ranks(a):
    order = np.argsort(a, kind="mergesort"); ranks = np.empty(len(a)); ranks[order] = np.arange(1, len(a) + 1)
    for v in np.unique(a):                                   # average ranks for ties
        m = a == v
        if m.sum() > 1: ranks[m] = ranks[m].mean()
    return ranks


def wilcoxon_exact(d, max_exact=MAX_EXACT):
    """Two-sided Wilcoxon signed-rank test. Returns (p, W) with W the sum of the ranks of the positive differences.
    Exact for up to max_exact non-zero pairs: the doubled ranks are integers, so the null distribution of 2W (every rank positive
    or negative with probability 1/2) is a convolution computed in O(n * sum of ranks). The two-sided P value is the probability of
    a statistic at least as far from the centre as the one observed, on either side (the definition of the exact enumeration)."""
    d = np.asarray([x for x in d if x != 0 and np.isfinite(x)], float); n = len(d)
    if n == 0: return 1.0, 0.0
    a = np.abs(d); ranks = _ranks(a); W = float(ranks[d > 0].sum())
    if n <= max_exact:
        r2 = np.rint(2 * ranks).astype(int); total = int(r2.sum())
        dist = np.zeros(total + 1); dist[0] = 1.0
        for r in r2:                                         # each rank enters with sign + or - (probability 1/2 each)
            new = 0.5 * dist.copy(); new[r:] += 0.5 * dist[: total + 1 - r]; dist = new
        w2 = int(round(2 * W)); lo, hi = min(w2, total - w2), max(w2, total - w2)
        return float(min(1.0, dist[: lo + 1].sum() + dist[hi:].sum())), W
    mu = n * (n + 1) / 4
    ties = np.unique(a, return_counts=True)[1]
    var = n * (n + 1) * (2 * n + 1) / 24 - float(np.sum(ties ** 3 - ties)) / 48
    z = (abs(W - mu) - 0.5) / math.sqrt(max(var, 1e-12))
    return float(min(1.0, math.erfc(max(z, 0.0) / math.sqrt(2)))), W


def boot_ci(d, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.RandomState(seed); d = np.asarray(d, float)
    m = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n_boot)])
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_table(rows, metric="final_acc", reference="rewire", out=None):
    """Paired comparison of `reference` against every other arm, per dataset and shift mode (descriptive with five seeds: the exact
    two-sided test cannot fall below P = 0.0625 with five pairs)."""
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in rows]); recs = []
    for (ds, sh), g in df.groupby(["dataset", "shift_mode"]):
        piv = g.pivot_table(index="seed", columns="arm", values=metric)
        if reference not in piv: continue
        for arm in piv.columns:
            if arm == reference: continue
            pair = piv[[reference, arm]].dropna()
            if len(pair) < 3: continue
            d = (pair[reference] - pair[arm]).values; p, _ = wilcoxon_exact(d); lo, hi = boot_ci(d)
            recs.append(dict(dataset=ds, shift_mode=sh, metric=metric, reference=reference, arm=arm, n_seeds=len(pair),
                             mean_ref=float(pair[reference].mean()), mean_arm=float(pair[arm].mean()), mean_diff=float(d.mean()),
                             ci_lo=lo, ci_hi=hi, cohen_dz=float(d.mean() / (d.std(ddof=1) + 1e-12)), p_wilcoxon=p))
    t = pd.DataFrame(recs)
    if out is not None and len(t):
        os.makedirs(out, exist_ok=True); t.to_csv(f"{out}/paired_{metric}.csv", index=False)
    return t


def pooled_table(rows, metric="final_acc", reference="rewire", out=None):
    """Paired test on the POOLED dataset x seed pairs (n = n_datasets x n_seeds, 30 in the campaign): the test with power. Holm's
    correction over the family of comparisons with the reference."""
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in rows]); recs = []
    for sh, g in df.groupby("shift_mode"):
        piv = g.pivot_table(index=["dataset", "seed"], columns="arm", values=metric)
        if reference not in piv: continue
        for arm in piv.columns:
            if arm == reference: continue
            pair = piv[[reference, arm]].dropna()
            if len(pair) < 5: continue
            d = (pair[reference] - pair[arm]).values; p, _ = wilcoxon_exact(d); lo, hi = boot_ci(d)
            recs.append(dict(shift_mode=sh, metric=metric, reference=reference, arm=arm, n_pairs=len(pair),
                             mean_ref=float(pair[reference].mean()), mean_arm=float(pair[arm].mean()), mean_diff=float(d.mean()),
                             ci_lo=lo, ci_hi=hi, cohen_dz=float(d.mean() / (d.std(ddof=1) + 1e-12)), p_wilcoxon=p,
                             win_rate=float((d > 0).mean())))
    t = pd.DataFrame(recs)
    if len(t):
        t["p_holm"] = np.nan                       # per group by INDEX: np.concatenate over groupby misaligns the rows
        for _, idx in t.groupby("shift_mode").groups.items(): t.loc[idx, "p_holm"] = holm(t.loc[idx, "p_wilcoxon"].values)
    if out is not None and len(t):
        os.makedirs(out, exist_ok=True); t.to_csv(f"{out}/pooled_{metric}.csv", index=False)
    return t


def holm(pvals):
    """Holm-Bonferroni adjusted p-values for a family of comparisons."""
    p = np.asarray(pvals, float); order = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (len(p) - i) * p[idx]); adj[idx] = min(1.0, run)
    return adj
