"""Every number of the paper, computed from the records: per-run scalars, aggregates over seeds, the three costs, paired statistics,
retention, savings, recovery, ablation, the gate, recurrence density, recognition, domain shift, the native protocols, the shrinkage
sensitivity, the pre-registered hypotheses and the hardware measurement. facts() flattens them into named values with the paper's
number formats; scripts/make_paper_figures.py writes them as LaTeX macros (\\defval{key}{value}) and as facts.json."""
import math, re
import numpy as np, pandas as pd
from ..metrics.stats import wilcoxon_exact, boot_ci, holm
from .style import DS_NAME, RAW, FEAT, GRADIENT, PROTECTED, ORDER

NAN = float("nan")
TOK = dict(rewire="rewire", rewire_gated="gated", rewire_gated_maha="maha", rewire_gated_t095="tnf", rewire_gated_b128="bott",
           abl_sketch_ridge="skridge", abl_sketch_lda="sklda", abl_dualview_ridge="dvridge", abl_width128="wone", abl_width256="wtwo",
           abl_select_errors="selerr", abl_select_random="selrand", abl_sketch_ridge_select_errors="skselerr", abl_sketch_ridge_select_random="skselrand",
           abl_sketch_ridge_woodbury="skwood", abl_sketch_ridge_gated="skgated", ctrl_relu="relu", ctrl_slda_input="slda", ridge_input="ridgein",
           ranpac="ranpac", hd_memory="hd", hlop="hlop", hlop_adaptive="hlopr", hlop_gated="hlopg", replay="reh", replay_iso="rehiso",
           replay_gated="rehg", ewc="ewc", naive="ft", dsd_snn="dsd")
DTOK = dict(mnist="mnist", nmnist="nmnist", shd="shd", ssc="ssc", cifar10="cifarten", cifar100="cifarhundred", tinyimagenet="tiny")
ABL_LADDER = ["abl_sketch_ridge", "abl_sketch_lda", "abl_dualview_ridge", "rewire"]
ABL_OTHER = ["abl_width128", "abl_width256", "abl_select_errors", "abl_select_random", "abl_sketch_ridge_select_errors", "abl_sketch_ridge_select_random",
             "abl_sketch_ridge_woodbury", "ctrl_relu", "ctrl_slda_input", "ridge_input", "ranpac", "hd_memory"]
GATE_VARIANTS = ["rewire", "rewire_gated", "rewire_gated_maha", "rewire_gated_t095", "rewire_gated_b128"]
MAIN_TABLE = ["rewire", "rewire_gated", "ctrl_slda_input", "ranpac", "hlop", "replay", "replay_iso", "ewc", "naive"]
SAVINGS_ARMS = ["rewire", "rewire_gated", "ctrl_slda_input", "ranpac", "hlop", "replay", "replay_iso", "ewc", "naive"]
RECOG_ARMS = ["rewire", "rewire_gated_maha", "ctrl_slda_input", "ranpac", "hlop", "replay", "replay_iso", "ewc", "naive"]
COST_ARMS = ["rewire", "rewire_gated", "ctrl_slda_input", "ranpac", "hlop", "replay", "ewc", "naive"]
GATE_BASE = {"rewire_gated": "rewire", "rewire_gated_maha": "rewire", "abl_sketch_ridge_gated": "abl_sketch_ridge", "hlop_gated": "hlop", "replay_gated": "replay"}


# ------------------------------------------------------------------------------------------------------------------ per-run scalars
def _chunk_costs(r):
    e = np.asarray(r["energy_curve"], float); inc = np.diff(np.concatenate([[0.0], e])) / 1000.0
    v = r["visits"]; kinds = np.array([x["kind"] for x in v]); wrote = np.array([bool(x.get("wrote", True)) for x in v])
    intro, rec = kinds == "intro", kinds == "recur"
    return (float(inc[intro].mean()) if intro.any() else NAN, float(inc[rec].mean()) if rec.any() else NAN,
            float((~wrote[rec]).mean()) if rec.any() else NAN)


def _f(x, scale=1.0):
    try: return NAN if x is None else float(x) * scale
    except (TypeError, ValueError): return NAN


def scalars(rows):
    recs = []
    for r in rows:
        adm, readm, skipped = _chunk_costs(r)
        rec_ret = [x["pre_comp"] for x in r["visits"] if x["kind"] == "recur"]
        cd = r.get("censored_detail") or {}
        recs.append(dict(dataset=r["dataset"], arm=r["arm"], seed=r["seed"], n_recur=r.get("n_recur"), theta=_f(r.get("theta_hi")), eps=_f(r.get("_eps")),
                         acc=_f(r["final_acc"], 100), bwt=_f(r["bwt"], 100), energy=_f(r["lifetime_energy_uJ"], 1e-3), front=_f(r.get("frontend_energy_uJ", 0), 1e-3),
                         adm=adm, readm=readm, skipped=skipped, stand=_f(r["standing_bytes"], 1 / 1024),
                         resid=_f(r.get("resident_bytes_total", r.get("resident_bytes")), 1 / 1024),
                         auroc=_f(r.get("recog_auroc_fam_x1", r.get("recog_auroc_fam"))), auroc_incl=_f(r.get("recog_auroc_fam")),
                         auroc_test=_f(r.get("fam_auroc_test")), auroc_comp=_f(r.get("recog_auroc_comp_x1", r.get("recog_auroc_comp"))),
                         ret=_f(r.get("retention_mean", np.mean(rec_ret) if rec_ret else NAN), 100),
                         S=(_f(r.get("S_ctrl")) if _f(r.get("E_novel")) >= 1.0 else NAN),          # undefined when a new class needs no step
                         Erel=_f(r.get("E_relearn")), Enov=_f(r.get("E_novel")),
                         S_AUR=(_f(r.get("S_AUR")) if _f(r.get("AUR_novel", 0.0)) <= 0.99 else NAN),
                         cens=_f((cd.get("relearn", 0) or 0) + (cd.get("novel", 0) or 0)),
                         time=_f(r.get("train_time_s")), acc_shifted=_f(r.get("final_acc_shifted"), 100), acc_unshifted=_f(r.get("final_acc_unshifted"), 100),
                         first_written=_f(r.get("first_shift_written_share"), 100), later_written=_f(r.get("later_shift_written_share"), 100),
                         ret_first=_f(r.get("retention_first_shift"), 100), ret_later=_f(r.get("retention_later_shift"), 100),
                         written=_f(r.get("recur_written_share"), 100), solves=_f(r.get("number_of_solves")), n_holdout=len(r.get("holdout") or [])))
    df = pd.DataFrame(recs)
    if len(df):
        ft = df[df.arm == "naive"].set_index(["dataset", "seed"])["adm"].to_dict()
        df["maint"] = [(0.0 if a not in GRADIENT else (NAN if a == "naive" else adm - ft.get((d, s), NAN))) for a, d, s, adm in zip(df.arm, df.dataset, df.seed, df.adm)]
    return df


def agg(df, keys=("dataset", "arm"), cols=None):
    cols = cols or [c for c in df.columns if c not in ("dataset", "arm", "seed") and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(list(keys))[cols]
    m, s, n = g.mean(), g.std(ddof=1), g.count()
    out = m.copy()
    for c in cols: out[c + "_sd"] = s[c]; out[c + "_sem"] = s[c] / np.sqrt(n[c].clip(lower=1)); out[c + "_n"] = n[c]
    return out


# ------------------------------------------------------------------------------------------------------------------ paired statistics
def paired_pooled(df, metric="acc", ref="rewire", arms=None):
    piv = df.pivot_table(index=["dataset", "seed"], columns="arm", values=metric)
    recs = []
    for a in (arms or [c for c in piv.columns if c != ref]):
        if a == ref or a not in piv or ref not in piv: continue
        pair = piv[[ref, a]].dropna()
        if len(pair) < 5: continue
        d = (pair[ref] - pair[a]).values; p, _ = wilcoxon_exact(d); lo, hi = boot_ci(d)
        recs.append(dict(arm=a, n=len(d), diff=float(d.mean()), lo=lo, hi=hi, win=float((d > 0).mean()), dz=float(d.mean() / (d.std(ddof=1) + 1e-12)), p=p))
    t = pd.DataFrame(recs)
    if len(t): t["p_holm"] = holm(t.p.values); t = t.sort_values("diff", ascending=False).reset_index(drop=True)
    return t


def paired_per_dataset(df, metric="acc", ref="rewire", arms=None):
    recs = []
    for (ds), g in df.groupby("dataset"):
        piv = g.pivot_table(index="seed", columns="arm", values=metric)
        for a in (arms or [c for c in piv.columns if c != ref]):
            if a == ref or a not in piv or ref not in piv: continue
            pair = piv[[ref, a]].dropna()
            if len(pair) < 2: continue
            d = (pair[ref] - pair[a]).values; p, _ = wilcoxon_exact(d); lo, hi = boot_ci(d)
            recs.append(dict(dataset=ds, arm=a, n=len(d), diff=float(d.mean()), lo=lo, hi=hi, p=p))
    return pd.DataFrame(recs)


# ------------------------------------------------------------------------------------------------------------------ curves
def retention_curves(rows):
    out = {}
    for (ds, a) in sorted({(r["dataset"], r["arm"]) for r in rows}):
        cs = [[x["pre_comp"] for x in r["visits"] if x["kind"] == "recur"] for r in rows if r["dataset"] == ds and r["arm"] == a]
        L = min(map(len, cs)) if cs else 0
        if L: out[(ds, a)] = 100 * np.nanmean(np.array([c[:L] for c in cs]), 0)
    return out


def stream_curves(rows):
    out = {}
    for (ds, a) in sorted({(r["dataset"], r["arm"]) for r in rows}):
        rs = [r for r in rows if r["dataset"] == ds and r["arm"] == a]
        L = min(len(r["acc_curve"]) for r in rs)
        acc = 100 * np.mean([r["acc_curve"][:L] for r in rs], 0); en = np.mean([r["energy_curve"][:L] for r in rs], 0) / 1000
        n_intro = sum(1 for x in rs[0]["visits"] if x["kind"] == "intro")
        out[(ds, a)] = dict(acc=acc, energy=en, n_intro=n_intro, n_seeds=len(rs))
    return out


def recovery(rows):
    out = {}
    for (ds, a) in sorted({(r["dataset"], r["arm"]) for r in rows}):
        rs = [r for r in rows if r["dataset"] == ds and r["arm"] == a and r.get("rec_curve_seen")]
        if not rs: continue
        bud = sorted(int(b) for b in rs[0]["rec_curve_seen"])
        out[(ds, a)] = dict(budgets=bud, seen=[100 * np.mean([r["rec_curve_seen"][str(b)] for r in rs]) for b in bud],
                            novel=[100 * np.mean([r["rec_curve_novel"][str(b)] for r in rs]) for b in bud])
    return out


# ------------------------------------------------------------------------------------------------------------------ the gate and the sweeps
def gate_sweep(camp):
    """theta sweep: per dataset, gated arm and theta: share of recurrences skipped, lifetime energy saved and the accuracy change
    against the ungated base arm on the same seeds (main grid)."""
    main = scalars(camp.main); gate = scalars(camp.gate) if camp.gate else pd.DataFrame(columns=main.columns)
    seeds = sorted(set(gate.seed)) if len(gate) else []
    rows = pd.concat([gate, main[main.arm.isin(GATE_BASE) & main.seed.isin(seeds)]]) if len(gate) else main[main.arm.isin(GATE_BASE)]
    base = main.set_index(["dataset", "arm", "seed"])
    recs = []
    for (ds, a, th), g in rows.groupby(["dataset", "arm", "theta"]):
        b = GATE_BASE.get(a)
        if b is None: continue
        sv, da, sk = [], [], []
        for _, r in g.iterrows():
            k = (ds, b, r.seed)
            if k in base.index:
                sv.append(100 * (1 - r.energy / base.loc[k, "energy"])); da.append(r.acc - base.loc[k, "acc"]); sk.append(100 * r.skipped)
        recs.append(dict(dataset=ds, arm=a, theta=th, skipped=np.mean(sk) if sk else NAN, saved=np.mean(sv) if sv else NAN, dacc=np.mean(da) if da else NAN, n=len(sv)))
    return pd.DataFrame(recs)


def density(camp):
    """Recurrence density: lifetime energy and final accuracy against R, per dataset and sweep arm (the main grid supplies its R)."""
    from ..campaign import SWEEP_ARMS
    df = pd.concat([scalars(camp.sweep), scalars([r for r in camp.main if r["arm"] in SWEEP_ARMS])]) if camp.sweep else scalars([r for r in camp.main if r["arm"] in SWEEP_ARMS])
    a = agg(df, keys=("dataset", "arm", "n_recur"), cols=["energy", "acc", "skipped"]).reset_index()
    growth = []
    for (ds, arm), g in a.groupby(["dataset", "arm"]):
        g = g.sort_values("n_recur")
        if len(g) >= 2: growth.append(dict(dataset=ds, arm=arm, r_min=int(g.n_recur.iloc[0]), r_max=int(g.n_recur.iloc[-1]), growth=float(g.energy.iloc[-1] / g.energy.iloc[0]),
                                           acc_min=float(g.acc.iloc[0]), acc_max=float(g.acc.iloc[-1])))
    return a, pd.DataFrame(growth)


def shift_table(camp):
    if not camp.shift: return pd.DataFrame()
    s = scalars(camp.shift); m = scalars(camp.main).set_index(["dataset", "arm", "seed"])
    s["dacc_vs_unshifted"] = [r.acc - m.loc[(r.dataset, r.arm, r.seed), "acc"] if (r.dataset, r.arm, r.seed) in m.index else NAN for r in s.itertuples()]
    return agg(s, cols=["acc", "acc_shifted", "acc_unshifted", "ret_first", "ret_later", "first_written", "later_written", "energy", "dacc_vs_unshifted", "auroc"]).reset_index()


def native_table(camp):
    if not camp.native: return pd.DataFrame()
    from ..native import verdict
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "A"} for r in camp.native])
    df["acc"] *= 100; df["bwt"] *= 100; df["energy"] = df["lifetime_energy_uJ"] / 1000
    g = df.groupby(["protocol", "arm"]).agg(acc=("acc", "mean"), acc_sd=("acc", "std"), bwt=("bwt", "mean"), bwt_sd=("bwt", "std"),
                                             energy=("energy", "mean"), n=("seed", "count"), n_tasks=("n_tasks", "max")).reset_index()
    g["verdict"] = [verdict(p, a, ac / 100, bw / 100) for p, a, ac, bw in zip(g.protocol, g.arm, g.acc, g.bwt)]
    return g


def eps_tables(camp):
    sens = agg(scalars(camp.eps_sens), keys=("dataset", "eps"), cols=["acc", "energy"]).reset_index() if camp.eps_sens else pd.DataFrame()
    dev = scalars(camp.eps_dev).groupby("eps")["acc"].mean().reset_index() if camp.eps_dev else pd.DataFrame()
    return sens, dev, camp.meta.get("eps_selected")


# ------------------------------------------------------------------------------------------------------------------ hypotheses
def fidelity_failures(native):
    """The protected gradient baselines that fail their native-protocol check on any protocol (rewire/native.py). By the protocol's rule
    such a baseline is reported but not used in the comparisons (the best and the cheapest gradient-based baseline, the verdicts)."""
    if not len(native): return ()
    bad = set(native[native.verdict == "fail"].arm)
    return tuple(a for a in PROTECTED if a in bad)


def hypotheses(A, gs, datasets, ref=PROTECTED):
    """The pre-registered hypotheses, per dataset. H1: REWIRE within 2 points of the best gradient learner at <= 50% of the cheapest
    gradient learner's lifetime energy; H2: re-admission at <= 50% of the cheapest gradient learner's; H4: stream recognition AUROC
    >= 0.90 (cosine familiarity); H6: some gate setting saves >= 40% of lifetime energy at <= 2 points. The gradient learners of H1
    and H2 are the protected ones (HLOP, rehearsal, iso-memory rehearsal, EWC), as in the pre-registered code, less any that fails its
    fidelity check (`ref`); Results also keeps the outcome against the full set. H3 (retention, BWT) and H5 (the ablation ladder) are
    reported as values; their verdicts follow the author's pre-registered criteria."""
    out = []
    for ds in datasets:
        def v(a, c):
            try: return float(A.loc[(ds, a), c])
            except KeyError: return NAN
        prot = [a for a in ref if (ds, a) in A.index]
        best = max(prot, key=lambda a: v(a, "acc")) if prot else None; cheap = min(prot, key=lambda a: v(a, "energy")) if prot else None
        cheap_r = min(prot, key=lambda a: v(a, "readm")) if prot else None
        dacc = v("rewire", "acc") - v(best, "acc") if best else NAN; er = v("rewire", "energy") / v(cheap, "energy") if cheap else NAN
        rr = v("rewire", "readm") / v(cheap_r, "readm") if cheap_r else NAN
        h6 = []
        for a in ("rewire_gated", "rewire_gated_maha", "rewire_gated_t095", "rewire_gated_b128"):
            sv = 100 * (1 - v(a, "energy") / v("rewire", "energy")); da = v(a, "acc") - v("rewire", "acc")
            h6.append((a, sv, da, sv >= 40 and da >= -2))
        passing = [x for x in h6 if x[3]]
        closest = passing[0] if passing else max(h6, key=lambda x: (x[1] if x[2] >= -2 else -1e9, x[2]))
        out.append(dict(dataset=ds, h1=bool(dacc >= -2 and er <= 0.5), h1_dacc=dacc, h1_ratio=er, h1_best=best, h1_cheap=cheap,
                        h2=bool(rr <= 0.5), h2_ratio=rr, h2_cheap=cheap_r,
                        h3_rewire=v("rewire", "bwt"), h3_hlop=v("hlop", "bwt"), h3_reh=v("replay", "bwt"),
                        h4=bool(v("rewire", "auroc") >= 0.90), h4_auroc=v("rewire", "auroc"),
                        h5_relu=v("rewire", "acc") - v("ctrl_relu", "acc"), h5_view=v("rewire", "acc") - v("abl_sketch_lda", "acc"),
                        h5_lda=v("abl_sketch_lda", "acc") - v("abl_sketch_ridge", "acc"),
                        h6=bool(passing), h6_arm=closest[0], h6_saved=closest[1], h6_dacc=closest[2]))
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------------------------ the whole set
class Results:
    """All aggregates of a campaign, computed once."""
    def __init__(self, camp):
        self.camp = camp; self.datasets = camp.datasets
        self.df = scalars(camp.main)
        self.A = agg(self.df)
        self.native = native_table(camp)
        self.fid_fail = fidelity_failures(self.native)                                   # reported, not compared against
        self.cmp = tuple(a for a in PROTECTED if a not in self.fid_fail)                 # the comparison set of the verdicts and H1/H2
        others = [a for a in sorted(self.df.arm.unique()) if a != "rewire" and a not in self.fid_fail]   # the paired tests leave them out too
        self.pooled = paired_pooled(self.df, "acc", arms=others)
        self.per_ds = paired_per_dataset(self.df, "acc", arms=others)
        self.pooled_energy = paired_pooled(self.df, "energy", arms=others)
        self.ret_curves = retention_curves(camp.main)
        self.streams = stream_curves(camp.main)
        self.recovery = recovery(camp.main)
        self.gate = gate_sweep(camp)
        self.density, self.growth = density(camp)
        self.shift = shift_table(camp)
        self.eps_sens, self.eps_dev, self.eps_selected = eps_tables(camp)
        self.hyp = hypotheses(self.A, self.gate, self.datasets, ref=self.cmp)
        self.hyp_all = hypotheses(self.A, self.gate, self.datasets, ref=PROTECTED) if self.fid_fail else None
        self.akida = camp.akida

    def v(self, ds, arm, col):
        try: return float(self.A.loc[(ds, arm), col])
        except KeyError: return NAN

    def mean_over(self, arm, col, datasets=None):
        xs = [self.v(d, arm, col) for d in (datasets or self.datasets)]; xs = [x for x in xs if np.isfinite(x)]
        return float(np.mean(xs)) if xs else NAN

    def best_gradient(self, ds, arms=None):
        cand = [a for a in (arms or self.cmp) if (ds, a) in self.A.index]
        return max(cand, key=lambda a: self.v(ds, a, "acc")) if cand else None


# ------------------------------------------------------------------------------------------------------------------ number formats
def fmt_num(x, nd=1):
    if x is None or not np.isfinite(x): return "--"
    s = f"{abs(x):,.{nd}f}"
    if float(s.replace(",", "")) == 0: return f"{0:.{nd}f}"
    return ("\\ensuremath{-}" if x < 0 else "") + s


def fmt_signed(x, nd=1):
    if x is None or not np.isfinite(x): return "--"
    s = f"{abs(x):,.{nd}f}"
    if float(s.replace(",", "")) == 0: return f"{0:.{nd}f}"
    return ("+" if x > 0 else "\\ensuremath{-}") + s


def fmt_mj(x):
    if x is None or not np.isfinite(x): return "--"
    return fmt_num(x, 1 if abs(x) < 100 else 0)


def fmt_ratio(x):
    if x is None or not np.isfinite(x): return "--"
    return f"{x:.2f}" if x < 1 else f"{x:.1f}"


def fmt_p(p):
    if p is None or not np.isfinite(p): return "--"
    if p < 0.001: return "$P<0.001$"
    return f"$P={p:.3f}$" if p < 0.1 else f"$P={p:.2f}$"


def fmt_int(x):
    if x is None or not np.isfinite(x): return "--"
    return f"{int(round(x)):,}"


WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def words(n): return WORDS.get(int(n), str(int(n)))


def ds_list(dss):
    names = [DS_NAME.get(d, d) for d in dss]
    if not names: return "none"
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
