"""Named values for the paper: every number the text, the legends and the tables quote, under a systematic key, formatted as the
paper prints it. scripts/make_paper_figures.py writes them to macros/numbers.tex as \\defval{key}{formatted} and to facts.json
(with the unformatted value), so that the text never carries a hand-typed number.

Keys: <metric>.<arm>.<dataset> (arm and dataset tokens in compute.TOK and compute.DTOK), with .mean/.min/.max over datasets and
.rawmin/.rawmax/.featmin/.featmax over the inputs without a front end (pixels, events, spikes) and the ResNet-feature datasets."""
import re
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from .compute import (TOK, DTOK, NAN, fmt_num, fmt_signed, fmt_mj, fmt_ratio, fmt_p, fmt_int, words, ds_list, GATE_VARIANTS, ABL_LADDER, ABL_OTHER)
from .style import RAW, FEAT, PROTECTED, DS_NAME

METRICS = dict(acc=("acc", fmt_num), accsd=("acc_sd", fmt_num), bwt=("bwt", fmt_num), bwtsd=("bwt_sd", fmt_num), energy=("energy", fmt_mj),
               readm=("readm", fmt_mj), adm=("adm", fmt_mj), maint=("maint", fmt_mj), stand=("stand", fmt_int), resid=("resid", fmt_int),
               ret=("ret", fmt_num), S=("S", lambda x: fmt_num(x, 2)), erel=("Erel", fmt_num), enov=("Enov", fmt_num), cens=("cens", fmt_num),
               auroc=("auroc", lambda x: fmt_num(x, 2)), aurocsd=("auroc_sd", lambda x: fmt_num(x, 2)), auroctest=("auroc_test", lambda x: fmt_num(x, 2)),
               auroctestsd=("auroc_test_sd", lambda x: fmt_num(x, 2)), skip=("skipped", lambda x: fmt_num(100 * x, 0)), time=("time", lambda x: fmt_num(x, 0)),
               front=("front", lambda x: fmt_num(x / 1000, 1)))


def _tok(a): return TOK.get(a, a.replace("_", ""))


def ds_list_plain(items):
    items = list(items)
    if not items: return ""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
def _dtok(d): return DTOK.get(d, d)
def _ttok(th): return f"t{int(round(100 * th))}"
def _etok(e): return "e" + f"{float(e):g}".replace(".", "p")


class Facts(dict):
    def put(self, key, value, text=None, fmt=fmt_num):
        if text is None:
            text = fmt(value) if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool) else str(value)
        self[key] = dict(value=(float(value) if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool) else value), text=text)

    def range(self, key, values, fmt=fmt_num, datasets=None):
        """key.min, key.max (and the datasets where they occur), key.mean and key.absmax over a dict dataset -> value."""
        vals = {d: x for d, x in values.items() if x is not None and np.isfinite(x) and (datasets is None or d in datasets)}
        if not vals: return
        dmin, dmax = min(vals, key=vals.get), max(vals, key=vals.get)
        self.put(key + ".min", vals[dmin], fmt=fmt); self.put(key + ".max", vals[dmax], fmt=fmt)
        self.put(key + ".argmin", DS_NAME.get(dmin, dmin)); self.put(key + ".argmax", DS_NAME.get(dmax, dmax))
        self.put(key + ".mean", float(np.mean(list(vals.values()))), fmt=fmt)
        ufmt = fmt_num if fmt is fmt_signed else fmt               # magnitudes print without a sign
        self.put(key + ".absmax", float(max(abs(x) for x in vals.values())), fmt=ufmt)
        xs = list(vals.values())
        if all(x > 0 for x in xs) or all(x < 0 for x in xs):       # one sign: magnitudes, for text that already says the direction
            a = sorted(abs(x) for x in xs); self.put(key + ".amin", a[0], fmt=ufmt); self.put(key + ".amax", a[-1], fmt=ufmt)
        self.put(key + ".n", len(vals), text=str(len(vals)))
        self.put(key + ".npos", sum(x > 0 for x in vals.values()), text=str(sum(x > 0 for x in vals.values())))
        self.put(key + ".nneg", sum(x < 0 for x in vals.values()), text=str(sum(x < 0 for x in vals.values())))
        self.put(key + ".poslist", ds_list([d for d in vals if vals[d] > 0])); self.put(key + ".neglist", ds_list([d for d in vals if vals[d] < 0]))


def build(R):
    F = Facts(); D = R.datasets; raw = [d for d in D if d in RAW]; feat = [d for d in D if d in FEAT]
    # ---- meta
    F.put("meta.ndatasets", len(D), text=str(len(D))); F.put("meta.ndatasetswords", words(len(D)))
    F.put("meta.nraw", len(raw), text=str(len(raw))); F.put("meta.nrawwords", words(len(raw))); F.put("meta.nfeatwords", words(len(feat)))
    F.put("meta.datasets", ds_list(D)); F.put("meta.rawdatasets", ds_list(raw)); F.put("meta.featdatasets", ds_list(feat))
    seeds = R.camp.seeds; F.put("meta.nseeds", len(seeds), text=str(len(seeds))); F.put("meta.nseedswords", words(len(seeds)))
    # the smallest two-sided P the exact signed-rank test can give with one pair per seed (all signs alike): 2 / 2^n
    if seeds: F.put("stats.pmin", 2.0 / 2 ** len(seeds), fmt=lambda x: f"{x:.4g}")
    arms = sorted(set(R.df.arm)); F.put("meta.narms", len(arms), text=str(len(arms))); F.put("meta.ncomp", len(arms) - 1, text=str(len(arms) - 1))
    F.put("meta.nruns", len(R.df), text=fmt_int(len(R.df)))
    st = [s for s in R.camp.meta.get("status", []) if s.get("elapsed_min") is not None]
    if st: F.put("meta.hours", max(s["elapsed_min"] for s in st) / 60, fmt=lambda x: fmt_num(x, 1))
    tv = sorted({str((r.get("versions") or {}).get("torch", "")).split("+")[0] for r in R.camp.main} - {""})
    if tv: F.put("meta.torch", ", ".join(tv))
    env = "\n".join(R.camp.meta.get("environment", []))
    import re
    m = re.search(r"^(NVIDIA[^,\n]*|[A-Z][^,\n]*GPU[^,\n]*),", env, re.M)
    if m: F.put("meta.gpu", m.group(1).strip())
    elif "gpu: none" in env: F.put("meta.gpu", "a CPU without a GPU")
    m = re.search(r"cuda ([0-9.]+)", env)
    if m: F.put("meta.cuda", m.group(1))
    if R.camp.meta.get("manifests"): F.put("meta.codehash", str(R.camp.meta["manifests"][0].get("code_hash", ""))[:12])
    if R.camp.meta.get("determinism"):
        dd = R.camp.meta["determinism"][0]; mx = max((dd.get("max_abs_difference") or {}).values() or [0])
        def _sci(x):                                       # LaTeX, so that no '1e-06' glues digits to letters in the text layer
            if not x: return "0"
            e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
            return f"${m:.1f}\\times10^{{{e}}}$"
        F.put("meta.determinism", mx, fmt=_sci)
    col = [(r["dataset"], r["arm"], r["seed"]) for r in R.camp.main if r.get("collapsed")]
    F.put("meta.ncollapsed", len(col), text=str(len(col)))
    # in the text: how many runs on which dataset (no arm identifiers, which carry underscores); the full list is for the report
    from collections import Counter
    from .style import PROSE as _PROSE
    byds = Counter(DS_NAME.get(d, d) for d, a, s in col)
    parts = [f"{words(n)} on {ds}" for ds, n in byds.most_common()]
    F.put("meta.collapsed", (parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]) if parts else "none")
    F.put("meta.collapsed.detail", "; ".join(f"{DS_NAME.get(d, d)}, {_PROSE.get(a, a)}, seed {s}" for d, a, s in col) or "none")
    if R.eps_selected: F.put("eps.sel", float(R.eps_selected.get("selected", NAN)), fmt=lambda x: f"{x:g}")
    # ---- per arm and dataset
    for (d, a), row in R.A.iterrows():
        for key, (col, fmt) in METRICS.items():
            if col in row.index: F.put(f"{key}.{_tok(a)}.{_dtok(d)}", float(row[col]), fmt=fmt)
    for a in sorted(set(R.df.arm)):
        for key, (col, fmt) in METRICS.items():
            vals = {d: R.v(d, a, col) for d in D}
            if key == "skip": vals = {d: (x if np.isfinite(x) else NAN) for d, x in vals.items()}
            F.range(f"{key}.{_tok(a)}", vals, fmt=fmt)
            F.range(f"{key}.{_tok(a)}.raw", vals, fmt=fmt, datasets=raw); F.range(f"{key}.{_tok(a)}.feat", vals, fmt=fmt, datasets=feat)
        # differences with REWIRE (REWIRE minus the arm) and energy ratios (the arm over REWIRE)
        if a != "rewire":
            F.range(f"diff.{_tok(a)}", {d: R.v(d, "rewire", "acc") - R.v(d, a, "acc") for d in D}, fmt=fmt_signed)
            F.range(f"diff.{_tok(a)}.raw", {d: R.v(d, "rewire", "acc") - R.v(d, a, "acc") for d in D}, fmt=fmt_signed, datasets=raw)
            F.range(f"diff.{_tok(a)}.feat", {d: R.v(d, "rewire", "acc") - R.v(d, a, "acc") for d in D}, fmt=fmt_signed, datasets=feat)
            for d in D:
                F.put(f"diff.{_tok(a)}.{_dtok(d)}", R.v(d, "rewire", "acc") - R.v(d, a, "acc"), fmt=fmt_signed)
                F.put(f"diff.{_tok(a)}.{_dtok(d)}.abs", abs(R.v(d, "rewire", "acc") - R.v(d, a, "acc")), fmt=fmt_num)
            er = {d: R.v(d, "rewire", "energy") / R.v(d, a, "energy") for d in D}       # REWIRE's energy over the arm's
            F.range(f"eratio.{_tok(a)}", er, fmt=fmt_ratio)
            rat = {d: R.v(d, a, "energy") / R.v(d, "rewire", "energy") for d in D}
            F.range(f"ratio.{_tok(a)}", rat, fmt=fmt_ratio); [F.put(f"ratio.{_tok(a)}.{_dtok(d)}", x, fmt=fmt_ratio) for d, x in rat.items()]
            F.range(f"ratio.{_tok(a)}.raw", rat, fmt=fmt_ratio, datasets=raw); F.range(f"ratio.{_tok(a)}.feat", rat, fmt=fmt_ratio, datasets=feat)
            rr = {d: R.v(d, a, "readm") / R.v(d, "rewire", "readm") for d in D}
            F.range(f"rratio.{_tok(a)}", rr, fmt=fmt_ratio); [F.put(f"rratio.{_tok(a)}.{_dtok(d)}", x, fmt=fmt_ratio) for d, x in rr.items()]
    # ---- the protected gradient learners together (HLOP and rehearsal: the energy claim of the abstract)
    both = {}
    for a in ("hlop", "replay"):
        for d in D: both[f"{a}.{d}"] = R.v(d, a, "energy") / R.v(d, "rewire", "energy")
    xs = [x for x in both.values() if np.isfinite(x)]
    if xs:
        F.put("ratio.protected.min", min(xs), fmt=fmt_ratio); F.put("ratio.protected.max", max(xs), fmt=fmt_ratio)
        F.put("eratio.protected.min", 1 / max(xs), fmt=fmt_ratio); F.put("eratio.protected.max", 1 / min(xs), fmt=fmt_ratio)
    rr = [R.v(d, a, "readm") / R.v(d, "rewire", "readm") for a in ("hlop", "replay") for d in D]
    rr = [x for x in rr if np.isfinite(x)]
    if rr: F.put("rratio.protected.min", min(rr), fmt=fmt_ratio); F.put("rratio.protected.max", max(rr), fmt=fmt_ratio)
    # ---- savings: where REWIRE's saving is complete (no step to relearn), and its value where it is not
    comp = [d for d in D if np.isfinite(R.v(d, "rewire", "Erel")) and R.v(d, "rewire", "Erel") <= 1e-9]
    F.put("sav.complete.list", ds_list(comp)); F.put("sav.complete.n", len(comp), text=str(len(comp))); F.put("sav.complete.nwords", words(len(comp)))
    F.put("sav.complete.dtoks", " ".join(_dtok(d) for d in comp))
    F.range("S.rewire.partial", {d: R.v(d, "rewire", "S") for d in D if d not in comp}, fmt=lambda x: fmt_num(x, 2))
    # ---- HLOP's never-seen class after the largest few-shot budget: where it stays within 5 points of chance
    stuck = {}
    for (d, a), c in R.recovery.items():
        if a == "hlop" and c["novel"]: stuck[d] = c["novel"][-1]
    st = {d: x for d, x in stuck.items() if np.isfinite(x) and x <= 55.0}
    F.put("rec.hlop.stuck.list", ds_list([d for d in D if d in st])); F.put("rec.hlop.stuck.n", len(st), text=str(len(st)))
    if st: F.range("rec.hlop.stuck", st, fmt=fmt_num)
    if stuck: F.range("rec.hlop.novel.last", stuck, fmt=fmt_num)
    # ---- test-time recognition: the best any learner reaches on any dataset
    tt = [(R.v(d, a, "auroc_test"), a, d) for (d, a) in R.A.index if np.isfinite(R.v(d, a, "auroc_test"))]
    if tt:
        b = max(tt); F.put("auroctest.best.max", b[0], fmt=lambda x: fmt_num(x, 2))
        from .style import label
        F.put("auroctest.best.arm", label(b[1]).replace(" (ours)", "")); F.put("auroctest.best.dataset", DS_NAME.get(b[2], b[2]))
    # ---- protocol values from the configuration recorded with the runs (the text quotes them)
    def cfg(d, k):
        rs = [r for r in R.camp.main if r["dataset"] == d]
        return rs[0]["_cfg"].get(k) if rs else None
    d0 = D[0]
    rr = [r for r in R.camp.main if r["arm"] == "rewire"]
    if rr:
        for k in ("rw_hidden", "rw_sketch", "rw_gate_delta", "theta_hi", "full_steps", "chunk_target", "chunk_context", "batch", "lr"):
            x = rr[0]["_cfg"].get(k)
            if isinstance(x, (int, float)): F.put(f"cfg.{k}", float(x), fmt=lambda v: f"{v:g}")
    sh = [r for r in R.camp.shift] if R.camp.shift else []
    if sh:
        F.put("cfg.shiftdeg", float(sh[0]["_cfg"].get("shift_deg") or NAN), fmt=lambda x: f"{x:g}")
        F.put("cfg.shiftchannels", float(sh[0]["_cfg"].get("shift_channels") or NAN), fmt=lambda x: f"{x:g}")
    else:
        F.put("cfg.shiftdeg", float(cfg(d0, "shift_deg") or NAN), fmt=lambda x: f"{x:g}"); F.put("cfg.shiftchannels", float(cfg(d0, "shift_channels") or NAN), fmt=lambda x: f"{x:g}")
    if len(R.native) and "pmnist" in set(R.native.protocol):
        nt = int(R.native[R.native.protocol == "pmnist"].n_tasks.max()); F.put("cfg.pmnistpermutations", nt - 1, text=words(nt - 1))
    nr = lambda r: r.get("n_recur", (r.get("_cfg") or {}).get("n_recur"))
    rs_small = sorted({int(nr(r)) for r in (R.camp.sweep + R.camp.main) if nr(r) is not None and r["dataset"] not in ("cifar100", "tinyimagenet")})
    rs_big = sorted({int(nr(r)) for r in (R.camp.sweep + R.camp.main) if nr(r) is not None and r["dataset"] in ("cifar100", "tinyimagenet")})
    if rs_small: F.put("cfg.sweepsmall", ", ".join(map(str, rs_small)))
    r8 = [int(r["n_recur"]) for r in R.camp.main if r["dataset"] in ("mnist", "nmnist", "cifar10") and r.get("n_recur") is not None]
    if r8: F.put("cfg.nrecur.small", r8[0], text=str(r8[0]))
    if rs_big: F.put("cfg.sweepbig", ", ".join(map(str, rs_big)))
    buds = [c["budgets"][-1] for c in R.recovery.values() if c.get("budgets")]
    if buds: F.put("cfg.lastbudget", max(buds), text=str(max(buds))); F.put("cfg.lastbudgetwords", words(max(buds)))
    gseeds = sorted({r["seed"] for r in R.camp.gate}) if R.camp.gate else []
    F.put("meta.ngateseeds", len(gseeds), text=str(len(gseeds))); F.put("meta.ngateseedswords", words(len(gseeds)))
    # ---- per-dataset paired differences (bootstrap CI over seeds) and the pooled test
    for r in R.per_ds.itertuples():
        k = f"{_tok(r.arm)}.{_dtok(r.dataset)}"
        F.put(f"pdiff.{k}", r.diff, fmt=fmt_signed); F.put(f"plo.{k}", r.lo, fmt=fmt_signed); F.put(f"phi.{k}", r.hi, fmt=fmt_signed)
    for r in R.pooled.itertuples():
        k = _tok(r.arm)
        F.put(f"pd.{k}", r.diff, fmt=fmt_signed); F.put(f"plo.{k}", r.lo, fmt=fmt_signed); F.put(f"phi.{k}", r.hi, fmt=fmt_signed)
        F.put(f"pwin.{k}", 100 * r.win, fmt=lambda x: fmt_num(x, 0)); F.put(f"pp.{k}", r.p_holm, text=fmt_p(r.p_holm)); F.put(f"pdz.{k}", r.dz, fmt=lambda x: fmt_num(x, 2))
    if len(R.pooled):
        F.put("pool.nsig", int((R.pooled.p_holm < 0.05).sum()), text=str(int((R.pooled.p_holm < 0.05).sum())))
        F.put("pool.ncomp", len(R.pooled), text=str(len(R.pooled))); F.put("pool.npairs", int(R.pooled.n.max()), text=str(int(R.pooled.n.max())))
        ns = R.pooled[R.pooled.p_holm >= 0.05]
        from .style import label
        from .style import PROSE
        F.put("pool.nonsig", ds_list_plain([PROSE.get(a, a) for a in ns.arm]) or "none"); F.put("pool.nnonsig", len(ns), text=str(len(ns)))
    # ---- the best gradient-based baseline per dataset (HLOP, both rehearsals, EWC), and REWIRE against it: exceeds = the bootstrap interval
    #      of the paired difference above zero; matches = within H1's 2-point margin (mean over seeds); trails = more than 2 points behind
    verdicts = {}
    for d in D:
        b = R.best_gradient(d)
        if b is None: continue
        row = R.per_ds[(R.per_ds.dataset == d) & (R.per_ds.arm == b)]
        lo, hi = (float(row.lo.iloc[0]), float(row.hi.iloc[0])) if len(row) else (NAN, NAN)
        diff = R.v(d, "rewire", "acc") - R.v(d, b, "acc")
        v = "exceeds" if lo > 0 else ("matches" if diff >= -2.0 else "trails"); verdicts[d] = v
        from .style import label
        F.put(f"best.{_dtok(d)}", label(b).replace(" (ours)", "")); F.put(f"best.diff.{_dtok(d)}", R.v(d, "rewire", "acc") - R.v(d, b, "acc"), fmt=fmt_signed)
        F.put(f"best.verdict.{_dtok(d)}", v)
    n_ok = sum(v in ("exceeds", "matches") for v in verdicts.values())
    F.put("best.nok", n_ok, text=str(n_ok)); F.put("best.nokwords", words(n_ok)); F.put("best.nexceeds", sum(v == "exceeds" for v in verdicts.values()), text=str(sum(v == "exceeds" for v in verdicts.values())))
    F.put("best.trails", ds_list([d for d, v in verdicts.items() if v == "trails"])); F.put("best.ntrails", sum(v == "trails" for v in verdicts.values()), text=str(sum(v == "trails" for v in verdicts.values())))
    F.put("best.exceedslist", ds_list([d for d, v in verdicts.items() if v == "exceeds"]))
    F.range("aurocbest.rewire", {d: np.nanmax([R.v(d, "rewire", "auroc"), R.v(d, "rewire_gated_maha", "auroc")]) for d in D
                                 if np.isfinite(R.v(d, "rewire", "auroc")) or np.isfinite(R.v(d, "rewire_gated_maha", "auroc"))}, fmt=lambda x: fmt_num(x, 2))
    for d in D:
        b = R.best_gradient(d)
        if b is not None:
            F.put(f"best.acc.{_dtok(d)}", R.v(d, b, "acc"), fmt=fmt_num); F.put(f"best.diff.{_dtok(d)}.abs", abs(R.v(d, "rewire", "acc") - R.v(d, b, "acc")), fmt=fmt_num)
    # ---- the gate variants against ungated REWIRE (main grid, same seeds) and the theta sweep
    for a in GATE_VARIANTS[1:] + ["abl_sketch_ridge_gated", "hlop_gated", "replay_gated"]:
        base = {"abl_sketch_ridge_gated": "abl_sketch_ridge", "hlop_gated": "hlop", "replay_gated": "replay"}.get(a, "rewire")
        sv = {d: 100 * (1 - R.v(d, a, "energy") / R.v(d, base, "energy")) for d in D}; da = {d: R.v(d, a, "acc") - R.v(d, base, "acc") for d in D}
        for d in D:
            F.put(f"saved.{_tok(a)}.{_dtok(d)}", sv[d], fmt=lambda x: fmt_num(x, 0)); F.put(f"dacc.{_tok(a)}.{_dtok(d)}", da[d], fmt=fmt_signed)
        F.range(f"saved.{_tok(a)}", sv, fmt=lambda x: fmt_num(x, 0)); F.range(f"dacc.{_tok(a)}", da, fmt=fmt_signed)
        F.range(f"saved.{_tok(a)}.raw", sv, fmt=lambda x: fmt_num(x, 0), datasets=raw); F.range(f"saved.{_tok(a)}.feat", sv, fmt=lambda x: fmt_num(x, 0), datasets=feat)
        F.range(f"dacc.{_tok(a)}.raw", da, fmt=fmt_signed, datasets=raw); F.range(f"dacc.{_tok(a)}.feat", da, fmt=fmt_signed, datasets=feat)
    for r in R.gate.itertuples():
        k = f"{_tok(r.arm)}.{_dtok(r.dataset)}.{_ttok(r.theta)}"
        F.put(f"gate.skip.{k}", r.skipped, fmt=lambda x: fmt_num(x, 0)); F.put(f"gate.saved.{k}", r.saved, fmt=lambda x: fmt_num(x, 0))
        F.put(f"gate.dacc.{k}", r.dacc, fmt=fmt_signed)
    # ---- ablation ladder and controls (differences in points)
    ab = dict(lda=("abl_sketch_lda", "abl_sketch_ridge"), view=("rewire", "abl_sketch_lda"), viewridge=("abl_dualview_ridge", "abl_sketch_ridge"),
              ldaview=("rewire", "abl_dualview_ridge"), ladder=("rewire", "abl_sketch_ridge"), relu=("rewire", "ctrl_relu"),
              selerr=("abl_select_errors", "rewire"), selrand=("abl_select_random", "rewire"), skselerr=("abl_sketch_ridge_select_errors", "abl_sketch_ridge"),
              skselrand=("abl_sketch_ridge_select_random", "abl_sketch_ridge"), wone=("rewire", "abl_width128"), wtwo=("rewire", "abl_width256"),
              slda=("rewire", "ctrl_slda_input"), ridgein=("rewire", "ridge_input"), ranpac=("rewire", "ranpac"), hd=("rewire", "hd_memory"),
              skwood=("abl_sketch_ridge_woodbury", "abl_sketch_ridge"), sldaridge=("ctrl_slda_input", "ridge_input"))
    for k, (a, b) in ab.items():
        vals = {d: R.v(d, a, "acc") - R.v(d, b, "acc") for d in D}
        for d in D: F.put(f"abl.{k}.{_dtok(d)}", vals[d], fmt=fmt_signed)
        F.range(f"abl.{k}", vals, fmt=fmt_signed); F.range(f"abl.{k}.raw", vals, fmt=fmt_signed, datasets=raw); F.range(f"abl.{k}.feat", vals, fmt=fmt_signed, datasets=feat)
    # ---- recovery curves
    for (d, a), c in R.recovery.items():
        for b, s, n in zip(c["budgets"], c["seen"], c["novel"]):
            F.put(f"rec.{_tok(a)}.{_dtok(d)}.seen.b{b}", s, fmt=fmt_num); F.put(f"rec.{_tok(a)}.{_dtok(d)}.novel.b{b}", n, fmt=fmt_num)
    # ---- density and long streams
    for r in R.growth.itertuples():
        F.put(f"dens.growth.{_tok(r.arm)}.{_dtok(r.dataset)}", r.growth, fmt=lambda x: fmt_num(x, 2)); F.put(f"dens.rmin.{_dtok(r.dataset)}", r.r_min, text=str(r.r_min))
        F.put(f"dens.rmax.{_dtok(r.dataset)}", r.r_max, text=str(r.r_max))
    for a in sorted(set(R.growth.arm)) if len(R.growth) else []:
        g = R.growth[R.growth.arm == a]; vals = dict(zip(g.dataset, g.growth))
        F.range(f"dens.growth.{_tok(a)}", vals, fmt=lambda x: fmt_num(x, 2))
        F.range(f"dens.growth.{_tok(a)}.small", vals, fmt=lambda x: fmt_num(x, 2), datasets=[d for d in D if d not in ("cifar100", "tinyimagenet")])
        F.range(f"dens.growth.{_tok(a)}.big", vals, fmt=lambda x: fmt_num(x, 2), datasets=[d for d in D if d in ("cifar100", "tinyimagenet")])
    if len(R.growth):
        small = R.growth[~R.growth.dataset.isin(["cifar100", "tinyimagenet"])]
        if len(small): F.put("dens.rmax.small", int(small.r_max.max()), text=str(int(small.r_max.max()))); F.put("dens.rmin.small", int(small.r_min.min()), text=str(int(small.r_min.min())))
        big = R.growth[R.growth.dataset.isin(["cifar100", "tinyimagenet"])]
        if len(big): F.put("dens.rmax.big", int(big.r_max.max()), text=str(int(big.r_max.max())))
    for r in R.density.itertuples():
        F.put(f"dens.energy.{_tok(r.arm)}.{_dtok(r.dataset)}.r{int(r.n_recur)}", r.energy, fmt=fmt_mj); F.put(f"dens.acc.{_tok(r.arm)}.{_dtok(r.dataset)}.r{int(r.n_recur)}", r.acc, fmt=fmt_num)
    # ---- domain shift
    if len(R.shift):
        for r in R.shift.itertuples():
            k = f"{_tok(r.arm)}.{_dtok(r.dataset)}"
            for name, col, fmt in (("acc", "acc", fmt_num), ("accs", "acc_shifted", fmt_num), ("dacc", "dacc_vs_unshifted", fmt_signed),
                                   ("first", "first_written", lambda x: fmt_num(x, 0)), ("later", "later_written", lambda x: fmt_num(x, 0)),
                                   ("retfirst", "ret_first", fmt_num), ("retlater", "ret_later", fmt_num), ("energy", "energy", fmt_mj)):
                F.put(f"sh.{name}.{k}", getattr(r, col), fmt=fmt)
        for a in sorted(set(R.shift.arm)):
            g = R.shift[R.shift.arm == a]
            for name, col, fmt in (("acc", "acc", fmt_num), ("accs", "acc_shifted", fmt_num), ("dacc", "dacc_vs_unshifted", fmt_signed), ("first", "first_written", lambda x: fmt_num(x, 0)),
                                   ("later", "later_written", lambda x: fmt_num(x, 0)), ("retfirst", "ret_first", fmt_num)):
                F.range(f"sh.{name}.{_tok(a)}", dict(zip(g.dataset, getattr(g, col))), fmt=fmt)
        for a in ("hlop", "replay", "ctrl_slda_input", "naive"):
            vals = {d: (float(R.shift[(R.shift.dataset == d) & (R.shift.arm == "rewire")].acc_shifted.iloc[0]) - float(R.shift[(R.shift.dataset == d) & (R.shift.arm == a)].acc_shifted.iloc[0]))
                    for d in D if ((R.shift.dataset == d) & (R.shift.arm == a)).any() and ((R.shift.dataset == d) & (R.shift.arm == "rewire")).any()}
            F.range(f"sh.diff.{_tok(a)}", vals, fmt=fmt_signed)
            F.put(f"sh.ahead.{_tok(a)}", sum(v > 0 for v in vals.values()), text=str(sum(v > 0 for v in vals.values())))
    # ---- native protocols (task- and domain-incremental)
    if len(R.native):
        for r in R.native.itertuples():
            k = f"{_tok(r.arm)}.{r.protocol}"
            F.put(f"nat.acc.{k}", r.acc, fmt=fmt_num); F.put(f"nat.accsd.{k}", r.acc_sd, fmt=fmt_num); F.put(f"nat.bwt.{k}", r.bwt, fmt=fmt_num)
            F.put(f"nat.bwtsd.{k}", r.bwt_sd, fmt=fmt_num); F.put(f"nat.energy.{k}", r.energy, fmt=fmt_mj); F.put(f"nat.verdict.{k}", r.verdict)
            F.put(f"nat.tasks.{r.protocol}", r.n_tasks, text=str(int(r.n_tasks))); F.put(f"nat.tasks.{r.protocol}.words", words(r.n_tasks))
        for p in sorted(set(R.native.protocol)):
            g = R.native[R.native.protocol == p].set_index("arm")
            if "rewire" in g.index:
                for a in g.index:
                    if a != "rewire": F.put(f"nat.ratio.{_tok(a)}.{p}", g.loc[a, "energy"] / g.loc["rewire", "energy"], fmt=fmt_ratio)
            fails = [a for a in g.index if g.loc[a, "verdict"] == "fail"]
            from .style import label
            F.put(f"nat.fails.{p}", ", ".join(label(a) for a in fails) or "none"); F.put(f"nat.nfails.{p}", len(fails), text=str(len(fails)))
    # ---- shrinkage selection and sensitivity
    if len(R.eps_dev):
        for r in R.eps_dev.itertuples(): F.put(f"eps.dev.{_etok(r.eps)}", r.acc, fmt=fmt_num)
        F.put("eps.grid", ", ".join(f"{e:g}" for e in sorted(R.eps_dev.eps)))
    if len(R.eps_sens):
        for r in R.eps_sens.itertuples(): F.put(f"eps.sens.{_dtok(r.dataset)}.{_etok(r.eps)}", r.acc, fmt=fmt_num)
        rng = {d: float(g.acc.max() - g.acc.min()) for d, g in R.eps_sens.groupby("dataset")}
        F.range("eps.spread", rng, fmt=fmt_num)
    # ---- hypotheses
    for r in R.hyp.itertuples():
        d = _dtok(r.dataset)
        for h in ("h1", "h2", "h4", "h6"): F.put(f"hyp.{h}.{d}", "pass" if getattr(r, h) else "fail")
        F.put(f"hyp.h1dacc.{d}", r.h1_dacc, fmt=fmt_signed); F.put(f"hyp.h1ratio.{d}", r.h1_ratio, fmt=fmt_ratio); F.put(f"hyp.h2ratio.{d}", r.h2_ratio, fmt=fmt_ratio)
        F.put(f"hyp.h3.{d}", r.h3_rewire, fmt=fmt_num); F.put(f"hyp.h3hlop.{d}", r.h3_hlop, fmt=fmt_num); F.put(f"hyp.h3reh.{d}", r.h3_reh, fmt=fmt_num)
        F.put(f"hyp.h4auroc.{d}", r.h4_auroc, fmt=lambda x: fmt_num(x, 2)); F.put(f"hyp.h5relu.{d}", r.h5_relu, fmt=fmt_signed)
        F.put(f"hyp.h5view.{d}", r.h5_view, fmt=fmt_signed); F.put(f"hyp.h5lda.{d}", r.h5_lda, fmt=fmt_signed)
        from .style import label
        F.put(f"hyp.h6arm.{d}", {"rewire_gated": "g", "rewire_gated_maha": "g$_{\\rm M}$", "rewire_gated_t095": "g$_{.95}$", "rewire_gated_b128": "g$_{128}$"}.get(r.h6_arm, r.h6_arm))
        F.put(f"hyp.h6saved.{d}", r.h6_saved, fmt=lambda x: fmt_num(x, 0)); F.put(f"hyp.h6dacc.{d}", r.h6_dacc, fmt=fmt_signed)
    for h in ("h1", "h2", "h4", "h6"):
        n = int(R.hyp[h].sum()) if len(R.hyp) else 0
        F.put(f"hyp.{h}.npass", n, text=str(n)); F.put(f"hyp.{h}.npasswords", words(n))
        F.put(f"hyp.{h}.passlist", ds_list([d for d, ok in zip(R.hyp.dataset, R.hyp[h]) if ok]))
    # H3 and H5 as facts (their pass criteria are the author's): where REWIRE's backward transfer is at least rehearsal's / HLOP's, and
    # where each rung of the ladder (input view, discriminant, LIF layer) adds accuracy
    if len(R.hyp):
        for key, f in (("h3.reh", lambda r: r.h3_rewire >= r.h3_reh), ("h3.hlop", lambda r: r.h3_rewire >= r.h3_hlop),
                       ("h5.view", lambda r: r.h5_view > 0), ("h5.lda", lambda r: r.h5_lda > 0), ("h5.relu", lambda r: r.h5_relu > 0)):
            ok = [r.dataset for r in R.hyp.itertuples() if f(r)]
            F.put(f"hyp.{key}.npass", len(ok), text=str(len(ok))); F.put(f"hyp.{key}.npasswords", words(len(ok)))
            F.put(f"hyp.{key}.passlist", ds_list(ok)); F.put(f"hyp.{key}.faillist", ds_list([d for d in R.hyp.dataset if d not in ok]))
    # the comparison set: the protected baselines less those that fail their native-protocol check (reported, not compared against)
    from .style import label as _label
    F.put("fid.nexcluded", len(R.fid_fail), text=str(len(R.fid_fail))); F.put("fid.excluded", ds_list_plain([_label(a) for a in R.fid_fail]) or "none")
    names = (["HLOP"] if "hlop" in R.cmp else []) + (["rehearsal"] if ("replay" in R.cmp or "replay_iso" in R.cmp) else []) + (["EWC"] if "ewc" in R.cmp else [])
    F.put("cmp.names", ds_list_plain(names)); F.put("cmp.n", len(names), text=str(len(names)))
    if R.hyp_all is not None and len(R.hyp_all):
        for h in ("h1", "h2"):
            n = int(R.hyp_all[h].sum())
            F.put(f"hypall.{h}.npass", n, text=str(n)); F.put(f"hypall.{h}.npasswords", words(n))
            F.put(f"hypall.{h}.passlist", ds_list([d for d, ok in zip(R.hyp_all.dataset, R.hyp_all[h]) if ok]))
    # ---- front end
    fr = {d: R.v(d, "rewire", "front") for d in D if d in FEAT}
    if fr:
        F.range("front", {d: x / 1000 for d, x in fr.items()}, fmt=lambda x: fmt_num(x, 0))
        F.range("frontratio", {d: R.v(d, "rewire", "front") / R.v(d, "rewire", "energy") for d in fr}, fmt=lambda x: fmt_num(x, 0))
    # ---- Fig. 1e (drawn by hand): eight classes admitted in seven chunks (the first admits two), then R recurrences;
    #      E(R) = 7 x admission + R x re-admission (means over the datasets), as equation (1) with L = K - 1 + R
    for a in ("rewire", "rewire_gated", "hlop", "replay"):
        adm, rd = R.mean_over(a, "adm"), R.mean_over(a, "readm")
        F.put(f"fig1e.{_tok(a)}.slope", rd, fmt=fmt_mj); F.put(f"fig1e.{_tok(a)}.e0", 7 * adm, fmt=fmt_mj); F.put(f"fig1e.{_tok(a)}.e64", 7 * adm + 64 * rd, fmt=fmt_mj)
    # ---- panel letters of the per-dataset figures (six datasets: a-f; seven: a-g)
    F.put("fig.dslast", "abcdefghij"[len(D) - 1]); F.put("fig.costs", "abcdefghij"[len(D)])
    # ---- hardware (Akida), when measured
    for i, ak in enumerate(R.akida):
        for k, v in (ak.get("summary") or {}).items():
            if isinstance(v, (int, float)): F.put(f"akida.{k}.{ak.get('dataset', i)}", v, fmt=lambda x: fmt_num(x, 2 if abs(x) < 10 else 1))
    hw_facts(F, R)
    return F


def hw_facts(F, R):
    """Named values of the hardware measurement (hardware/akida_<dataset>.json): per dataset and as ranges over the datasets, the
    paired accuracy cost of the chip against the LIF layer on the same streams, the energy above the idle floor, the latency, the frame
    rate, the ratio of measured to modelled energy, and the checks (chip against software model, host against the campaign)."""
    hw = [ak for ak in R.akida if ak.get("backend") == "hw" and ak.get("per_seed")]
    if not hw: return
    loss, net, lat, fps, ratio, model, power = {}, {}, {}, {}, {}, {}, {}
    total, swfps = {}, {}
    same = repro = nseeds = nhw = 0; maxdiff = 0.0; floors = []
    for ak in hw:
        d = ak["dataset"]; rows = ak["per_seed"]; t = DTOK.get(d, d)
        def m(k): xs = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and np.isfinite(r[k])]; return float(np.mean(xs)) if xs else NAN
        loss[d] = float(np.mean([r["acc_lif"] - r["acc_chip"] for r in rows])); net[d] = m("net_energy_per_frame_uJ"); lat[d] = m("latency_ms")
        fps[d] = m("fps"); ratio[d] = m("measured_over_modelled"); model[d] = m("modelled_uJ"); power[d] = m("avg_power_mW")
        # the rest of the record: the energy per frame with the idle floor included, the frame rate of MetaTF's software model on the
        # host, the samples written and tested per stream, the mean number of non-zero inputs and hidden activations per frame (what the
        # operation-count model charges) and the spread of the power within a run (MetaTF's statistics)
        total[d] = m("energy_per_frame_uJ"); swfps[d] = m("sw_fps_software")
        F.put(f"hw.total.{t}", total[d], fmt=lambda x: fmt_num(x, 0)); F.put(f"hw.swfps.{t}", swfps[d], fmt=lambda x: fmt_num(x, 0))
        for k, col in (("nwritten", "n_written"), ("ntest", "n_test"), ("nnzin", "nnz_input"), ("nnzh", "nnz_hidden")):
            F.put(f"hw.{k}.{t}", m(col), fmt=lambda x: fmt_num(x, 0))
        stds = [float(x) for r in rows for st in (r.get("statistics") or []) for x in re.findall(r"Std\s+([0-9.]+)", str(st))]
        if stds: F.put(f"hw.powerstd.{t}", float(np.mean(stds)), fmt=fmt_num)
        # accuracies of the hardware run, rounded half up: they are means of five seeds on 1,600 test samples, so ties (52.65) occur
        hu = lambda x: str(Decimal(repr(float(x))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
        for k, col in (("emul", "acc_numpy"), ("chip", "acc_chip"), ("sim", "acc_sim"), ("lif", "acc_lif")): F.put(f"hw.{k}.{t}", m(col), fmt=hu)
        F.put(f"hw.emulmax.{t}", max(abs(r["acc_numpy"] - r["acc_sim"]) for r in rows), fmt=lambda x: fmt_num(x, 2))
        if np.isfinite(power[d]) and np.isfinite(m("floor_mW")): F.put(f"hw.floorshare.{t}", 100.0 * m("floor_mW") / power[d], fmt=lambda x: fmt_num(x, 0))
        # where the chip's accuracy stands among the campaign's methods on the same streams and seeds (chip minus method)
        chip = m("acc_chip")
        for arm in ("rewire", "ctrl_relu", "ctrl_slda_input", "hlop", "hlop_adaptive", "replay", "replay_iso", "ewc", "naive"):
            v = R.v(d, arm, "acc")
            if np.isfinite(v):
                F.put(f"hw.vs.{TOK.get(arm, arm)}.{t}", chip - v, fmt=fmt_signed); F.put(f"hw.vs.{TOK.get(arm, arm)}.{t}.abs", abs(chip - v))
        floors += [r["floor_mW"] for r in rows if isinstance(r.get("floor_mW"), (int, float))]
        nseeds += len(rows); same += sum(abs(r["acc_chip"] - r["acc_sim"]) < 1e-9 for r in rows)
        nhw += sum(all("Hardware" in q for q in (r.get("sequences") or ["?"])) for r in rows)
        for r in rows:
            if isinstance(r.get("acc_lif_campaign"), (int, float)):
                dd = abs(r["acc_lif"] - r["acc_lif_campaign"]); maxdiff = max(maxdiff, dd); repro += dd < 1e-3   # equal to float precision
        for k, v in (("loss", loss[d]), ("net", net[d]), ("latency", lat[d]), ("fps", fps[d]), ("ratio", ratio[d]), ("model", model[d]), ("power", power[d])):
            F.put(f"hw.{k}.{DTOK.get(d, d)}", v, fmt=(lambda x: fmt_num(x, 0)) if k in ("fps", "ratio", "power") else (lambda x: fmt_num(x, 2)) if k in ("latency", "model") else fmt_num)
    for k, vals, f in (("loss", loss, fmt_num), ("net", net, fmt_num), ("latency", lat, lambda x: fmt_num(x, 2)), ("ratio", ratio, lambda x: fmt_num(x, 0)),
                       ("fps", fps, lambda x: fmt_num(x, 0)), ("model", model, lambda x: fmt_num(x, 2)), ("total", total, lambda x: fmt_num(x, 0)),
                       ("swfps", swfps, lambda x: fmt_num(x, 0))):
        F.range(f"hw.{k}", vals, fmt=f)
    if floors: F.put("hw.floor_W", float(np.mean(floors)) / 1000, fmt=lambda x: fmt_num(x, 2))
    F.put("hw.nseeds", nseeds, text=str(nseeds)); F.put("hw.nsame", same, text=str(same)); F.put("hw.nrepro", repro, text=str(repro)); F.put("hw.nhw", nhw, text=str(nhw))
    for d in loss:
        if f"abl.relu.{DTOK.get(d, d)}" in F: F.put(f"hw.relu.{DTOK.get(d, d)}", abs(F[f'abl.relu.{DTOK.get(d, d)}']['value']))
    F.put("hw.maxdiff", maxdiff, fmt=lambda x: fmt_num(x, 2)); F.put("hw.ndatasets", len(hw), text=str(len(hw))); F.put("hw.ndatasetswords", words(len(hw)))
    F.put("hw.datasets", ds_list([ak["dataset"] for ak in hw]))
    v = hw[0].get("versions", {})
    for k in ("akida", "cnn2snn", "quantizeml", "python", "torch"):
        if v.get(k): F.put(f"hw.version.{k}", str(v[k]).replace("+cpu", ""))
    # the probe run before the measurement (hardware/probe_akida.py, its printed report kept as hardware/probe_report.txt): random weights
    # at the sizes of both networks, chip against MetaTF's software model, value by value
    from pathlib import Path
    for run in getattr(R.camp, "runs", []):
        rep = Path(run) / "hardware" / "probe_report.txt"
        if not rep.exists(): continue
        hits = re.findall(r"chip output == software model: ([0-9.]+)% of values equal; max \|diff\| ([0-9.eE+-]+)", rep.read_text(errors="replace"))
        if hits:
            F.put("hw.probe.n", len(hits), text=str(len(hits)))
            F.put("hw.probe.nexact", sum(float(a) == 100.0 and float(b) == 0.0 for a, b in hits), text=str(sum(float(a) == 100.0 and float(b) == 0.0 for a, b in hits)))
            frames = re.findall(r"chip, (\d+) frames", rep.read_text(errors="replace"))
            if frames: F.put("hw.probe.frames", int(frames[0]), fmt=fmt_int)
        break


def to_tex(F, path, header=""):
    def esc(t): return str(t).replace("%", "\\%").replace("#", "\\#")
    lines = ["% Named values of the paper, written by scripts/make_paper_figures.py from the run records. Do not edit by hand.", header,
             "\\providecommand{\\defval}[2]{\\expandafter\\gdef\\csname rv@#1\\endcsname{#2}}"]
    for k in sorted(F): lines.append(f"\\defval{{{k}}}{{{esc(F[k]['text'])}}}")
    open(path, "w").write("\n".join(lines) + "\n")
