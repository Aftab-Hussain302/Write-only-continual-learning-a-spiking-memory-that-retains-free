"""The paper's tables, each written as an \\input-ready .tex file holding only the tabular (the caption and the label stay in the
manuscript, so that merging a new version is a single \\input) and as a CSV with the unrounded values. Formats follow the manuscript:
booktabs rules, mean +- s.d. over seeds, points for accuracies, mJ for energies, KB for memory, best in bold and second underlined."""
import numpy as np, pandas as pd
from .compute import (MAIN_TABLE, SAVINGS_ARMS, RECOG_ARMS, COST_ARMS, GATE_VARIANTS, ABL_LADDER, ABL_OTHER, fmt_num, fmt_signed, fmt_mj, fmt_int, fmt_p, fmt_ratio, NAN)
from .style import DS_NAME, RAW, FEAT, TEX_LABEL, ORDER

HEAD = dict(rewire="REWIRE", rewire_gated="REWIRE-g", ctrl_slda_input="SLDA-in", ranpac="RanPAC", hlop="HLOP", replay="Reh.", replay_iso="Reh.-iso", ewc="EWC", naive="FT")


def _pm(m, s, nd=1): return "--" if not np.isfinite(m) else f"{fmt_num(m, nd)}$\\pm${fmt_num(s, nd) if np.isfinite(s) else '--'}"


def _write(path, tex, df=None):
    open(path + ".tex", "w").write(tex)
    if df is not None: df.to_csv(path + ".csv", index=False)


def _rank_marks(vals):
    """Bold the best and underline the second best (ties share the mark)."""
    fin = sorted({round(v, 1) for v in vals if np.isfinite(v)}, reverse=True)
    best = fin[0] if fin else None; second = fin[1] if len(fin) > 1 else None
    return [("b" if np.isfinite(v) and round(v, 1) == best else "u" if np.isfinite(v) and round(v, 1) == second else "") for v in vals]


def _mark(s, m): return f"\\textbf{{{s}}}" if m == "b" else (f"\\underline{{{s}}}" if m == "u" else s)


def table1_main(R, out):
    D = R.datasets; arms = [a for a in MAIN_TABLE if any((d, a) in R.A.index for d in D)]
    cols = "ll" + "r" * len(arms)
    fail = set(getattr(R, "fid_fail", ()))                   # baselines that fail their fidelity check: reported, not ranked or compared
    L = [f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{cols}@{{}}}}", "\\toprule",
         "Dataset & & " + " & ".join(HEAD.get(a, a) + ("$^{\\dagger}$" if a in fail else "") for a in arms) + " \\\\",
         " & & " + " & ".join("(ours)" if a in ("rewire", "rewire_gated") else "" for a in arms) + " \\\\", "\\midrule"]
    recs = []
    raw = [d for d in D if d in RAW]; feat = [d for d in D if d not in RAW]
    for grp in (raw, feat):
        for d in grp:
            for met in ("acc", "bwt"):
                vals = [R.v(d, a, met) for a in arms]; marks = _rank_marks([NAN if a in fail else x for a, x in zip(arms, vals)])
                cells = [_mark(_pm(R.v(d, a, met), R.v(d, a, met + "_sd")), m) for a, m in zip(arms, marks)]
                L.append(f"{DS_NAME.get(d, d) if met == 'acc' else ''} & {'ACC' if met == 'acc' else 'BWT'} & " + " & ".join(cells) + " \\\\")
                recs += [dict(dataset=d, metric=met, arm=a, mean=R.v(d, a, met), sd=R.v(d, a, met + "_sd")) for a in arms]
        if grp is raw and feat: L.append("\\midrule")
    L.append("\\midrule")
    for key, lab, f in (("energy", "Lifetime energy (mJ)", fmt_mj), ("readm", "Re-admission (mJ)", fmt_mj), ("stand", "Standing memory (KB)", fmt_int)):
        vals = [R.mean_over(a, key) for a in arms]
        L.append(("Cost (mean)" if key == "energy" else "") + f" & {lab} & " + " & ".join(f(v) for v in vals) + " \\\\")
        recs += [dict(dataset="mean", metric=key, arm=a, mean=v, sd=NAN) for a, v in zip(arms, vals)]
    L.append("\\bottomrule")
    if fail:
        names = " and ".join(HEAD.get(a, a) for a in arms if a in fail)
        L.append(f"\\multicolumn{{{len(arms) + 2}}}{{@{{}}l}}{{\\footnotesize $^{{\\dagger}}${names} fails its native-protocol check (Extended Data "
                 f"Table~\\ref{{edtab:native}}): reported, but not ranked or compared against.}} \\\\")
    L.append("\\end{tabular*}")
    _write(f"{out}/tab1_main", "\n".join(L) + "\n", pd.DataFrame(recs))


def ed_native(R, out):
    g = R.native
    if not len(g): _write(f"{out}/ed_tab1_native", "\\begin{tabular}{l}\\toprule native protocols not in the records\\\\\\bottomrule\\end{tabular}\n"); return
    L = ["\\begin{tabular}{llrrrl}", "\\toprule", "Protocol & Method & ACC & BWT & Energy (mJ) & Verdict \\\\", "\\midrule"]   # natural width
    names = dict(split="Split-MNIST (TIL)", pmnist="Permuted MNIST (DIL)")
    order = ["rewire", "ctrl_slda_input", "hlop", "ewc", "dsd_snn", "naive"]
    for p in [p for p in ("split", "pmnist") if p in set(g.protocol)]:
        gp = g[g.protocol == p].set_index("arm")
        for i, a in enumerate([a for a in order if a in gp.index] + [a for a in gp.index if a not in order]):
            r = gp.loc[a]; lab = TEX_LABEL.get(a, a)
            L.append(f"{names[p] if i == 0 else ''} & {lab} & {_pm(r.acc, r.acc_sd)} & {_pm(r.bwt, r.bwt_sd)} & {fmt_mj(r.energy)} & {r.verdict} \\\\")
        L.append("\\midrule" if p == "split" else "")
    L = [l for l in L if l] + ["\\bottomrule", "\\end{tabular}"]
    _write(f"{out}/ed_tab1_native", "\n".join(L) + "\n", g)


def ed_akida(R, out):
    if not R.akida:
        _write(f"{out}/ed_tab2_akida", "\\begin{tabular}{l}\\toprule Akida measurement not in the records (hardware/akida/run\\_akida.py writes it)\\\\\\bottomrule\\end{tabular}\n"); return
    rows = []
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrrrrr@{}}", "\\toprule",
         "Dataset & ACC chip & ACC sim. & ACC LIF & Floor (mW) & Energy/frame ($\\mu$J) & Net ($\\mu$J) & Latency (ms) & Measured/modelled \\\\", "\\midrule"]
    for ak in R.akida:
        s = ak.get("summary", {}); d = ak.get("dataset", "?")
        vals = [s.get(k, NAN) for k in ("acc_chip", "acc_sim", "acc_lif", "floor_mW", "energy_per_frame_uJ", "net_energy_per_frame_uJ", "latency_ms", "measured_over_modelled")]
        L.append(f"{DS_NAME.get(d, d)} & " + " & ".join(fmt_num(v, 1 if i < 3 else 2) for i, v in enumerate(vals)) + " \\\\"); rows.append(dict(dataset=d, **s))
    L += ["\\bottomrule", "\\end{tabular*}"]
    _write(f"{out}/ed_tab2_akida", "\n".join(L) + "\n", pd.DataFrame(rows))


def si_hardware(R, out):
    """The hardware measurement seed by seed (hardware/akida_<dataset>.json): accuracy with the LIF layer on the host and in the campaign,
    with the converted network in the NumPy emulation, in MetaTF's software model and on the chip; power, frame rate, energy per frame
    with the idle floor and above it, the operation-count model of the same layers and the ratio of measured (above idle) to modelled."""
    hw = [ak for ak in R.akida if ak.get("backend") == "hw" and ak.get("per_seed")]
    if not hw:
        _write(f"{out}/si_tab_hardware", "\\begin{tabular}{l}\\toprule No hardware measurement in the records\\\\\\bottomrule\\end{tabular}\n"); return
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llrrrrrrrrrrr@{}}", "\\toprule",
         " & & \\multicolumn{5}{c}{Final accuracy (\\%)} & & & \\multicolumn{3}{c}{Energy per frame ($\\mu$J)} & \\\\",
         "\\cmidrule(lr){3-7}\\cmidrule(lr){10-12}",
         "Dataset & Seed & LIF, host & LIF, campaign & Emulation & Software model & Chip & Power (mW) & Frames/s & Total & Above idle & Modelled & Ratio \\\\", "\\midrule"]
    rows = []
    for j, ak in enumerate(hw):
        d = ak["dataset"]; ps = sorted(ak["per_seed"], key=lambda r: r["seed"])
        cols = ("acc_lif", "acc_lif_campaign", "acc_numpy", "acc_sim", "acc_chip", "avg_power_mW", "fps", "energy_per_frame_uJ", "net_energy_per_frame_uJ",
                "modelled_uJ", "measured_over_modelled")
        nd = (2, 2, 2, 2, 2, 0, 0, 0, 1, 2, 0)
        for i, r in enumerate(ps):
            vals = [r.get(k, NAN) for k in cols]
            L.append(f"{DS_NAME.get(d, d) if i == 0 else ''} & {r['seed']} & " + " & ".join(fmt_num(v, n) for v, n in zip(vals, nd)) + " \\\\")
            rows.append(dict(dataset=d, seed=r["seed"], **{k: r.get(k) for k in cols}))
        means = [float(np.mean([r[k] for r in ps if isinstance(r.get(k), (int, float))])) if any(isinstance(r.get(k), (int, float)) for r in ps) else NAN for k in cols]
        L.append(f" & mean & " + " & ".join(fmt_num(v, n) for v, n in zip(means, nd)) + " \\\\")
        if j < len(hw) - 1: L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular*}"]
    _write(f"{out}/si_tab_hardware", "\n".join(L) + "\n", pd.DataFrame(rows))


def si_protocol(R, out):
    """The protocol, architecture and cost model, from the configuration recorded with the runs."""
    def c(d, k):
        rs = [r for r in R.camp.main if r["dataset"] == d]
        return rs[0]["_cfg"].get(k) if rs else None
    D = R.datasets; s8 = [d for d in D if c(d, "n_stream_classes") and c(d, "n_stream_classes") <= 16]; s80 = [d for d in D if d not in s8]
    def streams(ds):
        groups = {}
        for d in ds: groups.setdefault((c(d, "n_stream_classes"), c(d, "n_recur"), c(d, "n_holdout")), []).append(DS_NAME[d])
        return "; ".join(f"{', '.join(v)}: {k[0]} classes, {k[1]} recurrence chunks, {k[2]} held out" for k, v in groups.items())
    def eff(d, k):
        """The per-class pool and test sizes the streams actually used (recorded per run; the configured value caps them, and a dataset
        with fewer samples per class after the standardisation subset is removed gets fewer)."""
        rs = [r for r in R.camp.main if r["dataset"] == d]
        x = rs[0].get(k + "_eff") if rs else None
        return x if x is not None else c(d, k)
    def samples(ds):
        groups = {}
        for d in ds: groups.setdefault((eff(d, "n_train_per_class"), eff(d, "n_test_per_class")), []).append(DS_NAME[d])
        return "; ".join(f"{a} train / {b} test ({', '.join(v)})" for (a, b), v in groups.items())
    d0 = D[0]; eps = R.eps_selected.get("selected") if R.eps_selected else c(d0, "rw2_eps")
    rows = [("Datasets", ", ".join(DS_NAME[d] for d in D if d in RAW) + " (no front end); " + ", ".join(DS_NAME[d] for d in D if d in FEAT) + " (frozen ImageNet-pretrained ResNet-18, 512-d features, identical for every method)"),
            ("Streams", streams(D) + "; minimum gap of 2 chunks between visits of a class; recurrences drawn with probability proportional to the time since the last visit"),
            ("Samples per class", samples(D) + "; feature standardisation on a fixed 10\\% subset removed from every stream and test set"),
            ("Chunk", f"{c(d0, 'chunk_target')} target samples of the arriving class, of which the first {max(4, int(c(d0, 'chunk_target') or 80) // 5)} are held out and never written, + {c(d0, 'chunk_context')} context samples of the two most recent other classes; the first chunk introduces two classes (interleaved, no context)"),
            ("Substrate", f"{c(d0, 'rw_hidden')} leaky integrate-and-fire units, leak $\\beta=0.95$, threshold $u_{{\\rm th}}=1$, $T={c(d0, 'T')}$ (20 for spike trains), random afferent weights folded to a drive standard deviation of 1.5 on a calibration batch, never trained"),
            ("Memory", f"dual view of a $d_s={c(d0, 'rw_sketch')}$ rectified sketch and an input view ($\\le512$ dimensions); count-free discriminant with shrinkage $\\varepsilon={float(eps):g}$; one-hot ridge with $\\lambda=1$ (ridge ablations and controls); gate $\\delta=0.05$, $\\theta=0.85$, budget $B=128$"),
            ("Gradient baselines", f"{c(d0, 'hidden')} LIF units, surrogate gradient (slope 2), Adam, learning rate $10^{{-3}}$ scaled by $\\sqrt{{784/d_{{\\rm in}}}}$, class-balanced batches of up to {c(d0, 'batch')} samples, the same number from each class of a chunk ({3 * (int(c(d0, 'batch') or 32) // 3)} in a single-class chunk), {c(d0, 'full_steps')} updates per chunk; HLOP: the authors' Hebbian module; rehearsal: one reservoir buffer of 20 samples per stream class, filled from every sample and not class-balanced, or of as many samples as fit in REWIRE's standing memory (iso-memory); EWC: running-mean diagonal Fisher, $\\lambda_{{\\rm EWC}}=10^3$"),
            ("Energy", "$E_{\\rm MAC}=4.6$\\,pJ, $E_{\\rm AC}=0.9$\\,pJ at 45\\,nm; synaptic operations from the spikes emitted; one accumulate per input event per postsynaptic unit on event and spike data; backward factor 2; the shared front end reported separately"),
            ("Seeds and runs", f"{len(R.camp.seeds)} seeds per dataset and method, paired by the stream schedule ({len(set(R.df.arm))} methods and variants $\\times$ {len(D)} datasets $\\times$ {len(R.camp.seeds)} seeds = {len(R.df)} main-grid runs)")]
    L = ["\\begin{tabular}{p{0.17\\textwidth}p{0.77\\textwidth}}", "\\toprule", "Setting & Value \\\\", "\\midrule"] + [f"{a} & {b} \\\\" for a, b in rows] + ["\\bottomrule", "\\end{tabular}"]
    _write(f"{out}/si_tab_protocol", "\n".join(L) + "\n", pd.DataFrame(rows, columns=["setting", "value"]))


def si_paired(R, out):
    t = R.pooled
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrr@{}}", "\\toprule", "REWIRE minus & mean diff. (pts) & 95\\% CI & wins & $P$ (Holm) & pairs \\\\", "\\midrule"]
    for r in t.itertuples():
        star = "$^{*}$" if r.p_holm < 0.05 else ""
        p = "$<0.001$" if r.p_holm < 0.001 else f"{r.p_holm:.3f}"
        L.append(f"{TEX_LABEL.get(r.arm, r.arm)} & {fmt_signed(r.diff)} & [{fmt_signed(r.lo)}, {fmt_signed(r.hi)}] & {100 * r.win:.0f}\\% & {p}{star} & {r.n} \\\\")
    L += ["\\bottomrule", "\\end{tabular*}"]
    _write(f"{out}/si_tab_paired_pooled", "\n".join(L) + "\n", t)
    p = R.per_ds; D = R.datasets
    L = ["\\begin{tabular*}{\\textheight}{@{\\extracolsep{\\fill}}l" + "rr" * len(D) + "@{}}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DS_NAME[d]}}}" for d in D) + " \\\\",
         "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(D))), "REWIRE minus & " + " & ".join("diff. & 95\\% CI" for _ in D) + " \\\\", "\\midrule"]
    for a in t.arm:
        cells = []
        for d in D:
            r = p[(p.dataset == d) & (p.arm == a)]
            cells += ([fmt_signed(float(r["diff"].iloc[0])), f"[{fmt_signed(float(r.lo.iloc[0]))}, {fmt_signed(float(r.hi.iloc[0]))}]"] if len(r) else ["--", "--"])
        L.append(f"{TEX_LABEL.get(a, a)} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular*}"]
    _write(f"{out}/si_tab_paired_ds", "\n".join(L) + "\n", p)


def _per_ds_block(R, out, name, arms, quantities, extra_rows=None):
    D = R.datasets
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lp{3.2cm}" + "r" * len(D) + "@{}}", "\\toprule",
         "Method & Quantity & " + " & ".join(DS_NAME[d] for d in D) + " \\\\", "\\midrule"]; recs = []
    for a in arms:
        if not any((d, a) in R.A.index for d in D): continue
        for i, (lab, key, f) in enumerate(quantities):
            vals = [R.v(d, a, key) for d in D]
            cells = [("0$^{*}$" if key == "maint" and a not in ("hlop", "hlop_adaptive", "hlop_gated", "replay", "replay_iso", "replay_gated", "ewc", "naive")
                      else ("--" if key == "maint" and a == "naive" else f(v))) for v in vals]
            L.append(f"{TEX_LABEL.get(a, a) if i == 0 else ''} & {lab} & " + " & ".join(cells) + " \\\\")
            recs += [dict(method=a, quantity=lab, dataset=d, value=v) for d, v in zip(D, vals)]
    if extra_rows:
        L.append("\\midrule")
        for lab, q, vals in extra_rows: L.append(f"{lab} & {q} & " + " & ".join(vals) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular*}"]
    _write(f"{out}/{name}", "\n".join(L) + "\n", pd.DataFrame(recs))


def si_costs(R, out):
    front = [("Shared front end", "energy over the stream (J)", [fmt_num(R.v(d, "rewire", "front") / 1000, 1) if d in FEAT else "0" for d in R.datasets])]
    _per_ds_block(R, out, "si_tab_costs", COST_ARMS, [("admission (mJ)", "adm", fmt_mj), ("maintenance overhead (mJ)", "maint", fmt_mj), ("re-admission (mJ)", "readm", fmt_mj),
                                                       ("standing memory (KB)", "stand", fmt_int), ("resident memory (KB)", "resid", fmt_int)], front)


def si_savings(R, out):
    _per_ds_block(R, out, "si_tab_savings", SAVINGS_ARMS, [("retention at re-admission (\\%)", "ret", fmt_num), ("steps to relearn a forgotten class", "Erel", fmt_num),
                                                            ("steps to learn a never-seen class", "Enov", fmt_num), ("savings $S$", "S", lambda x: fmt_num(x, 2)),
                                                            ("censored events per seed", "cens", fmt_num)])


def si_recovery(R, out):
    D = R.datasets; bud = None; recs = []
    L = []
    for (d, a), c in R.recovery.items(): bud = c["budgets"]; break
    if bud is None: return
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ll" + "r" * len(bud) + "l@{}}", "\\toprule", "Method & class / steps & " + " & ".join(str(b) for b in bud) + " & dataset \\\\", "\\midrule"]
    for d in D:
        for a in ("rewire", "hlop", "replay", "naive"):
            c = R.recovery.get((d, a))
            if c is None: continue
            L.append(f"{TEX_LABEL.get(a, a)} & forgotten (seen) & " + " & ".join(fmt_num(v) for v in c["seen"]) + f" & {DS_NAME[d]} \\\\")
            L.append(" & never seen & " + " & ".join(fmt_num(v) for v in c["novel"]) + " & \\\\")
            recs += [dict(dataset=d, method=a, steps=b, seen=s, novel=n) for b, s, n in zip(c["budgets"], c["seen"], c["novel"])]
    L += ["\\bottomrule", "\\end{tabular*}"]
    _write(f"{out}/si_tab_recovery", "\n".join(L) + "\n", pd.DataFrame(recs))


def si_ablation(R, out):
    D = R.datasets; arms = [a for a in ABL_LADDER + ABL_OTHER if any((d, a) in R.A.index for d in D)]
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "r" * len(D) + "r@{}}", "\\toprule", "Method & " + " & ".join(DS_NAME[d] for d in D) + " & Energy (mJ) \\\\", "\\midrule"]; recs = []
    for i, a in enumerate(arms):
        lab = "REWIRE (dual view, discriminant readout)" if a == "rewire" else TEX_LABEL.get(a, a)
        L.append(f"{lab} & " + " & ".join(_pm(R.v(d, a, "acc"), R.v(d, a, "acc_sd")) for d in D) + f" & {fmt_mj(R.mean_over(a, 'energy'))} \\\\")
        if a == "rewire": L.append("\\midrule")
        recs += [dict(method=a, dataset=d, acc=R.v(d, a, "acc"), sd=R.v(d, a, "acc_sd"), energy_mean=R.mean_over(a, "energy")) for d in D]
    L += ["\\bottomrule", "\\end{tabular*}"]
    _write(f"{out}/si_tab_ablation", "\n".join(L) + "\n", pd.DataFrame(recs))


def si_gate(R, out):
    D = R.datasets
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llrrrrr@{}}", "\\toprule", "Dataset & & ungated & g & g$_{\\rm M}$ & g$_{.95}$ & g$_{128}$ \\\\", "\\midrule"]; recs = []
    for d in D:
        for i, (lab, key, f) in enumerate((("ACC", "acc", fmt_num), ("energy per re-admission (mJ)", "readm", fmt_mj), ("lifetime energy (mJ)", "energy", fmt_mj),
                                           ("recurrences skipped (\\%)", "skipped", lambda x: fmt_num(100 * x, 0)))):
            vals = [R.v(d, a, key) for a in GATE_VARIANTS]
            L.append(f"{DS_NAME[d] if i == 0 else ''} & {lab} & " + " & ".join(f(v) for v in vals) + " \\\\")
            recs += [dict(dataset=d, quantity=lab, arm=a, value=v) for a, v in zip(GATE_VARIANTS, vals)]
    L += ["\\bottomrule", "\\end{tabular*}"]
    _write(f"{out}/si_tab_gate", "\n".join(L) + "\n", pd.DataFrame(recs))


def si_sweep(R, out):
    dens = R.density
    if not len(dens): return
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ll" + "r" * 6 + "@{}}", "\\toprule", "Dataset & Method & \\multicolumn{5}{c}{lifetime energy (mJ) at $R$ recurrences} & growth \\\\", "\\midrule"]
    for d in R.datasets:
        g = dens[dens.dataset == d]; rs = sorted(set(int(x) for x in g.n_recur))
        if not rs: continue
        L.append(f"\\multicolumn{{2}}{{l}}{{{DS_NAME[d]}: $R=$ {', '.join(map(str, rs))}}} & " + " & ".join("" for _ in range(6)) + " \\\\")
        for a in ("rewire", "rewire_gated", "ctrl_slda_input", "hlop", "replay", "naive"):
            ga = g[g.arm == a].sort_values("n_recur")
            if not len(ga): continue
            cells = [fmt_mj(float(ga[ga.n_recur == r].energy.iloc[0])) if (ga.n_recur == r).any() else "--" for r in rs] + [""] * (5 - len(rs))
            gr = R.growth[(R.growth.dataset == d) & (R.growth.arm == a)]
            L.append(f" & {TEX_LABEL.get(a, a)} & " + " & ".join(cells[:5]) + f" & {fmt_num(float(gr.growth.iloc[0]), 2) if len(gr) else '--'}$\\times$ \\\\")
    L += ["\\bottomrule", "\\end{tabular*}"]
    _write(f"{out}/si_tab_sweep", "\n".join(L) + "\n", dens)


def si_recognition(R, out):
    D = R.datasets
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}p{3.6cm}p{0.9cm}" + "r" * len(D) + "@{}}", "\\toprule", "Score & & " + " & ".join(DS_NAME[d] for d in D) + " \\\\", "\\midrule"]; recs = []
    names = dict(rewire="REWIRE, cosine familiarity", rewire_gated_maha="REWIRE, Mahalanobis familiarity")
    for a in RECOG_ARMS:
        if not any((d, a) in R.A.index for d in D): continue
        lab = names.get(a, TEX_LABEL.get(a, a) + (" (output confidence)" if a in ("hlop", "replay", "replay_iso", "ewc", "naive") else ""))
        L.append(f"{lab} & stream & " + " & ".join(f"{fmt_num(R.v(d, a, 'auroc'), 2)}\\,$\\pm$\\,{fmt_num(R.v(d, a, 'auroc_sd'), 2)}" for d in D) + " \\\\")
        L.append(f" & test & " + " & ".join(f"{fmt_num(R.v(d, a, 'auroc_test'), 2)}\\,$\\pm$\\,{fmt_num(R.v(d, a, 'auroc_test_sd'), 2)}" for d in D) + " \\\\")
        recs += [dict(score=lab, dataset=d, stream=R.v(d, a, "auroc"), test=R.v(d, a, "auroc_test")) for d in D]
    L += ["\\midrule", "REWIRE, competence & stream & " + " & ".join(fmt_num(R.v(d, "rewire", "auroc_comp"), 2) for d in D) + " \\\\", "\\bottomrule", "\\end{tabular*}"]
    _write(f"{out}/si_tab_recognition", "\n".join(L) + "\n", pd.DataFrame(recs))


def si_hypotheses(R, out):
    h = R.hyp; D = R.datasets
    rr = ">{\\raggedright\\arraybackslash}"
    L = ["\\begin{tabular}{" + rr + "p{0.2\\textwidth}" + (rr + "p{0.115\\textwidth}") * len(D) + "}", "\\toprule", "Hypothesis & " + " & ".join(DS_NAME[d] for d in D) + " \\\\", "\\midrule"]
    def row(name, f): L.append(name + " & " + " & ".join(f(h[h.dataset == d].iloc[0]) for d in D) + " \\\\")
    row("H1: within 2\\,points of the best gradient-based baseline at $\\le50\\%$ of the cheapest one's lifetime energy",
        lambda r: f"{'pass' if r.h1 else 'fail'} ({fmt_signed(r.h1_dacc)}\\,pts, {fmt_ratio(r.h1_ratio)}$\\times$)")
    row("H2: re-admission at $\\le50\\%$ of the cheapest gradient-based baseline's", lambda r: f"{'pass' if r.h2 else 'fail'} ({fmt_ratio(r.h2_ratio)}$\\times$)")
    ha = getattr(R, "hyp_all", None)
    if ha is not None and len(ha):                      # the same two bars against the pre-registered set, failed baselines included
        from .style import label
        inc = " and ".join(label(a) for a in R.fid_fail)
        def row_all(name, f): L.append(name + " & " + " & ".join(f(ha[ha.dataset == d].iloc[0]) for d in D) + " \\\\")
        row_all(f"H1 with {inc} in the reference set (pre-registered set)", lambda r: f"{'pass' if r.h1 else 'fail'} ({fmt_signed(r.h1_dacc)}\\,pts, {fmt_ratio(r.h1_ratio)}$\\times$)")
        row_all(f"H2 with {inc} in the reference set (pre-registered set)", lambda r: f"{'pass' if r.h2 else 'fail'} ({fmt_ratio(r.h2_ratio)}$\\times$)")
    row("H3: retention (BWT; REWIRE vs HLOP, rehearsal)", lambda r: f"{fmt_num(r.h3_rewire)} vs {fmt_num(r.h3_hlop)}, {fmt_num(r.h3_reh)}")
    row("H4: stream recognition AUROC $\\ge0.90$", lambda r: f"{'pass' if r.h4 else 'fail'} ({fmt_num(r.h4_auroc, 2)})")
    row("H5: ablation ladder (LIF vs ReLU; input view; discriminant vs ridge; points)", lambda r: f"{fmt_signed(r.h5_relu)}; {fmt_signed(r.h5_view)}; {fmt_signed(r.h5_lda)}")
    lab6 = {"rewire_gated": "g", "rewire_gated_maha": "g$_{\\rm M}$", "rewire_gated_t095": "g$_{.95}$", "rewire_gated_b128": "g$_{128}$"}
    row("H6: a gate setting saves $\\ge40\\%$ energy at $\\le2$\\,points", lambda r: f"{'pass' if r.h6 else 'fail'} ({lab6.get(r.h6_arm, r.h6_arm)}: {fmt_num(r.h6_saved, 0)}\\% at {fmt_signed(r.h6_dacc)})")
    L += ["\\bottomrule", "\\end{tabular}"]
    _write(f"{out}/si_tab_hypotheses", "\n".join(L) + "\n", h)


def si_full(R, out):
    for d in R.datasets:
        L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrrr@{}}", "\\toprule", "Method & ACC & BWT & Lifetime (mJ) & Re-adm. (mJ) & AUROC & Stored (KB) \\\\", "\\midrule"]; recs = []
        for a in [a for a in ORDER if (d, a) in R.A.index]:
            L.append(f"{TEX_LABEL.get(a, a)} & {_pm(R.v(d, a, 'acc'), R.v(d, a, 'acc_sd'))} & {fmt_num(R.v(d, a, 'bwt'))} & {fmt_mj(R.v(d, a, 'energy'))} & {fmt_mj(R.v(d, a, 'readm'))} & "
                     f"{fmt_num(R.v(d, a, 'auroc'), 2)} & {fmt_int(R.v(d, a, 'stand'))} \\\\")
            recs.append(dict(method=a, acc=R.v(d, a, "acc"), acc_sd=R.v(d, a, "acc_sd"), bwt=R.v(d, a, "bwt"), energy=R.v(d, a, "energy"), readm=R.v(d, a, "readm"), auroc=R.v(d, a, "auroc"), stored_KB=R.v(d, a, "stand")))
        L += ["\\bottomrule", "\\end{tabular*}"]
        _write(f"{out}/si_tab_full_{d}", "\n".join(L) + "\n", pd.DataFrame(recs))


def si_shift(R, out):
    s = R.shift
    if not len(s): return
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llrrrrrr@{}}", "\\toprule",
         "Dataset & Method & ACC & ACC shifted & $\\Delta$ unshifted & first written (\\%) & later written (\\%) & Energy (mJ) \\\\", "\\midrule"]
    for d in R.datasets:
        g = s[s.dataset == d]
        for i, a in enumerate([a for a in ("rewire", "rewire_gated", "rewire_gated_maha", "ctrl_slda_input", "hlop", "replay", "naive") if (g.arm == a).any()]):
            r = g[g.arm == a].iloc[0]
            fw = fmt_num(r.first_written, 0) if a.startswith("rewire_gated") else "--"; lw = fmt_num(r.later_written, 0) if a.startswith("rewire_gated") else "--"
            L.append(f"{DS_NAME[d] if i == 0 else ''} & {TEX_LABEL.get(a, a)} & {_pm(r.acc, r.acc_sd)} & {fmt_num(r.acc_shifted)} & {fmt_signed(r.dacc_vs_unshifted)} & {fw} & {lw} & {fmt_mj(r.energy)} \\\\")
    L += ["\\bottomrule", "\\end{tabular*}"]
    _write(f"{out}/si_tab_shift", "\n".join(L) + "\n", s)


def si_eps(R, out):
    s, dev, sel = R.eps_sens, R.eps_dev, R.eps_selected
    if not len(s) and not len(dev): return
    eps = sorted(set(s.eps)) if len(s) else sorted(set(dev.eps))
    L = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "r" * len(eps) + "@{}}", "\\toprule", "Stream & " + " & ".join(f"$\\varepsilon={e:g}$" for e in eps) + " \\\\", "\\midrule"]
    if len(dev): L.append("Development & " + " & ".join(fmt_num(float(dev[dev.eps == e].acc.iloc[0])) if (dev.eps == e).any() else "--" for e in eps) + " \\\\"); L.append("\\midrule")
    for d in R.datasets:
        g = s[s.dataset == d] if len(s) else s
        if not len(g): continue
        L.append(f"{DS_NAME[d]} & " + " & ".join(_pm(float(g[g.eps == e].acc.iloc[0]), float(g[g.eps == e].acc_sd.iloc[0])) if (g.eps == e).any() else "--" for e in eps) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular*}"]
    _write(f"{out}/si_tab_eps", "\n".join(L) + "\n", s if len(s) else dev)


ALL = [table1_main, ed_native, ed_akida, si_hardware, si_protocol, si_paired, si_costs, si_savings, si_recovery, si_ablation, si_gate, si_sweep, si_recognition,
       si_hypotheses, si_full, si_shift, si_eps]
