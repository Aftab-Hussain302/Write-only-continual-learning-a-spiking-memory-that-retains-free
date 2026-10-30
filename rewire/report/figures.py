"""The paper's results figures, drawn from the records (never re-run). Each panel is a function drawing into given axes; layouts
assemble the panels into the paper's figure files: the NCS numbering (main Figs 3-5 and the Extended Data figures) and, on request, a
per-analysis layout (one file per analysis, fig06-fig14) for inspection. Every figure is written as EPS and PNG (300 dpi) plus a PDF copy for
the integrity scan; its source data go to an Excel workbook (one sheet per panel)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter, NullLocator, MaxNLocator, FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .style import (W1, W2, MM, OKABE, SHADE, TEXT, label, color, marker, filled, DS_NAME, DS_INPUT, DS_TWO_LINE, RAW, FEAT, panel_letter, save, SHORT)
from .compute import ABL_LADDER, NAN

PARETO_ARMS = ["rewire", "rewire_gated", "ctrl_slda_input", "ranpac", "hlop", "replay", "replay_iso", "naive"]
LINE_ARMS = ["rewire", "rewire_gated", "ctrl_slda_input", "hlop", "replay", "naive"]
RET_ARMS = ["rewire", "ctrl_slda_input", "hlop", "replay", "naive"]
SAV_ARMS = ["rewire", "ctrl_slda_input", "ranpac", "hlop", "replay", "naive"]
REC_ARMS = ["rewire", "hlop", "replay", "naive"]
COST_ARMS = ["rewire", "rewire_gated", "ctrl_slda_input", "ranpac", "hlop", "replay", "ewc", "naive"]
SHIFT_ARMS = ["rewire", "rewire_gated", "rewire_gated_maha", "ctrl_slda_input", "hlop", "replay", "naive"]
LETTERS = "abcdefghij"
ABL_ROWS = [("abl_sketch_ridge", "sketch only, ridge readout"), ("abl_sketch_lda", "sketch only, discriminant readout"),
            ("abl_dualview_ridge", "dual view, ridge readout"), ("abl_width128", "sketch 128"), ("abl_width256", "sketch 256"),
            ("ctrl_relu", "ReLU layer instead of LIF"), ("ctrl_slda_input", "SLDA on the inputs"), ("ranpac", "RanPAC-style (ReLU, ridge)")]


def _tint(hexcol, f=0.55):
    """An opaque lighter tint of a colour (EPS holds no transparency)."""
    h = hexcol.lstrip("#"); r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(int(c + (255 - c) * f) for c in (r, g, b))


def _ds_title(d): return f"{DS_NAME.get(d, d)} ({DS_INPUT.get(d, '')})"


def log_ticks(ax, lo, hi, axis="x"):
    """Readable ticks on a log axis: 1-2-5 or 1-3 steps, plain numbers (k for thousands), no minor labels."""
    lo, hi = max(lo, 1e-9), max(hi, lo * 1.01); a = ax.xaxis if axis == "x" else ax.yaxis
    t = []
    for mult in ((1,), (1, 3), (1, 2, 5)):
        t = [m * 10 ** e for e in range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1) for m in mult if lo <= m * 10 ** e <= hi]
        if len(t) >= 2: break
    if len(t) > 6: t = t[::2]
    if not t: return
    a.set_major_locator(FixedLocator(t)); a.set_major_formatter(FixedFormatter([f"{v:,.0f}" if v >= 1000 else f"{v:g}" for v in t]))
    a.set_minor_formatter(NullFormatter())


def _point(ax, x, y, a, xerr=None, yerr=None, ms=None):
    op = not filled(a); ms = ms or (8.5 if a == "rewire" else 5.5 if op else 4.8)
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=marker(a), color=color(a), ms=ms, mfc="white" if op else color(a),
                mec=color(a) if op else "black", mew=0.9 if op else 0.35, elinewidth=0.5, capsize=1.2, label=label(a), zorder=8 if a == "rewire" else (4 if op else 6))


# ------------------------------------------------------------------------------------------------------------------------- panels
def p_pareto(ax, R, d, data):
    """Accuracy against lifetime energy on one dataset (mean +- s.e.m. over seeds); fine-tuning at the bottom edge with its value."""
    pts = [(a, R.v(d, a, "energy"), R.v(d, a, "acc"), R.v(d, a, "energy_sem"), R.v(d, a, "acc_sem")) for a in PARETO_ARMS if np.isfinite(R.v(d, a, "acc"))]
    ys = [p[2] for p in pts if p[0] != "naive"]; lo, hi = min(ys), max(ys); pad = max(1.0, 0.12 * (hi - lo))
    ymin = lo - pad; ytop = min(hi + pad, 100.0 + 0.25 * pad); ax.set_ylim(ymin, ytop)
    for a, x, y, xe, ye in pts:
        if a == "naive" and y < ymin:
            _point(ax, x, ymin + 0.06 * (hi + pad - ymin), a, ms=4.5)
            ax.annotate(f"{y:.0f}%", (x, ymin + 0.06 * (hi + pad - ymin)), xytext=(4, 0), textcoords="offset points", va="center", fontsize=5.5, color=TEXT)
        else: _point(ax, x, y, a, xerr=xe, yerr=ye)
        data.append(dict(dataset=DS_NAME.get(d, d), method=label(a), lifetime_energy_mJ=x, lifetime_energy_sem=xe, accuracy=y, accuracy_sem=ye, n_seeds=R.v(d, a, "acc_n")))
    xs = [p[1] for p in pts]; ax.set_xscale("log"); ax.set_xlim(min(xs) / 1.6, max(xs) * 1.6); log_ticks(ax, min(xs) / 1.6, max(xs) * 1.6)
    ax.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10])); ax.set_title(_ds_title(d), fontsize=7)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}" if v <= 100 + 1e-9 else ""))


def p_costs(ax, R, data):
    """The three costs per chunk, mean over datasets and seeds: admission (light), maintenance overhead (hatched; zero by construction
    for the write-only methods) and re-admission (dark), logarithmic axis; the share of recurrences the gate skipped is printed."""
    x = np.arange(len(COST_ARMS)); w = 0.26
    for i, a in enumerate(COST_ARMS):
        adm, mt, rd = R.mean_over(a, "adm"), R.mean_over(a, "maint"), R.mean_over(a, "readm")
        ax.bar(i - w, adm, w, color=_tint(color(a), 0.55), edgecolor=_tint(color(a), 0.55), linewidth=0.4)
        if a in ("hlop", "replay", "ewc") and np.isfinite(mt) and mt > 0:
            ax.bar(i, mt, w, color="white", edgecolor=color(a), hatch="//////", linewidth=0.5)
        ax.bar(i + w, rd, w, color=color(a), edgecolor=color(a), linewidth=0.4)
        if a == "rewire_gated":
            sk = R.mean_over(a, "skipped"); ax.text(i + w, rd * 1.35, f"{100 * sk:.0f}%\nskipped", ha="center", va="bottom", fontsize=5.5, color=TEXT)
        data.append(dict(method=label(a), admission_mJ=adm, maintenance_mJ=(0.0 if a not in ("hlop", "replay", "ewc", "naive") else mt), readmission_mJ=rd))
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([SHORT.get(a, label(a)) for a in COST_ARMS], fontsize=5.5)
    ax.set_ylabel("energy per chunk (mJ, log)"); ax.grid(axis="x", visible=False)
    ys = [R.mean_over(a, c) for a in COST_ARMS for c in ("adm", "readm")]; ys = [y for y in ys if np.isfinite(y) and y > 0]
    if ys: ax.set_ylim(min(ys) / 2.2, max(ys) * 3.0); log_ticks(ax, min(ys) / 2.2, max(ys) * 3.0, axis="y")
    ax.legend(handles=[Patch(facecolor=_tint(OKABE["blue"], 0.55), edgecolor="none", label="admission"),
                       Patch(facecolor="white", edgecolor=OKABE["blue"], hatch="//////", label="maintenance overhead"),
                       Patch(facecolor=OKABE["blue"], edgecolor="none", label="re-admission")], loc="upper left", ncol=3, fontsize=5.5, handlelength=1.2)


def p_retention_curve(ax, R, d, data, arms=RET_ARMS, ylabel=True):
    for a in arms:
        c = R.ret_curves.get((d, a))
        if c is None: continue
        t = np.arange(1, len(c) + 1)
        ax.plot(t, c, color=color(a), lw=1.4 if a == "rewire" else 0.8, marker=marker(a), ms=5 if a == "rewire" else 3.2, markevery=max(1, len(t) // 6),
                mfc=color(a) if filled(a) else "white", mec="black" if filled(a) else color(a), mew=0.3 if filled(a) else 0.8, label=label(a), zorder=8 if a == "rewire" else 4)
        for i, v in enumerate(c): data.append(dict(dataset=DS_NAME.get(d, d), method=label(a), recurrence=i + 1, retention_pct=v))
    ax.axhline(50, color=OKABE["grey"], lw=0.5, ls=(0, (2, 2))); ax.set_ylim(45, 101)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=3))     # recurrences are counted, not measured
    ax.set_title(DS_NAME.get(d, d), fontsize=7); ax.set_xlabel("recurrence index")
    if ylabel: ax.set_ylabel("balanced accuracy on the\nreturning class (%)")


def p_savings_S(ax, R, data):
    """Ebbinghaus's savings S = 1 - E_relearn / E_new per dataset (mean of the per-seed ratios)."""
    D = R.datasets; x = np.arange(len(D)); off = np.linspace(-0.3, 0.3, len(SAV_ARMS)); floor = -1.0
    for j, a in enumerate(SAV_ARMS):
        ys = [R.v(d, a, "S") for d in D]
        op = not filled(a)
        yp = [max(y, floor - 0.05) if np.isfinite(y) else np.nan for y in ys]      # below -1: drawn at the lower edge, value printed
        ax.plot(x + off[j], yp, ls="none", marker=marker(a), ms=6.5 if a == "rewire" else 4.2, color=color(a), mfc="white" if op else color(a),
                mec=color(a) if op else "black", mew=0.8 if op else 0.3, label=label(a), zorder=8 if a == "rewire" else 4)
        for xi, y in zip(x + off[j], ys):
            if np.isfinite(y) and y < floor:
                ax.annotate(f"{y:.1f}".replace("-", "\u2212"), (xi, floor - 0.05), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=4.5, color=TEXT)
        for d, y in zip(D, ys): data.append(dict(dataset=DS_NAME.get(d, d), method=label(a), savings_S=y, steps_relearn=R.v(d, a, "Erel"), steps_new=R.v(d, a, "Enov"),
                                                 censored_per_seed=R.v(d, a, "cens")))
    ax.axhline(0, color=OKABE["grey"], lw=0.5); ax.axhline(1, color=OKABE["grey"], lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks(x); ax.set_xticklabels([DS_TWO_LINE.get(d, d) for d in D], fontsize=5.5); ax.grid(axis="x", visible=False)
    ax.set_ylabel("savings on relearning, S"); vals = [R.v(d, a, "S") for d in D for a in SAV_ARMS]
    lo = np.nanmin(vals) if np.any(np.isfinite(vals)) else np.nan
    ax.set_ylim(max(min(-0.1, lo - 0.1), floor - 0.15) if np.isfinite(lo) else -0.1, 1.12)


def p_ablation(ax, R, data):
    """Accuracy of each ablation and control relative to REWIRE: mean over the inputs without a front end (filled) and over the
    ResNet-feature datasets (open), with the range over the datasets (grey)."""
    D = R.datasets; raw = [d for d in D if d in RAW]; feat = [d for d in D if d in FEAT]
    rows = [(a, t) for a, t in ABL_ROWS if any(np.isfinite(R.v(d, a, "acc")) for d in D)]
    for i, (a, t) in enumerate(rows):
        y = len(rows) - 1 - i; diffs = {d: R.v(d, a, "acc") - R.v(d, "rewire", "acc") for d in D}
        vals = [v for v in diffs.values() if np.isfinite(v)]
        if not vals: continue
        ax.plot([min(vals), max(vals)], [y, y], color="#C8C8C4", lw=2.2, solid_capstyle="butt", zorder=1)
        c = OKABE["blue"] if a.startswith("abl") else OKABE["purple"]
        mr = np.nanmean([diffs[d] for d in raw]) if raw else NAN; mf = np.nanmean([diffs[d] for d in feat]) if feat else NAN
        ax.plot(mr, y, "o", color=c, mec="black", mew=0.3, ms=4.5, zorder=3); ax.plot(mf, y, "o", color=c, mfc="white", mew=1.0, ms=4.5, zorder=3)
        data.append(dict(method=t, mean_raw_inputs=mr, mean_resnet_features=mf, min=min(vals), max=max(vals), **{f"diff_{DS_NAME[d]}": diffs[d] for d in D}))
    ax.axvline(0, color="black", lw=0.6); ax.text(0, len(rows) - 0.35, " REWIRE", ha="left", va="bottom", fontsize=5.5, color=TEXT)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([t for _, t in rows][::-1], fontsize=6); ax.set_ylim(-1.9, len(rows) + 0.1)
    ax.set_xlabel("accuracy relative to REWIRE (points)"); ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=OKABE["blue"], mec="black", mew=0.3, ms=4.5, label="mean over pixel, event and spike inputs"),
                       Line2D([], [], marker="o", ls="none", color=OKABE["blue"], mfc="white", mew=1.0, ms=4.5, label="mean over ResNet features"),
                       Line2D([], [], color="#C8C8C4", lw=2.2, label="range over the datasets")], loc="lower left", fontsize=5.5, ncol=3,
              frameon=True, facecolor="white", edgecolor="white", framealpha=1.0)      # opaque, so that the REWIRE line does not cross the text


def p_gate_curve(ax, R, data):
    """The gate as an operating curve: lifetime energy saved against the accuracy change, for theta = 0.75, 0.85, 0.95 (light to
    dark), per dataset, against ungated REWIRE on the same seeds."""
    g = R.gate[R.gate.arm == "rewire_gated"] if len(R.gate) else None
    shades = {0.75: _tint(OKABE["green"], 0.6), 0.85: OKABE["green"], 0.95: "#00563F"}
    if g is None or not len(g): ax.text(0.5, 0.5, "gate sweep not in the records", transform=ax.transAxes, ha="center"); return
    labels = []
    for d in R.datasets:
        gd = g[g.dataset == d].sort_values("theta")
        if not len(gd): continue
        ax.plot(gd.dacc, gd.saved, color="#9A9A96", lw=0.6, zorder=1)
        for r in gd.itertuples():
            ax.plot(r.dacc, r.saved, "o", color=shades.get(round(r.theta, 2), OKABE["green"]), mec="black", mew=0.3, ms=4, zorder=3)
            data.append(dict(dataset=DS_NAME.get(d, d), theta=r.theta, accuracy_change_points=r.dacc, energy_saved_pct=r.saved, recurrences_skipped_pct=r.skipped))
        top = gd.iloc[0]; labels.append((top.dacc, top.saved, DS_NAME.get(d, d)))
    ax.axvline(0, color=OKABE["grey"], lw=0.5); ax.axhline(0, color=OKABE["grey"], lw=0.5)
    if labels: _annotate_apart(ax, labels, ground=True)
    ax.set_xlabel("accuracy change against ungated REWIRE (points)"); ax.set_ylabel("lifetime energy saved (%)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=c, mec="black", mew=0.3, ms=4, label=f"$\\theta$ = {t:.2f}" + (" (default)" if t == 0.85 else "")) for t, c in shades.items()],
              loc="best", fontsize=5.5)


def p_density(ax, R, data, datasets=None, arms=LINE_ARMS):
    """Lifetime energy against the number of recurrence chunks, relative to the shortest stream, mean over the 8- and 16-class
    streams (range shaded by lines): the slope is the cost of a re-admission."""
    dens = R.density; dss = datasets or [d for d in R.datasets if d not in ("cifar100", "tinyimagenet")]
    for a in arms:
        rel = {}
        for d in dss:
            g = dens[(dens.dataset == d) & (dens.arm == a)].sort_values("n_recur")
            if len(g) < 2: continue
            e0 = float(g.energy.iloc[0])
            for r, e in zip(g.n_recur, g.energy): rel.setdefault(int(r), []).append(float(e) / e0)
        if not rel: continue
        rs = sorted(rel); m = [np.mean(rel[r]) for r in rs]
        ax.plot(rs, m, color=color(a), lw=1.3 if a == "rewire" else 0.8, marker=marker(a), ms=5 if a == "rewire" else 3.2, mfc=color(a) if filled(a) else "white",
                mec="black" if filled(a) else color(a), mew=0.3 if filled(a) else 0.8, label=label(a), zorder=8 if a == "rewire" else 4)
        for r in rs: data.append(dict(method=label(a), n_recurrence_chunks=r, energy_relative_to_shortest=np.mean(rel[r]), min=min(rel[r]), max=max(rel[r]), n_datasets=len(rel[r])))
    ax.set_xscale("log", base=2); rs_all = sorted(set(int(x) for x in dens.n_recur)) if len(dens) else [8, 128]
    ax.xaxis.set_major_locator(FixedLocator(rs_all)); ax.xaxis.set_major_formatter(FixedFormatter([str(r) for r in rs_all])); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("recurrence chunks in the stream, R"); ax.set_ylabel("lifetime energy relative to the\nshortest stream")


def p_recognition(ax, R, data):
    """Recognition AUROC per dataset: REWIRE's cosine and Mahalanobis familiarity on the stream, the cosine familiarity at test time
    (seen against never-seen classes, hatched) and HLOP's output confidence on the stream."""
    D = R.datasets; x = np.arange(len(D)); w = 0.19
    series = [("rewire", "auroc", OKABE["blue"], None, "REWIRE, stream (cosine)"), ("rewire_gated_maha", "auroc", _tint(OKABE["blue"], 0.45), None, "REWIRE, stream (Mahalanobis)"),
              ("rewire", "auroc_test", "white", OKABE["blue"], "REWIRE, test (seen against never seen)"), ("hlop", "auroc", OKABE["vermillion"], None, "HLOP, stream (confidence)")]
    for j, (a, col, fc, hc, lab) in enumerate(series):
        ys = [R.v(d, a, col) for d in D]; es = [R.v(d, a, col + "_sd") for d in D]
        ax.bar(x + (j - 1.5) * w, ys, w, color=fc, edgecolor=hc or fc, hatch="//////" if hc else None, linewidth=0.5, label=lab)
        for d, y, e in zip(D, ys, es): data.append(dict(dataset=DS_NAME.get(d, d), score=lab, auroc=y, sd=e))
    ax.axhline(0.5, color=OKABE["grey"], lw=0.5, ls=(0, (2, 2)))
    ax.set_ylim(0.3, 1.04); ax.set_xticks(x); ax.set_xticklabels([DS_TWO_LINE.get(d, d) for d in D], fontsize=5.5); ax.grid(axis="x", visible=False)
    ax.set_ylabel("recognition AUROC"); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=5.2, ncol=4, handlelength=1.0)


def p_shift_acc(ax, R, data, arms=SHIFT_ARMS):
    """Domain-shifted recurrences: final accuracy on the classes that returned from their shifted domain, per dataset."""
    sh = R.shift; D = [d for d in R.datasets if len(sh) and (sh.dataset == d).any()]
    if not D: ax.text(0.5, 0.5, "domain-shift grid not in the records", transform=ax.transAxes, ha="center", fontsize=6); return
    x = np.arange(len(D)); off = np.linspace(-0.32, 0.32, len(arms))
    for j, a in enumerate(arms):
        ys = [float(sh[(sh.dataset == d) & (sh.arm == a)].acc_shifted.iloc[0]) if ((sh.dataset == d) & (sh.arm == a)).any() else NAN for d in D]
        op = not filled(a)
        ax.plot(x + off[j], ys, ls="none", marker=marker(a), ms=6.5 if a == "rewire" else 4.0, color=color(a), mfc="white" if op else color(a),
                mec=color(a) if op else "black", mew=0.8 if op else 0.3, label=label(a), zorder=8 if a == "rewire" else 4)
        for d, y in zip(D, ys): data.append(dict(dataset=DS_NAME.get(d, d), method=label(a), accuracy_on_shifted_classes=y))
    ax.set_xticks(x); ax.set_xticklabels([DS_TWO_LINE.get(d, d) for d in D], fontsize=5.5); ax.grid(axis="x", visible=False)
    ax.set_ylabel("accuracy on the classes\nthat returned shifted (%)")


def p_shift_written(ax, R, data):
    """The gate under domain shift: share of first shifted returns and of later returns that REWIRE-gated wrote (cosine and
    Mahalanobis familiarity)."""
    sh = R.shift; D = [d for d in R.datasets if len(sh) and (sh.dataset == d).any()]
    if not D: return
    x = np.arange(len(D)); w = 0.2
    series = [("rewire_gated", "first_written", OKABE["green"], None, "first shifted return (cosine)"), ("rewire_gated", "later_written", "white", OKABE["green"], "later returns (cosine)"),
              ("rewire_gated_maha", "first_written", "#00563F", None, "first shifted return (Mahalanobis)"), ("rewire_gated_maha", "later_written", "white", "#00563F", "later returns (Mahalanobis)")]
    for j, (a, col, fc, hc, lab) in enumerate(series):
        ys = [float(sh[(sh.dataset == d) & (sh.arm == a)][col].iloc[0]) if ((sh.dataset == d) & (sh.arm == a)).any() else NAN for d in D]
        ax.bar(x + (j - 1.5) * w, ys, w, color=fc, edgecolor=hc or fc, hatch="//////" if hc else None, linewidth=0.5, label=lab)
        for xi, y in zip(x + (j - 1.5) * w, ys):
            if np.isfinite(y) and y < 0.5: ax.text(xi, 1.5, "0", ha="center", va="bottom", fontsize=4.8, color=TEXT)
        for d, y in zip(D, ys): data.append(dict(dataset=DS_NAME.get(d, d), method=label(a), returns=lab, written_pct=y))
    ax.set_ylim(0, 112); ax.set_xticks(x); ax.set_xticklabels([DS_TWO_LINE.get(d, d) for d in D], fontsize=5.5); ax.grid(axis="x", visible=False)
    ax.set_ylabel("recurrences written (%)"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=5.2, ncol=2, handlelength=1.0)


def p_stream(axa, axe, R, d, data):
    """One stream chunk by chunk: accuracy on the classes seen so far (admission phase shaded) and cumulative energy (log)."""
    c0 = R.streams.get((d, "rewire"));
    if c0 is None: return
    n_intro = c0["n_intro"]
    for ax in (axa, axe):
        ax.axvspan(0.5, n_intro + 0.5, color=SHADE, zorder=0, lw=0)
    for a in LINE_ARMS:
        c = R.streams.get((d, a))
        if c is None: continue
        t = np.arange(1, len(c["acc"]) + 1); kw = dict(color=color(a), lw=1.3 if a == "rewire" else 0.8, marker=marker(a), ms=4.5 if a == "rewire" else 3,
                                                       markevery=max(1, len(t) // 6), mfc=color(a) if filled(a) else "white", mec="black" if filled(a) else color(a),
                                                       mew=0.3 if filled(a) else 0.8, label=label(a), zorder=8 if a == "rewire" else 4)
        axa.plot(t, c["acc"], **kw); axe.plot(t, c["energy"], **kw)
        for i in range(len(t)): data.append(dict(dataset=DS_NAME.get(d, d), method=label(a), chunk=int(t[i]), accuracy_seen_pct=c["acc"][i], cumulative_energy_mJ=c["energy"][i]))
    axe.set_yscale("log"); ys = [v for a in LINE_ARMS if (d, a) in R.streams for v in R.streams[(d, a)]["energy"] if v > 0]
    if ys: axe.set_ylim(min(ys) / 1.5, max(ys) * 1.5); log_ticks(axe, min(ys) / 1.5, max(ys) * 1.5, axis="y")
    axa.set_ylabel("accuracy on the\nclasses seen (%)"); axe.set_ylabel("cumulative energy\n(mJ, log)")
    for ax in (axa, axe):
        ax.set_xlabel("stream chunk"); ax.set_xlim(0.5, len(c0["acc"]) + 0.5); ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axa.set_title(_ds_title(d), fontsize=7)


def p_steps(ax, R, d, data):
    """Few-shot steps to criterion: a forgotten seen class (filled) and a never-seen class (hatched), S printed, 40-step cap."""
    x = np.arange(len(SAV_ARMS)); w = 0.38
    for i, a in enumerate(SAV_ARMS):
        er, en, s = R.v(d, a, "Erel"), R.v(d, a, "Enov"), R.v(d, a, "S")
        ax.bar(i - w / 2, er, w, color=color(a), edgecolor=color(a), linewidth=0.4)
        ax.bar(i + w / 2, en, w, color="white", edgecolor=color(a), hatch="//////", linewidth=0.5)
        if np.isfinite(s): ax.text(i, max(er if np.isfinite(er) else 0, en if np.isfinite(en) else 0) + 1.2, f"{s:.2f}".replace("-", "\u2212"), ha="center", va="bottom",
                                   fontsize=4.8, color=TEXT)
        data.append(dict(dataset=DS_NAME.get(d, d), method=label(a), steps_relearn=er, steps_new=en, savings_S=s))
    ax.set_ylim(0, 46); ax.set_xticks(x); ax.set_xticklabels([SHORT.get(a, label(a)).replace("Fine-\ntuning", "FT") for a in SAV_ARMS], fontsize=5); ax.grid(axis="x", visible=False)
    ax.set_title(DS_NAME.get(d, d), fontsize=7)


def p_recovery(ax, R, d, data):
    for a in REC_ARMS:
        c = R.recovery.get((d, a))
        if c is None: continue
        ax.plot(c["budgets"], c["seen"], color=color(a), lw=1.0, marker="o", ms=3, label=f"{label(a)}, forgotten class")
        ax.plot(c["budgets"], c["novel"], color=color(a), lw=0.8, ls=(0, (3, 1.5)), marker="s", ms=2.8, label=f"{label(a)}, never-seen class")
        for b, s, n in zip(c["budgets"], c["seen"], c["novel"]): data.append(dict(dataset=DS_NAME.get(d, d), method=label(a), steps=b, forgotten_class_pct=s, never_seen_class_pct=n))
    ax.set_ylim(45, 101); ax.set_title(DS_NAME.get(d, d), fontsize=7); ax.set_xlabel("few-shot steps"); ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def p_density_ds(axe, axa, R, d, data):
    dens = R.density
    for a in LINE_ARMS:
        g = dens[(dens.dataset == d) & (dens.arm == a)].sort_values("n_recur")
        if not len(g): continue
        kw = dict(color=color(a), lw=1.3 if a == "rewire" else 0.8, marker=marker(a), ms=4.5 if a == "rewire" else 3, mfc=color(a) if filled(a) else "white",
                  mec="black" if filled(a) else color(a), mew=0.3 if filled(a) else 0.8, label=label(a), zorder=8 if a == "rewire" else 4)
        axe.errorbar(g.n_recur, g.energy, yerr=g.energy_sem, **kw); axa.errorbar(g.n_recur, g.acc, yerr=g.acc_sem, **kw)
        for r in g.itertuples(): data.append(dict(dataset=DS_NAME.get(d, d), method=label(a), n_recurrence_chunks=int(r.n_recur), lifetime_energy_mJ=r.energy, energy_sem=r.energy_sem, accuracy=r.acc, accuracy_sem=r.acc_sem))
    rs = sorted(set(int(x) for x in dens[dens.dataset == d].n_recur))
    lab = [r for r in rs if r & (r - 1) == 0] or rs                      # powers of two get a label; the others a tick only
    for ax in (axe, axa):
        ax.set_xscale("log", base=2); ax.xaxis.set_major_locator(FixedLocator(lab)); ax.xaxis.set_major_formatter(FixedFormatter([str(r) for r in lab]))
        ax.xaxis.set_minor_locator(FixedLocator([r for r in rs if r not in lab])); ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xlabel("recurrence chunks, R")
    axe.set_yscale("log"); ys = [v for v in dens[dens.dataset == d].energy if v > 0]
    if ys: axe.set_ylim(min(ys) / 1.5, max(ys) * 1.5); log_ticks(axe, min(ys) / 1.5, max(ys) * 1.5, axis="y")
    axe.set_ylabel("lifetime energy\n(mJ, log)"); axa.set_ylabel("final accuracy (%)"); axe.set_title(_ds_title(d), fontsize=7)


def p_eps(ax, R, data):
    s = R.eps_sens
    if not len(s): ax.text(0.5, 0.5, "shrinkage sweep not in the records", transform=ax.transAxes, ha="center"); return
    # one method (REWIRE), so one colour: the datasets differ by marker and are labelled at the right end of their line
    marks = dict(mnist="o", nmnist="s", shd="^", ssc="v", cifar10="D", cifar100="P", tinyimagenet="X"); ends = []; grid = set()
    for i, d in enumerate(R.datasets):
        g = s[s.dataset == d].sort_values("eps")
        if not len(g): continue
        ax.errorbar(g.eps, g.acc, yerr=g.acc_sd, marker=marks.get(d, "o"), ms=3.2, lw=0.8, color=OKABE["blue"], mfc="white", mec=OKABE["blue"], mew=0.8,
                    elinewidth=0.5, capsize=1.2)
        ends.append((float(g.eps.iloc[-1]), float(g.acc.iloc[-1]), DS_NAME.get(d, d))); grid |= set(float(e) for e in g.eps)
        for r in g.itertuples(): data.append(dict(dataset=DS_NAME.get(d, d), shrinkage=r.eps, accuracy=r.acc, sd=r.acc_sd))
    if R.eps_selected and R.eps_selected.get("selected"): ax.axvline(float(R.eps_selected["selected"]), color=OKABE["grey"], lw=0.6, ls=(0, (2, 2)))
    ax.set_xscale("log"); ax.set_xlabel("LDA shrinkage, $\\varepsilon$"); ax.set_ylabel("final accuracy (%)")
    if grid:
        gl = sorted(grid); ax.xaxis.set_major_locator(FixedLocator(gl)); ax.xaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in gl]))
        ax.xaxis.set_minor_locator(NullLocator()); ax.set_xlim(gl[0] / 1.4, gl[-1] * 1.4)
    if ends: _annotate_apart(ax, ends, fontsize=5, dx=6, dy=-2, leader=True)


# ------------------------------------------------------------------------------------------------------------------------- layouts
def _legend_from(fig, axes, loc, ncol, bbox=None, fontsize=6, extra=()):
    seen = {}
    for a in axes:
        for h, l in zip(*a.get_legend_handles_labels()): seen.setdefault(l, h)
    for h, l in extra: seen.setdefault(l, h)
    if seen: fig.legend(seen.values(), seen.keys(), loc=loc, ncol=ncol, frameon=False, fontsize=fontsize, bbox_to_anchor=bbox, handletextpad=0.4, columnspacing=1.0)


def _annotate_apart(ax, items, fontsize=5, dx=3, dy=2, color_of=None, leader=False, ground=False):
    """Label points (x, y, text) with text offset to the upper right, then move labels down, one by one from the top, until no two
    overlap (labels of points that coincide would otherwise print on top of each other)."""
    fig = ax.figure; fig.canvas.draw(); r = fig.canvas.get_renderer(); pt = fig.dpi / 72.0
    lead = dict(arrowprops=dict(arrowstyle="-", lw=0.3, color=OKABE["grey"], shrinkA=0.5, shrinkB=2, relpos=(0, 0.5))) if leader else {}   # from the label's left edge
    anns = [ax.annotate(s, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize, color=(color_of(s) if color_of else TEXT),
                        annotation_clip=False, va="bottom", zorder=6 if ground else None,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.4) if ground else None, **lead) for x, y, s in items]
    fig.canvas.draw()
    boxes = [a.get_window_extent(r) for a in anns]; placed = []
    for i in sorted(range(len(anns)), key=lambda k: -boxes[k].y0):
        bb, shift = boxes[i], 0.0
        for _ in range(60):
            moved = bb.translated(0, -shift); hits = [p for p in placed if moved.overlaps(p)]
            if not hits: break
            shift += max(moved.y1 - p.y0 for p in hits) + 0.5 * pt
        if shift: anns[i].set_position((dx, dy - shift / pt))
        placed.append(bb.translated(0, -shift))
    return anns


def fig3(R, out, sd):
    """Fig. 3: accuracy against lifetime energy, one panel per dataset (a-f for six datasets, a-g for seven); then the three costs
    per chunk (the next letter)."""
    D = R.datasets; n = len(D); ncol = 3 if n <= 6 else 4; nrow = (n + ncol - 1) // ncol
    fig = plt.figure(figsize=(W2, (155 if nrow <= 2 else 200) * MM))
    gs = fig.add_gridspec(nrow + 1, ncol, left=0.075, right=0.985, top=0.965, bottom=0.075, hspace=0.62, wspace=0.28 if ncol == 3 else 0.34,
                          height_ratios=[1] * nrow + [1.12])
    axes = []; data = []
    for i, d in enumerate(D):
        ax = fig.add_subplot(gs[i // ncol, i % ncol]); p_pareto(ax, R, d, data); axes.append(ax)
        if i % ncol == 0: ax.set_ylabel("final accuracy (%)")
        if i + ncol >= n: ax.set_xlabel("lifetime energy (mJ, log)")
        panel_letter(fig, ax, LETTERS[i], dx=-0.055 if i % ncol == 0 else -0.04, dy=0.012)
    sd[f"a-{LETTERS[n - 1]} accuracy vs energy"] = data
    axg = fig.add_subplot(gs[nrow, 0:ncol - 1]); cd = []; p_costs(axg, R, cd); sd[f"{LETTERS[n]} three costs"] = cd; panel_letter(fig, axg, LETTERS[n], dx=-0.055, dy=0.012)
    _legend_from(fig, axes, "center left", 1, bbox=(0.705 if ncol == 3 else 0.765, 0.17 if nrow <= 2 else 0.13), fontsize=6)
    save(fig, f"{out}/fig3_accuracy_energy_costs")


def fig4(R, out, sd):
    """Fig. 4: a retention at re-admission (CIFAR-100 and SHD, or the first two datasets present); b savings on relearning;
    c ablation ladder."""
    fig = plt.figure(figsize=(W2, 155 * MM)); data_a, data_b, data_c = [], [], []
    gs = fig.add_gridspec(2, 4, left=0.08, right=0.985, top=0.955, bottom=0.075, hspace=0.75, wspace=0.55, height_ratios=[1, 1.15])
    pick = [d for d in ("cifar100", "shd") if d in R.datasets] or R.datasets[:2]
    axs = []
    for i, d in enumerate(pick[:2]):
        ax = fig.add_subplot(gs[0, i]); p_retention_curve(ax, R, d, data_a, ylabel=(i == 0)); axs.append(ax)
    panel_letter(fig, axs[0], "a", dx=-0.065, dy=0.015)
    axb = fig.add_subplot(gs[0, 2:]); p_savings_S(axb, R, data_b); panel_letter(fig, axb, "b", dx=-0.06, dy=0.015)
    axc = fig.add_subplot(gs[1, 1:]); p_ablation(axc, R, data_c); panel_letter(fig, axc, "c", dx=-0.255, dy=0.015)
    _legend_from(fig, axs + [axb], "center", 6, bbox=(0.5, 0.505), fontsize=6)
    sd["a retention at re-admission"] = data_a; sd["b savings S"] = data_b; sd["c ablation"] = data_c
    save(fig, f"{out}/fig4_retention_savings_ablation")


def fig5(R, out, sd):
    """Fig. 5: a the gate as an operating curve; b lifetime energy against recurrence density (long streams); c recognition AUROC;
    d domain-shifted recurrences (accuracy on the shifted classes); e what the gate writes under shift."""
    fig = plt.figure(figsize=(W2, 178 * MM)); da, db, dc, dd, de = [], [], [], [], []
    gs = fig.add_gridspec(3, 2, left=0.085, right=0.985, top=0.965, bottom=0.12, hspace=0.72, wspace=0.3, height_ratios=[1, 1, 1])
    axa = fig.add_subplot(gs[0, 0]); p_gate_curve(axa, R, da); panel_letter(fig, axa, "a", dx=-0.07, dy=0.012)
    axb = fig.add_subplot(gs[0, 1]); p_density(axb, R, db); panel_letter(fig, axb, "b", dx=-0.07, dy=0.012)
    axb.legend(fontsize=5.2, loc="best", handlelength=1.4)
    axc = fig.add_subplot(gs[1, :]); p_recognition(axc, R, dc); panel_letter(fig, axc, "c", dx=-0.07, dy=0.03)
    axd = fig.add_subplot(gs[2, 0]); p_shift_acc(axd, R, dd); panel_letter(fig, axd, "d", dx=-0.07, dy=0.012)
    if len(R.shift): axd.legend(fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, handlelength=1.0)
    axe = fig.add_subplot(gs[2, 1]); p_shift_written(axe, R, de); panel_letter(fig, axe, "e", dx=-0.07, dy=0.012)
    sd["a gate operating curve"] = da; sd["b recurrence density"] = db; sd["c recognition"] = dc; sd["d shift accuracy"] = dd; sd["e shift writes"] = de
    save(fig, f"{out}/fig5_gate_density_recognition_shift")


def _pair_grid(n):
    """Rows and figure height for one pair of axes per dataset, two datasets per row (an Extended Data page each)."""
    rows = (n + 1) // 2
    return rows, (222 if rows <= 3 else 232) * MM


def _gs_mm(fig, rows, cols, h_mm, top_mm, bottom_mm, gap_mm, **kw):
    """A grid spec whose top and bottom margins and row gap are given in mm, so that a taller figure gets taller axes, not wider gaps."""
    top, bottom = 1 - top_mm / h_mm, bottom_mm / h_mm
    ax_h = ((top - bottom) * h_mm - (rows - 1) * gap_mm) / rows
    return fig.add_gridspec(rows, cols, top=top, bottom=bottom, hspace=gap_mm / ax_h, **kw)


def ed_streams(R, out, sd):
    D = R.datasets; rows, h = _pair_grid(len(D)); fig = plt.figure(figsize=(W2, h)); data = []
    gs = _gs_mm(fig, rows, 4, h / MM, 7.0, 20.0, 24.0, left=0.075, right=0.985, wspace=0.62)
    axes = []
    for i, d in enumerate(D):
        r, c = i // 2, (i % 2) * 2
        axa = fig.add_subplot(gs[r, c]); axe = fig.add_subplot(gs[r, c + 1]); p_stream(axa, axe, R, d, data); axes += [axa]
        panel_letter(fig, axa, LETTERS[i], dx=-0.06, dy=0.012)
    _legend_from(fig, axes, "lower center", 7, bbox=(0.5, 0.0), fontsize=6, extra=[(Patch(facecolor=SHADE, edgecolor="none"), "admission chunks")])
    sd[f"a-{LETTERS[len(D) - 1]} streams"] = data
    save(fig, f"{out}/ed_fig3_streams")


def ed_retention(R, out, sd):
    D = R.datasets; ncol = 3 if len(D) <= 6 else 4; fig = plt.figure(figsize=(W2, 155 * MM)); data = []
    gs = _gs_mm(fig, 2, ncol, 155.0, 5.5, 18.7, 20.0, left=0.09, right=0.985, wspace=0.18)
    axes = []
    for i, d in enumerate(D[:2 * ncol]):
        ax = fig.add_subplot(gs[i // ncol, i % ncol]); p_retention_curve(ax, R, d, data, ylabel=(i % ncol == 0)); axes.append(ax)
        if i % ncol: ax.set_yticklabels([])
        panel_letter(fig, ax, LETTERS[i], dx=-0.02 if i % ncol else -0.075, dy=0.012)
    _legend_from(fig, axes, "lower center", 5, bbox=(0.5, 0.0)); sd[f"a-{LETTERS[len(D) - 1]} retention"] = data
    save(fig, f"{out}/ed_fig4_retention")


def ed_savings_recovery(R, out, sd):
    D = R.datasets; ncol = 3 if len(D) <= 6 else 4; fig = plt.figure(figsize=(W2, 222 * MM)); d1, d2 = [], []
    gs = _gs_mm(fig, 4, ncol, 222.0, 13.0, 22.0, 19.8, left=0.08, right=0.985, wspace=0.22)   # room for the key above
    for i, d in enumerate(D[:2 * ncol]):
        ax = fig.add_subplot(gs[i // ncol, i % ncol]); p_steps(ax, R, d, d1)
        if i % ncol == 0: ax.set_ylabel("few-shot steps\nto criterion")
        if i == 0: panel_letter(fig, ax, "a", dx=-0.075, dy=0.012)
    axes = []
    for i, d in enumerate(D[:2 * ncol]):
        ax = fig.add_subplot(gs[2 + i // ncol, i % ncol]); p_recovery(ax, R, d, d2); axes.append(ax)
        if i % ncol == 0: ax.set_ylabel("balanced accuracy\nof the class (%)")
        if i == 0: panel_letter(fig, ax, "b", dx=-0.075, dy=0.012)
    _legend_from(fig, axes, "lower center", 4, bbox=(0.5, 0.0), fontsize=5.5)
    fig.legend(handles=[Patch(facecolor="#7F7F7F", edgecolor="#7F7F7F", label="steps to relearn a forgotten seen class"),
                        Patch(facecolor="white", edgecolor="#7F7F7F", hatch="//////", label="steps to learn a never-seen class (both capped at 40)"),
                        Line2D([], [], ls="none", label="number above the bars: savings S")],
               loc="upper right", bbox_to_anchor=(0.985, 0.995), ncol=3, frameon=False, fontsize=5.5, handlelength=1.2)
    sd["a few-shot steps"] = d1; sd["b recovery curves"] = d2
    save(fig, f"{out}/ed_fig6_savings_recovery")


def ed_density(R, out, sd):
    D = R.datasets; rows, h = _pair_grid(len(D)); fig = plt.figure(figsize=(W2, h)); data = []
    gs = _gs_mm(fig, rows, 4, h / MM, 7.0, 20.0, 24.0, left=0.095, right=0.985, wspace=0.66)
    axes = []
    for i, d in enumerate(D):
        r, c = i // 2, (i % 2) * 2
        axe = fig.add_subplot(gs[r, c]); axa = fig.add_subplot(gs[r, c + 1]); p_density_ds(axe, axa, R, d, data); axes.append(axe)
        panel_letter(fig, axe, LETTERS[i], dx=-0.075, dy=0.012)
    _legend_from(fig, axes, "lower center", 6, bbox=(0.5, 0.0), fontsize=6); sd[f"a-{LETTERS[len(D) - 1]} density"] = data
    save(fig, f"{out}/ed_fig7_density")


def ed_shift(R, out, sd):
    H = 150.0; fig = plt.figure(figsize=(W2, H * MM)); d1, d2 = [], []
    gs = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=1 - 7.0 / H, bottom=28.0 / H, wspace=0.3)    # the keys go below the panels
    leg_y = -11.4 / (H - 7.0 - 28.0)                                       # the keys 11.4 mm below the axes
    ax = fig.add_subplot(gs[0, 0]); ret = []
    sh = R.shift
    if len(sh):
        D = [d for d in R.datasets if (sh.dataset == d).any()]; x = np.arange(len(D)); w = 0.18
        for j, a in enumerate(["rewire", "ctrl_slda_input", "hlop", "replay"]):
            for k, (col, hc) in enumerate((("ret_first", None), ("ret_later", color(a)))):
                ys = [float(sh[(sh.dataset == d) & (sh.arm == a)][col].iloc[0]) if ((sh.dataset == d) & (sh.arm == a)).any() else NAN for d in D]
                if k == 0: ax.plot(x + (j - 1.5) * w, ys, ls="none", marker=marker(a), color=color(a), mec="black", mew=0.3, ms=4.5, label=f"{label(a)}, first shifted return")
                else: ax.plot(x + (j - 1.5) * w, ys, ls="none", marker=marker(a), color=color(a), mfc="white", mew=0.9, ms=4.5, label=f"{label(a)}, later returns")
                for d, y in zip(D, ys): d1.append(dict(dataset=DS_NAME.get(d, d), method=label(a), returns="first shifted" if k == 0 else "later", retention_pct=y))
        ax.axhline(50, color=OKABE["grey"], lw=0.5, ls=(0, (2, 2))); ax.set_xticks(x); ax.set_xticklabels([DS_TWO_LINE.get(d, d) for d in D], fontsize=5.5)
        ax.set_ylabel("retention at re-admission (%)"); ax.grid(axis="x", visible=False)
        ax.legend(fontsize=5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, leg_y), handletextpad=0.3, columnspacing=0.8)
    panel_letter(fig, ax, "a", dx=-0.07, dy=0.012)
    ax2 = fig.add_subplot(gs[0, 1])
    if len(sh):
        D = [d for d in R.datasets if (sh.dataset == d).any()]; x = np.arange(len(D))
        for j, a in enumerate(["rewire", "rewire_gated", "rewire_gated_maha", "ctrl_slda_input", "hlop", "replay", "naive"]):
            ys = [float(sh[(sh.dataset == d) & (sh.arm == a)].energy.iloc[0]) if ((sh.dataset == d) & (sh.arm == a)).any() else NAN for d in D]
            ax2.plot(x + (j - 3) * 0.1, ys, ls="none", marker=marker(a), color=color(a), mfc="white" if not filled(a) else color(a), mec=color(a) if not filled(a) else "black",
                     mew=0.8 if not filled(a) else 0.3, ms=4.2, label=label(a))
            for d, y in zip(D, ys): d2.append(dict(dataset=DS_NAME.get(d, d), method=label(a), lifetime_energy_mJ=y))
        ax2.set_yscale("log"); ax2.set_xticks(x); ax2.set_xticklabels([DS_TWO_LINE.get(d, d) for d in D], fontsize=5.5); ax2.grid(axis="x", visible=False)
        ev = [r["lifetime_energy_mJ"] for r in d2 if np.isfinite(r["lifetime_energy_mJ"]) and r["lifetime_energy_mJ"] > 0]
        if ev: ax2.set_ylim(min(ev) / 1.5, max(ev) * 1.5); log_ticks(ax2, min(ev) / 1.5, max(ev) * 1.5, axis="y")
        ax2.set_ylabel("lifetime energy under shift (mJ, log)")
        ax2.legend(fontsize=5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, leg_y), handletextpad=0.3, columnspacing=0.8)
    panel_letter(fig, ax2, "b", dx=-0.07, dy=0.012)
    sd["a retention under shift"] = d1; sd["b energy under shift"] = d2
    save(fig, f"{out}/ed_fig8_shift")


def _photo(ax, path, labels=()):
    """A photograph of the hardware (cropped on disk, nothing else changed), with labels joined to what they name by a leader drawn
    twice (dark under light) so that it reads on any background."""
    from PIL import Image
    img = np.asarray(Image.open(path).convert("RGB")); ax.imshow(img, interpolation="lanczos"); ax.set_axis_off()
    for text, (tx, ty), (lx, ly), ha in labels:
        ax.plot([lx, tx], [ly, ty], color="black", lw=1.4, solid_capstyle="round", zorder=3)
        ax.plot([lx, tx], [ly, ty], color="white", lw=0.6, solid_capstyle="round", zorder=4)
        ax.plot([tx], [ty], "o", ms=2.6, color="white", mec="black", mew=0.5, zorder=5)
        ax.text(lx, ly, text, fontsize=5.5, ha=ha, va="center", color="black", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#555555", lw=0.4))


def _flow(ax):
    """What runs where in the hardware measurement: the host prepares the frame, the AKD1000 runs the converted expansion and the
    sketch, the host writes the features and solves."""
    from matplotlib.patches import FancyBboxPatch
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_axis_off()
    chip_fill, host_fill = _tint(OKABE["blue"], 0.85), "white"
    ax.add_patch(FancyBboxPatch((1, 30.5), 98, 41, boxstyle="round,pad=0,rounding_size=2", fc=_tint(OKABE["blue"], 0.93), ec=OKABE["blue"], lw=0.6))
    ax.text(3, 69.5, "AKD1000: inference only, nothing is trained", fontsize=5.5, color=OKABE["blue"], fontweight="bold", va="top")
    boxes = [(80.5, 98, "on the host CPU (Compute Module 4)", "event or spike frame, averaged over time and\nquantised to 4 bits (700 or 2,312 inputs)", host_fill),
             (47.5, 64, None, "fully connected layer, 4-bit weights, then 1,024\nbounded rectifiers, 4-bit: the expansion", chip_fill),
             (33.5, 45.5, None, "fully connected layer, 4-bit weights, then 512\nbounded rectifiers, 4-bit: the sketch", chip_fill),
             (2, 22, "on the host CPU", "dual view with the input view; write into the\nsums G, C, n; solve the count-free discriminant", host_fill)]
    for y0, y1, head, body, fc in boxes:
        ax.add_patch(FancyBboxPatch((6, y0), 88, y1 - y0, boxstyle="round,pad=0,rounding_size=1.5", fc=fc, ec="#555555", lw=0.5))
        if head: ax.text(8, y1 - 1.5, head, fontsize=5.5, fontweight="bold", va="top", color=TEXT)
        ax.text(50, (y0 + y1) / 2 - (2.4 if head else 0), body, fontsize=5.5, va="center", ha="center", color="black", linespacing=1.25)
    for xa, ya, yb, lab, ly in ((84, 80.5, 64.6, "PCIe", 76.0), (50, 47.4, 46.0, None, 0), (84, 33.4, 22.6, "PCIe", 26.5)):
        ax.annotate("", xy=(xa, yb), xytext=(xa, ya), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black", mutation_scale=6))
        if lab: ax.text(xa + 2, ly, lab, fontsize=5.5, va="center", color=TEXT)       # clear of the chip's frame


def ed_hardware(R, out, sd):
    """Extended Data figure of the hardware measurement (one page): the node (photographs in <run>/hardware/photos/, when present),
    what runs where, the accuracy cost of the converted expansion on the chip, the energy per frame measured against the
    operation-count model, and where the chip's accuracy stands among the campaign's methods on the same streams and seeds."""
    hw = [ak for ak in R.akida if ak.get("backend") == "hw" and ak.get("per_seed")]
    if not hw: return
    from pathlib import Path
    photos = {}
    for run in R.camp.runs:
        for name in ("node", "card"):
            p = Path(run) / "hardware" / "photos" / f"{name}.jpg"
            if p.exists() and name not in photos: photos[name] = str(p)
    H = 215.0 if photos else 120.0; top = H - 4.0
    fig = plt.figure(figsize=(W2, H * MM))
    def box(x0, y0, w, h): return [x0 / 180.0, y0 / H, w / 180.0, h / H]          # mm from the bottom left -> figure fraction
    letters = iter("abcdef")
    def letter(ax, dx=0.0, dy=0.004): panel_letter(fig, ax, next(letters), dx=dx, dy=dy)
    if "node" in photos:
        ax = fig.add_axes(box(4, top - 90, 84, 90))
        _photo(ax, photos["node"], [("Raspberry Pi Compute Module 4\n(8 GB RAM)", (488, 505), (640, 150), "center"),
                                    ("BrainChip AKD1000 PCIe card", (330, 650), (700, 585), "center"),
                                    ("Compute Module 4 IO board", (560, 870), (420, 945), "center")])
        letter(ax, dx=-0.018)
    if "card" in photos:
        ax = fig.add_axes(box(95, top - 43, 83, 43))
        _photo(ax, photos["card"], [("AKD1000 neuromorphic processor", (338, 150), (620, 40), "center"),
                                    ("PCIe edge connector", (520, 350), (300, 350), "right")])
        letter(ax, dx=-0.018)
    y_c = top - 98 if photos else H - 58
    ax = fig.add_axes(box(92, y_c, 87, 50) if photos else box(2, y_c, 72, 54)); _flow(ax); letter(ax)
    D = [ak["dataset"] for ak in hw]
    # d: accuracy cost on the chip, seed by seed, paired against the LIF layer on the same streams
    ax = fig.add_axes(box(17, 64, 62, 39) if photos else box(91, 13, 34, 42)); data = []
    for i, ak in enumerate(hw):
        rows = sorted(ak["per_seed"], key=lambda r: r["seed"]); dif = [r["acc_chip"] - r["acc_lif"] for r in rows]
        xs = i + np.linspace(-0.16, 0.16, len(dif))
        ax.plot(xs, dif, ls="none", marker="D", ms=3.2, mfc="white", mec=OKABE["blue"], mew=0.8, zorder=3)
        ax.plot([i - 0.26, i + 0.26], [np.mean(dif)] * 2, color=OKABE["blue"], lw=1.2, zorder=4)
        ax.text(i + 0.3, np.mean(dif), f"{np.mean(dif):+.1f}".replace("-", "\u2212"), fontsize=5.5, va="center", color=OKABE["blue"])
        for r, dd in zip(rows, dif): data.append(dict(dataset=DS_NAME.get(ak["dataset"], ak["dataset"]), seed=r["seed"], acc_lif=r["acc_lif"], acc_software_model=r["acc_sim"], acc_chip=r["acc_chip"], chip_minus_lif=dd))
    ax.axhline(0, color=OKABE["grey"], lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks(range(len(D))); ax.set_xticklabels([DS_TWO_LINE.get(d, DS_NAME.get(d, d)) for d in D], fontsize=5.5); ax.set_xlim(-0.6, len(D) - 0.15)
    ax.set_ylabel("chip minus LIF layer (points)"); ax.grid(axis="x", visible=False)
    lo = min(r["chip_minus_lif"] for r in data); ax.set_ylim(min(-1.0, lo - 1.0), 1.5)
    letter(ax, dx=-0.075); sd["accuracy on the chip"] = data
    # e: energy per frame, measured above the idle floor against the operation-count model of the same layers
    ax = fig.add_axes(box(107, 64, 70, 39) if photos else box(141, 13, 36, 42)); data = []; w = 0.36
    for i, ak in enumerate(hw):
        rows = ak["per_seed"]; meas = float(np.mean([r["net_energy_per_frame_uJ"] for r in rows])); mod = float(np.mean([r["modelled_uJ"] for r in rows]))
        lat = float(np.mean([r["latency_ms"] for r in rows]))
        ax.bar(i - w / 2, meas, width=w, color=OKABE["blue"], edgecolor="black", lw=0.4, zorder=3)
        ax.bar(i + w / 2, mod, width=w, color="white", edgecolor=OKABE["blue"], hatch="//////", lw=0.6, zorder=3)
        ax.text(i - w / 2, meas * 1.25, f"{meas:.1f}", fontsize=5.5, ha="center", va="bottom"); ax.text(i + w / 2, mod * 1.25, f"{mod:.2f}", fontsize=5.5, ha="center", va="bottom")
        data.append(dict(dataset=DS_NAME.get(ak["dataset"], ak["dataset"]), measured_above_idle_uJ=meas, modelled_uJ=mod, ratio=meas / mod, latency_ms=lat))
    ax.set_yscale("log"); ax.set_ylim(0.1, 2000); log_ticks(ax, 0.1, 1000, axis="y")
    ax.set_xticks(range(len(D))); ax.set_xlim(-0.6, len(D) - 0.4)
    ax.set_xticklabels([f"{DS_NAME.get(r['dataset'], r['dataset'])}\n{r['latency_ms']:.2f} ms, {r['ratio']:.0f}\u00d7" for r in data], fontsize=5.5)
    ax.set_ylabel("energy per frame ($\\mu$J, log)"); ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(fc=OKABE["blue"], ec="black", lw=0.4, label="measured, above idle"), Patch(fc="white", ec=OKABE["blue"], hatch="//////", lw=0.6, label="operation-count model")],
              fontsize=5.5, loc="upper left", frameon=False, handlelength=1.2, borderaxespad=0.2)
    letter(ax, dx=-0.075); sd["energy per frame"] = data
    if photos:
        # f: the chip's accuracy among the campaign's methods on the same streams and seeds (mean +- s.d. over the seeds)
        arms = ["rewire", "chip", "ctrl_relu", "ctrl_slda_input", "replay_iso", "replay", "hlop"]
        names = {"chip": "REWIRE, expansion on the AKD1000", "rewire": "REWIRE, LIF layer (GPU campaign)"}
        data = []; axes = []
        for i, ak in enumerate(hw):
            d = ak["dataset"]; ax = fig.add_axes(box(52 + i * 64, 12, 58, 36)); axes.append(ax)
            chip = np.array([r["acc_chip"] for r in ak["per_seed"]])
            for j, a in enumerate(arms):
                if a == "chip": m_, s_, c_, mk, fl = chip.mean(), chip.std(ddof=1), OKABE["blue"], "D", True
                else:
                    xs = R.df[(R.df.dataset == d) & (R.df.arm == a)].acc.to_numpy(float)
                    if not len(xs): continue
                    m_, s_, c_, mk, fl = xs.mean(), (xs.std(ddof=1) if len(xs) > 1 else 0.0), color(a), marker(a), filled(a)
                y = len(arms) - 1 - j
                ax.errorbar(m_, y, xerr=s_, fmt="none", ecolor=c_, elinewidth=0.6, capsize=1.5, zorder=3)
                ax.plot(m_, y, ls="none", marker=mk, ms=5 if a in ("rewire", "chip") else 4, color=c_, mfc=c_ if fl else "white",
                        mec="black" if fl else c_, mew=0.3 if fl else 0.8, zorder=4)
                data.append(dict(dataset=DS_NAME.get(d, d), method=names.get(a, label(a)), mean_final_acc=float(m_), sd=float(s_), n_seeds=int(len(chip) if a == "chip" else len(xs))))
            ax.set_yticks(range(len(arms))); ax.set_ylim(-0.6, len(arms) - 0.4)
            ax.set_yticklabels([names.get(a, label(a)) for a in reversed(arms)] if i == 0 else [], fontsize=5.5)
            ax.set_title(DS_NAME.get(d, d), fontsize=6.5); ax.set_xlabel("final accuracy (%)"); ax.grid(axis="y", visible=False)
            ax.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        letter(axes[0], dx=-0.27); sd["accuracy among the methods"] = data
    save(fig, f"{out}/ed_fig9_hardware")


def aux_fig1e(R, out, sd):
    """A reference drawing of Fig. 1e (the diagram itself is drawn by hand): lifetime energy against the number of recurrence chunks
    R for K = 8 classes admitted in K - 1 = 7 chunks (the first admits two), E(R) = (K - 1) x mean admission energy + R x mean
    re-admission energy (equation (1), L = K - 1 + R), with the means over the datasets of the records."""
    K, Rmax = 8, 64; fig = plt.figure(figsize=(W2, 50 * MM)); ax = fig.add_axes([0.08, 0.2, 0.62, 0.72]); data = []; items, cols = [], {}
    for a, ls in (("replay", "-"), ("hlop", "-"), ("rewire", "-"), ("rewire_gated", (0, (4, 2)))):
        adm, rd = R.mean_over(a, "adm"), R.mean_over(a, "readm")
        if not (np.isfinite(adm) and np.isfinite(rd)): continue
        e0, e1 = (K - 1) * adm, (K - 1) * adm + Rmax * rd
        ax.plot([0, Rmax], [e0, e1], color=color(a), lw=1.4, ls=ls); ax.plot([Rmax], [e1], "o", color=color(a), ms=5)
        s = f"{label(a).replace(' (ours)', '')} ({e0:,.1f} to {e1:,.1f})\n+{rd:.1f} mJ per recurrence"; items.append((Rmax, e1, s)); cols[s] = color(a)
        data.append(dict(method=label(a), K=K, mean_admission_mJ=adm, mean_readmission_mJ=rd, E0_mJ=e0, E64_mJ=e1))
    r8 = [r["n_recur"] for r in R.camp.main if r["dataset"] in ("mnist", "nmnist", "cifar10") and r.get("n_recur") is not None]
    if r8: ax.axvline(r8[0], color=OKABE["grey"], lw=0.5, ls=(0, (2, 2))); ax.text(r8[0] + 0.6, ax.get_ylim()[1] * 0.95, "the 8-class streams", fontsize=5.5, color=TEXT, va="top")
    ax.set_xlim(0, Rmax); ax.set_xticks(range(0, Rmax + 1, 8)); ax.set_xlabel(f"recurrence chunks R ({K} classes admitted once)")
    ax.set_ylabel("lifetime energy (mJ)"); sd["fig1e reference values"] = data
    if items: _annotate_apart(ax, items, fontsize=5.5, dx=8, dy=-6, color_of=cols.get, leader=True)
    save(fig, f"{out}/aux_fig1e_reference")


def si_eps(R, out, sd):
    fig = plt.figure(figsize=(W1, 62 * MM)); ax = fig.add_axes([0.14, 0.2, 0.66, 0.72]); data = []; p_eps(ax, R, data); sd["shrinkage sensitivity"] = data
    save(fig, f"{out}/si_fig2_shrinkage")


def si_gate_thresholds(R, out, sd):
    g = R.gate[R.gate.arm == "rewire_gated"] if len(R.gate) else None
    fig = plt.figure(figsize=(W2, 58 * MM)); data = []
    gs = fig.add_gridspec(1, 3, left=0.07, right=0.985, top=0.9, bottom=0.25, wspace=0.35)
    shades = {0.75: _tint(OKABE["green"], 0.6), 0.85: OKABE["green"], 0.95: "#00563F"}
    for k, (col, yl) in enumerate((("skipped", "recurrences skipped (%)"), ("saved", "lifetime energy saved (%)"), ("dacc", "accuracy change (points)"))):
        ax = fig.add_subplot(gs[0, k]); D = R.datasets; x = np.arange(len(D))
        if g is not None and len(g):
            for i, d in enumerate(D):
                gd = g[g.dataset == d].sort_values("theta")
                if not len(gd): continue
                ax.plot([i] * len(gd), gd[col], color="#C8C8C4", lw=1.2, zorder=1)
                for r in gd.itertuples():
                    ax.plot(i, getattr(r, col), "o", color=shades.get(round(r.theta, 2), OKABE["green"]), mec="black", mew=0.3, ms=4, zorder=3)
                    if k == 0: data.append(dict(dataset=DS_NAME.get(d, d), theta=r.theta, skipped_pct=r.skipped, energy_saved_pct=r.saved, accuracy_change=r.dacc))
        ax.axhline(0, color=OKABE["grey"], lw=0.5); ax.set_xticks(x); ax.set_xticklabels([DS_TWO_LINE.get(d, d) for d in D], fontsize=5.2); ax.set_ylabel(yl); ax.grid(axis="x", visible=False)
        panel_letter(fig, ax, "abc"[k], dx=-0.055, dy=0.02)
    fig.legend(handles=[Line2D([], [], marker="o", ls="none", color=c, mec="black", mew=0.3, ms=4, label=f"$\\theta$ = {t:.2f}") for t, c in shades.items()], loc="lower center", ncol=3, frameon=False, fontsize=6)
    sd["gate thresholds"] = data
    save(fig, f"{out}/si_fig1_gate_thresholds")


# --------------------------------------------------------------------------------------------------------------------- per-analysis layout
def separate(R, out, sd):
    """One file per analysis (fig06-fig14), in the same style, for inspection; not used by the manuscript."""
    D = R.datasets[:6]
    fig = plt.figure(figsize=(W2, 92 * MM)); gs = fig.add_gridspec(2, 3, left=0.075, right=0.985, top=0.94, bottom=0.25, hspace=0.62, wspace=0.28); axes = []; data = []
    for i, d in enumerate(D):
        ax = fig.add_subplot(gs[i // 3, i % 3]); p_pareto(ax, R, d, data); axes.append(ax)
        if i % 3 == 0: ax.set_ylabel("accuracy (%)")
        if i // 3 == 1: ax.set_xlabel("lifetime energy (mJ, log)")
    _legend_from(fig, axes, "lower center", 4, bbox=(0.5, 0.0)); sd["fig06"] = data; save(fig, f"{out}/fig06_accuracy_energy")
    for i, d in enumerate(D):
        fig = plt.figure(figsize=(W2, 57 * MM)); gs = fig.add_gridspec(1, 2, left=0.08, right=0.985, top=0.88, bottom=0.32, wspace=0.25); data = []
        axa, axe = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]); p_stream(axa, axe, R, d, data)
        _legend_from(fig, [axa], "lower center", 7, bbox=(0.5, 0.0), extra=[(Patch(facecolor=SHADE, edgecolor="none"), "admission chunks")])
        sd[f"fig07{'abcdef'[i]}"] = data; save(fig, f"{out}/fig07{'abcdef'[i]}_stream_{d}")
    fig = plt.figure(figsize=(W1, 50 * MM)); ax = fig.add_axes([0.15, 0.22, 0.83, 0.72]); data = []; p_costs(ax, R, data); sd["fig08a"] = data; save(fig, f"{out}/fig08a_three_costs")
    fig = plt.figure(figsize=(W1, 55 * MM)); gs = fig.add_gridspec(1, 2, left=0.17, right=0.98, top=0.9, bottom=0.36, wspace=0.12); data = []; axs = []
    for i, d in enumerate([d for d in ("cifar100", "shd") if d in D][:2]):
        ax = fig.add_subplot(gs[0, i]); p_retention_curve(ax, R, d, data, ylabel=(i == 0)); axs.append(ax)
        if i: ax.set_yticklabels([])
    _legend_from(fig, axs, "lower center", 3, bbox=(0.55, 0.0), fontsize=5.5); sd["fig08b"] = data; save(fig, f"{out}/fig08b_retention")
    ed_retention(R, out, sd); import os; os.replace(f"{out}/ed_fig4_retention.eps", f"{out}/fig09_retention_all.eps"); os.replace(f"{out}/ed_fig4_retention.png", f"{out}/fig09_retention_all.png"); os.replace(f"{out}/ed_fig4_retention.pdf", f"{out}/fig09_retention_all.pdf")
    fig = plt.figure(figsize=(W2, 70 * MM)); gs = fig.add_gridspec(2, 3, left=0.07, right=0.985, top=0.93, bottom=0.12, hspace=0.75, wspace=0.18); data = []
    for i, d in enumerate(D):
        ax = fig.add_subplot(gs[i // 3, i % 3]); p_steps(ax, R, d, data)
        if i % 3 == 0: ax.set_ylabel("few-shot steps\nto criterion")
    sd["fig10a"] = data; save(fig, f"{out}/fig10a_savings")
    fig = plt.figure(figsize=(W2, 82 * MM)); gs = fig.add_gridspec(2, 3, left=0.07, right=0.985, top=0.93, bottom=0.25, hspace=0.7, wspace=0.18); data = []; axs = []
    for i, d in enumerate(D):
        ax = fig.add_subplot(gs[i // 3, i % 3]); p_recovery(ax, R, d, data); axs.append(ax)
        if i % 3 == 0: ax.set_ylabel("balanced accuracy\nof the class (%)")
    _legend_from(fig, axs, "lower center", 4, bbox=(0.5, 0.0), fontsize=5.5); sd["fig10b"] = data; save(fig, f"{out}/fig10b_recovery")
    fig = plt.figure(figsize=(W1 * 1.45, 62 * MM)); ax = fig.add_axes([0.36, 0.3, 0.6, 0.66]); data = []; p_ablation(ax, R, data); sd["fig11"] = data; save(fig, f"{out}/fig11_ablation")
    fig = plt.figure(figsize=(W1, 62 * MM)); ax = fig.add_axes([0.17, 0.2, 0.8, 0.74]); data = []; p_gate_curve(ax, R, data); sd["fig12a"] = data; save(fig, f"{out}/fig12a_gate")
    si_gate_thresholds(R, out, sd); os.replace(f"{out}/si_fig1_gate_thresholds.eps", f"{out}/fig12b_gate_thresholds.eps"); os.replace(f"{out}/si_fig1_gate_thresholds.png", f"{out}/fig12b_gate_thresholds.png"); os.replace(f"{out}/si_fig1_gate_thresholds.pdf", f"{out}/fig12b_gate_thresholds.pdf")
    for i, d in enumerate(D):
        fig = plt.figure(figsize=(W2, 55 * MM)); gs = fig.add_gridspec(1, 2, left=0.08, right=0.985, top=0.88, bottom=0.34, wspace=0.25); data = []
        axe, axa = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]); p_density_ds(axe, axa, R, d, data); _legend_from(fig, [axe], "lower center", 6, bbox=(0.5, 0.0))
        panel_letter(fig, axe, "abcdef"[i], dx=-0.07, dy=0.03); sd[f"fig13{'abcdef'[i]}"] = data; save(fig, f"{out}/fig13{'abcdef'[i]}_density_{d}")
    fig = plt.figure(figsize=(W1, 58 * MM)); ax = fig.add_axes([0.14, 0.2, 0.84, 0.75]); data = []; p_recognition(ax, R, data); sd["fig14"] = data; save(fig, f"{out}/fig14_recognition")


NCS = [fig3, fig4, fig5, ed_streams, ed_retention, ed_savings_recovery, ed_density, ed_shift, ed_hardware, si_eps, si_gate_thresholds, aux_fig1e]
