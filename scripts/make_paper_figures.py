#!/usr/bin/env python3
"""Build every results figure, table and named value of the paper from one or more finished run folders. Nothing is re-run and
nothing is downloaded; a new output folder is created on every call, so earlier versions are never overwritten.

    python3 scripts/make_paper_figures.py --runs results/results_2026-11-02_0830 --name final
    python3 scripts/make_paper_figures.py --runs results/results_2026-11-02_0830 results/akida_2026-11-05 --name final

Writes paper_results_<name>_<YYYYmmdd-HHMMSS>/ (next to the first run folder unless --out is given):
  figures/        fig3_accuracy_energy_costs, fig4_retention_savings_ablation, fig5_gate_density_recognition_shift (main),
                  ed_fig3, ed_fig4, ed_fig6..ed_fig9 (Extended Data; Figs 1, 2 and 5 are diagrams drawn by hand), si_fig1..si_fig2
                  (Supplementary), aux_fig1e_reference (for redrawing Fig. 1e)
                  (with --layout both, also one file per analysis in separate/), each as EPS and PNG (300 dpi) with a PDF copy; 88 or 180 mm wide,
                  5-7 pt sans-serif text
  tables/         one \\input-ready .tex per table (the tabular only; caption and label stay in the manuscript) and a CSV each
  macros/numbers.tex   \\defval{key}{value} for every named value (facts.json holds the unformatted values)
  source_data/    one Excel workbook per figure (one sheet per panel), as Nature Portfolio asks
  manifest.json   the run folders, rows per stage, code hash, seeds, datasets, font, date and script version
  README.md       which file goes where in the manuscript
  integrity_report.txt   the scan of every figure's text layer for look-alike characters"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
VERSION = "2026-10-29"


def shown(p):
    """A run folder as recorded in the manifest: relative to the working directory when it lies inside it, else absolute."""
    p = Path(p).resolve()
    try: return str(p.relative_to(Path.cwd().resolve()))
    except ValueError: return str(p)


def write_source_data(sd, path):
    import pandas as pd
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for sheet, rows in sd.items():
            df = pd.DataFrame(rows) if rows else pd.DataFrame({"note": ["no data in the records for this panel"]})
            df.to_excel(w, sheet_name=sheet[:31].replace("/", "-"), index=False)


README = """# Paper results: {name}

Built by scripts/make_paper_figures.py (version {version}) on {date} from: {runs}
Rows: {rows}. Code hash of the run: {code}. Font: {font}.
{synthetic}
## Where each file goes in the manuscript (main.tex / supplementary.tex)

| file | place |
|---|---|
| figures/fig3_accuracy_energy_costs.eps | Fig. 3 (accuracy against lifetime energy per dataset; the three costs) |
| figures/fig4_retention_savings_ablation.eps | Fig. 4 (retention at re-admission, a; savings, b; ablation ladder, c) |
| figures/fig5_gate_density_recognition_shift.eps | Fig. 5 (gate, a; recurrence density, b; recognition, c; domain shift, d-e) |
| figures/ed_fig3_streams.eps | Extended Data Fig. 3 (the streams chunk by chunk) |
| figures/ed_fig4_retention.eps | Extended Data Fig. 4 (retention at re-admission, every dataset) |
| figures/ed_fig6_savings_recovery.eps | Extended Data Fig. 6 (few-shot steps and recovery curves) |
| figures/ed_fig7_density.eps | Extended Data Fig. 7 (recurrence density and long streams) |
| figures/ed_fig8_shift.eps | Extended Data Fig. 8 (domain-shifted recurrences) |
| figures/ed_fig9_hardware.eps | Extended Data Fig. 9 (the neuromorphic node, what runs where, accuracy and energy on the AKD1000; with the hardware records) |
| figures/si_fig1_gate_thresholds.eps, si_fig2_shrinkage.eps | Supplementary Figs 1-2 |
| figures/aux_fig1e_reference.eps | not in the paper: a reference drawing of Fig. 1e (drawn by hand) with the values of this run |
| figures/separate/fig06...fig14 | with --layout both: one file per analysis, for inspection (not used by the manuscript) |
| tables/tab1_main.tex | Table 1 |
| tables/ed_tab1_native.tex | Extended Data Table 1 |
| tables/si_tab_hardware.tex | Supplementary Table of the hardware measurement, seed by seed (with the hardware records) |
| tables/si_tab_*.tex | Supplementary Tables |
| macros/numbers.tex | every named value as \\defval{{key}}{{value}}; the manuscript quotes them as \\val{{key}} |
| facts.json | the same values with their unrounded numbers |
| source_data/*.xlsx | Source data, one workbook per figure |
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--name", default="final"); ap.add_argument("--out", default=None)
    ap.add_argument("--layout", choices=("ncs", "separate", "both"), default="ncs")
    ap.add_argument("--allow-synthetic", action="store_true", help="accept records of the synthetic pipeline test (testing the tools only)")
    ap.add_argument("--no-scan", action="store_true")
    a = ap.parse_args()
    from rewire.report import style
    from rewire.report.load import Campaign
    from rewire.report.compute import Results
    from rewire.report import figures, tables, facts as factsmod
    font = style.setup()
    camp = Campaign(a.runs, allow_synthetic=a.allow_synthetic); R = Results(camp)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(a.out) if a.out else Path(a.runs[0]).resolve().parent / f"paper_results_{a.name}_{stamp}"
    if out.exists(): out = Path(str(out) + "_" + stamp)
    for sub in ("figures", "tables", "macros", "source_data") + (("figures/separate",) if a.layout != "ncs" else ()): (out / sub).mkdir(parents=True, exist_ok=True)
    print(f"[paper] {camp.summary()}\n[paper] writing {out}", flush=True)
    made = []
    if a.layout in ("ncs", "both"):
        for f in figures.NCS:
            sd = {}; f(R, str(out / "figures"), sd); made.append(f.__name__)
            write_source_data(sd, str(out / "source_data" / f"{f.__name__}.xlsx"))
    if a.layout in ("separate", "both"):
        sd = {}; figures.separate(R, str(out / "figures" / "separate"), sd); write_source_data(sd, str(out / "source_data" / "separate_fig06-fig14.xlsx"))
    for t in tables.ALL: t(R, str(out / "tables"))
    F = factsmod.build(R)
    hdr = "% SYNTHETIC PIPELINE TEST: not results\n" if camp.synthetic else ""
    factsmod.to_tex(F, str(out / "macros" / "numbers.tex"), header=hdr)
    json.dump({k: v for k, v in F.items()}, open(out / "facts.json", "w"), indent=0, default=str)
    if not a.no_scan:
        try:
            from rewire.report.integrity import report
            pdfs = sorted(str(p) for p in (out / "figures").rglob("*.pdf"))
            (out / "integrity_report.txt").write_text(report(pdfs))
        except ImportError:
            print("[paper] integrity scan skipped: pip install pymupdf", flush=True)
    man = dict(version=VERSION, date=time.strftime("%Y-%m-%dT%H:%M:%S"), runs=[shown(r) for r in a.runs], rows=camp.summary(),
               code_hash=[m.get("code_hash") for m in camp.meta.get("manifests", [])], eps_selected=camp.meta.get("eps_selected"),
               font=font, figures=made, synthetic=camp.synthetic, n_values=len(F))
    json.dump(man, open(out / "manifest.json", "w"), indent=1, default=str)
    (out / "README.md").write_text(README.format(name=a.name, version=VERSION, date=man["date"], runs=", ".join(man["runs"]), rows=man["rows"],
                                                 code=", ".join(str(c)[:12] for c in man["code_hash"] if c) or "n/a", font=font,
                                                 synthetic="\n**SYNTHETIC PIPELINE TEST: these are not results.**\n" if camp.synthetic else ""))
    print(f"[paper] done: {len(made)} figure layouts, {len(tables.ALL)} table writers, {len(F)} named values -> {out}", flush=True)
    return out


if __name__ == "__main__": main()
