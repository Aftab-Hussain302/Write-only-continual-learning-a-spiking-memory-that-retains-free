# Paper results: final

Built by scripts/make_paper_figures.py (version 2026-10-29) on 2026-10-30T15:56:19 from: records/results_2026-10-29_1457
Rows: {'main': 870, 'shift': 210, 'sweep': 910, 'gate': 180, 'eps_sens': 150, 'eps_dev': 5, 'native': 60, 'akida': 2, 'datasets': ['mnist', 'nmnist', 'shd', 'cifar10', 'cifar100', 'tinyimagenet'], 'seeds': [0, 1, 2, 3, 4], 'arms': 29, 'synthetic': False}. Code hash of the run: 83030c41f78c. Font: Liberation Sans.

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
| macros/numbers.tex | every named value as \defval{key}{value}; the manuscript quotes them as \val{key} |
| facts.json | the same values with their unrounded numbers |
| source_data/*.xlsx | Source data, one workbook per figure |
