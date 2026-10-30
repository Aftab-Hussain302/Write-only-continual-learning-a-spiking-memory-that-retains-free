"""End-to-end self-test on the CPU (a few minutes, no datasets needed): every arm of the paper on a short synthetic stream, the
domain-shifted grid, the task- and domain-incremental native protocols, every metric finite, every diagnostic figure written, and the
guards on the statistics and the shift transform."""
import os, json, numpy as np, torch
from rewire.configure import config_for
from rewire.engine import run_all, run_grid
from rewire.native import run_native
from rewire.models.arms import PAPER_ARMS

cfg = config_for("synthetic", dict(seeds=[0], n_recur=6, full_steps=8, budget=8, n_train_per_class=120, n_test_per_class=40, recovery_budgets=[0, 1, 2],
                                   hidden=128, rw_hidden=128, rw_sketch=64, T=3, stem="conv", conv_stem=False, data_root="data"))
cfg["out"] = "results_selftest/none"; df = run_all(cfg, log=lambda *a: None)
assert len(df) == len(PAPER_ARMS), f"an arm failed: {set(PAPER_ARMS) - set(df.arm)}"
for k in ["final_acc", "bwt", "bwt_recur", "fam_auroc_test", "S_AUR", "AUR_seen", "AUR_novel", "RIF", "recog_auroc_fam", "recog_auroc_fam_x1", "retention_mean",
          "lifetime_energy_uJ", "standing_bytes", "resident_bytes_total"]:
    assert np.isfinite(df[k]).all(), f"non-finite {k}"
for f in ["pareto_acc_vs_energy.png", "acc_over_stream.png", "chunk_cost.png", "retention_at_recurrence.png", "recovery_curves.png", "leaderboard.csv", "metrics.csv"]:
    assert os.path.exists(f"results_selftest/none/{f}"), f"missing {f}"
rows = json.load(open("results_selftest/none/rows.json"))
rw = [r for r in rows if r["arm"] == "rewire"][0]; sk = [r for r in rows if r["arm"] == "abl_sketch_ridge"][0]
assert rw["standing_bytes"] > 0 and rw["feature_dim"] == 64 + min(28 * 28, cfg["rw_skip_max"]) and sk["feature_dim"] == 64, "REWIRE's dual view has sketch + input dimensions"
assert all(x["steps"] == 0 for x in rw["visits"]), "REWIRE must take no gradient steps"
gated = [r for r in rows if r["arm"] == "rewire_gated"][0]
assert all(x["write_cost_uJ"] == 0.0 for x in gated["visits"] if not x["wrote"]), "a skipped recurrence costs no write"
hd = [r for r in rows if r["arm"] == "hd_memory"][0]; assert hd["feature_dim"] == rw["feature_dim"], "HD bundling on REWIRE's features"
# ---- domain-shifted recurrences: the shift arms, every recurrence shifted, the first shifted return flagged
c2 = dict(cfg, shift_mode="domain", out="results_selftest/domain", arms=[a for a in cfg["shift_arms"] if a in PAPER_ARMS]); c2["_overrides"] = {"shift_mode": "domain"}
srows = run_grid(c2, lambda *a: None)
assert all(r["shift_mode"] == "domain" for r in srows) and all(any(v["first_shift"] for v in r["visits"]) for r in srows), "shifted recurrences recorded"
assert all(np.isfinite(r["final_acc_shifted"]) for r in srows), "accuracy on the shifted classes"
# ---- the native protocols (task- and domain-incremental), for a gradient and a write-only learner
nrows = run_native(dict(cfg, native_dataset="synthetic", native_tasks_pmnist=3, native_n_per_class=60, native_n_test=20), arms=["naive", "rewire"], seeds=(0,), epochs=1,
                   out="results_selftest/native", log=lambda *a: None)
assert len(nrows) == 4 and all(np.isfinite(r["acc"]) for r in nrows), "native protocols"
from rewire.dataloader import apply_shift
Xs = torch.rand(4, 1, 28, 28); c3 = dict(cfg, shift_mode="rotate")
assert torch.allclose(apply_shift(Xs, 3, 0, c3), apply_shift(Xs, 7, 0, c3)), "the shift must not depend on the class"
assert not torch.allclose(apply_shift(Xs, 3, 0, c3), apply_shift(Xs, 3, 1, c3)), "the shift should differ across seeds"
from rewire.metrics.stats import wilcoxon_exact, holm
assert wilcoxon_exact([0.05] * 8)[0] < 0.05 and wilcoxon_exact([0.0] * 8)[0] == 1.0, "paired test broken"
_p = np.array([0.30, 0.001, 0.02]); assert (holm(_p) >= _p).all(), "Holm correction must never lower a p-value"
print(f"SELFTEST PASSED: {len(PAPER_ARMS)} arms on the synthetic stream, the domain-shifted grid, the native protocols, figures, metrics finite, statistics guards")
