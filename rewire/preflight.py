import os
"""Fast scientific guards, run before launching expensive jobs. No dataset downloads."""
import tempfile
import numpy as np
import torch
from .configure import DEFAULT, PRESETS, PRESET_OF, config_for
from .dataloader.loaders import make_stream
from .models.arms import RidgeMemory, build
from .modules import DEV


def check(condition, name):
    if not condition: raise AssertionError(name)
    print("[PASS]", name, flush=True)


def main():
    check(torch.cuda.is_available() or os.environ.get("REWIRE_PREFLIGHT_CPU") == "1", "CUDA available on training host (set REWIRE_PREFLIGHT_CPU=1 to check the mathematics on CPU)")
    check(DEFAULT["hidden"] == DEFAULT["rw_hidden"] == 1024, "gradient baselines and REWIRE's expansion share the width 1024")
    check(len(DEFAULT["seeds"]) >= 5, "at least five seeds")
    from .configure import PAPER_DATASETS
    check(set(PAPER_DATASETS) == {"mnist", "cifar10", "cifar100", "nmnist", "shd", "tinyimagenet"} and all(PRESET_OF[d] in PRESETS for d in PRESET_OF),
          "six datasets (SSC optional) with a preset each")
    torch.manual_seed(0)
    h = torch.randn(19, 8, device=DEV); y = torch.randint(0, 3, (19,), device=DEV)
    A = h.T @ h + torch.eye(8, device=DEV)
    C = torch.zeros(8, 3, device=DEV); C.index_add_(1, y, h.T)
    expected = torch.linalg.solve(A, C)
    for perm in (torch.arange(19, device=DEV), torch.randperm(19, device=DEV)):
        m = RidgeMemory(8, 3, 1.0)
        for ids in perm.split(4): m.write(h[ids], y[ids])
        m.solve(); check(torch.allclose(m.W, expected, atol=3e-6), "incremental batch ridge equivalence and order independence")
    m = RidgeMemory(1, 2, 1.0)
    old = torch.ones(1, 1, device=DEV)
    m.write(old, torch.tensor([0], device=DEV)); m.solve()
    before = float((m.scores(old)[0, 0] - 1) ** 2)
    m.write(old, torch.tensor([1], device=DEV)); m.solve()
    after = float((m.scores(old)[0, 0] - 1) ** 2)
    check(after > before, "old class can worsen despite additive statistics")
    yy = torch.randint(0, 3, (15,), device=DEV); hh = torch.randn(15, 8, device=DEV)
    one, two = RidgeMemory(8, 3), RidgeMemory(8, 3)
    one.write(hh, yy); one.solve()
    two.write(hh[:7], yy[:7]); two.write(hh[7:], yy[7:]); two.solve()
    check(one.solves == 1 and torch.allclose(one.G, two.G, atol=3e-6)
          and torch.allclose(one.C, two.C, atol=3e-6)
          and torch.allclose(one.W, two.W, atol=3e-6), "one solve matches two-stage sufficient statistics")
    cfg = dict(DEFAULT, hidden=32, rw_hidden=32, rw_sketch=16, T=2,
               n_stream_classes=8, n_classes=10, n_train_per_class=35, n_test_per_class=10,
               chunk_target=12, chunk_context=6, n_recur=16)
    calib = torch.randn(12, 1, 28, 28, device=DEV)
    rw = build("rewire", cfg, 0, calib)
    check(rw.opt is None and rw.net.H == 32, "the deployed learner has no optimizer")
    check(rw.mem.head == "lda" and rw.skip and rw.d == 16 + min(28 * 28, cfg["rw_skip_max"]), "REWIRE = dual view [sketch ; input view] with the LDA readout")
    check(build("abl_sketch_ridge", cfg, 0, calib).mem.head == "ridge", "the sketch-only ablation keeps the ridge readout")
    check(rw.resident_bytes() > rw.standing_bytes(), "resident memory includes frozen tensors")
    gated = build("rewire_gated", {**cfg, "theta_hi": 0.0, "rw_gate_delta": 100.0}, 0, calib)
    labels0 = torch.zeros(len(calib), dtype=torch.long, device=DEV)
    empty_x, empty_y = calib[:0], labels0[:0]
    first = gated.visit(calib, labels0, empty_x, empty_y, 0, 0, True)
    second = gated.visit(calib, labels0, empty_x, empty_y, 0, 1, False)
    check(first["wrote"] and gated.mem.solves == 1 and second["wrote"] is False
          and second["write_cost_uJ"] == 0.0 and gated.opt is None, "skipped recurrence performs no solve/write/gradient update")
    for ds in ("mnist", "cifar100", "shd"):
        dc = config_for(ds)
        check(dc["n_stream_classes"] == {"mnist": 8, "cifar100": 80, "shd": 16}[ds], f"{ds} stream class count")
        check(all(isinstance(n, int) and n >= 0 for n in dc["sweep_recur"]), f"{ds} recurrence sweep configured")
    check(config_for("shd")["T"] == 20 and config_for("mnist")["T"] == 6, "SHD binned into 20 frames, frames into 6")
    rng = np.random.RandomState(0)
    X = rng.randn(600, 2).astype("float32"); labels = np.repeat(np.arange(10), 60)
    cfg.update(n_recur=16, min_recurrences_per_class=2)
    st = make_stream(X, labels, 0, cfg)
    counts = {c: 0 for c in st["stream_classes"]}
    for v in st["chunks"]:
        if v["kind"] == "recur": counts[v["cls"]] += 1
    check(min(counts.values()) >= 2, "balanced recurrence schedule")
    # ---- the fixes made before the final campaign (each one makes the code do what the paper says)
    st0 = make_stream(X, labels, 0, cfg); ch0 = st0["chunks"][0]; n_ho = max(4, len(ch0["target"]) // 5)
    held = labels[ch0["target"][:n_ho]]
    check(set(held.tolist()) == set(ch0["cls"]) and abs(int((held == ch0["cls"][0]).sum()) - int((held == ch0["cls"][1]).sum())) <= 1,
          "first chunk interleaved: the held-out slice holds both classes equally")
    Xf = torch.rand(24, 1, 28, 28, device=DEV); yf = torch.tensor([3, 5] * 12, device=DEV)
    rwf = build("rewire", cfg, 0, Xf); rwf.visit(Xf, yf, Xf[:0], yf[:0], [3, 5], 0, True)
    hf = rwf.feats(Xf[: max(4, len(yf) // 5)]); ff = rwf._fam(hf); yy = yf[: len(ff)]
    check(abs(rwf.fam_ref[3] - float(ff[yy == 3].mean())) < 1e-5 and abs(rwf.fam_ref[5] - float(ff[yy == 5].mean())) < 1e-5,
          "each class's familiarity reference comes from its own held-out samples")
    from .models.arms import rewire_standing_bytes
    iso = build("replay_iso", cfg, 0, calib); per = iso.bytes_per_sample(calib[0]); nb = iso.buffer_size(calib) * per
    check(rewire_standing_bytes(cfg, 28 * 28) == rw.standing_bytes() and rw.standing_bytes() - per < nb <= rw.standing_bytes(),
          "iso-memory rehearsal holds as many raw samples as fit in REWIRE's standing memory")
    hd = build("hd_memory", cfg, 0, calib)
    check(hd.d == rw.d and hd.skip, "hyperdimensional bundling runs on REWIRE's own dual-view features")
    from .dataloader import apply_shift
    ev = (torch.rand(3, 6, 2, 34, 34) < 0.1).float(); sh = apply_shift(ev, 0, 0, dict(cfg, shift_mode="domain", shift_kind="rotate_nearest"))
    check(set(torch.unique(sh).tolist()) <= {0.0, 1.0} and abs(float(sh.sum()) - float(ev.sum())) < 0.2 * float(ev.sum()),
          "event frames stay binary under the nearest-neighbour rotation and keep their event count (within the cropped corners)")
    sp = torch.zeros(2, 20, 1, 700, 1); sp[:, :, 0, 100, 0] = 1
    s1 = apply_shift(sp, 0, 0, dict(cfg, shift_mode="domain", shift_kind="channel", shift_channels=35))
    check(float(s1.sum()) == float(sp.sum()) and int(s1[0, 0, 0, :, 0].nonzero()[0]) in (65, 135), "spike trains move along the channel axis by the configured channels")
    check(torch.equal(apply_shift(ev, 3, 0, dict(cfg, shift_mode="domain", shift_kind="rotate_nearest")), apply_shift(ev, 7, 0, dict(cfg, shift_mode="domain", shift_kind="rotate_nearest"))),
          "the shift does not depend on the class")
    from .metrics.stats import wilcoxon_exact
    check(abs(wilcoxon_exact(np.linspace(0.5, 3, 30))[0] - 2 / 2 ** 30) < 1e-15 and abs(wilcoxon_exact([1, 2, 3, 4, 5])[0] - 0.0625) < 1e-12,
          "the pooled Wilcoxon test is exact (30 pairs)")
    check(config_for("mnist")["sweep_recur"][-1] == 128 and config_for("shd")["sweep_recur"][-1] == 128 and config_for("cifar100")["sweep_recur"][-1] == 64,
          "long streams: up to 128 recurrence chunks on the 8- and 16-class streams")
    check(config_for("shd")["shift_kind"] == "channel" and config_for("nmnist")["shift_kind"] == "rotate_nearest" and config_for("cifar10")["shift_kind"] == "rotate",
          "a domain shift is defined for every modality")
    from .results_io import load, ORDER
    from .models.arms import PAPER_ARMS
    check(set(config_for("mnist")["arms"]) == set(PAPER_ARMS) and set(PAPER_ARMS) <= set(ORDER), "every scheduled arm is in the registry and the report order")
    with tempfile.TemporaryDirectory() as out:
        from pathlib import Path
        import json
        f = Path(out) / "mnist" / "shift_none"; f.mkdir(parents=True)
        (f / "rows.json").write_text(json.dumps([{"arm": "rewire", "dataset": "mnist", "shift_mode": "none", "seed": 0, "collapsed": True, "final_acc": 0.1}]))
        check(bool(load(out).iloc[0].collapsed), "collapsed run loaded and retained")
    print("PREFLIGHT PASSED", flush=True)


if __name__ == "__main__": main()
