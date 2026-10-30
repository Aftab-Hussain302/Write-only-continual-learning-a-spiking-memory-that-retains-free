"""The properties the paper relies on, as executable checks (CPU, seconds):  python3 -m pytest tests -q
  Proposition 1  order invariance: the memory after any ordering of the writes equals batch fitting on the union
  Proposition 2  count-free discriminant: writing a class m times leaves its mean unchanged, so with the shared precision held fixed
                 the discriminant scores are identical; the ridge readout changes with m
  the LDA solve matches a NumPy reference; the recursive (Woodbury) ridge solve matches the direct one; a skipped recurrence costs
  no write; the exact Wilcoxon test matches brute-force enumeration; the domain shifts; the native protocols for a write-only arm."""
import itertools, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rewire.models.arms import RidgeMemory, build
from rewire.configure import DEFAULT
from rewire.modules import DEV

torch.manual_seed(0)
d, K, n = 12, 4, 60
H = torch.randn(n, d, device=DEV); Y = torch.randint(0, K, (n,), device=DEV)


def _fit(head, order, chunks=4, **kw):
    m = RidgeMemory(d, K, 1.0, head=head, **kw)
    for ids in order.split(n // chunks): m.write(H[ids], Y[ids])
    m.solve(); return m


def test_order_invariance_ridge_and_lda():
    for head in ("ridge", "lda"):
        a = _fit(head, torch.arange(n, device=DEV)); b = _fit(head, torch.randperm(n, device=DEV)); c = _fit(head, torch.randperm(n, device=DEV), chunks=1)
        for m in (b, c):
            assert torch.allclose(a.W, m.W, atol=1e-5) and torch.allclose(a.b, m.b, atol=1e-5), head


def test_count_free_discriminant():
    """Writing class 0 three times over (as a recurring class is): the class means are unchanged, so with the shared precision P held
    fixed the discriminant scores s_k(h) = h'P mu_k - mu_k'P mu_k / 2 are identical (Proposition 2); the counts enter only P. The ridge
    readout instead drifts towards the over-written class (its one-hot targets act as a prior)."""
    dup = torch.cat([torch.arange(n, device=DEV), torch.where(Y == 0)[0], torch.where(Y == 0)[0]])
    a = _fit("lda", torch.arange(n, device=DEV), chunks=1, keep_inverse=True); b = _fit("lda", dup, chunks=1, keep_inverse=True)
    Ma, Mb = a.C / a.n.clamp(min=1)[None], b.C / b.n.clamp(min=1)[None]
    assert torch.allclose(Ma, Mb, atol=1e-5), "class means are count-free"
    P = a.Lam
    sa = H @ (P @ Ma) - 0.5 * (Ma * (P @ Ma)).sum(0); sb = H @ (P @ Mb) - 0.5 * (Mb * (P @ Mb)).sum(0)
    assert torch.allclose(sa, sb, atol=1e-4), "with P held fixed, the scores do not depend on the counts"
    share = {}
    for head in ("ridge", "lda"):
        x = _fit(head, torch.arange(n, device=DEV), chunks=1); y_ = _fit(head, dup, chunks=1)
        share[head] = float((y_.scores(H).argmax(1) == 0).float().mean() - (x.scores(H).argmax(1) == 0).float().mean())
    assert share["ridge"] > 0, "the ridge readout drifts towards the over-written class"
    assert abs(share["lda"]) < share["ridge"], f"the LDA readout must drift less than ridge ({share})"


def test_lda_matches_numpy():
    m = _fit("lda", torch.arange(n, device=DEV), chunks=1, lda_eps=0.1)
    Hn, Yn = H.cpu().numpy().astype(np.float64), Y.cpu().numpy()
    mu = np.stack([Hn[Yn == k].mean(0) if (Yn == k).any() else np.zeros(d) for k in range(K)], 1)
    Sw = sum(((Hn[Yn == k] - mu[:, k]).T @ (Hn[Yn == k] - mu[:, k])) for k in range(K) if (Yn == k).any())
    Sigma = Sw / (n - len(set(Yn.tolist()))); A = Sigma + 0.1 * np.trace(Sigma) / d * np.eye(d)
    W = np.linalg.solve(A, mu); b = -0.5 * (mu * W).sum(0)
    assert np.allclose(m.W.cpu().numpy(), W, atol=1e-4) and np.allclose(m.b.cpu().numpy(), b, atol=1e-4)


def test_woodbury_matches_direct():
    a = _fit("ridge", torch.arange(n, device=DEV)); w = _fit("ridge", torch.arange(n, device=DEV), solver="woodbury", refresh_every=0)
    assert torch.allclose(a.W, w.W, atol=1e-4)


def _small_cfg(**kw):
    return dict(DEFAULT, hidden=32, rw_hidden=32, rw_sketch=16, T=2, n_classes=10, n_stream_classes=8, n_holdout=2, n_train_per_class=35, n_test_per_class=10,
                chunk_target=12, chunk_context=6, n_recur=16, stem="conv", conv_stem=False, **kw)


def test_write_only_and_gate():
    cfg = _small_cfg()
    calib = torch.rand(12, 1, 28, 28, device=DEV); y0 = torch.zeros(12, dtype=torch.long, device=DEV); e = calib[:0]; ey = y0[:0]
    rw = build("rewire", cfg, 0, calib)
    assert rw.opt is None and rw.mem.head == "lda" and rw.skip, "REWIRE: no optimizer, LDA readout, dual view"
    first = rw.visit(calib, y0, e, ey, 0, 0, True); assert first["wrote"] and rw.mem.solves == 1 and first["steps"] == 0
    g = build("rewire_gated", {**cfg, "theta_hi": 0.0, "rw_gate_delta": 100.0}, 0, calib)       # a gate that always recognises
    g.visit(calib, y0, e, ey, 0, 0, True); second = g.visit(calib, y0, e, ey, 0, 1, False)
    assert second["wrote"] is False and second["write_cost_uJ"] == 0.0 and g.mem.solves == 1, "a recognised recurrence is probed, not written"
    b = build("rewire_gated_b128", {**cfg, "theta_hi": 0.0, "rw_gate_delta": 100.0}, 0, calib)  # the budgeted gate writes until 128 samples are banked
    b.visit(calib, y0, e, ey, 0, 0, True); third = b.visit(calib, y0, e, ey, 0, 1, False)
    assert third["wrote"] is True, "below the budget the recurrence is written"


def test_exact_wilcoxon_matches_enumeration():
    from rewire.metrics.stats import wilcoxon_exact
    rng = np.random.RandomState(1)
    for _ in range(80):
        x = np.round(rng.randn(rng.randint(1, 11)) * 2) / 2; x = x[x != 0]
        if not len(x): continue
        a = np.abs(x); order = np.argsort(a, kind="mergesort"); r = np.empty(len(x)); r[order] = np.arange(1, len(x) + 1)
        for v in np.unique(a): r[a == v] = r[a == v].mean()
        W = r[x > 0].sum(); T = r.sum()
        p = np.mean([(sum(ri for ri, s in zip(r, sg) if s) >= max(W, T - W) - 1e-9) or (sum(ri for ri, s in zip(r, sg) if s) <= min(W, T - W) + 1e-9)
                     for sg in itertools.product([0, 1], repeat=len(x))])
        assert abs(wilcoxon_exact(x)[0] - p) < 1e-12


def test_domain_shifts():
    from rewire.dataloader import apply_shift
    cfg = _small_cfg(shift_mode="domain")
    img = torch.rand(4, 1, 28, 28); rot = apply_shift(img, 1, 0, dict(cfg, shift_kind="rotate"))
    assert rot.shape == img.shape and not torch.allclose(rot, img)
    ev = (torch.rand(2, 6, 2, 34, 34) < 0.1).float(); r2 = apply_shift(ev, 1, 0, dict(cfg, shift_kind="rotate_nearest"))
    assert set(torch.unique(r2).tolist()) <= {0.0, 1.0}
    sp = (torch.rand(2, 20, 1, 700, 1) < 0.05).float(); s2 = apply_shift(sp, 1, 0, dict(cfg, shift_kind="channel", shift_channels=35))
    assert float(s2.sum()) <= float(sp.sum()) and (torch.roll(sp, 35, -2)[..., 35:, :] == s2[..., 35:, :]).all() or (torch.roll(sp, -35, -2)[..., :-35, :] == s2[..., :-35, :]).all()


def test_native_protocols_write_only_and_gradient():
    from rewire.native import run_one_native
    cfg = _small_cfg(native_tasks_pmnist=3, batch=8)
    rng = np.random.RandomState(0); X = torch.tensor(rng.rand(10 * 30, 1, 28, 28).astype("float32"), device=DEV); y = torch.tensor(np.repeat(np.arange(10), 30), device=DEV)
    tr = {c: np.where(y.cpu().numpy() == c)[0][:20] for c in range(10)}; te = {c: np.where(y.cpu().numpy() == c)[0][20:] for c in range(10)}
    for arm in ("rewire", "naive"):
        for protocol in ("split", "pmnist"):
            row = run_one_native(arm, cfg, 0, protocol, X, y, tr, te, epochs=1, log=lambda *a: None)
            A = np.array(row["A"]); assert A.shape == (row["n_tasks"], row["n_tasks"]) and np.isfinite(np.diag(A)).all() and row["lifetime_energy_uJ"] > 0
            assert row["write_only"] == (arm == "rewire")
