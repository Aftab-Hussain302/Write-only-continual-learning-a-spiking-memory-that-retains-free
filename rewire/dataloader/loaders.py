import glob, math, os, numpy as np, torch, torch.nn.functional as F
from ..configure import DATASETS


EVENT = ("nmnist",)
SPIKES = ("shd", "ssc")


def synthetic_enabled():
    """REWIRE_SYNTHETIC_DATA=1 replaces every dataset by a synthetic stand-in of the same shape and class count. It exists only to
    test the pipeline end to end on a machine without the data; rows produced with it are marked (config["synthetic_data"]) and the
    figure script refuses them unless explicitly told otherwise."""
    return os.environ.get("REWIRE_SYNTHETIC_DATA") == "1"


def read_nmnist_bin(path):
    """Parse one N-MNIST .bin file. Five bytes per event: x, y, then one polarity bit and a 23 bit timestamp.
    Written out here so the pipeline needs no event-data library."""
    raw = np.fromfile(path, dtype=np.uint8)
    n = len(raw) // 5; raw = raw[: n * 5].reshape(n, 5).astype(np.uint32)
    x = raw[:, 0]; y = raw[:, 1]; p = (raw[:, 2] >> 7) & 1
    t = ((raw[:, 2] & 0x7F) << 16) | (raw[:, 3] << 8) | raw[:, 4]
    return x, y, p, t


def nmnist_frames(path, T, size=(34, 34)):
    """Events of one sample -> (T, 2, H, W) frames, binned into T equal time windows and clipped to [0,1]."""
    x, y, p, t = read_nmnist_bin(path)
    F_ = np.zeros((T, 2, size[0], size[1]), np.float32)
    if len(t) == 0: return F_
    b = np.clip(((t - t.min()) * T) // max(1, int(t.max() - t.min()) + 1), 0, T - 1)
    ok = (x < size[1]) & (y < size[0])
    np.add.at(F_, (b[ok], p[ok], y[ok], x[ok]), 1.0)
    return np.clip(F_, 0, 1)


def load_nmnist_local(root, cfg, per_class=None):
    """N-MNIST from an unzipped folder, no external library. Expects <root>/Train/<class>/*.bin (the layout of the official archive);
    <root>/nmnist/Train/... and <root>/Test/... are also accepted."""
    cands = [os.path.join(root, p) for p in ("Train", "nmnist/Train", "NMNIST/Train", "Test", "nmnist/Test")]
    base = next((c for c in cands if os.path.isdir(c)), None)
    if base is None: raise FileNotFoundError(f"no N-MNIST Train/ folder under {root}")
    per_class = per_class or (cfg["n_train_per_class"] + cfg["n_test_per_class"])
    Xs, ys = [], []
    for c in range(10):
        files = sorted(glob.glob(os.path.join(base, str(c), "*.bin")))[:per_class]
        if not files: raise FileNotFoundError(f"no .bin files for class {c} under {base}")
        for f in files: Xs.append(nmnist_frames(f, cfg["T"])); ys.append(c)
    X = np.stack(Xs); y = np.asarray(ys, np.int64)
    print(f"[data] N-MNIST from {base}: {len(y)} samples, frames {X.shape[1:]}")
    return X, y


def load_event_dataset(name, cfg, root="data", per_class=None):
    """N-MNIST, parsed directly from the official .bin files (no event-data library). No silent fallback: a run that quietly
    substituted MNIST for N-MNIST would duplicate a dataset in the results."""
    if name != "nmnist": raise ValueError(f"unknown event dataset {name}")
    last = None
    for r in (root, os.path.join(root, "nmnist")):
        try: return load_nmnist_local(r, cfg, per_class)
        except FileNotFoundError as e: last = e
    raise FileNotFoundError(f"N-MNIST not found under {root} (expected <root>/nmnist/Train/<digit>/*.bin): {last}")


def load_tinyimagenet(root, cfg, size=32, per_class=None):
    """Tiny-ImageNet-200 from the official folder layout (train/<wnid>/images/*.JPEG), read at `size` x `size` (64, the native size,
    when the frozen ResNet-18 front end is used). Only per_class images per class are materialised."""
    from PIL import Image
    base = os.path.join(root, "tiny-imagenet-200"); wnids = open(os.path.join(base, "wnids.txt")).read().split(); idx = {w: i for i, w in enumerate(wnids)}
    per_class = per_class or (cfg["n_train_per_class"] + cfg["n_test_per_class"])
    files = [(f, idx[w]) for w in wnids for f in sorted(glob.glob(os.path.join(base, "train", w, "images", "*.JPEG")))[:per_class]]
    # The images stay uint8 (N, 3, size, size): 100k images at 64x64 are 1.2 GB. Consumers scale by 1/255 per batch
    # (_resnet18_embed, shifted_features, the conv-stem path in engine.run_grid), so the values they see are unchanged.
    X = np.empty((len(files), 3, size, size), np.uint8); y = np.empty(len(files), np.int64)
    for i, (f, c) in enumerate(files):
        im = Image.open(f).convert("RGB")
        if im.size != (size, size): im = im.resize((size, size), Image.BILINEAR)
        X[i] = np.asarray(im, dtype=np.uint8).transpose(2, 0, 1); y[i] = c
    print(f"[data] Tiny-ImageNet-200 from {base}: {len(y)} samples at {size}x{size}, {len(wnids)} classes (uint8, {X.nbytes/1e9:.1f} GB)"); return X, y


def as_unit_float(x):
    """uint8 images -> float in [0, 1] (what every consumer expects); float inputs pass through unchanged."""
    if isinstance(x, np.ndarray): return x.astype(np.float32) / 255.0 if x.dtype == np.uint8 else x
    return x.float() / 255.0 if x.dtype == torch.uint8 else x


def _resnet18_embed(X, dev, net=None, size=224):
    """Frozen ImageNet ResNet-18 (global-pool, 512-d) features of X (N, C, H, W; float in [0, 1] or uint8), 256 images at a time.
    Pass `net` (from _resnet18_net) to reuse one loaded network over several calls."""
    net = net or _resnet18_net(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    F_ = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            x = as_unit_float(torch.as_tensor(np.asarray(X[i:i + 256]), device=dev)).float()   # uint8 batches are scaled here, never the whole set
            if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
            x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            F_.append(net((x - mean) / std).float().cpu())
    return torch.cat(F_).numpy().astype(np.float32)


def _resnet18_net(dev):
    import torchvision
    random_stem = os.environ.get("REWIRE_TEST_RANDOM_STEM") == "1"                 # tests only: no download, random weights
    net = torchvision.models.resnet18(weights=None if random_stem else torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity(); return net.eval().to(dev)


def _cache_tag(cfg):
    tag = "" if int(cfg.get("stem_resize", 224)) == 224 else f"_r{int(cfg['stem_resize'])}"
    if synthetic_enabled() or os.environ.get("REWIRE_TEST_RANDOM_STEM") == "1": tag += "_TEST"
    return tag


def shifted_features(Xraw, F0, seed, cfg, name, root):
    """Features of the seed's shifted images for every row, cached; plus a lookup from an unshifted feature row to its index."""
    kind = cfg.get("shift_kind", "rotate") if cfg["shift_mode"] == "domain" else cfg["shift_mode"]
    cache = os.path.join(root, f"_features_resnet18_{name}_{len(Xraw)}_{kind}{float(cfg['shift_deg']):g}_s{seed}{_cache_tag(cfg)}.npy")
    if os.path.exists(cache): F1 = np.load(cache)
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"; net = _resnet18_net(dev); Fs = []
        for i in range(0, len(Xraw), 2048):                                       # shift (CPU, 2048 images per draw) and embed chunk by
            xb = as_unit_float(torch.tensor(np.asarray(Xraw[i:i + 2048])))        # chunk: the shifted float32 set is never materialised in full
            Fs.append(_resnet18_embed(apply_shift(xb, -1, seed, cfg).numpy(), dev, net, int(cfg.get("stem_resize", 224))))
        F1 = np.concatenate(Fs); mu, sd = cfg["_feat_stats"]; F1 = (F1 - mu) / sd
        tmp = cache + f".tmp{os.getpid()}.npy"; np.save(tmp, F1); os.replace(tmp, cache); print(f"[data] shifted features ({kind}, seed {seed}) for {name}: {F1.shape} (cached)")
    lut = {hash(F0[i].tobytes()): i for i in range(len(F0))}
    return lut, torch.tensor(F1)


def pretrained_features(X, cfg, name, root):
    """Frozen ImageNet-pretrained ResNet-18 features (512-d, global pool), applied identically to every method as the shared
    input; the standard practice of analytic continual learning (ACIL, RanPAC) for natural-image benchmarks. Cached to disk after
    the first pass. Applied to CIFAR-10, CIFAR-100 and Tiny-ImageNet alike (cfg["stem_resnet_datasets"]); Tiny-ImageNet overlaps
    the front end's pretraining data, which the paper states and anchors its 80-class claims on CIFAR-100. Cost is charged once
    per sample as a fixed front end (stem_macs), identically for every method."""
    cache = os.path.join(root, f"_features_resnet18_{name}_{len(X)}{_cache_tag(cfg)}.npz")
    if os.path.exists(cache):
        z = np.load(cache); print(f"[data] pretrained ResNet-18 features from cache: {z['F'].shape}"); return z["F"], float(z["macs"]), (z["mu"], z["sd"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    F_ = _resnet18_embed(X, dev, size=int(cfg.get("stem_resize", 224)))
    rng = np.random.RandomState(1234); nrm = np.zeros(len(F_), bool); nrm[rng.choice(len(F_), max(64, len(F_) // 10), replace=False)] = True
    mu, sd = F_[nrm].mean(0), F_[nrm].std(0) + 1e-6; F_ = (F_ - mu) / sd          # inductive: statistics from a held-out normalisation
    F_[nrm] = np.nan                                                               # subset that is excluded from every stream and test set
    macs = 1.8e9                                                                   # ResNet-18 at 224x224, per sample (the reported front-end charge)
    tmp = cache + f".tmp{os.getpid()}.npz"; np.savez(tmp, F=F_, macs=macs, mu=mu, sd=sd); os.replace(tmp, cache); print(f"[data] pretrained ResNet-18 features for {name}: {F_.shape} (cached)"); return F_, macs, (mu, sd)


def load_spike_dataset(name, root, cfg):
    """Spiking Heidelberg Digits (SHD; Cramer et al., 2020) or Spiking Speech Commands (SSC): spoken words as 700-channel spike trains,
    no front end. Reads the official <name>_train.h5 (h5py) and bins spike times into T frames of 700 channels over the dataset's
    recording window (DATASETS[name]["max_time"]: 1.4 s for SHD, 1.0 s for SSC), counts clipped to 1, as (N, T, 1, 700, 1)."""
    import h5py
    p = next((q for q in (os.path.join(root, name, f"{name}_train.h5"), os.path.join(root, f"{name}_train.h5")) if os.path.exists(q)), None)
    if p is None: raise FileNotFoundError(f"{name}_train.h5 not found under {root}/{name} (download from https://zenkelab.org/datasets/)")
    T = cfg["T"]; per = cfg["n_train_per_class"] + cfg["n_test_per_class"]; K = DATASETS[name]["n_classes"]; tmax = float(DATASETS[name]["max_time"])
    with h5py.File(p, "r") as f:
        times, units, labels = f["spikes"]["times"], f["spikes"]["units"], np.asarray(f["labels"])
        keep = []
        for c in range(K): keep += list(np.where(labels == c)[0][:per])
        keep = sorted(keep); X = np.zeros((len(keep), T, 1, 700, 1), np.float32); y = labels[keep].astype(np.int64)
        for i, j in enumerate(keep):
            t = np.asarray(times[j]); u = np.asarray(units[j]).astype(np.int64); b = np.minimum((t / tmax * T).astype(np.int64), T - 1)
            np.add.at(X[i, :, 0, :, 0], (b, u), 1.0)
    X = np.clip(X, 0, 1); print(f"[data] {name.upper()} from {p}: {len(y)} samples, frames {X.shape[1:]}"); return X, y


def load_shd(root, cfg): return load_spike_dataset("shd", root, cfg)


def load_dataset(name="mnist", mode="torchvision", root="data", cfg=None):
    """X (N,C,H,W) float in [0,1] (uint8 for Tiny-ImageNet) or (N,T,C,H,W) frames for event and spike data; y (N,) int.
    MNIST, CIFAR-10 and CIFAR-100 are read by torchvision from the standard folders under `root` (downloaded there if absent and
    the machine is online); N-MNIST, SHD, SSC and Tiny-ImageNet are read from the layouts described in DATASETS.md. A dataset that
    cannot be loaded raises; nothing is ever substituted (the synthetic stand-ins exist only under REWIRE_SYNTHETIC_DATA=1)."""
    cfg = cfg or {}
    if name == "synthetic": return synthetic_dataset()
    if synthetic_enabled(): return synthetic_standin(name, cfg)
    if name in SPIKES: return load_spike_dataset(name, root, {"T": 20, "n_train_per_class": 300, "n_test_per_class": 100, **cfg})
    if name == "tinyimagenet":
        size = 64 if cfg.get("stem", "resnet18") == "resnet18" else 32
        return load_tinyimagenet(root, {"n_train_per_class": 400, "n_test_per_class": 100, **cfg}, size=size)
    if name in EVENT: return load_event_dataset(name, {"T": 6, "n_train_per_class": 300, "n_test_per_class": 100, **cfg}, root)
    if name not in ("mnist", "cifar10", "cifar100"): raise ValueError(f"unknown dataset {name}")
    from torchvision import datasets
    folder = {"mnist": "MNIST", "cifar10": "cifar-10-batches-py", "cifar100": "cifar-100-python"}[name]
    dl = not os.path.isdir(os.path.join(root, folder))
    cls = dict(mnist=datasets.MNIST, cifar10=datasets.CIFAR10, cifar100=datasets.CIFAR100)[name]
    ds = cls(root, train=True, download=dl)
    X = np.asarray(ds.data, dtype=np.float32) / 255.0; y = np.asarray(ds.targets, dtype=np.int64)
    if name in ("cifar10", "cifar100"): X = X.transpose(0, 3, 1, 2)
    X = X.reshape(-1, *DATASETS[name]["shape"]); print(f"[data] {name}: {len(y)} samples"); return X, y


def synthetic_dataset(n_per_class=400, shape=(1, 28, 28)):
    """A deterministic 10-class toy image set (noisy binary templates) for the self-test and the launcher's dry run. It is only ever
    loaded under the explicit name "synthetic" and is never substituted for a real dataset."""
    rng = np.random.RandomState(123); C, H, W = shape; tpl = (rng.rand(10, C, H, W) < 0.15).astype(np.float32)
    X = np.zeros((10 * n_per_class, C, H, W), np.float32); y = np.zeros(10 * n_per_class, np.int64)
    for c in range(10):
        X[c * n_per_class:(c + 1) * n_per_class] = np.abs(np.repeat(tpl[c][None], n_per_class, 0) - (rng.rand(n_per_class, C, H, W) < 0.02 + 0.01 * c).astype(np.float32)); y[c * n_per_class:(c + 1) * n_per_class] = c
    print(f"[data] synthetic 10-class toy set {shape}"); return X, y


def synthetic_standin(name, cfg):
    """A synthetic stand-in with the shape, dtype and class count of the real dataset `name` (pipeline tests only). Classes are
    noisy versions of smooth random prototypes, with enough overlap that no method is perfect; event and spike stand-ins are
    Bernoulli frames whose rates follow the class prototype."""
    print(f"[data] !!! SYNTHETIC STAND-IN for {name} (REWIRE_SYNTHETIC_DATA=1): not a real dataset, pipeline test only")
    K = int(DATASETS[name].get("n_classes", 10)); n = int(cfg.get("n_train_per_class", 300)) + int(cfg.get("n_test_per_class", 100))
    rng = np.random.RandomState(4321 + K)
    if name in SPIKES or name in EVENT:
        T = int(cfg.get("T", 20 if name in SPIKES else 6)); shp = (1, 700, 1) if name in SPIKES else (2, 34, 34)
        D = int(np.prod(shp)); base = rng.rand(K, D) ** 4 * 0.25; shared = rng.rand(D) ** 4 * 0.25
        X = np.zeros((K * n, T) + shp, np.float32); y = np.repeat(np.arange(K), n)
        for c in range(K):
            p = 0.55 * base[c] + 0.45 * shared
            for t in range(T):
                X[c * n:(c + 1) * n, t] = (rng.rand(n, D) < p * (0.6 + 0.8 * t / max(1, T - 1))).reshape((n,) + shp)
        return X, y.astype(np.int64)
    shp = (1, 28, 28) if name == "mnist" else ((3, 64, 64) if name == "tinyimagenet" else (3, 32, 32))
    C, H, W = shp; low = torch.nn.functional.interpolate(torch.tensor(rng.rand(K, C, 6, 6), dtype=torch.float32), size=(H, W), mode="bilinear", align_corners=False).numpy()
    shared = low.mean(0, keepdims=True)
    X = np.empty((K * n, C, H, W), np.float32); y = np.repeat(np.arange(K), n)
    for c in range(K):
        proto = 0.5 * low[c] + 0.5 * shared[0]
        X[c * n:(c + 1) * n] = np.clip(proto[None] + 0.28 * rng.randn(n, C, H, W), 0, 1)
    if name == "tinyimagenet": X = (X * 255).astype(np.uint8)
    return X, y.astype(np.int64)


def _fold(X):
    """(N,T,C,H,W) -> (N*T,C,H,W) plus an unfold function, so spatial ops work for static and event inputs alike."""
    if X.dim() == 5: return X.reshape(-1, *X.shape[2:]), (lambda Z, s=X.shape: Z.reshape(s[0], s[1], *Z.shape[1:]))
    return X, (lambda Z: Z)


def stem_transform(X, cfg, seed=0):
    """Fixed random convolutional stem for multi-channel data when the ResNet front end is not used (stem != "resnet18"), applied
    ONCE to the whole dataset and shared by every arm. Random and frozen: no trainable parameters, no standing memory (regenerated
    from the seed), only a fixed per-sample cost."""
    Xf, unfold = _fold(X)
    if not cfg.get("conv_stem", True): return X, 0
    if Xf.shape[1] == 1 and not cfg.get("conv_stem_gray", True): return X, 0
    X = Xf; g = torch.Generator().manual_seed(12345 + seed); C = X.shape[1]; ch, k, p = cfg["stem_ch"], cfg["stem_k"], cfg["stem_pool"]
    W = (torch.randn(ch, C, k, k, generator=g) / (C * k * k) ** 0.5)
    out = []
    for i in range(0, len(X), 512):
        h = F.conv2d(X[i:i + 512], W, stride=2, padding=k // 2); h = F.avg_pool2d(F.relu(h), p); out.append(h)
    Z = torch.cat(out); Zf = Z.flatten(1)
    Zf = (Zf - Zf.mean(0, keepdim=True)) / (Zf.std(0, keepdim=True) + 1e-6)     # per-feature standardisation: a centred drive keeps the
    Z = Zf.reshape(Z.shape)                                                     # LIF layer in its responsive range for every arm
    macs = ch * C * k * k * (X.shape[-1] // 2) ** 2
    print(f"[data] frozen conv stem: {tuple(X.shape[1:])} -> {tuple(Z.shape[1:])} ({Z[0].numel()} features/step, {macs/1e3:.0f}k MAC/step)")
    return unfold(Z), macs


def rotate(X, deg, mode="bilinear"):
    """Rotation of every frame by `deg` degrees about the centre (zero padding). mode="nearest" keeps binary event frames binary."""
    if len(X) == 0: return X
    X, unfold = _fold(X)
    th = math.radians(deg)
    m = torch.tensor([[math.cos(th), -math.sin(th), 0.0], [math.sin(th), math.cos(th), 0.0]], dtype=X.dtype, device=X.device)[None].repeat(len(X), 1, 1)
    return unfold(F.grid_sample(X, F.affine_grid(m, X.shape, align_corners=False), mode=mode, align_corners=False))


def channel_shift(X, s):
    """Translation of spike trains by s channels along the channel axis (dim -2 of (..., 700, 1)), zero-filled: the cochlear
    channels are ordered by frequency, so this is the spike-train analogue of a pitch or speaker change."""
    s = int(s); Y = torch.zeros_like(X)
    if s > 0: Y[..., s:, :] = X[..., :-s, :]
    elif s < 0: Y[..., :s, :] = X[..., -s:, :]
    else: Y = X.clone()
    return Y


def shift_kind(cfg):
    return cfg.get("shift_kind", "rotate") if cfg["shift_mode"] == "domain" else cfg["shift_mode"]


def apply_shift(X, cls, seed, cfg):
    """Domain shift: ONE transform per seed, identical for every class, so the transform carries no class information (an angle drawn
    per class would turn the rotation itself into a class cue).
    Feature inputs (2-D rows from a frozen front end) are mapped to the features of the shifted image through cfg["_shift_lut"].
    With shift_mode="domain" the transform is the dataset's own (configure.DATASETS[...]["shift"]): rotation of images (bilinear),
    rotation of event frames (nearest neighbour, frames stay binary) or translation of spike trains along the channel axis."""
    mode = cfg["shift_mode"]
    if mode == "none" or len(X) == 0: return X
    if X.dim() == 2 and "_shift_lut" in cfg:
        lut, F1 = cfg["_shift_lut"], cfg["_shifted"]; rows = X.detach().cpu().numpy()
        idx = [lut[hash(r.tobytes())] for r in rows]; return F1[idx].to(X.device)
    g = np.random.RandomState(seed); sign = float(g.choice([-1, 1])); kind = shift_kind(cfg)
    if kind == "rotate_nearest": return rotate(X, sign * cfg["shift_deg"], mode="nearest")
    if kind == "channel": return channel_shift(X, sign * cfg["shift_channels"])
    if kind in ("rotate", "mixed"): X = rotate(X, sign * cfg["shift_deg"])
    if kind in ("noise", "mixed"):                                                 # legacy (self-test only): not a physical shift for spikes
        gen = torch.Generator(device="cpu").manual_seed(seed)
        X = (X + cfg["shift_noise"] * torch.randn(X.shape, generator=gen).to(X.device)).clamp(0, 1)
    return X


def make_stream(X, y, seed, cfg):
    """Class-incremental-with-repetition stream (Hemati et al. 2023 semantics).
    First chunk introduces two classes together (their samples interleaved, so that the held-out slice holds both), then one new
    class per chunk, then n_recur recurrence chunks of seen classes (a class recurs only after >= recur_gap_min other chunks;
    longest-absent classes are favoured). Every single-class chunk carries chunk_context samples of the two most recent OTHER
    classes (identical for every arm). With shift_mode != none, every recurrence of a class comes from that class's shifted domain,
    and the class's test set switches to the shifted domain from its first shifted recurrence on (domain drift inside a class)."""
    rng = np.random.RandomState(1000 + seed); K = cfg["n_classes"]
    perm = rng.permutation(K); S = perm[:cfg["n_stream_classes"]].tolist(); H = perm[cfg["n_stream_classes"]:cfg["n_stream_classes"] + cfg["n_holdout"]].tolist()
    pool, test = {}, {}
    n_avail = min(len(np.where(y == c)[0]) for c in range(K)); n_te = min(cfg["n_test_per_class"], n_avail // 3); n_tr = min(cfg["n_train_per_class"], n_avail - n_te)
    if n_tr < cfg["n_train_per_class"]: print(f"[data] small dataset: {n_tr} train / {n_te} test per class")
    cfg = dict(cfg); cfg["chunk_target"] = min(cfg["chunk_target"], n_tr // 2); cfg["chunk_context"] = min(cfg["chunk_context"], n_tr // 4)
    for c in range(K):
        idx = np.where(y == c)[0]; rng.shuffle(idx); pool[c] = idx[:n_tr]; test[c] = idx[n_tr:n_tr + n_te]
    order = list(rng.permutation(S)); pair = (order[0], order[1]); seq = [(pair, "intro")] + [(c, "intro") for c in order[2:]]
    last_seen = {c: (0 if c in pair else order.index(c) - 1) for c in order}
    recurrence_counts = {c: 0 for c in S}
    flat = lambda c: list(c) if isinstance(c, tuple) else [c]
    for t in range(len(seq), len(seq) + cfg["n_recur"]):
        recent = [c2 for c, _ in seq[-cfg["recur_gap_min"]:] for c2 in flat(c)]
        cand = [c for c in S if c not in recent]; absent = np.array([t - last_seen[c] for c in cand], float)
        needed = [c for c in cand if recurrence_counts[c] < cfg.get("min_recurrences_per_class", 0)]
        if needed:
            low = min(recurrence_counts[c] for c in needed)
            cand = [c for c in needed if recurrence_counts[c] == low]
            absent = np.array([t - last_seen[c] for c in cand], float)
        c = int(rng.choice(cand, p=absent / absent.sum())); seq.append((c, "recur")); last_seen[c] = t
        recurrence_counts[c] += 1
    if min(recurrence_counts.values()) < cfg.get("min_recurrences_per_class", 0):
        raise ValueError("n_recur is insufficient to meet min_recurrences_per_class with this gap")
    chunks = []
    for i, (c, kind) in enumerate(seq):
        others = []
        for cc, _ in reversed(seq[:i]):
            for c2 in flat(cc):
                if c2 not in flat(c) and c2 not in others: others.append(c2)
            if len(others) == 2: break
        others = others[:2]
        if isinstance(c, tuple):
            parts = [rng.choice(pool[cc], cfg["chunk_target"] // 2, replace=False) for cc in c]
            tgt = np.stack(parts, 1).reshape(-1)                          # interleaved a, b, a, b, ...: the held-out slice holds both classes
            chunks.append(dict(t=i, cls=list(c), kind=kind, target=tgt, context=np.zeros(0, int), context_classes=[], shifted=False)); continue
        tgt = rng.choice(pool[c], cfg["chunk_target"], replace=False)
        ctx = [rng.choice(pool[o], cfg["chunk_context"] // max(1, len(others)), replace=False) for o in others]
        chunks.append(dict(t=i, cls=int(c), kind=kind, target=tgt, context=np.concatenate(ctx) if ctx else np.zeros(0, int), context_classes=others,
                           shifted=(kind == "recur" and cfg["shift_mode"] != "none")))
    return dict(stream_classes=S, holdout=H, order=order, chunks=chunks, pool=pool, test=test)
