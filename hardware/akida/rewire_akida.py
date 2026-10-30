"""REWIRE's frozen expansion on a BrainChip Akida processor: inference only (nothing is trained on the chip or anywhere else).

The chip computes the plastic-free part of the learner, the random expansion and its sketch; the host does what REWIRE does with
any features: it writes them into the sums (G, C, n) and solves the count-free discriminant (rewire.models.RidgeMemory). Because the
memory is a sum (Proposition 1), the final readout of an ungated stream equals the readout fitted to the multiset of every sample
the stream writes, so the final accuracy is computed by writing that multiset once.

The Akida 1.0 hardware (AKD1000) executes quantised layers whose units emit graded events; it has no leaky integrate-and-fire
dynamics. The expansion is therefore converted, not copied: the same random afferent weights and biases (calibrated as in REWIRE)
in a dense layer with the chip's quantised rectifier in place of the LIF rate, applied to the time-averaged frame, then the same
random sketch. Three accuracies are reported on the same streams: with the LIF layer (REWIRE as in the campaign), with the
converted network in a NumPy emulation and in MetaTF's software model (acc_sim), and with the features computed on the chip
(acc_chip). Energy is measured on the chip with MetaTF's power meter (idle floor, energy per frame) and compared with the
operation-count model of the paper for the same network (one accumulate per input event and per non-zero activation, per
postsynaptic unit, at E_AC).

Backends: "numpy" (always available; the converted network with uniform 4-bit activations, used to test the pipeline), "sw"
(MetaTF software model; needs akida, cnn2snn, quantizeml), "hw" (mapped on the first Akida device found)."""
import math, re, time
import numpy as np, torch

E_AC_PJ = 0.9          # pJ per accumulate at 45 nm (the paper's energy model)


# ------------------------------------------------------------------------------------------------------------------ data and stream
def prepare(dataset, data_root, seed, overrides=None):
    """The configuration, the data and the stream of one seed, exactly as the campaign builds them (no front end: pixels, events,
    spikes only)."""
    from rewire.configure import config_for
    from rewire.dataloader import load_dataset, make_stream
    cfg = config_for(dataset, dict(overrides or {})); cfg.setdefault("shift_mode", "none")
    X, y = load_dataset(dataset, cfg["data"], data_root, cfg)
    if X.ndim == 4 and X.shape[-3] > 1: raise SystemExit(f"{dataset}: the hardware measurement covers inputs without a front end (pixels, events, spikes)")
    st = make_stream(X, y, seed, cfg)
    return cfg, X, y, st


def flat_input(X):
    """The chip's input: the time-averaged frame for event and spike data (N, T, C, H, W), the flattened image otherwise."""
    X = np.asarray(X, dtype=np.float32)
    return X.reshape(X.shape[0], X.shape[1], -1).mean(1) if X.ndim == 5 else X.reshape(X.shape[0], -1)


def written_multiset(stream, cfg):
    """Indices and labels of every sample an ungated REWIRE writes along the stream (targets after the held-out slice, plus the
    context samples), with their multiplicity."""
    idx, lab = [], []
    for ch in stream["chunks"]:
        tgt = np.asarray(ch["target"]); cls = ch["cls"]
        hold = max(4, len(tgt) // 5)                                 # the held-out slice, as REWIRE.visit takes it
        if isinstance(cls, list):                                    # first chunk: two classes interleaved
            yt = np.array([cls[i % 2] for i in range(len(tgt))])
        else: yt = np.full(len(tgt), cls)
        idx += list(tgt[hold:]); lab += list(yt[hold:])
        ctx = np.asarray(ch["context"]); occ = ch.get("context_classes", [])
        if len(ctx):
            per = len(ctx) // max(1, len(occ)); yc = np.concatenate([np.full(per, c) for c in occ])[:len(ctx)]
            idx += list(ctx); lab += list(yc)
    return np.asarray(idx, int), np.asarray(lab, int)


def test_set(stream):
    S = stream["stream_classes"]
    idx = np.concatenate([stream["test"][c] for c in S]); return idx, S


# ------------------------------------------------------------------------------------------------------------------ the reference learner
def reference(cfg, seed, X, stream):
    """REWIRE (as in the campaign) built for this seed; its random weights are the ones the chip runs."""
    from rewire.models import build
    from rewire.modules import DEV
    torch.manual_seed(seed); np.random.seed(seed)                    # as rewire.engine.run_one: the expansion's weights follow the seed
    calib = torch.tensor(np.asarray(X)[stream["chunks"][0]["target"]], device=DEV)
    return build("rewire", cfg, seed, calib)


def expansion(arm):
    """The frozen random weights of the expansion and the sketch, after REWIRE's calibration, as NumPy arrays."""
    W = arm.net.fc1.weight.detach().cpu().numpy() * float(arm.net.gain); b = arm.net.fc1.bias.detach().cpu().numpy() * float(arm.net.gain)
    S = arm.S.detach().cpu().numpy(); Sb = arm.Sb.detach().cpu().numpy()
    P = None if arm.P is None else arm.P.detach().cpu().numpy()
    return dict(W=W, b=b, S=S, Sb=Sb, P=P)


def input_view(arm, Xf):
    v = Xf if arm.P is None else Xf @ arm.P.detach().cpu().numpy()
    return v.astype(np.float32)


def lda_accuracy(H, y, idx_w, y_w, idx_t, S, cfg):
    """Write the multiset (idx_w, y_w) of features H into REWIRE's memory, solve the count-free discriminant once and return the final
    accuracy on the test samples of the stream classes (prediction restricted to the stream classes)."""
    from rewire.models import RidgeMemory
    from rewire.modules import DEV
    Ht = torch.tensor(H, device=DEV, dtype=torch.float32)
    mem = RidgeMemory(H.shape[1], cfg["n_classes"], cfg["rw_lambda"], head="lda", lda_eps=float(cfg.get("rw2_eps", 0.1)))
    for i in range(0, len(idx_w), 4096):
        mem.write(Ht[idx_w[i:i + 4096]], torch.tensor(y_w[i:i + 4096], device=DEV))
    mem.solve()
    sc = Ht[idx_t] @ mem.W + mem.b
    mask = torch.full((cfg["n_classes"],), -1e9, device=DEV); mask[list(S)] = 0
    pred = (sc + mask).argmax(1).cpu().numpy()
    return float((pred == y[idx_t]).mean())


def dual(r, v, calib):
    """The dual view h = [g r ; v] with REWIRE's one label-free gain from the calibration batch."""
    g = float(v[calib].std() / (r[calib].std() + 1e-6))
    return np.concatenate([g * r, v], 1)


# ------------------------------------------------------------------------------------------------------------------ the converted network
INPUT_BITS = 4          # Akida 1.0 runs a FullyConnected input layer on 1-, 2- or 4-bit inputs (MetaTF 2.10 rejects values above 15)


def to_input(Xf, bits=INPUT_BITS):
    """Inputs as the chip takes them: unsigned `bits`-bit integers (stored as uint8). Event counts, spike counts and pixels are
    non-negative; one scale for the whole dataset, from its 99.9th percentile."""
    top = 2 ** bits - 1
    scale = top / max(float(np.percentile(Xf, 99.9)), 1e-6)
    return np.clip(np.round(Xf * scale), 0, top).astype(np.uint8), scale


def quant(x, bits, vmax):
    q = (2 ** bits) - 1
    return np.clip(np.round(x / vmax * q), 0, q) * (vmax / q)


def numpy_chip(E, Xu8, scale, bits=4, w_bits=4):
    """NumPy emulation of the converted network: 4-bit weights (per output unit), rectifier activations quantised to `bits` with
    a per-layer range from the data. Returns the sketch features and the count of non-zero activations per frame (for the model)."""
    def qw(W):
        m = np.abs(W).max(1, keepdims=True) + 1e-12; q = 2 ** (w_bits - 1) - 1
        return np.round(W / m * q) * (m / q)
    x = Xu8.astype(np.float32) / scale
    a1 = np.maximum(x @ qw(E["W"]).T + E["b"], 0); v1 = float(np.percentile(a1, 99.5)) or 1.0; a1 = quant(a1, bits, v1)
    a2 = np.maximum(a1 @ qw(E["S"].T).T + E["Sb"], 0); v2 = float(np.percentile(a2, 99.5)) or 1.0; a2 = quant(a2, bits, v2)
    nnz = dict(input=float((Xu8 > 0).sum(1).mean()), hidden=float((a1 > 0).sum(1).mean()))
    return a2.astype(np.float32), nnz


def _keras():
    """Keras 2 as quantizeml uses it: tf_keras where installed (TensorFlow >= 2.16), else TensorFlow's own Keras."""
    try:
        import tf_keras as keras
    except ImportError:
        from tensorflow import keras
    return keras


def ranges(E, x):
    """The rectifiers' ranges, from the data as in numpy_chip: the 99.5th percentile of each layer's rectified output."""
    a1 = np.maximum(x @ E["W"].T + E["b"], 0); v1 = float(np.percentile(a1, 99.5)) or 1.0
    a2 = np.maximum(np.minimum(a1, v1) @ E["S"] + E["Sb"], 0); v2 = float(np.percentile(a2, 99.5)) or 1.0
    return v1, v2


def keras_model(E, v1, v2):
    """The converted network as a Keras model: dense -> bounded rectifier -> dense -> bounded rectifier, on (1, 1, d_in) inputs (Akida
    1.0 runs a dense layer as FullyConnected on a 1 x 1 x d_in frame and needs bounded rectifiers)."""
    keras = _keras()
    d_in, H, d_s = E["W"].shape[1], E["W"].shape[0], E["S"].shape[1]
    inp = keras.layers.Input((1, 1, d_in))
    l1 = keras.layers.Dense(H, name="expansion"); x = l1(inp); x = keras.layers.ReLU(max_value=v1, name="rate")(x)
    l2 = keras.layers.Dense(d_s, name="sketch"); x = l2(x); x = keras.layers.ReLU(max_value=v2, name="sketch_relu")(x)
    m = keras.Model(inp, x); l1.set_weights([E["W"].T, E["b"]]); l2.set_weights([E["S"], E["Sb"]])
    return m


def akida_model(E, calib_u8, scale, act_bits=4, weight_bits=4):
    """Quantise (quantizeml) and convert (cnn2snn) the network for Akida 1.0 (AKD1000). The input layer takes the 4-bit frame (to_input);
    its scale is folded into the first layer's weights. Akida 1.0 constraints (checked with MetaTF 2.10 / quantizeml 0.12): 1-, 2- or 4-bit
    weights in every FullyConnected layer, the first included; bounded rectifiers; an output quantiser on the last rectifier."""
    from quantizeml.models import quantize, QuantizationParams
    from cnn2snn import convert, set_akida_version, AkidaVersion
    E2 = dict(E); E2["W"] = E["W"] / scale                           # the network sees uint8 inputs directly
    xc = calib_u8.reshape(len(calib_u8), -1).astype(np.float32)
    v1, v2 = ranges(E2, xc)
    km = keras_model(E2, v1, v2)
    qp = QuantizationParams(input_weight_bits=weight_bits, weight_bits=weight_bits, activation_bits=act_bits)
    qc = {"sketch_relu": {"output_quantizer": {"bitwidth": act_bits, "signed": False, "axis": "per-tensor", "buffer_bitwidth": 32}}}
    with set_akida_version(AkidaVersion.v1):
        qm = quantize(km, q_config=qc, qparams=qp, samples=xc.reshape(len(xc), 1, 1, -1), num_samples=len(xc), epochs=1)
        return convert(qm)


def akida_features(model, Xu8, device=None, repeats=1):
    """Features of the converted network computed by MetaTF's software model (device None) or on the chip (model mapped to device).
    On the chip, power measurement is enabled and the run's statistics are returned with the features. MetaTF reports the statistics of
    the last forward call (akida.statistics: fps; powers Avg/Min/Max/Std in mW over that call; energy = Avg power x duration / frames, in
    mJ per frame, the idle floor included), so all frames go through one call, repeated `repeats` times for a longer power trace."""
    x = np.ascontiguousarray(Xu8.reshape(len(Xu8), 1, 1, -1))
    meas = {}
    if device is not None:
        model.map(device)
        meas["sequences"] = [f"{getattr(s, 'name', '?')}: {getattr(s, 'backend', '?')}" for s in getattr(model, "sequences", [])]
        try:
            device.soc.power_measurement_enabled = True; time.sleep(1.0)            # MetaTF's own CLI waits 1 s before reading the floor
            meas["floor_mW"] = float(device.soc.power_meter.floor)
        except Exception as e:
            meas["floor_mW"] = float("nan"); meas["power_error"] = f"{type(e).__name__}: {e}"
    t0 = time.time(); F = np.asarray(model.forward(x), dtype=np.float32).reshape(len(x), -1); dt = time.time() - t0
    if device is not None:
        runs = []
        for _ in range(max(1, repeats)):
            model.forward(x); st = model.statistics
            runs.append(dict(fps=st.fps, energy_mJ=st.energy, powers=dict(st.powers or {}), text=str(st)))
        meas["statistics"] = [r["text"] for r in runs]
        fps = [r["fps"] for r in runs if r["fps"]]; en = [r["energy_mJ"] for r in runs if r["energy_mJ"]]
        pw = [r["powers"].get("Avg") for r in runs if r["powers"].get("Avg") is not None]
        meas["fps"] = float(np.mean(fps)) if fps else len(x) / dt
        meas["energy_per_frame_uJ"] = 1000.0 * float(np.mean(en)) if en else float("nan")
        meas["avg_power_mW"] = float(np.mean(pw)) if pw else float("nan")
        meas["latency_ms"] = 1000.0 / meas["fps"] if meas["fps"] else float("nan")
        if math.isfinite(meas["avg_power_mW"]) and math.isfinite(meas["floor_mW"]) and meas["fps"]:
            meas["net_energy_per_frame_uJ"] = 1000.0 * (meas["avg_power_mW"] - meas["floor_mW"]) / meas["fps"]   # mW / fps = mJ per frame
        else: meas["net_energy_per_frame_uJ"] = float("nan")
        model.map(None)
    else: meas["fps_software"] = len(x) / dt
    return F, meas


def modelled_energy_uJ(nnz, H, d_s):
    """The paper's operation-count model for the converted network: one accumulate per input event and per non-zero hidden activation,
    per postsynaptic unit, at E_AC."""
    return (nnz["input"] * H + nnz["hidden"] * d_s) * E_AC_PJ * 1e-6
