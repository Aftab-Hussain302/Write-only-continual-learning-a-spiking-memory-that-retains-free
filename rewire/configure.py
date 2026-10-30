"""Single source of truth for every knob: the protocol, the learners, the cost model, the stages of the final campaign and
the presets. Every run records the configuration it used in rows.json; `--set key=value` on the command line overrides any key.

Three continual-learning scenarios are run (van de Ven et al. 2022):
  class-incremental with repetition (the main grid, its sweeps and the domain-shifted recurrences; Hemati et al. 2023),
  task-incremental Split-MNIST and domain-incremental Permuted MNIST (the native protocols of DSD-SNN and HLOP; rewire/native.py)."""
DEFAULT = dict(
    # ---- data / protocol (class-incremental learning with repetition, Hemati et al. 2023)
    dataset="mnist", data_root="data", data="torchvision",
    seeds=[0, 1, 2, 3, 4],
    arms=None,                                  # None: every arm of the paper (models.arms.PAPER_ARMS)
    n_classes=10, n_stream_classes=8, n_holdout=2,   # per-dataset overrides in DATASETS below
    n_recur=16, min_recurrences_per_class=0,
    chunk_target=80, chunk_context=40,          # a chunk: 80 samples of the arriving class + 40 of the two most recent other classes
    n_train_per_class=300, n_test_per_class=100,
    recur_gap_min=2,                            # a class recurs only after >= 2 other chunks
    # ---- domain-shifted recurrences: with shift_mode="domain" every recurrence of a class arrives from that class's shifted
    #      domain, and its test set switches with it. The shift itself is defined per modality (DATASETS[...]["shift"]):
    #      "rotate" (images; bilinear, before the frozen front end where there is one), "rotate_nearest" (event frames, which
    #      stay binary), "channel" (spike trains: translation along the tonotopic channel axis, zero-filled). One transform per
    #      seed (the sign of the rotation or translation is drawn from the seed), identical for every class and every method.
    #      The legacy modes "rotate" | "noise" | "mixed" remain for the self-test only.
    shift_mode="none", shift_deg=25.0, shift_channels=35, shift_noise=0.25,
    # ---- shared spiking backbone (the gradient baselines; REWIRE's expansion uses rw_hidden)
    hidden=1024, T=6, beta=0.95, thr=1.0, surrogate_slope=2.0,
    drive_std=1.5,                              # input drive scale folded into the first-layer weights at calibration (LIF operating point)
    # ---- REWIRE: frozen random LIF expansion -> sketch (+ input view) -> analytic memory
    rw_hidden=1024,                             # LIF units of the frozen expansion
    rw_sketch=512,                              # sketch width d_s (the Gram matrix is (d_s + input view)^2)
    rw_lambda=1.0,                              # ridge regularisation (the ridge ablations and controls)
    rw_head="ridge",                            # "lda": the count-free readout (REWIRE); "ridge": one-hot ridge (ablation)
    rw_lda_eps=0.01,                            # LDA shrinkage of the base configuration (REWIRE and the LDA ablations use rw2_eps)
    rw_solver="direct", rw_refresh_every=25,    # "woodbury": recursive update of (G + lambda I)^-1 per written batch (ridge only)
    rw_skip_input=False, rw_skip_max=512, rw_skip_scale=1.0,   # dual view: [spiking sketch ; input view], input projected to <= 512 dims
    rw_fam="cos",                               # familiarity: cosine to the class means, or "maha" (Mahalanobis under the LDA covariance)
    rw_gated=False, rw_gate_delta=0.05, theta_hi=0.85, rw_gate_budget=0,   # the recognition gate: familiarity >= tau_k - delta AND competence >= theta_hi
    rw_select="none", rw_select_keep=0.25,      # selective write on recurrences (the count-bias control): "errors" | "random"
    rw_no_lif=False, rw_raw_ridge=False,        # controls: ReLU random layer instead of LIF; no expansion at all (ridge / SLDA on the inputs)
    rw2_sketch=0, rw2_eps=0.1,                  # REWIRE: sketch width (0 = rw_sketch) and LDA shrinkage. In the final campaign rw2_eps is
                                                # selected by the eps_dev stage (below) before any evaluation run and passed to every unit.
    event_input_cost="spikes",                  # input-layer charge on event frames: one accumulate per input event ("spikes") | dense_once | dense_per_frame
    stem="resnet18", stem_resnet_datasets=["cifar10", "cifar100", "tinyimagenet"],   # frozen ImageNet-pretrained ResNet-18 features (512-d), identical for every method
    stem_resize=224,                            # input size of the frozen ResNet-18 (224 in every reported run)
    conv_stem=True, conv_stem_gray=False, stem_ch=32, stem_k=5, stem_pool=4,        # fixed random conv stem for multi-channel data when stem != resnet18
    # ---- gradient baselines
    lr=0.001, batch=32, full_steps=40,          # 40 surrogate-gradient updates per chunk
    hlop_k=[16, 8], hlop_grow=4, hlop_hebb_lr=0.01, hlop_hebb_iters=5,   # authors' module; the subspace has room for every stream class
    dsd_init=64, dsd_grow=32, dsd_hmax=320, dsd_alpha=2.0, dsd_gamma=0.99, dsd_p0=5.0, dsd_eta=10.0, dsd_rho_p=0.5, dsd_epochs_mult=1,
    ewc_lambda=1000.0, replay_M="per_class", replay_per_class=20,
    # ---- evaluation
    tau=0.80, budget=40, shots=10, n_forgotten_events=3, tau_rel=0.95,
    recovery_budgets=[0, 1, 2, 5, 10],          # few-shot in-context recovery curve (0, 1, 2, 5, 10 updates of `shots` samples)
    # ---- cost model (Horowitz, ISSCC 2014, 45 nm; the Spikformer convention)
    pj_mac=4.6, pj_sop=0.9, backward_factor=2.0,
    # ---- sweeps of the class-incremental protocol
    sweep_recur=[8, 16, 32],                    # recurrence density (presets below extend it)
    sweep_theta=[0.75, 0.85, 0.95],             # the gate's competence threshold (seeds 0-2)
    # ---- the arms of each stage of the final campaign (scripts/run_all.py)
    shift_arms=["rewire", "rewire_gated", "rewire_gated_maha", "ctrl_slda_input", "hlop", "replay", "naive"],
    native_arms=["naive", "ewc", "hlop", "dsd_snn", "rewire", "ctrl_slda_input"],
    native_protocols=["split", "pmnist"],       # task-incremental Split-MNIST (5 tasks x 2 classes, masked head) and domain-incremental
    native_tasks_pmnist=10,                     # Permuted MNIST (the unpermuted task + 9 permutations, one 10-way head)
    native_epochs=3, native_n_per_class=300, native_n_test=100,
    # ---- the LDA shrinkage, selected once on a development stream that shares no seed with the evaluation (rule fixed in advance:
    #      highest final accuracy on the development stream; a tie within 0.1 points goes to the value closest to 0.1)
    eps_grid=[0.01, 0.03, 0.1, 0.3, 1.0], eps_dev_dataset="cifar100", eps_dev_seed=100,
    # ---- io
    out="results",
)

PRESETS = dict(
    # fair     : the 8- and 16-class streams (MNIST, CIFAR-10, N-MNIST, SHD, SSC): 24 recurrence chunks, 500 train / 200 test per class;
    #            recurrence density up to 128 recurrence chunks (long streams)
    # fair_big : the 80-class streams (CIFAR-100, Tiny-ImageNet): 16 recurrence chunks, per-class counts from DATASETS
    fair=dict(n_recur=24, n_train_per_class=500, n_test_per_class=200, sweep_recur=[8, 16, 32, 64, 128]),
    fair_big=dict(n_recur=16, n_train_per_class=500, n_test_per_class=100, sweep_recur=[8, 16, 32, 64]),
)
PRESET_OF = dict(mnist="fair", cifar10="fair", nmnist="fair", shd="fair", ssc="fair", cifar100="fair_big", tinyimagenet="fair_big")
PAPER_DATASETS = ["mnist", "nmnist", "shd", "cifar10", "cifar100", "tinyimagenet"]   # SSC is optional: add "ssc" to --datasets
DATASET_EXTRA = dict(shd={"T": 20}, ssc={"T": 20})   # spike trains are binned into 20 frames

# Per-dataset protocol: class counts, stream split, per-class sample counts (these win over the preset) and the domain shift.
DATASETS = dict(mnist=dict(shape=(1, 28, 28), shift="rotate"),
                synthetic=dict(shape=(1, 28, 28), shift="rotate"),     # synthetic: the self-test's toy set, never a substitute
                cifar10=dict(shape=(3, 32, 32), shift="rotate"),
                nmnist=dict(shape=(2, 34, 34), shift="rotate_nearest"),
                shd=dict(shape=(1, 700, 1), n_classes=20, n_stream_classes=16, n_holdout=4, n_train_per_class=300, n_test_per_class=100,
                         shift="channel", max_time=1.4),
                ssc=dict(shape=(1, 700, 1), n_classes=35, n_stream_classes=28, n_holdout=7, n_train_per_class=300, n_test_per_class=100,
                         shift="channel", max_time=1.0),
                cifar100=dict(shape=(3, 32, 32), n_classes=100, n_stream_classes=80, n_holdout=20, n_train_per_class=300, n_test_per_class=100,
                              shift="rotate"),
                tinyimagenet=dict(shape=(3, 32, 32), n_classes=200, n_stream_classes=80, n_holdout=20, n_train_per_class=400, n_test_per_class=100,
                                  shift="rotate"))
DATASET_NAME = dict(mnist="MNIST", cifar10="CIFAR-10", nmnist="N-MNIST", shd="SHD", ssc="SSC", cifar100="CIFAR-100", tinyimagenet="Tiny-ImageNet",
                    synthetic="Synthetic")
INPUT_KIND = dict(mnist="pixels", nmnist="events", shd="spikes", ssc="spikes", cifar10="ResNet features", cifar100="ResNet features",
                  tinyimagenet="ResNet features", synthetic="pixels")


def apply_dataset_defaults(cfg, name):
    """Per-dataset protocol overrides from DATASETS, then explicit --set values."""
    cfg = dict(cfg); cfg["dataset"] = name
    for k, v in DATASETS.get(name, {}).items():
        if k == "shape": continue
        cfg["shift_kind" if k == "shift" else k] = v
    for k, v in cfg.get("_overrides", {}).items(): cfg[k] = v
    return cfg


def config_for(dataset, overrides=None, preset=None):
    """The full configuration of one dataset: DEFAULT + its preset + its dataset table + the dataset's extra settings + overrides."""
    cfg = dict(DEFAULT); cfg.update(PRESETS[preset or PRESET_OF.get(dataset, "fair")]); cfg.update(DATASET_EXTRA.get(dataset, {}))
    cfg["_overrides"] = dict(overrides or {}); cfg.update(cfg["_overrides"])
    cfg = apply_dataset_defaults(cfg, dataset)
    if cfg.get("arms") is None:
        from .models.arms import PAPER_ARMS
        cfg["arms"] = list(PAPER_ARMS)
    return cfg
