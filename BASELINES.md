# Baselines, controls and attribution

Every learner runs on the same stream, the same front end, the same 1,024-unit width and the same cost model
(`rewire/configure.py`). The arm names below are the ones used in the code (`rewire/models/arms.py`, `ARMS`).

* `hlop`, `hlop_adaptive`, `hlop_gated`: Xiao, Meng, Zhang, He, Lin, "Hebbian Learning based Orthogonal Projection for Continual
  Learning of Spiking Neural Networks", ICLR 2024 (github.com/pkuxmq/HLOP-SNN). `rewire/baselines/hlop_official/hlop_module.py`
  is the authors' Hebbian module, unmodified, Apache License 2.0 (LICENSE and NOTICE included). The published method is task- and
  domain-incremental; its use on a class-incremental stream with repetition is our adaptation: subspace capacity for every class;
  `hlop` freezes the consolidated subspace as published; `hlop_adaptive` refreshes it on recurrences; `hlop_gated` skips the update
  of a recurrence when its own held-out competence reaches theta (the competence half of REWIRE's gate, as a control).
* `replay`, `replay_iso`, `replay_gated`: rehearsal with a reservoir buffer (Chaudhry et al., "On Tiny Episodic Memories in
  Continual Learning", 2019): one reservoir of 20 samples per stream class, filled by reservoir sampling from every sample of every
  chunk and therefore not class-balanced; `replay_iso` holds as many raw samples as fit in REWIRE's standing memory
  (sized from REWIRE's own feature dimension, `rewire_standing_bytes`); `replay_gated` is the competence-gated control.
* `ewc`: Kirkpatrick et al., PNAS 2017, with a running-mean diagonal Fisher.
* `naive`: fine-tuning without protection, the unprotected reference.
* `dsd_snn`: Han et al., IJCAI 2023, our re-implementation of dynamic structure development; it runs on the native protocols only
  (`rewire/native.py`), where the gradient baselines are checked against their own protocols before the campaign's comparison.
* `ctrl_slda_input`: streaming LDA on the inputs (Hayes & Kanan, "Lifelong Machine Learning with Deep Streaming Linear
  Discriminant Analysis", CVPR-W 2020): REWIRE's readout on the raw input, without the spiking layer and the sketch.
* `ridge_input`: ridge regression on the inputs held as sufficient statistics (the analytic class-incremental recipe of Zhuang et
  al., ACIL, NeurIPS 2022).
* `ranpac`: a RanPAC-style control (McDonnell et al., NeurIPS 2023): a rectifier random layer and the random sketch with the ridge
  readout, no spiking layer. `ctrl_relu` is the same substitution inside REWIRE's full configuration (dual view, discriminant).
* `hd_memory`: write-once hyperdimensional bundling (Kanerva, Cognitive Computation 2009) on REWIRE's own dual-view features.
* Energy coefficients: Horowitz, ISSCC 2014 (45 nm); the accounting convention of Zhou et al., Spikformer, ICLR 2023.
* Class-incremental learning with repetition: Hemati et al., CoLLAs 2023.
* The frozen spiking expansion with a linear readout follows the liquid-state-machine / reservoir tradition (Maass, Natschlaeger,
  Markram 2002; Jaeger 2001); the dual view (input beside the reservoir state) is standard echo-state-network practice.
