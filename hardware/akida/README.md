# REWIRE's expansion on BrainChip Akida (inference only)

`run_akida.py` measures the part of REWIRE that a neuromorphic processor would run, the frozen random expansion and its sketch,
on a BrainChip Akida device, and writes `<run>/hardware/akida_<dataset>.json` into the campaign's run folder, where
`scripts/make_paper_figures.py` reads it (Extended Data Fig. 9, the Supplementary Table of the hardware measurement and the named
values `hw.*` and `akida.*`). Two optional files in the same folder are read as well: `probe_report.txt`, the printed report of
`probe_akida.py` (named values `hw.probe.*`), and `photos/node.jpg` and `photos/card.jpg`, the photographs of panels a and b of the figure.

Nothing is trained on the chip or anywhere else. The chip computes features; the host writes them into REWIRE's sums and solves the
count-free discriminant exactly as in the campaign. Because the memory is a sum (Proposition 1), the final readout of an ungated
stream equals the readout fitted to the multiset of every sample the stream writes, so each stream is written once. The script checks
this against the campaign's own REWIRE run of the same seed (`acc_lif` against `acc_lif_campaign`); on the synthetic test streams the
two agree exactly for REWIRE, the sketch-only discriminant and the sketch-only ridge learner.

## What runs where

| part | where |
|---|---|
| time-averaged event or spike frame, as unsigned 4-bit input (Akida 1.0 limit for a FullyConnected input layer) | host, once per sample |
| expansion: the REWIRE layer's random afferent weights and biases (after its calibration), 4-bit weights, rectifier bounded at the 99.5th percentile of its output on the first chunk, 4-bit activations | chip |
| sketch: the same random map, 4-bit weights, bounded rectifier, 4-bit activations | chip |
| input view, dual view, write into (G, C, n), solve of the discriminant | host |

Akida 1.0 (AKD1000) executes quantised layers whose units emit graded events; it has no leaky integrate-and-fire dynamics. The
expansion is therefore converted, not copied, and the accuracy is reported three ways on the same streams: with the LIF layer
(`acc_lif`, REWIRE as in the campaign), with the converted network in MetaTF's software model (`acc_sim`; also in a NumPy emulation,
`acc_numpy`) and with the features computed on the chip (`acc_chip`).

## Energy

With the device mapped, MetaTF's power meter gives the idle floor (`floor_mW`), the energy per frame of a run (`energy_per_frame_uJ`)
and the frame rate; `net_energy_per_frame_uJ` subtracts floor power x frame time. MetaTF 2.10 reports the energy per frame with the floor included (akida/statistics.py). `modelled_uJ` is the
paper's operation-count model of the same network (one accumulate per input event and per non-zero hidden activation, per
postsynaptic unit, at 0.9 pJ), and `measured_over_modelled` their ratio.

## Running it

Tested with MetaTF 2.10 (akida 2.10.0, cnn2snn 2.10.0, quantizeml 0.12.1, TensorFlow 2.15.1, Python 3.11): quantisation, conversion
for Akida 1.0 and the software model on synthetic data, and the mapping of both network sizes (700 and 2,312 inputs) entirely onto
hardware on MetaTF's virtual AKD1000 (one sequence, one pass). The node used for the paper is a Raspberry Pi Compute Module 4 (8 GB) on
its IO board with an AKD1000 PCIe card (aarch64, Raspberry Pi OS), with these versions installed for the system Python 3.11. Akida 1.0 constraints met by `rewire_akida.akida_model`: 1-, 2- or
4-bit weights in every FullyConnected layer, the first included; inputs of at most 4 bits; bounded rectifiers; an output quantiser on
the last rectifier.

    python3 hardware/akida/extract_for_akida.py --src <uploaded archives> --data_root ~/data   # on a node with little disk: only the files read
    python3 hardware/akida/probe_akida.py 2>&1 | tee probe_report.txt     # one minute, no data: map, run, compare with software, power
    python3 hardware/akida/run_akida.py --data_root ~/data --run results/results_<date> --datasets shd nmnist --backend hw

Without a board, `--backend sw` runs MetaTF's software model only (no energy), and `--backend numpy` tests the whole pipeline with a
NumPy emulation of the converted network (`REWIRE_SYNTHETIC_DATA=1` for synthetic data). MetaTF reports the statistics of the last
forward call only (frames per second; power Avg/Min/Max/Std in mW; energy = average power x duration / frames, idle floor included),
so `run_akida.py` sends every frame of a seed through one call, `--repeats` times (default 3), and reports the means; the energy above
the floor is (average power - floor) / frame rate.
