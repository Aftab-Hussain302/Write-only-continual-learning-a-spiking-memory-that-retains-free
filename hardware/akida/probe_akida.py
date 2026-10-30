#!/usr/bin/env python3
"""Probe the AKD1000 with REWIRE's converted expansion before the measurement (about a minute; no dataset needed).

    python3 hardware/akida/probe_akida.py 2>&1 | tee probe_report.txt

Builds the converted network exactly as run_akida.py does (rewire_akida.akida_model), with random weights at the sizes of SHD
(700 -> 1024 -> 512) and N-MNIST (2312 -> 1024 -> 512), maps it on the first Akida device, runs random 4-bit frames through the chip
and through MetaTF's software model, and reports whether the outputs agree, where each layer runs and what the power meter returns."""
import os, platform, sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def step(name, fn):
    t0 = time.time()
    try:
        out = fn(); print(f"[ok]   {name} ({time.time() - t0:.1f} s)", flush=True); return out
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        tb = traceback.format_exc().strip().splitlines(); print("       " + (tb[-3].strip() if len(tb) > 2 else ""), flush=True); return None


def main():
    print(f"python {platform.python_version()} on {platform.machine()}")
    for mod in ("numpy", "tensorflow", "quantizeml", "cnn2snn", "akida", "torch", "h5py"):
        try: print(f"   {mod:11s} {__import__(mod).__version__}")
        except Exception as e: print(f"   {mod:11s} not importable ({type(e).__name__})")
    try: print("   board: " + open("/proc/device-tree/model").read().strip("\x00\n"))
    except Exception: pass
    import akida
    import rewire_akida as RA
    devs = step("akida.devices()", lambda: akida.devices())
    if not devs: print("no Akida device: check the driver (lspci shows the card; 'akida devices' should list it)"); return
    dev = devs[0]; print(f"   device: {getattr(dev, 'desc', dev)}; version {getattr(dev, 'version', '?')}")
    for name, d_in in (("SHD-size", 700), ("N-MNIST-size", 2312)):
        print(f"\n== {name} network: {d_in} -> 1024 -> 512")
        rng = np.random.RandomState(0); H, d_s = 1024, 512
        E = dict(W=(rng.randn(H, d_in) / np.sqrt(d_in)).astype(np.float32) * 3, b=(rng.randn(H) * 0.1).astype(np.float32),
                 S=(rng.randn(H, d_s) / np.sqrt(H)).astype(np.float32), Sb=(rng.randn(d_s) * 0.1).astype(np.float32))
        Xq, sc = RA.to_input(rng.poisson(1.0, (2000, d_in)).astype(np.float32))
        m = step("quantise and convert for Akida 1.0", lambda: RA.akida_model(E, Xq[:80], sc))
        if m is None: continue
        sw = step("software model, 2000 frames", lambda: RA.akida_features(m, Xq, None))
        hw = step("chip, 2000 frames x 3 runs (power measurement on)", lambda: RA.akida_features(m, Xq, dev, repeats=3))
        if sw is None or hw is None: continue
        a, c = sw[0], hw[0]; meas = hw[1]
        print(f"   chip output == software model: {np.mean(a == c) * 100:.2f}% of values equal; max |diff| {np.abs(a - c).max():.4g}")
        print(f"   where it ran: {meas.get('sequences')}")
        for k in ("floor_mW", "avg_power_mW", "fps", "latency_ms", "energy_per_frame_uJ", "net_energy_per_frame_uJ", "power_error"):
            if k in meas: print(f"   {k}: {meas[k]}")
        for t in meas.get("statistics", []): print("   " + t.replace("\n", " | "))
    print("\nPROBE DONE")


if __name__ == "__main__": main()
