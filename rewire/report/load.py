"""Load one or more finished run folders (the tree written by scripts/merge_results.py) into one Campaign object. Nothing is run and
nothing is downloaded: only the saved records are read.

    <run>/<dataset>/shift_none/rows.json        main grid          <run>/<dataset>/shift_domain/rows.json   domain-shifted recurrences
    <run>/<dataset>/sweep_recur/n<R>/rows.json  density, long      <run>/<dataset>/sweep_gate/t<theta>/rows.json   gate operating curve
    <run>/<dataset>/eps_sens/e<eps>/rows.json   shrinkage sweep    <run>/eps/dev/e<eps>/rows.json, <run>/eps_selected.json
    <run>/native/native_rows.json               TIL / DIL          <run>/hardware/akida*.json                  the Akida measurement
    <run>/MANIFEST.json, ENVIRONMENT.md, status.json, determinism.json"""
import glob, json, os, re
from pathlib import Path
from .style import DS_ORDER

DROP = ("per_class_curve", "steps_curve")       # not needed by any paper figure or table; dropped to keep memory small


def _read(path):
    rows = json.load(open(path))
    for r in rows:
        for k in DROP: r.pop(k, None)
        cfg = r.get("config") or {}
        r["_synthetic"] = bool(cfg.get("synthetic_data", False)); r["_eps"] = cfg.get("rw2_eps")
        r["_n_stream"] = cfg.get("n_stream_classes"); r["_n_classes"] = cfg.get("n_classes"); r["_cfg"] = {k: cfg.get(k) for k in (
            "n_recur", "n_stream_classes", "n_holdout", "n_classes", "chunk_target", "chunk_context", "rw2_eps", "theta_hi", "shift_kind",
            "shift_deg", "shift_channels", "full_steps", "hidden", "rw_hidden", "rw_sketch", "T", "n_train_per_class", "n_test_per_class",
            "rw_gate_delta", "recovery_budgets", "batch", "lr")}
        r.pop("config", None)
    return rows


class Campaign:
    def __init__(self, runs, allow_synthetic=False):
        self.runs = [Path(r) for r in runs]
        self.main, self.shift, self.sweep, self.gate, self.eps_sens, self.eps_dev, self.native, self.akida = [], [], [], [], [], [], [], []
        self.meta = dict(runs=[str(r) for r in self.runs], manifests=[], environment=[], status=[], determinism=[], eps_selected=None)
        for run in self.runs:
            for f in sorted(glob.glob(str(run / "*" / "shift_none" / "rows.json"))): self.main += _read(f)
            for f in sorted(glob.glob(str(run / "*" / "shift_domain" / "rows.json"))): self.shift += _read(f)
            for f in sorted(glob.glob(str(run / "*" / "sweep_recur" / "n*" / "rows.json"))): self.sweep += _read(f)
            for f in sorted(glob.glob(str(run / "*" / "sweep_gate" / "t*" / "rows.json"))): self.gate += _read(f)
            for f in sorted(glob.glob(str(run / "*" / "eps_sens" / "e*" / "rows.json"))): self.eps_sens += _read(f)
            for f in sorted(glob.glob(str(run / "eps" / "dev" / "e*" / "rows.json"))): self.eps_dev += _read(f)
            nf = run / "native" / "native_rows.json"
            if nf.exists(): self.native += json.load(open(nf))
            for f in sorted(glob.glob(str(run / "hardware" / "akida*.json"))) + sorted(glob.glob(str(run / "akida*.json"))):
                try: self.akida.append(json.load(open(f)))
                except Exception: pass
            for name, key in (("MANIFEST.json", "manifests"), ("status.json", "status"), ("determinism.json", "determinism")):
                p = run / name
                if p.exists():
                    try: self.meta[key].append(json.loads(p.read_text()))
                    except Exception: pass
            if (run / "ENVIRONMENT.md").exists(): self.meta["environment"].append((run / "ENVIRONMENT.md").read_text())
            for p in (run / "eps_selected.json", run / "eps" / "eps_selected.json"):
                if p.exists() and self.meta["eps_selected"] is None: self.meta["eps_selected"] = json.loads(p.read_text())
        synth = any(r["_synthetic"] for r in self.main + self.shift + self.sweep + self.gate + self.eps_sens) or any(r.get("synthetic_data") for r in self.native)
        self.synthetic = synth
        if synth and not allow_synthetic:
            raise SystemExit("These records come from the synthetic pipeline test (REWIRE_SYNTHETIC_DATA=1), not from the datasets. "
                             "Refusing to build paper figures from them; pass --allow-synthetic only to test the tools.")
        if not self.main: raise SystemExit(f"no main-grid rows.json found under {', '.join(map(str, self.runs))}")
        present = {r["dataset"] for r in self.main}
        self.datasets = [d for d in DS_ORDER if d in present] + sorted(present - set(DS_ORDER))
        self.arms = sorted({r["arm"] for r in self.main})
        self.seeds = sorted({r["seed"] for r in self.main})
        # the sweep's default point (n_recur of the main grid) and the gate's default theta come from the main grid itself
        self.sweep_default = [r for r in self.main]
        self.gate_default = [r for r in self.main]

    def rows(self, which="main", dataset=None, arm=None):
        rs = getattr(self, which)
        return [r for r in rs if (dataset is None or r["dataset"] == dataset) and (arm is None or r["arm"] == arm)]

    def summary(self):
        return dict(main=len(self.main), shift=len(self.shift), sweep=len(self.sweep), gate=len(self.gate), eps_sens=len(self.eps_sens),
                    eps_dev=len(self.eps_dev), native=len(self.native), akida=len(self.akida), datasets=self.datasets, seeds=self.seeds,
                    arms=len(self.arms), synthetic=self.synthetic)
