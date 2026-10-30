"""Loader of a finished run's rows.json files into one flat table (used by the preflight and the reporting tools)."""
import glob, json
import pandas as pd
from .models.arms import ARMS

ORDER = list(ARMS)
SCALARS = ("dataset", "shift_mode", "arm", "seed", "n_recur", "theta_hi", "final_acc", "bwt", "bwt_recur", "forgetting", "recog_auroc_fam",
           "recog_auroc_fam_x1", "fam_auroc_test", "retention_mean", "recur_written_share", "final_acc_shifted", "final_acc_unshifted",
           "first_shift_written_share", "later_shift_written_share", "retention_first_shift", "retention_later_shift",
           "lifetime_energy_uJ", "frontend_energy_uJ", "total_energy_uJ", "standing_bytes", "resident_bytes", "resident_bytes_total", "optimizer_bytes",
           "number_of_solves", "hidden_width", "sketch_width", "feature_dim", "collapsed", "S_ctrl", "S_AUR", "AUR_seen", "AUR_novel", "E_relearn", "E_novel",
           "censored", "RIF", "energy_per_acc_point", "train_time_s")


def load(res, pattern="*/shift_*/rows.json"):
    rows = []
    for f in sorted(glob.glob(f"{res}/{pattern}")):
        for r in json.load(open(f)): rows.append({k: r.get(k) for k in SCALARS})
    d = pd.DataFrame(rows)
    if len(d): d["collapsed"] = d.collapsed.fillna(False).astype(bool)
    return d
