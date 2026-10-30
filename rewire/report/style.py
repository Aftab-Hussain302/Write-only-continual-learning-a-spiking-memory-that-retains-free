"""One visual language for every figure of the paper: Nature Computational Science sizes (88 mm single column, 180 mm double column,
5-7 pt sans-serif text, 8 pt bold lowercase panel letters, lines 0.5-1 pt), Okabe-Ito colours (colour-vision safe; no red/green
pair), one colour, marker and label per method taken from this table, no transparency (EPS cannot hold alpha), no rotated tick
labels, TrueType fonts embedded (fonttype 42) so that the text stays editable and searchable."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

MM = 1 / 25.4
W1, W2 = 88 * MM, 180 * MM                     # single and double column widths (inches)

OKABE = dict(blue="#0072B2", green="#009E73", vermillion="#D55E00", orange="#E69F00", sky="#56B4E9", purple="#CC79A7",
             yellow="#F0E442", black="#000000", grey="#7F7F7F")
SHADE = "#EEEEEC"                               # opaque light grey for shaded spans (no alpha)
GRID = "#E3E3E3"
TEXT = "#333333"

# arm -> label, colour, marker, filled. The paper's names; REWIRE blue, its gate bluish green, HLOP vermillion,
# rehearsal orange, fine-tuning grey (the colours of the diagrams).
ARM_STYLE = {
    "rewire":            ("REWIRE (ours)", OKABE["blue"], "*", True),
    "rewire_gated":      ("REWIRE-gated (ours)", OKABE["green"], "P", True),
    "rewire_gated_maha": ("REWIRE-gated, Mahalanobis", OKABE["green"], "p", False),
    "rewire_gated_t095": ("REWIRE-gated, $\\theta$ = 0.95", OKABE["green"], "h", False),
    "rewire_gated_b128": ("REWIRE-gated, budget 128", OKABE["green"], "H", False),
    "abl_sketch_ridge":  ("Sketch only, ridge readout", OKABE["blue"], "o", False),
    "abl_sketch_ridge_gated": ("Sketch only, ridge readout, gated", OKABE["blue"], "D", False),
    "abl_sketch_lda":    ("Sketch only, discriminant readout", OKABE["blue"], "v", False),
    "abl_dualview_ridge": ("Dual view, ridge readout", OKABE["blue"], "^", False),
    "abl_width128":      ("REWIRE, sketch 128", OKABE["blue"], "<", False),
    "abl_width256":      ("REWIRE, sketch 256", OKABE["blue"], ">", False),
    "abl_select_errors": ("REWIRE, selective write (errors)", OKABE["blue"], "x", False),
    "abl_select_random": ("REWIRE, selective write (random)", OKABE["blue"], "+", False),
    "abl_sketch_ridge_select_errors": ("Sketch only, ridge, selective write (errors)", OKABE["blue"], "x", False),
    "abl_sketch_ridge_select_random": ("Sketch only, ridge, selective write (random)", OKABE["blue"], "+", False),
    "abl_sketch_ridge_woodbury": ("Sketch only, ridge, recursive solve", OKABE["blue"], "1", False),
    "ctrl_slda_input":   ("SLDA on the inputs", OKABE["purple"], "s", True),
    "ctrl_relu":         ("ReLU layer instead of LIF", OKABE["purple"], "D", False),
    "ranpac":            ("RanPAC-style (ReLU, ridge)", OKABE["sky"], "o", False),
    "ridge_input":       ("Ridge on the inputs", OKABE["purple"], "v", False),
    "hd_memory":         ("HD bundling", OKABE["purple"], "^", False),
    "hlop":              ("HLOP", OKABE["vermillion"], "s", True),
    "hlop_adaptive":     ("HLOP-R (subspace refreshed)", OKABE["vermillion"], "D", False),
    "hlop_gated":        ("HLOP, competence-gated", OKABE["vermillion"], "o", False),
    "replay":            ("Rehearsal", OKABE["orange"], "s", True),
    "replay_iso":        ("Rehearsal (iso-memory)", OKABE["orange"], "D", False),
    "replay_gated":      ("Rehearsal, competence-gated", OKABE["orange"], "o", False),
    "ewc":               ("EWC", OKABE["black"], "^", True),
    "naive":             ("Fine-tuning", OKABE["grey"], "X", True),
    "dsd_snn":           ("DSD-SNN", OKABE["grey"], "v", False),
}
# names in running text (lower case where the paper writes them so; no "(ours)")
PROSE = {"rewire": "REWIRE", "rewire_gated": "REWIRE-gated", "rewire_gated_maha": "REWIRE-gated with the Mahalanobis familiarity",
         "rewire_gated_t095": "REWIRE-gated at theta 0.95", "rewire_gated_b128": "the budgeted gate", "abl_sketch_ridge": "the sketch-only ridge learner",
         "abl_sketch_ridge_gated": "the gated sketch-only ridge learner", "abl_sketch_lda": "the sketch-only discriminant", "abl_dualview_ridge": "the dual view with a ridge readout",
         "abl_width128": "the 128-wide sketch", "abl_width256": "the 256-wide sketch", "abl_select_errors": "the selective write of errors",
         "abl_select_random": "the random selective write", "abl_sketch_ridge_select_errors": "the sketch-only ridge learner writing errors",
         "abl_sketch_ridge_select_random": "the sketch-only ridge learner writing a random subset", "abl_sketch_ridge_woodbury": "the recursive solve",
         "ctrl_relu": "the rectifier substitution", "ctrl_slda_input": "SLDA on the inputs", "ranpac": "the RanPAC-style control", "ridge_input": "ridge on the inputs",
         "hd_memory": "hyperdimensional bundling", "hlop": "HLOP", "hlop_adaptive": "HLOP-R", "hlop_gated": "competence-gated HLOP", "replay": "rehearsal",
         "replay_iso": "iso-memory rehearsal", "replay_gated": "competence-gated rehearsal", "ewc": "EWC", "naive": "fine-tuning", "dsd_snn": "DSD-SNN"}
SHORT = {"rewire": "REWIRE", "rewire_gated": "REWIRE-gated", "ctrl_slda_input": "SLDA-in", "ranpac": "RanPAC", "hlop": "HLOP",
         "replay": "Rehearsal", "replay_iso": "Reh.-iso", "ewc": "EWC", "naive": "Fine-\ntuning", "hlop_gated": "HLOP-gated", "replay_gated": "Reh.-gated"}
# table labels (LaTeX)
TEX_LABEL = {a: s[0].replace("$\\theta$ = 0.95", "$\\theta=0.95$") for a, s in ARM_STYLE.items()}
TEX_LABEL.update({"rewire": "REWIRE", "rewire_gated": "REWIRE-gated", "rewire_gated_maha": "REWIRE-gated, Mahalanobis familiarity",
                  "rewire_gated_t095": "REWIRE-gated, $\\theta=0.95$", "rewire_gated_b128": "REWIRE-gated, budget $B=128$",
                  "abl_select_errors": "REWIRE, selective write (errors and a random quarter)", "abl_select_random": "REWIRE, selective write (random subset)",
                  "abl_sketch_ridge_select_errors": "Sketch only, ridge, selective write (errors and a random quarter)",
                  "abl_sketch_ridge_select_random": "Sketch only, ridge, selective write (random subset)",
                  "abl_sketch_ridge_woodbury": "Sketch only, ridge, recursive (Woodbury) solve", "abl_width128": "REWIRE, sketch width 128",
                  "abl_width256": "REWIRE, sketch width 256", "hd_memory": "Hyperdimensional bundling", "replay": "Rehearsal (reservoir, 20 per stream class)"})

ORDER = ["rewire", "rewire_gated", "rewire_gated_maha", "rewire_gated_t095", "rewire_gated_b128",
         "abl_sketch_ridge", "abl_sketch_lda", "abl_dualview_ridge", "abl_width128", "abl_width256", "abl_select_errors", "abl_select_random",
         "abl_sketch_ridge_select_errors", "abl_sketch_ridge_select_random", "abl_sketch_ridge_woodbury", "abl_sketch_ridge_gated",
         "ctrl_relu", "ctrl_slda_input", "ridge_input", "ranpac", "hd_memory",
         "hlop", "hlop_adaptive", "hlop_gated", "replay", "replay_iso", "replay_gated", "ewc", "naive"]

DS_ORDER = ["mnist", "nmnist", "shd", "ssc", "cifar10", "cifar100", "tinyimagenet"]
DS_NAME = dict(mnist="MNIST", nmnist="N-MNIST", shd="SHD", ssc="SSC", cifar10="CIFAR-10", cifar100="CIFAR-100", tinyimagenet="Tiny-ImageNet")
DS_INPUT = dict(mnist="pixels", nmnist="events", shd="spikes", ssc="spikes", cifar10="ResNet features", cifar100="ResNet features", tinyimagenet="ResNet features")
DS_TWO_LINE = dict(mnist="MNIST", nmnist="N-MNIST", shd="SHD", ssc="SSC", cifar10="CIFAR-10", cifar100="CIFAR-\n100", tinyimagenet="Tiny-\nImageNet")
RAW = ("mnist", "nmnist", "shd", "ssc")         # no front end: pixels, events, spikes
FEAT = ("cifar10", "cifar100", "tinyimagenet")  # the frozen ResNet-18 features
GRADIENT = ("hlop", "hlop_adaptive", "hlop_gated", "replay", "replay_iso", "replay_gated", "ewc", "naive")
PROTECTED = ("hlop", "replay", "replay_iso", "ewc")   # the gradient-based continual learners (fine-tuning is the unprotected reference)


def label(a): return ARM_STYLE.get(a, (a,))[0]
def color(a): return ARM_STYLE.get(a, (a, OKABE["grey"]))[1]
def marker(a): return ARM_STYLE.get(a, (a, None, "o"))[2]
def filled(a): return ARM_STYLE.get(a, (a, None, None, True))[3]


def _font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for f in ("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "TeX Gyre Heros", "DejaVu Sans"):
        if f in names: return f
    return "DejaVu Sans"


FONT = None


def setup():
    """The rcParams of every paper figure. Returns the font family actually used (reported in the manifest)."""
    global FONT
    FONT = _font()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": 6, "axes.titlesize": 7, "axes.labelsize": 6.5,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "legend.title_fontsize": 6,
        "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold", "mathtext.sf": FONT, "mathtext.cal": FONT, "mathtext.tt": FONT,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "lines.linewidth": 0.9, "lines.markersize": 4, "patch.linewidth": 0.5, "hatch.linewidth": 0.5, "errorbar.capsize": 1.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4, "axes.axisbelow": True,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": 0.4, "legend.columnspacing": 1.0, "legend.borderaxespad": 0.2,
        "savefig.dpi": 300, "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.unicode_minus": True, "axes.titlelocation": "left", "axes.titlepad": 3, "axes.labelpad": 2,
        "figure.constrained_layout.use": False,
        "axes.formatter.useoffset": False, "axes.formatter.limits": (-5, 6),   # no "1e9"-style offset or exponent text on linear axes
    })
    return FONT


def panel_letter(fig, ax, letter, dx=-0.0, dy=0.0):
    """Nature panel letters: bold lowercase, 8 pt, at the top left of the axes' bounding box."""
    bb = ax.get_position(); fig.text(bb.x0 + dx, bb.y1 + dy, letter, fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, path_noext, formats=("eps", "png", "pdf")):
    """EPS (the submission format) and PNG at 300 dpi; a PDF copy is kept for the integrity scan and for Overleaf previews."""
    import logging, os
    logging.getLogger("matplotlib.backends.backend_ps").setLevel(logging.ERROR)
    os.makedirs(os.path.dirname(path_noext), exist_ok=True)
    for f in formats: fig.savefig(f"{path_noext}.{f}", format=f, dpi=300)
    plt.close(fig)
