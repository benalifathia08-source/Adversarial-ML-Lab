"""Figures for the README, built only from results/*.json. Run: python -m advml.figures"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from advml.config import FIGURES  # noqa: E402
from advml.results import load_json  # noqa: E402

INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
RED, BLUE, GREEN, ORANGE, GREY = "#cf222e", "#0969da", "#1a7f37", "#bc4c00", "#8c959f"
NL = "\n"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 150,
})


def _pct(ax, axis: str = "y", decimals: int = 0) -> None:
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(PercentFormatter(1.0, decimals=decimals))


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def evasion() -> None:
    r = load_json("m3_evasion")
    d = load_json("m5_defenses")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.25})
    one, both = r["constrained_one_sided"], r["constrained_both_ends"]
    budgets = list(both["whitebox_pgd_mlp"])
    x = [float(b) * 100 for b in budgets]
    ax.plot(x, [both["whitebox_pgd_mlp"][b]["evasion_rate"] for b in budgets], "-o", color=RED,
            label="Both ends (C2 / exfiltration), white-box vs MLP")
    ax.plot(x, [both["blackbox_random_search_lgbm"][b] for b in budgets], "--s", color=ORANGE,
            label="Both ends, black-box (100 queries) vs LightGBM")
    ax.plot(x, [one["whitebox_pgd_mlp"][b]["evasion_rate"] for b in budgets], "-o", color=BLUE,
            label="One side only, white-box vs MLP")
    hardened = d["adversarial_training"]["whitebox_both_ends"]
    ax.plot([float(b) * 100 for b in hardened], list(hardened.values()), "-D", color=GREEN,
            label="Both ends vs adversarially trained MLP (about 0%)")
    ax.set_xscale("log")
    ax.set_xticks(x, [f"+{int(v)}%" for v in x])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    _pct(ax)
    ax.set_xlabel("Traffic the attacker may add (budget)")
    ax.set_ylabel("Detected attacks that evade")
    ax.set_title("Realistic evasion (every flow stays valid)")
    ax.legend(loc="upper left", fontsize=8.5)

    paper = r["paper_attacks"]
    names = ["fgsm_eps0.1", "pgd_eps0.1", "pgd_eps1.0"]
    labels = [f"FGSM{NL}ε=0.1", f"PGD{NL}ε=0.1", f"PGD{NL}ε=1.0", f"HopSkipJump{NL}(LightGBM)"]
    hsj = r["hopskipjump_lgbm"]
    ev = [paper[n]["evasion_rate"] for n in names] + [hsj["evasion_rate"]]
    va = [paper[n]["evaded_and_valid"] for n in names] + [hsj["evaded_and_valid"]]
    pos = list(range(len(labels)))
    ax2.bar([p - 0.2 for p in pos], ev, 0.4, color=GREY, label="Evades the model")
    ax2.bar([p + 0.2 for p in pos], va, 0.4, color=RED, label="…and is a valid flow")
    for p, v in zip(pos, va, strict=True):
        ax2.text(p + 0.2, v + 0.02, f"{v:.0%}", ha="center", fontsize=8.5, color=INK)
    ax2.set_xticks(pos, labels, fontsize=8.5)
    ax2.set_ylim(0, 1.15)
    _pct(ax2)
    ax2.set_title("“Paper” attacks (any feature)")
    ax2.legend(loc="upper center", fontsize=8.5, ncols=2)
    _save(fig, "evasion")


def importance() -> None:
    r = load_json("m3_evasion")["importance"]
    groups = list(r["groups"].items())[:15][::-1]
    colors = {"one_sided": BLUE, "both_ends": RED, "nobody": GREY}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh([g for g, _ in groups], [v["recall_drop"] for _, v in groups],
            color=[colors[v["controlled_by"]] for _, v in groups])
    share = r["share_of_importance"]
    handles = [Patch(color=BLUE, label=f"attacker, one side: {share['one_sided']:.0%}"),
               Patch(color=RED, label=f"attacker owning both ends: {share['both_ends']:.0%}"),
               Patch(color=GREY, label=f"nobody (sensor / protocol): {share['nobody']:.0%}")]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, title="Who controls it (share of importance)",
              title_fontsize=8.5)
    ax.set_xlabel("Drop in detection rate when the feature is shuffled")
    _pct(ax, "x", decimals=1)
    ax.set_title("What the detector relies on, and who controls it")
    _save(fig, "importance")


def backdoor() -> None:
    r = load_json("m4_poisoning")
    rates = list(r["backdoor"])
    x = [float(k) * 100 for k in rates]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(x, [r["backdoor"][k]["backdoor_success"] for k in rates], "-o", color=RED,
            label="Triggered attacks let through")
    ax.plot(x, [r["backdoor"][k]["recall"] for k in rates], "-s", color=BLUE, label="Detection rate on normal attacks")
    ax.axhline(r["clean"]["backdoor_success"], color=GREY, ls=":", lw=1)
    ax.text(x[-1], r["clean"]["backdoor_success"] + 0.03, "same trigger on a clean model", ha="right", fontsize=8.5,
            color=MUTED)
    for xi, k in zip(x, rates, strict=True):
        ax.annotate(f"{r['backdoor'][k]['poison_rows']} rows", (xi, r["backdoor"][k]["backdoor_success"]),
                    textcoords="offset points", xytext=(0, -16), ha="center", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{v:g}%" for v in x])
    ax.minorticks_off()
    ax.set_ylim(0, 1.05)
    _pct(ax)
    ax.set_xlabel("Poisoned share of the training set")
    ax.set_title("Backdoor: packets padded to 1,337 bytes walk past the model")
    ax.legend(loc="center right", fontsize=8.5)
    _save(fig, "backdoor")


def poison_defenses() -> None:
    p = load_json("m4_poisoning")
    d = p["backdoor_defenses"]["0.01"]
    rows = [("No defense", p["backdoor"]["0.01"]["backdoor_success"], None)]
    names = {"activation_clustering_art": "Activation clustering (ART)", "spectral_signature": "Spectral signature",
             "label_check_cross_validated": "Cross-validated label check",
             "trusted_core_set": "Trusted core set (5% verified)",
             "trusted_core_set_plus_spike_check": "Trusted core set + spike check"}
    for key, label in names.items():
        rows.append((label, d[key]["after_retraining"]["backdoor_success"], d[key]["detection"]["poison_caught"]))
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    colors = [GREEN if v < 0.05 else (ORANGE if v < 0.5 else RED) for _, v, _ in rows]
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=colors)
    for i, (_, v, caught) in enumerate(rows):
        note = f"{v:.1%}" + (f"   (caught {caught:.0%} of the poison)" if caught is not None else "")
        ax.text(v + 0.01, i, note, va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, 1.35)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    _pct(ax, "x")
    ax.set_xlabel("Backdoor success after the defense and retraining (1% poison)")
    ax.set_title("Poisoning defenses: standard detectors fail, layered checks work")
    _save(fig, "poison_defenses")


def evasion_defenses() -> None:
    d = load_json("m5_defenses")
    entries = [(f"Baseline{NL}MLP", d["baseline"], "whitebox_both_ends"),
               (f"Adversarial{NL}training", d["adversarial_training"], "whitebox_both_ends"),
               (f"Robust features{NL}(released)", d["robust_features"], "whitebox_both_ends"),
               (f"MLP OR{NL}LightGBM", d["ensemble_or"], "whitebox_mlp_then_transfer_both_ends")]
    labels = [e[0] for e in entries]
    white = [e[1][e[2]]["1.0"] for e in entries]
    black = [e[1]["blackbox_both_ends_b1.0"] for e in entries]
    fpr = [e[1]["clean"]["fpr"] for e in entries]
    recall = [e[1]["clean"]["recall"] for e in entries]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"wspace": 0.2})
    pos = list(range(len(labels)))
    ax.bar([p - 0.2 for p in pos], white, 0.4, color=RED, label="White-box (attacker knows the defense)")
    ax.bar([p + 0.2 for p in pos], black, 0.4, color=ORANGE, label="Black-box, 200 queries")
    for p, w, b in zip(pos, white, black, strict=True):
        ax.text(p - 0.2, w + 0.015, f"{w:.1%}", ha="center", fontsize=8)
        ax.text(p + 0.2, b + 0.015, f"{b:.1%}", ha="center", fontsize=8)
    ax.set_xticks(pos, labels, fontsize=9)
    _pct(ax)
    ax.set_ylim(0, 0.85)
    ax.set_title("Evasion, both-ends attacker, +100% traffic")
    ax.legend(fontsize=8.5, loc="upper center")
    ax2.bar([p - 0.2 for p in pos], recall, 0.4, color=BLUE, label="Detection rate (recall)")
    ax2.bar([p + 0.2 for p in pos], fpr, 0.4, color=GREY, label="False-positive rate")
    for p, r_, f in zip(pos, recall, fpr, strict=True):
        ax2.text(p - 0.2, r_ + 0.015, f"{r_:.1%}", ha="center", fontsize=8)
        ax2.text(p + 0.2, f + 0.015, f"{f:.1%}", ha="center", fontsize=8)
    ax2.axhline(0.15, color=RED, ls="--", lw=1, label="Gate limit: 15% false positives")
    ax2.set_xticks(pos, labels, fontsize=9)
    _pct(ax2)
    ax2.set_ylim(0, 1.25)
    ax2.set_title("Cost on clean traffic (validation)")
    ax2.legend(fontsize=8.5, loc="upper center", ncols=3)
    _save(fig, "evasion_defenses")


def main() -> None:
    for fn in (evasion, importance, backdoor, poison_defenses, evasion_defenses):
        fn()
        print("figure:", fn.__name__)


if __name__ == "__main__":
    main()
