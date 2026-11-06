"""Figures of the design note upqc.tex from the summaries of two simulations of ls_upqc.slx.

    python3 upqc_figures.py [figure_folder]        (default ../figures)

Input (folder data/ next to this script), written by upqc_summary.m from the results of the default case of
init_model.m, logging step 25 us (param.time.tc_decimation = 40). The repository ignores *.csv: only the
*_numbers.txt files are committed; to redraw the figures, run the simulations, save the workspace and call
upqc_summary(matfile, 'data/s1', true) (and 'data/s0', ...) from this folder:
    s1_env.csv, s1_numbers.txt, s1_wave.csv    with the storage (param.sim.storage = 1)
    s0_env.csv, s0_numbers.txt                 without the storage (param.sim.storage = 0)
    s1_lim2_env.csv, s1_lim2_numbers.txt       with the storage and the DAB of the branch main: output of the
                                               DC-link voltage PI (phase shift) limited to +-2 x pi/2; optional
    s1_lim1_k1_env.csv, s1_lim1_k1_numbers.txt with the storage, limit +-1 and the gains of main (kp_udc = 1,
                                               ki_udc = 35); optional, when the default gains are other
The *_env.csv files have one row every 10 ms with the quantities over the last grid period: sequence
components of the grid and load voltages (rms phase values), grid sequence powers, load power, rms currents,
DC-link voltage (mean, min, max over the period), battery and DAB power. The *_wave.csv file has the grid and
load voltages, the grid currents and the DC-link voltage from 0.72 s to 0.82 s every 0.2 ms.
Output: upqc_storage.png, upqc_no_storage.png, upqc_fault_onset.png, upqc_currents.png and, with the optional
inputs, upqc_dab_fix.png.
Needs numpy and matplotlib.
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
U_PH = 690 / np.sqrt(3)             # [V] rated rms phase voltage, 1 pu of the sequence plots
I_VS_NOM = 250e3 / np.sqrt(3) / 690 / 0.9   # [A] rated rms current of the 250 kW data set (is_nom)
FAULT = (0.75, 1.665)               # [s] FRT test 25.4 of the default case
Q_STEP = 2.0                        # [s] step of the reactive current reference of the SAF
UDC_REF = 1200                      # [V]


def load(name):
    with open(os.path.join(DATA, name), newline="") as f:
        rows = list(csv.reader(f))
    hdr = rows[0]
    m = np.array([[float(x) for x in r] for r in rows[1:]])
    return {h: m[:, k] for k, h in enumerate(hdr)}


def events(ax):
    ax.axvspan(*FAULT, color="0.85", zorder=0, lw=0)
    ax.axvline(Q_STEP, color="0.4", ls=":", lw=0.8)
    ax.grid(True, lw=0.3)


def seq_and_power(axs, e):
    ax = axs[0]
    events(ax)
    ax.plot(e["t"], e["Ug1p"] / U_PH, color="tab:blue", label="grid, positive")
    ax.plot(e["t"], e["Ug1n"] / U_PH, color="tab:blue", ls="--", label="grid, negative")
    ax.plot(e["t"], e["Ul1p"] / U_PH, color="tab:red", label="load, positive")
    ax.plot(e["t"], e["Ul1n"] / U_PH, color="tab:red", ls="--", label="load, negative")
    ax.set_ylabel("voltage\nsequences [pu]")
    ax.set_ylim(-0.05, 1.45)
    ax.legend(ncol=4, fontsize=7, loc="upper right")
    ax = axs[1]
    events(ax)
    ax.plot(e["t"], (e["Pg"] + e["Pg1n"]) / 1e3, color="tab:blue", label="grid P")
    ax.plot(e["t"], e["Qg"] / 1e3, color="tab:blue", ls="--", label="grid Q (positive seq.)")
    ax.plot(e["t"], e["Pl"] / 1e3, color="tab:red", label="load P")
    ax.set_ylabel("power\n[kW, kvar]")
    ax.legend(ncol=3, fontsize=7, loc="upper right")
    ax = axs[2]
    events(ax)
    ax.fill_between(e["t"], e["udc_min"], e["udc_max"], color="tab:green", alpha=0.3, lw=0, label="min-max in the period")
    ax.plot(e["t"], e["udc_mean"], color="tab:green", label="mean")
    ax.axhline(UDC_REF, color="k", ls="--", lw=0.8, label="reference")
    ax.set_ylabel("DC link [V]")
    ax.legend(ncol=3, fontsize=7, loc="lower right")


def fig_case(e, title, out, storage):
    fig, axs = plt.subplots(4, 1, figsize=(6.3, 7.6), sharex=True)
    seq_and_power(axs, e)
    ax = axs[3]
    events(ax)
    if storage:
        ax.plot(e["t"], e["pb_mean"] / 1e3, color="tab:purple", label="battery (positive: discharge)")
        ax.plot(e["t"], e["pdab_out_mean"] / 1e3, color="tab:orange", ls="--", label="DAB output to the DC link")
        ax.set_ylabel("power [kW]")
    else:
        ax.plot(e["t"], e["Is_saf_rms"], color="tab:olive", label="SAF stack")
        ax.plot(e["t"], e["Is_vs_rms"], color="tab:cyan", label="VS stack")
        ax.plot(e["t"], e["Ig1p"], color="tab:blue", label="grid, positive seq.")
        ax.plot(e["t"], e["Ig1n"], color="tab:blue", ls="--", label="grid, negative seq.")
        ax.set_ylabel("current [A rms]")
    ax.legend(ncol=2, fontsize=7, loc="upper left")
    ax.set_xlabel("time [s]")
    ax.set_xlim(0, e["t"][-1])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_wave(w, out):
    fig, axs = plt.subplots(3, 1, figsize=(6.3, 6.0), sharex=True)
    t = w["t"] * 1e3
    for ax, keys, lab in ((axs[0], ("uga", "ugb", "ugc"), "grid voltage [V]"),
                          (axs[1], ("ula", "ulb", "ulc"), "load voltage [V]"),
                          (axs[2], ("iga", "igb", "igc"), "grid current [A]")):
        for k, c in zip(keys, ("tab:blue", "tab:red", "tab:green")):
            ax.plot(t, w[k], color=c, lw=0.9, label=k[-1])
        ax.axvline(FAULT[0] * 1e3, color="0.4", ls=":", lw=0.8)
        ax.set_ylabel(lab)
        ax.grid(True, lw=0.3)
    axs[0].legend(ncol=3, fontsize=7, loc="upper right", title="phase", title_fontsize=7)
    axs[2].set_xlabel("time [ms]")
    axs[2].set_xlim(t[0], t[-1])
    fig.suptitle("With the storage: start of the fault (test 25.4 at 750 ms)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_currents(e1, e0, out):
    fig, axs = plt.subplots(2, 1, figsize=(6.3, 4.6), sharex=True)
    for ax, e, title in ((axs[0], e1, "with the storage"), (axs[1], e0, "without the storage")):
        events(ax)
        ax.plot(e["t"], e["Is_saf_rms"] / I_VS_NOM, color="tab:olive", label="SAF stack")
        ax.plot(e["t"], e["Is_vs_rms"] / I_VS_NOM, color="tab:cyan", label="VS stack")
        ax.plot(e["t"], e["Il_rms"] / I_VS_NOM, color="tab:red", ls="--", label="load")
        ax.axhline(1, color="k", ls=":", lw=0.8)
        ax.set_ylabel("rms current /\nrated current")
        ax.set_title(title, fontsize=8)
        ax.legend(ncol=3, fontsize=7, loc="upper left")
    axs[1].set_xlabel("time [s]")
    axs[1].set_xlim(0, e1["t"][-1])
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_dab_fix(cases, out):
    """cases: list of (env, colour, label), the last one the default case."""
    fig, axs = plt.subplots(2, 1, figsize=(6.3, 4.8), sharex=True)
    ax = axs[0]
    events(ax)
    for e, c, lab in cases:
        ax.fill_between(e["t"], e["udc_min"], e["udc_max"], color=c, alpha=0.2, lw=0)
        ax.plot(e["t"], e["udc_mean"], color=c, label=lab)
    ax.axhline(UDC_REF, color="k", ls="--", lw=0.8)
    ax.set_ylabel("DC link [V]\n(mean, min-max)")
    ax.legend(ncol=1, fontsize=7, loc="upper left")
    ax = axs[1]
    events(ax)
    for e, c, lab in cases:
        ax.plot(e["t"], e["pb_mean"] / 1e3, color=c, label=lab)
    ax.set_ylabel("battery power [kW]\n(positive: discharge)")
    axs[1].set_xlabel("time [s]")
    axs[1].set_xlim(0, cases[-1][0]["t"][-1])
    fig.suptitle("With the storage: DC-link voltage control of the DAB", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "figures")
    os.makedirs(out, exist_ok=True)
    e1 = load("s1_env.csv")
    e0 = load("s0_env.csv")
    fig_case(e1, "With the storage (param.sim.storage = 1)", os.path.join(out, "upqc_storage.png"), True)
    fig_case(e0, "Without the storage (param.sim.storage = 0)", os.path.join(out, "upqc_no_storage.png"), False)
    fig_wave(load("s1_wave.csv"), os.path.join(out, "upqc_fault_onset.png"))
    fig_currents(e1, e0, os.path.join(out, "upqc_currents.png"))
    cases = []
    if os.path.exists(os.path.join(DATA, "s1_lim2_env.csv")):
        cases.append((load("s1_lim2_env.csv"), "tab:gray", "phase shift up to $\\pi$ (main)"))
    if os.path.exists(os.path.join(DATA, "s1_lim1_k1_env.csv")):
        cases.append((load("s1_lim1_k1_env.csv"), "tab:orange", "up to $\\pi/2$, gains of main"))
    if cases:
        cases.append((e1, "tab:green", "default case"))
        fig_dab_fix(cases, os.path.join(out, "upqc_dab_fix.png"))
    print("figures in", os.path.abspath(out))


if __name__ == "__main__":
    main()
