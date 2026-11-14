#!/usr/bin/env python3
"""Data files of the run figures of three_phase_dab.tex, from the saved runs of the Simscape model.

Run from the document folder:  python3 py/three_phase_dab_runs.py <folder of sim_results_01.mat and _03.mat>
The runs are the MATLAB v7.3 files saved by init_model.m / the model of pwr-control/advanced-dcdc-converters
(analysis_three_phase_dab), tc = 100 ns, logged window of 20 ms at the end of the run; they are not kept in this
repository. Needs numpy and h5py. Writes into figures/:
  run01_period.dat, run03_period.dat   one period of the phase-a currents of run 01 (forward, 1 pu) and run 03
                                       (light load, 0.1 pu), averaged over the periods of the logged window and
                                       aligned on the rising edge of leg a of bridge 1, with the DC offset of the
                                       window removed, next to the harmonic model of three_phase_dab_figures.py at
                                       the measured V1, V2 and phi
  run01_dc.dat                         run 01: DC input current of side 1 and phase shift, averaged over each period
                                       of the logged window (limit cycle of the phase quantization)
"""
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import three_phase_dab_figures as tf  # noqa: E402

FIG = tf.FIG
TC = 100e-9                                     # [s] step of the runs
NP = int(round(1 / (tf.FS * TC)))               # samples per period, 1000


def load(path):
    """Logged signals (names without _sim), samples x channels."""
    d = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            if k.endswith("_sim"):
                a = np.array(f[k])
                d[k[:-4]] = a.T if a.shape[0] < a.shape[1] else a
    d["t"] = d.pop("time_tc")[:, 0]
    return d


def edges(u, V):
    """Commutations of a bridge from its phase-to-star voltages: sample of the new level, leg, direction."""
    du = np.diff(u, axis=0)
    n = np.where(np.max(np.abs(du), axis=1) > 0.5 * V)[0]
    k = np.argmax(np.abs(du[n]), axis=1)
    s = np.sign(du[n, k])
    return n + 1, k, s


def phase_shift(e1, e2):
    """Delay (samples) of every edge of bridge 2 after the last edge of the same leg and direction of bridge 1."""
    n1, k1, s1 = e1
    rows = []
    for n, k, s in zip(*e2):
        m = (k1 == k) & (s1 == s) & (n1 <= n)
        if m.any():
            rows.append((n, n - n1[m].max()))
    return np.array(rows)


def run_period(path, name):
    d = load(path)
    u1, u2 = d["u1_dab_transformer_modA"], d["u2_dab_transformer_modA"]
    i1, i2 = d["i1_dab_transformer_modA"], d["i2_dab_transformer_modA"]
    V1 = d["dab_voltage_input_modA"][:, 0].mean()
    V2 = d["dab_voltage_output_modA"][:, 0].mean()
    e1, e2 = edges(u1, V1), edges(u2, V2)
    dl = phase_shift(e1, e2)
    phi = 2 * np.pi * tf.FS * TC * dl[:, 1].mean()
    # periods that start on a rising edge of leg a of bridge 1
    start = e1[0][(e1[1] == 0) & (e1[2] > 0)]
    start = start[start + NP <= i1.shape[0]]
    off1, off2 = i1[:, 0].mean(), i2[:, 0].mean()
    ia = np.mean([i1[n:n + NP, 0] for n in start], axis=0) - off1
    ja = np.mean([i2[n:n + NP, 0] for n in start], axis=0) - off2
    o = tf.op_point(tf.P, phi, V1, V2)
    th = 2 * np.pi * np.arange(0, NP + 1, 2) / NP
    ia_m = np.array([tf.waveform(o["ph"], 3, t) for t in th])
    ja_m = np.array([tf.waveform(o["ph"], 4, t) for t in th])
    k = np.arange(0, NP + 1, 2) % NP
    with open(os.path.join(FIG, name), "w") as f:
        f.write("theta i1a i2a im i1a_h i2a_h im_h\n")
        for j in range(th.size):
            f.write(f"{np.degrees(th[j]):.2f} {ia[k[j]]:.2f} {ja[k[j]]:.2f} {ia[k[j]] - ja[k[j]]:.3f} "
                    f"{ia_m[j]:.2f} {ja_m[j]:.2f} {ia_m[j] - ja_m[j]:.3f}\n")
    print(f"{os.path.basename(path)}: V1 = {V1:.1f} V, V2 = {V2:.1f} V, phi = {np.degrees(phi):.2f} deg over "
          f"{dl.shape[0]} edges, {start.size} periods; phase a: I1 {np.sqrt(np.mean(i1[:, 0] ** 2)):.1f} A rms "
          f"(model {o['I1rms']:.1f}), offsets i1 {off1:.2f} A, i2 {off2:.2f} A; rms of the difference run - model "
          f"over the period: {np.sqrt(np.mean((ia[k] - ia_m) ** 2)):.2f} A")
    return d, e1, e2, dl, V1, V2


def run_dc(d, dl, name):
    """Period averages of the DC input current and of the phase shift over the logged window."""
    I1 = d["dab_current_input_modA"][:, 0]
    t = d["t"]
    npd = I1.size // NP
    per = (dl[:, 0] // NP).astype(int)
    with open(os.path.join(FIG, name), "w") as f:
        f.write("t idc1 phi\n")
        for p in range(npd):
            sel = per == p
            if not sel.any():
                continue
            ph = np.degrees(2 * np.pi * tf.FS * TC * dl[sel, 1].mean())
            f.write(f"{1e3 * (t[p * NP] - t[0]):.2f} {I1[p * NP:(p + 1) * NP].mean():.2f} {ph:.3f}\n")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    d, e1, e2, dl, V1, V2 = run_period(os.path.join(folder, "sim_results_01.mat"), "run01_period.dat")
    run_dc(d, dl, "run01_dc.dat")
    run_period(os.path.join(folder, "sim_results_03.mat"), "run03_period.dat")
