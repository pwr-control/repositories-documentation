#!/usr/bin/env python3
"""Data files of the figures of cllc_dcx.tex (pgfplots reads them with \\addplot table).

Run from the document folder:  python3 py/cllc_dcx_figures.py
Writes into figures/:
  fha_gain_50kHz.dat, fha_gain_35kHz.dat   first-harmonic gain n u2/u1 and input impedance angle of the CLLC
                                           tank against fn = fs/fr, at 10, 50 and 100 % load (the circuit of
                                           cllc_analysis.m of pwr-control/solid-state-transformers: Lr1, Cr1,
                                           Lm, Lr2 and Cr2 reflected to side 1, FHA load Rac)
  tank_waveforms_50kHz.dat, tank_waveforms_35kHz.dat
                                           idealized steady-state waveforms of the DCX over two switching
                                           periods: bridge voltage, resonant currents, magnetizing current,
                                           resonant capacitor voltage
The numbers are those of cllc_dcx_param_example.m and cllc_dcx_param_example_35kHz.m of pwr-control/library.
Only the standard library is used.
"""
import cmath
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(FIG, exist_ok=True)

STAGES = {
    "50kHz": dict(u1=800.0, n1=11, n2=11, P=50e3, fs=50e3, fn=0.95, Lr1=12e-6, Lm=400e-6, Coss=0.79e-9),
    "35kHz": dict(u1=790.0, n1=9, n2=8, P=50e3, fs=35e3, fn=0.95, Lr1=16e-6, Lm=560e-6, Coss=0.79e-9),
}


def derived(s):
    d = dict(s)
    d["n"] = s["n1"] / s["n2"]
    d["fr"] = s["fs"] / s["fn"]
    d["Cr1"] = 1.0 / ((2 * math.pi * d["fr"]) ** 2 * s["Lr1"])
    d["Lr2"] = s["Lr1"] / d["n"] ** 2
    d["Cr2"] = d["Cr1"] * d["n"] ** 2
    d["u2"] = s["u1"] / d["n"]
    d["idc1"] = s["P"] / s["u1"]
    d["idc2"] = s["P"] / d["u2"]
    d["Rac"] = 8 * d["n"] ** 2 * d["u2"] ** 2 / (math.pi ** 2 * s["P"])
    d["Z0"] = math.sqrt(s["Lr1"] / d["Cr1"])
    d["Q"] = d["Z0"] / d["Rac"]
    d["Ln"] = s["Lm"] / s["Lr1"]
    d["I1pk"] = math.pi / 2 * d["idc1"] / s["fn"]          # resonant half sine carrying idc1 per half period
    d["I1rms"] = d["I1pk"] * math.sqrt(s["fn"] / 2)
    d["Impk"] = s["u1"] / (4 * s["Lm"] * s["fs"])
    d["Vcr1pk"] = d["I1pk"] / (2 * math.pi * d["fr"] * d["Cr1"])
    return d


def fha_gain(d, path):
    """|n u2/u1| and angle of Zin against fn, FHA, loads 10 / 50 / 100 %."""
    loads = (0.1, 0.5, 1.0)
    with open(path, "w") as f:
        f.write("fn " + " ".join(f"M{int(100*l)} " for l in loads) + " ".join(f"phiZ{int(100*l)} " for l in loads) + "\n")
        npts = 401
        for k in range(npts):
            fn = 0.5 + 1.0 * k / (npts - 1)
            w = 2 * math.pi * fn * d["fr"]
            s = 1j * w
            ZLs1 = s * d["Lr1"]
            ZCs1 = 1 / (s * d["Cr1"])
            ZLm = s * d["Lm"]
            ZLs2 = s * d["Lr2"] * d["n"] ** 2            # reflected to side 1
            ZCs2 = 1 / (s * d["Cr2"] / d["n"] ** 2)
            Ms, phis = [], []
            for l in loads:
                R = d["Rac"] / l
                ZLout = ZLs2 + ZCs2 + R
                Zpout = ZLout * ZLm / (ZLout + ZLm)
                Zin = ZLs1 + ZCs1 + Zpout
                H = Zpout / Zin * R / ZLout
                Ms.append(abs(H))
                phis.append(math.degrees(cmath.phase(Zin)))
            f.write(f"{fn:.4f} " + " ".join(f"{m:.5f}" for m in Ms) + " " + " ".join(f"{p:.2f}" for p in phis) + "\n")


def tank_waveforms(d, path):
    """Idealized DCX waveforms over two switching periods, time in us.

    Half period Ts/2 = 1/(2 fs): the resonant current is a half sine of period 1/fr (it lasts 1/(2 fr) < Ts/2),
    then zero until the next commutation (only the magnetizing current flows); the magnetizing current is a
    triangle of slope u1/Lm; the resonant capacitor voltage is -Vpk cos(wr t) during the half sine and constant
    in the gap. i_r2 is referred to side 1 (n i_r2) and equals the resonant current; i_r1 = resonant + i_m.
    """
    fs, fr, u1 = d["fs"], d["fr"], d["u1"]
    Ts = 1 / fs
    th = Ts / 2                     # half period
    tr = 1 / (2 * fr)               # duration of the half sine
    wr = 2 * math.pi * fr
    npts = 1601
    with open(path, "w") as f:
        f.write("t_us u_b1 i_res i_m i_r1 u_cr1\n")
        for k in range(npts):
            t = 2 * Ts * k / (npts - 1)
            m = int(t // th)         # half period index
            tau = t - m * th
            sgn = 1.0 if m % 2 == 0 else -1.0
            u_b1 = sgn * u1
            if tau < tr:
                i_res = sgn * d["I1pk"] * math.sin(wr * tau)
                u_cr1 = -sgn * d["Vcr1pk"] * math.cos(wr * tau)
            else:
                i_res = 0.0
                u_cr1 = sgn * d["Vcr1pk"]
            i_m = sgn * (-d["Impk"] + 2 * d["Impk"] * tau / th)     # triangle, -Impk -> +Impk in a positive half period
            f.write(f"{t*1e6:.4f} {u_b1:.1f} {i_res:.3f} {i_m:.3f} {i_res + i_m:.3f} {u_cr1:.2f}\n")


if __name__ == "__main__":
    for name, s in STAGES.items():
        d = derived(s)
        fha_gain(d, os.path.join(FIG, f"fha_gain_{name}.dat"))
        tank_waveforms(d, os.path.join(FIG, f"tank_waveforms_{name}.dat"))
        print(f"{name}: fr = {d['fr']/1e3:.2f} kHz, Cr1 = {d['Cr1']*1e6:.3f} uF, Z0 = {d['Z0']:.2f} Ohm, "
              f"Rac = {d['Rac']:.2f} Ohm, Q = {d['Q']:.3f}, Ln = {d['Ln']:.1f}, I1pk = {d['I1pk']:.1f} A, "
              f"I1rms = {d['I1rms']:.1f} A, Im = {d['Impk']:.1f} A, Vcr1pk = {d['Vcr1pk']:.0f} V")
        w = 2 * math.pi * s["fs"]
        X = 2 * (w * s["Lr1"] - 1 / (w * d["Cr1"]))
        Pmax = (4 / math.pi * s["u1"]) * (4 / math.pi * d["n"] * d["u2"]) / (2 * abs(X))
        Rnum = (w ** 2 * s["Lr1"] + 1 / d["Cr1"])
        print(f"        X(fs) = {X:.3f} Ohm, P(phi) = {Pmax/1e3:.0f} kW sin(phi), 50 kW at {math.degrees(math.asin(s['P']/Pmax)):.1f} deg, "
              f"R_num/tc = {Rnum:.3e} Ohm/s -> {Rnum*50e-9:.3f} Ohm at 50 ns, {Rnum*500e-9:.2f} at 500 ns, {Rnum*1e-6:.2f} at 1 us")
        for l in (0.1, 0.5, 1.0):
            # gain at fs
            wfs = 2 * math.pi * s["fs"]; ss = 1j * wfs
            R = d["Rac"] / l
            ZLout = ss * d["Lr2"] * d["n"]**2 + 1/(ss * d["Cr2"]/d["n"]**2) + R
            Zpout = ZLout * ss*s["Lm"] / (ZLout + ss*s["Lm"])
            Zin = ss*s["Lr1"] + 1/(ss*d["Cr1"]) + Zpout
            H = Zpout/Zin*R/ZLout
            print(f"        load {int(l*100):3d} %: |M(fs)| = {abs(H):.4f}, Zin = {abs(Zin):.2f} Ohm at {math.degrees(cmath.phase(Zin)):.1f} deg")
