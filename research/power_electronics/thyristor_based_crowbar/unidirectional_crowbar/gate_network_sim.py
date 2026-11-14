#!/usr/bin/env python3
"""
Time-domain response of the passive anode-to-gate trigger network of the
crowbar thyristor (5STP 26N6500) during the breaker-opening transient.

The anode-cathode voltage v_AK(t) is *forced* (taken from the simulated load
voltage: raised-cosine swing -3 kV -> +2.8 kV in ~5 us, decay to the +2.5 kV
MOV plateau, thyristor turn-on after t_gd, voltage collapse).  The network
(D1, C_h || R_C, C_g + R_gs, TVS + R_s, R_GK, D_GK, gate) is integrated with
explicit Euler at 1 ns.  Purpose: gate-current waveforms of the two paths.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

# ------------------------------------------------------------------ parameters
P = dict(
    Cg=2.2e-9, Rgs=22.0,          # dynamic path
    Ch=22e-9, RC=22e3,            # hold capacitor and reset resistor (N to K)
    Vst=520.0, r_tvs=6.0, Rs=330.0,   # static path: 3 x 5KP150A (~173 V each) + Rs
    RGK=10.0,                     # gate-cathode resistor
    VfD1=5.0, RD1=1.0,            # D1 string (5 diodes) forward model
    Vg0=1.2, rg=0.7,              # gate-cathode V-I: v_G = Vg0 + rg*i
    VfT=2.4,                      # TVS stack forward drop (3 x 0.8 V)
)
# forced anode voltage
V_NEG, V_PK, V_PL = -3000.0, 2800.0, 2500.0   # service, overshoot, MOV plateau
T_R = 5.0e-6            # swing duration (raised cosine)
TAU_OS = 1.0e-6         # overshoot decay
T_GD = 2.5e-6           # assumed gate-controlled delay from first gate current
T_F = 1.5e-6            # anode voltage fall time at turn-on
V_ON = 30.0             # residual on-state voltage during spreading
TAU_ON = 10e-6


def make_vak(t, tr=T_R, vpk=V_PK, tgd=T_GD, t_on_override=None):
    dv = vpk - V_NEG
    # start of swing such that v_AK(0) = 0
    th0 = np.arccos(1.0 - 2.0 * (-V_NEG) / dv)
    ts = -th0 / np.pi * tr
    te = ts + tr
    v = np.full_like(t, V_NEG)
    m = (t >= ts) & (t <= te)
    v[m] = V_NEG + dv * (1.0 - np.cos(np.pi * (t[m] - ts) / tr)) / 2.0
    m2 = t > te
    v[m2] = V_PL + (vpk - V_PL) * np.exp(-(t[m2] - te) / TAU_OS)
    # thyristor turn-on
    t_on = tgd if t_on_override is None else t_on_override
    m3 = t >= t_on
    v_on0 = np.interp(t_on, t, v)
    fall = np.clip(1.0 - (t[m3] - t_on) / T_F, 0.0, 1.0)
    v[m3] = v_on0 * fall + V_ON * (1.0 - fall) * np.exp(-(t[m3] - t_on) / TAU_ON)
    return v, ts, te, t_on


def simulate(t, vak, P):
    dt = t[1] - t[0]
    n = len(t)
    vN = 0.0; vCg = 0.0
    out = {k: np.zeros(n) for k in ("vN", "vG", "iCg", "iS", "iG", "iD1", "iRGK")}
    for k in range(n):
        v = vak[k]
        # solve gate node by bisection (KCL at G)
        def kcl(vG):
            iCg = (vN - vCg - vG) / P["Rgs"]
            vT = vN - vG
            if vT > P["Vst"]:
                iS = (vT - P["Vst"]) / (P["Rs"] + P["r_tvs"])
            elif vT < -P["VfT"]:
                iS = (vT + P["VfT"]) / P["Rs"]
            else:
                iS = 0.0
            ig = (vG - P["Vg0"]) / P["rg"] if vG > P["Vg0"] else 0.0
            iRGK = vG / P["RGK"]
            iDGK = (vG + 0.7) / 0.05 if vG < -0.7 else 0.0
            return iCg + iS - ig - iRGK - iDGK, iCg, iS, ig, iRGK
        lo, hi = -50.0, 50.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            f, *_ = kcl(mid)
            if f > 0:
                lo = mid
            else:
                hi = mid
        vG = 0.5 * (lo + hi)
        _, iCg, iS, ig, iRGK = kcl(vG)
        iD1 = max(0.0, (v - vN - P["VfD1"]) / P["RD1"])
        iRC = vN / P["RC"]
        dvN = (iD1 - iRC - iCg - iS) / P["Ch"]
        out["vN"][k] = vN; out["vG"][k] = vG; out["iCg"][k] = iCg; out["iS"][k] = iS
        out["iG"][k] = ig; out["iD1"][k] = iD1; out["iRGK"][k] = iRGK
        vN += dvN * dt
        vCg += iCg / P["Cg"] * dt
    return out


def summary(t, vak, o, P, label):
    dt = t[1] - t[0]
    iG = o["iG"]
    print(f"--- {label}")
    print(f"  max dv/dt (rise)     : {np.max(np.gradient(vak, dt)[t < 0.5e-6])/1e6:8.0f} V/us")
    print(f"  peak gate current    : {iG.max():6.2f} A at t = {t[np.argmax(iG)]*1e6:5.2f} us")
    print(f"  peak dynamic (Cg)    : {o['iCg'].max():6.2f} A")
    print(f"  peak static (TVS)    : {o['iS'].max():6.2f} A")
    print(f"  peak D1 current      : {o['iD1'].max():6.1f} A")
    print(f"  peak gate voltage    : {o['vG'].max():6.2f} V,  min {o['vG'].min():6.2f} V")
    for thr in (0.66, 2.0, 5.0):
        m = iG > thr
        if m.any():
            print(f"  i_G > {thr:4.2f} A from {t[m][0]*1e6:6.2f} to {t[m][-1]*1e6:6.2f} us  ({(t[m][-1]-t[m][0])*1e6:5.1f} us)")
    Q = np.trapezoid(iG, t)
    print(f"  gate charge          : {Q*1e6:6.1f} uC")
    E_tvs = np.trapezoid(np.where(o['iS'] > 0, o['iS'] * P['Vst'], 0.0), t)
    E_rs = np.trapezoid(o['iS']**2 * P['Rs'], t)
    E_rgs = np.trapezoid(o['iCg']**2 * P['Rgs'], t)
    print(f"  energy TVS / Rs / Rgs: {E_tvs*1e3:6.1f} / {E_rs*1e3:6.1f} / {E_rgs*1e3:6.2f} mJ (window)")
    print(f"  v_N at end of window : {o['vN'][-1]:7.0f} V  (rest via R_C, tau = {P['RC']*P['Ch']*1e3:.2f} ms, E = {0.5*P['Ch']*o['vN'][-1]**2*1e3:.1f} mJ)")


if __name__ == "__main__":
    dt = 1e-9
    t = np.arange(-3.0e-6, 25.0e-6, dt)
    vak, ts, te, t_on = make_vak(t)
    o = simulate(t, vak, P)
    summary(t, vak, o, P, "nominal: 5 us swing, t_gd = 2.5 us, Rs = 330, Cg = 2.2 nF, Ch = 22 nF")

    # sensitivity: fast edge (2 us swing ~ 4.5 kV/us) and slow edge (20 us swing)
    for tr, lab in ((2.0e-6, "fast edge, 2 us swing"), (20e-6, "slow edge, 20 us swing (t_on forced at 3 us after zero crossing)")):
        vak2, *_ , t_on2 = make_vak(t, tr=tr, tgd=(2.5e-6 if tr < 5e-6 else 3.0e-6))
        o2 = simulate(t, vak2, P)
        summary(t, vak2, o2, P, lab)

    # no-hold-capacitor variant for comparison
    P2 = dict(P); P2["Ch"] = 0.5e-9
    o3 = simulate(t, vak, P2)
    summary(t, vak, o3, P2, "without hold capacitor (Ch = 0.5 nF stray)")

    # -------------------------------------------------------------- figure
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
        "axes.linewidth": 0.6, "grid.linewidth": 0.4, "grid.color": "#c8c8c8",
        "xtick.direction": "in", "ytick.direction": "in",
    })
    tu = t * 1e6
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.2, 4.6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 1.25], hspace=0.12))
    a1.plot(tu, vak / 1e3, color="#333333", lw=1.4)
    a1.plot(tu, o["vN"] / 1e3, color="#333333", lw=1.0, ls=(0, (4, 2)))
    a1.axhline(0, color="#999999", lw=0.5)
    a1.set_ylabel(r"$v_{AK}$, $v_{N}$ [kV]")
    a1.set_ylim(-3.3, 3.3)
    a1.text(-2.8, 0.4, r"$v_{AK}$ (forced)", fontsize=8)
    a1.text(12.5, 1.55, r"$v_{N}$ (node N, hold capacitor)", fontsize=8)
    a1.annotate("", xy=(t_on * 1e6, -1.0), xytext=(t_on * 1e6, -2.9),
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#333333"))
    a1.text(t_on * 1e6 + 0.25, -2.8, r"anode current starts ($t_{gd}$)", fontsize=8)
    a1.grid(True)

    a2.plot(tu, o["iG"], color="#2a78d6", lw=1.6, label=r"$i_G$ total (into gate)")
    a2.plot(tu, o["iCg"], color="#eb6834", lw=1.1, ls=(0, (5, 2)), label=r"dynamic path $C_g\,\mathrm{d}v/\mathrm{d}t$")
    a2.plot(tu, o["iS"], color="#1baf7a", lw=1.1, ls=(0, (2, 1.5)), label=r"static path (TVS, $R_s$)")
    a2.axhline(10.0, color="#999999", lw=0.6, ls=":")
    a2.text(-2.6, 10.25, r"$I_{FGM}$ (datasheet)", fontsize=8, ha="left")
    a2.axhline(2.0, color="#999999", lw=0.6, ls=":")
    a2.text(24.5, 2.25, r"$I_{GM}$ min. recommended", fontsize=8, ha="right")
    a2.set_ylim(-0.5, 11.5)
    a2.set_ylabel(r"gate current [A]")
    a2.set_xlabel(r"time from $v_{AK}$ zero crossing [$\mu$s]")
    a2.set_xlim(-3, 25)
    a2.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.95, edgecolor="#cccccc")
    a2.grid(True)
    fig.savefig("/home/claude/crowbar/gate_response.pdf", bbox_inches="tight")
    print("figure written")
