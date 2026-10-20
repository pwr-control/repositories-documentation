#!/usr/bin/env python3
"""
lrc_filter_example.py -- numbers of the worked examples of the note
"L(RC) filter design guidelines" (research/power_electronics/lrc_filter_design_guidelines).

Part 0: exact switching ripple of the converter-side inductor current (SPWM, SVM, DPWM1).
Part A: grid-side L(RC)L filter of a 400 V / 250 kVA active front end.
Part B: sine (LC / L(RC)) output filter of a 110 kW drive, induction motor and PMSM.

Only numpy is required:  python3 lrc_filter_example.py
"""
import numpy as np

PI = np.pi


# ============================================================================
# PART 0 - inductor current ripple of a two-level VSC
# ============================================================================
def period_ripple(m, theta, mode, Vdc=1.0, T=1.0, L=1.0):
    """Exact piecewise-linear switching ripple of the phase-a inductor current over one
    switching period of a two-level VSC (double-edge regular-sampled PWM, isolated neutral,
    capacitor/grid a short at the switching frequency).

    m     : modulation index, phase peak / (Vdc/2)   (linear range: 1 SPWM, 2/sqrt(3) SVM)
    theta : angle of the reference vector, phase a at its peak for theta = 0
    mode  : 'spwm' (no zero sequence), 'svm' (min-max zero sequence, equal zero-vector split),
            'dpwm1' (60-degree discontinuous, the phase with the largest |v| is clamped)
    Returns (peak-to-peak ripple, mean square of the ripple over the period); with Vdc=T=L=1
    the values are the coefficients k of  delta_i = k Vdc/(fsw L).
    """
    Vref = m * Vdc / 2
    va, vb, vc = (Vref * np.cos(theta + ph) for ph in (0.0, -2 * PI / 3, 2 * PI / 3))
    if mode == 'svm':
        v0 = -(max(va, vb, vc) + min(va, vb, vc)) / 2
    elif mode == 'spwm':
        v0 = 0.0
    elif mode == 'dpwm1':
        vs = [va, vb, vc]
        k = int(np.argmax(np.abs(vs)))
        v0 = (Vdc / 2 if vs[k] > 0 else -Vdc / 2) - vs[k]
    else:
        raise ValueError(mode)
    d = np.array([0.5 + (v + v0) / Vdc for v in (va, vb, vc)])   # leg duty cycles
    tr = (1 - d) * T / 2                                           # rising edges, first half period
    order = np.argsort(tr)
    times = [0.0] + [tr[k] for k in order] + [T / 2]
    s = np.zeros(3)
    segs = []
    for j in range(len(times) - 1):
        van = Vdc * (s[0] - s.sum() / 3)                           # phase-a voltage to load neutral
        segs.append((times[j + 1] - times[j], van))
        if j < 3:
            s[order[j]] = 1
    full = segs + segs[::-1]                                       # symmetric second half
    i, pts, msq, mean = 0.0, [0.0], 0.0, 0.0
    for dur, van in full:
        a, i = i, i + (van - va) * dur / L
        pts.append(i)
        msq += dur * (a * a + a * i + i * i) / 3
        mean += dur * (a + i) / 2
    pts = np.array(pts)
    return pts.max() - pts.min(), msq / T - (mean / T) ** 2


def ripple_coefficients(m, mode, npts=720):
    """(k_pp,max, k_pp,mean, k_rms) over one fundamental period for the given m and modulator."""
    th = np.linspace(0, 2 * PI, npts, endpoint=False)
    pp, ms = zip(*(period_ripple(m, t, mode) for t in th))
    pp = np.array(pp)
    return pp.max(), pp.mean(), np.sqrt(np.mean(ms))


def hdf(m, mode):
    """Harmonic distortion factor (Holmes-Lipo): I_ripple,rms = Vdc/(24 fsw L) sqrt(HDF)."""
    if mode == 'spwm':
        return 1.5 * m ** 2 - 4 * np.sqrt(3) / PI * m ** 3 + 9 / 8 * m ** 4
    return 1.5 * m ** 2 - 4 * np.sqrt(3) / PI * m ** 3 + (27 / 16 - 81 * np.sqrt(3) / (64 * PI)) * m ** 4


def k_peak(m, mode):
    """Closed forms at the phase-voltage peak (theta = 0)."""
    return {'spwm': m / 4 * (1 - m / 2), 'svm': m / 4 * (1 - 3 * m / 4), 'dpwm1': m / 2 * (1 - 3 * m / 4)}[mode]


def k_zero(m):
    """Closed form at the zero crossing of the phase voltage (all modulators)."""
    return np.sqrt(3) * m / 12


print("=" * 78)
print("PART 0 - inductor ripple coefficients k, delta_i = k Vdc/(fsw L)")
print("=" * 78)
print("m       " + "".join(f"{mode:>30s}" for mode in ('spwm', 'svm', 'dpwm1')))
print("        " + "  max pp  mean pp   rms   " * 3)
for mm in (0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 2 / np.sqrt(3)):
    row = f"{mm:<7.3f} "
    for mode in ('spwm', 'svm', 'dpwm1'):
        if mode == 'spwm' and mm > 1:
            row += " " * 30
            continue
        a, b, c = ripple_coefficients(mm, mode)
        row += f"   1/{1/a:5.2f}  1/{1/b:5.2f}  1/{1/c:5.1f} "
    print(row)
print("closed forms, m = 1 (SVM): peak-region k = m/4 (1 - 3m/4) = "
      f"{k_peak(1, 'svm'):.4f}, zero crossing k = sqrt(3) m/12 = {k_zero(1):.4f} = 1/{1/k_zero(1):.2f};"
      f" rms from HDF = {np.sqrt(hdf(1, 'svm'))/24:.4f}, SPWM {np.sqrt(hdf(1, 'spwm'))/24:.4f}")
print("global worst case: SPWM 1/6.93 (m = 1), SVM and DPWM1 1/6 (m = 2/sqrt3; DPWM1 also at m = 2/3)")


# ============================================================================
# damping branch, L-(C1 || C2 R2) filter, n = C2/C1 (see the damped LC filter note)
# ============================================================================
def rH(n):   # optimum r = R2/R0 for the peak of the voltage transfer function
    return np.sqrt((1 + n) * (2 + n) / 2) / n


def rP(n):   # pole placement, real pole at w0
    return (2 + n) / (2 * n)


def rZ(n):   # Erickson optimum for the output impedance
    return np.sqrt((2 + n) * (4 + 3 * n) / (2 * n ** 2 * (4 + n)))


def zetaP(n):
    return n / (2 * np.sqrt(2 * (2 + n)))


# ============================================================================
# PART A - grid-side L(RC)L filter, 400 V / 250 kVA AFE
# ============================================================================
print()
print("=" * 78)
print("PART A - grid-side filter, 400 V / 250 kVA active front end")
print("=" * 78)
Vll, Sn, fg, Vdc, fsw = 400.0, 250e3, 50.0, 700.0, 4000.0
fs = 2 * fsw                    # double-update sampling
wg = 2 * PI * fg
wsw = 2 * PI * fsw
Vph = Vll / np.sqrt(3)
In = Sn / (np.sqrt(3) * Vll)
Ipk = In * np.sqrt(2)
Zb = Vll ** 2 / Sn
Lb = Zb / wg
Cb = 1 / (wg * Zb)
print(f"G1  In = {In:.1f} A rms ({Ipk:.0f} A pk)   Zb = {Zb:.3f} ohm   Lb = {Lb*1e3:.3f} mH   Cb = {Cb*1e3:.3f} mF")
Ssc_min, Ssc_max = 5e6, 20e6
Lg_max = Vll ** 2 / (wg * Ssc_min)
Lg_min = Vll ** 2 / (wg * Ssc_max)
print(f"    grid: Ssc = {Ssc_min/1e6:.0f}-{Ssc_max/1e6:.0f} MVA  ->  Lg = {Lg_min*1e6:.0f}-{Lg_max*1e6:.0f} uH"
      f"  (Ssc/Sn = {Ssc_min/Sn:.0f}-{Ssc_max/Sn:.0f})")

# G2 converter-side inductor from the ripple at rated operation
k_r = 0.25
dI_max = k_r * Ipk
m_rated = 2 * Vph * np.sqrt(2) * 1.05 / Vdc       # rated operation, 5% drop on the filter
kmax, kmean, krms = ripple_coefficients(m_rated, 'svm')
L1_min = kmax * Vdc / (fsw * dI_max)
L1 = 200e-6
dI = kmax * Vdc / (fsw * L1)
I1_sw = krms * Vdc / (fsw * L1)                     # switching-ripple rms, assigned to the fsw group
print(f"G2  ripple target {k_r*100:.0f}% of Ipk = {dI_max:.0f} A pp at m = {m_rated:.2f} (SVM, k = 1/{1/kmax:.2f})"
      f" -> L1 >= {L1_min*1e6:.0f} uH ; chosen L1 = {L1*1e6:.0f} uH ({L1/Lb*100:.1f}% pu): "
      f"dI_pp,max = {dI:.0f} A ({dI/Ipk*100:.0f}%), mean pp = {kmean*Vdc/(fsw*L1):.0f} A, rms = {I1_sw:.1f} A")
print(f"    worst case over m (m = 2/sqrt3): Vdc/(6 fsw L1) = {Vdc/(6*fsw*L1):.0f} A ; indicative Vdc/(8 fsw L1) = {Vdc/(8*fsw*L1):.0f} A")

# G3 capacitor
x_c = 0.05
Cf = 250e-6
Qc = wg * Cf * Vll ** 2
print(f"G3  Cf <= {x_c*100:.0f}% Cb = {x_c*Cb*1e6:.0f} uF -> Cf = {Cf*1e6:.0f} uF (star), Qc = {Qc/1e3:.1f} kvar "
      f"({Qc/Sn*100:.1f}%), delta equivalent {Cf/3*1e6:.0f} uF")

# G4 attenuation requirement and L2
I_lim = 0.003 * In                          # IEEE 519, h >= 35, Isc/IL < 20
k_a = I_lim / I1_sw
L2_min = (1 + 1 / k_a) / (wsw ** 2 * Cf)
L2 = 125e-6
k_a_act = 1 / abs(1 - wsw ** 2 * L2 * Cf)
r = L2 / L1
print(f"G4  Ig(fsw) <= 0.3% In = {I_lim:.2f} A ; I1(fsw) ~ {I1_sw:.1f} A -> k_a <= {k_a:.3f} ({20*np.log10(k_a):.1f} dB)"
      f" -> L2 >= {L2_min*1e6:.0f} uH ; chosen L2 = {L2*1e6:.0f} uH (r = {r:.2f}), k_a = {k_a_act:.3f},"
      f" Ig(fsw) = {k_a_act*I1_sw:.2f} A = {k_a_act*I1_sw/In*100:.2f}% In")
print(f"    Liserre form 1/|1+r(1-w^2 L1 Cf)| = {1/abs(1+r*(1-wsw**2*L1*Cf)):.3f}")


def fres(L1, L2, C):
    return np.sqrt((L1 + L2) / (L1 * L2 * C)) / (2 * PI)


fr0 = fres(L1, L2, Cf)
fr_min = fres(L1, L2 + Lg_max, Cf)
fr_max = fres(L1, L2 + Lg_min, Cf)
print(f"G5  f_res = {fr0:.0f} Hz (Lg = 0); with Lg: {fr_min:.0f}-{fr_max:.0f} Hz ; "
      f"limits 10 fg = {10*fg:.0f} Hz, fsw/2 = {fsw/2:.0f} Hz ; critical fs/6 = {fs/6:.0f} Hz (fs = {fs/1e3:.0f} kHz)")
f_trap = 1 / (2 * PI * np.sqrt(L2 * Cf))
print(f"    current trap f2 = 1/(2 pi sqrt(L2 Cf)) = {f_trap:.0f} Hz ; L1+L2 = {(L1+L2)/Lb*100:.1f}% pu")

# G6 damping (a) series resistor
Rd = 1 / (3 * 2 * PI * fr0 * Cf)
Ic_g = wg * Cf * Vph
Ic_sw = I1_sw
P_Rd = 3 * Rd * (Ic_g ** 2 + Ic_sw ** 2)
Zc_sw = np.sqrt(Rd ** 2 + 1 / (wsw * Cf) ** 2)
k_a_Rd = Zc_sw / np.sqrt(Rd ** 2 + (wsw * L2 - 1 / (wsw * Cf)) ** 2)
zeta_Rd = Rd / 2 * np.sqrt(Cf / (L1 * L2 / (L1 + L2)))
print(f"G6a series Rd = 1/(3 w_res Cf) = {Rd*1e3:.0f} mohm ; zeta ~ {zeta_Rd:.2f} ; Ic(50 Hz) = {Ic_g:.1f} A, Ic(fsw) ~ {Ic_sw:.1f} A"
      f" -> P_Rd = {P_Rd:.0f} W ({P_Rd/Sn*100:.2f}%) ; attenuation at fsw with Rd: {k_a_Rd:.3f}")
# (b) R-C parallel branch, n = 1, same total capacitance
n = 1.0
C1 = Cf / (1 + n)
C2 = n * C1
Lp = L1 * L2 / (L1 + L2)
R0 = np.sqrt(Lp / C1)
R2 = rP(n) * R0
I2 = wg * C2 * Vph / np.sqrt(1 + (wg * C2 * R2) ** 2)
P2_f = 3 * R2 * I2 ** 2
Vc_sw = I1_sw / (wsw * C1)
P2_sw = 3 * Vc_sw ** 2 / R2
f0_1 = 1 / (2 * PI * np.sqrt(Lp * C1))
print(f"G6b R-C branch n = {n:.0f}: C1 = C2 = {C1*1e6:.0f} uF, Lp = L1||L2 = {Lp*1e6:.1f} uH, R0 = {R0:.3f} ohm, "
      f"f0(C1) = {f0_1:.0f} Hz ; rP = {rP(n):.2f} (rZ = {rZ(n):.2f}, rH = {rH(n):.2f}) -> R2 = {R2:.2f} ohm, zeta = {zetaP(n):.2f}")
print(f"    I2(50 Hz) = {I2:.1f} A -> P2,fund = {P2_f:.0f} W ; Vc(fsw) ~ {Vc_sw:.1f} V -> P2,sw ~ {P2_sw:.0f} W ; "
      f"total {P2_f+P2_sw:.0f} W ({(P2_f+P2_sw)/Sn*100:.2f}%)")

# G7 ratings
print(f"G7  L1: {In:.0f} A rms, peak {Ipk + Vdc/(6*fsw*L1)/2:.0f} A (saturation, worst-case ripple), "
      f"ripple rms {I1_sw:.0f} A at {fsw/1e3:.0f} kHz ; L2: {In:.0f} A rms, line reactor ; "
      f"Cf: >= {1.1*Vll:.0f} V ac, Ic ~ {np.sqrt(Ic_g**2+Ic_sw**2):.0f} A rms")

# ============================================================================
# PART B - sine filter between inverter and motor
# ============================================================================
print()
print("=" * 78)
print("PART B - sine filter, 110 kW / 400 V drive, induction motor and PMSM")
print("=" * 78)
Vdc, Vll, Pn = 650.0, 400.0, 110e3
Vph = Vll / np.sqrt(3)

# cable / reflected wave
v_c = 150.0            # m/us
tr_inv = 0.2           # us, IGBT edge
l_cable = 150.0        # m
Gm = 0.9
l_crit = v_c * tr_inv / 2
Vpk = Vdc * (1 + Gm * min(1, l_cable / l_crit))
print(f"cable: v = {v_c:.0f} m/us, tr = {tr_inv:.1f} us -> l_crit = {l_crit:.0f} m ; l = {l_cable:.0f} m, Gm = {Gm}"
      f" -> Vpk ~ {Vpk:.0f} V ({Vpk/Vdc:.2f} Vdc) ; NEMA MG1-31 limit {3.1*Vll:.0f} V pk, MG1-30 1000 V")
OF_adm = 0.2
l_crit_req = l_cable * Gm / OF_adm
tr_req = 2 * l_crit_req / v_c
print(f"       overshoot <= {OF_adm*100:.0f}% -> l_crit >= {l_crit_req:.0f} m -> tr >= {tr_req:.1f} us"
      f" (dv/dt <= {Vdc/tr_req:.0f} V/us) ; dv/dt filter f0 <= 0.3/tr = {0.3/tr_req*1e3:.0f} kHz")
Cdv = 0.22e-6
print(f"       dv/dt filter with C = {Cdv*1e6:.2f} uF: loss bound 3 fsw C Vdc^2 = {3*4000*Cdv*Vdc**2:.0f} W at 4 kHz")

cases = {
    "IM  110 kW 4-pole": dict(In=195.0, f1max=75.0, Lsig_pu=0.18, fsw=4000.0, Rm_pu=0.10, Lf=230e-6),
    "PMSM 110 kW (IPM)": dict(In=180.0, f1max=200.0, Lsig_pu=0.30, fsw=8000.0, Rm_pu=0.02, Lf=150e-6),
}
krms_full = ripple_coefficients(2 / np.sqrt(3), 'svm')[2]   # rms ripple at full modulation
for name, c in cases.items():
    In, f1max, fsw = c["In"], c["f1max"], c["fsw"]
    w1max = 2 * PI * f1max
    wsw = 2 * PI * fsw
    Zb = Vph / In
    Lb = Zb / (2 * PI * 50)
    Cb = 1 / (2 * PI * 50 * Zb)
    Lsig = c["Lsig_pu"] * Lb
    Rm = c["Rm_pu"] * Zb
    f0_lo, f0_hi = 8 * f1max, fsw / 4
    print(f"\n{name}: In = {In:.0f} A, f1max = {f1max:.0f} Hz, fsw = {fsw/1e3:.0f} kHz, Zb = {Zb:.3f} ohm, "
          f"Lb = {Lb*1e3:.2f} mH, L_hf = {Lsig*1e3:.2f} mH ({c['Lsig_pu']} pu)")
    print(f"M2  window 8 f1max = {f0_lo:.0f} Hz <= f0 <= fsw/4 = {f0_hi:.0f} Hz" + ("  (infeasible)" if f0_lo > f0_hi else ""))
    f0_t = 0.9 * f0_hi
    # M3 inductor: drop at f1max and ripple
    eps = 0.08
    Lf_max = eps * Vph / (w1max * In)
    Lf = c["Lf"]
    dI = Vdc / (6 * fsw * Lf)                       # worst case over m (full voltage at f1max)
    I_rms_sw = krms_full * Vdc / (fsw * Lf)
    print(f"M3  drop <= {eps*100:.0f}% at f1max -> Lf <= {Lf_max*1e6:.0f} uH ; chosen Lf = {Lf*1e6:.0f} uH ({Lf/Lb*100:.1f}% pu): "
          f"drop {w1max*Lf*In/Vph*100:.1f}% at f1max, {2*PI*50*Lf*In/Vph*100:.1f}% at 50 Hz ; "
          f"dI_pp,max = {dI:.0f} A ({dI/(In*np.sqrt(2))*100:.0f}% of Ipk), ripple rms {I_rms_sw:.1f} A")
    # M4 capacitor
    Cf = 1 / ((2 * PI * f0_t) ** 2 * Lf)
    Cf = round(Cf * 1e6 / 10) * 10e-6
    f0 = 1 / (2 * PI * np.sqrt(Lf * Cf))
    Ic_max = w1max * Cf * Vph
    print(f"M4  f0 target {f0_t:.0f} Hz -> Cf = {Cf*1e6:.0f} uF (star, {Cf/Cb*100:.1f}% Cb), f0 = {f0:.0f} Hz ; "
          f"Ic = {2*PI*50*Cf*Vph:.1f} A at 50 Hz, {Ic_max:.1f} A at f1max ({Ic_max/In*100:.0f}% In)")
    # M5 loaded resonance and attenuation
    fp = f0 * np.sqrt(1 + Lf / Lsig)
    Lp = Lf * Lsig / (Lf + Lsig)
    att = 1 / ((fsw / fp) ** 2 - 1)
    G = Rm / (Rm ** 2 + (2 * PI * fp * Lsig) ** 2)
    zeta_m = 0.5 * G * np.sqrt(Lp / Cf)
    print(f"M5  loaded resonance fp = f0 sqrt(1 + Lf/L_hf) = {fp:.0f} Hz ({fp/f0:.2f} f0) ; |vc/vi|(fsw) = {att:.3f} ({20*np.log10(att):.0f} dB) ; "
          f"Lp = Lf||L_hf = {Lp*1e6:.0f} uH ; motor damping zeta_m ~ {zeta_m:.4f} (Rm = {c['Rm_pu']} pu)")
    # M6 damping
    Rd = 1 / (3 * 2 * PI * fp * Cf)
    Ic_sw = I_rms_sw
    P_Rd = 3 * Rd * (Ic_max ** 2 + Ic_sw ** 2)
    zeta_Rd = Rd / 2 * np.sqrt(Cf / Lp)
    n = 1.0
    C1 = Cf; C2 = n * C1
    R0 = np.sqrt(Lp / C1); R2 = rP(n) * R0
    I2 = w1max * C2 * Vph / np.sqrt(1 + (w1max * C2 * R2) ** 2)
    P2 = 3 * R2 * I2 ** 2 + 3 * (Ic_sw / (wsw * C1)) ** 2 / R2
    print(f"M6  series Rd = 1/(3 wp Cf) = {Rd*1e3:.0f} mohm (zeta ~ {zeta_Rd:.2f}): P = {P_Rd:.0f} W ({P_Rd/Pn*100:.2f}%) at f1max ; "
          f"R-C branch n = 1 (C2 = {C2*1e6:.0f} uF added, R2 = {R2:.2f} ohm, zeta = {zetaP(n):.2f}): P = {P2:.0f} W ({P2/Pn*100:.2f}%) at f1max")
    # M7 control
    print(f"M7  inverter current bandwidth <= fsw/10 = {fsw/10:.0f} Hz ; stator-current loop <= fp/5 = {fp/5:.0f} Hz ; "
          f"HF injection f_inj <= fp/3 = {fp/3:.0f} Hz")
