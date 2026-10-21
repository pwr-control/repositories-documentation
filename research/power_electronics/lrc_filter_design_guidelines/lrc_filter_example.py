#!/usr/bin/env python3
"""
lrc_filter_example.py -- numbers of the worked examples of the note
"L(RC) filter design guidelines" (research/power_electronics/lrc_filter_design_guidelines).

Part 0: exact switching ripple of the converter-side inductor current (SPWM, SVM, DPWM1),
        including the split of the ripple into the carrier clusters (FFT over the fundamental).
Part 1: grid-side L(RC)L sine filter of a 480 V / 60 Hz / 500 kVA active front end, fsw = 2 kHz.
Part 2: L(RC) sine filter of a 450 kW / 480 V drive at fsw = 2 kHz, induction motor and PMSM.
Part 3: comparison table AFE / IM / PMSM.

Only numpy is required:  python3 lrc_filter_example.py
"""
import numpy as np

PI = np.pi
SQ2, SQ3 = np.sqrt(2.0), np.sqrt(3.0)


# ============================================================================
# PART 0 - inductor current ripple of a two-level VSC
# ============================================================================
def period_wave(m, theta, mode, Vdc=1.0, T=1.0, L=1.0, npts=128):
    """Exact piecewise-linear switching ripple of the phase-a inductor current over one
    switching period of a two-level VSC (double-edge regular-sampled PWM, isolated neutral,
    capacitor/grid a short at the switching frequency), sampled at npts points.

    m     : modulation index, phase peak / (Vdc/2)   (linear range: 1 SPWM, 2/sqrt(3) SVM)
    theta : angle of the reference vector, phase a at its peak for theta = 0
    mode  : 'spwm' (no zero sequence), 'svm' (min-max zero sequence, equal zero-vector split),
            'dpwm1' (60-degree discontinuous, the phase with the largest |v| is clamped)
    With Vdc = T = L = 1 the values are the coefficients k of  delta_i = k Vdc/(fsw L).
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
    tb, ib, i = [0.0], [0.0], 0.0
    for dur, van in full:
        i += (van - va) * dur / L
        tb.append(tb[-1] + dur)
        ib.append(i)
    return np.interp(np.arange(npts) * T / npts, np.array(tb), np.array(ib))


def period_ripple(m, theta, mode):
    """(peak-to-peak ripple, mean square of the ripple) over one switching period, Vdc = T = L = 1."""
    w = period_wave(m, theta, mode, npts=4096)
    return w.max() - w.min(), np.mean((w - w.mean()) ** 2)


def ripple_coefficients(m, mode, npts=720):
    """(k_pp,max, k_pp,mean, k_rms) over one fundamental period for the given m and modulator."""
    th = np.linspace(0, 2 * PI, npts, endpoint=False)
    pp, ms = zip(*(period_ripple(m, t, mode) for t in th))
    pp = np.array(pp)
    return pp.max(), pp.mean(), np.sqrt(np.mean(ms))


def cluster_split(m, mode, N, nf=3, npts=128):
    """Ripple waveform over nf fundamental periods with pulse ratio N = fsw/f1 (N*nf integer),
    FFT, and RMS of the carrier clusters k*fsw +- fsw/2, k = 1..4, plus the largest lines.
    Returns (k_rms total, [k_rms of clusters 1..4], dict{harmonic order: k_rms of the line})."""
    nsw = int(round(N * nf))
    assert abs(nsw - N * nf) < 1e-9, "N*nf must be an integer"
    w = np.concatenate([period_wave(m, 2 * PI * (j + 0.5) / N, mode, npts=npts) for j in range(nsw)])
    w -= w.mean()
    X = np.fft.rfft(w) / len(w)
    f = np.fft.rfftfreq(len(w), d=1.0 / npts)       # in units of fsw
    p = 2 * np.abs(X) ** 2
    tot = np.sqrt(p.sum())
    cl = [np.sqrt(p[(f > k - 0.5) & (f <= k + 0.5)].sum()) for k in (1, 2, 3, 4)]
    lines = {}
    for i in np.argsort(p)[::-1][:8]:
        lines[round(f[i] * N, 2)] = np.sqrt(p[i])
    return tot, cl, lines


def hdf(m, mode):
    """Harmonic distortion factor (Holmes-Lipo): I_ripple,rms = Vdc/(24 fsw L) sqrt(HDF)."""
    if mode == 'spwm':
        return 1.5 * m ** 2 - 4 * SQ3 / PI * m ** 3 + 9 / 8 * m ** 4
    return 1.5 * m ** 2 - 4 * SQ3 / PI * m ** 3 + (27 / 16 - 81 * SQ3 / (64 * PI)) * m ** 4


def k_peak(m, mode):
    """Closed forms at the phase-voltage peak (theta = 0)."""
    return {'spwm': m / 4 * (1 - m / 2), 'svm': m / 4 * (1 - 3 * m / 4), 'dpwm1': m / 2 * (1 - 3 * m / 4)}[mode]


def k_zero(m):
    """Closed form at the zero crossing of the phase voltage (all modulators)."""
    return SQ3 * m / 12


def inv(x):
    return f"1/{1/x:.1f}"


print("=" * 78)
print("PART 0 - inductor ripple coefficients k, delta_i = k Vdc/(fsw L)")
print("=" * 78)
print("m       " + "".join(f"{mode:>30s}" for mode in ('spwm', 'svm', 'dpwm1')))
print("        " + "  max pp  mean pp   rms   " * 3)
for mm in (0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 2 / SQ3):
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
print("\ncarrier clusters (pulse ratio fsw/f1 = 33.33, SVM):")
for mm in (1.0, 1.03, 2 / SQ3):
    tot, cl, lines = cluster_split(mm, 'svm', 2000 / 60)
    print(f"m = {mm:.3f}: total {inv(tot)}, clusters fsw {inv(cl[0])} ({cl[0]/tot*100:.0f}%), 2fsw {inv(cl[1])},"
          f" 3fsw {inv(cl[2])}, 4fsw {inv(cl[3])}")
    print("          largest lines (order h: k_rms): " + ", ".join(f"{h}: {inv(v)}" for h, v in sorted(lines.items())))


# ============================================================================
# damping branch, L-(C1 || C2 R2) filter, n = C2/C1 (see the damped LC filter note)
# ============================================================================
def rho_H(n):   # optimum rho = R2/R0 for the peak of the voltage transfer function
    return np.sqrt((1 + n) * (2 + n) / 2) / n


def rho_P(n):   # pole placement, real pole at w0
    return (2 + n) / (2 * n)


def rho_Z(n):   # Erickson optimum for the output impedance
    return np.sqrt((2 + n) * (4 + 3 * n) / (2 * n ** 2 * (4 + n)))


def zeta_P(n):
    return n / (2 * np.sqrt(2 * (2 + n)))


def Zc_rc(w, C1, C2, R2):
    """shunt impedance of C1 || (R2 + C2)"""
    z1 = 1 / (1j * w * C1)
    z2 = R2 + 1 / (1j * w * C2)
    return z1 * z2 / (z1 + z2)


def Zc_rd(w, C, Rd):
    return Rd + 1 / (1j * w * C)


def lcl_ratio(w, Z1, Z2, Zc):
    """i2/i1 and i1/v1 of the generic L - Zc - Z2 structure (eq. tf of the note)"""
    i2_i1 = Zc / (Zc + Z2)
    i1_v1 = (Zc + Z2) / (Z1 * Z2 + Zc * (Z1 + Z2))
    return i2_i1, i1_v1


def peak_tf(L1, L2, Zc_fun, fmax=5000.0):
    """peak of |i2/i1| (resonance) of the damped LCL over 100 Hz .. fmax"""
    f = np.linspace(100, fmax, 20000)
    w = 2 * PI * f
    g = np.abs(Zc_fun(w) / (Zc_fun(w) + 1j * w * L2))
    # resonance of the L1 - Zc - L2 structure: |i2/v1| normalised to the undamped slope
    h = np.abs(1 / (1j * w * L1 * 1j * w * L2 + Zc_fun(w) * (1j * w * (L1 + L2))))
    i = np.argmax(h)
    return f[i], h[i], g


# ============================================================================
# PART 1 - grid-side L(RC)L sine filter, 480 V / 60 Hz / 500 kVA AFE, fsw = 2 kHz
# ============================================================================
print()
print("=" * 78)
print("PART 1 - AFE grid-side L(RC)L filter: 480 V, 60 Hz, 500 kVA, Vdc = 800 V, fsw = 2 kHz")
print("=" * 78)
Vll, fg, Sn, Vdc, fsw = 480.0, 60.0, 500e3, 800.0, 2000.0
fs = 2 * fsw                      # double update
wg, wsw = 2 * PI * fg, 2 * PI * fsw
Vph = Vll / SQ3
In = Sn / (SQ3 * Vll)
Ipk = SQ2 * In
Zb = Vll ** 2 / Sn
Lb = Zb / wg
Cb = 1 / (wg * Zb)
print(f"A1  Vph = {Vph:.1f} V, In = {In:.0f} A ({Ipk:.0f} A pk), Zb = {Zb:.3f} ohm, Lb = {Lb*1e6:.0f} uH, Cb = {Cb*1e3:.2f} mF")
Ssc_min, Ssc_max = 12e6, 48e6
Lg_max, Lg_min = Vll ** 2 / (wg * Ssc_min), Vll ** 2 / (wg * Ssc_max)
print(f"    grid Ssc = {Ssc_min/1e6:.0f}-{Ssc_max/1e6:.0f} MVA (Isc/IL = {Ssc_min/Sn:.0f}-{Ssc_max/Sn:.0f}) -> Lg = {Lg_min*1e6:.1f}-{Lg_max*1e6:.1f} uH"
      f" ({Lg_min/Lb*100:.1f}-{Lg_max/Lb*100:.1f}% pu); pulse ratio fsw/f1 = {fsw/fg:.2f}, fs = {fs:.0f} Hz, fs/6 = {fs/6:.0f} Hz")

# A2 modulation index and ripple coefficients (drop of the filter in quadrature with the current, unity PF)
eps_L = 0.3
m_r = 2 * SQ2 * Vph * np.sqrt(1 + eps_L ** 2) / Vdc
kmax, kmean, krms = ripple_coefficients(m_r, 'svm')
tot, cl, lines = cluster_split(m_r, 'svm', fsw / fg)
h_p, h_m = round(fsw / fg + 2, 2), round(fsw / fg - 2, 2)
k_line, k_line_m = lines[h_p], lines[h_m]
k_cl1 = cl[0]
k_line2 = lines[round(2 * fsw / fg + 1, 2)]
print(f"A2  m = 2 sqrt2 Vph sqrt(1 + eps_L^2) / Vdc = {m_r:.3f} (eps_L = {eps_L}); k_pp,max = sqrt3 m/12 = {kmax:.4f} = {inv(kmax)}, k_pp,mean = {inv(kmean)},"
      f" k_rms = {inv(krms)} (HDF: {inv(np.sqrt(hdf(m_r,'svm'))/24)})")
print(f"    first cluster {inv(k_cl1)} ({k_cl1/tot*100:.0f}% of the ripple rms); lines fsw-2f1 (h = {h_m}): {inv(k_line_m)},"
      f" fsw+2f1 (h = {h_p}): {inv(k_line)}; second cluster {inv(cl[1])}, lines 2fsw+-f1: {inv(k_line2)}")

# A3 converter-side inductor
for k_r in (0.25, 0.40):
    print(f"A3  k_r = {k_r:.2f}: L1 >= k Vdc/(fsw k_r Ipk) = {kmax*Vdc/(fsw*k_r*Ipk)*1e6:.0f} uH ({kmax*Vdc/(fsw*k_r*Ipk)/Lb*100:.1f}% pu)")
L1 = 180e-6
dI_max = kmax * Vdc / (fsw * L1)
dI_mean = kmean * Vdc / (fsw * L1)
I_rip = krms * Vdc / (fsw * L1)
I1_cl = k_cl1 * Vdc / (fsw * L1)
I1_line = k_line * Vdc / (fsw * L1)
I1_line_m = k_line_m * Vdc / (fsw * L1)
I1_line2 = k_line2 * Vdc / (fsw * L1)
print(f"    chosen L1 = {L1*1e6:.0f} uH ({L1/Lb*100:.1f}% pu): dI_pp,max = {dI_max:.0f} A ({dI_max/Ipk*100:.0f}% of Ipk), mean {dI_mean:.0f} A,"
      f" I_rip,rms = {I_rip:.1f} A ({I_rip/In*100:.1f}% In); worst case Vdc/(6 fsw L1) = {Vdc/(6*fsw*L1):.0f} A, rule Vdc/(8 fsw L1) = {Vdc/(8*fsw*L1):.0f} A")
print(f"    first cluster I1,sw = {I1_cl:.1f} A; lines: fsw-2f1 {I1_line_m:.1f} A, fsw+2f1 {I1_line:.1f} A, 2fsw+-f1 {I1_line2:.1f} A; saturation current >= {Ipk + dI_max/2:.0f} A")

# A4 capacitor
x = 0.05
C1 = 280e-6
Qc = wg * C1 * Vll ** 2
Ic_f = wg * C1 * Vph
print(f"A4  C1 <= {x} Cb = {x*Cb*1e6:.0f} uF -> C1 = {C1*1e6:.0f} uF star ({C1/3*1e6:.1f} uF delta), x = {C1/Cb*100:.2f}%, Qc = {Qc/1e3:.1f} kvar,"
      f" Ic(f1) = {Ic_f:.1f} A; 1/(wsw^2 C1) = {1/(wsw**2*C1)*1e6:.1f} uH")

# A5 attenuation requirement: IEEE 519, 20 < Isc/IL < 50 -> 1.0 % (23 <= h < 35), 0.5 % (35 <= h <= 50)
lim_lo, lim_hi = 0.010 * In, 0.005 * In
w_line = 2 * PI * (fsw + 2 * fg)
w_line_m = 2 * PI * (fsw - 2 * fg)
k_a_req = lim_hi / I1_line
k_a_req_m = lim_lo / I1_line_m
L2p_req = (1 + 1 / k_a_req) / (w_line ** 2 * C1)
L2p_req_m = (1 + 1 / k_a_req_m) / (w_line_m ** 2 * C1)
print(f"A5  limits: {lim_lo:.1f} A (23 <= h < 35), {lim_hi:.1f} A (h >= 35); binding line fsw+2f1 = {fsw+2*fg:.0f} Hz (h = {h_p}):"
      f" k_a <= {k_a_req:.3f} ({20*np.log10(k_a_req):.1f} dB) -> L2' >= (1 + 1/k_a)/(w^2 C1) = {L2p_req*1e6:.0f} uH; line fsw-2f1: k_a <= {k_a_req_m:.3f} -> L2' >= {L2p_req_m*1e6:.0f} uH")
A = k_line * Vdc / fsw            # I1_line = A / L1
a = w_line ** 2 * C1
L1o = np.sqrt(A / (a * lim_hi))   # minimum L1 + L2' of the plain LCL: L2' = L1 + 1/(w^2 C1)
print(f"    minimum total inductance (plain LCL, this C1): L1 = {L1o*1e6:.0f} uH, L2' = L1 + 1/(w^2 C1) = {(L1o + 1/a)*1e6:.0f} uH, total {(2*L1o+1/a)*1e6:.0f} uH ({(2*L1o+1/a)/Lb*100:.0f}% pu)")
L2 = 180e-6
for Lg, tag in ((Lg_min, "Lg,min"), (Lg_max, "Lg,max")):
    L2p = L2 + Lg
    ka = 1 / (w_line ** 2 * L2p * C1 - 1)
    kam = 1 / (w_line_m ** 2 * L2p * C1 - 1)
    print(f"    chosen L2 = {L2*1e6:.0f} uH (r = {L2/L1:.2f}), {tag}: L2' = {L2p*1e6:.1f} uH, k_a(fsw+2f1) = {ka:.3f} ({20*np.log10(ka):.1f} dB),"
          f" I2(fsw+2f1) = {ka*I1_line:.2f} A ({ka*I1_line/In*100:.2f}% In); I2(fsw-2f1) = {kam*I1_line_m:.2f} A ({kam*I1_line_m/In*100:.2f}%)")
V1 = abs(Vph + 1j * wg * (L1 + L2) * In)
print(f"    L1 + L2 = {(L1+L2)*1e6:.0f} uH = {(L1+L2)/Lb*100:.1f}% pu: quadrature drop, V1 = {V1:.0f} V ph (+{(V1/Vph-1)*100:.1f}%), m = {2*SQ2*V1/Vdc:.3f} at unity PF (limit 1.155)")


def fres(L1, L2, C):
    return np.sqrt((L1 + L2) / (L1 * L2 * C)) / (2 * PI)


# A6 resonance
fr_min, fr_max, fr_0 = fres(L1, L2 + Lg_max, C1), fres(L1, L2 + Lg_min, C1), fres(L1, L2, C1)
f2_min = 1 / (2 * PI * np.sqrt((L2 + Lg_max) * C1))
print(f"A6  f_res = {fr_0:.0f} Hz (Lg = 0), {fr_min:.0f} Hz (Lg,max) ... {fr_max:.0f} Hz (Lg,min); bounds 10 f1 = {10*fg:.0f} Hz, fsw/2 = {fsw/2:.0f} Hz;"
      f" fs/6 = {fs/6:.0f} Hz (0.8-1.2: {0.8*fs/6:.0f}-{1.2*fs/6:.0f} Hz) -> band above fs/6: i1 feedback inherently stable; f2 = {f2_min:.0f} Hz")

# A7 damping: R-C branch added to C1 (L(RC)L), n = C2/C1, vs series resistor
Lp_min = L1 * (L2 + Lg_min) / (L1 + L2 + Lg_min)
Lp_max = L1 * (L2 + Lg_max) / (L1 + L2 + Lg_max)
R0 = np.sqrt(Lp_min / C1)
print(f"A7  Lp = L1 || L2' = {Lp_min*1e6:.1f} uH (Lg,min) ... {Lp_max*1e6:.1f} uH (Lg,max); R0 = sqrt(Lp/C1) = {R0:.3f} ohm")
AFE_damp = {}
for n in (1.0, 0.5, 0.25):
    C2 = n * C1
    R2 = rho_P(n) * R0
    I2f = wg * C2 * Vph / np.sqrt(1 + (wg * C2 * R2) ** 2)
    P_f = 3 * R2 * I2f ** 2
    Zc = Zc_rc(wsw, C1, C2, R2)
    vc_sw = I1_cl * abs(Zc)
    Ib = vc_sw / abs(R2 + 1 / (1j * wsw * C2))
    P_sw = 3 * R2 * Ib ** 2
    ka_d = abs(Zc_rc(w_line, C1, C2, R2) / (Zc_rc(w_line, C1, C2, R2) + 1j * w_line * (L2 + Lg_min)))
    fpk, hpk, _ = peak_tf(L1, L2 + Lg_min, lambda w: Zc_rc(w, C1, C2, R2))
    Qtot = wg * (C1 + C2) * Vll ** 2
    AFE_damp[n] = (C2, R2, zeta_P(n), P_f + P_sw, ka_d)
    print(f"    R-C n = {n}: C2 = {C2*1e6:.0f} uF added (total {(C1+C2)/Cb*100:.1f}% Cb, Qc = {Qtot/1e3:.1f} kvar), rho_P = {rho_P(n):.2f} (rho_Z {rho_Z(n):.2f}, rho_H {rho_H(n):.2f})"
          f" -> R2 = {R2:.2f} ohm, zeta = {zeta_P(n):.3f}")
    print(f"           I2(f1) = {I2f:.1f} A -> P_f = {P_f:.0f} W; v_c,sw = {vc_sw:.1f} V, I_branch,sw = {Ib:.1f} A -> P_sw = {P_sw:.0f} W; total {P_f+P_sw:.0f} W ({(P_f+P_sw)/Sn*100:.2f}% Sn);"
          f" k_a(fsw+2f1) with the branch = {ka_d:.3f} -> I2 = {ka_d*I1_line:.2f} A; resonant peak at {fpk:.0f} Hz")
Rd = 1 / (3 * 2 * PI * fr_max * C1)
zeta_Rd = Rd / 2 * np.sqrt(C1 / Lp_min)
P_Rd = 3 * Rd * (Ic_f ** 2 + I_rip ** 2)
ka_Rd = abs(Zc_rd(w_line, C1, Rd) / (Zc_rd(w_line, C1, Rd) + 1j * w_line * (L2 + Lg_min)))
print(f"    series Rd = 1/(3 w_res C1) = {Rd*1e3:.0f} mohm, zeta = {zeta_Rd:.2f}: P = 3 Rd (Ic^2 + Irip^2) = {P_Rd:.0f} W ({P_Rd/Sn*100:.2f}%),"
      f" k_a(fsw+2f1) = {ka_Rd:.3f} -> I2 = {ka_Rd*I1_line:.2f} A (limit {lim_hi:.1f} A)")

# A8 ratings and control
n_sel = 0.25
C2, R2, zeta_sel, P_sel, ka_sel = AFE_damp[n_sel]
vc_sw_pk = SQ2 * I1_cl * abs(Zc_rc(wsw, C1, C2, R2))
print(f"A8  selected n = {n_sel}: C1 = {C1*1e6:.0f} uF, C2 = {C2*1e6:.0f} uF, R2 = {R2:.2f} ohm, zeta = {zeta_sel:.3f}, P = {P_sel:.0f} W")
print(f"    L1: {In:.0f} A rms, saturation >= {Ipk + dI_max/2:.0f} A, ripple {I_rip:.0f} A rms at 2 kHz; L2: {In:.0f} A line reactor;"
      f" C1: >= {1.1*Vll:.0f} V ac, v_c,sw,pk = {vc_sw_pk:.0f} V, I = {np.sqrt(Ic_f**2+I1_cl**2):.0f} A rms; C2: {wg*C2*Vph:.1f} A at f1")
print(f"    control: fc <= f_res,min/3 = {fr_min/3:.0f} Hz (fsw/{fsw/(fr_min/3):.1f}); PLL <= 20 Hz at Ssc,min; delay 1.5 Ts = {1.5/fs*1e6:.0f} us")

# A9 how to lighten the filter: (i) trap at fsw, (ii) fsw = 4 kHz
Lt = 1 / (wsw ** 2 * C1)
w2 = 2 * PI * (2 * fsw + fg)
Zt = 1j * (w2 * Lt - 1 / (w2 * C1))
ka2_req = lim_hi / I1_line2
L2p_trap = (abs(Zt) / ka2_req - abs(Zt)) / w2
print(f"A9  (i) trap Lt = 1/(wsw^2 C1) = {Lt*1e6:.1f} uH in series with C1: lines at fsw+-2f1 notched; at 2fsw+f1 ({2*fsw+fg:.0f} Hz) Z_c = j{Zt.imag:.3f} ohm (inductive),"
      f" I1 = {I1_line2:.1f} A -> k_a <= {ka2_req:.3f} -> L2' >= {L2p_trap*1e6:.0f} uH (inductive divider)")
# (ii) 4 kHz: same C1, same limit (h = 64-70 > 50: 0.5 %), same k_r
fsw4 = 4000.0
tot4, cl4, lines4 = cluster_split(m_r, 'svm', fsw4 / fg)
k_line4 = lines4[round(fsw4 / fg + 2, 2)]
w4 = 2 * PI * (fsw4 + 2 * fg)
L1_4 = kmax * Vdc / (fsw4 * 0.40 * Ipk)
I1_4 = k_line4 * Vdc / (fsw4 * L1_4)
L2p_4 = (1 + I1_4 / lim_hi) / (w4 ** 2 * C1)
print(f"    (ii) fsw = 4 kHz, same k_r and C1: L1 = {L1_4*1e6:.0f} uH ({L1_4/Lb*100:.1f}%), I1(fsw+2f1) = {I1_4:.1f} A -> L2' >= {L2p_4*1e6:.0f} uH ({L2p_4/Lb*100:.1f}%),"
      f" total {(L1_4+L2p_4)/Lb*100:.0f}% pu instead of {(L1+L2)/Lb*100:.0f}%")
AFE = dict(Ln=L1, Lpu=L1/Lb, L2=L2, L2pu=L2/Lb, C=C1, Cpu=C1/Cb, C2=C2, R2=R2, zeta=zeta_sel, Ploss=P_sel/Sn, fr=fr_0, ratio=fr_0/fsw, ka=ka_sel, dI=dI_max/Ipk, Irip=I_rip/In,
           Ic=Ic_f/In, eps=(L1+L2)/Lb, In=In, Ipk=Ipk + dI_max/2, m=2*SQ2*V1/Vdc, bw=fr_min/3, Vdc=Vdc)

# ============================================================================
# PART 1b - L(RC) without L2 (grid as second inductance): check of a 690 V / 250 kW / 4 kHz design
#           and its version aligned with the criteria of Example 1
# ============================================================================
print()
print("=" * 78)
print("PART 1b - 690 V / 250 kW / 4 kHz AFE: L(RC) with the grid as L2, and the aligned L(RC)L")
print("=" * 78)
Vll, fg, Sn, Vdc, fsw = 690.0, 50.0, 275e3, 1100.0, 4000.0
fs = 2 * fsw
wg, wsw = 2 * PI * fg, 2 * PI * fsw
Vph = Vll / SQ3
In = Sn / (SQ3 * Vll)
Ipk = SQ2 * In
Zb = Vll ** 2 / Sn
Lb = Zb / wg
Cb = 1 / (wg * Zb)
m_r = 2 * SQ2 * Vph * 1.02 / Vdc
kmax, kmean, krms = ripple_coefficients(m_r, 'svm')
tot, cl, lines = cluster_split(m_r, 'svm', fsw / fg, nf=1)
N = fsw / fg
k_p, k_m = lines[round(N + 2, 2)], lines[round(N - 2, 2)]
print(f"C1  In = {In:.0f} A ({Ipk:.0f} A pk), Zb = {Zb:.3f} ohm, Lb = {Lb*1e6:.0f} uH, Cb = {Cb*1e6:.0f} uF; m = {m_r:.3f}: k_pp,max = {inv(kmax)}, k_rms = {inv(krms)},"
      f" lines fsw-2f1 {inv(k_m)}, fsw+2f1 {inv(k_p)}, first cluster {inv(cl[0])}; fs/6 = {fs/6:.0f} Hz (zone {0.8*fs/6:.0f}-{1.2*fs/6:.0f} Hz), fsw/2 = {fsw/2:.0f} Hz")
L1, C, Rd = 500e-6, 200e-6, 0.050
for k_r in (0.25, 0.40):
    print(f"    ripple k_r = {k_r}: L1 >= {kmax*Vdc/(fsw*k_r*Ipk)*1e6:.0f} uH")
dI = kmax * Vdc / (fsw * L1)
I_rip = krms * Vdc / (fsw * L1)
I1_p, I1_m, I1_cl = k_p * Vdc / (fsw * L1), k_m * Vdc / (fsw * L1), cl[0] * Vdc / (fsw * L1)
w_p = 2 * PI * (fsw + 2 * fg)
lim05, lim10 = 0.005 * In, 0.010 * In
print(f"C2  L1 = {L1*1e6:.0f} uH ({L1/Lb*100:.1f}%): dI_pp,max = {dI:.0f} A ({dI/Ipk*100:.0f}%), I_rip = {I_rip:.1f} A, I1(fsw+2f1) = {I1_p:.2f} A, I1(fsw-2f1) = {I1_m:.2f} A,"
      f" saturation {Ipk + dI/2:.0f} A; limits 0.5% = {lim05:.2f} A, 1.0% = {lim10:.2f} A")
Ic = wg * C * Vph
print(f"C3  C = {C*1e6:.0f} uF ({C/Cb*100:.1f}% Cb): Qc = {wg*C*Vll**2/1e3:.1f} kvar, Ic = {Ic:.1f} A ({Ic/In*100:.0f}% In), v_c,sw = {I1_cl/(wsw*C):.1f} V;"
      f" Rd = {Rd*1e3:.0f} mohm: P_Rd = {3*Rd*(Ic**2+I_rip**2):.0f} W, attenuation penalty {np.sqrt(1+(wsw*Rd*C)**2):.3f}")
print("C4  grid as L2 (no L2), over the grid inductance:")
for Lg in (25e-6, 50e-6, 75e-6, 100e-6):
    Zc = Zc_rd(w_p, C, Rd)
    ka = abs(Zc / (Zc + 1j * w_p * Lg))
    Lp = L1 * Lg / (L1 + Lg)
    zeta = Rd / 2 * np.sqrt(C / Lp)
    print(f"    Lg = {Lg*1e6:.0f} uH ({Lg/Lb*100:.1f}%): f2 = {1/(2*PI*np.sqrt(Lg*C)):.0f} Hz, f_res = {fres(L1, Lg, C):.0f} Hz ({fres(L1, Lg, C)/fsw:.2f} fsw),"
          f" k_a(fsw+2f1) = {ka:.2f}, I2 = {ka*I1_p:.2f} A ({ka*I1_p/In*100:.2f}% In), zeta = {zeta:.3f}")
print(f"    f_res enters the 0.8-1.2 fs/6 zone for Lg > {1/((2*PI*1.2*fs/6)**2*C)*1e6:.0f} uH; with C = {C*1e6:.0f} uF, f_res >= 1.2 fs/6 at Lg,max = 100 uH would need"
      f" L1||L2' <= {1/((2*PI*1.2*fs/6)**2*C)*1e6:.0f} uH: no room for an L2")
print("C5  aligned design: same L1, C1 = 5% bound or less, L2 such that f_res stays in [1.2 fs/6, fsw/2] over Lg = 25-100 uH, R-C branch n = 0.25")
C1, L2 = 60e-6, 140e-6
for Lg in (25e-6, 50e-6, 100e-6):
    L2p = L2 + Lg
    ka = 1 / (w_p ** 2 * L2p * C1 - 1)
    print(f"    C1 = {C1*1e6:.0f} uF ({C1/Cb*100:.1f}%), L2 = {L2*1e6:.0f} uH ({L2/Lb*100:.1f}%), Lg = {Lg*1e6:.0f}: f_res = {fres(L1, L2p, C1):.0f} Hz ({fres(L1, L2p, C1)/fsw:.2f} fsw),"
          f" k_a = {ka:.3f}, I2(fsw+2f1) = {ka*I1_p:.2f} A ({ka*I1_p/In*100:.2f}%)")
n = 0.25
Lp = L1 * (L2 + 25e-6) / (L1 + L2 + 25e-6)
R0 = np.sqrt(Lp / C1)
C2 = n * C1
R2 = rho_P(n) * R0
I2f = wg * C2 * Vph / np.sqrt(1 + (wg * C2 * R2) ** 2)
Zc = Zc_rc(wsw, C1, C2, R2)
vcs = I1_cl * abs(Zc)
Ib = vcs / abs(R2 + 1 / (1j * wsw * C2))
ka_d = abs(Zc_rc(w_p, C1, C2, R2) / (Zc_rc(w_p, C1, C2, R2) + 1j * w_p * (L2 + 25e-6)))
V1 = abs(Vph + 1j * wg * (L1 + L2) * In)
print(f"    R-C: C2 = {C2*1e6:.0f} uF, Lp = {Lp*1e6:.0f} uH, R0 = {R0:.2f} ohm, R2 = rho_P R0 = {R2:.2f} ohm, zeta = {zeta_P(n):.3f}, P = {3*R2*I2f**2:.0f} + {3*R2*Ib**2:.0f} W,"
      f" k_a with branch (Lg = 25) = {ka_d:.3f} -> I2 = {ka_d*I1_p:.2f} A; Ic(C1) = {wg*C1*Vph:.1f} A ({C1/Cb*100:.1f}%), total C = {(C1+C2)/Cb*100:.1f}% Cb;"
      f" L1 + L2 = {(L1+L2)/Lb*100:.1f}% pu, V1 = {V1:.0f} V (+{(V1/Vph-1)*100:.1f}%), m = {2*SQ2*V1/Vdc:.3f}")

# ============================================================================
# PART 2 - L(RC) sine filter, 450 kW / 480 V drive at 2 kHz, IM and PMSM
# ============================================================================
print()
print("=" * 78)
print("PART 2 - sine filter, 450 kW / 480 V / 60 Hz drive, Vdc = 800 V, fsw = 2 kHz: IM and PMSM")
print("=" * 78)
Pn, Vll, Vdc, fsw = 450e3, 480.0, 800.0, 2000.0
Vph = Vll / SQ3
wsw = 2 * PI * fsw

# cable / reflected wave
v_c, tr_inv, l_cable, Gm = 150.0, 0.3, 150.0, 0.8
l_crit = v_c * tr_inv / 2
Vpk = Vdc * (1 + Gm * min(1, l_cable / l_crit))
print(f"B1  cable v = {v_c:.0f} m/us, tr = {tr_inv:.1f} us -> l_crit = {l_crit:.1f} m; l = {l_cable:.0f} m, Gm = {Gm} -> Vpk = {Vpk:.0f} V ({Vpk/Vdc:.2f} Vdc);"
      f" MG1-31 {3.1*Vll:.0f} V, MG1-30 1000 V")
OF = 0.2
l_crit_req = l_cable * Gm / OF
tr_req = 2 * l_crit_req / v_c
Cdv = 0.22e-6
print(f"    dv/dt filter for OF <= {OF}: l_crit >= {l_crit_req:.0f} m -> tr >= {tr_req:.1f} us (f0 <= {0.3/tr_req*1e3:.0f} kHz, dv/dt <= {Vdc/tr_req:.0f} V/us),"
      f" C = {Cdv*1e6:.2f} uF: loss bound 3 fsw C Vdc^2 = {3*fsw*Cdv*Vdc**2:.0f} W")

cases = {
    "IM":   dict(In=650.0, f1max=60.0, Lhf_pu=0.18, Rhf_pu=0.10, pf=0.87, Lf=150e-6, label="IM 450 kW 4-pole"),
    "PMSM": dict(In=590.0, f1max=60.0, Lhf_pu=0.30, Rhf_pu=0.02, pf=0.95, Lf=160e-6, label="PMSM (IPM) 450 kW 8-pole"),
}
m_full = 2 / SQ3
tot_f, cl_f, lines_f = cluster_split(m_full, 'svm', fsw / 60.0)
krms_full = ripple_coefficients(m_full, 'svm')[2]
print(f"    ripple at full modulation m = 2/sqrt3: k_pp,max = 1/6, k_rms = {inv(krms_full)}, first cluster {inv(cl_f[0])}")
OUT = {}
for key, c in cases.items():
    In, f1max, Lf = c["In"], c["f1max"], c["Lf"]
    Ipk = SQ2 * In
    w1 = 2 * PI * f1max
    Zb = Vph / In
    Lb = Zb / w1
    Cb = 1 / (w1 * Zb)
    Lhf = c["Lhf_pu"] * Lb
    Rhf = c["Rhf_pu"] * Zb
    print(f"\n{c['label']}: In = {In:.0f} A ({Ipk:.0f} A pk), f1max = {f1max:.0f} Hz, Zb = {Zb:.4f} ohm, Lb = {Lb*1e6:.0f} uH, Cb = {Cb*1e3:.2f} mF,"
          f" L_hf = {Lhf*1e6:.0f} uH ({c['Lhf_pu']} pu), R_hf = {Rhf*1e3:.1f} mohm")
    # B2 window
    print(f"B2  window: 8 f1max = {8*f1max:.0f} Hz <= f0 <= fsw/4 = {fsw/4:.0f} Hz (fsw/3 = {fsw/3:.0f} Hz); fsw/f1max = {fsw/f1max:.1f} (needs 24-40)")
    # B3 inductor
    eps = 0.08
    Lf_drop = eps * Vph / (w1 * In)
    k_r = 0.5
    Lf_rip = Vdc / (6 * fsw * k_r * Ipk)
    dI = Vdc / (6 * fsw * Lf)
    I_rip = krms_full * Vdc / (fsw * Lf)
    I1_cl = cl_f[0] * Vdc / (fsw * Lf)
    eps_act = w1 * Lf * In / Vph
    # inverter voltage with the drop (vectorial), PF of the load, capacitor current added later
    print(f"B3  drop <= {eps*100:.0f}%: Lf <= {Lf_drop*1e6:.0f} uH -> ripple would be {Vdc/(6*fsw*Lf_drop)/Ipk*100:.0f}%; ripple k_r = {k_r}: Lf >= Vdc/(6 fsw k_r Ipk) = {Lf_rip*1e6:.0f} uH")
    print(f"    chosen Lf = {Lf*1e6:.0f} uH ({Lf/Lb*100:.1f}% pu): drop {eps_act*100:.1f}% at f1max; dI_pp,max = {dI:.0f} A ({dI/Ipk*100:.0f}% of Ipk),"
          f" I_rip,rms = {I_rip:.1f} A ({I_rip/In*100:.1f}% In), first cluster {I1_cl:.1f} A; saturation >= {Ipk + dI/2:.0f} A")
    # B4 capacitor from the loaded resonance target fp = fsw/3
    fp_t = fsw / 3
    f0_t = fp_t / np.sqrt(1 + Lf / Lhf)
    Cf_t = 1 / ((2 * PI * f0_t) ** 2 * Lf)
    Cf = round(Cf_t * 1e6 / 20) * 20e-6
    f0 = 1 / (2 * PI * np.sqrt(Lf * Cf))
    fp = f0 * np.sqrt(1 + Lf / Lhf)
    Ic = w1 * Cf * Vph
    print(f"B4  target fp = fsw/3 = {fp_t:.0f} Hz -> f0 = fp/sqrt(1 + Lf/Lhf) = {f0_t:.0f} Hz ({f0_t/f1max:.1f} f1max) -> Cf = 1/((2 pi f0)^2 Lf) = {Cf_t*1e6:.0f} uF")
    print(f"    chosen Cf = {Cf*1e6:.0f} uF ({Cf/Cb*100:.1f}% Cb): f0 = {f0:.0f} Hz ({f0/f1max:.1f} f1max), fp = {fp:.0f} Hz (fsw/{fsw/fp:.2f}), Ic(f1max) = {Ic:.1f} A ({Ic/In*100:.1f}% In)")
    # inverter current and voltage at rated load (vector): is at PF, ic in quadrature
    pf = c["pf"]
    Is = In * (pf - 1j * np.sqrt(1 - pf ** 2))          # lagging
    Vc = Vph + 0j
    Ii = Is + 1j * w1 * Cf * Vc
    Vi = Vc + 1j * w1 * Lf * Ii
    m_need = 2 * SQ2 * abs(Vi) / Vdc
    print(f"    rated point: I_i = {abs(Ii):.0f} A (PF {abs(Ii.real)/abs(Ii):.3f}), V_i = {abs(Vi):.0f} V ph ({abs(Vi)*SQ3:.0f} V ll), m = {m_need:.3f} (limit 1.155)")
    # B5 loaded resonance, attenuation, machine damping
    att = 1 / ((fsw / fp) ** 2 - 1)
    Lp = Lf * Lhf / (Lf + Lhf)
    G = Rhf / (Rhf ** 2 + (2 * PI * fp * Lhf) ** 2)
    zeta_m = 0.5 * G * np.sqrt(Lp / Cf)
    vc_sw = I1_cl / (wsw * Cf)
    print(f"B5  |vc/vi|(fsw) = 1/((fsw/fp)^2 - 1) = {att:.3f} ({20*np.log10(att):.1f} dB); Lp = Lf || Lhf = {Lp*1e6:.0f} uH; G_m = {G*1e3:.1f} mS, zeta_m = {zeta_m:.4f};"
          f" v_c,sw = {vc_sw:.1f} V rms ({vc_sw/Vph*100:.1f}% Vph), peak {SQ2*vc_sw:.0f} V")
    # dead-time harmonics
    print(f"    5th/7th at f1max: {5*f1max:.0f}/{7*f1max:.0f} Hz, amplification 1/|1-(f/fp)^2| = {1/abs(1-(5*f1max/fp)**2):.2f} / {1/abs(1-(7*f1max/fp)**2):.2f}")
    # B6 damping
    res = {}
    for n in (1.0, 0.5, 0.25):
        C1 = Cf
        C2 = n * C1
        R0 = np.sqrt(Lp / C1)
        R2 = rho_P(n) * R0
        I2f = w1 * C2 * Vph / np.sqrt(1 + (w1 * C2 * R2) ** 2)
        P_f = 3 * R2 * I2f ** 2
        Zc = Zc_rc(wsw, C1, C2, R2)
        vcs = I1_cl * abs(Zc)
        Ib = vcs / abs(R2 + 1 / (1j * wsw * C2))
        P_sw = 3 * R2 * Ib ** 2
        Ic_tot = w1 * (C1 + C2) * Vph
        res[n] = (C2, R2, zeta_P(n), P_f + P_sw, Ic_tot)
        print(f"B6  R-C n = {n}: C2 = {C2*1e6:.0f} uF added, R0 = {R0:.3f} ohm, rho_P = {rho_P(n):.2f} -> R2 = {R2:.2f} ohm, zeta = {zeta_P(n):.3f};"
              f" I2(f1) = {I2f:.1f} A: P_f = {P_f:.0f} W, P_sw = {P_sw:.0f} W, total {P_f+P_sw:.0f} W ({(P_f+P_sw)/Pn*100:.2f}% Pn); Ic total {Ic_tot:.0f} A ({Ic_tot/In*100:.0f}% In)")
    Rd = 1 / (3 * 2 * PI * fp * Cf)
    zeta_Rd = Rd / 2 * np.sqrt(Cf / Lp)
    P_Rd = 3 * Rd * (Ic ** 2 + I_rip ** 2)
    print(f"    series Rd = 1/(3 wp Cf) = {Rd*1e3:.0f} mohm (zeta {zeta_Rd:.2f}): P = {P_Rd:.0f} W ({P_Rd/Pn*100:.2f}% Pn); attenuation penalty sqrt(1+(wsw Rd Cf)^2) = {np.sqrt(1+(wsw*Rd*Cf)**2):.2f}")
    # B7 ratings and control
    print(f"B7  Lf: {In:.0f} A rms + ripple {I_rip:.0f} A rms, saturation >= {Ipk + dI/2:.0f} A; Cf: >= {1.1*Vll:.0f} V ac + {SQ2*vc_sw:.0f} V pk ripple, I = {np.sqrt(Ic**2+I_rip**2):.0f} A rms")
    print(f"    control: alpha_i <= 2 pi fsw/10 = 2 pi {fsw/10:.0f} Hz; alpha_v ~ alpha_i/4; stator-current loop <= fp/5 = {fp/5:.0f} Hz; f_inj <= fp/3 = {fp/3:.0f} Hz")
    C2s, R2s, zs, Ps, _ = res[0.25]
    OUT[key] = dict(Ln=Lf, Lpu=Lf/Lb, C=Cf, Cpu=Cf/Cb, C2=C2s, R2=R2s, zeta=zs, Ploss=Ps/Pn, f0=f0, fp=fp, ratio=fp/fsw, ka=att, dI=dI/Ipk, Irip=I_rip/In, Ic=Ic/In, eps=eps_act,
                    In=In, Ipk=Ipk + dI/2, zeta_m=zeta_m, res=res, Rd=Rd, P_Rd=P_Rd/Pn, m=m_need, Lhf=Lhf, bw=fp/5)

# ============================================================================
# PART 3 - comparison
# ============================================================================
print()
print("=" * 78)
print("PART 3 - comparison AFE / IM / PMSM (450 kW class, 480 V, 2 kHz)")
print("=" * 78)
print(f"{'':28s}{'AFE':>16s}{'IM':>16s}{'PMSM':>16s}")
rows = [
    ("converter-side L [uH] (% pu)", f"{AFE['Ln']*1e6:.0f} ({AFE['Lpu']*100:.1f})", f"{OUT['IM']['Ln']*1e6:.0f} ({OUT['IM']['Lpu']*100:.1f})", f"{OUT['PMSM']['Ln']*1e6:.0f} ({OUT['PMSM']['Lpu']*100:.1f})"),
    ("second inductance [uH]", f"L2 = {AFE['L2']*1e6:.0f} ({AFE['L2pu']*100:.1f})", f"L_hf = {OUT['IM']['Lhf']*1e6:.0f} (18)", f"L_hf = {OUT['PMSM']['Lhf']*1e6:.0f} (30)"),
    ("capacitance C1 [uF] (% Cb)", f"{AFE['C']*1e6:.0f} ({AFE['Cpu']*100:.1f})", f"{OUT['IM']['C']*1e6:.0f} ({OUT['IM']['Cpu']*100:.1f})", f"{OUT['PMSM']['C']*1e6:.0f} ({OUT['PMSM']['Cpu']*100:.1f})"),
    ("resonance [Hz] (/fsw)", f"{AFE['fr']:.0f} ({AFE['ratio']:.2f})", f"{OUT['IM']['fp']:.0f} ({OUT['IM']['ratio']:.2f})", f"{OUT['PMSM']['fp']:.0f} ({OUT['PMSM']['ratio']:.2f})"),
    ("attenuation at fsw [dB]", f"{20*np.log10(AFE['ka']):.1f}", f"{20*np.log10(OUT['IM']['ka']):.1f}", f"{20*np.log10(OUT['PMSM']['ka']):.1f}"),
    ("ripple pp max (% Ipk)", f"{AFE['dI']*100:.0f}", f"{OUT['IM']['dI']*100:.0f}", f"{OUT['PMSM']['dI']*100:.0f}"),
    ("ripple rms (% In)", f"{AFE['Irip']*100:.1f}", f"{OUT['IM']['Irip']*100:.1f}", f"{OUT['PMSM']['Irip']*100:.1f}"),
    ("fundamental drop (%)", f"{AFE['eps']*100:.1f}", f"{OUT['IM']['eps']*100:.1f}", f"{OUT['PMSM']['eps']*100:.1f}"),
    ("capacitor current (% In)", f"{AFE['Ic']*100:.1f}", f"{OUT['IM']['Ic']*100:.1f}", f"{OUT['PMSM']['Ic']*100:.1f}"),
    ("modulation index", f"{AFE['m']:.2f}", f"{OUT['IM']['m']:.2f}", f"{OUT['PMSM']['m']:.2f}"),
    ("R-C branch n = 0.25: C2 [uF]", f"{AFE['C2']*1e6:.0f}", f"{OUT['IM']['C2']*1e6:.0f}", f"{OUT['PMSM']['C2']*1e6:.0f}"),
    ("  R2 [ohm], zeta", f"{AFE['R2']:.2f}, {AFE['zeta']:.2f}", f"{OUT['IM']['R2']:.2f}, {OUT['IM']['zeta']:.2f}", f"{OUT['PMSM']['R2']:.2f}, {OUT['PMSM']['zeta']:.2f}"),
    ("  loss (% rating)", f"{AFE['Ploss']*100:.2f}", f"{OUT['IM']['Ploss']*100:.2f}", f"{OUT['PMSM']['Ploss']*100:.2f}"),
    ("machine damping zeta_m", "-", f"{OUT['IM']['zeta_m']:.3f}", f"{OUT['PMSM']['zeta_m']:.4f}"),
    ("L peak current [A]", f"{AFE['Ipk']:.0f}", f"{OUT['IM']['Ipk']:.0f}", f"{OUT['PMSM']['Ipk']:.0f}"),
    ("current-loop bandwidth [Hz]", f"{AFE['bw']:.0f}", f"{OUT['IM']['bw']:.0f}", f"{OUT['PMSM']['bw']:.0f}"),
]
for r in rows:
    print(f"{r[0]:28s}{r[1]:>16s}{r[2]:>16s}{r[3]:>16s}")
