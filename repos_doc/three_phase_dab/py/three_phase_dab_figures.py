#!/usr/bin/env python3
"""Data files of the figures of three_phase_dab.tex (pgfplots reads them with \\addplot table).

Run from the document folder:  python3 py/three_phase_dab_figures.py
Writes into figures/:
  power_phi.dat          power against the phase shift: De Doncker's formula at d = 1 and d = 0.85, the
                         harmonic model with Lm, Rfe and the resistances at d = 1, phase current rms
  waveforms_nominal.dat  steady state at 1200 V / 1200 V and phi = 30 deg over one period: phase-to-star-point
                         voltages of the two bridges, phase currents, magnetizing current, DC currents of the
                         two bridges
  zvs_map.dat            lowest load with ZVS of bridge 1 and of bridge 2 against d = V2 n1/(n2 V1), forward
                         power, V1 = 1200 V, dead time 500 ns
  flux_start.dat         flux space vector of the transformers over the first two periods from zero flux: plain
                         six-step start and the start that holds the first state for a whole step and skips the
                         next one
The model is the one of dab3_harmonic_op.m of pwr-control/advanced-dcdc-converters (analysis_three_phase_dab):
per phase, T equivalent referred to the primary (R1 + L1, Lm // Rfe, L2' + R2'), six-step voltages of both bridges
(floating star points, harmonics h = 6k +- 1), bridge 2 lagging bridge 1 by phi. The numbers are those of
private/dab3_param.m. The script also prints the derived quantities quoted in the note.
Only the standard library is used.
"""
import cmath
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(FIG, exist_ok=True)

P_NOM = 250e3                       # [W]
FS = 10e3                           # [Hz] transformer frequency
V_NOM = 1200.0                      # [V] both sides, n = 14:14
L_TOT = 28.0e-6                     # [H] series inductance per phase, Ld1 + Ld2'
RDS = 1.04e-3                       # [Ohm] SiC module, danfoss_SKM1700MB20R4S2I4
COSS = 3.28e-9                      # [F]
TD = 500e-9                         # [s] dead time of the sizing
PHI_MAX = math.pi / 3               # [rad] limit of the modulator
P = dict(V1=V_NOM, V2=V_NOM, n1=14, n2=14, fs=FS,
         L1=L_TOT / 2, R1=1.3e-3 + RDS, L2r=L_TOT / 2, R2r=1.3e-3 + RDS, Lm=1e-3, Rfe=3e3)


def harmonics(hmax):
    """Odd harmonics without the triplen ones: 1, 5, 7, 11, 13, ..."""
    return [h for h in range(1, hmax + 1, 2) if h % 3]


def phasors(p, phi, V1, V2, hmax):
    """Phasors of each harmonic (v = Im{V e^{j h w t}}): h, U1, U2, I1 (out of leg a of bridge 1), J2 (into leg a of
    bridge 2, referred to the primary)."""
    n12 = p["n1"] / p["n2"]
    w = 2 * math.pi * p["fs"]
    out = []
    for h in harmonics(hmax):
        U1 = 2 * V1 / (h * math.pi)
        U2 = 2 * V2 * n12 / (h * math.pi) * cmath.exp(-1j * h * phi)
        Z1 = p["R1"] + 1j * h * w * p["L1"]
        Z2 = p["R2r"] + 1j * h * w * p["L2r"]
        Ym = 1 / (1j * h * w * p["Lm"]) + 1 / p["Rfe"]
        E = (U1 / Z1 + U2 / Z2) / (1 / Z1 + Ym + 1 / Z2)
        out.append((h, U1, U2, (U1 - E) / Z1, (E - U2) / Z2))
    return out


def op_point(p, phi, V1=None, V2=None, hmax=2001):
    """DC powers, rms currents and switching currents (> 0: the current charges the leg capacitances the right way)."""
    V1 = p["V1"] if V1 is None else V1
    V2 = p["V2"] if V2 is None else V2
    n12 = p["n1"] / p["n2"]
    ph = phasors(p, phi, V1, V2, hmax)
    P1 = 1.5 * sum((U1 * I1.conjugate()).real for h, U1, U2, I1, J2 in ph)
    P2 = 1.5 * sum((U2 * J2.conjugate()).real for h, U1, U2, I1, J2 in ph)
    I1rms = math.sqrt(sum(abs(I1) ** 2 for h, U1, U2, I1, J2 in ph) / 2)
    I2rms = math.sqrt(sum(abs(J2) ** 2 for h, U1, U2, I1, J2 in ph) / 2) * n12
    i_sw1 = -sum(I1.imag for h, U1, U2, I1, J2 in ph)                                   # leg a rises at 0
    i_sw2 = sum((J2 * cmath.exp(1j * h * phi)).imag for h, U1, U2, I1, J2 in ph) * n12  # leg a rises at phi
    return dict(P1=P1, P2=P2, I1rms=I1rms, I2rms=I2rms, i_sw1=i_sw1, i_sw2=i_sw2, ph=ph)


def waveform(ph, k, theta):
    """Im{sum X_h e^{j h theta}} of the phasor k of the list (3: I1, 4: J2)."""
    return sum((x[k] * cmath.exp(1j * x[0] * theta)).imag for x in ph)


def power_ideal(V1, V2r, phi):
    """De Doncker: lossless, Lm infinite, L = L_TOT per phase."""
    k = V1 * V2r / (2 * math.pi * FS * L_TOT)
    a = abs(phi)
    if a <= math.pi / 3:
        pa = a * (2 / 3 - a / (2 * math.pi))
    else:
        pa = a - a ** 2 / math.pi - math.pi / 18
    return math.copysign(k * pa, phi)


def phi_ideal(V1, V2r, P):
    """Inverse of power_ideal for |phi| <= pi/3."""
    k = V1 * V2r / (2 * math.pi * FS * L_TOT)
    x = abs(P) / k
    return math.copysign(2 * math.pi / 3 - math.sqrt((2 * math.pi / 3) ** 2 - 2 * math.pi * x), P)


def six_step(theta):
    """Upper switch of legs a, b, c (leg a rises at theta = 0, b and c 120 and 240 deg later)."""
    return [1.0 if (theta - k * 2 * math.pi / 3) % (2 * math.pi) < math.pi else 0.0 for k in range(3)]


def phase_voltage(theta, V):
    s = six_step(theta)
    return V / 3 * (2 * s[0] - s[1] - s[2])


def power_phi(path):
    with open(path, "w") as f:
        f.write("phi P_d100 P_d085 P_h100 I1rms_h100\n")
        for k in range(0, 121):
            phi = math.radians(k)
            o = op_point(P, phi, hmax=601)
            f.write(f"{k} {power_ideal(V_NOM, V_NOM, phi) / 1e3:.2f} {power_ideal(V_NOM, 0.85 * V_NOM, phi) / 1e3:.2f} "
                    f"{o['P2'] / 1e3:.2f} {o['I1rms']:.2f}\n")


def waveforms_nominal(path, phi=math.pi / 6, npts=720):
    o = op_point(P, phi)
    ph = o["ph"]
    th = [2 * math.pi * k / npts for k in range(npts)]
    ia = [waveform(ph, 3, t) for t in th]
    ja = [waveform(ph, 4, t) for t in th]
    sh = npts // 3
    with open(path, "w") as f:
        f.write("theta v1a v2a i1a i2a im idc1 idc2\n")
        for k in range(npts + 1):
            kk = k % npts
            t = 2 * math.pi * k / npts
            s1 = six_step(t)
            s2 = six_step(t - phi)
            i1 = [ia[kk], ia[(kk - sh) % npts], ia[(kk - 2 * sh) % npts]]
            j2 = [ja[kk], ja[(kk - sh) % npts], ja[(kk - 2 * sh) % npts]]
            idc1 = sum(s * i for s, i in zip(s1, i1))
            idc2 = sum(s * j for s, j in zip(s2, j2))
            f.write(f"{360 * k / npts:.2f} {phase_voltage(t, V_NOM):.1f} {phase_voltage(t - phi, V_NOM):.1f} "
                    f"{ia[kk]:.2f} {ja[kk]:.2f} {ia[kk] - ja[kk]:.3f} {idc1:.2f} {idc2:.2f}\n")
    return o


def zvs_map(path):
    """Lowest load (%) above which the switching current of each bridge charges the leg within the dead time."""
    rows = []
    for jd in range(41):
        d = 0.80 + 0.01 * jd
        V2 = d * V_NOM
        req1, req2 = 2 * COSS * V_NOM / TD, 2 * COSS * V2 / TD
        m1, m2 = [], []
        loads = [0.005 * k for k in range(1, 201)]
        for x in loads:
            o = op_point(P, phi_ideal(V_NOM, V2, x * P_NOM), V_NOM, V2, hmax=1001)
            m1.append(o["i_sw1"] - req1)                # margin of the switching current
            m2.append(o["i_sw2"] - req2)

        def boundary(m):
            """Load above which the margin stays positive, interpolated between the grid points."""
            for k in range(len(m) - 1, -1, -1):
                if m[k] < 0:
                    if k == len(m) - 1:
                        return 1.0
                    return loads[k] + (loads[k + 1] - loads[k]) * (-m[k]) / (m[k + 1] - m[k])
            return 0.0

        rows.append((d, 100 * boundary(m1), 100 * boundary(m2)))
    with open(path, "w") as f:
        f.write("d x1 x2 xboth\n")
        for d, x1, x2 in rows:
            f.write(f"{d:.2f} {x1:.1f} {x2:.1f} {max(x1, x2):.1f}\n")
    return rows


def flux_start(path, nstep=60):
    """Flux space vector (alpha, beta) of the phase-to-star volt-seconds from zero flux, two periods, mVs."""
    seq = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1)]
    dt = 1 / (6 * FS * nstep)

    def run(states):
        lam = [0.0, 0.0, 0.0]
        tr = [(0.0, 0.0)]
        for s in states:
            m = sum(s) / 3
            v = [V_NOM * (x - m) for x in s]
            for _ in range(nstep):
                lam = [l + vi * dt for l, vi in zip(lam, v)]
                al = (2 / 3) * (lam[0] - 0.5 * lam[1] - 0.5 * lam[2])
                be = (2 / 3) * (math.sqrt(3) / 2) * (lam[1] - lam[2])
                tr.append((1e3 * al, 1e3 * be))
        return tr

    plain = run([seq[k % 6] for k in range(12)])
    skip = run([seq[0]] + [seq[k % 6] for k in range(2, 13)])
    with open(path, "w") as f:
        f.write("a_plain b_plain a_skip b_skip\n")
        for (a1, b1), (a2, b2) in zip(plain, skip):
            f.write(f"{a1:.3f} {b1:.3f} {a2:.3f} {b2:.3f}\n")
    n = 6 * nstep
    mean = lambda tr, i: sum(x[i] for x in tr[-n:]) / n
    return (mean(plain, 0), mean(plain, 1)), (mean(skip, 0), mean(skip, 1))


def compare_single_phase(phi=math.pi / 6, npts=3600, cdc=500e-6):
    """DAB3 and single-phase DAB at the same point (250 kW, 1200 V / 1200 V, 10 kHz, phi = 30 deg), both lossless with
    Lm infinite: series inductance, currents, DC-link ripple."""
    p = dict(P, R1=0.0, R2r=0.0, Lm=1e6, Rfe=1e12)
    o = op_point(p, phi, hmax=4001)
    th = [2 * math.pi * k / npts for k in range(npts)]
    ia = [waveform(o["ph"], 3, t) for t in th]
    sh = npts // 3
    idc3 = [sum(s * i for s, i in zip(six_step(th[k]), (ia[k], ia[(k - sh) % npts], ia[(k - 2 * sh) % npts])))
            for k in range(npts)]
    # single-phase DAB, square waves: P = V1 V2 phi (1 - phi/pi) / (w L1), trapezoidal current at d = 1
    w = 2 * math.pi * FS
    L1 = V_NOM ** 2 * phi * (1 - phi / math.pi) / (w * P_NOM)
    ip = V_NOM * phi / (w * L1)

    def i1(t):
        t = t % (2 * math.pi)
        x = t if t < math.pi else t - math.pi
        v = -ip + 2 * ip * x / phi if x < phi else ip
        return v if t < math.pi else -v

    i_sp = [i1(t) for t in th]
    idc1 = [i if t < math.pi else -i for t, i in zip(th, i_sp)]          # DC current of bridge 1

    def ripple(idc):
        m = sum(idc) / npts
        q, qs = 0.0, []
        for x in idc:
            q += (x - m) / (FS * npts)
            qs.append(q)
        return m, math.sqrt(sum((x - m) ** 2 for x in idc) / npts), (max(qs) - min(qs)) / cdc

    r3, r1 = ripple(idc3), ripple(idc1)
    rms_sp = math.sqrt(sum(i * i for i in i_sp) / npts)
    out = dict(
        L3=L_TOT, L1=L1,
        I3=o["I1rms"], I1=rms_sp, I3pk=max(abs(x) for x in ia), I1pk=ip,
        isw3=(o["i_sw1"], o["i_sw2"]), isw1=(ip, ip),
        cond3=6 * RDS * o["I1rms"] ** 2, cond1=4 * RDS * rms_sp ** 2,
        toff3=12 * o["i_sw1"], toff1=8 * ip,
        dc3=r3, dc1=r1,
        lam3=V_NOM / (9 * FS), lam1=V_NOM / (4 * FS),
        va3=3 * V_NOM * math.sqrt(2) / 3 * o["I1rms"], va1=V_NOM * rms_sp)
    return out


if __name__ == "__main__":
    w = 2 * math.pi * FS
    print(f"L = {1e6 * L_TOT:.1f} uH per phase, Pmax = {7 * math.pi / 36 * V_NOM ** 2 / (w * L_TOT) / 1e3:.0f} kW at 90 deg, "
          f"{power_ideal(V_NOM, V_NOM, PHI_MAX) / 1e3:.0f} kW at phi_max = 60 deg "
          f"({power_ideal(V_NOM, 0.85 * V_NOM, PHI_MAX) / 1e3:.0f} kW with d = 0.85)")
    print(f"flux linkage of the six-step: {1e3 * V_NOM / (9 * FS):.2f} mVs peak; "
          f"ZVS current in {1e9 * TD:.0f} ns: {2 * COSS * V_NOM / TD:.1f} A at {V_NOM:.0f} V")
    for deg in (20, 25, 30, 35, 40):
        Lx = V_NOM ** 2 / (w * P_NOM) * math.radians(deg) * (2 / 3 - math.radians(deg) / (2 * math.pi))
        print(f"  phi_nom {deg} deg: L = {1e6 * Lx:.1f} uH")
    power_phi(os.path.join(FIG, "power_phi.dat"))
    o = waveforms_nominal(os.path.join(FIG, "waveforms_nominal.dat"))
    print(f"nominal point, phi = 30 deg: P1 = {o['P1'] / 1e3:.1f} kW, P2 = {o['P2'] / 1e3:.1f} kW, "
          f"I1 = {o['I1rms']:.1f} A rms, i_sw1 = {o['i_sw1']:.1f} A, i_sw2 = {o['i_sw2']:.1f} A, "
          f"transitions {1e9 * 2 * COSS * V_NOM / o['i_sw1']:.0f} / {1e9 * 2 * COSS * V_NOM / o['i_sw2']:.0f} ns")
    rows = zvs_map(os.path.join(FIG, "zvs_map.dat"))
    for d, x1, x2 in rows:
        if abs(round(100 * d) % 5) == 0:
            print(f"  d = {d:.2f}: ZVS of bridge 1 from {x1:.1f} %, bridge 2 from {x2:.1f} %, both from {max(x1, x2):.1f} %")
    c = compare_single_phase()
    print("DAB3 against the single-phase DAB, 250 kW, 1200 V / 1200 V, 10 kHz, phi = 30 deg, lossless:")
    print(f"  series inductance {1e6 * c['L3']:.1f} uH per phase / {1e6 * c['L1']:.1f} uH")
    print(f"  current {c['I3']:.1f} A rms, {c['I3pk']:.1f} A peak / {c['I1']:.1f} A rms, {c['I1pk']:.1f} A peak")
    print(f"  switching current {c['isw3'][0]:.1f}, {c['isw3'][1]:.1f} A / {c['isw1'][0]:.1f} A")
    print(f"  conduction with {1e3 * RDS:.2f} mOhm per switch: {c['cond3']:.0f} W (12 switches) / {c['cond1']:.0f} W (8)")
    print(f"  sum of the turned-off currents per period: {c['toff3']:.0f} A / {c['toff1']:.0f} A")
    print(f"  DC link 1: {c['dc3'][0]:.1f} A mean, {c['dc3'][1]:.1f} A rms ripple at 60 kHz, {c['dc3'][2]:.2f} V p-p with 500 uF / "
          f"{c['dc1'][0]:.1f} A, {c['dc1'][1]:.1f} A rms at 20 kHz, {c['dc1'][2]:.2f} V p-p")
    print(f"  transformer flux linkage {1e3 * c['lam3']:.2f} mVs per phase / {1e3 * c['lam1']:.2f} mVs; "
          f"winding VA per side {c['va3'] / 1e3:.0f} / {c['va1'] / 1e3:.0f} kVA")
    m_plain, m_skip = flux_start(os.path.join(FIG, "flux_start.dat"))
    print(f"flux start: mean flux vector over the second period, plain start ({m_plain[0]:.2f}, {m_plain[1]:.2f}) mVs, "
          f"start with the skipped state ({m_skip[0]:.2f}, {m_skip[1]:.2f}) mVs")
