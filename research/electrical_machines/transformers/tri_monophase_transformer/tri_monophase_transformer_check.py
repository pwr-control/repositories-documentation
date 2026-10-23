#!/usr/bin/env python3
# tri_monophase_transformer_check.py
#
# Verification of the Simscape component tri_monophase_transformer.ssc
# (two star-connected three-phase primaries on a three-limb core, one
# single-phase secondary). Pure Python 3 / NumPy (+ Matplotlib for the
# figures): the script rebuilds the SAME equations of the .ssc file as a
# linear descriptor system E x' = A x + B u, solves it in the frequency
# domain (sinusoidal steady state) and in the time domain (trapezoidal
# rule), prints the quantities quoted in the technical note and draws its
# figures into ./figures/.
#
# Usage:  python3 tri_monophase_transformer_check.py [--no-fig]
#
# MIT License, Copyright (c) 2026 Davide Bagnara

import sys
import numpy as np

# --------------------------------------------------------------------------
# Default parameters (identical to the defaults of the .ssc component)
# --------------------------------------------------------------------------
PAR = dict(
    Sr=100e3,   # rated apparent power, single-phase output   [VA]
    V1=400.0,   # primary 1 rated line-to-line voltage, rms  [V]
    V2=400.0,   # primary 2 rated line-to-line voltage, rms  [V]
    V3=230.0,   # secondary rated voltage, rms               [V]
    fr=50.0,    # rated frequency                            [Hz]
    r1=0.005, l1=0.030,   # primary 1 coil resistance / leakage   [pu]
    r2=0.005, l2=0.030,   # primary 2 coil resistance / leakage   [pu]
    r3=0.005, l3=0.030,   # secondary resistance / leakage        [pu]
    lm=100.0,   # positive-sequence magnetizing inductance   [pu]
    lm0=0.5,    # zero-sequence magnetizing inductance       [pu]
    rfe=500.0,  # core-loss resistance per limb              [pu]
    ws=np.array([1.0, -1.0, 0.0]),  # secondary turns distribution on limbs a,b,c
)

ALPHA = np.exp(1j * 2 * np.pi / 3)   # a = 1 /_ 120 deg


def derived(p):
    """Physical quantities from the per-unit parameters (same formulas as the
    parameters(Access=private) block of the .ssc file)."""
    w = np.asarray(p['ws'], float)
    wb = 2 * np.pi * p['fr']
    Zb1 = p['V1'] ** 2 / p['Sr']
    Zb2 = p['V2'] ** 2 / p['Sr']
    Zb3 = p['V3'] ** 2 / p['Sr']
    # |w_a + a^2 w_b + a w_c|  (balanced positive-sequence flux, a-b-c sequence)
    gw = np.sqrt((w[0] - 0.5 * (w[1] + w[2])) ** 2 + 0.75 * (w[2] - w[1]) ** 2)
    d = dict(
        w=w, wb=wb, Zb1=Zb1, Zb2=Zb2, Zb3=Zb3, gw=gw,
        n2=p['V2'] / p['V1'],
        n3=np.sqrt(3) * p['V3'] / (gw * p['V1']),
        R1=p['r1'] * Zb1, L1=p['l1'] * Zb1 / wb,
        R2=p['r2'] * Zb2, L2=p['l2'] * Zb2 / wb,
        R3=p['r3'] * Zb3, L3=p['l3'] * Zb3 / wb,
        Lm=p['lm'] * Zb1 / wb, Lm0=p['lm0'] * Zb1 / wb,
        Rfe=p['rfe'] * Zb1,
    )
    return d


# --------------------------------------------------------------------------
# Descriptor system  E x' = A x + B u
# x = [i1a i1b i1c  i2a i2b i2c  i3  la lb lc  vn1 vn2]
# u = [vA1 vB1 vC1  vA2 vB2 vC2]   (phase-to-ground source voltages)
# --------------------------------------------------------------------------
IX = dict(i1=[0, 1, 2], i2=[3, 4, 5], i3=6, lam=[7, 8, 9], vn1=10, vn2=11)


def build(p, prim2='open', neutral1='float', neutral2='float',
          load='open', RL=0.0, LL=0.0):
    """prim2: 'open' | 'fed' (primary 2 driven by u[3:6]);
    neutralX: 'float' | 'ground';  load: 'open' | 'short' | 'rl'."""
    d = derived(p)
    w, n2, n3 = d['w'], d['n2'], d['n3']
    E = np.zeros((12, 12))
    A = np.zeros((12, 12))
    B = np.zeros((12, 6))
    c = (d['Lm'] - d['Lm0']) / 3.0
    # primary 1 coils: L1 i1k' + lk' = -R1 i1k - vn1 + vk
    for k in range(3):
        r = k
        E[r, IX['i1'][k]] = d['L1']
        E[r, IX['lam'][k]] = 1.0
        A[r, IX['i1'][k]] = -d['R1']
        A[r, IX['vn1']] = -1.0
        B[r, k] = 1.0
    # primary 2 coils
    for k in range(3):
        r = 3 + k
        if prim2 == 'fed':
            E[r, IX['i2'][k]] = d['L2']
            E[r, IX['lam'][k]] = n2
            A[r, IX['i2'][k]] = -d['R2']
            A[r, IX['vn2']] = -1.0
            B[r, 3 + k] = 1.0
        else:                       # open: i2k = 0
            A[r, IX['i2'][k]] = 1.0
    # secondary: (L3+LL) i3' - n3 sum_k w_k lk' = -(R3+RL) i3
    r = 6
    if load == 'open':
        A[r, IX['i3']] = 1.0
    else:
        if load == 'short':
            RL, LL = 0.0, 0.0
        E[r, IX['i3']] = d['L3'] + LL
        for k in range(3):
            E[r, IX['lam'][k]] = -n3 * w[k]
        A[r, IX['i3']] = -(d['R3'] + RL)
    # magnetizing: lk = Lm im_k - c sum_j im_j,
    # im_k = i1k + n2 i2k - n3 w_k i3 - lk'/Rfe
    for k in range(3):
        r = 7 + k
        E[r, IX['lam'][k]] += d['Lm'] / d['Rfe']
        A[r, IX['lam'][k]] = -1.0
        A[r, IX['i1'][k]] += d['Lm']
        A[r, IX['i2'][k]] += d['Lm'] * n2
        A[r, IX['i3']] += -d['Lm'] * n3 * w[k]
        for j in range(3):
            E[r, IX['lam'][j]] += -c / d['Rfe']
            A[r, IX['i1'][j]] += -c
            A[r, IX['i2'][j]] += -c * n2
            A[r, IX['i3']] += c * n3 * w[j]
    # neutrals
    r = 10
    if neutral1 == 'float':
        for k in range(3):
            A[r, IX['i1'][k]] = 1.0
    else:
        A[r, IX['vn1']] = 1.0
    r = 11
    if prim2 == 'fed' and neutral2 == 'float':
        for k in range(3):
            A[r, IX['i2'][k]] = 1.0
    else:
        A[r, IX['vn2']] = 1.0
    return E, A, B, d


def phasor(E, A, B, U, w):
    """Sinusoidal steady state: (jwE - A) X = B U, U = complex peak values."""
    return np.linalg.solve(1j * w * E - A, B @ U)


def source(Vll, phase_deg=0.0, seq=+1):
    """Balanced three-phase set of phase-to-ground voltages, complex peaks."""
    Vp = np.sqrt(2) * Vll / np.sqrt(3) * np.exp(1j * np.deg2rad(phase_deg))
    return np.array([Vp, Vp * ALPHA ** (-seq), Vp * ALPHA ** (-2 * seq)])


def seqcomp(I):
    """Symmetrical components (zero, positive, negative) of a 3-vector."""
    a = ALPHA
    I0 = (I[0] + I[1] + I[2]) / 3
    Ip = (I[0] + a * I[1] + a ** 2 * I[2]) / 3
    In = (I[0] + a ** 2 * I[1] + a * I[2]) / 3
    return I0, Ip, In


def rms(x):
    return np.abs(x) / np.sqrt(2)


def ang(x):
    return np.rad2deg(np.angle(x))


def report(title, X, U, d, p, prim2):
    i1 = X[IX['i1']]
    i2 = X[IX['i2']]
    i3 = X[IX['i3']]
    lam = X[IX['lam']]
    w = d['w']
    e3 = d['n3'] * (1j * d['wb']) * (w @ lam)
    v3 = e3 - (d['R3'] + 1j * d['wb'] * d['L3']) * i3
    vAB = U[0] - U[1]
    print('\n' + '=' * 72)
    print(title)
    print('-' * 72)
    print('  secondary: V3 = %8.2f V rms  /_ %7.2f deg (ref. V_AB),  I3 = %8.2f A rms'
          % (rms(v3), ang(v3 / vAB), rms(i3)))
    print('  primary 1 line currents [A rms]: A %8.3f  B %8.3f  C %8.3f'
          % tuple(rms(i1)))
    if prim2 == 'fed':
        print('  primary 2 line currents [A rms]: A %8.3f  B %8.3f  C %8.3f'
              % tuple(rms(i2)))
    Itot = i1 + i2
    I0, Ip, In = seqcomp(Itot)
    print('  total line current sequence components [A rms]: I0 %7.3f  I+ %7.3f  I- %7.3f'
          % (rms(I0), rms(Ip), rms(In)))
    # powers (complex peaks -> S = V I* / 2)
    S1 = 0.5 * np.sum(U[0:3] * np.conj(i1))
    S2 = 0.5 * np.sum(U[3:6] * np.conj(i2)) if prim2 == 'fed' else 0.0
    S3 = 0.5 * v3 * np.conj(i3)
    Pcu = 0.5 * (d['R1'] * np.sum(np.abs(i1) ** 2) + d['R2'] * np.sum(np.abs(i2) ** 2)
                 + d['R3'] * np.abs(i3) ** 2)
    e = 1j * d['wb'] * lam
    Pfe = 0.5 * np.sum(np.abs(e) ** 2) / d['Rfe']
    print('  P_in = %9.1f W   P_out = %9.1f W   P_cu = %8.1f W   P_fe = %8.1f W   balance = %.2e W'
          % (S1.real + np.real(S2), S3.real, Pcu, Pfe,
             S1.real + np.real(S2) - S3.real - Pcu - Pfe))
    print('  Q_in = %9.1f var  Q_out = %9.1f var' % (S1.imag + np.imag(S2), S3.imag))
    print('  limb flux linkages (primary 1) [V s rms]: a %.4f  b %.4f  c %.4f'
          % tuple(rms(lam)))
    return dict(v3=v3, i3=i3, i1=i1, i2=i2, lam=lam, e3=e3)


# --------------------------------------------------------------------------
# Time-domain simulation (trapezoidal rule, backward Euler on the first step)
# --------------------------------------------------------------------------
def simulate(E, A, B, ufun, t_end, h, x0=None):
    """Trapezoidal rule on E x' = A x + B u (backward Euler on the first step)."""
    n = int(round(t_end / h))
    t = np.arange(n + 1) * h
    x = np.zeros((n + 1, 12))
    if x0 is not None:
        x[0] = x0
    M_be = E / h - A
    M_tr = E / h - A / 2
    for k in range(n):
        if k == 0:
            rhs = (E / h) @ x[k] + B @ ufun(t[k + 1])
            x[k + 1] = np.linalg.solve(M_be, rhs)
        else:
            rhs = (E / h + A / 2) @ x[k] + B @ (0.5 * (ufun(t[k]) + ufun(t[k + 1])))
            x[k + 1] = np.linalg.solve(M_tr, rhs)
    return t, x


def main():
    make_fig = '--no-fig' not in sys.argv
    p = dict(PAR)
    d = derived(p)
    print('Derived quantities (default parameters, ws = %s):' % np.array2string(d['w']))
    print('  g_w = %.4f   n2 = N2/N1 = %.4f   n3 = N3/N1 = %.4f' % (d['gw'], d['n2'], d['n3']))
    print('  Zb1 = %.4f Ohm  Zb3 = %.4f Ohm  R1 = %.3f mOhm  L1 = %.3f uH  R3 = %.3f mOhm  L3 = %.3f uH'
          % (d['Zb1'], d['Zb3'], 1e3 * d['R1'], 1e6 * d['L1'], 1e3 * d['R3'], 1e6 * d['L3']))
    print('  Lm = %.3f mH  Lm0 = %.3f mH  Rfe = %.1f Ohm' % (1e3 * d['Lm'], 1e3 * d['Lm0'], d['Rfe']))
    Ib1 = p['Sr'] / (np.sqrt(3) * p['V1'])
    I3n = p['Sr'] / p['V3']
    RLn = p['V3'] ** 2 / p['Sr']
    print('  base currents: Ib1 = %.2f A (3-ph base)   I3n = %.2f A   rated load RL = %.4f Ohm'
          % (Ib1, I3n, RLn))

    U1 = source(p['V1'])
    Uopen = np.concatenate([U1, np.zeros(3)])

    # --- 1. no load ---------------------------------------------------------
    E, A, B, d = build(p, prim2='open', load='open')
    X = phasor(E, A, B, Uopen, d['wb'])
    r = report('TEST 1 - no load, primary 1 at rated voltage, primary 2 open, secondary open', X, Uopen, d, p, 'open')
    # open-circuit voltage of primary 2
    lam = X[IX['lam']]
    e2 = d['n2'] * 1j * d['wb'] * lam
    print('  primary 2 open-circuit line-to-line voltage: %.2f V rms' % rms(e2[0] - e2[1]))
    print('  expected magnetizing current Ib1/lm = %.3f A, core-loss current = %.3f A'
          % (Ib1 / p['lm'], (p['V1'] / np.sqrt(3)) / (p['rfe'] * d['Zb1'])))

    # --- 2. rated resistive load -------------------------------------------
    E, A, B, d = build(p, prim2='open', load='rl', RL=RLn)
    X = phasor(E, A, B, Uopen, d['wb'])
    r = report('TEST 2 - rated resistive load on the secondary, primary 1 alone', X, Uopen, d, p, 'open')
    zsc = (p['r3'] + 1j * p['l3']) + 2 * (p['r1'] + 1j * p['l1'])
    print('  expected: V3 ~ V3n/|1 + z_sc| = %.2f V with z_sc = z3 + 2 z1, I_A ~ I_B ~ n3*I3 = %.1f A'
          % (p['V3'] / abs(1 + zsc), d['n3'] * rms(r['i3'])))

    # --- 3. short circuit ---------------------------------------------------
    E, A, B, d = build(p, prim2='open', load='short')
    X = phasor(E, A, B, Uopen, d['wb'])
    r = report('TEST 3 - secondary short circuit at rated primary voltage', X, Uopen, d, p, 'open')
    print('  I3/I3n = %.3f pu  -> z_sc = %.4f pu  (expected |z3 + 2 z1| = %.4f pu, u_cc = %.2f %%)'
          % (rms(r['i3']) / I3n, I3n / rms(r['i3']), abs(zsc), 100 * abs(zsc)))

    # --- 4. both primaries in parallel on the same source -------------------
    Upar = np.concatenate([U1, U1])
    E, A, B, d = build(p, prim2='fed', load='rl', RL=RLn)
    X = phasor(E, A, B, Upar, d['wb'])
    r = report('TEST 4 - primaries 1 and 2 in parallel on the same source, rated load', X, Upar, d, p, 'fed')
    E, A, B, d = build(p, prim2='fed', load='short')
    X = phasor(E, A, B, Upar, d['wb'])
    zsc2 = (p['r3'] + 1j * p['l3']) + (p['r1'] + 1j * p['l1'])
    print('  short circuit with both primaries: I3/I3n = %.3f pu -> z_sc = %.4f pu (expected |z3 + 2 (z1||z2)| = %.4f pu)'
          % (rms(X[IX['i3']]) / I3n, I3n / rms(X[IX['i3']]), abs(zsc2)))

    # --- 5. two primaries fed by two sources with a phase displacement ------
    for delta in (2.0, 5.0):
        U2 = source(p['V2'], phase_deg=delta)
        Ud = np.concatenate([U1, U2])
        E, A, B, d = build(p, prim2='fed', load='open')
        X = phasor(E, A, B, Ud, d['wb'])
        r = report('TEST 5 - primary 2 fed by a second source displaced by %.0f deg, no load' % delta, X, Ud, d, p, 'fed')
        icirc_pu = 2 * np.sin(np.deg2rad(delta) / 2) / abs((p['r1'] + 1j * p['l1']) + (p['r2'] + 1j * p['l2']))
        print('  circulating current I_A1/Ib1 = %.3f pu  (estimate 2 sin(delta/2)/|z1+z2| = %.3f pu)'
              % (rms(r['i1'][0]) / Ib1, icirc_pu))

    # --- 6. T-type distribution [1 -1/2 -1/2] ------------------------------
    pT = dict(p)
    pT['ws'] = np.array([1.0, -0.5, -0.5])
    E, A, B, dT = build(pT, prim2='open', load='rl', RL=RLn)
    X = phasor(E, A, B, Uopen, dT['wb'])
    r = report('TEST 6 - T-type secondary distribution ws = [1 -1/2 -1/2], rated load', X, Uopen, dT, pT, 'open')
    print('  n3 = %.4f, expected I_A = n3*I3n = %.1f A, I_B = I_C = %.1f A' % (dT['n3'], dT['n3'] * I3n, dT['n3'] * I3n / 2))
    E, A, B, dT = build(pT, prim2='open', load='short')
    X = phasor(E, A, B, Uopen, dT['wb'])
    print('  short circuit: z_sc = %.4f pu (expected |z3 + 2 z1| = %.4f pu, same as the line-to-line distribution)'
          % (I3n / rms(X[IX['i3']]), abs(zsc)))

    # --- 7. single-limb distribution [1 0 0] with and without neutral -------
    pS = dict(p)
    pS['ws'] = np.array([1.0, 0.0, 0.0])
    for neu in ('float', 'ground'):
        E, A, B, dS = build(pS, prim2='open', neutral1=neu, load='rl', RL=RLn)
        X = phasor(E, A, B, Uopen, dS['wb'])
        r = report('TEST 7 - single-limb distribution ws = [1 0 0], rated load, neutral N1 %s' % neu, X, Uopen, dS, pS, 'open')
        if neu == 'ground':
            zsc1 = (p['r3'] + 1j * p['l3']) + 3 * (p['r1'] + 1j * p['l1'])
            print('  expected with neutral: V3 ~ V3n/|1 + z_sc| with z_sc = z3 + 3 z1 -> %.2f V' % (p['V3'] / abs(1 + zsc1)))
        else:
            print('  without neutral the load current must return through the zero-sequence magnetizing path (lm0 = %.2f pu): the output collapses' % p['lm0'])

    # --- 8. time domain: no-load steady state, rated load connected at t = 2T
    T = 1 / p['fr']
    h = T / 2000
    def ufun(t):
        v = np.real(U1 * np.exp(1j * d['wb'] * t))
        return np.concatenate([v, np.zeros(3)])
    E, A, B, d = build(p, prim2='open', load='open')
    X0 = phasor(E, A, B, Uopen, d['wb'])          # consistent initial state
    t1, x1 = simulate(E, A, B, ufun, 2 * T, h, x0=np.real(X0))
    E, A, B, d = build(p, prim2='open', load='rl', RL=RLn)
    t2, x2 = simulate(E, A, B, lambda t: ufun(t + t1[-1]), 4 * T, h, x0=x1[-1])
    t = np.concatenate([t1, t1[-1] + t2[1:]])
    x = np.vstack([x1, x2[1:]])
    i1 = x[:, IX['i1']]
    i3 = x[:, IX['i3']]
    lam = x[:, IX['lam']]
    v3 = np.where(t > t1[-1], RLn * i3, d['n3'] * np.gradient(lam @ d['w'], t))
    last = t >= 5 * T
    print('\n' + '=' * 72)
    print('TEST 8 - time domain (trapezoidal rule): no load, rated load connected at t = 2T')
    print('-' * 72)
    print('  last-cycle rms: V3 = %.2f V, I3 = %.2f A, I_A = %.2f A, I_B = %.2f A, I_C = %.2f A'
          % (np.sqrt(np.mean(v3[last] ** 2)), np.sqrt(np.mean(i3[last] ** 2)),
             np.sqrt(np.mean(i1[last, 0] ** 2)), np.sqrt(np.mean(i1[last, 1] ** 2)),
             np.sqrt(np.mean(i1[last, 2] ** 2))))
    print('  (the phasor solution of test 2 gives V3 = 225.65 V, I3 = 426.55 A, I_A = 244.85 A, I_B = 246.26 A, I_C = 1.47 A)')

    if make_fig:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import os
        os.makedirs('figures', exist_ok=True)
        plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3})
        fig, ax = plt.subplots(3, 1, figsize=(6.3, 6.4), sharex=True)
        ax[0].plot(1e3 * t, i1[:, 0], label=r'$i_{A1}$')
        ax[0].plot(1e3 * t, i1[:, 1], label=r'$i_{B1}$')
        ax[0].plot(1e3 * t, i1[:, 2], label=r'$i_{C1}$')
        ax[0].set_ylabel('primary 1 line currents [A]')
        ax[0].legend(loc='upper left', ncol=3)
        ax[1].plot(1e3 * t, v3, 'k', label=r'$v_3$')
        ax[1].set_ylabel(r'$v_3$ [V]')
        ax1b = ax[1].twinx()
        ax1b.plot(1e3 * t, i3, 'C3', label=r'$i_3$')
        ax1b.set_ylabel(r'$i_3$ [A]', color='C3')
        ax1b.grid(False)
        ax[2].plot(1e3 * t, lam[:, 0], label=r'$\psi_a$')
        ax[2].plot(1e3 * t, lam[:, 1], label=r'$\psi_b$')
        ax[2].plot(1e3 * t, lam[:, 2], label=r'$\psi_c$')
        ax[2].set_ylabel('limb flux linkages [V s]')
        ax[2].set_xlabel('time [ms]')
        ax[2].legend(loc='upper left', ncol=3)
        for a_ in ax:
            a_.axvline(1e3 * t1[-1], color='0.5', lw=0.8, ls=':')
        fig.tight_layout()
        fig.savefig('figures/load_step_rated_load.pdf')

        # line currents for the three secondary distributions (steady state)
        fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.6))
        cases = [(np.array([1.0, -1.0, 0.0]), r'$\mathbf{w}=[1,-1,0]$', 'float'),
                 (np.array([1.0, -0.5, -0.5]), r'$\mathbf{w}=[1,-\frac{1}{2},-\frac{1}{2}]$', 'float'),
                 (np.array([1.0, 0.0, 0.0]), r'$\mathbf{w}=[1,0,0]$, N1 grounded', 'ground')]
        for a_, (wv, lab, neu) in zip(ax, cases):
            pc = dict(p)
            pc['ws'] = wv
            E, A, B, dc = build(pc, prim2='open', neutral1=neu, load='rl', RL=RLn)
            X = phasor(E, A, B, Uopen, dc['wb'])
            I = X[IX['i1']] / np.sqrt(2)
            Vref = U1 / np.sqrt(2) * 0.9      # phase voltages, scaled to the plot
            for k, nm in enumerate(('A', 'B', 'C')):
                a_.annotate('', xy=(Vref[k].real, Vref[k].imag), xytext=(0, 0),
                            arrowprops=dict(arrowstyle='->', color='0.65', lw=0.8))
                a_.text(Vref[k].real * 1.12, Vref[k].imag * 1.12, r'$\underline{V}_%s$' % nm,
                        color='0.45', ha='center', va='center', fontsize=7)
            for k, (c_, nm) in enumerate(zip(('C0', 'C1', 'C2'), ('A', 'B', 'C'))):
                if abs(I[k]) < 15:
                    continue
                a_.annotate('', xy=(I[k].real, I[k].imag), xytext=(0, 0),
                            arrowprops=dict(arrowstyle='-|>', color=c_, lw=1.8, shrinkA=0, shrinkB=0))
                a_.text(I[k].real * 1.15, I[k].imag * 1.15 + 20 * np.sign(I[k].imag + 1e-9) * (abs(I[k].imag) < 40),
                        r'$\underline{I}_%s$' % nm, color=c_, ha='center', va='center', fontsize=8)
            a_.set_aspect('equal')
            a_.set_xlim(-450, 450)
            a_.set_ylim(-450, 450)
            a_.set_xticks([-400, -200, 0, 200, 400])
            a_.set_yticks([-400, -200, 0, 200, 400])
            a_.tick_params(labelsize=7)
            a_.set_title(lab, fontsize=8)
            a_.set_xlabel('Re [A rms]', fontsize=8)
        ax[0].set_ylabel('Im [A rms]', fontsize=8)
        fig.tight_layout()
        fig.savefig('figures/line_currents_distributions.pdf')
        print('\nfigures written to ./figures/')


if __name__ == '__main__':
    main()
