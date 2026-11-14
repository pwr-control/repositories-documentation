#!/usr/bin/env python3
"""Figures and numbers of the note three_phase_plls_comparison (pwr-control/repositories-documentation).

Run from this folder:
    python3 make_figures.py [path/to/sim_results_01.mat]
Writes into ../figures/:  *.pdf and three_phase_plls_numbers.txt (every number quoted in the note, with LaTeX table rows).
Needs numpy, matplotlib and (for the logged run) h5py. The .mat file is the logged run of frt_dqpll.slx of
pwr-control/modelization-and-control (foundations_of_power_electronics/three_phase_dqPLLs_analysis, 2.25 s, VDE-AR-N
4110 test 25.4 asymmetric type 1 from 0.75 s for 915 ms, saved with `save sim_results_01.mat` after the run, v7.3).
Without the file the analytic and synthetic figures are produced and the logged-run figures are skipped.
The PLL algorithms are in pll_models.py (same folder).
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pll_models import *

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, '..', 'figures'); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'lines.linewidth': 1.0,
                     'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 120, 'axes.titlesize': 8.5})
W = 6.3                                     # [in] figure width = 16 cm text width
lines = []
def say(s=''):
    print(s); lines.append(s)
def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches='tight'); plt.close(fig); say('  -> figures/%s' % name)
def deg(x): return np.degrees(x)

p = Params(); ts = p.ts; w0 = p.omega_n; f0 = w0 / TWO_PI
COL = {'dq': 'C0', 'dq_frt': 'C9', 'fht': 'C1', 'ddsrf_model': '0.45', 'ddsrf_pos': 'C2', 'ddsrf_imp': 'C3'}
LAB = {'dq': 'dq-PLL ($k_p$ 1, $k_i$ 80)', 'dq_frt': 'dq-PLL fault gains ($k_p$ 0.2, $k_i$ 80)', 'fht': 'FHT-PLL',
       'ddsrf_model': 'DDSRF-PLL as built (negative feed)', 'ddsrf_pos': 'DDSRF-PLL, positive feed ($k_p$ 1, $k_i$ 80)',
       'ddsrf_imp': 'DDSRF-PLL improved ($k_p$ 0.32, $k_i$ 8, $\\omega$ init 1 pu)'}
KP_IMP, KI_IMP, CLAMP_IMP = 0.32, 8.0, (0.5, 1.5)

say('three_phase_plls_comparison: numbers of the note (%s)' % time.strftime('%Y-%m-%d'))
say('ts = %.0f us, omega_n = 2 pi %.0f Hz' % (ts * 1e6, f0))

# =====================================================================================================================
# 1. analytic part
# =====================================================================================================================
say('\n=== 1. dq-PLL loop: T(s) = omega_n (kp s + ki) / (s^2 + omega_n kp s + omega_n ki)')
def T_dq(s, kp, ki): return w0 * (kp * s + ki) / (s ** 2 + w0 * kp * s + w0 * ki)
def L_dq(s, kp, ki, wf=None):
    g = w0 * (kp * s + ki) / s ** 2
    return g * (wf / (s + wf)) if wf else g
def margins(kp, ki, wf=None):
    ws = np.logspace(0, 4.5, 40000); G = np.array([L_dq(1j * w, kp, ki, wf) for w in ws]); i = np.argmin(np.abs(np.abs(G) - 1))
    ph = np.degrees(np.unwrap(np.angle(G)))
    return ws[i], 180 + ph[i]
GAINS = [(1.0, 80.0, 'normal'), (0.2, 80.0, 'fault (kp 0.2)'), (0.2, 3.2, 'alternative fault (kp 0.2, ki 3.2)'), (KP_IMP, KI_IMP, 'DDSRF improved (kp 0.32, ki 8)')]
say('%-38s %8s %8s %6s %8s %8s %8s %10s %12s' % ('gain set', 'wn rad/s', 'wn Hz', 'zeta', '|T(2w0)|', '|T(6w0)|', '|T(12w0)|', 'wc, PM', 'wc, PM w/ LPF'))
tab_dq = []
for kp, ki, name in GAINS:
    wn = np.sqrt(w0 * ki); z = w0 * kp / (2 * wn); wc, pm = margins(kp, ki); wcf, pmf = margins(kp, ki, w0)
    say('%-38s %8.1f %8.2f %6.3f %8.3f %8.4f %8.4f %5.0f %4.0f  %5.0f %6.1f' % (name, wn, wn / TWO_PI, z, abs(T_dq(2j * w0, kp, ki)), abs(T_dq(6j * w0, kp, ki)), abs(T_dq(12j * w0, kp, ki)), wc, pm, wcf, pmf))
    tab_dq.append((name, kp, ki, wn, wn / TWO_PI, z, abs(T_dq(2j * w0, kp, ki)), abs(T_dq(6j * w0, kp, ki)), wc, pm, wcf, pmf))
say('LaTeX rows (gain set & kp & ki & wn [rad/s] & wn [Hz] & zeta & |T(j2w0)| & |T(j6w0)| & wc [rad/s] & PM & wc with LPF & PM with LPF):')
for r in tab_dq:
    say('  %s & %.2g & %g & %.0f & %.1f & %.2f & %.3f & %.3f & %.0f & \\ang{%.0f} & %.0f & \\ang{%.0f} \\\\' % r)
# discrete loop poles of stage B (forward Euler integrator, theta one step late)
say('discrete loop poles (theta[k+1] = theta[k] + ts w0 (kp e + i1), i1[k] = i1[k-1] + ki ts e):')
for kp, ki, name in GAINS[:3]:
    A = np.array([[1 - ts * w0 * (kp + ki * ts), ts * w0], [-ki * ts, 1.0]]); ev = np.linalg.eigvals(A)
    say('  %-36s z = %s -> s = %s rad/s (continuous %s)' % (name, np.round(ev, 5), np.round(np.log(ev) / ts, 1), np.round(np.roots([1, w0 * kp, w0 * ki]), 1)))

# ---- figure: closed loop of the dq-PLL and open loop of the DDSRF-PLL (with the LPF in the loop)
fr = np.logspace(0, 3.3, 600); ww = TWO_PI * fr
fig, ax = plt.subplots(1, 2, figsize=(W, 2.6))
for (kp, ki, name), c in zip(GAINS[:3], ['C0', 'C9', 'C4']):
    ax[0].semilogx(fr, 20 * np.log10(np.abs(T_dq(1j * ww, kp, ki))), color=c, label='$k_p$ %.2g, $k_i$ %g' % (kp, ki))
for h, lab in [(2, '100 Hz'), (6, '300 Hz')]:
    ax[0].axvline(h * f0, color='k', ls=':', lw=0.7); ax[0].text(h * f0 * 1.05, -38, lab, fontsize=6.5)
ax[0].set_xlabel('frequency [Hz]'); ax[0].set_ylabel('$|T(j\\omega)|$ [dB]'); ax[0].set_title('dq-PLL closed loop: phase from the detector input'); ax[0].legend(); ax[0].set_ylim(-40, 10)
for (kp, ki), c, wf in [((1.0, 80.0), 'C0', None), ((1.0, 80.0), 'C2', w0), ((KP_IMP, KI_IMP), 'C3', w0)]:
    G = np.array([L_dq(1j * w, kp, ki, wf) for w in ww]); wc, pm = margins(kp, ki, wf)
    lab = '$k_p$ %.2g, $k_i$ %g%s: $\\omega_c$ %.0f rad/s, PM %.0f$^\\circ$' % (kp, ki, ', LPF $\\omega_f = \\omega_0$' if wf else '', wc, pm)
    ax[1].semilogx(fr, 20 * np.log10(np.abs(G)), color=c, label=lab); ax[1].plot(wc / TWO_PI, 0, 'o', color=c, ms=4)
ax[1].axhline(0, color='k', lw=0.6)
ax[1].set_xlabel('frequency [Hz]'); ax[1].set_ylabel('$|L(j\\omega)|$ [dB]'); ax[1].set_title('open loop of the phase loop (dot: crossover)'); ax[1].legend(fontsize=6); ax[1].set_ylim(-40, 40)
savefig(fig, 'pll_loops.pdf')

# ---- FHT: continuous, intended discrete, actual discrete (delayed innovation), consistent states, delta = 0
say('\n=== 2. first harmonic tracker: H1 = (l1 s + l1 d w0 + l2)/D, H2 = (l2 s - l1 w0^2)/D, D = s^2 + (l1 + d w0) s + l1 d w0 + w0^2 + l2')
l1c, l2c = p.L_fht; d = p.delta_fht
say('continuous gains l1 = %.3f omega0 = %.1f 1/s, l2 = %.4f omega0^2 = %.0f 1/s^2; discrete gains Ld = [%.5f %.3f] (designed on I + A ts)' % (l1c / w0, l1c, l2c / w0 ** 2, l2c, p.Ld_fht[0], p.Ld_fht[1]))
def fht_matrices(delta=None):
    pp = Params(); pp.delta_fht = p.delta_fht if delta is None else delta
    if delta is not None:   # recompute the gains for this delta (same poles)
        A = np.array([[0.0, 1.0], [-w0 ** 2, -pp.delta_fht * w0]]); pp.Ld_fht = acker_observer(np.eye(2) + A * ts, np.array([[1.0, 0.0]]), np.exp(ts * np.array([-1.0, -4.0]) * w0))
    fh = FHT(pp); Ad = np.array([[fh.a11, fh.a12], [fh.a21, fh.a22]]); L = np.array([fh.l1, fh.l2]); return Ad, L
def G_cont(w, delta=d):
    A_ = np.array([[0.0, 1.0], [-w0 ** 2, -delta * w0]]); L = acker_observer(A_, np.array([[1.0, 0.0]]), np.array([-1.0, -4.0]) * w0); l1, l2 = L
    s = 1j * w; D = s ** 2 + (l1 + delta * w0) * s + l1 * delta * w0 + w0 ** 2 + l2
    H1 = (l1 * s + l1 * delta * w0 + l2) / D; H2 = (l2 * s - l1 * w0 ** 2) / D; return 0.5 * (H1 - 1j * H2 / w0), H1
def G_disc(w, kind='actual', delta=None):
    """complex gain of the positive sequence calculator for a component e^{j w t} (w signed) of alpha + j beta"""
    Ad, L = fht_matrices(delta); z = np.exp(1j * w * ts); C = np.array([1.0, 0.0])
    if kind == 'actual':          # innovation with the output of the previous step: x[k+1] = Ad x[k] + L (u[k] - x1[k-1])
        Fa = np.block([[Ad, -L[:, None]], [np.array([[1.0, 0.0, 0.0]])]]); La = np.array([L[0], L[1], 0.0])
        M = np.linalg.inv(z * np.eye(3) - Fa) @ La; h1 = M[0]; h2 = z * M[1] / w0
    else:
        F = Ad - np.outer(L, C); M = np.linalg.inv(z * np.eye(2) - F) @ L; h1 = M[0]
        h2 = (z if kind == 'intended' else 1.0) * M[1] / w0      # 'consistent': v = x2[k]/w0 (same instant as the output)
    return 0.5 * (h1 - 1j * h2), h1
Ad, L = fht_matrices(); Fa = np.block([[Ad, -L[:, None]], [np.array([[1.0, 0.0, 0.0]])]]); F = Ad - np.outer(L, [1, 0])
say('poles: intended z = %s (s = %s rad/s); actual (delayed innovation) z = %s (s = %s rad/s)' % (np.round(np.linalg.eigvals(F), 4), np.round(np.log(np.linalg.eigvals(F)) / ts, 0), np.round(np.linalg.eigvals(Fa), 4), np.round(np.log(np.linalg.eigvals(Fa).astype(complex)) / ts, 0).real))
COMP = [('fundamental +50 Hz', w0), ('negative sequence -50 Hz', -w0), ('+49 Hz', TWO_PI * 49), ('+51 Hz', TWO_PI * 51), ('-51 Hz', -TWO_PI * 51), ('DC', 1e-6),
        ('2nd (+100 Hz)', 2 * w0), ('5th (-250 Hz)', -5 * w0), ('7th (+350 Hz)', 7 * w0), ('11th (-550 Hz)', -11 * w0), ('13th (+650 Hz)', 13 * w0)]
say('positive sequence calculator gain |G|, angle [deg]: continuous | actual discrete | consistent states | consistent, delta 0')
tab_fht = []
for name, w in COMP:
    gc, _ = G_cont(w); ga, _ = G_disc(w, 'actual'); gk, _ = G_disc(w, 'consistent'); g0, _ = G_disc(w, 'consistent', 0.0)
    say('  %-26s %.4f %7.2f | %.4f %7.2f | %.4f %7.2f | %.4f %7.2f' % (name, abs(gc), deg(np.angle(gc)), abs(ga), deg(np.angle(ga)), abs(gk), deg(np.angle(gk)), abs(g0), deg(np.angle(g0))))
    tab_fht.append((name, abs(gc), deg(np.angle(gc)), abs(ga), deg(np.angle(ga)), abs(gk), deg(np.angle(gk)), abs(g0), deg(np.angle(g0))))
say('LaTeX rows (component & continuous & actual & consistent & consistent delta 0; each |G| / angle):')
for r in tab_fht:
    say('  %s & %.3f / \\ang{%.1f} & %.3f / \\ang{%.1f} & %.3f / \\ang{%.1f} & %.3f / \\ang{%.1f} \\\\' % r)
_, h1a = G_disc(w0, 'actual'); _, h1i = G_disc(w0, 'intended')
say('single FHT output at +50 Hz: actual %.4f, %.2f deg; intended %.4f, %.2f deg' % (abs(h1a), deg(np.angle(h1a)), abs(h1i), deg(np.angle(h1i))))
# figure: |G| over signed frequency, and the phase around the fundamental
fs = np.linspace(-700, 700, 2801); ws = TWO_PI * fs
fig, ax = plt.subplots(1, 2, figsize=(W, 2.6))
for kind, c, lab in [('cont', 'k', 'continuous time'), ('actual', 'C1', 'library C code (as is)'), ('consistent', 'C2', 'consistent states $x_1[k], x_2[k]$'), ('cons0', 'C3', 'consistent states, $\\delta = 0$')]:
    if kind == 'cont': g = np.array([G_cont(w)[0] for w in ws])
    elif kind == 'cons0': g = np.array([G_disc(w, 'consistent', 0.0)[0] for w in ws])
    else: g = np.array([G_disc(w, kind)[0] for w in ws])
    ax[0].plot(fs, np.abs(g), color=c, label=lab, lw=1.0 if kind != 'cont' else 1.4)
    m = (fs > 40) & (fs < 60); ax[1].plot(fs[m], deg(np.angle(g[m])), color=c, label=lab, lw=1.0 if kind != 'cont' else 1.4)
for h in [-11, -5, 1, 7, 13]: ax[0].axvline(h * f0, color='k', ls=':', lw=0.5)
ax[0].set_xlabel('frequency of the $\\alpha\\beta$ component [Hz] (negative = negative sequence)'); ax[0].set_ylabel('$|G|$ [pu/pu]'); ax[0].set_title('FHT positive-sequence calculator: gain'); ax[0].legend(fontsize=6); ax[0].set_ylim(0, 1.1)
ax[1].set_xlabel('frequency [Hz]'); ax[1].set_ylabel('angle of $G$ [deg]'); ax[1].set_title('phase of $G$ near 50 Hz'); ax[1].axhline(0, color='k', lw=0.5)
savefig(fig, 'fht_frontend.pdf')

# ---- DDSRF decoupling: poles vs omega_f
say('\n=== 3. DDSRF decoupling network: 1 - F(s) F(s + j 2 w0) = 0 -> s = -(wf + j w0) +- sqrt(wf^2 - w0^2)')
tab_dd = []
for name, wf in [('w0/2', w0 / 2), ('w0/sqrt(2)', w0 / np.sqrt(2)), ('w0 (model, tau = 1/w0)', w0), ('1.5 w0', 1.5 * w0), ('2 w0', 2 * w0)]:
    r = np.sqrt(complex(wf ** 2 - w0 ** 2)); s1 = -(wf + 1j * w0) + r; s2 = -(wf + 1j * w0) - r; slow = -max(s1.real, s2.real)
    F_ = lambda s: wf / (s + wf)
    say('  wf = %-24s tau %.2f ms  poles %s %s  slowest decay %.0f 1/s (%.2f ms)  |F(j w0)| %.3f |F(j2w0)| %.3f |F(j6w0)| %.3f' % (name, 1e3 / wf, np.round(s1, 1), np.round(s2, 1), slow, 1e3 / slow, abs(F_(1j * w0)), abs(F_(2j * w0)), abs(F_(6j * w0))))
    tab_dd.append((name.replace('w0', '$\\omega_0$').replace('sqrt(2)', '$\\sqrt{2}$'), 1e3 / wf, s1.real, s1.imag, s2.real, s2.imag, 1e3 / slow, abs(F_(2j * w0)), abs(F_(6j * w0))))
say('LaTeX rows (wf & tau [ms] & pole 1 & pole 2 & slowest time constant [ms] & |F(j2w0)| & |F(j6w0)|):')
for r in tab_dd:
    say('  %s & %.2f & $%.0f %+.0fj$ & $%.0f %+.0fj$ & %.2f & %.3f & %.3f \\\\' % r)
fig, ax = plt.subplots(1, 2, figsize=(W, 2.6))
k = np.linspace(0.2, 3.0, 400); wfv = k * w0; r = np.sqrt((wfv ** 2 - w0 ** 2).astype(complex))
s1 = -(wfv + 1j * w0) + r; s2 = -(wfv + 1j * w0) - r
for s_, lab in [(s1, 'pole 1'), (s2, 'pole 2')]:
    sc = ax[0].scatter(s_.real / w0, s_.imag / w0, c=k, cmap='viridis', s=6, label=lab)
for kk, mk in [(1 / np.sqrt(2), 's'), (1.0, 'o'), (2.0, '^')]:
    rr = np.sqrt(complex(kk ** 2 - 1)); ax[0].plot([(-(kk + 1j) + rr).real, (-(kk + 1j) - rr).real], [(-(kk + 1j) + rr).imag, (-(kk + 1j) - rr).imag], mk, color='r', ms=5, mfc='none', label='$\\omega_f/\\omega_0$ = %.3g' % kk)
plt.colorbar(sc, ax=ax[0], label='$\\omega_f / \\omega_0$'); ax[0].set_xlabel('Re(s) / $\\omega_0$'); ax[0].set_ylabel('Im(s) / $\\omega_0$'); ax[0].set_title('poles of the decoupling network'); ax[0].legend(fontsize=6, loc='lower left')
ax[1].plot(k, -np.maximum(s1.real, s2.real) / w0, color='C0'); ax[1].set_xlabel('$\\omega_f / \\omega_0$'); ax[1].set_ylabel('slowest decay rate / $\\omega_0$'); ax[1].set_title('decay of the sequence estimation error'); ax[1].axvline(1, color='r', ls=':', lw=0.8)
fig.tight_layout(); savefig(fig, 'ddsrf_poles.pdf')

# =====================================================================================================================
# 2. synthetic tests
# =====================================================================================================================
say('\n=== 4. synthetic tests (ideal three-phase set, ts = 125 us, every loop starts locked at 1 pu)')
T_END, T_EV, T_EV2 = 1.0, 0.4, 0.8
t = np.arange(0, T_END, ts)
def seg(tt, f=f0, phi=0.0, Vp=1.0, Vn=0.0, phi_n=0.0, harmonics=()):
    return three_phase(tt, Vp=Vp, Vn=Vn, phi_n=phi_n, f=f, phi_p=phi, harmonics=harmonics)
def scenario(name):
    m = t >= T_EV; m2 = (t >= T_EV) & (t < T_EV2); fr_ = np.full_like(t, 1.0); ref = w0 * t
    u = seg(t)
    if name == 'phase_step':  u[:, m] = seg(t[m], phi=np.radians(30)); ref = w0 * t + np.where(m, np.radians(30), 0.0)
    elif name == 'freq_step': u[:, m] = seg(t[m], f=51.0, phi=w0 * T_EV - TWO_PI * 51 * T_EV); ref = np.where(m, TWO_PI * 51 * t + w0 * T_EV - TWO_PI * 51 * T_EV, w0 * t); fr_ = np.where(m, 1.02, 1.0)
    elif name == 'unbalance': u[:, m2] = seg(t[m2], Vp=0.625, Vn=0.375, phi_n=np.pi)
    elif name == 'sag':       u[:, m2] = seg(t[m2], Vp=0.3)
    elif name == 'harmonics': u[:, m] = seg(t[m], harmonics=[(5, 0.05, -1), (7, 0.03, +1), (11, 0.02, -1), (13, 0.015, +1)])
    elif name == 'dc_offset': u[0, m] += 0.02
    return u, wrap(ref), fr_
def variants_synth(u):
    return {'dq': run_dq_pll(u, t, p, i1_init=1.0), 'dq_frt': run_dq_pll(u, t, p, kp=p.pll_p_frt, i1_init=1.0),
            'fht': run_fht_pll(u, t, p, i1_init=1.0, reset=False),
            'ddsrf_pos': run_ddsrf_pll(u, t, p, feed='pos', sign=+1.0, i1_init=1.0, reset=False),
            'ddsrf_imp': run_ddsrf_pll(u, t, p, feed='pos', sign=+1.0, i1_init=1.0, reset=False, kp=KP_IMP, ki=KI_IMP, clamp=CLAMP_IMP)}
SCEN = [('phase_step', 'phase jump +30 deg'), ('freq_step', 'frequency step 50 -> 51 Hz'), ('sag', 'balanced sag to 0.3 pu (400 ms)'),
        ('unbalance', 'type D dip: pos. 0.625, neg. 0.375 pu (400 ms)'), ('harmonics', 'harmonics 5th 5 %, 7th 3 %, 11th 2 %, 13th 1.5 %'), ('dc_offset', 'DC offset 2 % on phase a')]
synth = {}
for sc, title in SCEN:
    u, ref, fr_ = scenario(sc); v = variants_synth(u); synth[sc] = (u, ref, fr_, v)
    say('%s:' % title)
    for k_, r in v.items():
        e = wrap(r['phase'] - ref); w = r['omega']; pre = (t > 0.3) & (t < T_EV); post = (t >= T_EV) & (t < T_EV2); tail = (t > T_EV2 - 0.15) & (t < T_EV2)
        off = e[pre].mean(); ee = np.abs(e - off); idx = np.where(post & (ee > np.radians(0.5)))[0]
        r['m'] = dict(offset=deg(off), peak=deg(ee[post].max()), ripple=deg(e[tail].max() - e[tail].min()), wpeak=np.abs(w[post] - fr_[post]).max(), wripple=w[tail].max() - w[tail].min(), settle=1e3 * (t[idx.max()] - T_EV) if idx.size else 0.0)
        say('  %-10s offset %6.2f deg  peak %6.2f deg  ripple pp %6.3f deg  omega peak %.3f pu  omega ripple pp %.4f  settle(0.5 deg) %5.1f ms' % ((k_,) + tuple(r['m'].values())))
say('LaTeX rows per variant (scenario & peak & settle & ripple & omega peak):')
for k_ in ('dq', 'dq_frt', 'fht', 'ddsrf_pos', 'ddsrf_imp'):
    row = [LAB[k_]]
    for sc, _ in SCEN:
        m = synth[sc][3][k_]['m']; row.append('%.3g / %.0f / %.3g' % (m['peak'], m['settle'], m['ripple']))
    say('  ' + ' & '.join(row) + ' \\\\')
# figures: steps (phase, frequency, sag) and disturbances (unbalance, harmonics, dc)
def plot_synth(scs, name, zoom):
    fig, axs = plt.subplots(len(scs), 2, figsize=(W, 1.9 * len(scs) + 0.4), gridspec_kw={'width_ratios': [2.2, 1]})
    for i, sc in enumerate(scs):
        u, ref, fr_, v = synth[sc]; title = dict(SCEN)[sc]
        for k_ in ('dq', 'dq_frt', 'fht', 'ddsrf_pos', 'ddsrf_imp'):
            e = deg(wrap(v[k_]['phase'] - ref)); e -= v[k_]['m']['offset'] * (sc not in ('phase_step',))
            axs[i, 0].plot(t, e, color=COL[k_], label=LAB[k_], lw=0.9)
            axs[i, 1].plot(t, v[k_]['omega'], color=COL[k_], lw=0.9)
        axs[i, 0].set_xlim(zoom[sc]); axs[i, 1].set_xlim(zoom[sc]); axs[i, 0].set_ylabel('phase error [deg]'); axs[i, 1].set_ylabel('$\\omega$ [pu]')
        axs[i, 0].set_title(title, loc='left'); axs[i, 1].set_title('frequency', loc='left')
        if sc in ('harmonics', 'dc_offset', 'unbalance'):
            axs[i, 0].set_ylim(*{'harmonics': (-0.4, 0.4), 'dc_offset': (-5, 5), 'unbalance': (-25, 25)}[sc])
    axs[-1, 0].set_xlabel('time [s]'); axs[-1, 1].set_xlabel('time [s]')
    h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95)); savefig(fig, name)
plot_synth(['phase_step', 'freq_step', 'sag'], 'synth_steps.pdf', {'phase_step': (0.38, 0.65), 'freq_step': (0.38, 0.65), 'sag': (0.38, 0.65)})
plot_synth(['unbalance', 'harmonics', 'dc_offset'], 'synth_disturbances.pdf', {'unbalance': (0.38, 0.65), 'harmonics': (0.60, 0.65), 'dc_offset': (0.60, 0.70)})

# =====================================================================================================================
# 3. logged run of the model
# =====================================================================================================================
mat = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'sim_results_01.mat')
if not os.path.exists(mat):
    say('\n=== 5. logged run: file %s not found, figures skipped' % mat); open(os.path.join(FIG, 'three_phase_plls_numbers.txt'), 'w').write('\n'.join(lines) + '\n'); sys.exit(0)
import h5py
with h5py.File(mat, 'r') as f:
    def get(name):
        a = np.array(f[name]); return a if a.ndim == 2 and a.shape[0] == 3 else (a.T if a.ndim == 2 and a.shape[1] == 3 else a.ravel())
    tl = get('time_ts_afe_sim'); u = get('ugd_abc_grid_sim'); ph_log = get('phase_grid_fhtsr_dqpll_sim'); om_log = get('omega_grid_fhtsr_dqpll_sim')
    usr = get('u_fhtsr_alpha_sim') + 1j * get('u_fhtsr_beta_sim'); ufa = get('u_fht_alpha_sim') + 1j * get('u_fht_beta_sim')
say('\n=== 5. logged run %s: %d samples, %.3f-%.3f s' % (os.path.basename(mat), tl.size, tl[0], tl[-1]))
T_F0, T_F1 = 0.75, 0.75 + 0.915
al, be = clarke(*u); ang = np.unwrap(np.arctan2(be, al)); mfit = (tl > 0.30) & (tl < 0.74)
slope, th0 = np.polyfit(tl[mfit], ang[mfit], 1); th_ref = wrap(slope * tl + th0)
say('reference angle: linear fit of the alpha-beta angle on 0.30-0.74 s: f = %.4f Hz (%.5f pu), theta0 = %.4f rad; logged FHT-PLL omega mean %.5f pu' % (slope / TWO_PI, slope / w0, th0, om_log[mfit].mean()))
Np = int(round(1 / f0 / ts)); zz = (al + 1j * be) * np.exp(-1j * (slope * tl + th0)); zc = np.cumsum(np.concatenate([[0], zz])); pos = (zc[Np:] - zc[:-Np]) / Np
zn = (al + 1j * be) * np.exp(1j * (slope * tl + th0)); zcn = np.cumsum(np.concatenate([[0], zn])); neg = (zcn[Np:] - zcn[:-Np]) / Np; tp = tl[Np - 1:]
for lo, hi, name in [(0.5, 0.74, 'before the fault'), (0.9, 1.6, 'in the fault'), (1.9, 2.2, 'after the fault')]:
    mm = (tp > lo) & (tp < hi); say('  one-period DFT %-16s |pos| %.4f pu angle %6.2f deg   |neg| %.4f pu' % (name, np.abs(pos[mm]).mean(), deg(np.angle(pos[mm])).mean(), np.abs(neg[mm]).mean()))
e_sr = wrap(np.angle(-usr) - th_ref); e_fh = wrap(np.angle(ufa) - th_ref); mm = (tl > 0.5) & (tl < 0.74); mf = (tl > 1.2) & (tl < 1.65)
say('FHT front end (logged): angle(u_fht) - ref = %.3f deg; angle(-u_sr) - ref = %.3f deg before the fault, %.3f deg mean and %.2f deg pp in the fault' % (deg(e_fh[mm].mean()), deg(e_sr[mm].mean()), deg(e_sr[mf].mean()), deg(e_sr[mf].max() - e_sr[mf].min())))
e_log = wrap(ph_log - th_ref); say('logged FHT-PLL phase - ref: %.3f deg before the fault' % deg(e_log[mm].mean()))
# runs
t0 = time.time()
frt = ((tl >= T_F0) & (tl < T_F1)).astype(float)            # frt_active is not logged: taken equal to the fault window
runs = {'dq': run_dq_pll(u, tl, p), 'dq_sched': run_dq_pll(u, tl, p, kp_sched=np.where(frt > 0.5, p.pll_p_frt, p.pll_p)), 'fht': run_fht_pll(u, tl, p),
        'ddsrf_model': run_ddsrf_pll(u, tl, p, feed='neg', sign=-1.0), 'ddsrf_pos': run_ddsrf_pll(u, tl, p, feed='pos', sign=+1.0),
        'ddsrf_pos_init1': run_ddsrf_pll(u, tl, p, feed='pos', sign=+1.0, i1_init=1.0),
        'ddsrf_imp': run_ddsrf_pll(u, tl, p, feed='pos', sign=+1.0, i1_init=1.0, kp=KP_IMP, ki=KI_IMP, clamp=CLAMP_IMP)}
say('offline runs: %.1f s' % (time.time() - t0))
ev = np.abs(wrap(ph_log - runs['fht']['phase'])); say('check: offline FHT-PLL vs logged phase: max |diff| %.2e rad (%.2e after 0.02 s); omega max |diff| %.2e pu' % (ev.max(), ev[tl > 0.02].max(), np.abs(om_log - runs['fht']['omega']).max()))
LABL = dict(LAB); LABL['dq_sched'] = 'dq-PLL as in the model ($k_p$ 1 $\\to$ 0.2 in the fault)'; LABL['ddsrf_pos_init1'] = 'DDSRF-PLL, positive feed, $\\omega$ init 1 pu'
COLL = dict(COL); COLL['dq_sched'] = 'C0'; COLL['ddsrf_pos_init1'] = 'C8'
say('metrics (phase error vs the reference angle; offset = mean before the fault; lock = last time |e - offset| > 1 deg before the fault):')
say('%-16s %8s %7s %9s %9s %9s %9s %9s %9s' % ('variant', 'offset', 'lock s', 'in:peak', 'in:settle', 'ripple pp', 'w ripple', 'out:peak', 'out:settle'))
rows = []
for k_, r in runs.items():
    e = wrap(r['phase'] - th_ref); w = r['omega']; off = e[mm].mean(); ee = e - off
    idx = np.where((tl < 0.74) & (np.abs(ee) > np.radians(1)))[0]; lock = tl[idx.max()] if idx.size else 0.0
    win_in = (tl >= T_F0) & (tl < T_F0 + 0.3); win_out = (tl >= T_F1) & (tl < T_F1 + 0.3)
    s_in = np.where((tl >= T_F0) & (tl < T_F1) & (np.abs(ee) > np.radians(1)))[0]; s_out = np.where((tl >= T_F1) & (np.abs(ee) > np.radians(1)))[0]
    m = dict(offset=deg(off), lock=lock, in_peak=deg(np.abs(ee[win_in]).max()), in_settle=1e3 * (tl[s_in.max()] - T_F0) if s_in.size else 0.0,
             ripple=deg(ee[mf].max() - ee[mf].min()), wripple=w[mf].max() - w[mf].min(), out_peak=deg(np.abs(ee[win_out]).max()), out_settle=1e3 * (tl[s_out.max()] - T_F1) if s_out.size else 0.0,
             in_wpeak=np.abs(w[win_in] - slope / w0).max(), out_wpeak=np.abs(w[win_out] - slope / w0).max())
    r['m'] = m
    say('%-16s %8.3f %7.3f %9.2f %9.1f %9.3f %9.4f %9.2f %9.1f   (omega peaks %.3f / %.3f pu)' % (k_, m['offset'], m['lock'], m['in_peak'], m['in_settle'], m['ripple'], m['wripple'], m['out_peak'], m['out_settle'], m['in_wpeak'], m['out_wpeak']))
    rows.append('  %s & %.2f & %.0f & %.1f & %.0f & %.2f & %.3f & %.1f & %.0f \\\\' % (LABL[k_], m['offset'], 1e3 * m['lock'], m['in_peak'], m['in_settle'], m['ripple'], m['wripple'], m['out_peak'], m['out_settle']))
say('LaTeX rows (variant & offset [deg] & lock [ms] & fault-in peak [deg] & settle [ms] & ripple pp [deg] & omega ripple pp [pu] & recovery peak [deg] & settle [ms]):'); [say(r_) for r_ in rows]
# ---- figure: grid voltage and sequences
fig, axs = plt.subplots(1, 3, figsize=(W, 2.0), gridspec_kw={'width_ratios': [1, 1, 1.3]})
for i, (lo, hi, ttl) in enumerate([(0.73, 0.80, 'fault start (0.75 s)'), (1.64, 1.71, 'fault end (1.665 s)')]):
    mz = (tl > lo) & (tl < hi)
    for j, lab in enumerate('abc'): axs[i].plot(tl[mz], u[j, mz], label='$u_%s$' % lab, lw=0.8)
    axs[i].set_xlim(lo, hi); axs[i].set_title(ttl, loc='left'); axs[i].set_xlabel('time [s]')
axs[0].set_ylabel('capacitor voltage [pu]'); axs[0].legend(fontsize=6, ncol=3, loc='lower left')
axs[2].plot(tp, np.abs(pos), label='positive sequence', color='C2'); axs[2].plot(tp, np.abs(neg), label='negative sequence', color='C3'); axs[2].set_xlabel('time [s]'); axs[2].set_ylabel('one-period DFT [pu]'); axs[2].set_title('sequence magnitudes', loc='left'); axs[2].legend(fontsize=6); axs[2].set_xlim(0, tl[-1])
fig.tight_layout(); savefig(fig, 'logged_voltage.pdf')
# ---- figure: the three PLLs as built on the logged run (phase error, omega), full record and zooms
def plot_logged(keys, name, remove_offset=True, ylim=None, ylim_zoom=None):
    fig, axs = plt.subplots(2, 3, figsize=(W, 3.9), gridspec_kw={'width_ratios': [2, 1, 1]})
    for k_ in keys:
        r = runs[k_]; e = deg(wrap(r['phase'] - th_ref)) - (r['m']['offset'] if remove_offset else 0.0)
        for j, (lo, hi) in enumerate([(0, tl[-1]), (0.74, 0.95), (1.65, 1.86)]):
            mz = (tl >= lo) & (tl <= hi); axs[0, j].plot(tl[mz], e[mz], color=COLL[k_], label=LABL[k_], lw=0.8); axs[1, j].plot(tl[mz], r['omega'][mz], color=COLL[k_], lw=0.8); axs[0, j].set_xlim(lo, hi); axs[1, j].set_xlim(lo, hi)
    for j, ttl in enumerate(['whole record', 'fault start', 'fault end']): axs[0, j].set_title(ttl, loc='left'); axs[1, j].set_xlabel('time [s]')
    axs[0, 0].set_ylabel('phase error [deg]' + (' (offset removed)' if remove_offset else '')); axs[1, 0].set_ylabel('$\\omega$ [pu]'); axs[0, 0].legend(fontsize=6, loc='upper right')
    if ylim: axs[0, 0].set_ylim(*ylim)
    if ylim_zoom: axs[0, 1].set_ylim(*ylim_zoom); axs[0, 2].set_ylim(*ylim_zoom)
    for j in range(3): axs[1, j].set_ylim(0.6, 1.4) if j == 0 else axs[1, j].set_ylim(0.85, 1.15)
    fig.tight_layout(); savefig(fig, name)
plot_logged(['dq_sched', 'fht', 'ddsrf_model'], 'logged_as_built.pdf', remove_offset=False, ylim=(-180, 180), ylim_zoom=(-180, 180))
plot_logged(['dq', 'dq_sched', 'fht', 'ddsrf_imp'], 'logged_compare.pdf', remove_offset=True, ylim=(-25, 25), ylim_zoom=(-25, 25))
# ---- figure: start-up and the 100 Hz ripple
fig, axs = plt.subplots(2, 2, figsize=(W, 3.9))
for k_ in ('dq', 'fht', 'ddsrf_pos', 'ddsrf_pos_init1', 'ddsrf_imp'):
    r = runs[k_]; mz = tl < 0.35; axs[0, 0].plot(tl[mz], deg(wrap(r['phase'] - th_ref))[mz], color=COLL[k_], label=LABL[k_], lw=0.8); axs[1, 0].plot(tl[mz], r['omega'][mz], color=COLL[k_], lw=0.8)
axs[0, 0].set_title('start (zero frequency, as the model; or $\\omega$ init 1 pu)', loc='left'); axs[0, 0].set_ylabel('phase error [deg]'); axs[1, 0].set_ylabel('$\\omega$ [pu]'); axs[1, 0].set_xlabel('time [s]'); axs[0, 0].set_xlim(0, 0.35); axs[1, 0].set_xlim(0, 0.35)
for k_ in ('dq', 'dq_sched', 'fht', 'ddsrf_pos_init1', 'ddsrf_imp'):
    r = runs[k_]; mz = (tl > 1.40) & (tl < 1.44); e = deg(wrap(r['phase'] - th_ref)) - r['m']['offset']; axs[0, 1].plot(tl[mz], e[mz], color=COLL[k_], label=LABL[k_], lw=0.9); axs[1, 1].plot(tl[mz], r['omega'][mz], color=COLL[k_], lw=0.9)
axs[0, 1].set_title('in the fault: 100 Hz ripple (offset removed)', loc='left'); axs[1, 1].set_xlabel('time [s]'); axs[0, 1].set_xlim(1.40, 1.44); axs[1, 1].set_xlim(1.40, 1.44)
h0, l0 = axs[0, 0].get_legend_handles_labels(); h1, l1 = axs[0, 1].get_legend_handles_labels()
hh = dict(zip(l0 + l1, h0 + h1)); fig.legend(list(hh.values()), list(hh.keys()), loc='lower center', ncol=3, fontsize=6.5, frameon=False)
fig.tight_layout(rect=(0, 0.09, 1, 1)); savefig(fig, 'logged_startup_ripple.pdf')
# ---- figure: DDSRF-PLL as built, fixed feed, improved: phase error, omega and the sequence outputs
fig, axs = plt.subplots(3, 1, figsize=(W, 5.6), sharex=True)
for k_ in ('ddsrf_model', 'ddsrf_pos', 'ddsrf_imp'):
    r = runs[k_]
    if k_ != 'ddsrf_pos': axs[0].plot(tl, deg(wrap(r['phase'] - th_ref)), color=COLL[k_], label=LABL[k_], lw=0.8)
    axs[1].plot(tl, r['omega'], color=COLL[k_], lw=0.8, label=LABL[k_])
    axs[2].plot(tl, np.hypot(r['pos_xi'], r['pos_eta']), color=COLL[k_], lw=0.8, label=LABL[k_] + ': $|u_{pos}|$'); axs[2].plot(tl, np.hypot(r['neg_xi'], r['neg_eta']), color=COLL[k_], lw=0.8, ls='--', label='$|u_{neg}|$')
axs[2].plot(tp, np.abs(pos), 'k:', lw=0.9, label='true positive sequence (DFT)'); axs[2].plot(tp, np.abs(neg), 'k-.', lw=0.9, label='true negative sequence (DFT)')
axs[0].set_ylabel('phase error [deg]'); axs[1].set_ylabel('$\\omega$ [pu]'); axs[2].set_ylabel('sequence outputs [pu]'); axs[2].set_xlabel('time [s]'); axs[0].set_xlim(0, tl[-1]); axs[1].set_ylim(-1.3, 1.3)
axs[0].set_title('phase error (as built, improved)', loc='left'); axs[1].set_title('estimated frequency (also the positive feed with zero start, locked on $-\\omega$)', loc='left'); axs[2].set_title('outputs of the DDSRF block: $|u_{pos}|$ solid, $|u_{neg}|$ dashed', loc='left')
h, l = axs[1].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, fontsize=6.5, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1)); savefig(fig, 'logged_ddsrf.pdf')
# numbers for the FHT lead vs logged and the ddsrf_model description
r = runs['ddsrf_model']; e = wrap(r['phase'] - th_ref)
say('DDSRF-PLL as built: omega mean 0.5-0.74 s %.4f pu, phase error at 0.5/0.7 s %.1f/%.1f deg; in the fault (1.2-1.65 s) omega %.4f pu, error mean %.1f deg pp %.2f deg; |u_neg| output in the fault %.3f pu' % (
    r['omega'][mm].mean(), deg(e[np.searchsorted(tl, 0.5)]), deg(e[np.searchsorted(tl, 0.7)]), r['omega'][mf].mean(), deg(e[mf].mean()), deg(e[mf].max() - e[mf].min()), np.hypot(r['neg_xi'], r['neg_eta'])[mf].mean()))
r = runs['ddsrf_pos']; say('DDSRF-PLL positive feed, zero start: omega at 0.7 s %.4f pu (locked on -omega), |u_neg| output %.3f pu' % (r['omega'][np.searchsorted(tl, 0.7)], np.hypot(r['neg_xi'], r['neg_eta'])[np.searchsorted(tl, 0.7)]))
open(os.path.join(FIG, 'three_phase_plls_numbers.txt'), 'w').write('\n'.join(lines) + '\n')
say('done')
