#!/usr/bin/env python3
"""Figures and numbers of the note single_phase_pll (pwr-control/repositories-documentation,
repos_doc/single_phase_lock_loop), rev. 01.

Run from this folder:
    python3 make_figures.py [path/to/sim_results_cases.mat]
Writes into ../figures/: *.pdf and single_phase_pll_numbers.txt (every number quoted in the note, with LaTeX table rows).
Needs numpy, scipy and matplotlib. The PLLs are the step-by-step replicas of sp_pll_models.py (same folder): they
reproduce the Simulink model single_phase_pll_analysis.slx of pwr-control/modelization-and-control
(foundations_of_power_electronics/single_phase_dqPLLs_analysis) within 2e-12 rad. The optional .mat file holds the
six test cases simulated by the model: the file written by run_test_cases(1:6, 'sim_results_cases.mat') in the model
folder (struct array results), or a flat export with the matrices caseN = [t u theta_g omega_g, phase and omega of the
Park PLL, FHT-PLL, DDSRF-PLL]. When given, the script compares it with the replicas and uses the logged signals for
the tables and figures of the test cases; without it, everything comes from the replicas.
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal as ss
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sp_pll_models as m

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5, 'lines.linewidth': 1.0,
                     'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 120, 'axes.titlesize': 8.5})
WID = 6.3                                    # [in] text width
lines = []


def say(s=''):
    print(s)
    lines.append(s)


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches='tight')
    plt.close(fig)
    say('  -> figures/%s' % name)


def deg(x):
    return np.degrees(x)


pp = np.polynomial.polynomial
P = m.default_params()
ts, w0 = P['ts'], P['w0']
f0 = w0 / m.TWO_PI
wq = 1.0 / P['tau_qsg']                      # corner of the low passes of the Park QSG
wf = 1.0 / P['tau_ddsrf']                    # corner of the low passes of the DDSRF
Pu = m.fht_variant(P, 'undamped')
NAMES = ('park', 'fht', 'ddsrf')
TITLE = {'park': 'Park PLL', 'fht': 'FHT-PLL', 'ddsrf': 'DDSRF-PLL'}
COL = {'park': 'C0', 'fht': 'C1', 'ddsrf': 'C2'}
t_run = np.arange(8001) * ts                 # 1 s, as the model

say('single_phase_pll: numbers of the note (%s)' % time.strftime('%Y-%m-%d'))
say('ts = %.0f us, omega_n = 2 pi %.0f Hz = %.2f rad/s' % (ts * 1e6, f0, w0))

# =====================================================================================================================
# 1. quadrature signal generators: complex gain G(nu) on u = cos(nu t): alpha + j beta = G(nu) e^{j nu t} + G(-nu) e^{-j nu t}
# =====================================================================================================================
say('\n=== 1. front ends: alpha + j beta = G(nu) e^{j nu t} + G(-nu) e^{-j nu t} for u = cos(nu t)')


def Q_park(s):
    """beta / u of the Park QSG (SOGI quadrature output with k = wq / w0), PLL locked at w0."""
    return w0 * wq / (s ** 2 + wq * s + w0 ** 2)


def D_park(s):
    """alpha_hat / u of the Park QSG (band pass, not used by the phase detector)."""
    return wq * s / (s ** 2 + wq * s + w0 ** 2)


def G_park(nu):
    """the phase detector of the block rotates u + j beta: alpha is the raw input."""
    return 0.5 * (1.0 + 1j * Q_park(1j * nu))


def G_sogi(nu):
    """the same generator with alpha = alpha_hat (SOGI-QSG as in the literature), for reference."""
    return 0.5 * (D_park(1j * nu) + 1j * Q_park(1j * nu))


def fht_h(Q, nu):
    """x1 / u and x2 / (w u) of the discrete tracker of Q at the frequency nu (outputs = a priori state x[k])."""
    z = np.exp(1j * nu * ts)
    Acl = Q['Ad_fht'] - np.outer(Q['Ld_fht'], Q['C_fht'])
    x = np.linalg.solve(z * np.eye(Acl.shape[0]) - Acl, Q['Ld_fht'])
    return x[0], x[1] / Q['w_fht']


def G_fht(nu, Q=P):
    h1, h2 = fht_h(Q, nu)
    return 0.5 * (h1 - 1j * h2)               # FHTalpha1: alpha = x1, beta = -x2 / w


def G_fht3(nu, Q=P):
    return G_fht(nu, Q) ** 2                  # stage A of the FHT-PLL: u_sr = -G^2 (sign taken by the gain -1)


FRONT = [('Park QSG (alpha = u)', G_park), ('SOGI k = 0.5 (alpha = D u)', G_sogi), ('FHT (class)', G_fht),
         ('FHT undamped', lambda nu: G_fht(nu, Pu)), ('FHT-PLL stage A (class)', G_fht3),
         ('FHT-PLL stage A undamped', lambda nu: G_fht3(nu, Pu))]
say('%-28s %8s %8s %8s %8s %8s %8s %8s %8s %8s %10s %8s' % ('front end', '|G(w0)|', 'ang deg', '|G(-w0)|', '|2G(0)|',
    'h2 +/-', 'h3 +/-', 'h5 +/-', 'h7 +/-', '', '49 Hz ang', 'img'))
fe_rows = []
for name, G in FRONT:
    g0 = G(w0)
    hs = []
    for h in (2, 3, 5, 7):
        hs += [abs(G(h * w0)) / abs(g0), abs(G(-h * w0)) / abs(g0)]
    g49 = G(2 * np.pi * 49)
    dc = abs(2 * G(1e-9)) / abs(g0)
    say('%-28s %8.4f %8.3f %8.4f %8.3f ' % (name, abs(g0), deg(np.angle(g0)), abs(G(-w0)) / abs(g0), dc)
        + ' '.join('%.2f/%.2f' % (hs[2 * i], hs[2 * i + 1]) for i in range(4))
        + '  %10.3f %8.4f' % (deg(np.angle(g49)), abs(G(-2 * np.pi * 49)) / abs(g49)))
    fe_rows.append((name, abs(g0), deg(np.angle(g0)), abs(G(-w0)) / abs(g0), dc, hs, deg(np.angle(g49)),
                    abs(G(-2 * np.pi * 49)) / abs(g49)))
say('LaTeX rows (front end & |G(w0)| & angle & image & DC & h2 & h3 & h5 & h7 (+/-) & angle at 49 Hz & image at 49 Hz):')
for r in fe_rows:
    say('  %s & %.4f & \\ang{%.2f} & %.4f & %.2f & %s & \\ang{%.2f} & %.4f \\\\' % (
        r[0], r[1], r[2], r[3], r[4], ' & '.join('%.2f/%.2f' % (r[5][2 * i], r[5][2 * i + 1]) for i in range(4)),
        r[6], r[7]))
# the tracker as an observer: poles, SOGI equivalence, forward Euler and damping
l1c, l2c = (5 - P['fht_delta']) * w0, 4 * w0 ** 2 - P['fht_delta'] * w0 * (5 - P['fht_delta']) * w0 - w0 ** 2
say('tracker (class): continuous gains l1 = %.3f w0, l2 = %.4f w0^2 (poles -w0, -4 w0); discrete Ld = %s' % (
    l1c / w0, l2c / w0 ** 2, np.array2string(P['Ld_fht'], precision=5)))
say('  undamped exact: Ld = %s' % np.array2string(Pu['Ld_fht'], precision=5))
say('  x1/u at DC = (p1 p2 - w0^2)/(p1 p2) = 3/4 for poles -w0, -4 w0 (undamped model); SOGI when p1 p2 = w0^2, k = -(p1+p2)/w0')
zf = np.linalg.eigvals(P['Ad_fht'])
zu = np.linalg.eigvals(Pu['Ad_fht'])
say('  oscillator model: |eig(I + A ts)| = %.6f (sigma = %.3f 1/s), continuous -delta w0/2 = %.3f 1/s; exact undamped |eig| = %.6f' % (
    abs(zf[0]), np.log(abs(zf[0])) / ts, -P['fht_delta'] * w0 / 2, abs(zu[0])))
for name, Q in (('class', P), ('undamped', Pu)):
    h1, h2 = fht_h(Q, w0)
    say('  %-9s at 50 Hz: alpha %.4f / %.3f deg, beta = -x2/w %.4f / %.3f deg (ideal -90)' % (
        name, abs(h1), deg(np.angle(h1)), abs(h2), deg(np.angle(-h2))))

# ---- figure: gains of the front ends
nu = np.linspace(-400, 400, 3201) * 2 * np.pi
fig, ax = plt.subplots(1, 2, figsize=(WID, 2.5), gridspec_kw={'width_ratios': [1.6, 1]})
for (name, G), c, ls in zip([FRONT[0], FRONT[1], FRONT[2], FRONT[4]], ['C0', '0.55', 'C2', 'C1'], ['-', ':', '-', '--']):
    g = np.array([abs(G(x)) for x in nu]) / abs(G(w0))
    ax[0].plot(nu / 2 / np.pi, g, color=c, ls=ls, label=name)
for h in (-7, -5, -3, -1, 0, 1, 3, 5, 7):
    ax[0].axvline(h * f0, color='k', lw=0.5, alpha=0.25)
ax[0].set_xlabel('signed frequency of the component $e^{j\\nu t}$ [Hz]')
ax[0].set_ylabel('$|G(\\nu)|\\,/\\,|G(\\omega_0)|$')
ax[0].set_ylim(0, 1.15)
ax[0].set_xlim(-400, 400)
ax[0].legend(loc='upper left', fontsize=6)
ax[0].set_title('gain on the components of $\\alpha + j\\beta$ (negative: image)')
fz = np.linspace(47, 53, 241)
for (name, G), c, ls in zip([FRONT[2], FRONT[3], FRONT[4], FRONT[5]], ['C2', 'C2', 'C1', 'C1'], ['-', ':', '-', ':']):
    ax[1].plot(fz, [deg(np.angle(G(2 * np.pi * f))) for f in fz], color=c, ls=ls, label=name)
ax[1].axhline(0, color='k', lw=0.5)
ax[1].set_xlabel('frequency [Hz]')
ax[1].set_ylabel('angle of $G$ [deg]')
ax[1].set_title('trackers at fixed $\\omega_0$: phase near 50 Hz')
ax[1].legend(fontsize=5.5, loc='lower left')
fig.tight_layout()
savefig(fig, 'qsg_gains.pdf')

# =====================================================================================================================
# 2. loop models
# =====================================================================================================================
say('\n=== 2. loop models')


def Kpd(s, V=1.0):
    """averaged phase detector of the Park PLL (QSG in the loop): eta = V (s + wq) / (2 s + wq) (phase error)."""
    return V * (s + wq) / (2 * s + wq)


def Kpd_f(s, V=1.0):
    """the same detector read after the low-pass filter (variant eta_filtered): eta_f = V wq / (2 s + wq)."""
    return V * wq / (2 * s + wq)


def park_loop(kp, ki, V=1.0, filt=False):
    """open loop of the Park PLL: w0 (kp s + ki) / s^2 times the averaged detector."""
    K = Kpd_f if filt else Kpd
    return dict(L=lambda s: PI_(s, kp, ki) * K(s, V) / s, pre=None, post=obs, kp=kp, ki=ki, V=V, filt=filt)


def D_eff(s):
    """decoupling network + low passes of the DDSRF on a small phase error (exact for a positive-sequence input)."""
    num = wf * (s ** 3 + 2 * wf * s ** 2 + 4 * w0 ** 2 * s + 4 * w0 ** 2 * wf)
    den = s ** 4 + 4 * wf * s ** 3 + (4 * wf ** 2 + 4 * w0 ** 2) * s ** 2 + 8 * w0 ** 2 * wf * s + 4 * w0 ** 2 * wf ** 2
    return num / den


def D_dd(s):
    """the same network on a component e^{st} of the positive frame (complex coefficients)."""
    return wf * (s + 2j * w0) / (s ** 2 + 2 * (wf + 1j * w0) * s + 2j * w0 * wf)


def PI_(s, kp, ki):
    return w0 * (kp * s + ki) / s                       # omega in pu times w0


def obs(s):
    """double-integrator observer, continuous poles -w0 and -4 w0 (output phase of the Park PLL)."""
    return (5 * w0 * s + 4 * w0 ** 2) / (s ** 2 + 5 * w0 * s + 4 * w0 ** 2)


def env(G, W):
    """phase transfer of a front end outside the loop: A(W) = (G~(W) + conj G~(-W)) / 2, G~(W) = G(w0 + W) / G(w0)."""
    g0 = G(w0)
    return (G(w0 + W) / g0 + np.conj(G(w0 - W) / g0)) / 2


KP_F, KI_F = 0.3, 3.0                                   # Park PLL with the PI on the filtered eta (variant)
LOOPS = {
    'park': dict(park_loop(1.0, 0.0), lab='Park PLL, 1 pu ($k_p$ 1, $k_i$ 0)'),
    'park02': dict(park_loop(1.0, 0.0, 0.2), lab='Park PLL, 0.2 pu'),
    'park_ki': dict(park_loop(1.0, 20.0), lab='Park PLL $k_i$ 20, 1 pu'),
    'park_ki02': dict(park_loop(1.0, 20.0, 0.2), lab='Park PLL $k_i$ 20, 0.2 pu'),
    'park_f': dict(park_loop(KP_F, KI_F, 1.0, True), lab='Park PLL, PI on filtered $\\eta$ ($k_p$ 0.3, $k_i$ 3), 1 pu'),
    'park_f02': dict(park_loop(KP_F, KI_F, 0.2, True), lab='Park PLL, PI on filtered $\\eta$, 0.2 pu'),
    'fht': dict(L=lambda s: PI_(s, 1.0, 80.0) / s, pre=lambda W: env(G_fht3, W), post=None,
                lab='FHT-PLL ($k_p$ 1, $k_i$ 80)'),
    'ddsrf': dict(L=lambda s: PI_(s, 0.32, 8.0) * D_eff(s) / s, pre=lambda W: env(G_fht, W), post=None,
                  lab='DDSRF-PLL ($k_p$ 0.32, $k_i$ 8)'),
    'ddsrf_F': dict(L=lambda s: PI_(s, 0.32, 8.0) * (wf / (s + wf)) / s, pre=lambda W: env(G_fht, W), post=None,
                    lab='DDSRF-PLL, LPF only in the loop'),
}


def H_overall(k, W):
    d = LOOPS[k]
    s = 1j * W
    L = d['L'](s)
    H = L / (1 + L)
    if d['pre'] is not None:
        H = H * d['pre'](W)
    if d['post'] is not None:
        H = H * d['post'](s)
    return H


def margins(k):
    ws = np.logspace(0, 4, 40000)
    L = np.array([LOOPS[k]['L'](1j * w) for w in ws])
    i = np.where(np.abs(L) < 1)[0][0]
    return ws[i], 180 + deg(np.angle(L[i]))


def poles(k):
    """closed-loop poles from the rational loop (sampled: fit with the known structure)."""
    if k.startswith('park'):
        d = LOOPS[k]
        V, kp, ki = d['V'], d['kp'], d['ki']
        # L = w0 V (kp s + ki)(s + wq) / (s^2 (2 s + wq)), or w0 V (kp s + ki) wq / (s^2 (2 s + wq)) when filtered
        num = pp.polymul([w0 * V * ki, w0 * V * kp], [wq] if d['filt'] else [wq, 1.0])
        den = pp.polymul([0, 0, 1.0], [wq, 2.0])
    elif k == 'fht':
        num, den = np.array([w0 * 80.0, w0 * 1.0]), np.array([0, 0, 1.0])
    else:
        nD = wf * np.array([4 * w0 ** 2 * wf, 4 * w0 ** 2, 2 * wf, 1.0])
        dD = np.array([4 * w0 ** 2 * wf ** 2, 8 * w0 ** 2 * wf, 4 * wf ** 2 + 4 * w0 ** 2, 4 * wf, 1.0])
        if k == 'ddsrf_F':
            nD, dD = np.array([wf]), np.array([wf, 1.0])
        num = pp.polymul([w0 * 8.0, w0 * 0.32], nD)
        den = pp.polymul([0, 0, 1.0], dD)
    r = np.roots(pp.polyadd(den, num)[::-1])
    return r[np.argsort(-r.real)]


say('%-36s %10s %8s %10s %10s %10s  %s' % ('loop', 'wc rad/s', 'PM deg', '|T(j2w0)|', 'H -3dB', 'H peak dB', 'closed-loop poles'))
loop_rows = {}
for k, d in LOOPS.items():
    wc, pm = margins(k)
    ws = np.logspace(0, 3.6, 3000)
    H = np.array([H_overall(k, w) for w in ws])
    bw = ws[np.where(np.abs(H) < 1 / np.sqrt(2))[0][0]]
    L2 = d['L'](2j * w0)
    T2 = abs(L2 / (1 + L2))
    pl = poles(k)
    loop_rows[k] = (wc, pm, T2, bw, 20 * np.log10(np.max(np.abs(H))), pl)
    say('%-36s %10.1f %8.1f %10.3f %10.1f %10.2f  %s' % (d['lab'].replace('$', ''), wc, pm, T2, bw,
        20 * np.log10(np.max(np.abs(H))), np.array2string(np.round(pl[:4], 1), max_line_width=200)))
say('Park PLL frequency step error (ki = 0): dtheta = dw / (w0 kp V) = %.3f deg/Hz at 1 pu, %.3f deg/Hz at 0.2 pu' % (
    deg(2 * np.pi / w0), deg(2 * np.pi / (0.2 * w0))))
for W in (100.0, 200.0, 314.16):
    say('  DDSRF network at W = %5.1f rad/s: LPF %.3f / %.1f deg, D_eff %.3f / %.1f deg' % (
        W, abs(wf / (1j * W + wf)), deg(np.angle(wf / (1j * W + wf))), abs(D_eff(1j * W)), deg(np.angle(D_eff(1j * W)))))
for W in (50.0, 100.0, 157.0, 250.0, 300.0):
    a1, a3 = env(G_fht, W), env(G_fht3, W)
    say('  envelope of the front ends at W = %5.1f rad/s: FHT %.3f / %.1f deg, FHT-PLL stage A %.3f / %.1f deg' % (
        W, abs(a1), deg(np.angle(a1)), abs(a3), deg(np.angle(a3))))

# ---- closed-loop phase response of the replicas: small phase modulation of the input
say('closed loop, replica (small phase modulation) against the model, |H| / angle [deg]:')
eps = 2e-3
tm = np.arange(int(1.6 / ts)) * ts
selm = tm > 0.6
WV = (31.4, 62.8, 100.0, 157.0, 200.0)
meas = {k: [] for k in NAMES + ('park_f',)}
for W in WV:
    u = np.cos(w0 * tm + eps * np.sin(W * tm))
    row = '  W %6.1f rad/s:' % W
    for k, fn in (('park', lambda u: m.park_pll(u, P)), ('fht', lambda u: m.fht_pll(u, P)),
                  ('ddsrf', lambda u: m.ddsrf_pll(u, P, i1_init=1.0)),
                  ('park_f', lambda u: m.park_pll(u, P, kp=KP_F, ki=KI_F, eta_filtered=True))):
        ph = np.unwrap(fn(u)['phase'] - w0 * tm)
        M = np.column_stack([np.cos(W * tm[selm]), np.sin(W * tm[selm]), np.ones(selm.sum()), tm[selm]])
        c, *_ = np.linalg.lstsq(M, ph[selm], rcond=None)
        X = complex(c[1], c[0]) / eps
        meas[k].append(X)
        Hm = H_overall(k, W)
        row += '  %s %.3f/%6.1f (model %.3f/%6.1f)' % (k, abs(X), deg(np.angle(X)), abs(Hm), deg(np.angle(Hm)))
    Hf = H_overall('ddsrf_F', W)
    say(row + '  [DDSRF with the LPF only: %.3f/%6.1f]' % (abs(Hf), deg(np.angle(Hf))))

# ---- figure: loops
fr = np.logspace(0, 3, 500)
ww = 2 * np.pi * fr
fig, ax = plt.subplots(2, 2, figsize=(WID, 4.2), sharex='col')
for k, c, ls in (('park', 'C0', '-'), ('park02', 'C0', ':'), ('park_f', 'C0', '--'), ('fht', 'C1', '-'),
                 ('ddsrf', 'C2', '-'), ('ddsrf_F', 'C2', ':')):
    L = np.array([LOOPS[k]['L'](1j * w) for w in ww])
    ax[0, 0].semilogx(fr, 20 * np.log10(np.abs(L)), color=c, ls=ls, label=LOOPS[k]['lab'])
    ax[1, 0].semilogx(fr, deg(np.unwrap(np.angle(L))), color=c, ls=ls)
    wc, pm = margins(k)
    ax[0, 0].plot(wc / 2 / np.pi, 0, 'o', color=c, ms=3)
for k, c, ls in (('park', 'C0', '-'), ('park02', 'C0', ':'), ('park_f', 'C0', '--'), ('fht', 'C1', '-'),
                 ('ddsrf', 'C2', '-')):
    H = np.array([H_overall(k, w) for w in ww])
    ax[0, 1].semilogx(fr, 20 * np.log10(np.abs(H)), color=c, ls=ls, label=LOOPS[k]['lab'])
    ax[1, 1].semilogx(fr, deg(np.unwrap(np.angle(H))), color=c, ls=ls)
for k in NAMES + ('park_f',):
    X = np.array(meas[k])
    mk = '+' if k == 'park_f' else 'x'
    ax[0, 1].plot(np.array(WV) / 2 / np.pi, 20 * np.log10(np.abs(X)), mk, color=COL.get(k, 'C0'), ms=4)
    ax[1, 1].plot(np.array(WV) / 2 / np.pi, deg(np.angle(X)), mk, color=COL.get(k, 'C0'), ms=4)
ax[0, 0].set_ylim(-30, 50)
ax[0, 0].set_ylabel('$|L(j\\omega)|$ [dB]')
ax[1, 0].set_ylabel('$\\angle L$ [deg]')
ax[1, 0].set_ylim(-200, -60)
ax[0, 0].set_title('open loop (dots: crossover)')
ax[0, 0].legend(fontsize=5.5, loc='upper right')
ax[0, 1].set_ylim(-30, 5)
ax[0, 1].set_ylabel('$|\\theta/\\theta_g|$ [dB]')
ax[1, 1].set_ylabel('angle [deg]')
ax[1, 1].set_ylim(-270, 10)
ax[0, 1].set_title('input phase to PLL phase (x, +: replicas)')
for a in ax[1]:
    a.set_xlabel('frequency [Hz]')
fig.tight_layout()
savefig(fig, 'pll_loops.pdf')

# =====================================================================================================================
# 3. test cases (replicas, or the logs of the model when given)
# =====================================================================================================================
say('\n=== 3. test cases of the model (param.test, prv.test_param)')
logs = None
if len(sys.argv) > 1:
    import scipy.io as sio
    logs = sio.loadmat(sys.argv[1], squeeze_me=True, struct_as_record=False)
    if 'results' in logs:                    # file of run_test_cases: struct array results(k).t, .u, .park.phase, ...
        logs = {'case%d' % int(x.id): np.column_stack([x.t, x.u, x.theta_g, x.omega_g, x.park.phase, x.park.omega,
                                                        x.fht.phase, x.fht.omega, x.ddsrf.phase, x.ddsrf.omega])
                for x in np.atleast_1d(logs['results'])}
    say('logs: %s' % os.path.basename(sys.argv[1]))
RUN = {}
for c in range(1, 7):
    T, name = m.test_param(c)
    u, thg, wg = m.test_signal(t_run, T, w0)
    r = m.run_all(u, P)
    if logs is not None:
        X = logs['case%d' % c]
        dmax = max(np.max(np.abs(m.wrap(r[k]['phase'] - X[:, 4 + 2 * j]))) for j, k in enumerate(NAMES))
        say('  case %d: max |replica - log| = %.1e rad' % (c, dmax))
        u, thg, wg = X[:, 1], X[:, 2], X[:, 3]
        for j, k in enumerate(NAMES):
            r[k]['phase'], r[k]['omega'] = X[:, 4 + 2 * j], X[:, 5 + 2 * j]
    mt = {k: m.metrics(t_run, thg, wg, r[k]['phase'], r[k]['omega'], T) for k in NAMES}
    RUN[c] = dict(T=T, name=name, u=u, thg=thg, wg=wg, r=r, mt=mt)
say('%-6s %-10s %9s %9s %9s %9s %9s %9s %9s %11s' % ('case', 'PLL', 'lock ms', 'pre deg', 'pre pp', 'peak deg', 'settle ms',
                                               'end deg', 'end pp', 'w err pu'))
for c in range(1, 7):
    for k in NAMES:
        x = RUN[c]['mt'][k]
        say('%-6d %-10s %9.1f %9.2f %9.2f %9.2f %9.1f %9.2f %9.2f %11.2e' % (
            c, k, 1e3 * x['lock_time'], x['pre_mean'], x['pre_pp'], x['peak'], 1e3 * x['settle'], x['end_mean'],
            x['end_pp'], x['w_end_mean']))


def fmt(x, f='%.2f', nan='--'):
    return nan if not np.isfinite(x) else f % x


say('LaTeX rows (case & PLL & lock ms & offset before the event & peak & settle ms & end error & end ripple pp):')
for c in range(1, 7):
    for j, k in enumerate(NAMES):
        x = RUN[c]['mt'][k]
        say('  %s & %s & %s & %s & %s & %s & %s & %s \\\\' % (str(c) if j == 0 else '', TITLE[k], fmt(1e3 * x['lock_time'], '%.0f'),
            fmt(x['pre_mean']), fmt(x['peak'], '%.1f'), fmt(1e3 * x['settle'], '%.0f'), fmt(x['end_mean']),
            fmt(x['end_pp'])))

# ---- figure: the six cases
fig, ax = plt.subplots(3, 2, figsize=(WID, 6.0), sharex=True)
for c in range(1, 7):
    a = ax[(c - 1) // 2, (c - 1) % 2]
    for k in NAMES:
        a.plot(t_run, RUN[c]['mt'][k]['err'], color=COL[k], lw=0.7, label=TITLE[k])
    a.set_title('case %d: %s' % (c, RUN[c]['name']), fontsize=6.5)
    a.set_ylim(-25, 25)
    if (c - 1) % 2 == 0:
        a.set_ylabel('phase error [deg]')
ax[0, 0].legend(loc='lower right')
for a in ax[2]:
    a.set_xlabel('t [s]')
fig.tight_layout()
savefig(fig, 'cases_as_built.pdf')

# ---- linear model against the simulation: +30 deg jump of case 2


def step_lin(k, tt):
    """step response of the overall phase transfer, from its frequency response (inverse FFT, 2^16 points)."""
    N = 2 ** 16
    Tw = N * ts
    fgrid = np.fft.rfftfreq(N, ts)
    Wg = 2 * np.pi * fgrid
    Wg[0] = 1e-6
    H = np.array([H_overall(k, w) for w in Wg])
    h = np.fft.irfft(H, N) / ts                         # impulse response
    y = np.cumsum(h) * ts
    return np.interp(tt, np.arange(N) * ts, y)


X2 = RUN[2]
sel = (t_run >= 0.4) & (t_run < 0.55)
tt = t_run[sel] - 0.4
nav = int(round(np.pi / w0 / ts))
fig, ax = plt.subplots(1, 3, figsize=(WID, 2.2), sharey=True)
say('+30 deg jump (case 2): simulation against the linear model')
for a, k in zip(ax, NAMES):
    e = X2['mt'][k]['err'][sel] - X2['mt'][k]['pre_mean']
    y = 30 * (step_lin(k, tt) - 1)
    avg = np.convolve(e, np.ones(nav) / nav, mode='same')
    k10 = tt > 0.012
    say('  %-6s max|model - sim| %.2f deg, max|model - sim averaged over 10 ms| %.2f deg (t > 12 ms), overshoot sim %.2f model %.2f deg' % (
        k, np.max(np.abs(y - e)), np.max(np.abs(y - avg)[k10]), np.max(e), np.max(y)))
    a.plot(1e3 * tt, e, color=COL[k], lw=0.8, label='simulation')
    a.plot(1e3 * tt, y, 'k--', lw=0.8, label='linear model')
    a.set_title(TITLE[k])
    a.set_xlabel('time after the jump [ms]')
ax[0].set_ylabel('phase error [deg]')
ax[0].legend()
fig.tight_layout()
savefig(fig, 'jump_validation.pdf')

# ---- ripple of the detector disturbances (DC offset, harmonics): linear prediction against the simulation
say('steady-state ripple: linear prediction (components of the input through G, the loop and, for the DDSRF, the network)')


def ripple_pred(k, comps, g0):
    """comps: list of (a, h, psi) for a cos(h Phi + psi) with Phi the angle of the fundamental (h = 0: DC a cos psi);
    returns the phase ripple theta~(t) of the PLL k (Park: averaged, through T / Kpd and the observer)."""
    tt = np.arange(0, 0.2, ts / 4)
    th = np.zeros_like(tt)
    G = {'park': G_park, 'fht': G_fht3, 'ddsrf': G_fht}[k]
    G0 = G(w0)
    for a, h, psi in comps:
        for sgn in (+1, -1):
            nu = sgn * h * w0
            if h == 0 and sgn < 0:
                continue
            amp = a * G(nu if h else 1e-9) * (2.0 if h == 0 else 1.0)
            c = amp * np.exp(1j * sgn * psi) * np.exp(-1j * np.angle(G0)) / abs(G0)
            W = nu - w0
            s = 1j * W
            L = LOOPS[k]['L'](s)
            if k == 'park':
                tf = (L / (1 + L)) / Kpd(s) * obs(s)
            elif k == 'fht':
                tf = L / (1 + L)
            else:
                tf = PI_(s, 0.32, 8.0) / s * D_dd(s) / (1 + L)
            th += np.imag(c * tf * np.exp(1j * W * tt))
    return deg(np.ptp(th))


def comps_of(T):
    """components of the test signal relative to the fundamental angle Phi = phi + theta0."""
    out = []
    for h, a, ps in zip(T['harm_order'], T['harm_amp'], T['harm_phase']):
        if a:
            out.append((a, h, ps - np.pi / 2 - h * T['theta0']))      # a sin(h phi + ps) = a cos(h Phi + ps - pi/2 - h theta0)
    if T['dc']:
        out.append((T['dc'], 0, 0.0))
    return out


rip_rows = []
for c in (1, 4, 5):
    T = RUN[c]['T']
    row = '  case %d:' % c
    rr = [c]
    for k in NAMES:
        pr = ripple_pred(k, comps_of(T), None)
        sim = RUN[c]['mt'][k]['end_pp']
        row += '  %s pred %.2f sim %.2f' % (k, pr, sim)
        rr += [pr, sim]
    say(row)
    rip_rows.append(rr)
say('LaTeX rows (case & Park pred & sim & FHT-PLL pred & sim & DDSRF pred & sim):')
for r in rip_rows:
    say('  %d & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\' % tuple(r))

# ---- the sag edges: angle of the front-end vectors and phase error
X6 = RUN[6]
T6, u6 = X6['T'], X6['u']
fq = [m.FHT(P, -1.0)]
ab_f = np.array([fq[0].step(x) for x in u6])
ang_fht = deg(m.wrap(np.arctan2(ab_f[:, 1], ab_f[:, 0]) - X6['thg']))
pk = m.park_pll(u6, P)
ang_park = deg(m.wrap(np.arctan2(pk['beta'], u6) - X6['thg']))
fig, ax = plt.subplots(2, 2, figsize=(WID, 3.6), sharex='col', sharey='row')
for j, (t1, t2) in enumerate(((0.245, 0.30), (0.495, 0.55))):
    s6 = (t_run >= t1) & (t_run < t2)
    ax[0, j].plot(1e3 * t_run[s6], ang_park[s6], color='C0', lw=0.8, label='Park QSG: $\\angle(u + j\\beta) - \\theta_g$')
    ax[0, j].plot(1e3 * t_run[s6], ang_fht[s6], color='C2', lw=0.8, label='FHT: $\\angle(\\alpha + j\\beta) - \\theta_g$')
    for k in NAMES:
        ax[1, j].plot(1e3 * t_run[s6], X6['mt'][k]['err'][s6], color=COL[k], lw=0.8, label=TITLE[k])
    ax[1, j].set_xlabel('t [ms]')
    ax[0, j].set_title('sag start (0.25 s)' if j == 0 else 'sag end (0.5 s)')
ax[0, 0].set_ylabel('angle of the QSG vector [deg]')
ax[1, 0].set_ylabel('PLL phase error [deg]')
ax[0, 0].legend(fontsize=6)
ax[1, 0].legend(fontsize=6)
fig.tight_layout()
savefig(fig, 'sag_edges.pdf')
s6 = (t_run >= 0.25) & (t_run < 0.27)
s7 = (t_run >= 0.5) & (t_run < 0.52)
say('sag edges (case 6): largest angle error of the QSG vector in the first 20 ms: Park %.1f / %.1f deg, FHT %.1f / %.1f deg (start / end)' % (
    np.max(np.abs(ang_park[s6])), np.max(np.abs(ang_park[s7])), np.max(np.abs(ang_fht[s6])), np.max(np.abs(ang_fht[s7]))))
for k in NAMES:
    e = X6['mt'][k]['err']
    say('  %-6s largest PLL error after the sag start %.1f deg, after the sag end %.1f deg' % (
        k, np.max(np.abs(e[(t_run >= 0.25) & (t_run < 0.5)] - X6['mt'][k]['end_mean'])),
        np.max(np.abs(e[t_run >= 0.5] - X6['mt'][k]['end_mean']))))

# =====================================================================================================================
# 4. proposed changes, tested on the replicas
# =====================================================================================================================
say('\n=== 4. variants (replicas)')
sp = lambda k: np.roots([1.0, k, 1.0])                  # observer poles of a SOGI with gain k, in units of w0
Pdc1 = m.fht_variant(P, 'dc', pole_dc=-0.1)
Pdc5 = m.fht_variant(P, 'dc', pole_dc=-0.5)
Ps1 = m.fht_variant(P, 'undamped', poles=sp(1.0))
VAR = [
    ('park', 'as built ($k_i$ 0)', lambda u: m.park_pll(u, P)),
    ('park', '$k_i$ 20', lambda u: m.park_pll(u, P, ki=20.0)),
    ('park', '$k_i$ 20, normalized $\\eta$', lambda u: m.park_pll(u, P, ki=20.0, normalize=True)),
    ('park', '$k_i$ 20, DC estimate $0.02\\,\\omega_0$', lambda u: m.park_pll(u, P, ki=20.0, dc_gain=0.02 * w0)),
    ('park', 'PI on filtered $\\eta$, $k_p$ 0.3, $k_i$ 3', lambda u: m.park_pll(u, P, kp=KP_F, ki=KI_F, eta_filtered=True)),
    ('fht', 'as built', lambda u: m.fht_pll(u, P)),
    ('fht', 'undamped FHT', lambda u: m.fht_pll(u, Pu)),
    ('fht', 'undamped FHT, DC state $-0.1\\,\\omega_0$', lambda u: m.fht_pll(u, Pdc1)),
    ('fht', 'FHT poles of a SOGI $k = 1$', lambda u: m.fht_pll(u, Ps1)),
    ('ddsrf', 'as built', lambda u: m.ddsrf_pll(u, P)),
    ('ddsrf', 'start 1 pu, undamped FHT', lambda u: m.ddsrf_pll(u, Pu, i1_init=1.0)),
    ('ddsrf', 'start 1 pu, DC state $-0.5\\,\\omega_0$', lambda u: m.ddsrf_pll(u, Pdc5, i1_init=1.0)),
    ('ddsrf', 'start 1 pu, FHT poles SOGI $k = 1$', lambda u: m.ddsrf_pll(u, Ps1, i1_init=1.0)),
]
VR = {}
for c in range(1, 7):
    T = RUN[c]['T']
    u, thg, wg = m.test_signal(t_run, T, w0)
    for i, (k, lab, f) in enumerate(VAR):
        o = f(u)
        VR[(i, c)] = m.metrics(t_run, thg, wg, o['phase'], o['omega'], T)
        VR[(i, c)]['o'] = o
say('%-48s %7s %7s %7s | %7s | %7s %7s | %7s | %7s | %7s' % ('variant', 'c1 lock', 'c1 pp', 'c1 pk', 'c2 set', 'c3 end',
    'c3 pk', 'c4 pp', 'c5 pp', 'c6 pk'))
var_rows = []
for i, (k, lab, f) in enumerate(VAR):
    r = [VR[(i, c)] for c in range(1, 7)]
    row = (TITLE[k], lab, 1e3 * r[0]['lock_time'], r[0]['end_pp'], r[0]['peak'], 1e3 * r[1]['settle'],
           r[2]['end_mean'], r[2]['peak'], r[3]['end_pp'], r[4]['end_pp'], r[5]['peak'])
    var_rows.append(row)
    plain = k + ' ' + lab.replace('$', '').replace('\\,', ' ').replace('\\omega_0', 'w0')
    say('%-48s %7.0f %7.2f %7.1f | %7.0f | %7.2f %7.2f | %7.2f | %7.2f | %7.1f' % ((plain,) + row[2:]))
say('LaTeX rows (PLL & variant & lock c1 ms & pp c1 & peak c1 & settle c2 ms & end c3 & peak c3 & pp c4 & pp c5 & peak c6):')
for r in var_rows:
    say('  %s & %s & %.0f & %.2f & %.1f & %.0f & %.2f & %.2f & %.2f & %.2f & %.1f \\\\' % r)

# lock time against the initial phase (no event)
say('lock time [ms] against the initial phase of the input (no event): DDSRF start 0 (clamp 0.5) / 1 pu, Park, FHT-PLL')
lock_rows = []
for th0 in (0, 45, 90, 135, 170, 180):
    T = dict(RUN[6]['T'])
    T.update(theta0=np.radians(th0), t_sag=np.inf, t_rec=np.inf)
    u, thg, wg = m.test_signal(t_run, T, w0)
    res = []
    for f in (lambda u: m.ddsrf_pll(u, P), lambda u: m.ddsrf_pll(u, P, i1_init=1.0), lambda u: m.park_pll(u, P),
              lambda u: m.fht_pll(u, P)):
        e = deg(m.wrap(f(u)['phase'] - thg))
        out = np.where(np.abs(e) >= 5)[0]
        res.append(1e3 * (t_run[out[-1] + 1] if out.size else 0.0))
    lock_rows.append([th0] + res)
    say('  %3d deg: DDSRF %5.0f / %5.0f   Park %4.0f   FHT-PLL %4.0f' % tuple([th0] + res))
say('LaTeX rows (initial phase & DDSRF start 0 & DDSRF start 1 pu & Park & FHT-PLL):')
for r in lock_rows:
    say('  \\ang{%d} & %.0f & %.0f & %.0f & %.0f \\\\' % tuple(r))
# Park loop with ki
for k in ('park_ki', 'park_ki02', 'park_f', 'park_f02'):
    wc, pm, T2, bw, pk_db, pl = loop_rows[k]
    say('  %s: crossover %.1f rad/s, PM %.1f deg, poles %s' % (k, wc, pm, np.round(pl, 1)))

# ---- figure: variants
fig, ax = plt.subplots(2, 2, figsize=(WID, 4.0))
a = ax[0, 0]
T = dict(RUN[6]['T'])
T.update(theta0=np.radians(90.0), t_sag=np.inf, t_rec=np.inf)
u, thg, wg = m.test_signal(t_run, T, w0)
for f, c_, ls, lab in ((lambda u: m.ddsrf_pll(u, P), 'C2', ':', 'DDSRF-PLL as built (start 0)'),
                       (lambda u: m.ddsrf_pll(u, P, i1_init=1.0), 'C2', '-', 'DDSRF-PLL start 1 pu'),
                       (lambda u: m.park_pll(u, P), 'C0', '-', 'Park PLL'), (lambda u: m.fht_pll(u, P), 'C1', '-', 'FHT-PLL')):
    a.plot(1e3 * t_run, deg(m.wrap(f(u)['phase'] - thg)), color=c_, ls=ls, lw=0.8, label=lab)
a.set_xlim(0, 250)
a.set_ylim(-100, 100)
a.set_xlabel('t [ms]')
a.set_ylabel('phase error [deg]')
a.set_title('start-up, input $90^\\circ$ ahead of the PLLs')
a.legend(fontsize=5.5)
SHORT = {0: 'Park, $k_i$ 0', 1: 'Park, $k_i$ 20', 3: 'Park, $k_i$ 20 + DC estimate', 4: 'Park, filtered $\\eta$',
         5: 'FHT-PLL as built', 6: 'FHT-PLL undamped', 7: 'FHT-PLL + DC state', 8: 'FHT-PLL, SOGI $k$ = 1',
         9: 'DDSRF as built', 10: 'DDSRF undamped', 11: 'DDSRF + DC state', 12: 'DDSRF, SOGI $k$ = 1'}
a = ax[0, 1]
for i, c_, ls in ((0, 'C0', ':'), (1, 'C0', '-'), (5, 'C1', ':'), (6, 'C1', '-'), (9, 'C2', ':'), (10, 'C2', '-')):
    a.plot(t_run, VR[(i, 3)]['err'], color=c_, ls=ls, lw=0.8, label=SHORT[i])
a.set_xlim(0.35, 1.0)
a.set_ylim(-1, 7.5)
a.set_xlabel('t [s]')
a.set_title('frequency step to 49 Hz at 0.4 s (case 3)')
a.legend(fontsize=5, ncol=2, loc='upper right', columnspacing=0.8, handlelength=1.6)
a = ax[1, 0]
for i, c_, ls in ((1, 'C0', ':'), (3, 'C0', '-'), (6, 'C1', ':'), (7, 'C1', '-'), (10, 'C2', ':'), (11, 'C2', '-')):
    a.plot(t_run, VR[(i, 4)]['err'] - VR[(i, 4)]['pre_mean'], color=c_, ls=ls, lw=0.8, label=SHORT[i])
a.set_xlim(0.35, 0.7)
a.set_ylim(-2.5, 5)
a.set_xlabel('t [s]')
a.set_ylabel('phase error [deg]')
a.set_title('DC offset of 0.02 pu from 0.4 s (case 4)')
a.legend(fontsize=5, ncol=2, loc='upper right', columnspacing=0.8, handlelength=1.6)
a = ax[1, 1]
for i, c_, ls in ((0, 'C0', ':'), (4, 'C0', '-'), (6, 'C1', ':'), (8, 'C1', '-'), (10, 'C2', ':'), (12, 'C2', '-')):
    a.plot(t_run, VR[(i, 1)]['err'], color=c_, ls=ls, lw=0.8, label=SHORT[i])
a.set_xlim(0.7, 0.8)
a.set_ylim(-2.5, 4)
a.set_xlabel('t [s]')
a.set_title('case 1, 4th and 5th harmonic 10 %: selective variants')
a.legend(fontsize=5, ncol=2, loc='upper right', columnspacing=0.8, handlelength=1.6)
fig.tight_layout()
savefig(fig, 'variants.pdf')

with open(os.path.join(FIG, 'single_phase_pll_numbers.txt'), 'w') as fh:
    fh.write('\n'.join(lines) + '\n')
print('numbers -> figures/single_phase_pll_numbers.txt')
