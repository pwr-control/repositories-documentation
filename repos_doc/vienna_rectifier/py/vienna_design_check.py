"""Design check of vienna_rectifier (pwr-control/modelization-and-control, foundations_of_power_electronics/
vienna_rectifier): linear checks (the same as private/vienna_design.m of the model) and the switched model of
vienna_sim.py (step tc = 0.625 us, the control every 125 us as the model does it).

usage: python3 vienna_design_check.py [out_dir]     prints the results, writes the figures (pdf + png) to out_dir

numpy, scipy, numba, matplotlib.
"""
import math, os, sys
import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vienna_sim as vs

OUT = sys.argv[1] if len(sys.argv) > 1 else '.'
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------------------------------- figure style
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'          # categorical slots 1-3 (validated, fixed order)
INK, INK2, GRID, SHADE = '#0b0b0b', '#52514e', '#e3e2de', '#ecebe7'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'legend.frameon': False,
    'axes.edgecolor': '#b9b8b2', 'axes.linewidth': 0.6, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
    'lines.linewidth': 1.1, 'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 200, 'figure.dpi': 100,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, name + '.png'), bbox_inches='tight')
    plt.close(fig)


# ================================================================================ linear checks
d = vs.Data()
w = d.w; ts = d.ts
udc0 = d.udc_nom
Ku = udc0 / math.sqrt(3)
g = Ku / d.ibez
lines = []


def out(s=''):
    print(s)
    lines.append(s)


# 1. operating point
Z1 = d.R1 + 1j * w * d.L1
Z2 = d.R2 + 1j * w * d.L2
Yc = 1 / (d.Rc + 1 / (1j * w * d.Cf))
P_dc = udc0 ** 2 / d.R_load + udc0 ** 2 / (2 * d.R_meas)
I1 = P_dc / (1.5 * d.E_pk)
for _ in range(60):
    a = 1 + Z2 * Yc; b = Z2 * I1
    qa = abs(a) ** 2; qb = 2 * (np.conj(a) * b).real; qc = abs(b) ** 2 - d.E_pk ** 2
    Ug = (-qb + math.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa)
    vconv = Ug - Z1 * I1
    I1 = I1 * P_dc / (1.5 * vconv.real * I1)
I2 = I1 + Ug * Yc
u_pu = abs(vconv) / Ku
delta = -np.angle(vconv)
out('LINEAR CHECKS (same as private/vienna_design.m)')
out(f'1. operating point: {P_dc/1e3:.1f} kW at {udc0:.0f} V (R = {d.R_load} Ohm); E = {d.E_pk:.1f} V peak (secondary, '
    f'no load); capacitor {Ug:.1f} V ({Ug/d.ubez:.3f} pu); converter current {I1:.1f} A peak ({I1/d.ibez:.3f} pu); '
    f'grid current {abs(I2):.1f} A, power factor {math.cos(np.angle(I2)):.3f}')
out(f'   converter voltage {abs(vconv):.1f} V = {u_pu:.3f} pu of udc/sqrt(3) (margin {100*(1-u_pu):.1f} %), '
    f'delta = {math.degrees(delta):.2f} deg')

# 2. LCL resonance
A = np.array([[-(d.R1 + d.Rc) / d.L1, -1 / d.L1, d.Rc / d.L1], [1 / d.Cf, 0, -1 / d.Cf],
              [d.Rc / d.L2, 1 / d.L2, -(d.Rc + d.R2) / d.L2]])
B = np.array([1 / d.L1, 0, 0])
f_res = math.sqrt((d.L1 + d.L2) / (d.L1 * d.L2 * d.Cf)) / 2 / math.pi
ev = np.linalg.eigvals(A); ev = ev[ev.imag > 1e-6]; ev = ev[np.argmax(abs(ev))]
z_pas = -ev.real / abs(ev)
out(f'2. LCL: L1 {d.L1*1e6:.1f} uH, Cf {d.Cf*1e6:.0f} uF, Rc {d.Rc*1e3:.0f} mOhm, L2 {d.L2*1e6:.1f} uH: resonance '
    f'{f_res:.0f} Hz against fs/6 = {1/(6*ts):.0f} Hz, passive damping {100*z_pas:.1f} %')

# 3. current loop (complex alpha-beta)
M = np.zeros((4, 4)); M[:3, :3] = A * ts; M[:3, 3] = B * ts
E4 = expm(M); Ad = E4[:3, :3]; Bd = E4[:3, 3]
rho = np.exp(1j * w * ts)


def current_cl(kp, ki):
    Acl = np.zeros((5, 5), complex)
    Acl[:3, :3] = Ad; Acl[:3, 3] = -Bd * g * (kp + ki * ts); Acl[:3, 4] = Bd * g * rho
    Acl[3, 0] = 1; Acl[4, 3] = -ki * ts; Acl[4, 4] = rho
    Bcl = np.concatenate([Bd * g * (kp + ki * ts), [0, ki * ts]])
    return Acl, Bcl


def current_open(z, kp=d.kp_i, ki=d.ki_i):
    G = np.array([np.linalg.solve(zz * np.eye(3) - Ad, Bd)[0] * g for zz in z])
    return (kp + ki * ts * z / (z - rho)) * G / z


Acl, Bcl = current_cl(d.kp_i, d.ki_i)
p = np.linalg.eigvals(Acl)
s_ab = np.log(p) / ts
k_lcl = np.argmax(abs(s_ab.imag))
f_lcl = abs(s_ab[k_lcl].imag) / 2 / math.pi; z_lcl = -s_ab[k_lcl].real / abs(s_ab[k_lcl])
s_dq = s_ab - 1j * w
slow = s_dq[abs(s_dq) < 0.5 * 2 * math.pi * f_lcl]
s_slow = slow[np.argmax(slow.real)]


def margins_c(Lf, f0, sgn):
    f = np.logspace(math.log10(1.02 * f0) if sgn > 0 else -1, math.log10(0.4999 / ts), 40000)
    L = Lf(np.exp(1j * 2 * math.pi * sgn * f * ts))
    mag = abs(L); ph = sgn * np.unwrap(np.angle(L)) * 180 / math.pi
    k = np.where((mag[:-1] >= 1) & (mag[1:] < 1))[0]
    fc = pm = gm = None; k0 = 0
    if len(k):
        k0 = k[0]; fc = f[k0]; pm = (ph[k0] + 360) % 360 - 180
    q = np.floor((ph + 180) / 360)
    kk = np.where(q[k0:-1] != q[k0 + 1:])[0]
    if len(kk):
        gm = -20 * math.log10(mag[k0 + kk[0]])
    return fc, pm, gm


fcp, pmp, gmp = margins_c(current_open, 50.0, 1)
fcn, pmn, gmn = margins_c(current_open, 50.0, -1)
kp_lim = next((kp for kp in np.linspace(0.02, 3, 750) if max(abs(np.linalg.eigvals(current_cl(kp, d.ki_i)[0]))) >= 1),
              None)
out(f'3. current loop kp {d.kp_i}, ki {d.ki_i}: pole radius {max(abs(p)):.4f}; positive sequence crossover {fcp:.1f} Hz, '
    f'PM {pmp:.1f} deg, GM {gmp:.1f} dB; negative sequence crossover {fcn:.1f} Hz, PM {pmn:.1f} deg, GM {gmn:.1f} dB')
out(f'   LCL mode with the loop {f_lcl:.0f} Hz, damping {100*z_lcl:.1f} %; slowest pole in dq {s_slow.real:.1f} '
    f'{s_slow.imag:+.1f}j rad/s ({-1e3/s_slow.real:.1f} ms); unstable from kp = {kp_lim:.2f}')

# 4. udc loop
C_ser = d.C1 * d.C2 / (d.C1 + d.C2); C_np = d.C1 + d.C2
K = 1.5 * vconv.real * d.ibez / (C_ser * udc0 * d.udc_nom)
a_load = 2 / C_ser * (1 / d.R_load + 1 / (2 * d.R_meas))
ea = math.exp(-a_load * ts)


def Tdd(z):
    Cr = np.array([1, 0, 0, 0, 0])
    T = []
    for zz in z:
        Tp = Cr @ np.linalg.solve(zz * rho * np.eye(5) - Acl, Bcl)
        Tn = Cr @ np.linalg.solve(np.conj(zz) * rho * np.eye(5) - Acl, Bcl)
        T.append((Tp + np.conj(Tn)) / 2)
    return np.array(T)


def udc_open(z):
    return (d.kp_u + d.ki_u * ts * z / (z - 1)) * Tdd(z) * K * (1 - ea) / a_load / (z - ea) / z


fu = np.logspace(-1, math.log10(0.4999 / ts), 4000)
Lu = udc_open(np.exp(1j * 2 * math.pi * fu * ts))
mag = abs(Lu); ph = np.unwrap(np.angle(Lu)) * 180 / math.pi
k = np.where((mag[:-1] >= 1) & (mag[1:] < 1))[0][0]
kg = np.where((ph[:-1] > -180) & (ph[1:] <= -180))[0]
fc_u, pm_u, gm_u = fu[k], 180 + ph[k], (-20 * math.log10(mag[kg[0]]) if len(kg) else float('nan'))
out(f'4. udc loop kp {d.kp_u}, ki {d.ki_u}: C {C_ser*1e3:.1f} mF, plant {K:.1f} 1/s per pu, load pole {a_load:.1f} '
    f'rad/s: crossover {fc_u:.1f} Hz, PM {pm_u:.1f} deg, GM {gm_u:.1f} dB')

# 5. Vienna constraint and neutral point
th = np.arange(3600) / 3600 * 2 * math.pi
ph3 = np.array([0, -2 * math.pi / 3, 2 * math.pi / 3])[:, None]
m0 = 2 / math.sqrt(3) * u_pu * np.cos(th - delta + ph3)
ii = I1 * np.cos(th + ph3)
m_svm = m0 - 0.5 * (m0.max(0) + m0.min(0))


def vienna_zs(m0, i):
    m = np.zeros_like(m0); ok = np.zeros(m0.shape[1], bool)
    for k in range(m0.shape[1]):
        lo = -1 - m0[:, k].min(); hi = 1 - m0[:, k].max()
        for x in range(3):
            if i[x, k] > 0:
                lo = max(lo, -m0[x, k])
            elif i[x, k] < 0:
                hi = min(hi, -m0[x, k])
        v0 = -0.5 * (m0[:, k].max() + m0[:, k].min())
        if lo <= hi:
            ok[k] = True; v0 = min(max(v0, lo), hi)
        m[:, k] = m0[:, k] + v0
    return m, ok


m_vi, ok_vi = vienna_zs(m0, ii)
mis_svm = 360 * np.mean(m_svm[0] * ii[0] < 0); mis_vi = 360 * np.mean(m_vi[0] * ii[0] < 0)
m_eff = np.where(m_svm * ii >= 0, m_svm, 0.0)
i_np = ((1 - abs(m_eff)) * ii).sum(0)
v_np = np.cumsum(i_np) * (2 * math.pi / w / len(th)) / C_np; v_np -= v_np.mean()
X = np.fft.rfft(i_np) / len(th) * 2
unb_lin = 2 * (v_np.max() - v_np.min())
out(f'5. Vienna constraint: misaligned {mis_svm:.2f} deg per period with the min-max zero sequence, {mis_vi:.2f} deg '
    f'with the zero sequence on the current sign (feasible {100*ok_vi.mean():.0f} %); i_np third harmonic {abs(X[3]):.1f} A, '
    f'vC2 - vC1 ripple {unb_lin:.1f} V p-p')

# 6. PLL
tau = 1 / w
wn = math.sqrt(w * d.pll_ki); zeta = w * d.pll_kp / (2 * wn)
pp = np.roots([tau, 1, w * d.pll_kp, w * d.pll_ki])
fp = np.logspace(-1, 3, 20000)
sp = 1j * 2 * math.pi * fp
Lp = w * (d.pll_kp * sp + d.pll_ki) / sp ** 2 / (tau * sp + 1)
k = np.where((abs(Lp[:-1]) >= 1) & (abs(Lp[1:]) < 1))[0][0]
out(f'6. DDSRF-PLL kp {d.pll_kp}, ki {d.pll_ki}: wn {wn:.1f} rad/s, damping {zeta:.2f} without the filter; with the '
    f'filter crossover {fp[k]:.1f} Hz, PM {180+math.degrees(np.angle(Lp[k])):.1f} deg, poles '
    + ', '.join(f'{q.real:.1f}{q.imag:+.1f}j' for q in pp))

# 7. start. The udc PI of the model (udc_ctrl) is in velocity form with back-calculation: it stores x = out - kp e also
# while its output is clamped, so out(k) = out(k-1) + kp (e(k) - e(k-1)) + ki ts e(k). The reference starts at 0 with udc
# at 1 pu: the output sits at the lower limit with a positive stored state, and turns positive when the error rises
# above -x0/(kp + ki ts) (the integral of the first steps neglected), while udc is still above its reference.
u_diode = math.sqrt(3) * d.E_pk
e0 = 0.0 - d.udc_nom / udc0
out0 = min(max((d.kp_u + d.ki_u * ts) * e0, -d.i_lim), d.i_lim)
x0 = out0 - d.kp_u * e0
e_pos = -x0 / (d.kp_u + d.ki_u * ts)
out(f'7. start: diode level {u_diode:.0f} V ({u_diode/udc0:.3f} pu); R C = {d.R_load*C_ser*1e3:.0f} ms; udc PI with the '
    f'reference from 0: error {e0:.2f} pu, output at the limit {out0:.2f} pu, stored state {x0:+.2f} pu, output positive '
    f'from an error of {e_pos:.2f} pu (udc still above its reference)')

# ================================================================================ switched runs
T = 1.25


def metrics(r):
    t = r['t']; dt = r['dt']
    n = int(round(0.02 / dt))
    perr = np.degrees(r['perr'])
    bad = np.where(np.abs(perr) > 5)[0]
    t_lock = (t[bad.max() + 1] if bad.max() + 1 < len(t) else float('nan')) if len(bad) else 0.0
    thd2, f2, X2 = vs.thd(r['i2'][0], dt)
    thd1, f1, _ = vs.thd(r['i1'][0], dt)
    h = {hh: abs(X2[2 * hh]) / f2 * 100 for hh in (5, 7, 11, 13)}
    unb = r['vC2'][-n:] - r['vC1'][-n:]
    ks = np.where(np.abs(r['udc'] - udc0) > 0.02 * udc0)[0]
    i1 = r['i1'][:, -n:]; m = r['mc'][:, -n:]
    mis = (np.sign(m) * np.sign(i1) < 0) & (np.abs(i1) > 1.0)
    return dict(ipk=r['ipk'].max(), t_ipk=t[np.argmax(r['ipk'])], udc_max=r['udc'].max(),
                udc_min=r['udc'][t < 0.3].min(), t_settle=(t[ks.max()] if len(ks) else 0.0), udc_end=r['udc'][-1],
                t_lock=t_lock, thd2=thd2 * 100, thd1=thd1 * 100, h=h, f2=f2, unb_pp=unb.max() - unb.min(),
                unb_mean=unb.mean(), iq=np.mean(r['iq'][-160:]), mis=mis.mean(1).mean() * 360,
                open=np.mean(r['open'][t > 0.5]) / 3 * 100)


runs = {}
dl = vs.Data(); dl.R_load = 1e4
cases = [
    ('as saved: PLL on the negative sequence', dict(pll_legacy=True, pll_i1_init=0.0)),
    ('fixed PLL, integrator from 0', dict(pll_i1_init=0.0)),
    ('model as fixed (init_model.m)', dict()),
    ('  no dead time', dict(dead_time=False)),
    ('  zero sequence on the current sign', dict(vienna_cm=2)),
    ('  load connected at 0.2 s', dict(d=dl, R_load_step_t=0.2, R_load_step=5.0)),
    ('  load at 0.2 s, feed-forward, ref. 1 pu', dict(d=dl, R_load_step_t=0.2, R_load_step=5.0, v_ff=1.0, ref_init=1.0)),
    # start-up options, each from the model as fixed
    ('  udc PI: lower limit 0, cond. integration', dict(udc_aw=1, i_ref_min=0.0)),
    ('  udc PI: lower limit 0 only', dict(i_ref_min=0.0)),
    ('  udc PI: cond. integration only', dict(udc_aw=1)),
    ('  udc reference from 1 pu', dict(ref_init=1.0)),
    ('  feed-forward, reference from 0', dict(v_ff=1.0)),
    ('  feed-forward, reference from 1 pu', dict(v_ff=1.0, ref_init=1.0)),
    ('  feed-forward, lower limit 0, cond. int.', dict(v_ff=1.0, udc_aw=1, i_ref_min=0.0)),
    ('  DC link precharged to 980 V', dict(udc_init=980.0)),
]
out()
out(f'SWITCHED MODEL (vienna_sim.py, {T} s, step {d.tc*1e6:.3f} us)')
out(f'{"case":44s} {"ipk/A":>6s} {"at/ms":>6s} {"udc max":>7s} {"min":>5s} {"2% at":>6s} {"lock":>5s} {"THD ig":>6s} '
    f'{"is":>5s} {"h5":>5s} {"h7":>5s} {"h11":>5s} {"h13":>5s} {"unb pp":>6s} {"mis":>5s}')
for name, kw in cases:
    r = vs.run(T_end=T, log_from=T - 0.04, **kw)
    mm = metrics(r)
    runs[name] = (r, mm)
    lock = f'{mm["t_lock"]*1e3:5.0f}' if np.isfinite(mm['t_lock']) else '   no'
    out(f'{name:44s} {mm["ipk"]:6.0f} {mm["t_ipk"]*1e3:6.1f} {mm["udc_max"]:7.0f} {mm["udc_min"]:5.0f} '
        f'{mm["t_settle"]*1e3:6.0f} {lock} {mm["thd2"]:5.2f}% {mm["thd1"]:4.2f}% {mm["h"][5]:5.2f} {mm["h"][7]:5.2f} '
        f'{mm["h"][11]:5.2f} {mm["h"][13]:5.2f} {mm["unb_pp"]:6.2f} {mm["mis"]:5.1f}')
out('   ipk: largest converter current (any phase) and when; udc min over the first 0.3 s; 2% at: last time udc is out of '
    '1070 V +-2 %; lock: PLL error below 5 deg from then on; THD ig / is: grid / converter current, last two periods; '
    'h5 ... h13: grid current harmonics in % of the fundamental; unb pp: ripple of vC2 - vC1 [V]; mis: degrees per '
    'period where the pole reference and the current have opposite signs (phase average)')
for key in ('  load connected at 0.2 s', '  load at 0.2 s, feed-forward, ref. 1 pu'):
    rr = runs[key][0]
    tt0 = rr['t']; sel0 = tt0 < 0.2; sel1 = (tt0 >= 0.1) & (tt0 < 0.2)
    uu = rr['udc'][sel1]; Cs = d.C1 * d.C2 / (d.C1 + d.C2)
    p_in = (0.5 * Cs * (uu[-1] ** 2 - uu[0] ** 2) / (tt0[sel1][-1] - tt0[sel1][0]) + np.mean(uu ** 2) / 1e4
            + np.mean(uu ** 2) / (2 * d.R_meas))
    out(f'   {key.strip()}, before the load step: largest current {rr["ipk"][sel0].max():.1f} A, udc max '
        f'{rr["udc"][sel0].max():.0f} V, power into the DC link over 0.1 ... 0.2 s {p_in / 1e3:.2f} kW')
r0, m0r = runs['model as fixed (init_model.m)']
out(f'   model as fixed, steady state: iq {m0r["iq"]:+.3f} pu, unbalance mean {m0r["unb_mean"]:+.2f} V, open phases '
    f'{m0r["open"]:.2f} % of the time, grid current fundamental {m0r["f2"]:.1f} A peak')
open(os.path.join(OUT, 'vienna_design_check_output.txt'), 'w').write('\n'.join(lines) + '\n')

# ================================================================================ figures
# F1 start-up: the model as saved, the fixed PLL from 0, the model as fixed
sel = [('model as fixed (init_model.m)', 'model as fixed', C1), ('fixed PLL, integrator from 0', 'PLL integrator from 0', C2),
       ('as saved: PLL on the negative sequence', 'as saved (PLL on the negative sequence)', C3)]
fig, ax = plt.subplots(3, 1, figsize=(6.3, 5.6), sharex=True)
for key, lab, col in sel:
    r = runs[key][0]
    ax[0].plot(r['t'] * 1e3, r['udc'], color=col, label=lab)
    ax[1].plot(r['t'] * 1e3, r['ipk'], color=col, label=lab)
    ax[2].plot(r['t'] * 1e3, np.degrees(r['perr']), color=col, label=lab)
r = runs['model as fixed (init_model.m)'][0]
ax[0].plot(r['t'] * 1e3, r['ref'] * udc0, color=INK2, lw=0.8, label='udc reference')
ax[0].set_ylabel('udc [V]'); ax[0].set_ylim(0, 2000)
ax[0].legend(loc='lower right', ncol=2)
ax[1].set_ylabel('peak converter current [A]')
ax[2].set_ylabel('PLL phase error [deg]'); ax[2].set_ylim(-180, 180); ax[2].set_yticks([-180, -90, 0, 90, 180])
ax[2].set_xlabel('time [ms]'); ax[2].set_xlim(0, 400)
ax[0].set_title('Start-up: DC-link voltage, peak converter current per control period, PLL error')
fig.tight_layout(); save(fig, 'fig_startup')

# F2 start-up options
sel = [('model as fixed (init_model.m)', 'model as fixed', C1),
       ('  udc PI: lower limit 0, cond. integration', 'udc PI: lower limit 0, conditional integration', C2),
       ('  feed-forward, reference from 1 pu', 'capacitor-voltage feed-forward + udc reference from 1 pu', C3)]
fig, ax = plt.subplots(2, 1, figsize=(6.3, 3.9), sharex=True)
for key, lab, col in sel:
    r = runs[key][0]
    ax[0].plot(r['t'] * 1e3, r['udc'], color=col, label=lab)
    ax[1].plot(r['t'] * 1e3, r['ipk'], color=col, label=lab)
ax[0].set_ylabel('udc [V]'); ax[0].legend(loc='lower right')
ax[1].set_ylabel('peak converter current [A]'); ax[1].set_xlabel('time [ms]'); ax[1].set_xlim(0, 500)
ax[0].set_title('Start-up options (switched model)')
fig.tight_layout(); save(fig, 'fig_startup_options')

# F2b start-up: output of the udc PI and active current over the first 60 ms
fig, ax = plt.subplots(2, 1, figsize=(6.3, 4.2), sharex=True)
for key, lab, col in (('model as fixed (init_model.m)', 'model as fixed', C1),
                      ('  udc PI: lower limit 0, cond. integration', 'lower limit 0, cond. integration', C2)):
    r = runs[key][0]
    tm = r['t'] * 1e3
    ax[0].plot(tm, r['udc'], color=col, label=lab)
    ax[1].plot(tm, r['id'] * d.ibez, color=col, label=lab + ': active current')
    ax[1].plot(tm, r['iref'] * d.ibez, color=col, ls=(0, (4, 2)), lw=1.0, label=lab + ': reference')
r = runs['model as fixed (init_model.m)'][0]
ax[0].plot(r['t'] * 1e3, r['ref'] * udc0, color=INK2, lw=0.8, label='udc reference (both)')
ax[0].set_ylabel('udc [V]'); ax[0].set_ylim(0, 1300); ax[0].legend(loc='lower right')
ax[1].axhline(0.0, color=INK2, lw=0.6)
ax[1].set_ylabel('active current [A, peak]'); ax[1].set_xlabel('time [ms]'); ax[1].set_xlim(0, 60)
ax[1].set_ylim(-650, 900); ax[1].legend(loc='lower right', ncol=2, fontsize=7)
ax[0].set_title('Start-up: DC voltage, output of the DC-voltage PI and active current (switched model)')
fig.tight_layout(); save(fig, 'fig_startup_pi')

# F3 steady state: grid currents and converter current near a zero crossing
r, _ = runs['model as fixed (init_model.m)']
rv, _ = runs['  zero sequence on the current sign']
dt = r['dt']; n = r['i2'].shape[1]; tt = (np.arange(n) * dt + r['log_from']) * 1e3
fig, ax = plt.subplots(2, 1, figsize=(6.3, 4.4))
for x, col, nm in ((0, C1, 'u'), (1, C2, 'v'), (2, C3, 'w')):
    ax[0].plot(tt, r['i2'][x], color=col, lw=0.9, label='grid current ' + nm)
ax[0].set_ylabel('[A]'); ax[0].set_xlabel('time [ms]'); ax[0].set_ylim(-320, 420)
ax[0].legend(loc='upper center', ncol=3)
ax[0].set_title('Grid currents in steady state (model as fixed)')
# converter current of phase u minus its fundamental, over one period from its positive-going zero crossing
def deviation(rr):
    nw = int(round(2 * ts / dt))                                    # one carrier period
    x = np.convolve(np.concatenate([rr['i1'][0], rr['i1'][0][:nw]]), np.ones(nw) / nw, mode='valid')[:rr['i1'].shape[1]]
    X = np.fft.rfft(x) / len(x) * 2
    k = np.arange(len(x))
    fund = abs(X[2]) * np.cos(2 * np.pi * 2 * k / len(x) + np.angle(X[2]))
    ang = (2 * np.pi * 2 * k / len(x) + np.angle(X[2]) + np.pi / 2) % (2 * np.pi)   # 0 at the zero crossing to +
    o = np.argsort(ang[:len(x) // 2])
    return ang[:len(x) // 2][o] * 180 / np.pi, (x - fund)[:len(x) // 2][o]


a_m, e_m = deviation(r)
a_v, e_v = deviation(rv)
ax[1].plot(a_m, e_m, color=C1, label='min-max zero sequence (model)')
ax[1].plot(a_v, e_v, color=C2, label='zero sequence on the current sign (option)')
ax[1].set_xlim(0, 360); ax[1].set_xticks(range(0, 361, 45))
ax[1].set_xlabel('angle of the fundamental [deg] (0: zero crossing of the current to positive)'); ax[1].set_ylabel('[A]')
ax[1].set_title('Converter current of phase u minus its fundamental, averaged over a carrier period')
ax[1].legend(loc='upper right', ncol=2)
fig.tight_layout(); save(fig, 'fig_steady')

# F4 spectrum of the grid current
_, X0 = vs.thd(r['i2'][0], dt)[1:]
f0a = vs.thd(r['i2'][0], dt)[1]
f0v, Xv = vs.thd(rv['i2'][0], dt)[1:]
hh = np.arange(2, 41)
a0 = np.array([abs(X0[2 * h]) for h in hh]) / f0a * 100
av = np.array([abs(Xv[2 * h]) for h in hh]) / f0v * 100
fig, ax = plt.subplots(figsize=(6.3, 2.7))
wb = 0.38
ax.bar(hh - wb / 2 - 0.02, a0, wb, color=C1, label=f'min-max zero sequence (model), THD {runs["model as fixed (init_model.m)"][1]["thd2"]:.2f} %')
ax.bar(hh + wb / 2 + 0.02, av, wb, color=C2, label=f'zero sequence on the current sign, THD {runs["  zero sequence on the current sign"][1]["thd2"]:.2f} %')
ax.set_xlabel('harmonic order'); ax.set_ylabel('% of the fundamental'); ax.set_xlim(1.3, 40.7)
ax.set_xticks([5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 35, 37])
ax.grid(axis='x', visible=False)
ax.legend(loc='upper right'); ax.set_title('Harmonics of the grid current (switched model, steady state)')
fig.tight_layout(); save(fig, 'fig_spectrum')

# F5 neutral point
unb = r['vC2'] - r['vC1']
fig, ax = plt.subplots(figsize=(6.3, 2.5))
ax.plot(tt, unb - unb.mean(), color=C1, lw=0.8, label='switched model')
Xi = np.fft.rfft(r['i1'][0]) / n * 2
ang = (w * (tt / 1e3 - r['log_from']) + np.angle(Xi[2])) % (2 * math.pi)      # angle of the current of phase u
ax.plot(tt, np.interp(ang, th, 2 * v_np, period=2 * math.pi), color=C2, lw=1.4,
        label='linear estimate: 2 x integral of (1 - |m|) i over C1 + C2')
ax.set_xlabel('time [ms]'); ax.set_ylabel('vC2 - vC1 - mean [V]'); ax.set_ylim(-3.6, 4.6)
ax.legend(loc='upper right', ncol=2); ax.set_title('Neutral-point ripple in steady state')
fig.tight_layout(); save(fig, 'fig_neutral_point')

# F6 loops
ff = np.logspace(0, math.log10(0.4999 / ts), 1500)
Lpos = current_open(np.exp(1j * 2 * math.pi * ff * ts))
Lneg = current_open(np.exp(-1j * 2 * math.pi * ff * ts))
Luu = udc_open(np.exp(1j * 2 * math.pi * ff * ts))
fig, ax = plt.subplots(2, 1, figsize=(6.3, 4.4), sharex=True)
for L, col, lab, sg in ((Lpos, C1, 'current loop, positive sequence', 1), (Lneg, C2, 'current loop, negative sequence', -1),
                        (Luu, C3, 'DC-voltage loop', 1)):
    ax[0].semilogx(ff, 20 * np.log10(abs(L)), color=col, label=lab)
    phs = sg * np.unwrap(np.angle(L)) * 180 / math.pi
    ax[1].semilogx(ff, phs, color=col, label=lab)
ax[0].axhline(0, color=INK2, lw=0.7)
ax[1].axhline(-180, color=INK2, lw=0.7)
ax[0].axvline(f_res, color=INK2, lw=0.7); ax[0].text(f_res * 1.05, 25, 'LCL resonance', color=INK2, fontsize=7)
ax[0].set_ylabel('|L| [dB]'); ax[0].set_ylim(-60, 40); ax[0].legend(loc='lower left')
ax[1].set_ylabel('phase of L [deg]'); ax[1].set_xlabel('frequency [Hz] (negative sequence: |f|)')
ax[1].set_ylim(-540, 90); ax[1].set_yticks([-540, -360, -180, 0])
ax[0].set_title('Open loops (one control period of delay, zero-order hold)')
fig.tight_layout(); save(fig, 'fig_loops')

# F7 Vienna window over one period
deg = th * 180 / math.pi
fig, ax = plt.subplots(figsize=(6.3, 2.8))
mis_band = (m_svm[0] * ii[0] < 0)
y0, y1 = -1.15, 1.15
ax.fill_between(deg, y0, y1, where=mis_band, color=SHADE, step='mid', label='pole reference and current of opposite sign')
ax.plot(deg, m_svm[0], color=C1, label='pole reference, min-max zero sequence')
ax.plot(deg, m_vi[0], color=C2, label='pole reference, zero sequence on the current sign')
ax.plot(deg, ii[0] / d.ibez, color=C3, label='converter current [pu]')
ax.set_xlim(0, 360); ax.set_ylim(y0, y1); ax.set_xticks(range(0, 361, 45))
ax.set_xlabel('angle of the converter current of phase u [deg]'); ax.set_ylabel('[pu]')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
ax.set_title(f'Phase u over one period at the operating point (|u| = {u_pu:.3f} pu, delta = {math.degrees(delta):.1f} deg)')
fig.tight_layout(); save(fig, 'fig_vienna_window')
print('figures written to', OUT)
