"""Linear analysis of the grid-forming controller of grid_forming_dqctrl (afe_dq_ctrl_pwm) / grid_forming_dq.slx.

Companion script of grid_forming_control.tex: reproduces the numbers quoted in the note and draws its
figures (python3, numpy, scipy, matplotlib).

    python3 grid_forming_control_analysis.py [figures_dir]

Model: complex space vectors in the positive synchronous frame (x = x_xi + j x_eta); the plant is
balanced, so every loop is a complex-coefficient SISO system and a negative-sequence component at the
fundamental appears at the frequency -2 f0. Discrete time with the control period ts = 1/(2 fpwm),
exact (zero-order hold) discretization of the plant, one-step acquisition delay, controller equations
written exactly as in afe_dq_ctrl_pwm.m (PI with the updated state in the output, resonant term in the
semi-implicit form with a = 2 sin(wr ts/2), FHT with the second-order Taylor matrix of the C code).
Plant data: study model grid_forming_dq.slx (400 V, 250 kW, 50 Hz, 4 kHz double update), see
foundations_of_power_electronics/grid_forming/single_phase_load/init_model.m of modelization-and-control.
"""
import os
import sys
import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 9, 'axes.grid': True,
                     'grid.alpha': 0.35, 'legend.fontsize': 7.5, 'axes.titlesize': 9, 'figure.dpi': 100})
FIGW = 6.3     # [in] text width of the note


def pr(*a):
    print(*a)


# =============================================================================================== 1. data
pr('=' * 100)
pr('1. STUDY MODEL DATA (grid_forming_dq.slx, 400 V / 250 kW / 50 Hz / 4 kHz double update)')
us_nom, pwr_nom, f0 = 400.0, 250e3, 50.0
w0 = 2*np.pi*f0
is_nom = pwr_nom/np.sqrt(3)/us_nom/0.9                 # [A] rms, cos(phi) = 0.9 (afe_hw_param)
ubez = us_nom*np.sqrt(2/3)                             # [V] phase peak
ibez = is_nom*np.sqrt(2)                               # [A] phase peak
zbez = ubez/ibez
udc = 660.0
k_mod = udc/np.sqrt(3)/ubez                            # converter voltage per unit of modulation index
fpwm = 4e3
ts = 1/(2*fpwm)
fs = 1/ts
L1, R1 = 330e-6, 157*0.05*330e-6                       # converter-side reactor
CF, RC = 360e-6, 50e-3                                 # filter capacitor with its series resistance
L2, R2 = 80e-6, 157*0.05*80e-6                         # grid-side reactor
# transformer Dyn11 1 MVA 400/400 V, ucc 5 %, eta 95 %: leakage and copper resistance, primary star equivalent
i1n = 1e6/np.sqrt(3)/400
Lsig = 400*0.05/np.sqrt(3)/i1n/w0
Rsig = 0.05*1e6/3/i1n**2
# RL load 230^2*3/275e3*[0.9 1.1 0.95] ohm and 1 mH per phase on the secondary star, mean value referred to the
# primary star equivalent (n1 = 24 delta turns, n2 = 14 star turns)
kratio = (24/np.sqrt(3))/14
Rload = np.mean(230**2*3/275e3*np.array([0.9, 1.1, 0.95]))*kratio**2
Lload = 1e-3*kratio**2
Lg, Rg = L2 + Lsig + Lload, R2 + Rsig + Rload          # everything beyond the capacitor node
# per unit (inductances and capacitances in seconds)
L1p, R1p = L1/zbez, R1/zbez
Cp, RCp = CF*zbez, RC/zbez
L2p, R2p = L2/zbez, R2/zbez
Lgp, Rgp = Lg/zbez, Rg/zbez
f_res = 1/(2*np.pi*np.sqrt(L1*L2*CF/(L1 + L2)))
f_lc1 = 1/(2*np.pi*np.sqrt(L1*CF))
pr(f'  ubez = {ubez:.1f} V, ibez = {ibez:.1f} A, zbez = {zbez:.4f} ohm, k_mod = udc/sqrt(3)/ubez = {k_mod:.4f}')
pr(f'  ts = {ts*1e6:.0f} us, fs = {fs:.0f} Hz, fs/6 = {fs/6:.0f} Hz; LCL resonance (stiff grid) {f_res:.0f} Hz, L1-CF resonance {f_lc1:.0f} Hz')
pr(f'  pu plant: L1 = {L1p*1e3:.4f} ms, R1 = {R1p:.4f}, CF = {Cp*1e3:.4f} ms (b = {w0*Cp:.4f} pu), RC = {RCp:.4f}, '
   f'Lg = {Lgp*1e3:.4f} ms, Rg = {Rgp:.4f} (load R = {Rload/zbez:.3f} pu)')

# controller of the study model
kp_i, ki_i = 0.75, 35.0
kp_u, ki_u = 0.5, 18.0
kp_n, ki_n = kp_u, ki_u
kr = 2*ki_i
wr = 2*w0
delta_fht = 0.05

# =============================================================================================== 2. plant
# stationary frame, complex space vectors, pu: x = [i1, uc, i2], inputs [m, i_dist]
A = np.array([[-(R1p + RCp)/L1p, -1/L1p, RCp/L1p],
              [1/Cp, 0.0, -1/Cp],
              [RCp/Lgp, 1/Lgp, -(RCp + Rgp)/Lgp]])
B = np.array([[k_mod/L1p, 0.0], [0.0, -1/Cp], [0.0, 0.0]])
C_i1 = np.array([1.0, 0.0, 0.0])
C_ug = np.array([(1 - L2p/Lgp)*RCp, (1 - L2p/Lgp), -(1 - L2p/Lgp)*RCp + (L2p/Lgp)*Rgp - R2p])   # terminal voltage after L2
Adq = A - 1j*w0*np.eye(3)
Mz = expm(np.block([[Adq, B], [np.zeros((2, 5))]])*ts)
Ad, Bd = Mz[:3, :3], Mz[:3, 3:]


def P_dq_s(s):
    """plant m -> i1 in the positive frame, continuous time (no delay)"""
    return C_i1 @ np.linalg.solve(s*np.eye(3) - Adq, B[:, 0])


def P_dq_z(z):
    """plant m -> i1, discrete (ZOH), without the computation delay"""
    return C_i1 @ np.linalg.solve(z*np.eye(3) - Ad, Bd[:, 0])


# =============================================================================================== 3. current loop
pr('\n' + '=' * 100)
pr('3. CURRENT LOOP')
wc = kp_i*k_mod/L1p
pm = 90 - np.degrees(1.5*ts*wc) - np.degrees(np.arctan(ki_i/(kp_i*wc)))
pr(f'  simplified design (L1 only, delay 1.5 ts): crossover {wc:.0f} rad/s = {wc/2/np.pi:.0f} Hz, '
   f'PI zero {ki_i/kp_i:.1f} rad/s = {ki_i/kp_i/2/np.pi:.1f} Hz, delay phase {np.degrees(1.5*ts*wc):.1f} deg, phase margin {pm:.0f} deg')
P0 = P_dq_s(0.0)
pr(f'  loaded plant m -> i1 at DC (positive sequence): |P(0)| = {abs(P0):.3f}, kp*P = {abs(kp_i*P0):.3f}, '
   f'T_P(0) = kp P/(1 + kp P) = {abs(kp_i*P0/(1 + kp_i*P0)):.3f}')
for ff in (-2*f0, 2*f0):
    Pz = P_dq_s(2j*np.pi*ff)
    pr(f'  loaded plant at {ff:+.0f} Hz (dq): |P| = {abs(Pz):.3f} /_ {np.degrees(np.angle(Pz)):+.1f} deg')


def resonator(wr_, kr_, ts_, method, delta=0.0):
    """discrete resonant term kr s/(s^2 + 2 delta wr s + wr^2): x(k+1) = Ar x + Br e, r = Cr x + Dr e"""
    Ac = np.array([[0.0, 1.0], [-wr_**2, -2*delta*wr_]])
    Bc = np.array([[0.0], [1.0]])
    Cc = np.array([[0.0, kr_]])
    if method == 'code':                      # afe_dq_ctrl_pwm: semi-implicit Euler, a = 2 sin(wr ts/2), undamped
        a = 2*np.sin(wr_*ts_/2)
        Ar = np.array([[1.0, -a], [a, 1 - a*a]])
        Br = np.array([[ts_*kr_], [a*ts_*kr_]])
        Cr = np.array([[1.0, -a]])
        Dr = ts_*kr_
    elif method == 'zoh':                     # resonant_pi_param of the study model: exact, output from the delayed state
        Mm = expm(np.block([[Ac, Bc], [np.zeros((1, 3))]])*ts_)
        Ar, Br, Cr, Dr = Mm[:2, :2], Mm[:2, 2:], Cc, 0.0
    elif method == 'fwd':                     # library ctrl_afe_setup.resonant_pi: forward Euler
        Ar, Br, Cr, Dr = np.eye(2) + Ac*ts_, Bc*ts_, Cc, 0.0
    elif method == 'none':
        Ar, Br, Cr, Dr = np.zeros((2, 2)), np.zeros((2, 1)), np.zeros((1, 2)), 0.0
    else:
        raise ValueError(method)
    return Ar, Br, Cr, Dr


def R_z(z, res):
    Ar, Br, Cr, Dr = res
    return (Cr @ np.linalg.solve(z*np.eye(2) - Ar, Br))[0, 0] + Dr


def C_pi_z(z, kp, ki):
    """PI as coded: x(k+1) = x(k) + ki ts e(k), m(k) = kp e(k) + x(k+1)"""
    return kp + ki*ts*z/(z - 1)


def L_current_z(z, res, kp=kp_i, ki=ki_i):
    return (C_pi_z(z, kp, ki) + R_z(z, res))*P_dq_z(z)/z


def current_loop_ss(res, kp=kp_i, ki=ki_i):
    """closed current loop, states [x(3), d_i, x_I, x_r(2)], inputs [i_ref, i_dist], output i1"""
    Ar, Br, Cr, Dr = res
    n = 7
    I = np.eye(n, dtype=complex)
    e_x, e_u = -I[3], np.array([1.0, 0.0])                       # e = i_ref - d_i
    r_x, r_u = Cr[0, 0]*I[5] + Cr[0, 1]*I[6] + Dr*e_x, Dr*e_u
    m_x, m_u = (kp + ki*ts)*e_x + I[4] + r_x, (kp + ki*ts)*e_u + r_u
    Acl = np.zeros((n, n), complex)
    Bcl = np.zeros((n, 2), complex)
    Acl[:3, :3] = Ad
    Acl[:3, :] += np.outer(Bd[:, 0], m_x)
    Bcl[:3, :] += np.outer(Bd[:, 0], m_u)
    Bcl[:3, 1] += Bd[:, 1]
    Acl[3, :3] = C_i1
    Acl[4, :] = I[4] + ki*ts*e_x
    Bcl[4, :] = ki*ts*e_u
    Acl[5:7, :] = Ar @ np.vstack([I[5], I[6]]) + np.outer(Br[:, 0], e_x)
    Bcl[5:7, :] = np.outer(Br[:, 0], e_u)
    Ccl = I[0].copy()
    return Acl, Bcl, Ccl


def modes(Acl):
    ev = np.linalg.eigvals(Acl)
    s = np.log(ev)/ts
    order = np.argsort(-np.abs(ev))
    return ev[order], s[order]


def fmt_modes(s, n=4):
    return ', '.join(f'({sk.imag/2/np.pi:+.0f} Hz, {sk.real:.1f} 1/s)' for sk in s[:n])


def margins(Lfun, nf=24001):
    f = np.linspace(-fs/2*0.999, fs/2*0.999, nf) + 0.0123
    z = np.exp(1j*2*np.pi*f*ts)
    L = np.array([Lfun(zz) for zz in z])
    return f, L, np.min(np.abs(1 + L))


def simulate(Acl, Bcl, u, x0=None):
    n = Acl.shape[0]
    x = np.zeros(n, complex) if x0 is None else x0.copy()
    X = np.zeros((len(u), n), complex)
    for k in range(len(u)):
        X[k] = x
        x = Acl @ x + Bcl @ u[k]
    return X


def settle(err, ref_amp, tol=0.02):
    env = np.abs(err)/ref_amp
    idx = np.where(env > tol)[0]
    return (idx[-1] + 1)*ts if len(idx) else 0.0


pr('\n  resonant term: design rule of the pole displacement near z_r = exp(+-j wr ts):')
pr('     s - (+-j wr) = -rho S(z_r) P(z_r)/(z_r^2 ts), rho = residue of R(z) at z_r (~ kr z_r ts/2), S = 1/(1 + C_PI P/z)')
cases = [('PI only', 'none', 0.0, 0.0),
         ('PI + R code kr 35', 'code', 35.0, 0.0),
         ('PI + R code kr 70', 'code', 70.0, 0.0),
         ('PI + R code kr 140', 'code', 140.0, 0.0),
         ('PI + R code kr 210', 'code', 210.0, 0.0),
         ('PI + R zoh kr 70 (model)', 'zoh', 70.0, 0.0),
         ('PI + R fwd Euler kr 70 (library form)', 'fwd', 70.0, 0.0),
         ('PI + R fwd Euler kr 45 delta 0.015 (library values)', 'fwd', 45.0, 0.015)]
Tsim = 1.2
N = int(Tsim/ts)
t = np.arange(N)*ts
res_cl = {}
for name, meth, kr_, dl in cases:
    res = resonator(wr, kr_, ts, meth, dl)
    Acl, Bcl, Ccl = current_loop_ss(res)
    ev, s = modes(Acl)
    f, L, s_m = margins(lambda z: L_current_z(z, res))
    evr = np.linalg.eigvals(res[0]) if meth != 'none' else np.array([0.0])
    # tracking of a negative-sequence reference 0.2 pu at -2 f0 (error seen by the controller)
    u = np.zeros((N, 2), complex)
    u[:, 0] = 0.2*np.exp(-1j*wr*t)
    X = simulate(Acl, Bcl, u)
    err = u[:, 0] - X[:, 3]
    tset = settle(err, 0.2)
    T_neg = (Ccl @ np.linalg.solve(np.exp(-1j*wr*ts)*np.eye(7) - Acl, Bcl[:, 0]))
    pr(f'\n  [{name}]  resonator |eig| = {np.abs(evr).max():.6f}')
    pr(f'    modulus margin min|1 + L| = {s_m:.3f}; closed-loop max|z| = {np.abs(ev).max():.5f}; least damped modes: {fmt_modes(s)}')
    pr(f'    closed loop i1/i_ref at -2 f0: {abs(T_neg):.3f} /_ {np.degrees(np.angle(T_neg)):+.1f} deg; '
       f'negative-sequence reference: 2 % settling {tset*1e3:.0f} ms, max |err|/|ref| {np.abs(err).max()/0.2:.2f}')
    if meth in ('code', 'zoh'):
        for sg in (-1, +1):
            zr = np.exp(1j*sg*wr*ts)
            eps = 1e-7
            rho = (zr*eps)*R_z(zr + zr*eps, res)          # residue of R(z) at z_r
            Pz = P_dq_z(zr)
            S = 1/(1 + C_pi_z(zr, kp_i, ki_i)*Pz/zr)
            ds = -rho*S*Pz/(zr**2*ts)
            k_near = np.argmin(np.abs(s - 1j*sg*wr))
            pr(f'    mode at {sg*2*f0:+.0f} Hz: rule sigma = {ds.real:.1f} 1/s (rho/(kr ts/2) = {abs(rho/(kr_*ts/2)):.3f}), '
               f'eigenvalue sigma = {s[k_near].real:.1f} 1/s at {s[k_near].imag/2/np.pi:+.1f} Hz; |T_PI(z_r)|/|C_PI| = {abs(S*Pz/zr):.3f}')
    res_cl[name] = dict(f=f, L=L, t=t, err=np.abs(err)/0.2)

# ---- figure: open-loop frequency response of the current loop (dq, negative frequencies = negative sequence)
fig, ax = plt.subplots(2, 1, figsize=(FIGW, 4.6), sharex=True)
for name, lab, st in [('PI only', 'PI ($k_{p,i}$ = 0.75, $k_{i,i}$ = 35 s$^{-1}$)', '-'),
                      ('PI + R code kr 70', 'PI + resonant ($k_r$ = 70 s$^{-1}$, $\\omega_r = 2\\omega_0$)', '--')]:
    r = res_cl[name]
    sel = np.abs(r['f']) <= 1500
    ax[0].plot(r['f'][sel], 20*np.log10(np.abs(r['L'][sel])), st, lw=1.1, label=lab)
    ax[1].plot(r['f'][sel], np.degrees(np.angle(r['L'][sel])), st, lw=1.1, label=lab)
ax[0].axhline(0, color='k', lw=0.5)
ax[0].set_ylabel('$|L_i|$ [dB]')
ax[0].set_ylim(-30, 50)
ax[0].legend(loc='upper right')
ax[0].set_title('current loop: open-loop response in the positive dq frame ($f<0$: negative sequence)')
ax[1].set_ylabel('arg $L_i$ [deg]')
ax[1].set_xlabel('frequency in the dq frame [Hz]')
ax[1].set_ylim(-360, 180)
ax[1].set_yticks([-360, -270, -180, -90, 0, 90, 180])
for a_ in ax:
    a_.axvline(-2*f0, color='gray', lw=0.6, ls=':')
    a_.axvline(2*f0, color='gray', lw=0.6, ls=':')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_current_loop_bode.pdf'))
plt.close(fig)

# ---- figure: tracking of a negative-sequence current reference
fig, ax = plt.subplots(1, 1, figsize=(FIGW, 2.9))
for name, lab in [('PI only', 'PI only'), ('PI + R code kr 35', '$k_r = k_{i,i}$ = 35 s$^{-1}$'),
                  ('PI + R code kr 70', '$k_r = 2k_{i,i}$ = 70 s$^{-1}$'), ('PI + R code kr 140', '$k_r = 4k_{i,i}$ = 140 s$^{-1}$'),
                  ('PI + R fwd Euler kr 45 delta 0.015 (library values)', 'forward Euler resonator ($k_r$ = 45, $\\delta$ = 0.015): unstable')]:
    r = res_cl[name]
    sel = r['t'] <= 0.5
    ax.plot(r['t'][sel]*1e3, r['err'][sel], lw=1.1, label=lab)
ax.set_xlabel('t [ms]')
ax.set_ylabel('$|e_i|/|i^*_n|$')
ax.set_yscale('log')
ax.set_ylim(1e-3, 3)
ax.legend(loc='center right')
ax.set_title('current loop: error envelope, negative-sequence reference step 0.2 pu at $-2f_0$')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_negseq_tracking.pdf'))
plt.close(fig)

# =============================================================================================== 4. discretization of the resonant term
pr('\n' + '=' * 100)
pr('4. DISCRETIZATION OF THE RESONANT INTERNAL MODEL (undamped oscillator at wr): pole modulus and frequency')


def disc_poles(x, method):
    """poles of the discrete undamped oscillator for x = wr ts, as (|z|, angle) for the given method"""
    Ac = np.array([[0.0, 1.0], [-1.0, 0.0]])          # wr = 1, ts = x
    if method == 'forward Euler':
        Az = np.eye(2) + Ac*x
    elif method == 'backward Euler':
        Az = np.linalg.inv(np.eye(2) - Ac*x)
    elif method == 'Taylor 2':
        Az = np.eye(2) + Ac*x + 0.5*Ac@Ac*x*x
    elif method == 'Tustin':
        Az = np.linalg.solve(np.eye(2) - Ac*x/2, np.eye(2) + Ac*x/2)
    elif method in ('exact (ZOH)', 'Tustin prewarped', 'semi-implicit, a = 2 sin(x/2)'):
        if method == 'semi-implicit, a = 2 sin(x/2)':
            a = 2*np.sin(x/2)
            Az = np.array([[1.0, -a], [a, 1 - a*a]])
        else:
            Az = expm(Ac*x)
    elif method == 'semi-implicit, a = x':
        Az = np.array([[1.0, -x], [x, 1 - x*x]])
    else:
        raise ValueError(method)
    ev = np.linalg.eigvals(Az)
    k = np.argmax(ev.imag)
    return abs(ev[k]), np.angle(ev[k])


methods = ['forward Euler', 'backward Euler', 'Taylor 2', 'Tustin', 'semi-implicit, a = x', 'semi-implicit, a = 2 sin(x/2)', 'exact (ZOH)']
apps = [('study model, 2 f0 = 100 Hz, ts = 125 us', 2*np.pi*100, 125e-6),
        ('TELEMA VFC0, 2 f0 = 33.3 Hz, ts = 200 us', 2*np.pi*100/3, 200e-6),
        ('FHT internal model, f0 = 50 Hz, ts = 125 us', 2*np.pi*50, 125e-6),
        ('FHT internal model, f0 = 16.7 Hz, ts = 200 us', 2*np.pi*50/3, 200e-6)]
for lab, wr_, ts_ in apps:
    x = wr_*ts_
    pr(f'\n  {lab}: wr ts = {x:.4f}')
    for m in methods:
        mod, ang = disc_poles(x, m)
        pr(f'    {m:32s} |z| = {mod:.7f}  ->  sigma = ln|z|/ts = {np.log(mod)/ts_:+8.3f} 1/s;  '
           f'resonance {ang/ts_/2/np.pi:9.4f} Hz (error {(ang/x - 1)*100:+.4f} %)')

xg = np.logspace(-2.3, -0.3, 300)
fig, ax = plt.subplots(1, 2, figsize=(FIGW, 2.8))
styles = {'forward Euler': ('C3', '-'), 'backward Euler': ('C0', '-'), 'Taylor 2': ('C1', '-'), 'Tustin': ('C2', '-'),
          'semi-implicit, a = x': ('C4', '--'), 'exact (ZOH)': ('k', '-')}
for m, (c, st) in styles.items():
    mods = np.array([disc_poles(x, m)[0] for x in xg])
    angs = np.array([disc_poles(x, m)[1] for x in xg])
    ax[0].plot(xg, np.abs(np.log(mods))/xg, st, color=c, lw=1.1, label=m)
    ax[1].plot(xg, np.abs(angs/xg - 1), st, color=c, lw=1.1, label=m)
for a_ in ax:
    a_.set_xscale('log')
    a_.set_yscale('log')
    a_.set_xlabel('$\\omega_r t_s$ [rad]')
    for lab, wr_, ts_ in apps[:2]:
        a_.axvline(wr_*ts_, color='gray', lw=0.6, ls=':')
ax[0].set_ylabel('$|\\ln|z||/(\\omega_r t_s) = |\\sigma|/\\omega_r$')
ax[0].set_title('pole modulus error')
ax[0].text(0.03, 0.04, 'Tustin, semi-implicit, ZOH: 0', transform=ax[0].transAxes, fontsize=7.5)
ax[0].text(0.03, 0.12, 'forward and backward Euler coincide', transform=ax[0].transAxes, fontsize=7.5)
ax[0].set_ylim(1e-7, 1)
ax[1].set_ylabel('$|\\Delta\\omega_r|/\\omega_r$')
ax[1].set_title('resonance frequency error')
ax[1].text(0.03, 0.04, 'semi-implicit with $2\\sin(x/2)$, ZOH: 0', transform=ax[1].transAxes, fontsize=7.5)
ax[1].set_ylim(1e-7, 1)
ax[1].legend(loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_resonant_discretization.pdf'))
plt.close(fig)

# =============================================================================================== 5. FHT and sequence separation
pr('\n' + '=' * 100)
pr('5. FIRST HARMONIC TRACKER (resonant observer) AND SEQUENCE SEPARATION')


def fht_gains(w, delta, ts_, kind):
    """observer gains for poles at (-1, -4) w; kind: 'cont' (continuous A), 'fwd' (design on I + A ts, as the models),
    'exact' (design on expm(A ts))"""
    Ac = np.array([[0.0, 1.0], [-w*w, -delta*w]])
    p = np.array([-1.0, -4.0])*w
    if kind == 'cont':
        Am, pp = Ac, p
    elif kind == 'fwd':
        Am, pp = np.eye(2) + Ac*ts_, np.exp(p*ts_)
    elif kind == 'exact':
        Am, pp = expm(Ac*ts_), np.exp(p*ts_)
    a1, a0 = -(pp[0] + pp[1]), pp[0]*pp[1]
    l1 = Am[0, 0] + Am[1, 1] + a1
    l2 = Am[1, 0] + (a0 - (Am[0, 0] - l1)*Am[1, 1])/Am[0, 1]
    return np.array([l1, l2])


def fht_discrete_A(w, delta, ts_, kind):
    Ac = np.array([[0.0, 1.0], [-w*w, -delta*w]])
    if kind == 'taylor2':                      # first_harmonic_tracker.c
        return np.eye(2) + Ac*ts_ + 0.5*Ac@Ac*ts_*ts_
    if kind == 'fwd':
        return np.eye(2) + Ac*ts_
    if kind == 'exact':
        return expm(Ac*ts_)
    raise ValueError(kind)


def fht_response_c(s, w, delta, L):
    """continuous observer: uhat/u and v/u"""
    Ac = np.array([[0.0, 1.0], [-w*w, -delta*w]])
    X = np.linalg.solve(s*np.eye(2) - (Ac - np.outer(L, [1.0, 0.0])), L)
    return X[0], X[1]/w


def fht_response_d(z, Af, L, w, v_timing):
    """discrete observer as coded: x(k+1) = Af x + L (u - x1); uhat = x1(k); v = x2(k+1)/w ('after') or x2(k)/w ('before')"""
    X = np.linalg.solve(z*np.eye(2) - (Af - np.outer(L, [1.0, 0.0])), L)
    uhat = X[0]
    v = X[1]*(z if v_timing == 'after' else 1.0)/w
    return uhat, v


for lab, w_, ts_ in [('study model 50 Hz, 125 us', w0, ts), ('TELEMA VFC0 16.7 Hz, 200 us', 2*np.pi*50/3, 200e-6)]:
    pr(f'\n  {lab}:')
    for dl in (0.05, 0.0):
        Lc = fht_gains(w_, dl, ts_, 'cont')
        Ld = fht_gains(w_, dl, ts_, 'fwd')
        pr(f'    delta = {dl}: continuous gains l1 = {Lc[0]:.2f} = {Lc[0]/w_:.3f} w0, l2 = {Lc[1]:.0f} = {Lc[1]/w_**2:.4f} w0^2; '
           f'discrete gains (design on I + A ts) Ld = [{Ld[0]:.4f}, {Ld[1]:.3f}]')
    pr('    leakage of a positive-sequence input into the negative-sequence estimate |u-|/|u| (and |u+|/|u| for a negative-sequence input):')
    for dl in (0.05, 0.0):
        for kind_A, kind_L, timing in [('cont', 'cont', 'cont'), ('taylor2', 'fwd', 'after'), ('taylor2', 'fwd', 'before'),
                                       ('fwd', 'fwd', 'after'), ('exact', 'exact', 'before')]:
            out = []
            for sg in (+1, -1):
                if kind_A == 'cont':
                    uh, v = fht_response_c(1j*sg*w_, w_, dl, fht_gains(w_, dl, ts_, 'cont'))
                else:
                    uh, v = fht_response_d(np.exp(1j*sg*w_*ts_), fht_discrete_A(w_, dl, ts_, kind_A), fht_gains(w_, dl, ts_, kind_L), w_, timing)
                up, un = 0.5*(uh - 1j*v), 0.5*(uh + 1j*v)
                out.append((abs(un) if sg > 0 else abs(up), abs(up) if sg > 0 else abs(un)))
            pr(f'      delta {dl:<5} A {kind_A:8s} gains {kind_L:6s} v {timing:7s}: +w0 input -> |u-| = {out[0][0]*100:6.3f} % (|u+| = {out[0][1]:.4f}); '
               f'-w0 input -> |u+| = {out[1][0]*100:6.3f} % (|u-| = {out[1][1]:.4f})')
    # observer poles and settling (continuous)
    pr(f'    observer poles (-1, -4) w0 = {-w_:.1f}, {-4*w_:.1f} 1/s: slow time constant {1e3/w_:.1f} ms, 2 % settling of the sequence estimate about {4e3/w_:.0f} ms')

# ---- figure: sequence separation, frequency response (continuous, delta 0 and 0.05) and start-up transient
fig, ax = plt.subplots(1, 2, figsize=(FIGW, 2.8))
fgrid = np.linspace(-5*f0, 5*f0, 2001)
for dl, st in ((0.0, '-'), (0.05, '--')):
    Lc = fht_gains(w0, dl, ts, 'cont')
    up = np.zeros(len(fgrid), complex)
    un = np.zeros(len(fgrid), complex)
    for k, ff in enumerate(fgrid):
        uh, v = fht_response_c(2j*np.pi*ff, w0, dl, Lc)
        up[k], un[k] = 0.5*(uh - 1j*v), 0.5*(uh + 1j*v)
    ax[0].plot(fgrid/f0, np.abs(up), st, color='C0', lw=1.1, label=f'$|u^+/u|$, $\\delta$ = {dl}')
    ax[0].plot(fgrid/f0, np.abs(un), st, color='C3', lw=1.1, label=f'$|u^-/u|$, $\\delta$ = {dl}')
ax[0].set_xlabel('$f/f_0$ of the input ($f<0$: negative sequence)')
ax[0].set_ylabel('magnitude')
ax[0].set_ylim(0, 1.1)
ax[0].legend(loc='upper left')
ax[0].set_title('sequence separation, frequency response')
# start-up: positive-sequence input 1 pu applied at t = 0, discrete observer as coded (delta 0.05, Taylor 2)
Nf = int(0.06/ts)
tf_ = np.arange(Nf)*ts
for dl, timing, lab, st in ((0.05, 'after', '$\\delta$ = 0.05, $\\hat v$ after the update (as coded)', '--'), (0.0, 'before', '$\\delta$ = 0, $\\hat v$ before the update', '-')):
    Af = fht_discrete_A(w0, dl, ts, 'taylor2')
    Lf_ = fht_gains(w0, dl, ts, 'fwd')
    x = np.zeros(2, complex)
    up_t = np.zeros(Nf, complex)
    un_t = np.zeros(Nf, complex)
    for k in range(Nf):
        u = np.exp(1j*w0*tf_[k])
        e = u - x[0]
        xn = Af @ x + Lf_*e
        uh = x[0]
        v = (xn[1] if timing == 'after' else x[1])/w0
        up_t[k], un_t[k] = 0.5*(uh - 1j*v), 0.5*(uh + 1j*v)
        x = xn
    ax[1].plot(tf_*1e3, np.abs(up_t), st, color='C0', lw=1.1, label='$|u^+|$, ' + lab)
    ax[1].plot(tf_*1e3, np.abs(un_t), st, color='C3', lw=1.1, label='$|u^-|$, ' + lab)
ax[1].set_xlabel('t [ms]')
ax[1].set_ylabel('[pu]')
ax[1].set_ylim(0, 1.25)
ax[1].legend(loc='center right', fontsize=6.8)
ax[1].set_title('positive-sequence input 1 pu at t = 0')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_fht_sequence_separation.pdf'))
plt.close(fig)

# =============================================================================================== 6. full loop
pr('\n' + '=' * 100)
pr('6. FULL LOOP: voltage loops (positive and negative sequence), FHT, current loop with the resonant term')


def full_loop_ss(res, kpu=kp_u, kiu=ki_u, kpn=kp_n, kin=ki_n, nsc=True, delta=delta_fht, v_timing='after', fht_A='taylor2'):
    """states [x(3), d_i, d_u, xf(2), x_I, x_r(2), x_U, x_N], inputs [u_ref, i_dist]"""
    Ar, Br, Cr, Dr = res
    n = 12
    I = np.eye(n, dtype=complex)
    Af = fht_discrete_A(w0, delta, ts, fht_A)
    Lf_ = fht_gains(w0, delta, ts, 'exact' if fht_A == 'exact' else 'fwd')
    r1, r2 = np.exp(-1j*w0*ts), np.exp(-2j*w0*ts)
    # FHT on the delayed voltage sample, positive-frame representation of the stationary states
    ef = I[4] - I[5]
    xf1n = Af[0, 0]*I[5] + Af[0, 1]*I[6] + Lf_[0]*ef
    xf2n = Af[1, 0]*I[5] + Af[1, 1]*I[6] + Lf_[1]*ef
    uhat = I[5]
    v = (xf2n if v_timing == 'after' else I[6])/w0
    up, un = 0.5*(uhat - 1j*v), 0.5*(uhat + 1j*v)
    # positive-sequence voltage PI (reference u_ref): output with the updated state
    ep_x, ep_u = -up, np.array([1.0, 0.0])
    irp_x, irp_u = (kpu + kiu*ts)*ep_x + I[10], (kpu + kiu*ts)*ep_u
    # negative-sequence voltage PI in the negative frame, written in the positive frame (reference 0)
    if nsc:
        irn_x = -(kpn + kin*ts)*un + I[11]
        xN_next = r2*(I[11] - kin*ts*un)
    else:
        irn_x, xN_next = np.zeros(n, complex), np.zeros(n, complex)
    ir_x, ir_u = irp_x + irn_x, irp_u
    # current loop
    e_x, e_u = ir_x - I[3], ir_u
    r_x, r_u = Cr[0, 0]*I[8] + Cr[0, 1]*I[9] + Dr*e_x, Dr*e_u
    m_x, m_u = (kp_i + ki_i*ts)*e_x + I[7] + r_x, (kp_i + ki_i*ts)*e_u + r_u
    Acl = np.zeros((n, n), complex)
    Bcl = np.zeros((n, 2), complex)
    Acl[:3, :3] = Ad
    Acl[:3, :] += np.outer(Bd[:, 0], m_x)
    Bcl[:3, :] += np.outer(Bd[:, 0], m_u)
    Bcl[:3, 1] += Bd[:, 1]
    Acl[3, :3] = C_i1
    Acl[4, :3] = C_ug
    Acl[5, :] = r1*xf1n
    Acl[6, :] = r1*xf2n
    Acl[7, :] = I[7] + ki_i*ts*e_x
    Bcl[7, :] = ki_i*ts*e_u
    Acl[8:10, :] = Ar @ np.vstack([I[8], I[9]]) + np.outer(Br[:, 0], e_x)
    Bcl[8:10, :] = np.outer(Br[:, 0], e_u)
    Acl[10, :] = I[10] + kiu*ts*ep_x
    Bcl[10, :] = kiu*ts*ep_u
    Acl[11, :] = xN_next
    outs = {'i1': I[0], 'ug': np.concatenate([C_ug, np.zeros(9)]).astype(complex), 'up': up, 'un': un, 'ir': ir_x, 'm': m_x}
    return Acl, Bcl, outs


def demod(sig, t_, f_, Nper):
    """component of a positive-frame signal at frequency f_ (dq), averaged over Nper samples (envelope)"""
    ker = np.ones(Nper)/Nper
    return np.convolve(sig*np.exp(-2j*np.pi*f_*t_), ker, mode='full')[:len(sig)]


Tsim2 = 1.6
N2 = int(Tsim2/ts)
t2 = np.arange(N2)*ts
Nper = int(round(1/f0/ts))
t_load = 0.8
full = {}
configs = [('current loop only (no voltage loop)', 'none', 0.0, False, dict(kpu=0.0, kiu=0.0)),
           ('voltage loop, no compensator', 'none', 0.0, False, {}),
           ('compensator, PI only in the current loop', 'none', 0.0, True, {}),
           ('compensator, kr 35', 'code', 35.0, True, {}),
           ('compensator, kr 70', 'code', 70.0, True, {}),
           ('compensator, kr 140', 'code', 140.0, True, {}),
           ('compensator, kr 70, fwd Euler resonator', 'fwd', 70.0, True, {}),
           ('compensator, kr 70, FHT delta 0 and v before the update', 'code', 70.0, True, dict(delta=0.0, v_timing='before'))]
for name, meth, kr_, nsc, kw in configs:
    res = resonator(wr, kr_, ts, meth)
    Acl, Bcl, outs = full_loop_ss(res, nsc=nsc, **kw)
    ev, s = modes(Acl)
    u = np.zeros((N2, 2), complex)
    u[:, 0] = 1.0
    sel = t2 >= t_load
    u[sel, 1] = 0.3*np.exp(-1j*wr*t2[sel])                      # negative-sequence load current 0.3 pu
    X = simulate(Acl, Bcl, u)
    ug = X @ outs['ug']
    un_act = demod(ug, t2, -2*f0, Nper)                          # actual negative-sequence terminal voltage
    up_act = demod(ug, t2, 0.0, Nper)                            # actual positive-sequence terminal voltage
    un_est = X @ outs['un']
    k0 = np.searchsorted(t2, t_load)
    kb = k0 - 2
    i1 = X[:, 0]
    pr(f'\n  [{name}] max|z| = {np.abs(ev).max():.6f}; least damped / slowest modes: {fmt_modes(s, 5)}')
    pr(f'    steady state before the unbalance: |u+| = {abs(up_act[kb]):.4f} pu, |i1| = {abs(i1[kb]):.3f} pu, '
       f'FHT estimate |u-| = {abs(un_est[kb]):.4f} pu (leakage); start-up 2 % settling of |u+| {settle(up_act[:k0] - up_act[kb], 1.0)*1e3:.0f} ms')
    un_s = un_act[k0:]
    pr(f'    negative-sequence load current 0.3 pu: |u-| peak {np.abs(un_s).max():.3f} pu, after 0.8 s {abs(un_s[-1]):.4f} pu, '
       f'2 % settling {settle(un_s - un_s[-1], 0.3)*1e3:.0f} ms; |u+| after 0.8 s {abs(up_act[-1]):.4f} pu')
    full[name] = dict(t=t2, un=np.abs(un_act), up=np.abs(up_act), un_est=np.abs(un_est), Acl=Acl, Bcl=Bcl, outs=outs)

# ---- output impedance of the converter in the dq frame
fimp = np.linspace(-400, 400, 1601) + 0.0123
fig, ax = plt.subplots(1, 1, figsize=(FIGW, 2.9))
for name, lab in [('current loop only (no voltage loop)', 'current loop only (current source)'),
                  ('voltage loop, no compensator', 'with the positive-sequence voltage loop'),
                  ('compensator, PI only in the current loop', '+ negative-sequence loop, PI only in the current loop'),
                  ('compensator, kr 70', '+ negative-sequence loop and resonant term ($k_r$ = 70 s$^{-1}$)')]:
    d = full[name]
    Z = np.array([abs((d['outs']['ug'] @ np.linalg.solve(np.exp(2j*np.pi*ff*ts)*np.eye(12) - d['Acl'], d['Bcl'][:, 1]))) for ff in fimp])
    ax.plot(fimp, Z, lw=1.1, label=lab)
ax.set_yscale('log')
ax.set_ylim(1e-3, 10)
ax.set_xlabel('frequency in the dq frame [Hz] ($0$: positive sequence, $-2f_0$: negative sequence)')
ax.set_ylabel('$|Z_{out}|$ [pu]')
ax.axvline(-2*f0, color='gray', lw=0.6, ls=':')
ax.legend(loc='lower left', fontsize=7)
ax.set_title('output impedance $|u_g/i_{dist}|$ at the converter terminals')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_output_impedance.pdf'))
plt.close(fig)

# ---- figure: start-up of |u+| and response of |u-| to the negative-sequence load step
fig, ax = plt.subplots(1, 2, figsize=(FIGW, 2.8))
for name, lab in [('compensator, kr 70', 'FHT $\\delta$ = 0.05, $\\hat v$ after the update (as coded)'),
                  ('compensator, kr 70, FHT delta 0 and v before the update', 'FHT $\\delta$ = 0, $\\hat v$ before the update')]:
    d = full[name]
    sl = d['t'] <= 0.6
    ax[0].plot(d['t'][sl]*1e3, d['up'][sl], lw=1.1, label=lab)
ax[0].set_xlabel('t [ms]')
ax[0].set_ylabel('$|u^+|$ [pu]')
ax[0].set_ylim(0, 1.2)
ax[0].legend(loc='lower right')
ax[0].set_title('start-up: $u^*$ = 1 pu at t = 0, actual $|u^+|$')
for name, lab in [('voltage loop, no compensator', 'no compensator'),
                  ('compensator, PI only in the current loop', 'compensator, PI only'),
                  ('compensator, kr 35', 'compensator, $k_r$ = 35 s$^{-1}$'),
                  ('compensator, kr 70', 'compensator, $k_r$ = 70 s$^{-1}$'),
                  ('compensator, kr 140', 'compensator, $k_r$ = 140 s$^{-1}$')]:
    d = full[name]
    sl = (d['t'] >= t_load - 0.02) & (d['t'] <= t_load + 0.7)
    ax[1].plot((d['t'][sl] - t_load)*1e3, d['un'][sl], lw=1.1, label=lab)
ax[1].set_xlabel('t [ms]')
ax[1].set_ylabel('$|u^-|$ [pu]')
ax[1].set_ylim(0, 0.2)
ax[1].legend(loc='upper right')
ax[1].set_title('negative-sequence load 0.3 pu at t = 0')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_negseq_load_step.pdf'))
plt.close(fig)

# =============================================================================================== 7. voltage loop figures
pr('\n' + '=' * 100)
pr('7. POSITIVE-SEQUENCE VOLTAGE LOOP, simplified: plant i1 -> u with the current loop ideal and the load as a resistance')
Zl = Rload/zbez
pr(f'  dominant pole s = -ki_u Z_l/(1 + kp_u Z_l) = {-ki_u*Zl/(1 + kp_u*Zl):.1f} 1/s (Z_l = {Zl:.3f} pu); '
   f'capacitor pole 1/(C Z_l) = {1/(Cp*Zl):.0f} rad/s, cross-coupling w0 C Z_l = {w0*Cp*Zl:.3f}')
pr('\nfigures written to', OUT)
