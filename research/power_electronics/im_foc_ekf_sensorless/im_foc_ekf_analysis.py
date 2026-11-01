#!/usr/bin/env python3
"""Companion script of the note im_foc_ekf_sensorless (pwr-control, repositories-documentation).

Induction machine of foundations_of_power_electronics/im_analysis_and_design and inv_im (ABB M3BP 355MLB 6, 261 kW,
460 V, 60 Hz, six poles) fed by a two-level inverter (4 kHz PWM, double update, control at 8 kHz) through a 230 uH
inductor from a 1200 V DC link. The script prints every number quoted in the note and writes the figures in
./figures:

  1. machine data, bases and the normalized coefficients of im_setup / ekf_im_setup (library);
  2. design of the field-oriented control (current, flux and speed loops): open loops of the gains of the model
     and of the proposed gains (fig_foc_loops.pdf);
  3. discretization of the EKF prediction (forward Euler against second-order Taylor) and observability of the
     linearized model versus stator frequency (fig_observability.pdf);
  4. averaged time-domain simulations (machine model with the series inductor, discrete control at 8 kHz, one sample
     of computation delay plus the PWM hold, 12-bit current quantization and noise, dead-time voltage error and its
     feed-forward compensation):
       R1  encoder FOC; the proposed EKF (second-order and Euler prediction) and the EKF of the library run in open
           loop on the same data (fig_ekf_openloop.pdf);
       R2  sensorless FOC with the proposed EKF in the loop: magnetization, ramp to rated speed, rated torque, ramp
           down to 5 % speed, rated torque at 5 % speed (fig_sensorless_run.pdf); R2b the same without dead-time
           compensation;
       R3  as R2 with the stator and rotor resistances of the machine 30 % above the model and the dead time
           compensated at 90 % (R3) and at 80 % (R3b) (fig_sensitivity.pdf);
       R4  slow reversal through zero speed with an active load of 0.2 pu, nominal and as R3 (fig_reversal.pdf);
       sensitivity table at 5 % speed, one error at a time.

Run: python3 im_foc_ekf_analysis.py  (numpy, scipy, matplotlib; about five minutes on one core).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 1.1})
RNG = np.random.default_rng(1)


# ---------------------------------------------------------------------------------------------------------------
# 1. machine (im_setup of the library) and bases
# ---------------------------------------------------------------------------------------------------------------
def im_machine(pwr_nom=261e3, u_nom=460, i_nom=411, freq_nom=60, rpm_load=1195, number_poles=6, eta=0.95,
               Rs=2e-3, u_no_load=460, i_no_load=180, f_no_load=60, u_cc=77, i_cc=470, f_cc=50, Jm=5,
               load_friction_factor=0.75):
    m = dict(pwr_nom=pwr_nom, u_nom=u_nom, i_nom=i_nom, freq_nom=freq_nom, Rs=Rs, Jm=Jm, i_no_load=i_no_load,
             rpm_load=rpm_load)
    m['pp'] = number_poles / 2
    m['omega_bez'] = 2 * np.pi * freq_nom
    m['omega_m_bez'] = m['omega_bez'] / m['pp']
    m['omega_m_load'] = rpm_load / 60 * 2 * np.pi
    m['slip_nom'] = (freq_nom * 60 / m['pp'] - rpm_load) / (freq_nom * 60 / m['pp'])
    m['torque_nom'] = pwr_nom / eta / m['omega_m_bez']
    Rr = Rs
    Ls = u_no_load / np.sqrt(3) / i_no_load / (2 * np.pi * f_no_load)
    Zs = (u_cc / np.sqrt(3)) / i_cc
    Lsig = np.sqrt(Zs ** 2 - (Rs + Rr) ** 2) / (2 * np.pi * f_cc)
    Lds = Lsig / 2; Ldr = Lsig - Lds; Lm = Ls - Lds; Lr = Lm + Ldr
    w, s = m['omega_bez'], m['slip_nom']
    Rr_t, dR, Rmax, tol = Rs / 5, Rs / 100, Rs * 5, 0.05
    while Rr_t < Rmax:
        a = w * Lm * Rr_t / s; b = w * Lm * a
        c = (Rr_t / s) ** 2 + (w * Lr) ** 2
        d = (Rr_t / s) ** 2 + w ** 2 * Ldr * Lr
        e = w ** 2 * Lm * Lr
        ZRe = Rs + b / c; ZIm = w * Lds + w * Lm * d / c; Z2 = ZRe ** 2 + ZIm ** 2
        IsRe = u_nom / np.sqrt(3) * ZRe / Z2; IsIm = -u_nom / np.sqrt(3) * ZIm / Z2
        IrRe = (a * IsIm - e * IsRe) / c; IrIm = -(a * IsRe + e * IsIm) / c
        Tn = 3 * (IrRe ** 2 + IrIm ** 2) * Rr_t * (1 - s) / s / m['omega_m_load']
        if m['torque_nom'] * (1 - tol) <= Tn <= m['torque_nom'] * (1 + tol):
            break
        Rr_t += dR
    m.update(Rr=Rr_t, Ls=Ls, Lds=Lds, Ldr=Ldr, Lm=Lm, Lr=Lr, T_circuit=Tn)
    m['sigmaLs'] = Ls * (1 - Lm ** 2 / Ls / Lr)
    m['tau_r'] = Lr / Rr_t
    m['ubez'] = u_nom / np.sqrt(1.5); m['ibez'] = i_nom * np.sqrt(2)
    m['Xbez'] = m['ubez'] / m['ibez']; m['Lbez'] = m['Xbez'] / w
    m['psi_bez'] = m['ubez'] / w
    m['psi_nom'] = i_no_load * np.sqrt(2) * Lm
    m['T_bez'] = 1.5 * m['pp'] * m['psi_bez'] * m['ibez']          # natural torque base [Nm]
    m['load_friction'] = m['torque_nom'] * load_friction_factor / m['omega_m_bez']
    # normalized coefficients of im_setup (time normalized by omega_bez)
    n = {}
    n['Ls'] = Ls / m['Lbez']; n['Lm'] = Lm / m['Lbez']; n['Lr'] = Lr / m['Lbez']
    n['Rs'] = Rs / m['Xbez']; n['Rr'] = Rr_t / m['Xbez']
    n['alpha'] = n['Rr'] / n['Lr']
    n['sigma'] = n['Ls'] * (1 - n['Lm'] ** 2 / n['Ls'] / n['Lr'])
    n['beta'] = n['Lm'] / n['sigma'] / n['Lr']
    n['gamma'] = n['Rs'] / n['sigma'] + n['beta'] * n['alpha'] * n['Lm']
    n['Jm_lib'] = 0.5 * Jm * m['omega_m_bez'] / m['torque_nom']      # [s] Jm_norm of im_setup
    n['mu_lib'] = n['Lm'] / n['Jm_lib'] / n['Lr']
    m['n'] = n
    m['tau_m'] = Jm * m['omega_m_bez'] / m['T_bez']                  # [s] mechanical time constant, natural base
    return m


M = im_machine()
N = M['n']
UDC = 1200.0                       # battery / DC-link voltage [V]
LFI = 230e-6; RLFI = 157 * 0.05 * LFI
TS = 125e-6                        # control period (4 kHz PWM, double update)
WB = M['omega_bez']
U_CTRL = UDC / np.sqrt(3)          # 1 pu of the control voltage (linear limit of the SVM) [V peak]
K_UC = U_CTRL / M['ubez']          # machine pu of voltage per control pu
KT = N['Lm'] / N['Lr']             # torque per pu of psi_r x i_s (natural torque base)

# series inductor in pu and the total transient parameters seen by the inverter
N['sigma_t'] = N['sigma'] + LFI / M['Lbez']
N['Rs_t'] = N['Rs'] + RLFI / M['Xbez']
N['beta_t'] = N['Lm'] / N['sigma_t'] / N['Lr']
N['gamma_t'] = N['Rs_t'] / N['sigma_t'] + N['beta_t'] * N['alpha'] * N['Lm']

print('=== 1. Machine, bases and normalized coefficients ===')
print(f"Ls {M['Ls']*1e3:.4f} mH, Lm {M['Lm']*1e3:.4f} mH, Lr {M['Lr']*1e3:.4f} mH, sigma*Ls {M['sigmaLs']*1e3:.4f} mH, "
      f"Rs {M['Rs']*1e3:.2f} mOhm, Rr {M['Rr']*1e3:.3f} mOhm, tau_r {M['tau_r']:.3f} s")
print(f"bases: ubez {M['ubez']:.1f} V, ibez {M['ibez']:.1f} A, Zbez {M['Xbez']:.4f} Ohm, Lbez {M['Lbez']*1e3:.4f} mH, "
      f"psi_bez {M['psi_bez']:.4f} Vs, T_bez(3/2 pp psi i) {M['T_bez']:.0f} Nm, torque_nom(lib) {M['torque_nom']:.0f} Nm, "
      f"shaft torque at the rated point {M['pwr_nom']/M['omega_m_load']:.0f} Nm")
print(f"normalized: Ls {N['Ls']:.4f}, Lm {N['Lm']:.4f}, Lr {N['Lr']:.4f}, sigma {N['sigma']:.5f}, Rs {N['Rs']:.5f}, "
      f"Rr {N['Rr']:.5f}, alpha {N['alpha']:.6f}, beta {N['beta']:.3f}, gamma {N['gamma']:.5f}")
print(f"with the series inductor: sigma_t {N['sigma_t']:.5f} (x{N['sigma_t']/N['sigma']:.2f}), Rs_t {N['Rs_t']:.5f}, "
      f"beta_t {N['beta_t']:.3f}, gamma_t {N['gamma_t']:.5f}")
print(f"mechanics: tau_m = J*w_m_bez/T_bez = {M['tau_m']:.4f} s; Jm_norm(lib) = 1/2*J*w_m_bez/torque_nom = {N['Jm_lib']:.4f} s, "
      f"mu_norm(lib) = {N['mu_lib']:.3f}")
print(f"1 pu of control voltage = udc/sqrt(3) = {U_CTRL:.1f} V = {K_UC:.3f} ubez")

# ---------------------------------------------------------------------------------------------------------------
# 2. FOC design
# ---------------------------------------------------------------------------------------------------------------
print('\n=== 2. Field-oriented control: loop design ===')
w = np.logspace(-2, np.log10(np.pi / TS), 20000)
z = np.exp(1j * w * TS)
L_i_s = N['sigma_t'] / WB                                   # [pu s] transient inductance in pu
R_i = N['Rs_t'] + N['Rr'] * KT ** 2                         # [pu] resistance seen by the current loop
tau_el = L_i_s / R_i


def margins(L):
    mag = np.abs(L)
    k = np.where((mag[:-1] >= 1) & (mag[1:] < 1))[0]
    if len(k) == 0:
        return np.nan, np.nan
    k = k[0]
    wc = np.interp(1, [mag[k + 1], mag[k]], [w[k + 1], w[k]])
    ph = np.interp(wc, w, np.unwrap(np.angle(L)))
    return wc, 180 + np.degrees(ph)


def loop_current(kp_c, ki_c):
    """current loop, gains in control pu (pu of udc/sqrt(3) per pu of ibez)"""
    C = kp_c + ki_c * TS * z / (z - 1)
    P = K_UC / (R_i + 1j * w * L_i_s)
    return C * P * np.exp(-1j * w * 1.5 * TS)


# gains of the model (control pu) and proposed gains (crossover 200 Hz, zero on the electrical pole)
G_MODEL = dict(kp_i=0.25, ki_i=18.0, kp_psi=2.0, ki_psi=20.0, kp_w=2.5, ki_w=18.0)
fc_i = 200.0
kp_i_p = 2 * np.pi * fc_i * L_i_s / K_UC
ki_i_p = kp_i_p / tau_el
fc_w = 3.0
kp_w_p = 2 * np.pi * fc_w * M['tau_m']                       # torque (pu of T_bez) per pu of speed
ki_w_p = kp_w_p * 2 * np.pi * fc_w / 4
G_PROP = dict(kp_i=kp_i_p, ki_i=ki_i_p, kp_psi=6.0, ki_psi=6.0 / M['tau_r'], kp_w=kp_w_p, ki_w=ki_w_p)

L_i_m = loop_current(G_MODEL['kp_i'], G_MODEL['ki_i']); wc_im, pm_im = margins(L_i_m)
L_i_p = loop_current(G_PROP['kp_i'], G_PROP['ki_i']); wc_ip, pm_ip = margins(L_i_p)
print(f"current loop: L' = sigma*Ls + LFi = {(M['sigmaLs']+LFI)*1e3:.3f} mH, R' = {R_i*M['Xbez']*1e3:.2f} mOhm, "
      f"electrical time constant {tau_el*1e3:.1f} ms; Nyquist {1/2/TS:.0f} Hz")
print(f"  model    kp {G_MODEL['kp_i']:.3f}, ki {G_MODEL['ki_i']:.1f} 1/s: crossover {wc_im/2/np.pi:.0f} Hz, PM {pm_im:.0f} deg")
print(f"  proposed kp {G_PROP['kp_i']:.3f}, ki {G_PROP['ki_i']:.2f} 1/s (zero on the pole): crossover {wc_ip/2/np.pi:.0f} Hz, "
      f"PM {pm_ip:.0f} deg; the decoupling feed-forward removes the cross terms and the back-EMF")
# flux loop: plant psi/(Lm_n isx) = 1/(1 + s tau_r)
L_psi_m = (G_MODEL['kp_psi'] + G_MODEL['ki_psi'] * TS * z / (z - 1)) / (1 + 1j * w * M['tau_r'])
L_psi_p = (G_PROP['kp_psi'] + G_PROP['ki_psi'] * TS * z / (z - 1)) / (1 + 1j * w * M['tau_r'])
wc_pm, pm_pm = margins(L_psi_m); wc_pp, pm_pp = margins(L_psi_p)
print(f"flux loop: model kp 2 / ki 20: {wc_pm/2/np.pi:.2f} Hz, PM {pm_pm:.0f} deg; proposed kp 6 / ki = kp/tau_r = "
      f"{G_PROP['ki_psi']:.2f}: {wc_pp/2/np.pi:.2f} Hz, PM {pm_pp:.0f} deg, closed loop tau_r/kp = {M['tau_r']/6:.3f} s")
# speed loop: model: output isy, torque kT*psi*isy; proposed: output torque, isy = T/(kT psi)
a_w = M['load_friction'] * M['omega_m_bez'] / M['T_bez'] / M['tau_m']   # friction pole [1/s]
Gi_m = 1 / (1 + 1j * w / wc_im); Gi_p = 1 / (1 + 1j * w / wc_ip)
L_w_m = (G_MODEL['kp_w'] + G_MODEL['ki_w'] * TS * z / (z - 1)) * KT / M['tau_m'] / (1j * w + a_w) * Gi_m
L_w_p = (G_PROP['kp_w'] + G_PROP['ki_w'] * TS * z / (z - 1)) / M['tau_m'] / (1j * w + a_w) * Gi_p
wc_wm, pm_wm = margins(L_w_m); wc_wp, pm_wp = margins(L_w_p)
print(f"speed loop: tau_m {M['tau_m']:.3f} s, friction pole {a_w:.2f} rad/s; model kp 2.5 / ki 18 (output isy): "
      f"{wc_wm/2/np.pi:.2f} Hz, PM {pm_wm:.0f} deg; proposed kp {G_PROP['kp_w']:.2f} / ki {G_PROP['ki_w']:.1f} (output torque): "
      f"{wc_wp/2/np.pi:.2f} Hz, PM {pm_wp:.0f} deg")

fig, ax = plt.subplots(2, 1, figsize=(6.6, 5.2), sharex=True)
for L, lab, ls, c in [(L_i_m, 'corrente, modello', '--', 'C0'), (L_i_p, 'corrente, proposto', '-', 'C0'),
                      (L_psi_m, 'flusso, modello', '--', 'C1'), (L_psi_p, 'flusso, proposto', '-', 'C1'),
                      (L_w_m, 'velocità, modello', '--', 'C2'), (L_w_p, 'velocità, proposto', '-', 'C2')]:
    ax[0].semilogx(w / 2 / np.pi, 20 * np.log10(np.abs(L)), ls, color=c, label=lab)
    ax[1].semilogx(w / 2 / np.pi, np.degrees(np.unwrap(np.angle(L))), ls, color=c)
ax[0].axhline(0, color='k', lw=0.6); ax[1].axhline(-180, color='k', lw=0.6)
ax[0].set_ylim(-60, 80); ax[1].set_ylim(-270, 0)
ax[0].set_ylabel('|L| [dB]'); ax[1].set_ylabel('fase [deg]'); ax[1].set_xlabel('frequenza [Hz]')
ax[1].legend(*ax[0].get_legend_handles_labels(), fontsize=7, ncol=3, loc='lower left')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_foc_loops.pdf')); plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# EKF model (proposed): states [i_alpha, i_beta, psi_alpha, psi_beta, omega, T_load] in pu, time in seconds
# ---------------------------------------------------------------------------------------------------------------
class EKF:
    """Extended Kalman filter of the induction machine in the stationary frame.

    n = 6: x = [i_a, i_b, psi_a, psi_b, w, TL] with the mechanical equation (torque base T_bez, tau_m);
    n = 5: x = [i_a, i_b, psi_a, psi_b, w] with the speed as a random walk.
    order = 2 (default): second-order Taylor prediction x+ = x + ts f + ts^2/2 A f, F = I + ts A + (ts A)^2/2;
    order = 1: forward Euler x+ = x + ts f, F = I + ts A (unstable flux mode above about 0.3 pu of speed at 8 kHz);
    cov_order: order of F for the covariance only (default = order; 1 halves the cost and gives the same results).
    Joseph-form update. par: dict with sigma, Rs, Lm, Lr, alpha (normalized), tau_m [s]."""

    def __init__(self, par, n=6, q=None, r=None, x0=None, p0=1e-3, order=2, cov_order=None):
        self.n = n
        self.order = order
        self.cov_order = order if cov_order is None else cov_order
        self.p = par
        self.x = np.zeros(n) if x0 is None else np.array(x0, float)
        self.P = np.eye(n) * p0
        self.Q = np.diag(q)
        self.R = np.eye(2) * r
        s, Lm, Lr, a = par['sigma'], par['Lm'], par['Lr'], par['alpha']
        self.b = Lm / s / Lr
        self.g = par['Rs'] / s + self.b * a * Lm
        self.kT = Lm / Lr

    def f(self, x, u):
        p, b, g = self.p, self.b, self.g
        ia, ib, pa, pb, om = x[0], x[1], x[2], x[3], x[4]
        a, Lm, s = p['alpha'], p['Lm'], p['sigma']
        d = np.zeros(self.n)
        d[0] = WB * (-g * ia + b * a * pa + b * om * pb + u[0] / s)
        d[1] = WB * (-g * ib + b * a * pb - b * om * pa + u[1] / s)
        d[2] = WB * (a * Lm * ia - a * pa - om * pb)
        d[3] = WB * (a * Lm * ib - a * pb + om * pa)
        if self.n == 6:
            d[4] = (self.kT * (pa * ib - pb * ia) - x[5]) / p['tau_m']
        return d

    def jac(self, x):
        """continuous-time Jacobian df/dx [1/s]"""
        p, b, g = self.p, self.b, self.g
        ia, ib, pa, pb, om = x[0], x[1], x[2], x[3], x[4]
        a, Lm = p['alpha'], p['Lm']
        A = np.zeros((self.n, self.n))
        A[0, :5] = WB * np.array([-g, 0, b * a, b * om, b * pb])
        A[1, :5] = WB * np.array([0, -g, -b * om, b * a, -b * pa])
        A[2, :5] = WB * np.array([a * Lm, 0, -a, -om, -pb])
        A[3, :5] = WB * np.array([0, a * Lm, om, -a, pa])
        if self.n == 6:
            A[4, :6] = np.array([-self.kT * pb, self.kT * pa, self.kT * ib, -self.kT * ia, 0, -1]) / p['tau_m']
        return A

    def phi(self, x, order=None):
        """discrete transition matrix of the linearized model (order 1 or 2)"""
        A = self.jac(x) * TS
        return np.eye(self.n) + A + (A @ A / 2 if (self.order if order is None else order) == 2 else 0)

    def step(self, u, y, hold=False):
        """u: voltage applied during [k, k+1] (machine pu, alpha-beta); y: currents measured at k+1 (pu).
        hold=True: standstill known (pre-magnetization): speed and load states held at zero, not updated."""
        Q = self.Q
        if hold:
            self.x[4:] = 0.0
            self.P[4:, :] = 0.0; self.P[:, 4:] = 0.0
            Q = self.Q.copy(); Q[4:, 4:] = 0.0
            self.held = True
        elif getattr(self, 'held', False):
            for k in range(4, self.n):
                self.P[k, k] = 1e-4
            self.held = False
        A = self.jac(self.x)
        fx = self.f(self.x, u)
        xp = self.x + TS * fx
        if self.order == 2:
            xp = xp + TS ** 2 / 2 * A @ fx
        F = self.phi(self.x, self.cov_order)
        Pp = F @ self.P @ F.T + Q
        S = Pp[:2, :2] + self.R
        K = Pp[:, :2] @ np.linalg.inv(S)
        self.x = xp + K @ (y - xp[:2])
        IKC = np.eye(self.n); IKC[:, :2] -= K
        self.P = IKC @ Pp @ IKC.T + K @ self.R @ K.T
        return self.x


class EKFLibrary:
    """EKF of the library (ekf_im_setup) as implemented in inv_im_param.slx (ekf/extended_kalman_observer):
    prediction x+ = A_d(x) x + ts*w_b*B u with A_d = I + ts*w_b*(A1 + A2 x1 + A3 x2 + A4 x3 + A5 x4) + ts*A6 x5
    (the A6 term without w_b), covariance with the full Jacobian I + ts*w_b*(A1 + ... + A6 x5), Q and R of
    ekf_im_setup, x(0) = [0 0 1 1 0 0]*ts, P(0) = 0; coefficients of the machine alone (no series inductor)."""

    def __init__(self, n_coef, sigma_key='sigma', Rs_key='Rs'):
        s = n_coef[sigma_key]; Lm, Lr, a = n_coef['Lm'], n_coef['Lr'], n_coef['alpha']
        b = Lm / s / Lr; g = n_coef[Rs_key] / s + b * a * Lm
        mu = n_coef['mu_lib']; jm = n_coef['Jm_lib']
        Z = np.zeros((6, 6))
        A1 = Z.copy(); A1[0, 0] = -g; A1[0, 2] = b * a; A1[1, 1] = -g; A1[1, 3] = b * a
        A1[2, 0] = a * Lm; A1[2, 2] = -a; A1[3, 1] = a * Lm; A1[3, 3] = -a; A1[4, 5] = -1 / jm
        A2 = Z.copy(); A2[4, 3] = -mu
        A3 = Z.copy(); A3[4, 2] = mu
        A4 = Z.copy(); A4[1, 4] = -b; A4[3, 4] = 1; A4[4, 1] = mu
        A5 = Z.copy(); A5[0, 4] = b; A5[2, 4] = -1; A5[4, 0] = -mu
        A6 = Z.copy(); A6[0, 3] = b; A6[1, 2] = -b; A6[2, 3] = -1; A6[3, 2] = 1
        self.A = [A1, A2, A3, A4, A5, A6]
        self.B = np.zeros((6, 2)); self.B[0, 0] = 1 / s; self.B[1, 1] = 1 / s
        self.Q = TS / 10 * np.diag([TS ** 2, TS ** 2, TS, TS, 1, 1])
        self.R = TS * np.eye(2)
        self.x = np.array([0, 0, 1, 1, 0, 0]) * TS
        self.P = np.zeros((6, 6))

    def step(self, u, y):
        A1, A2, A3, A4, A5, A6 = self.A
        x = self.x
        Asum = A1 + A2 * x[0] + A3 * x[1] + A4 * x[2] + A5 * x[3]
        Ad = np.eye(6) + TS * WB * Asum + TS * A6 * x[4]
        Ac = np.eye(6) + TS * WB * (Asum + A6 * x[4])
        xp = Ad @ x + TS * WB * self.B @ u
        Pp = Ac @ self.P @ Ac.T + self.Q
        S = Pp[:2, :2] + self.R
        K = Pp[:, :2] @ np.linalg.inv(S)
        self.x = xp + K @ (y - xp[:2])
        self.P = (np.eye(6) - K @ np.hstack([np.eye(2), np.zeros((2, 4))])) @ Pp
        return self.x


PAR_NOM = dict(sigma=N['sigma_t'], Rs=N['Rs_t'], Lm=N['Lm'], Lr=N['Lr'], alpha=N['alpha'], tau_m=M['tau_m'])

# noise: 12-bit acquisition on +-1049.8 A, 1 A rms of white noise on each phase
Q_ADC = 1049.835 / 2048
SIG_I = 1.0
r_meas = (SIG_I ** 2 + Q_ADC ** 2 / 12) / M['ibez'] ** 2
# process noise per step (pu^2): currents from 2 % voltage uncertainty, flux model, speed / load random walks
dq_i = (TS * WB * 0.02 / N['sigma_t']) ** 2
Q6 = [dq_i, dq_i, 1e-10, 1e-10, 1e-10, 1e-6]
Q5 = [dq_i, dq_i, 1e-10, 1e-10, 2e-8]
print(f"\nEKF noise: R = {r_meas:.3e} pu^2 (sigma {np.sqrt(r_meas)*M['ibez']:.2f} A); Q(i) = {dq_i:.2e}, Q(psi) = 1e-10, "
      f"Q6(w) = 1e-10, Q6(TL) = 1e-6, Q5(w) = 2e-8 per step")

# ---------------------------------------------------------------------------------------------------------------
# 3. discretization of the prediction and observability of the linearized model
# ---------------------------------------------------------------------------------------------------------------
def steady_state(w_s, T_pu, psi=1.0):
    """steady state in rotor-flux coordinates (flux on the real axis): x (6) and the slip pulsation [pu]"""
    isy = T_pu / (KT * psi)
    isx = psi / N['Lm']
    w_slip = isy / isx * N['alpha']                         # pu (time normalized)
    w_r = w_s - w_slip
    x = np.array([isx, isy, psi, 0.0, w_r, T_pu])
    return x, w_slip


ekf_tmp = EKF(PAR_NOM, n=6, q=[1] * 6, r=1)
th_b = WB * TS
w_crit = np.sqrt(2 * N['alpha'] / th_b)
x_rated = steady_state(1.0 + 1e-9, 1.0)[0]
x_rated[4] = 1.0
mods = {}
for lab, order in [('Euler', 1), ('Taylor 2', 2)]:
    ekf_tmp.order = order
    mods[lab] = np.max(np.abs(np.linalg.eigvals(ekf_tmp.phi(x_rated)[:4, :4])))
from scipy.linalg import expm
mods['exact'] = np.max(np.abs(np.linalg.eigvals(expm(ekf_tmp.jac(x_rated)[:4, :4] * TS))))
print(f"\n=== 3. Prediction and observability ===")
print(f"flux mode, forward Euler: |1 + wb ts (-alpha + j w)| - 1 = (w wb ts)^2/2 - alpha wb ts: growth {(th_b)**2/2:.2e} - "
      f"{N['alpha']*th_b:.2e} per step at 1 pu; unstable above w = sqrt(2 alpha/(wb ts)) = {w_crit:.3f} pu")
print(f"electrical 4x4 model at rated speed and torque, largest |eigenvalue|: Euler {mods['Euler']:.6f}, Taylor 2 "
      f"{mods['Taylor 2']:.6f}, exact {mods['exact']:.6f}")


def obs_sigma_min(w_s, T_pu, n):
    """smallest singular value of the observability matrix of the model linearized around the steady state,
    written in the synchronous frame (LTI) with the time normalized by omega_b; states in pu"""
    x = steady_state(w_s, T_pu)[0][:n]
    e = EKF(PAR_NOM, n=n, q=[1] * n, r=1)
    A = e.jac(x) / WB
    Jr = np.zeros((n, n)); Jr[0, 1] = Jr[2, 3] = -1; Jr[1, 0] = Jr[3, 2] = 1
    A = A - w_s * Jr
    C = np.zeros((2, n)); C[0, 0] = C[1, 1] = 1
    O = np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)])
    return np.linalg.svd(O, compute_uv=False)[-1]


f_s = np.linspace(-0.15, 0.15, 601)
fig, ax = plt.subplots(figsize=(6.6, 2.9))
obs = {}
for (n_, T), c, ls in [((5, 0.0), 'C0', '-'), ((5, 1.0), 'C1', '-'), ((6, 0.0), 'C0', '--'), ((6, 1.0), 'C1', '--')]:
    o = np.array([obs_sigma_min(fs, T, n_) for fs in f_s])
    obs[(n_, T)] = o
    ax.semilogy(f_s * M['freq_nom'], o, ls, color=c, label=f'{n_} stati, coppia {T:.1f} pu')
ax.set_xlabel('frequenza di statore [Hz]'); ax.set_ylabel(r'$\sigma_{min}(\mathcal{O})$')
ax.legend(fontsize=7, ncol=2, loc='lower right')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_observability.pdf')); plt.close(fig)
for (n_, T), o in obs.items():
    slip = steady_state(0.0, T)[1]
    print(f"observability, {n_} states, torque {T:.1f} pu: sigma_min at 0 Hz {np.interp(0, f_s, o):.2e}, at 0.6 Hz "
          f"{np.interp(0.01, f_s, o):.2e}, at 3 Hz {np.interp(0.05, f_s, o):.2e}, at 9 Hz {np.interp(0.15, f_s, o):.2e}; "
          f"rotor speed on the unobservable line {-slip:+.5f} pu ({-slip*M['freq_nom']:+.3f} Hz)")


# ---------------------------------------------------------------------------------------------------------------
# 4. averaged time-domain simulation
# ---------------------------------------------------------------------------------------------------------------
T_DEAD = 3e-6                      # dead time of inv_im (param.inv.pwm.dead_time) [s]
VDT = T_DEAD * 4000 * UDC          # dead-time voltage error per phase: t_d f_pwm u_dc = 14.4 V (3 us, 4 kHz, 1200 V)
I_BAND = 10.0                      # linear band of the sign used by the dead-time compensation [A] (1.7 % of ibez)
# voltage given to the EKF in inv_im: u_im_scale_ekf = (2/3)^2 udc_nom/ubez times the control voltage, whereas the
# space-vector modulator (gain sqrt(3)/2 + sv_pwm_process, unit 2/3 udc) makes 1 control pu = udc/sqrt(3)
U_SCALE_LIB = (2 / 3) ** 2 / (1 / np.sqrt(3))


def simulate(scn, t_end, sensorless=False, ekf_n=6, plant_scale=None, deadtime=True, dt_comp=1.0, dt_band=None,
             lib_ekf=(), euler_ekf=False, nsub=4, t_hold=1.0, cov_order=None):
    """scn(t) -> (speed reference before the filter [pu], external load torque [pu of T_bez]).
    Control: FOC in rotor-flux coordinates with the proposed gains, flux P + feed-forward + clamped integral,
    isy = T_ref/(kT psi), current PI with decoupling and vector limit |u| <= 1 (control pu), angle advance 1.5 ts.
    Angle/flux from the current-model observer + encoder (sensorless=False) or from the EKF (sensorless=True).
    deadtime: dead-time voltage error -VDT sign(i) on each phase of the inverter; dt_comp: feed-forward compensation
    dt_comp*VDT*sat(i_ref/dt_band) added to the PWM command (0 = none, dt_band = I_BAND by default); the EKF gets
    the voltage of the current
    controller (without the compensation). lib_ekf: voltage scales of the library EKFs run in open loop on the same
    data (1.0 = voltage in machine pu, U_SCALE_LIB = scale of inv_im); euler_ekf: the proposed EKF with the Euler
    prediction also runs in open loop."""
    ps = dict(Rs=M['Rs'], Rr=M['Rr'], Lm=M['Lm'], Lr=M['Lr'], Ls=M['Ls'])
    for k, v in (plant_scale or {}).items():
        if k == 'Lm':                                   # magnetizing inductance: leakages unchanged
            dL = M['Lm'] * (v - 1)
            ps['Lm'] += dL; ps['Lr'] += dL; ps['Ls'] += dL
        else:
            ps[k] *= v
    ps['sigmaLs'] = ps['Ls'] - ps['Lm'] ** 2 / ps['Lr']
    pp, J, bfr = M['pp'], M['Jm'], M['load_friction']
    Ltot = ps['sigmaLs'] + LFI
    kr = ps['Lm'] / ps['Lr']
    Rtot = ps['Rs'] + RLFI + ps['Rr'] * kr ** 2
    ib, ub = M['ibez'], M['ubez']

    def f(x, u, Text):
        i, psi, wm = x[0], x[1], x[2].real
        we = pp * wm
        di = (u - Rtot * i + kr * (ps['Rr'] / ps['Lr'] - 1j * we) * psi) / Ltot
        dpsi = ps['Rr'] / ps['Lr'] * (ps['Lm'] * i - psi) + 1j * we * psi
        T = 1.5 * pp * kr * (np.conj(psi) * i).imag
        dwm = (T - bfr * wm - Text) / J
        return np.array([di, dpsi, dwm, we])

    g = G_PROP
    tau_f = 1 / (2 * np.pi * 10); ad = np.exp(-TS / tau_f)
    r1 = r2 = 0.0
    x = np.array([0j, 0j, 0j, 0j])
    psi_obs = 0j                                         # current-model observer (rotor frame, pu)
    xw = xpsi = 0.0; xi = 0j
    u_cmd = 0j                                           # control output of the previous step (alpha-beta, machine pu)
    u_app = 0j                                           # voltage applied in the current interval
    ekf = EKF(PAR_NOM, n=ekf_n, q=Q6 if ekf_n == 6 else Q5, r=r_meas,
              x0=[0, 0, 1e-3, 0, 0, 0][:ekf_n], p0=1e-4, cov_order=cov_order)
    libs = [(EKFLibrary(N), sc) for sc in lib_ekf]
    eul = EKF(PAR_NOM, n=6, q=Q6, r=r_meas, x0=[0, 0, 1e-3, 0, 0, 0], p0=1e-4, order=1) if euler_ekf else None
    u_pwm_cmd = 0j; u_pwm_app = 0j                      # PWM command with the dead-time compensation
    N_ = int(round(t_end / TS)); h = TS / nsub
    rec = np.zeros((N_, 26))
    band = I_BAND if dt_band is None else dt_band
    a3 = np.exp(-2j * np.pi / 3)                        # phase b = Re(i e^{-j2pi/3}), alpha-beta = 2/3 (a + b e^{j2pi/3} + c e^{-j2pi/3})
    for k in range(N_):
        t = k * TS
        w_ref_raw, Text = scn(t)
        i_s, psi_r, wm, th = x[0], x[1], x[2].real, x[3].real
        # ---- measurements at t_k: phase currents with noise and 12-bit quantization, encoder angle and speed
        ia = i_s.real + SIG_I * RNG.standard_normal(); ib_ = (i_s * a3).real + SIG_I * RNG.standard_normal()
        ia = np.round(ia / Q_ADC) * Q_ADC; ib_ = np.round(ib_ / Q_ADC) * Q_ADC
        i_meas = (ia + 1j * (ia + 2 * ib_) / np.sqrt(3)) / ib
        th_enc = th; w_enc = wm / M['omega_m_bez']
        # ---- estimators (the EKF uses the voltage applied in the last interval and the current sampled now)
        xe = ekf.step(np.array([u_app.real, u_app.imag]), np.array([i_meas.real, i_meas.imag]), hold=(t < t_hold))
        xls = [lb.step(sc * np.array([u_app.real, u_app.imag]), np.array([i_meas.real, i_meas.imag])) for lb, sc in libs]
        if eul is not None:
            xu = eul.step(np.array([u_app.real, u_app.imag]), np.array([i_meas.real, i_meas.imag]), hold=(t < t_hold))
        psi_now = psi_obs
        psi_obs = psi_now + TS * WB * N['alpha'] * (N['Lm'] * i_meas * np.exp(-1j * th_enc) - psi_now)
        if sensorless:
            psi_hat_ab = xe[2] + 1j * xe[3]
            w_hat = xe[4]
        else:
            psi_hat_ab = psi_obs * np.exp(1j * th_enc)
            w_hat = w_enc
        th_s = np.angle(psi_hat_ab) if abs(psi_hat_ab) > 1e-6 else 0.0
        psi_mag = abs(psi_hat_ab)
        is_xy = i_meas * np.exp(-1j * th_s)
        # ---- speed reference: ramp/hold profile through the 10 Hz second-order filter
        r1 = ad * r1 + (1 - ad) * w_ref_raw; r2 = ad * r2 + (1 - ad) * r1
        # ---- speed PI (output torque, pu of T_bez), clamping
        e = r2 - w_hat
        T_ref = g['kp_w'] * e + xw
        T_lim = 1.4 * KT
        T_sat = min(max(T_ref, -T_lim), T_lim)
        if T_sat == T_ref or np.sign(e) != np.sign(T_ref):
            xw += g['ki_w'] * TS * e
        # ---- flux: feed-forward + P + clamped integral; isx limits 0.05 .. 0.8 pu
        psi_ref = 1.0
        ep = psi_ref - psi_mag
        up = psi_ref + g['kp_psi'] * ep + xpsi
        ups = min(max(up, 0.05 * N['Lm']), 0.8 * N['Lm'])
        if ups == up:
            xpsi += g['ki_psi'] * TS * ep
        isx_ref = ups / N['Lm']
        isy_ref = T_sat / (KT * max(psi_mag, 0.3))
        isy_ref = min(max(isy_ref, -1.4), 1.4)
        # ---- current PI in x/y with decoupling (machine pu), vector limit |u| <= 1 control pu
        w_slip = N['alpha'] * isy_ref / max(isx_ref, 0.05)
        w_s = w_hat + w_slip
        ex = isx_ref - is_xy.real; ey = isy_ref - is_xy.imag
        kp_m = g['kp_i'] * K_UC; ki_m = g['ki_i'] * K_UC
        u_ff = -w_s * N['sigma_t'] * is_xy.imag + 1j * w_s * (N['sigma_t'] * is_xy.real + KT * psi_mag)
        u_pi = kp_m * (ex + 1j * ey) + xi
        u_xy = u_pi + u_ff
        mag = abs(u_xy) / K_UC
        if mag > 1.0:
            u_xy = u_xy / mag
        else:
            xi += ki_m * TS * (ex + 1j * ey)
        # angle advance 1.5 ts: the voltage computed now is applied during [t_k+1, t_k+2]
        th_app = th_s + 1.5 * TS * WB * w_s
        u_new = u_xy * np.exp(1j * th_app)
        # dead-time compensation from the phase-current references at the application instant
        comp = 0j
        if dt_comp > 0:
            i_ref = (isx_ref + 1j * isy_ref) * np.exp(1j * th_app) * ib
            c3 = dt_comp * VDT * np.clip(np.array([i_ref.real, (i_ref * a3).real, (i_ref * a3.conjugate()).real]) / band, -1, 1)
            comp = 2 / 3 * (c3[0] + c3[1] * a3.conjugate() + c3[2] * a3) / ub
        # ---- plant over [t_k, t_k+1] with the voltage computed at t_k-1
        u_app = u_cmd; u_pwm_app = u_pwm_cmd
        u_cmd = u_new; u_pwm_cmd = u_new + comp
        uV = u_pwm_app * ub
        if deadtime:
            iabc = [i_s.real, (i_s * a3).real, (i_s * a3.conjugate()).real]
            du = -VDT * np.array([np.sign(v) for v in iabc])
            uV = uV + 2 / 3 * (du[0] + du[1] * a3.conjugate() + du[2] * a3)
        Tm = 0.0
        for j in range(nsub):
            k1 = f(x, uV, Text * M['T_bez']); k2 = f(x + h / 2 * k1, uV, Text * M['T_bez'])
            k3 = f(x + h / 2 * k2, uV, Text * M['T_bez']); k4 = f(x + h * k3, uV, Text * M['T_bez'])
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        T_e = 1.5 * pp * kr * (np.conj(psi_r) * i_s).imag / M['T_bez']
        th_true = np.angle(psi_r) if abs(psi_r) > 1e-9 else 0.0
        err_th = np.angle(np.exp(1j * (th_s - th_true)))
        row = [t, r2, w_enc, w_hat, abs(psi_r) / M['psi_bez'], psi_mag, err_th, isx_ref, isy_ref, is_xy.real,
               is_xy.imag, abs(i_s) / np.sqrt(2), T_e, Text + M['load_friction'] * wm / M['T_bez'], xe[4],
               xe[5] if ekf_n == 6 else np.nan, np.angle(np.exp(1j * (np.angle(xe[2] + 1j * xe[3]) - th_true))),
               np.nan, np.nan, abs(u_app) / K_UC, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        if eul is not None:
            row[20] = xu[4]
            row[21] = np.angle(np.exp(1j * (np.arctan2(xu[3], xu[2]) - th_true)))
        for j, xl in enumerate(xls):
            row[[17, 22][j]] = xl[4]
            row[[18, 23][j]] = np.angle(np.exp(1j * (np.arctan2(xl[3], xl[2]) - th_true)))
            row[[24, 25][j]] = np.hypot(xl[2], xl[3])
        rec[k] = row
    cols = ['t', 'w_ref', 'w', 'w_hat', 'psi', 'psi_hat', 'err_th', 'isx_ref', 'isy_ref', 'isx', 'isy', 'is_rms',
            'Te', 'Tload', 'w_ekf', 'TL_ekf', 'err_th_ekf', 'w_lib', 'err_th_lib', 'u', 'w_eul', 'err_th_eul',
            'w_lib2', 'err_th_lib2', 'psi_lib', 'psi_lib2']
    return {c: rec[:, i] for i, c in enumerate(cols)}


def profile_R1(t):
    """magnetization 0-1 s, ramp to rated speed in 1 s, external load 0.2 pu at 2.5 s (with the friction of the
    model, 0.63 pu at rated speed, the total is 0.83 pu = rated torque)"""
    w = 0.0 if t < 1.0 else min(t - 1.0, 1.0)
    T = 0.2 if t >= 2.5 else 0.0
    return w, T


def profile_R2(t):
    """magnetization, ramp to rated speed, external load 0.2 pu at 2.5 s (rated torque with the friction), ramp down
    to 5 % speed, external load ramped from 0.2 to 0.8 pu in 1 s from 5.5 s (rated torque at 5 % speed)"""
    if t < 1.0:
        w = 0.0
    elif t < 2.0:
        w = t - 1.0
    elif t < 3.5:
        w = 1.0
    elif t < 4.45:
        w = 1.0 - (t - 3.5)
    else:
        w = 0.05
    T = (0.2 if t >= 2.5 else 0.0) + 0.6 * min(max(t - 5.5, 0.0), 1.0)
    return w, T


def profile_rev(t):
    """slow reversal through zero speed with an active load of 0.2 pu: +5 % from 1 s, load from 1.5 s, reference
    ramp at -0.05 pu/s from 2.5 s to -5 % (the stator frequency crosses zero at about 3.5 s)"""
    w = 0.0 if t < 1.0 else (0.05 if t < 2.5 else max(0.05 - 0.05 * (t - 2.5), -0.05))
    T = 0.2 if t >= 1.5 else 0.0
    return w, T


def profile_low(t):
    """5 % speed from 1 s, load 0.2 pu at 1.5 s, ramped to 0.8 pu between 3.0 s and 4.0 s"""
    w = 0.0 if t < 1.0 else 0.05
    T = (0.2 if t >= 1.5 else 0.0) + 0.6 * min(max(t - 3.0, 0.0), 1.0)
    return w, T


def at(d, key, t0):
    return np.interp(t0, d['t'], d[key])




def win(d, key, t0, t1, fun=np.mean):
    mk = (d['t'] >= t0) & (d['t'] < t1)
    return fun(d[key][mk])


def report(d, name, windows):
    for t0, t1, lab in windows:
        print(f"[{name}] {lab}: speed {win(d, 'w', t0, t1):+.4f} pu, estimate {win(d, 'w_hat', t0, t1):+.4f}, error "
              f"{win(d, 'w_hat', t0, t1) - win(d, 'w', t0, t1):+.4f} pu; flux {win(d, 'psi', t0, t1):.3f} pu (estimate "
              f"{win(d, 'psi_hat', t0, t1):.3f}); angle error {np.degrees(win(d, 'err_th', t0, t1)):+.2f} deg (rms "
              f"{np.degrees(win(d, 'err_th', t0, t1, np.std)):.2f}); torque {win(d, 'Te', t0, t1):.3f} pu, load "
              f"{win(d, 'Tload', t0, t1):.3f} pu, EKF load {win(d, 'TL_ekf', t0, t1):.3f} pu; current "
              f"{win(d, 'is_rms', t0, t1):.0f} A rms")
    mk = d['t'] > 1.0
    k = np.argmax(np.abs(d['w_hat'] - d['w']) * mk)
    print(f"[{name}] max speed estimation error after 1 s {d['w_hat'][k] - d['w'][k]:+.4f} pu at {d['t'][k]:.2f} s; max "
          f"angle error after 1.2 s {np.degrees(np.max(np.abs(d['err_th'][d['t'] > 1.2]))):.1f} deg; min speed after "
          f"4.5 s {np.min(d['w'][d['t'] > 4.5]) if d['t'][-1] > 4.5 else np.nan:+.4f} pu; peak current "
          f"{np.max(d['is_rms']):.0f} A rms")


def plot_run(d, fname, zoom_t0):
    fig, ax = plt.subplots(4, 1, figsize=(6.6, 8.4), sharex=True)
    ax[0].plot(d['t'], d['w_ref'], 'k--', lw=0.8, label='riferimento filtrato')
    ax[0].plot(d['t'], d['w'], 'k', label='velocità vera')
    ax[0].plot(d['t'], d['w_hat'], 'C0', lw=0.8, label='stima EKF')
    ax[0].set_ylabel('velocità [pu]'); ax[0].legend(fontsize=7, loc='upper right')
    axz = ax[0].inset_axes([0.62, 0.12, 0.25, 0.4])
    mk = d['t'] > zoom_t0
    axz.plot(d['t'][mk], d['w'][mk], 'k'); axz.plot(d['t'][mk], d['w_hat'][mk], 'C0', lw=0.6)
    axz.set_ylim(0.0, 0.1); axz.tick_params(labelsize=6); axz.set_title('5 %', fontsize=6)
    ax[1].plot(d['t'], d['psi'], 'k', label='flusso rotorico vero')
    ax[1].plot(d['t'], d['psi_hat'], 'C0', lw=0.8, label='stima EKF')
    ax[1].plot(d['t'], np.degrees(d['err_th']) / 10, 'C3', lw=0.6, label='errore angolo / 10 [deg]')
    ax[1].set_ylabel('[pu]'); ax[1].legend(fontsize=7, loc='lower right'); ax[1].set_ylim(-0.6, 1.2)
    ax[2].plot(d['t'], d['Tload'], 'k', label='coppia di carico vera (attrito + esterna)')
    ax[2].plot(d['t'], d['TL_ekf'], 'C0', lw=0.8, label='stima EKF')
    ax[2].plot(d['t'], d['Te'], 'C2', lw=0.7, label='coppia elettromagnetica')
    ax[2].set_ylabel('coppia [pu]'); ax[2].legend(fontsize=7, loc='upper right'); ax[2].set_ylim(-0.05, 1.3)
    ax[3].plot(d['t'], d['is_rms'], 'k', label='|i| [A rms]')
    ax[3].plot(d['t'], d['u'] * 400, 'C1', lw=0.8, label='|u| x 400 [pu di udc/sqrt(3)]')
    ax[3].set_ylabel('A rms'); ax[3].set_xlabel('tempo [s]'); ax[3].legend(fontsize=7, loc='upper right')
    ax[3].set_ylim(0, 580)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)


if __name__ == '__main__':
    print('\n=== 4. Simulations ===')
    # R1: encoder FOC, the estimators run in open loop on the same data
    d1 = simulate(profile_R1, 3.5, sensorless=False, ekf_n=6, lib_ekf=(1.0, U_SCALE_LIB), euler_ekf=True)
    m_ss = d1['t'] > 3.0
    ramp = (d1['t'] > 1.2) & (d1['t'] < 2.0)
    for lab, kw, kth in [('proposed EKF, Taylor 2', 'w_ekf', 'err_th_ekf'), ('proposed EKF, Euler', 'w_eul', 'err_th_eul'),
                         ('library EKF, voltage in machine pu', 'w_lib', 'err_th_lib'),
                         (f'library EKF, voltage x {U_SCALE_LIB:.3f} (inv_im)', 'w_lib2', 'err_th_lib2')]:
        ew = d1[kw] - d1['w']
        print(f"R1 {lab}: speed error rms during the ramp {np.sqrt(np.mean(ew[ramp] ** 2)):.4f} pu; at rated speed and "
              f"torque mean {np.mean(ew[m_ss]):+.4f} pu, rms {np.std(ew[m_ss]):.4f} pu; flux angle error mean "
              f"{np.degrees(np.mean(d1[kth][m_ss])):+.2f} deg, rms {np.degrees(np.std(d1[kth][m_ss])):.2f} deg")
    print(f"R1 flux magnitude at rated speed: actual {np.mean(d1['psi'][m_ss]):.3f} pu, library EKF {np.mean(d1['psi_lib'][m_ss]):.3f} pu "
          f"(voltage in machine pu), {np.mean(d1['psi_lib2'][m_ss]):.3f} pu (voltage x {U_SCALE_LIB:.3f})")
    print(f"R1 load torque at rated speed: actual (friction + external) {np.mean(d1['Tload'][m_ss]):.3f} pu, proposed EKF "
          f"{np.mean(d1['TL_ekf'][m_ss]):.3f} pu")
    fig, ax = plt.subplots(3, 1, figsize=(6.6, 6.8), sharex=True)
    ax[0].plot(d1['t'], d1['w'], 'k', label='velocità (encoder)')
    ax[0].plot(d1['t'], d1['w_ekf'], 'C0', label='EKF proposto (Taylor 2)')
    ax[0].plot(d1['t'], d1['w_eul'], 'C1', lw=0.8, label='EKF proposto (Eulero)')
    ax[0].plot(d1['t'], d1['w_lib'], 'C3', lw=0.6, alpha=0.8, label='EKF della libreria')
    ax[0].set_ylabel('velocità [pu]'); ax[0].legend(fontsize=7, loc='lower right'); ax[0].set_ylim(-0.3, 1.3)
    ax[1].plot(d1['t'], np.degrees(d1['err_th_ekf']), 'C0', label='EKF proposto (Taylor 2)')
    ax[1].plot(d1['t'], np.degrees(d1['err_th_eul']), 'C1', lw=0.8, label='EKF proposto (Eulero)')
    ax[1].plot(d1['t'], np.degrees(d1['err_th_lib']), 'C3', lw=0.6, alpha=0.8, label='EKF della libreria')
    ax[1].set_ylabel('errore angolo flusso [deg]'); ax[1].set_ylim(-30, 30); ax[1].legend(fontsize=7, ncol=3)
    ax[2].plot(d1['t'], d1['Tload'], 'k', label='coppia di carico (attrito + esterna)')
    ax[2].plot(d1['t'], d1['TL_ekf'], 'C0', label='stima EKF proposto')
    ax[2].plot(d1['t'], d1['Te'], 'C2', lw=0.7, label='coppia elettromagnetica')
    ax[2].set_ylabel('coppia [pu]'); ax[2].set_xlabel('tempo [s]'); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_ekf_openloop.pdf')); plt.close(fig)

    # R2: sensorless, nominal parameters, dead time compensated; R2b: without compensation; R3: Rs, Rr +30 %, 80 %
    w2 = [(3.0, 3.5, 'rated speed and torque'), (5.0, 5.5, '5 % speed, 0.23 pu'), (7.0, 7.5, '5 % speed, rated torque')]
    d2 = simulate(profile_R2, 7.5, sensorless=True, dt_comp=1.0)
    report(d2, 'R2 nominal', w2)
    d2b = simulate(profile_R2, 7.5, sensorless=True, dt_comp=0.0)
    report(d2b, 'R2b no dead-time compensation', w2)
    d3 = simulate(profile_R2, 7.5, sensorless=True, dt_comp=0.9, plant_scale=dict(Rr=1.3, Rs=1.3))
    report(d3, 'R3 Rs, Rr +30 %, compensation 90 %', w2)
    d3b = simulate(profile_R2, 7.5, sensorless=True, dt_comp=0.8, plant_scale=dict(Rr=1.3, Rs=1.3))
    report(d3b, 'R3b Rs, Rr +30 %, compensation 80 %', w2)
    plot_run(d2, 'fig_sensorless_run.pdf', 4.6)
    fig, ax = plt.subplots(3, 1, figsize=(6.6, 6.4), sharex=True)
    mk = d3['t'] >= 3.0
    ax[0].plot(d3['t'][mk], d3['w_ref'][mk], 'k--', lw=0.8, label='riferimento filtrato')
    for d, c, lab in [(d3, 'C0', 'compensazione 90 %'), (d3b, 'C3', 'compensazione 80 %')]:
        ax[0].plot(d['t'][mk], d['w'][mk], color=c, label=f'velocità vera, {lab}')
        ax[0].plot(d['t'][mk], d['w_hat'][mk], ':', color=c, lw=1.0, label=f'stima, {lab}')
        ax[1].plot(d['t'][mk], np.degrees(d['err_th'][mk]), color=c, lw=0.7, label=lab)
        ax[2].plot(d['t'][mk], d['psi'][mk], color=c, lw=0.8, label=f'flusso vero, {lab}')
        ax[2].plot(d['t'][mk], d['psi_hat'][mk], ':', color=c, lw=1.0, label=f'stima, {lab}')
    ax[0].set_ylabel('velocità [pu]'); ax[0].set_ylim(-0.3, 1.1); ax[0].legend(fontsize=6, loc='upper right')
    axz = ax[0].inset_axes([0.36, 0.32, 0.28, 0.58])
    mz = d3['t'] > 4.6
    for d, c in [(d3, 'C0'), (d3b, 'C3')]:
        axz.plot(d['t'][mz], d['w'][mz], color=c, lw=0.8); axz.plot(d['t'][mz], d['w_hat'][mz], ':', color=c, lw=0.8)
    axz.set_ylim(-0.1, 0.1); axz.tick_params(labelsize=6); axz.set_title('dopo 4,6 s', fontsize=6)
    ax[1].set_ylabel('errore angolo [deg]'); ax[1].set_ylim(-60, 60); ax[1].legend(fontsize=7)
    ax[2].set_ylabel('flusso [pu]'); ax[2].set_ylim(0.0, 1.6); ax[2].legend(fontsize=6, ncol=2, loc='lower left')
    ax[2].set_xlabel('tempo [s]')
    for a_ in ax:
        a_.axvspan(5.5, 6.5, color='0.9', lw=0)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_sensitivity.pdf')); plt.close(fig)

    # R4: slow reversal through zero speed with an active load
    w4 = [(2.0, 2.5, '+5 %'), (3.3, 3.7, 'zero crossing'), (5.5, 6.0, '-5 %')]
    d4 = simulate(profile_rev, 6.0, sensorless=True, dt_comp=1.0)
    report(d4, 'R4 nominal', w4)
    d4b = simulate(profile_rev, 6.0, sensorless=True, dt_comp=0.9, plant_scale=dict(Rr=1.3, Rs=1.3))
    report(d4b, 'R4b Rs, Rr +30 %, compensation 90 %', w4)
    fig, ax = plt.subplots(3, 1, figsize=(6.6, 6.4), sharex=True)
    ax[0].plot(d4['t'], d4['w_ref'], 'k--', lw=0.8, label='riferimento filtrato')
    for d, c, lab in [(d4, 'C0', 'nominale'), (d4b, 'C3', 'Rs, Rr +30 %, compensazione 90 %')]:
        ax[0].plot(d['t'], d['w'], color=c, label=f'velocità vera, {lab}')
        ax[0].plot(d['t'], d['w_hat'], ':', color=c, lw=0.9, label=f'stima, {lab}')
        ax[1].plot(d['t'], np.degrees(d['err_th']), color=c, lw=0.8, label=lab)
        ax[2].plot(d['t'], d['psi'], color=c, lw=0.8, label=f'flusso vero, {lab}')
    ax[0].set_ylabel('velocità [pu]'); ax[0].legend(fontsize=6, loc='lower left'); ax[0].set_ylim(-0.08, 0.08)
    ax[1].set_ylabel('errore angolo [deg]'); ax[1].legend(fontsize=7); ax[1].set_ylim(-60, 60)
    ax[2].set_ylabel('flusso [pu]'); ax[2].set_xlabel('tempo [s]'); ax[2].legend(fontsize=7); ax[2].set_ylim(0.6, 1.1)
    for a_ in ax:
        a_.axvspan(3.35, 3.65, color='0.85', lw=0)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_reversal.pdf')); plt.close(fig)

    # sensitivity at 5 % speed: one error at a time
    print('\nsensitivity at 5 % speed (profile_low): speed error [pu] and flux angle error [deg], light load 0.23 pu / '
          'rated torque 0.83 pu')
    cases = [('nominal', {}, 1.0, None), ('dead-time compensation 85 %', {}, 0.85, None),
             ('dead-time compensation 90 %', {}, 0.9, None), ('dead-time compensation 110 %', {}, 1.1, None),
             ('dead-time compensation 115 %', {}, 1.15, None), ('compensation band 29 A (5 %)', {}, 1.0, 0.05 * M['ibez']),
             ('Rs +30 %', dict(Rs=1.3), 1.0, None), ('Rs -30 %', dict(Rs=0.7), 1.0, None),
             ('Rr +30 %', dict(Rr=1.3), 1.0, None), ('Rr -30 %', dict(Rr=0.7), 1.0, None),
             ('Lm -10 %', dict(Lm=0.9), 1.0, None), ('Lm +10 %', dict(Lm=1.1), 1.0, None)]
    for lab, sc, cmp_, bnd in cases:
        d = simulate(profile_low, 5.5, sensorless=True, dt_comp=cmp_, dt_band=bnd, plant_scale=sc)
        res = []
        for t0, t1 in [(2.5, 3.0), (5.0, 5.5)]:
            res.append((win(d, 'w', t0, t1), win(d, 'w_hat', t0, t1) - win(d, 'w', t0, t1),
                        np.degrees(win(d, 'err_th', t0, t1)), win(d, 'psi', t0, t1)))
        print(f"  {lab:30s} light: speed {res[0][0]:+.4f}, error {res[0][1]:+.4f}, angle {res[0][2]:+6.2f}, flux "
              f"{res[0][3]:.3f} | rated: speed {res[1][0]:+.4f}, error {res[1][1]:+.4f}, angle {res[1][2]:+6.2f}, "
              f"flux {res[1][3]:.3f}")

    # checks quoted in the note (run last, so that the noise realizations of the runs above do not change)
    print('\nchecks')
    d = simulate(profile_R1, 3.5, sensorless=False, cov_order=1)
    m_ss = d['t'] > 3.0
    print(f"R1 with the second-order state prediction and the first-order covariance: angle error "
          f"{np.degrees(np.mean(d['err_th_ekf'][m_ss])):+.2f} deg, speed error {np.mean(d['w_ekf'][m_ss] - d['w'][m_ss]):+.4f} pu")
    d = simulate(profile_R2, 7.5, sensorless=True, cov_order=1)
    report(d, 'R2 with first-order covariance', w2)
    d = simulate(profile_rev, 6.0, sensorless=True, deadtime=False, dt_comp=0.0)
    report(d, 'R4c without dead time', w4)
    d = simulate(profile_rev, 6.0, sensorless=True, dt_comp=1.0, dt_band=3.0)
    report(d, 'R4d compensation band 3 A', w4)
