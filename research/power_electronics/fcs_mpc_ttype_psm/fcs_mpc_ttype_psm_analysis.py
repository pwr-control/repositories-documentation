#!/usr/bin/env python3
"""Companion script of the note fcs_mpc_ttype_psm.tex (finite-set predictive current control of the T-type
inverter of the wind generator model mpc_ttype_psm of modelization-and-control: analysis and fixes).

Run: python3 fcs_mpc_ttype_psm_analysis.py [output folder]
     -> fcs_mpc_ttype_psm_analysis_output.txt (every number of the note) and the PDF figures of the note in
        <output folder>/figures (default output folder: the folder of this script). About 4 minutes.
     python3 with numpy and matplotlib.

What it computes (the same numbers and formulas as init_model.m and private/*.m of the model)
 1. Per-unit chain of the fixed model: voltage of the predictive control and of the EKF in pu of ubez,
    inductance and resistance of the prediction and of the EKF (q axis of the machine + output inductor),
    neutral-point gain, operating point at rated speed.
 2. The predictive current loop seen as a deadbeat loop, before and after the fix (closed-loop poles).
 3. A switched simulation (vector level) of the whole chain, 1.5 s:
    shaft speed of fixed_speed_setting (step to rpm_sim at 0.025 s through a 0.1 s lag), torque curve with the
    1 Hz speed filter and the saturation, current reference on the q axis (id = 0) at the control angle,
    current and DC voltages sampled, quantized (1/2^11 pu) and one control step late, control clock 1.001
    (block global timing/time_inv1), encoder-sensorless switch.
    Fixed model: predictive control = class MPCNew (line-by-line copy of the MATLAB Function mpc_ttype_gate,
    checked against an Octave run of the MATLAB code), EKF block with L = Lq + LFi, R = Rs + RLFi, no A6 path
    in the state prediction, voltage = output ualpha, ubeta of the predictive control (vector of two steps
    before, pu of ubez) with gains 1, speed output from the double integrator observer on the EKF angle.
    Model before the fix (case A): the old mpc_ttype_gate (voltage in pu of udc_nom, no delay compensation,
    k_dc = 0), EKF with Ls, the A6 path without omega_bez, voltage 3*u*udc_nom/(sqrt(3)*ubez) of this step.
    Plant: PMSM in the rotor frame (Ld, Lq of one system + 230 uH output inductor), T-type bridge with ideal
    switches, split DC link 2 x 14.4 mF with 2 mOhm ESR each (block dclink), battery = open-circuit voltage at
    soc 0.85 + R0 + R1//C1 of the init. Dead time of the fixed gate generator: every turn-on is delayed by
    dead_time, so at a level change the leg keeps the old level for dead_time when the current flows the
    way that needs the switch that turns on (to a higher level with i > 0, to a lower level with i < 0), and
    takes the new level at once when a diode takes the current. Case A has no dead time.
 4. Gate timing of the T-type legs before and after the fix (filter time constant tc, CMOS interlock between Q1
    and Q4, new turn-on delay on the four gates), on the tc grid of the Simscape local solver.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- line-by-line copy of the MATLAB Function mpc_ttype_gate (fixed model) ----
STATES = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], float)


class MPCNew:
    def __init__(self, sw_mode='count'):
        self.sw_mode = sw_mode
        self.u_last = np.zeros(2)
        self.u_last2 = np.zeros(2)
        self.s_last = np.zeros(3)

    def step(self, vdc1, vdc2, ialpha_ref, ibeta_ref, ialpha, ibeta, omega_hat, psi_alpha_hat, psi_beta_hat,
             L_norm, R_norm, u_scale, np_gain, k_kalman, k_dc, k_sw, ts_pu):
        sqrt3_2 = 0.866025403784439
        inv_sqrt3 = 0.577350269189626
        g = ts_pu / L_norm
        ealpha = -omega_hat * psi_beta_hat
        ebeta = omega_hat * psi_alpha_hat
        ia_m = ialpha
        ib_m = -0.5 * ialpha + sqrt3_2 * ibeta
        ic_m = -0.5 * ialpha - sqrt3_2 * ibeta
        s = self.s_last
        inp_last = float(s[0] == 0) * ia_m + float(s[1] == 0) * ib_m + float(s[2] == 0) * ic_m
        ialpha0 = ialpha + g * (self.u_last[0] - R_norm * ialpha - ealpha)
        ibeta0 = ibeta + g * (self.u_last[1] - R_norm * ibeta - ebeta)
        dv0 = (vdc1 - vdc2) + np_gain * inp_last * ts_pu
        ia0 = ialpha0
        ib0 = -0.5 * ialpha0 + sqrt3_2 * ibeta0
        ic0 = -0.5 * ialpha0 - sqrt3_2 * ibeta0
        minJ = 1e18
        best = 13
        ua_best = 0.0
        ub_best = 0.0
        for n in range(27):
            sa, sb, sc = STATES[n]
            va = float(sa == 1) * vdc1 - float(sa == -1) * vdc2
            vb = float(sb == 1) * vdc1 - float(sb == -1) * vdc2
            vc = float(sc == 1) * vdc1 - float(sc == -1) * vdc2
            ua = u_scale * (va - 0.5 * vb - 0.5 * vc) * 2 / 3
            ub = u_scale * (vb - vc) * inv_sqrt3
            ia_p = ialpha0 + g * (ua - R_norm * ialpha0 - ealpha)
            ib_p = ibeta0 + g * (ub - R_norm * ibeta0 - ebeta)
            inp = float(sa == 0) * ia0 + float(sb == 0) * ib0 + float(sc == 0) * ic0
            dv = dv0 + np_gain * inp * ts_pu
            if self.sw_mode == 'count':
                nsw = abs(sa - self.s_last[0]) + abs(sb - self.s_last[1]) + abs(sc - self.s_last[2])
            else:
                nsw = (ua - self.u_last[0]) ** 2 + (ub - self.u_last[1]) ** 2
            J = k_kalman * ((ialpha_ref - ia_p) ** 2 + (ibeta_ref - ib_p) ** 2) + k_dc * dv ** 2 + k_sw * nsw
            if J < minJ:
                minJ = J
                best = n
                ua_best = ua
                ub_best = ub
        ualpha, ubeta = self.u_last2[0], self.u_last2[1]
        self.u_last2 = self.u_last.copy()
        self.u_last = np.array([ua_best, ub_best])
        self.s_last = STATES[best].copy()
        st = STATES[best]
        gates = []
        for x in st:
            gates += [15.0 * (x == 1), 15.0 * (x >= 0), 15.0 * (x <= 0), 15.0 * (x == -1)]
        return np.array(gates), ualpha, ubeta, best


OUTDIR = sys.argv[1] if len(sys.argv) > 1 else HERE
FIGDIR = os.path.join(OUTDIR, 'figures')
os.makedirs(FIGDIR, exist_ok=True)
OUT = open(os.path.join(OUTDIR, 'fcs_mpc_ttype_psm_analysis_output.txt'), 'w')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.write(s + '\n')


# -------------------------------------------------------------------------------------------------------------
# parameters (the same numbers and formulas as init_model.m and private/*.m)
# -------------------------------------------------------------------------------------------------------------
fpwm = 4e3
ts = 1 / fpwm / 2                       # 125 us: control step used by the constants
clock_gain = 1.001                      # global timing/time_inv1: the control runs at 8008 Hz
Tc = ts / clock_gain                    # real control period
tc = ts / 200                           # 0.625 us: Simscape local solver step, gate filter time constant
td = 3e-6                               # dead time (param.time.dead_time)

# machine: private/psm_machine_param.m, WindGen, one of the six systems
n_sys, pwr_nom, i_nom, rpm_nom, poles, eta = 6, 1600e3, 2062.0, 17.8, 104, 96.0
Ld, Lq = 1.04e-3, 1.26e-3
omega_m_bez = rpm_nom / 60 * 2 * np.pi
torque_nom = pwr_nom / omega_m_bez
Rs = (pwr_nom / 3) * (1 - eta / 100) / i_nom ** 2
Rs_m, Ld_m, Lq_m, Ls_m = Rs * n_sys, Ld * n_sys, Lq * n_sys, (Ld + Lq) / 2 * n_sys
pp = poles / 2
ibez = i_nom / n_sys * np.sqrt(2)
tau_bez = torque_nom / n_sys
psi_m = tau_bez / (1.5 * pp * ibez)
omega_bez = omega_m_bez * pp
ubez = psi_m * omega_bez
Xbez = ubez / ibez
Lbez = Xbez / omega_bez
Rs_n, Ld_n, Lq_n, Ls_n = Rs_m / Xbez, Ld_m / Lbez, Lq_m / Lbez, Ls_m / Lbez
P_bez = tau_bez * omega_m_bez

# inverter (private/inv_hw_param.m) and DC link block
udc_nom, LFi = 1070.0, 230e-6
RLFi = 157 * 0.05 * LFi
C_dc = 14.4e-3                          # CFi1 = CFi2
r_esr = 2e-3                            # series resistance of each capacitor in the dclink library block
L_dt, L_qt, R_t = Ld_m + LFi, Lq_m + LFi, Rs_m + RLFi

# battery (private/battery_param.m + legacy changes R0/2, R1/2, C1/50): 1200 V, 250 kW, 1C capacity, soc 0.85
ts_dab = 1 / (3 * fpwm) / 2
n_cells = int(np.floor(1200 / 3.6) - 1)
I_bat_nom = 250e3 / 1200
R0_b = 0.5 * 250e3 * (1 - 0.95) / I_bat_nom ** 2 / 2
R1_b = 0.5 * 250e3 * (1 - 0.95) / I_bat_nom ** 2 / 2
C1_b = 0.5 / 50
soc = 0.85
ocv_cell = -1.031 * np.exp(-35 * soc) + 3.485 + 0.2156 * soc - 0.05 * np.log(1 - soc + ts_dab)
U_ocv = n_cells * ocv_cell

# control (init_model.m, private/psm_ctrl_param.m)
q_adc = 1 / 2 ** 11
rpm_base = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 17.8, 18.0, 18.2])
torque_base = 1e3 / n_sys * np.array([50, 75, 100, 125, 150, 175, 200, 225, 275, 350, 450, 600, 750, 885, 885])
g0_1Hz = 1 * ts * 2 * np.pi
jm_ekf = 1e6
Q_ekf = ts * 0.1 * np.diag([ts ** 2, ts ** 2, 0.5, 0.5, 1, 1])
R_ekf = ts * 1.0 * np.eye(2)
S3_2 = np.sqrt(3) / 2
# fixed model: param.psm.ctrl.mpc.* (the EKF uses the same L_norm and R_norm)
MPC = dict(L_norm=L_qt / Lbez, R_norm=R_t / Xbez, u_scale=udc_nom / ubez,
           np_gain=ibez / (C_dc * udc_nom * omega_bez), k_kalman=1.0, k_dc=1.0, k_sw=1e-5, ts_pu=ts * omega_bez)
# model before the fix
ts_pu = ts * omega_bez
u_scale_ekf_old = udc_nom / np.sqrt(3) / ubez   # u_psm_scale_ekf before the fix (EKF input 3*u*u_psm_scale_ekf)
UA_UNIT = STATES[:, 0] - 0.5 * STATES[:, 1] - 0.5 * STATES[:, 2]
UB_UNIT = S3_2 * STATES[:, 1] - S3_2 * STATES[:, 2]


def lookup_linear(x, xp, fp):
    """1-D Lookup Table with linear interpolation and linear extrapolation (Simulink default)."""
    if x <= xp[0]:
        return fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    if x >= xp[-1]:
        return fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return float(np.interp(x, xp, fp))


def torque_pu(i_d, i_q):
    """Electromagnetic torque in pu of tau_bez (currents in pu of ibez)."""
    return i_q + (Ld_n - Lq_n) * i_d * i_q


log('=' * 110)
log('mpc_ttype_psm design check after the control fixes (numbers of init_model.m)')
log('=' * 110)

# -------------------------------------------------------------------------------------------------------------
# 1. per unit chain
# -------------------------------------------------------------------------------------------------------------
log('\n1. Per-unit chain (one of the six systems)')
log(f'   machine: ubez = {ubez:.1f} V peak, ibez = {ibez:.1f} A peak, tau_bez = {tau_bez/1e3:.2f} kNm, '
    f'omega_bez = {omega_bez:.2f} rad/s ({omega_bez/2/np.pi:.2f} Hz), 1 pu power = {P_bez/1e3:.1f} kW')
log(f'   Rs = {Rs_m*1e3:.2f} mOhm, Ld = {Ld_m*1e3:.2f} mH ({Ld_n:.4f} pu), Lq = {Lq_m*1e3:.2f} mH ({Lq_n:.4f} pu), '
    f'Ls = {Ls_n:.4f} pu; with the 230 uH inductor the bridge sees {L_dt/Lbez:.4f} pu (d) and {L_qt/Lbez:.4f} pu (q)')
log(f'   battery: {n_cells} cells, open circuit {U_ocv:.0f} V at soc {soc} ({U_ocv/udc_nom:.3f} pu of udc_nom '
    f'{udc_nom:.0f} V), R0 = {R0_b*1e3:.0f} mOhm, R1 = {R1_b*1e3:.0f} mOhm, C1 = {C1_b*1e3:.0f} mF')
log('   fixed predictive control: state voltages from vdc1 (P) and vdc2 (N) in pu of udc_nom, times '
    f'u_scale = udc_nom/ubez = {MPC["u_scale"]:.4f} -> pu of ubez;')
log(f'   prediction and EKF with L_norm = (Lq + LFi)/Lbez = {MPC["L_norm"]:.4f} pu and R_norm = (Rs + RLFi)/Xbez = '
    f'{MPC["R_norm"]:.4f} pu (before: Ls_norm = {Ls_n:.4f}, Rs_norm = {Rs_n:.4f});')
log(f'   neutral point: np_gain = ibez/(C*udc_nom*omega_bez) = {MPC["np_gain"]:.4f} (before: 1/1e-3 = 1000), k_dc = 1;')
log(f'   ts_pu = ts*omega_bez = {MPC["ts_pu"]:.5f}; EKF voltage = output of the predictive control (pu of ubez), gains 1 '
    f'(before: 3*u*{u_scale_ekf_old:.4f} = {3*u_scale_ekf_old/(udc_nom/ubez):.3f} x the true voltage).')
T_curve = lookup_linear(rpm_nom, rpm_base, torque_base)
iq_rated = -T_curve / tau_bez
ud_ss = -omega_bez * L_qt * iq_rated * ibez
uq_ss = R_t * iq_rated * ibez + omega_bez * psi_m
u_ss = np.hypot(ud_ss, uq_ss)
log(f'   rated speed: torque curve {T_curve/1e3:.1f} kNm per system -> iq_ref = {iq_rated:.3f} pu (id = 0), bridge voltage '
    f'{u_ss:.0f} V peak = {u_ss/ubez:.2f} pu of ubez = {u_ss/(U_ocv/np.sqrt(3)):.2f} of the linear limit udc/sqrt(3)')
log('   With L_norm = Lq + LFi the flux of the EKF is the "active flux" psi_m + (Ld - Lq)*id: it lies on the d axis for')
log('   any id, so the EKF angle has no saliency error (with Ls it lagged by about atan((Lq_t - Ls)*iq/psi_m)).')
log(f'   Saliency error of the EKF before the fix at rated torque: {np.degrees(np.arctan((L_qt/Lbez - Ls_n)*iq_rated)):.1f} deg')

# -------------------------------------------------------------------------------------------------------------
# 2. the predictive loop as a deadbeat loop
# -------------------------------------------------------------------------------------------------------------
log('\n2. Predictive current loop seen as a deadbeat loop (no vector limit, one axis)')
log('   g = (true voltage / predicted voltage)*(L_prediction / L_true).')
log('   Before: no delay compensation -> e(k+1) = e(k) - g*e(k-1), poles z^2 - z + g = 0.')
for name, g_scale, Ltrue in (('before, q axis', udc_nom / ubez, L_qt / Lbez),
                             ('before, d axis', udc_nom / ubez, L_dt / Lbez)):
    g = g_scale * Ls_n / Ltrue
    z = np.roots([1, -1, g])
    log(f'   {name:16s}: g = {g:.3f}, |z| = {np.abs(z[0]):.3f} '
        f'({"unstable" if np.abs(z[0]) > 1 else "stable"}, oscillation at {abs(np.angle(z[0]))/(2*np.pi*Tc):.0f} Hz)')
log('   With g = 1 (right voltage base, no delay compensation) |z| = 1: undamped oscillation at f_s/6.')
g1 = np.roots([1, -1, 1.0])
log(f'   check, g = 1  : |z| = {np.abs(g1[0]):.3f}, oscillation at {abs(np.angle(g1[0]))/(2*np.pi*Tc):.0f} Hz')
log('   After: delay compensated (current of now predicted with the vector of the last step)')
log('   -> e(k+1) = (1 - g)*e(k-1), poles z^2 = 1 - g: the error is multiplied by 1 - g every two steps.')
for name, Ltrue in (('after, q axis', L_qt / Lbez), ('after, d axis', L_dt / Lbez)):
    g = MPC['L_norm'] / Ltrue
    log(f'   {name:16s}: g = {g:.3f}, 1 - g = {1-g:+.3f}, |z| = {np.sqrt(abs(1-g)):.3f} (deadbeat for g = 1)')

# -------------------------------------------------------------------------------------------------------------
# 3. switched simulation
# -------------------------------------------------------------------------------------------------------------


def ekf_matrices(rs, ls, jm):
    A1 = np.zeros((6, 6)); A1[0, 0] = A1[1, 1] = -rs / ls; A1[4, 5] = -1 / jm
    A2 = np.zeros((6, 6)); A2[4, 3] = -1 / jm
    A3 = np.zeros((6, 6)); A3[4, 2] = 1 / jm
    A4 = np.zeros((6, 6)); A4[1, 4] = -1 / ls; A4[3, 4] = 1; A4[4, 1] = 1 / jm
    A5 = np.zeros((6, 6)); A5[0, 4] = 1 / ls; A5[2, 4] = -1; A5[4, 0] = -1 / jm
    A6 = np.zeros((6, 6)); A6[0, 3] = 1 / ls; A6[1, 2] = -1 / ls; A6[2, 3] = -1; A6[3, 2] = 1
    B = np.zeros((6, 2)); B[0, 0] = B[1, 1] = 1 / ls
    return A1, A2, A3, A4, A5, A6, B


class EKF:
    """Block EKF_bemf_observer/extended_kalman_observer: Unit Delay1 holds the corrected state of the previous step,
    the outputs (flux, angle, speed) are read from it; the step builds x(k|k) = F_pred*x + ts*omega_bez*B*u with
    F_pred = I + ts*omega_bez*(A1 + A2*ia + A3*ib + A4*psia + A5*psib) + ts*A6_pred*omega (copy A_ekf of the state
    prediction; A6_pred = A6 before the fix, zeros after), P = F*P*F' + Q with the full F (copy correlation_matrix/
    A_ekf, omega_bez on all paths), L = P*C'/(C*P*C' + R), x <- x(k|k) + L*(i - C*x(k|k)), P <- (I - L*C)*P."""

    def __init__(self, rs, ls, a6_pred):
        self.M = ekf_matrices(rs, ls, jm_ekf)
        self.a6_pred = a6_pred
        self.x = np.array([0, 0, 1, 1, 0, 0], float) * ts
        self.P = np.zeros((6, 6))
        self.I6 = np.eye(6)

    def outputs(self):
        return np.arctan2(self.x[3], self.x[2]), self.x[4], self.x[2], self.x[3]

    def step(self, u, i):
        A1, A2, A3, A4, A5, A6, B = self.M
        ia, ib, pa, pb, w, tl = self.x
        J5 = A1 + A2 * ia + A3 * ib + A4 * pa + A5 * pb
        F_cov = self.I6 + ts * omega_bez * (J5 + A6 * w)
        F_pred = self.I6 + ts * omega_bez * J5 + ts * (A6 if self.a6_pred else 0.0) * w
        xp = F_pred @ self.x + ts * omega_bez * (B @ u)
        P = F_cov @ self.P @ F_cov.T + Q_ekf
        S = P[0:2, 0:2] + R_ekf
        L = P[:, 0:2] @ np.linalg.inv(S)
        self.P = P - L @ P[0:2, :]
        self.x = xp + L @ (i - xp[0:2])


class DIO:
    """Block double_integrator_observer (inside extended_kalman_observer and in encoder-sensorless-switch):
    e = wrap(phase - theta), omega(k+1) = omega + ts*komega*e/omega_bez (pu), theta(k+1) = wrap(theta) +
    ts*ktheta*e + ts*omega_bez*omega; outputs wrap(theta(k)) and omega(k); poles [-1 -4]*omega_bez."""

    def __init__(self):
        A = np.array([[1.0, ts], [0.0, 1.0]])
        p = np.exp(np.array([-1.0, -4.0]) * omega_bez * ts)
        a1, a0 = -(p[0] + p[1]), p[0] * p[1]
        l1 = A[0, 0] + A[1, 1] + a1
        l2 = A[1, 0] + (a0 - (A[0, 0] - l1) * A[1, 1]) / A[0, 1]
        self.komega, self.ktheta = l2 / ts, l1 / ts
        self.th, self.w = np.pi, 0.0

    def outputs(self):
        return np.angle(np.exp(1j * self.th)), self.w

    def step(self, phase):
        e = np.angle(np.exp(1j * (phase - np.angle(np.exp(1j * self.th)))))
        th_w = np.angle(np.exp(1j * self.th))
        self.th, self.w = th_w + ts * self.ktheta * e + ts * omega_bez * self.w, self.w + ts * self.komega * e / omega_bez


def speed_profile(t, rpm_sim):
    """fixed_speed_setting: PS Step 0 -> rpm_sim at 0.025 s, PS Transfer Function lag 0.1 s.
    Returns the electrical speed [rad/s] and the electrical angle [rad] (rotor at 0 at t = 0)."""
    t0, tau = 0.025, 0.1
    w_end = omega_bez * rpm_sim / rpm_nom
    if t <= t0:
        return 0.0, 0.0
    x = t - t0
    return w_end * (1 - np.exp(-x / tau)), w_end * (x - tau * (1 - np.exp(-x / tau)))


def plant_rhs(t, x, s, pl):
    """x = [i_d, i_q, v1, v2, v_rc] (true rotor frame, capacitor voltages, battery RC voltage); s = leg levels."""
    i_d, i_q, v1, v2, vrc = x
    w, th = speed_profile(t, pl['rpm'])
    c, sn = np.cos(th), np.sin(th)
    ia = c * i_d - sn * i_q
    ibeta = sn * i_d + c * i_q
    i_ph = (ia, -0.5 * ia + S3_2 * ibeta, -0.5 * ia - S3_2 * ibeta)
    iP = sum(i for k, i in enumerate(i_ph) if s[k] > 0)
    iN = sum(i for k, i in enumerate(i_ph) if s[k] < 0)
    # battery current into P (charging when negative): OCV - R0*ib - vrc = v1 + v2 + r*(iC1 + iC2)
    i_bat = (U_ocv - vrc - v1 - v2 + r_esr * (iP - iN)) / (R0_b + 2 * r_esr)
    iC1 = i_bat - iP
    iC2 = i_bat + iN
    v1t = v1 + r_esr * iC1
    v2t = v2 + r_esr * iC2
    vph = [v1t if sx > 0 else (-v2t if sx < 0 else 0.0) for sx in s]
    u_al = 2 / 3 * (vph[0] - 0.5 * vph[1] - 0.5 * vph[2])
    u_be = (vph[1] - vph[2]) / np.sqrt(3)
    u_d = c * u_al + sn * u_be
    u_q = -sn * u_al + c * u_be
    Ldt, Lqt = pl['L'] * Ld_m + LFi, pl['L'] * Lq_m + LFi
    did = (u_d - R_t * i_d + w * Lqt * i_q) / Ldt
    diq = (u_q - R_t * i_q - w * Ldt * i_d - w * psi_m) / Lqt
    C = C_dc * pl['C']
    return np.array([did, diq, iC1 / C, iC2 / C, (i_bat - vrc / R1_b) / C1_b]), i_bat


def rk4(t, x, s, h, nsub, pl):
    dt = h / nsub
    ibat = 0.0
    for _ in range(nsub):
        k1, ibat = plant_rhs(t, x, s, pl)
        k2, _ = plant_rhs(t + dt / 2, x + dt / 2 * k1, s, pl)
        k3, _ = plant_rhs(t + dt / 2, x + dt / 2 * k2, s, pl)
        k4, _ = plant_rhs(t + dt, x + dt * k3, s, pl)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    return x, ibat


def simulate(name, model='new', angle='ekf', dead_time=td, L_factor=1.0, C_factor=1.0, rpm_sim=rpm_nom,
             speed_out='dio', k_sw=None, k_dc=None, sw_mode='count', t_end=1.5, nsub=8):
    """One run of the whole chain. model 'new': the fixed model; 'old': the model before the fix (case A).
    angle 'encoder': reference angle and torque-curve speed from the encoder (use_psm_encoder = 1, angle of the
    last step through phase_acquisition, the double integrator observer tracks it without error).
    L_factor, C_factor: machine inductances and DC-link capacitance of the plant times this factor (the control
    keeps the values of the init). speed_out 'dio': speed output of the EKF block = double integrator observer on
    the EKF angle (fixed model); 'state': the speed state of the EKF. k_sw, sw_mode: weight and kind of the
    switching term of the predictive control ('count': number of level changes, 'du2': change of the voltage
    vector squared). The run stops if a DC-link capacitor goes below 5 % of the link voltage."""
    pl = dict(L=L_factor, C=C_factor, rpm=rpm_sim)
    if model == 'new':
        ekf = EKF(MPC['R_norm'], MPC['L_norm'], a6_pred=False)
        mpc = MPCNew(sw_mode)
        dio = DIO()
        k_sw = MPC['k_sw'] if k_sw is None else k_sw
        k_dc = MPC['k_dc'] if k_dc is None else k_dc
    else:
        ekf = EKF(Rs_n, Ls_n, a6_pred=True)
        mpc = None
    N = int(round(t_end / Tc))
    v0 = U_ocv / 2                          # Simscape starts from steady state (DoDC on): DC link at the battery voltage
    x = np.array([0.0, 0.0, v0, v0, 0.0])
    i_prev_meas = np.zeros(2)
    vdc_prev_meas = np.array([v0, v0]) / udc_nom
    th_prev, w_prev = 0.0, 0.0
    u_prev_old = np.zeros(2)                # old code: persistent ualpha_prev, ubeta_prev
    spd_flt = 0.0
    s_prev = np.array([0, 0, 0])
    keys = ('t', 'w', 'w_hat', 'th_err', 'id', 'iq', 'iq_ref', 'Te', 'v1', 'v2', 'ibat', 'ia_err', 'ib_err',
            'ia', 'ib', 'ia_ref', 'ib_ref', 'state')
    rec = {k: np.zeros(N) for k in keys}
    trans = {k: 0 for k in ('OP', 'PO', 'ON', 'NO', 'PN', 'NP')}
    trans_t0 = 1.0
    nlev = 0
    diverged = False
    np_collapse = None
    for k in range(N):
        t = k * Tc
        w_true, th_true = speed_profile(t, rpm_sim)
        i_d, i_q, v1, v2, vrc = x
        c, sn = np.cos(th_true), np.sin(th_true)
        ia_t = (c * i_d - sn * i_q) / ibez
        ib_t = (sn * i_d + c * i_q) / ibez
        # acquisition: quantized phase currents (pu), then alpha/beta; one step of delay (Unit Delay)
        i_a, i_b, i_c = ia_t, -0.5 * ia_t + S3_2 * ib_t, -0.5 * ia_t - S3_2 * ib_t
        iq3 = np.round(np.array([i_a, i_b, i_c]) / q_adc) * q_adc
        i_meas = np.array([2 / 3 * (iq3[0] - 0.5 * iq3[1] - 0.5 * iq3[2]), 2 / 3 * S3_2 * (iq3[1] - iq3[2])])
        i_used = i_prev_meas.copy()
        i_prev_meas = i_meas
        vdc_meas = np.round(np.array([v1, v2]) / udc_nom / q_adc) * q_adc
        vdc_used = vdc_prev_meas.copy()
        vdc_prev_meas = vdc_meas
        # EKF outputs (state of the previous step)
        th_ekf, w_ekf, psa, psb = ekf.outputs()
        if model == 'new':
            _, w_dio = dio.outputs()
            dio.step(th_ekf)
            if speed_out == 'dio':
                w_ekf = w_dio
        # encoder (one step late) or observer; the switch uses the encoder while the speed is zero (omega_th = 0)
        if model == 'old':
            th_enc, w_enc = th_true, w_true / omega_bez        # as the first check
        else:
            th_enc, w_enc = th_prev, w_prev / omega_bez
        th_prev, w_prev = th_true, w_true
        if angle == 'encoder' or w_true == 0.0:
            th_ctrl, w_ctrl = th_enc, w_enc
        else:
            th_ctrl, w_ctrl = th_ekf, w_ekf
        # torque reference: |speed| through the 1 Hz filter, x rpm_nom, lookup, x -1, / tau_bez, saturation <= 0
        spd_flt = (1 - g0_1Hz) * spd_flt + g0_1Hz * abs(w_ctrl)
        iq_ref = min(0.0, -lookup_linear(spd_flt * rpm_nom, rpm_base, torque_base) / tau_bez)
        i_ref = np.array([-iq_ref * np.sin(th_ctrl), iq_ref * np.cos(th_ctrl)])
        # predictive control
        if model == 'new':
            _, ua_out, ub_out, n = mpc.step(vdc_used[0], vdc_used[1], i_ref[0], i_ref[1], i_used[0], i_used[1],
                                            w_ekf, psa, psb, MPC['L_norm'], MPC['R_norm'], MPC['u_scale'],
                                            MPC['np_gain'], MPC['k_kalman'], k_dc, k_sw, MPC['ts_pu'])
            u_ekf = np.array([ua_out, ub_out]) * 1.0 * 1.0     # mp control Gain2/Gain3 = 1, u_psm_scale_ekf = 1
        else:   # mpc_ttype_gate before the fix
            Vdc3 = (vdc_used[0] + vdc_used[1]) / 3
            ua = Vdc3 * UA_UNIT
            ub = Vdc3 * UB_UNIT
            ia_p = (1 - Rs_n / Ls_n * ts_pu) * i_used[0] + w_ekf / Ls_n * psb * ts_pu + ua * ts_pu / Ls_n
            ib_p = (1 - Rs_n / Ls_n * ts_pu) * i_used[1] - w_ekf / Ls_n * psa * ts_pu + ub * ts_pu / Ls_n
            J = (i_ref[0] - ia_p) ** 2 + (i_ref[1] - ib_p) ** 2     # k_kalman = 1, k_dc = k_sw = 0
            n = int(np.argmin(J))
            u_prev_old = np.array([ua[n], ub[n]])
            u_ekf = 3 * u_prev_old * u_scale_ekf_old
        s = STATES[n].astype(int)
        ekf.step(u_ekf, i_used)
        # level changes of the legs
        if t >= trans_t0:
            for ph in range(3):
                a, b = s_prev[ph], s[ph]
                if a != b:
                    nlev += 1
                    trans[{(0, 1): 'OP', (1, 0): 'PO', (0, -1): 'ON', (-1, 0): 'NO', (1, -1): 'PN', (-1, 1): 'NP'}[(a, b)]] += 1
        # dead time: a leg keeps its old level when the switch that turns on has to take the current
        if dead_time > 0 and model == 'new':
            i_ph = (ia_t, -0.5 * ia_t + S3_2 * ib_t, -0.5 * ia_t - S3_2 * ib_t)
            s_dt = s.copy()
            for ph in range(3):
                if s[ph] > s_prev[ph] and i_ph[ph] > 0:
                    s_dt[ph] = s_prev[ph]
                elif s[ph] < s_prev[ph] and i_ph[ph] < 0:
                    s_dt[ph] = s_prev[ph]
            x, _ = rk4(t, x, s_dt, dead_time, 1, pl)
            x, ibat = rk4(t + dead_time, x, s, Tc - dead_time, nsub, pl)
        else:
            x, ibat = rk4(t, x, s, Tc, nsub, pl)
        s_prev = s
        th_err = np.angle(np.exp(1j * (th_ctrl - th_true))) if w_true > 0 else 0.0
        vals = dict(t=t, w=w_true / omega_bez, w_hat=w_ekf, th_err=th_err * 180 / np.pi, id=i_d / ibez, iq=i_q / ibez,
                    iq_ref=iq_ref, Te=torque_pu(i_d / ibez, i_q / ibez), v1=v1, v2=v2, ibat=ibat,
                    ia_err=ia_t - i_ref[0], ib_err=ib_t - i_ref[1], ia=ia_t, ib=ib_t, ia_ref=i_ref[0],
                    ib_ref=i_ref[1], state=n)
        for kk, vv in vals.items():
            rec[kk][k] = vv
        if not np.all(np.isfinite(x)) or abs(x[0]) > 20 * ibez or abs(x[1]) > 20 * ibez:
            diverged = True
        if min(x[2], x[3]) < 0.05 * (x[2] + x[3]):
            np_collapse = t
        if diverged or np_collapse is not None:
            for kk in keys:
                rec[kk] = rec[kk][:k + 1]
            break
    t_win = max(0.0, (rec['t'][-1] - trans_t0)) if not (diverged or np_collapse is not None) else 0.0
    rec['trans'] = {kk: vv / t_win if t_win > 0 else np.nan for kk, vv in trans.items()}
    rec['f_dev'] = nlev / 3 / t_win / 2 if t_win > 0 else np.nan     # average device switching frequency per leg
    rec['diverged'] = diverged
    rec['np_collapse'] = np_collapse
    rec['name'] = name
    return rec


def summary(rec, t0=1.2):
    m = rec['t'] >= t0
    if rec['diverged'] or rec['np_collapse'] is not None or not np.any(m):
        return None
    w = rec['w'][m]
    ia_err, ib_err = rec['ia_err'][m], rec['ib_err'][m]
    err_rms = np.sqrt(np.mean(ia_err ** 2 + ib_err ** 2) / 2)
    st = STATES[rec['state'][m].astype(int)]
    dv = rec['v1'][m] - rec['v2'][m]
    return dict(o_leg=np.mean(st == 0), o_none=np.mean(np.all(st != 0, axis=1)),
                th_err=np.mean(rec['th_err'][m]), th_std=np.std(rec['th_err'][m]),
                w_ratio=np.mean(rec['w_hat'][m]) / np.mean(w),
                id=np.mean(rec['id'][m]), iq=np.mean(rec['iq'][m]), iq_ref=np.mean(rec['iq_ref'][m]),
                Te=np.mean(rec['Te'][m]), id_std=np.std(rec['id'][m]), iq_std=np.std(rec['iq'][m]), err=err_rms,
                dv_mean=np.mean(dv), dv_max=np.max(np.abs(dv)), dv_end=dv[-1],
                dv_max_run=np.max(np.abs(rec['v1'] - rec['v2'])),
                udc=np.mean(rec['v1'][m] + rec['v2'][m]), P=-np.mean(rec['ibat'][m] * (rec['v1'][m] + rec['v2'][m])),
                f_dev=rec['f_dev'])


log('\n3. Switched simulation of the whole chain (1.5 s, values averaged over 1.2 ... 1.5 s, one of the six systems)')
CASES = [
    ('A model before the fix', dict(model='old')),
    ('N1 fixed model (sensorless, dead time 3 us)', dict()),
    ('N2 fixed model, encoder angle (use_psm_encoder = 1)', dict(angle='encoder')),
    ('N3 fixed model without dead time', dict(dead_time=0.0)),
    ('N4 fixed model, machine inductances +20 % (control unchanged)', dict(L_factor=1.2)),
    ('N5 fixed model, machine inductances -15 % (control unchanged)', dict(L_factor=0.85)),
    ('N6 fixed model, DC-link capacitance -50 % (control unchanged)', dict(C_factor=0.5)),
    ('N7 fixed model, shaft at 12 rpm (partial load)', dict(rpm_sim=12.0)),
    ('N8 fixed model, machine inductances -20 % (limit of the sensorless control)', dict(L_factor=0.8)),
    ('N9 fixed model, machine inductances -20 %, encoder angle', dict(L_factor=0.8, angle='encoder')),
    ('S1 as N1 without the switching term (k_sw = 0)', dict(k_sw=0.0)),
    ('S2 as N7 (12 rpm) without the switching term (k_sw = 0)', dict(rpm_sim=12.0, k_sw=0.0)),
    ('S3 as N1 with the speed state of the EKF as speed output (no double integrator observer)', dict(speed_out='state')),
    ('D1 as N1 without the neutral-point term (k_dc = 0)', dict(k_dc=0.0)),
]
results = {}
for name, kw in CASES:
    rec = simulate(name, **kw)
    results[name] = rec
    s = summary(rec)
    if s is None:
        if rec['np_collapse'] is not None:
            log(f'   {name}: NEUTRAL POINT LOST at t = {rec["np_collapse"]:.3f} s (one DC-link capacitor below 5 % of '
                f'the link voltage; run stopped)')
        else:
            log(f'   {name}: DIVERGED at t = {rec["t"][-1]:.3f} s (current above 20 pu)')
        continue
    curve = -lookup_linear(np.mean(rec['w'][rec['t'] >= 1.2]) * rpm_nom, rpm_base, torque_base) / tau_bez
    log(f'   {name}:')
    log(f'      control angle error {s["th_err"]:+6.2f} deg (std {s["th_std"]:.2f}), EKF speed {s["w_ratio"]:.4f} x true | '
        f'id {s["id"]:+.3f} (std {s["id_std"]:.3f}), iq {s["iq"]:+.3f} (std {s["iq_std"]:.3f}) pu for iq_ref {s["iq_ref"]:+.3f} '
        f'| torque {s["Te"]:+.3f} pu, torque curve at the true speed {curve:+.3f} pu ({(s["Te"]/curve-1)*100:+.1f} %)')
    log(f'      current error rms {s["err"]:.4f} pu ({s["err"]*ibez:.1f} A) | DC link {s["udc"]:.0f} V, vdc1 - vdc2 mean '
        f'{s["dv_mean"]:+.1f} V, max |.| {s["dv_max"]:.1f} V (whole run {s["dv_max_run"]:.1f} V) | '
        f'battery charged with {s["P"]/1e3:.0f} kW | '
        f'switching {s["f_dev"]:.0f} Hz per device on average')
    tr = rec['trans']
    log(f'      a leg is at O {s["o_leg"]*100:.1f} % of the time, {s["o_none"]*100:.1f} % of the states have no leg at O | '
        f'level changes per second (3 legs): O<->P {tr["OP"]+tr["PO"]:.0f}, O<->N {tr["ON"]+tr["NO"]:.0f}, '
        f'P<->N {tr["PN"]+tr["NP"]:.0f}')

# figures of the note (PDF, figures/), lines rasterized to keep the files small
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'lines.linewidth': 0.8, 'pdf.fonttype': 42})
NA, N1, N2, N6, N7 = CASES[0][0], CASES[1][0], CASES[2][0], CASES[6][0], CASES[7][0]
S1, S2, D1 = CASES[10][0], CASES[11][0], CASES[13][0]
COL = {'A': 'C0', 'N1': 'C1', 'N2': 'C2', 'N6': 'C3', 'N7': 'C4', 'D1': 'C7'}


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, name), dpi=300)
    plt.close(fig)


fig, ax = plt.subplots(4, 1, figsize=(5.6, 4.8), sharex=True)
for name, lab, st in ((NA, 'A (before the fix)', '-'), (N1, 'N1 (fixed, sensorless)', '-'), (N2, 'N2 (fixed, encoder)', '--')):
    r = results[name]
    c = COL[name.split(' ')[0]]
    ax[0].plot(r['t'], r['iq'], st, color=c, label=lab, rasterized=True)
    ax[1].plot(r['t'], r['id'], st, color=c, label=lab, rasterized=True)
    ax[2].plot(r['t'], r['th_err'], st, color=c, label=lab, rasterized=True)
    ax[3].plot(r['t'], r['w_hat'], st, color=c, label=lab, rasterized=True)
for name, lab, st in ((NA, '$i_q^*$ of A', 'k:'), (N1, '$i_q^*$ of N1', 'k-')):
    r = results[name]
    ax[0].plot(r['t'], r['iq_ref'], st, lw=0.7, label=lab, rasterized=True)
ax[3].plot(results[NA]['t'], results[NA]['w'], 'k-', lw=0.7, label='true speed', rasterized=True)
ax[0].set_ylabel('$i_q$ [pu]'); ax[1].set_ylabel('$i_d$ [pu]'); ax[2].set_ylabel('angle error [deg]')
ax[3].set_ylabel('speed estimate [pu]'); ax[3].set_xlabel('time [s]'); ax[2].set_ylim(-30, 15); ax[3].set_ylim(-1.6, 1.2)
for a in ax:
    a.grid(True, lw=0.3)
ax[0].legend(loc='upper right', ncol=3)                 # one legend for the three cases, in the empty corner
ax[3].legend(ax[3].get_lines()[-1:], ['true speed'], loc='lower right')
savefig(fig, 'fig_startup.pdf')

fig, ax = plt.subplots(2, 1, figsize=(6.7, 3.8), sharex=True)
for a, cases in ((ax[0], ((NA, 'A (before the fix)'), (D1, 'D1 (fixed, without the neutral-point term, $k_{dc} = 0$)'))),
                 (ax[1], ((N1, 'N1 (fixed)'), (N6, 'N6 (fixed, DC-link C $-50\\,\\%$)'), (N7, 'N7 (fixed, 12 rpm)')))):
    for name, lab in cases:
        r = results[name]
        key = name.split(' ')[0]
        a.plot(r['t'], r['v1'] - r['v2'], color=COL[key], lw=0.6, label=lab, rasterized=True,
               zorder={'N1': 3, 'N7': 2}.get(key, 1))
    a.set_ylabel('$v_{dc1} - v_{dc2}$ [V]'); a.grid(True, lw=0.3); a.legend(loc='upper left', ncol=3)
ax[0].set_title('neutral point not controlled'); ax[1].set_title('neutral point controlled'); ax[1].set_ylim(-25, 25)
ax[-1].set_xlabel('time [s]')
savefig(fig, 'fig_neutral_point.pdf')

fig, ax = plt.subplots(3, 1, figsize=(6.7, 4.3), sharex=True)
for a, name, lab in zip(ax, (NA, N1, N7), ('A (before the fix), 17.8 rpm', 'N1 (fixed, sensorless), 17.8 rpm', 'N7 (fixed, sensorless), 12 rpm')):
    r = results[name]
    m = (r['t'] >= 1.40) & (r['t'] <= 1.45)
    a.plot(r['t'][m], r['ia'][m], color=COL[name.split(' ')[0]], lw=0.7, label='$i_\\alpha$')
    a.plot(r['t'][m], r['ia_ref'][m], 'k--', lw=0.7, label='$i_\\alpha^*$')
    a.set_title(lab); a.set_ylabel('[pu]'); a.grid(True, lw=0.3); a.legend(loc='upper right')
ax[-1].set_xlabel('time [s]')
savefig(fig, 'fig_current_zoom.pdf')

cats = ('O$\\leftrightarrow$P', 'O$\\leftrightarrow$N', 'P$\\leftrightarrow$N')
groups = ((NA, 'A, 17.8 rpm'), (S1, 'fixed, $k_{sw} = 0$, 17.8 rpm'), (N1, 'fixed, 17.8 rpm'),
          (S2, 'fixed, $k_{sw} = 0$, 12 rpm'), (N7, 'fixed, 12 rpm'))
fig, ax = plt.subplots(1, 1, figsize=(6.7, 2.5))
wbar = 0.16
for gi, (name, lab) in enumerate(groups):
    tr = results[name]['trans']
    vals = (tr['OP'] + tr['PO'], tr['ON'] + tr['NO'], tr['PN'] + tr['NP'])
    ax.bar(np.arange(3) + (gi - 2) * wbar, vals, wbar, label=lab)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(cats); ax.set_ylabel('level changes per second (3 legs)')
ax.grid(True, axis='y', lw=0.3); ax.legend(ncol=2)
savefig(fig, 'fig_switching.pdf')

# -------------------------------------------------------------------------------------------------------------
# 4. gate timing of the T-type legs
# -------------------------------------------------------------------------------------------------------------
log('\n4. Gate timing of the T-type legs on the tc grid of the Simscape local solver (gate 0/15 V, filter [tc 1],')
log('   CMOS logic threshold 7.5 V and Q1-Q4 interlock in dead_time_leg_generator_1, switch threshold 5.5 V).')
log('   The command changes at a control instant; the bridge sees the gates at t0 + k*tc.')


def gate_level(t, rise, delay=0.0, logic=False):
    """Gate voltage at t after a command step (rise True: 0 -> 15 V, False: 15 -> 0 V). Filter [tc 1]; logic: the
    CMOS block switches its 15 V output when the filtered command crosses 7.5 V; delay: turn-on delay (min of the
    signal and its copy delayed by delay; it acts on the rising edge only)."""
    def filt(tt):
        if tt <= 0:
            return 0.0 if rise else 15.0
        return 15.0 * (1 - np.exp(-tt / tc)) if rise else 15.0 * np.exp(-tt / tc)

    def sig(tt):
        v = filt(tt)
        return (15.0 if v > 7.5 else 0.0) if logic else v
    return min(sig(t), sig(t - delay)) if (delay > 0 and rise) else sig(t)


for label, delay in (('before the fix (no turn-on delay)', 0.0), (f'after the fix (turn-on delay {td*1e6:.0f} us on the four gates)', td)):
    log(f'   {label}:')
    for change, on_dev, off_dev in (('O -> P: Q3 off, Q1 on (Q2 on)', ('Q1', True), ('Q3', False)),
                                    ('P -> O: Q1 off, Q3 on (Q2 on)', ('Q3', False), ('Q1', True)),
                                    ('O -> N: Q2 off, Q4 on (Q3 on)', ('Q4', True), ('Q2', False)),
                                    ('N -> O: Q4 off, Q2 on (Q3 on)', ('Q2', False), ('Q4', True))):
        both = []
        for kk in range(1, 12):
            tt = kk * tc
            g_on = gate_level(tt, True, delay, logic=on_dev[1])
            g_off = gate_level(tt, False, 0.0, logic=off_dev[1])
            both.append(g_on > 5.5 and g_off > 5.5)
        first_on = next(kk for kk in range(1, 40) if gate_level(kk * tc, True, delay, logic=on_dev[1]) > 5.5)
        last_off = next(kk for kk in range(1, 40) if gate_level(kk * tc, False, 0.0, logic=off_dev[1]) <= 5.5)
        log(f'      {change:34s}: {off_dev[0]} off from step {last_off} ({last_off*tc*1e6:.3f} us), {on_dev[0]} on from step '
            f'{first_on} ({first_on*tc*1e6:.3f} us) -> {"SHORT CIRCUIT of a capacitor at step 1" if any(both) else "no overlap"}')
log('   Before the fix the first step after O -> P / O -> N saw Q3 (Q2) still at 15*exp(-1) = 5.52 V (on) and Q1 (Q4)')
log('   already on: one step of short circuit of a 14.4 mF capacitor per change. After the fix every switch turns on')
log(f'   {td*1e6:.0f} us later, after the one that turns off has gone below 5.5 V (two steps, 1.25 us).')
fig, ax = plt.subplots(1, 2, figsize=(6.7, 2.5), sharey=True)
tt = np.linspace(-0.5e-6, 5e-6, 1101)
kk = np.arange(0, 9)
for a, (lab, delay) in zip(ax, (('before the fix', 0.0), (f'after the fix (turn-on delay {td*1e6:.0f} $\\mu$s)', td))):
    q3 = [gate_level(x, False, 0.0, logic=False) for x in tt]
    q1 = [gate_level(x, True, delay, logic=True) for x in tt]
    a.plot(tt * 1e6, q3, 'C0', label='Q3 (turns off)')
    a.plot(tt * 1e6, q1, 'C3', label='Q1 (turns on)')
    a.plot(kk * tc * 1e6, [gate_level(x * tc, False, 0.0, logic=False) for x in kk], 'C0o', ms=3)
    a.plot(kk * tc * 1e6, [gate_level(x * tc, True, delay, logic=True) for x in kk], 'C3s', ms=3)
    a.axhline(5.5, color='k', lw=0.6, ls='--')
    both = [x for x in kk if x > 0 and gate_level(x * tc, False, 0.0) > 5.5 and gate_level(x * tc, True, delay, logic=True) > 5.5]
    for x in both:
        a.axvspan((x - 0.5) * tc * 1e6, (x + 0.5) * tc * 1e6, color='r', alpha=0.15, lw=0)
    a.set_title(lab); a.set_xlabel('time after the command O$\\rightarrow$P [$\\mu$s]'); a.grid(True, lw=0.3)
ax[0].set_ylabel('gate voltage [V]'); ax[0].legend(loc='center right')
savefig(fig, 'fig_gate_timing.pdf')
OUT.close()
