#!/usr/bin/env python3
"""Design check of mpc_psm (mpc_psm.slx), before and after the control fixes, with the numbers of init_model.m.

Run: python3 design_check_mpc_psm.py [output folder]
     -> design_check_mpc_psm_output.txt and the PNG figures (default: next to this file).
     mpc_port_2l.py (copies of the MATLAB Function mp control/mpc_pu_d, old and new) must be in the same folder.

What it checks
 1. Per-unit chain: voltage of the predictive control and of the EKF in pu of ubez, inductance and resistance
    of the prediction and of the EKF, operating point at rated speed.
 2. The predictive current loop seen as a deadbeat loop, before and after the fix (closed-loop poles).
 3. A switched simulation (vector level) of the whole chain, 1.25 s (StopTime of the model):
    shaft speed of fixed_speed_setting (step to rpm_sim at 0.025 s through a 0.1 s lag), torque curve with the
    1 Hz speed filter and the saturation, current reference on the q axis (id = 0) at the control angle,
    current and DC voltage sampled, quantized (1/2^11 pu) and one control step late, control clock 1.001
    (block global timing/time_inv1), encoder-sensorless switch (encoder while the speed is zero).
    Fixed model: predictive control = mpc_port_2l.MPCNew (copy of the new script, checked against Octave),
    EKF with L = Lq + LFi, R = Rs + RLFi, no A6 path in the state prediction, voltage = output ualpha, ubeta
    of the predictive control (vector of the last step, pu of ubez) through the Unit Delay of mp control
    (gains 1), speed output from the double integrator observer on the EKF angle, EKF process noise q_ekf = 1/100.
    Model before the fix (cases A, A0): mpc_port_2l.MPCOld (7 vectors, zero vector = all upper switches,
    restricted transitions, voltage in pu of udc_nom with vdc = 1, no delay compensation, Ls and Rs), EKF with
    Ls, the A6 path without omega_bez, voltage 3/2*u*u_psm_scale_ekf through the Unit Delay (the vector of the
    last step), speed output = speed state of the EKF, q_ekf = 1/10. Case A has the bridge block of the dev model with its
    two built-in passives on (1 mH + 10 mOhm per phase, 2 mF across the DC link), case A0 has them off.
    Plant: PMSM in the rotor frame (Ld, Lq of one system + 230 uH output inductor), two-level bridge with ideal
    switches, DC link 2 x 14.4 mF in series with 2 mOhm each (block dclink), battery = open-circuit voltage at
    soc 0.85 + R0 + R1//C1 of the init. Dead time of the gate generator (dead_time_generator_cmos_simulink):
    the switch that turns on waits until the RC copy (time constant sqrt(2)*dead_time) of the other gate has
    fallen below 7.5 V, ln(2)*sqrt(2)*dead_time = 2.94 us; during that time the leg keeps the old level when
    the current flows the way that needs the switch that turns on (to P with i > 0, to N with i < 0) and takes
    the new level at once when a diode takes the current.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mpc_port_2l import MPCNew, MPCOld, STATES  # noqa: E402

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else HERE
os.makedirs(OUTDIR, exist_ok=True)
OUT = open(os.path.join(OUTDIR, 'design_check_mpc_psm_output.txt'), 'w')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.write(s + '\n')


# -------------------------------------------------------------------------------------------------------------
# parameters (the same numbers and formulas as init_model.m and private/*.m)
# -------------------------------------------------------------------------------------------------------------
fpwm = 15e3
ts = 1 / fpwm                           # 66.7 us: control step (single update)
clock_gain = 1.001                      # global timing/time_inv1: the control runs at 15015 Hz
Tc = ts / clock_gain                    # real control period
tc = ts / 200                           # Simscape local solver step, gate filter time constant
td_param = 3e-6                         # param.time.dead_time
td = np.log(2) * np.sqrt(2) * td_param  # turn-on delay of the CMOS gate generator: 2.94 us

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

# inverter (private/inv_hw_param.m), DC link block, bridge block
udc_nom, LFi = 1070.0, 230e-6
RLFi = 157 * 0.05 * LFi
C_dc = 14.4e-3 / 2                      # CFi1 and CFi2 (14.4 mF each) in series
r_esr = 2 * 2e-3                        # 2 mOhm in each capacitor of the dclink library block
L_br, R_br, C_br = 1e-3, 10e-3, 2e-3    # built-in passives of the Three-Phase Bridge (ideal) block (defaults)

# battery (private/battery_param.m + legacy changes R0/2, R1/2, C1/50): 1200 V, 250 kW, 1C capacity, soc 0.85
ts_dab = 1 / 7.5e3 / 2
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
Q_DIAG = np.diag([ts ** 2, ts ** 2, 0.5, 0.5, 1, 1])
Q_EKF_OLD, Q_EKF_NEW = 0.1, 0.01     # q_ekf: legacy 1/10, fixed model 1/100 (Q = ts*q_ekf*Q_DIAG)
R_ekf = ts * 1.0 * np.eye(2)
S3_2 = np.sqrt(3) / 2
L_t = Lq_m + LFi
# fixed model: param.psm.ctrl.mpc.* (the EKF uses the same L_norm and R_norm)
MPC = dict(L_norm=L_t / Lbez, R_norm=(Rs_m + RLFi) / Xbez, u_scale=udc_nom / ubez, k_kalman=1.0, k_sw=1e-5)
# model before the fix
u_psm_scale_ekf_old = np.sqrt(3) / 2 * 2 / 3 * udc_nom / ubez


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
log('mpc_psm design check, before and after the control fixes (numbers of init_model.m)')
log('=' * 110)

# -------------------------------------------------------------------------------------------------------------
# 1. per unit chain
# -------------------------------------------------------------------------------------------------------------
log('\n1. Per-unit chain (one of the six systems)')
log(f'   machine: ubez = {ubez:.1f} V peak, ibez = {ibez:.1f} A peak, tau_bez = {tau_bez/1e3:.2f} kNm, '
    f'omega_bez = {omega_bez:.2f} rad/s ({omega_bez/2/np.pi:.2f} Hz), 1 pu power = {P_bez/1e3:.1f} kW')
log(f'   Rs = {Rs_m*1e3:.2f} mOhm, Ld = {Ld_m*1e3:.2f} mH ({Ld_n:.4f} pu), Lq = {Lq_m*1e3:.2f} mH ({Lq_n:.4f} pu), '
    f'Ls = (Ld + Lq)/2 = {Ls_n:.4f} pu; with the 230 uH inductor the bridge sees {(Ld_m+LFi)/Lbez:.4f} pu (d) and '
    f'{L_t/Lbez:.4f} pu (q)')
log(f'   battery: {n_cells} cells, open circuit {U_ocv:.0f} V at soc {soc} ({U_ocv/udc_nom:.3f} pu of udc_nom '
    f'{udc_nom:.0f} V), R0 = {R0_b*1e3:.0f} mOhm, R1 = {R1_b*1e3:.0f} mOhm, C1 = {C1_b*1e3:.0f} mF')
log('   Before the fix the predictive control computes the vectors from vdc = 1 (Constant) without any voltage')
log('   scale: 2/3 for the vector [1 0 0], in pu of udc_nom, while its model needs pu of ubez.')
log(f'   The true vector is 2/3*udc/ubez = {2/3*U_ocv/ubez:.3f} pu: the prediction sees the voltage '
    f'{U_ocv/ubez:.2f} times too small.')
log(f'   EKF voltage before the fix: 3/2*u*u_psm_scale_ekf = 3/2*u*{u_psm_scale_ekf_old:.4f} -> '
    f'{3/2*u_psm_scale_ekf_old*2/3:.3f} pu for [1 0 0], {3/2*u_psm_scale_ekf_old*2/3/(2/3*U_ocv/ubez):.3f} x the true '
    f'voltage, and the vector of the last step (one step too new).')
log(f'   Fixed predictive control: state voltages from the measured vdc (pu of udc_nom) times u_scale = '
    f'udc_nom/ubez = {MPC["u_scale"]:.4f} -> pu of ubez;')
log(f'   prediction and EKF with L_norm = (Lq + LFi)/Lbez = {MPC["L_norm"]:.4f} pu and R_norm = (Rs + RLFi)/Xbez = '
    f'{MPC["R_norm"]:.4f} pu (before: Ls_norm = {Ls_n:.4f}, Rs_norm = {Rs_n:.4f});')
log('   the EKF gets the voltage of the vector of two steps before (output of the predictive control = vector of the')
log('   last step, then the Unit Delay of mp control), gains 1, u_psm_scale_ekf = 1.')
T_curve = lookup_linear(rpm_nom, rpm_base, torque_base)
iq_rated = -T_curve / tau_bez
ud_ss = -omega_bez * L_t * iq_rated * ibez
uq_ss = (Rs_m + RLFi) * iq_rated * ibez + omega_bez * psi_m
u_ss = np.hypot(ud_ss, uq_ss)
log(f'   rated speed: torque curve {T_curve/1e3:.1f} kNm per system -> iq_ref = {iq_rated:.3f} pu (id = 0), bridge voltage '
    f'{u_ss:.0f} V peak = {u_ss/ubez:.2f} pu of ubez = {u_ss/(U_ocv/np.sqrt(3)):.2f} of the linear limit udc/sqrt(3)')
log('   With L_norm = Lq + LFi the flux of the EKF is the "active flux" psi_m + (Ld - Lq)*id: it lies on the d axis for')
log('   any id, so the EKF angle has no saliency error (with Ls it lagged by about atan((Lq_t - Ls)*iq/psi_m)).')
log(f'   Saliency error of the EKF before the fix at rated torque: {np.degrees(np.arctan((L_t/Lbez - Ls_n)*iq_rated)):.1f} deg')
lag = 3 * ts * omega_bez
log(f'   Angle delay of the chain: the EKF angle used at step k is the one of t(k-2) and the vector acts until t(k+1):')
log(f'   3 steps = {np.degrees(lag):.2f} deg at rated speed, i.e. id = {np.sin(lag)*iq_rated:+.3f} pu for the rated iq (not '
    f'compensated, small).')

# -------------------------------------------------------------------------------------------------------------
# 2. the predictive loop as a deadbeat loop
# -------------------------------------------------------------------------------------------------------------
log('\n2. Predictive current loop seen as a deadbeat loop (no vector limit, one axis)')
log('   g = (true voltage / predicted voltage)*(L_prediction / L_true).')
log('   Before: no delay compensation -> e(k+1) = e(k) - g*e(k-1), poles z^2 - z + g = 0.')
for name, Ltrue in (('before, q axis', L_t), ('before, d axis', Ld_m + LFi),
                    ('before, q axis, bridge 1 mH on', L_t + L_br)):
    g = (U_ocv / ubez) * Ls_n / (Ltrue / Lbez)
    z = np.roots([1, -1, g])
    log(f'   {name:30s}: g = {g:.3f}, |z| = {np.abs(z[0]):.3f} '
        f'({"unstable" if np.abs(z[0]) > 1 else "stable"}, oscillation at {abs(np.angle(z[0]))/(2*np.pi*Tc):.0f} Hz)')
log('   With g = 1 (right voltage base, no delay compensation) |z| = 1: undamped oscillation at f_s/6.')
log('   After: delay compensated (current of now predicted with the vector of the last step)')
log('   -> e(k+1) = (1 - g)*e(k-1), poles z^2 = 1 - g: the error is multiplied by 1 - g every two steps.')
for name, Ltrue in (('after, q axis', L_t), ('after, d axis', Ld_m + LFi)):
    g = MPC['L_norm'] / (Ltrue / Lbez)
    log(f'   {name:30s}: g = {g:.3f}, 1 - g = {1-g:+.3f}, |z| = {np.sqrt(abs(1-g)):.3f} (deadbeat for g = 1)')

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

    def __init__(self, rs, ls, a6_pred, q_ekf):
        self.M = ekf_matrices(rs, ls, jm_ekf)
        self.a6_pred = a6_pred
        self.Q = ts * q_ekf * Q_DIAG
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
        P = F_cov @ self.P @ F_cov.T + self.Q
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
    """x = [i_d, i_q, v_c, v_rc] (true rotor frame, DC-link capacitor voltage, battery RC voltage);
    s = leg states (1 = P, 0 = N)."""
    i_d, i_q, vc, vrc = x
    w, th = speed_profile(t, pl['rpm'])
    c, sn = np.cos(th), np.sin(th)
    ia = c * i_d - sn * i_q
    ibeta = sn * i_d + c * i_q
    i_ph = (ia, -0.5 * ia + S3_2 * ibeta, -0.5 * ia - S3_2 * ibeta)
    i_dc = sum(i for k, i in enumerate(i_ph) if s[k] > 0)          # current out of P into the bridge
    # battery current (charging when negative): OCV - R0*ib - vrc = vc + r*(ib - i_dc)
    i_bat = (U_ocv - vrc - vc + pl['r'] * i_dc) / (R0_b + pl['r'])
    iC = i_bat - i_dc
    v_t = vc + pl['r'] * iC
    vph = [v_t * sx for sx in s]
    u_al = 2 / 3 * (vph[0] - 0.5 * vph[1] - 0.5 * vph[2])
    u_be = (vph[1] - vph[2]) / np.sqrt(3)
    u_d = c * u_al + sn * u_be
    u_q = -sn * u_al + c * u_be
    Ldt, Lqt, Rt = pl['L'] * Ld_m + LFi + pl['Lx'], pl['L'] * Lq_m + LFi + pl['Lx'], Rs_m + RLFi + pl['Rx']
    did = (u_d - Rt * i_d + w * Lqt * i_q) / Ldt
    diq = (u_q - Rt * i_q - w * Ldt * i_d - w * psi_m) / Lqt
    return np.array([did, diq, iC / pl['C'], (i_bat - vrc / R1_b) / C1_b]), i_bat


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


def simulate(name, model='new', angle='ekf', dead_time=td, L_factor=1.0, rpm_sim=rpm_nom, speed_out='dio',
             k_sw=None, bridge_passives=False, q_ekf=None, t_end=1.25, nsub=4):
    """One run of the whole chain. model 'new': the fixed model; 'old': the model before the fix.
    angle 'encoder': reference angle and torque-curve speed from the encoder (use_psm_encoder = 1, angle of the
    last step through phase_acquisition, the double integrator observer tracks it without error).
    L_factor: machine inductances of the plant times this factor (the control keeps the values of the init).
    speed_out 'dio': speed output of the EKF block = double integrator observer on the EKF angle (fixed model);
    'state': the speed state of the EKF (model before the fix). bridge_passives: 1 mH + 10 mOhm per phase and
    2 mF across the DC link inside the bridge block (defaults of the block in the dev model). q_ekf: process noise
    of the EKF (Q = ts*q_ekf*diag([ts^2 ts^2 1/2 1/2 1 1])), default 1/100 (fixed model) or 1/10 (before the fix)."""
    pl = dict(L=L_factor, rpm=rpm_sim, Lx=L_br if bridge_passives else 0.0, Rx=R_br if bridge_passives else 0.0,
              C=C_dc + (C_br if bridge_passives else 0.0), r=r_esr)
    if model == 'new':
        ekf = EKF(MPC['R_norm'], MPC['L_norm'], a6_pred=False, q_ekf=Q_EKF_NEW if q_ekf is None else q_ekf)
        mpc = MPCNew()
        k_sw = MPC['k_sw'] if k_sw is None else k_sw
    else:
        ekf = EKF(Rs_n, Ls_n, a6_pred=True, q_ekf=Q_EKF_OLD if q_ekf is None else q_ekf)
        mpc = MPCOld()
        k_sw = 0.0
    dio = DIO()
    N = int(round(t_end / Tc))
    x = np.array([0.0, 0.0, U_ocv, 0.0])    # Simscape starts from steady state (DoDC on): DC link at the battery voltage
    i_prev_meas = np.zeros(2)
    vdc_prev_meas = U_ocv / udc_nom
    th_prev, w_prev = 0.0, 0.0
    u_delay = np.zeros(2)                   # Unit Delay / Unit Delay1 of mp control (after Gain7, Gain1)
    spd_flt = 0.0
    s_prev = np.array([1, 1, 1])
    keys = ('t', 'w', 'w_hat', 'th_err', 'id', 'iq', 'iq_ref', 'Te', 'vdc', 'ibat', 'ia_err', 'ib_err',
            'ia', 'ib', 'ia_ref', 'ib_ref', 'state')
    rec = {k: np.zeros(N) for k in keys}
    t_cnt0 = 1.0
    nchg = 0
    diverged = False
    for k in range(N):
        t = k * Tc
        w_true, th_true = speed_profile(t, rpm_sim)
        i_d, i_q, vc, vrc = x
        c, sn = np.cos(th_true), np.sin(th_true)
        ia_t = (c * i_d - sn * i_q) / ibez
        ib_t = (sn * i_d + c * i_q) / ibez
        # acquisition: quantized phase currents (pu), then alpha/beta; one step of delay (Unit Delay)
        i_a, i_b, i_c = ia_t, -0.5 * ia_t + S3_2 * ib_t, -0.5 * ia_t - S3_2 * ib_t
        iq3 = np.round(np.array([i_a, i_b, i_c]) / q_adc) * q_adc
        i_meas = np.array([2 / 3 * (iq3[0] - 0.5 * iq3[1] - 0.5 * iq3[2]), 2 / 3 * S3_2 * (iq3[1] - iq3[2])])
        i_used = i_prev_meas.copy()
        i_prev_meas = i_meas
        vdc_meas = np.round(vc / udc_nom / q_adc) * q_adc
        vdc_used = vdc_prev_meas
        vdc_prev_meas = vdc_meas
        # EKF outputs (state of the previous step), double integrator observer on the EKF angle
        th_ekf, w_ekf, psa, psb = ekf.outputs()
        _, w_dio = dio.outputs()
        dio.step(th_ekf)
        if speed_out == 'dio':
            w_ekf = w_dio
        # encoder (one step late) or observer; the switch uses the encoder while the speed is zero (omega_th = 0)
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
        # predictive control and voltage to the observers (Gain7/Gain1, then Unit Delay)
        if model == 'new':
            ua_out, ub_out, _, n = mpc.step(vdc_used, i_ref[0], i_ref[1], i_used[0], i_used[1], w_ekf, psa, psb,
                                            MPC['u_scale'], MPC['L_norm'], omega_bez, MPC['R_norm'],
                                            MPC['k_kalman'], k_sw, ts)
            u_mp = np.array([ua_out, ub_out]) * 1.0
            u_ekf = u_delay * 1.0                       # u_psm_scale_ekf = 1
        else:
            ua_out, ub_out, n = mpc.step(1.0, i_ref[0], i_ref[1], i_used[0], i_used[1], w_ekf, psa, psb,
                                         Ls_n, Ls_n, omega_bez, Rs_n, 1.0, 0.0, ts)
            u_mp = np.array([ua_out, ub_out]) * 1.5
            u_ekf = u_delay * u_psm_scale_ekf_old
        u_delay = u_mp
        s = STATES[n].astype(int)
        ekf.step(u_ekf, i_used)
        # legs that change state
        if t >= t_cnt0:
            nchg += int(np.sum(s != s_prev))
        # dead time: a leg keeps its old state when the switch that turns on has to take the current
        if dead_time > 0:
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
                    iq_ref=iq_ref, Te=torque_pu(i_d / ibez, i_q / ibez), vdc=vc, ibat=ibat,
                    ia_err=ia_t - i_ref[0], ib_err=ib_t - i_ref[1], ia=ia_t, ib=ib_t, ia_ref=i_ref[0],
                    ib_ref=i_ref[1], state=n)
        for kk, vv in vals.items():
            rec[kk][k] = vv
        if not np.all(np.isfinite(x)) or abs(x[0]) > 20 * ibez or abs(x[1]) > 20 * ibez:
            diverged = True
            for kk in keys:
                rec[kk] = rec[kk][:k + 1]
            break
    t_win = max(0.0, rec['t'][-1] - t_cnt0) if not diverged else 0.0
    rec['f_dev'] = nchg / 3 / t_win / 2 if t_win > 0 else np.nan    # average switching frequency per device
    rec['diverged'] = diverged
    rec['name'] = name
    return rec


def summary(rec, t0=1.0):
    m = rec['t'] >= t0
    if rec['diverged'] or not np.any(m):
        return None
    w = rec['w'][m]
    ia_err, ib_err = rec['ia_err'][m], rec['ib_err'][m]
    err_rms = np.sqrt(np.mean(ia_err ** 2 + ib_err ** 2) / 2)
    on = np.nonzero(rec['iq_ref'] < -0.05)[0]
    t_on = rec['t'][on[0]] if on.size else np.nan
    m_on = (rec['t'] >= t_on) & (rec['t'] < t0)
    return dict(t_on=t_on, th_start=np.max(np.abs(rec['th_err'][m_on])) if np.any(m_on) else np.nan,
                th_err=np.mean(rec['th_err'][m]), th_std=np.std(rec['th_err'][m]),
                w_ratio=np.mean(rec['w_hat'][m]) / np.mean(w),
                id=np.mean(rec['id'][m]), iq=np.mean(rec['iq'][m]), iq_ref=np.mean(rec['iq_ref'][m]),
                Te=np.mean(rec['Te'][m]), id_std=np.std(rec['id'][m]), iq_std=np.std(rec['iq'][m]), err=err_rms,
                udc=np.mean(rec['vdc'][m]), P=-np.mean(rec['ibat'][m] * rec['vdc'][m]), f_dev=rec['f_dev'],
                zero=np.mean(np.isin(rec['state'][m], (0, 7))))


log('\n3. Switched simulation of the whole chain (1.25 s, values averaged over 1.0 ... 1.25 s, one of the six systems)')
CASES = [
    ('A model before the fix (dev model, bridge passives on)', dict(model='old', speed_out='state', bridge_passives=True)),
    ('A0 model before the fix, bridge passives off', dict(model='old', speed_out='state')),
    ('N1 fixed model (sensorless, dead time 3 us)', dict()),
    ('N2 fixed model, encoder angle (use_psm_encoder = 1)', dict(angle='encoder')),
    ('N3 fixed model without dead time', dict(dead_time=0.0)),
    ('N4 fixed model, shaft at 12 rpm (partial load)', dict(rpm_sim=12.0)),
    ('N5 fixed model with the bridge passives still on (1 mH, 2 mF)', dict(bridge_passives=True)),
    ('L1 fixed model, machine inductances +20 % (control unchanged)', dict(L_factor=1.2)),
    ('L2 fixed model, machine inductances -10 % (control unchanged)', dict(L_factor=0.9)),
    ('L3 fixed model, machine inductances -20 % (control unchanged)', dict(L_factor=0.8)),
    ('L4 fixed model, machine inductances -25 % (control unchanged)', dict(L_factor=0.75)),
    ('L5 fixed model, machine inductances -25 %, encoder angle', dict(L_factor=0.75, angle='encoder')),
    ('E1 as N1 with the legacy EKF noise q_ekf = 1/10', dict(q_ekf=0.1)),
    ('E2 as L2 (inductances -10 %) with q_ekf = 1/10', dict(L_factor=0.9, q_ekf=0.1)),
    ('E3 inductances -15 % with q_ekf = 1/10', dict(L_factor=0.85, q_ekf=0.1)),
    ('S1 as N1 without the switching term (k_sw = 0)', dict(k_sw=0.0)),
    ('S2 as N1 with k_sw = 1e-4', dict(k_sw=1e-4)),
    ('S3 as N1 with k_sw = 1e-3', dict(k_sw=1e-3)),
    ('S4 as N1 with the speed state of the EKF as speed output (no double integrator observer)', dict(speed_out='state')),
]
results = {}
for name, kw in CASES:
    rec = simulate(name, **kw)
    results[name] = rec
    s = summary(rec)
    if s is None:
        log(f'   {name}: DIVERGED at t = {rec["t"][-1]:.3f} s (current above 20 pu)')
        continue
    curve = -lookup_linear(np.mean(rec['w'][rec['t'] >= 1.0]) * rpm_nom, rpm_base, torque_base) / tau_bez
    log(f'   {name}:')
    log(f'      control angle error {s["th_err"]:+6.2f} deg (std {s["th_std"]:.2f}), EKF speed {s["w_ratio"]:.4f} x true | '
        f'id {s["id"]:+.3f} (std {s["id_std"]:.3f}), iq {s["iq"]:+.3f} (std {s["iq_std"]:.3f}) pu for iq_ref {s["iq_ref"]:+.3f} '
        f'| torque {s["Te"]:+.3f} pu, torque curve at the true speed {curve:+.3f} pu ({(s["Te"]/curve-1)*100:+.1f} %)')
    log(f'      current error rms {s["err"]:.4f} pu ({s["err"]*ibez:.1f} A) | DC link {s["udc"]:.0f} V | battery charged with '
        f'{s["P"]/1e3:.0f} kW | switching {s["f_dev"]:.0f} Hz per device on average | zero vector {s["zero"]*100:.0f} % of the steps')
    log(f'      start: the torque reference leaves zero at {s["t_on"]:.3f} s (filtered speed 4 rpm), largest angle error from '
        f'then to 1.0 s: {s["th_start"]:.1f} deg')

# figures
NAME = {c[0].split(' ')[0]: c[0] for c in CASES}
SEL = [NAME['A'], NAME['N1'], NAME['N2']]
fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
for name, st in zip(SEL, ['-', '--', '-.']):
    r = results[name]
    lab = name.split(' ')[0]
    ax[0].plot(r['t'], r['iq'], st, lw=0.8, label='iq true, ' + lab)
    ax[1].plot(r['t'], r['id'], st, lw=0.8, label='id true, ' + lab)
    ax[2].plot(r['t'], r['th_err'], st, lw=0.8, label='control angle error, ' + lab)
    ax[3].plot(r['t'], r['w_hat'], st, lw=0.8, label='EKF speed, ' + lab)
for name, st in zip((NAME['A'], NAME['N1']), ('k-', 'k--')):
    r = results[name]
    ax[0].plot(r['t'], r['iq_ref'], st, lw=0.7, label='iq_ref, ' + name.split(' ')[0])
ax[3].plot(results[NAME['N1']]['t'], results[NAME['N1']]['w'], 'k', lw=0.8, label='true speed')
ax[0].set_ylabel('iq [pu]'); ax[1].set_ylabel('id [pu]'); ax[2].set_ylabel('angle error [deg]')
ax[3].set_ylabel('speed [pu]'); ax[3].set_xlabel('time [s]'); ax[2].set_ylim(-40, 40)
ax[0].set_title('mpc_psm, switched simulation: before the fix (A) and fixed model (N1 sensorless, N2 encoder)')
for a in ax:
    a.grid(True); a.legend(fontsize=7, loc='best')
fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, 'mpc_psm_startup.png'), dpi=200); plt.close(fig)

ZOOM = [NAME['A'], NAME['A0'], NAME['N1'], NAME['S2']]
fig, ax = plt.subplots(len(ZOOM), 1, figsize=(11, 2.6 * len(ZOOM)), sharex=True)
for a, name in zip(ax, ZOOM):
    r = results[name]
    m = (r['t'] >= 1.15) & (r['t'] <= 1.25)
    a.plot(r['t'][m], r['ia'][m], lw=0.7, label='i_alpha true')
    a.plot(r['t'][m], r['ia_ref'][m], 'k', lw=0.8, label='i_alpha reference')
    a.set_title(name, fontsize=9); a.set_ylabel('[pu]'); a.grid(True); a.legend(fontsize=7, loc='upper right')
ax[-1].set_xlabel('time [s]')
fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, 'mpc_psm_current_zoom.png'), dpi=200); plt.close(fig)
OUT.close()
