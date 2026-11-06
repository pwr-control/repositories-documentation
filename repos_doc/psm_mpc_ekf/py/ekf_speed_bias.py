"""Why is the speed state of the mpc_psm EKF 0.4 % low? Isolated test of the EKF (same equations as the block,
see design_check_mpc_psm.py class EKF) on exact data of its own model: no switching, no dead time, no saliency.

Plant (pu, time in 1/omega_bez): L di/dt = u - R i - e, e = j*w*psi, dpsi/dt = j*w*psi, |psi| = 1 (active flux),
integrated exactly (matrix exponential). The voltage of each step (zero-order hold) is the one
that brings the current to the reference rotating with the rotor (deadbeat on the exact plant), so the current
is iq_ref on the q axis at every sample. At step k the EKF gets the voltage applied on [k-1, k) and i(k), as in
the model after the fix.
Variants: forward Euler prediction (the block), exact prediction (matrix exponential, as the plant), Euler + exact rotation
of the flux only, Taylor-2 prediction; q_ekf 1/100 and 1/10; clock 1.001 (data at ts/1.001, EKF with ts).
"""
import numpy as np


def expm(Mx, terms=24):
    """Matrix exponential, Taylor series with scaling and squaring (small matrices, small norms here)."""
    nrm = np.linalg.norm(Mx, 1)
    sq = max(0, int(np.ceil(np.log2(nrm))) + 1) if nrm > 0.5 else 0
    A = Mx / 2 ** sq
    E = np.eye(Mx.shape[0]); T = np.eye(Mx.shape[0])
    for k in range(1, terms):
        T = T @ A / k
        E = E + T
    for _ in range(sq):
        E = E @ E
    return E

fpwm = 15e3
ts = 1 / fpwm
n_sys, pwr_nom, i_nom, rpm_nom, poles, eta = 6, 1600e3, 2062.0, 17.8, 104, 96.0
Ld, Lq = 1.04e-3, 1.26e-3
omega_m_bez = rpm_nom / 60 * 2 * np.pi
torque_nom = pwr_nom / omega_m_bez
Rs = (pwr_nom / 3) * (1 - eta / 100) / i_nom ** 2
Rs_m, Lq_m = Rs * n_sys, Lq * n_sys
pp = poles / 2
ibez = i_nom / n_sys * np.sqrt(2)
psi_m = torque_nom / n_sys / (1.5 * pp * ibez)
omega_bez = omega_m_bez * pp
ubez = psi_m * omega_bez
Xbez = ubez / ibez
Lbez = Xbez / omega_bez
LFi = 230e-6
RLFi = 157 * 0.05 * LFi
L_n = (Lq_m + LFi) / Lbez
R_n = (Rs_m + RLFi) / Xbez
jm_ekf = 1e6
Q_DIAG = np.diag([ts ** 2, ts ** 2, 0.5, 0.5, 1, 1])
R_EKF = ts * 1.0 * np.eye(2)
J2 = np.array([[0.0, -1.0], [1.0, 0.0]])


def ekf_matrices(rs, ls, jm):
    A1 = np.zeros((6, 6)); A1[0, 0] = A1[1, 1] = -rs / ls; A1[4, 5] = -1 / jm
    A2 = np.zeros((6, 6)); A2[4, 3] = -1 / jm
    A3 = np.zeros((6, 6)); A3[4, 2] = 1 / jm
    A4 = np.zeros((6, 6)); A4[1, 4] = -1 / ls; A4[3, 4] = 1; A4[4, 1] = 1 / jm
    A5 = np.zeros((6, 6)); A5[0, 4] = 1 / ls; A5[2, 4] = -1; A5[4, 0] = -1 / jm
    A6 = np.zeros((6, 6)); A6[0, 3] = 1 / ls; A6[1, 2] = -1 / ls; A6[2, 3] = -1; A6[3, 2] = 1
    B = np.zeros((6, 2)); B[0, 0] = B[1, 1] = 1 / ls
    return A1, A2, A3, A4, A5, A6, B


M = ekf_matrices(R_n, L_n, jm_ekf)


def f_model(x, u):
    """Continuous model of the EKF (time in 1/omega_bez)."""
    A1, A2, A3, A4, A5, A6, B = M
    J5 = A1 + A2 * x[0] + A3 * x[1] + A4 * x[2] + A5 * x[3]
    return J5 @ x + B @ u


def disc(w, h):
    """Exact discretization of the plant / of the EKF model with the speed w held over the step h (pu time):
    s = [i_alpha, i_beta, psi_alpha, psi_beta], ds/dt = A s + B u."""
    A = np.zeros((4, 4))
    A[0:2, 0:2] = -R_n / L_n * np.eye(2)
    A[0:2, 2:4] = -w / L_n * J2
    A[2:4, 2:4] = w * J2
    Bm = np.zeros((4, 2)); Bm[0:2, :] = np.eye(2) / L_n
    Mx = np.zeros((6, 6)); Mx[0:4, 0:4] = A; Mx[0:4, 4:6] = Bm
    E = expm(Mx * h)
    return E[0:4, 0:4], E[0:4, 4:6]


def run(variant='euler', q_ekf=0.01, w=1.0, iq=-0.874, clock=1.0, t_end=2.0, w0=None):
    h_true = ts / clock * omega_bez          # true step (pu time)
    h = ts * omega_bez                       # step used by the EKF
    n = int(t_end / ts)
    th0 = 0.3
    psi = np.array([np.cos(th0), np.sin(th0)])
    rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    i = rot(th0) @ np.array([0.0, iq])
    x = np.array([i[0], i[1], psi[0], psi[1], w if w0 is None else w0, 0.0])   # EKF starts on the truth
    P = np.zeros((6, 6))
    Q = ts * q_ekf * Q_DIAG
    I6 = np.eye(6)
    A1, A2, A3, A4, A5, A6, B = M
    w_log, th_err = [], []
    th = th0
    Phi, Gam = disc(w, h_true)
    sx = np.concatenate([i, psi])
    for k in range(n):
        th_next = th + w * h_true
        i_target = rot(th_next) @ np.array([0.0, iq])
        u = np.linalg.solve(Gam[0:2, :], i_target - Phi[0:2, :] @ sx)
        sx = Phi @ sx + Gam @ u
        i, psi = sx[0:2], sx[2:4]
        th = th_next
        # EKF step with u (applied on the last interval) and the current at the end of it
        J5 = A1 + A2 * x[0] + A3 * x[1] + A4 * x[2] + A5 * x[3]
        F_cov = I6 + h * (J5 + A6 * x[4])
        if variant == 'euler':
            xp = (I6 + h * J5) @ x + h * (B @ u)
        elif variant == 'exact':
            Pe, Ge = disc(x[4], h)
            xp = x.copy()
            xp[0:4] = Pe @ x[0:4] + Ge @ u
        elif variant == 'euler_rot':          # Euler for the current, exact rotation for the flux
            xp = (I6 + h * J5) @ x + h * (B @ u)
            xp[2:4] = rot(x[4] * h) @ x[2:4]
        elif variant == 'taylor2':            # x + h f + h^2/2 * Jf * f (second order)
            fx = f_model(x, u)
            Jf = J5 + A6 * x[4]
            xp = x + h * fx + h * h / 2 * (Jf @ fx)
        Pp = F_cov @ P @ F_cov.T + Q
        S = Pp[0:2, 0:2] + R_EKF
        K = Pp[:, 0:2] @ np.linalg.inv(S)
        P = Pp - K @ Pp[0:2, :]
        x = xp + K @ (i - xp[0:2])
        if k * ts >= t_end - 0.5:
            w_log.append(x[4])
            th_err.append(np.angle(np.exp(1j * (np.arctan2(x[3], x[2]) - th))))
    return np.mean(w_log) / w, np.degrees(np.mean(th_err)), np.hypot(x[2], x[3])


if __name__ == '__main__':
    print(f'ts = {ts*1e6:.2f} us, theta per step at 1 pu = ts*omega_bez = {ts*omega_bez:.5f} rad '
          f'({np.degrees(ts*omega_bez):.3f} deg); theta/2 = {ts*omega_bez/2*100:.3f} %; L = {L_n:.4f}, R = {R_n:.4f}')
    cases = [
        ('Euler (the block), q 1/100', dict()),
        ('Euler, q 1/10', dict(q_ekf=0.1)),
        ('Euler, no load (iq = 0)', dict(iq=0.0)),
        ('Euler, half speed (w = 0.5)', dict(w=0.5)),
        ('Euler + clock 1.001', dict(clock=1.001)),
        ('exact prediction (expm) ', dict(variant='exact')),
        ('Euler current + exact flux rotation', dict(variant='euler_rot')),
        ('Taylor-2 prediction', dict(variant='taylor2')),
        ('Taylor-2 + clock 1.001', dict(variant='taylor2', clock=1.001)),
    ]
    for name, kw in cases:
        r, the, psin = run(**kw)
        print(f'{name:40s}: speed state {r:.5f} x true ({(r-1)*100:+.3f} %), angle error {the:+.3f} deg, |psi| {psin:.4f}')
