"""Offline models of the three grid PLLs of frt_dqpll.slx (three_phase_dqPLLs_analysis, pwr-control/modelization-and-control).

Companion module of the note three_phase_plls_comparison (pwr-control/repositories-documentation). Every block is
written step by step as the Simulink diagram and the library C code do it (control period ts, one-step delays,
clamps), so that the three loops can be run on the same logged grid voltage:
  dq_pll    : Clarke -> normalisation -> rotation(theta) -> PI loop (clamped integrator, scheduled kp) -> theta, DI observer
  fht_pll   : Clarke -> FHT on alpha and beta (in-phase + quadrature) -> positive sequence (sign -1) -> rotation -> loop
  ddsrf_pll : FHT on each phase -> Clarke -> DDSRF decoupling (LPF tau) -> loop fed with the NEGATIVE sequence (as in
              the model) or with the positive sequence (fixed); options: PI gains, initial frequency, integrator clamp,
              pre-filter on/off, DC branch
The FHT-PLL branch was checked against the logged run sim_results_01 of the model: phase error below 4e-5 rad,
FHT signals below 1e-6 pu. numpy only.
"""

import numpy as np

TWO_PI = 2 * np.pi

def wrap(x):                              # Stateflow bounder: -pi + mod(u + 3 pi, 2 pi)
    return -np.pi + np.mod(x + 3 * np.pi, TWO_PI)

def clarke(a, b, c):                      # amplitude invariant
    return 2.0 / 3.0 * (a - 0.5 * b - 0.5 * c), (b - c) / np.sqrt(3.0)

def rot_pos(al, be, g):                   # alphabeta2xieta_pos
    return al * np.cos(g) + be * np.sin(g), -al * np.sin(g) + be * np.cos(g)

def rot_neg(al, be, g):                   # alphabeta2xieta_neg
    return al * np.cos(g) - be * np.sin(g), al * np.sin(g) + be * np.cos(g)

def acker_observer(A, C, poles):
    """L = acker(A', C', poles)' for a 2x2 system (Ackermann on the observability form)."""
    n = A.shape[0]
    At, Bt = A.T, C.T.reshape(n, 1)
    ctrb = np.hstack([np.linalg.matrix_power(At, k) @ Bt for k in range(n)])
    phi = np.eye(n)
    for p in poles:
        phi = phi @ (At - p * np.eye(n))
    K = np.zeros((1, n)); K[0, -1] = 1.0
    K = K @ np.linalg.inv(ctrb) @ phi
    return K.T.reshape(n)                  # column L

class Params:
    """numbers of ctrl_afe_setup(ts, omega_base) + grid_three_phase_emulator used by the PLL blocks"""
    def __init__(self, ts=125e-6, omega_n=TWO_PI * 50):
        self.ts, self.omega_n = ts, omega_n
        # PI gains (ctrl_afe_setup)
        self.pll_i1, self.pll_p, self.pll_p_frt = 80.0, 1.0, 0.2
        self.pll_i1_ddsrt, self.pll_p_ddsrt = 80.0, 1.0
        self.pll_i1_fht, self.pll_p_fht = 80.0, 1.0
        self.tau_ddsrf = 1.0 / omega_n
        # double integrator observer: Aso = [0 1; 0 0], Adso = I + Aso ts, Cso = [1 0], poles [-1 -4] omega
        Aso = np.array([[0.0, 1.0], [0.0, 0.0]]); self.Adso = np.eye(2) + Aso * ts; self.Cso = np.array([[1.0, 0.0]])
        pd = np.exp(np.array([-1.0, -4.0]) * omega_n * ts)
        self.Ldso = acker_observer(self.Adso, self.Cso, pd)
        # first harmonic tracker (fht_grid): omega_fht = omega, delta 0.05, Ad = I + A ts (forward Euler) for the design
        self.omega_fht, self.delta_fht = omega_n, 0.05
        Afht = np.array([[0.0, 1.0], [-omega_n ** 2, -self.delta_fht * omega_n]]); Cfht = np.array([[1.0, 0.0]])
        Ad = np.eye(2) + Afht * ts
        self.L_fht = acker_observer(Afht, Cfht, np.array([-1.0, -4.0]) * omega_n)          # continuous gains
        self.Ld_fht = acker_observer(Ad, Cfht, np.exp(ts * np.array([-1.0, -4.0]) * omega_n))  # discrete gains (used)
        self.t_reset = 0.0125                                                              # Step block of the FHT reset

class FHT:
    """first_harmonic_tracker_process of the library (Taylor-2 state matrix, forward-Euler-designed gains).
    output u_hat = state1 BEFORE the update (one step late), u_hat_dot = state2 AFTER the update."""
    def __init__(self, p, dtype=np.float64):
        self.p, self.dt = p, dtype
        a = -p.omega_fht ** 2; b = -p.delta_fht * p.omega_fht; ts = p.ts
        self.a11 = 1.0 + 0.5 * a * ts * ts; self.a12 = ts + 0.5 * b * ts * ts
        self.a21 = a * ts + 0.5 * a * b * ts * ts; self.a22 = 1.0 + b * ts + 0.5 * (a + b * b) * ts * ts
        self.l1, self.l2 = float(p.Ld_fht[0]), float(p.Ld_fht[1])
        self.reset()
    def reset(self):
        self.s1 = self.s2 = self.out = 0.0
    def step(self, u, reset=False):
        if reset: self.reset()
        err = u - self.out
        s1 = self.l1 * err + self.a11 * self.s1 + self.a12 * self.s2
        s2 = self.l2 * err + self.a21 * self.s1 + self.a22 * self.s2
        self.out = self.s1            # old state1
        self.s1, self.s2 = s1, s2
        return self.out, self.s2      # output_pu_hat, output_pu_hat_dot

class LPF:
    """library discrete-time-low-pass-filter: y[k] = ts/tau x[k-1] + (1 - ts/tau) y[k-1] (no direct feedthrough).
    output() advances the state and returns y[k]; push(x) stores x[k] for the next step."""
    def __init__(self, ts, tau):
        self.g = ts / tau; self.x1 = 0.0; self.y = 0.0
    def output(self):
        self.y = self.g * self.x1 + (1.0 - self.g) * self.y
        return self.y
    def push(self, x):
        self.x1 = x
    def step(self, x):
        y = self.output(); self.push(x); return y

class DIObserver:
    """double_integrator_observer: x[k+1] = Adso x + Ldso wrap(theta - wrap(Cso x)); omega_hat = x2 (from the delayed state)"""
    def __init__(self, p):
        self.p = p; self.x = np.zeros(2)
    def step(self, theta):
        e = wrap(theta - wrap(self.x[0]))
        omega_hat = self.x[1]
        self.x = self.p.Adso @ self.x + self.p.Ldso * e
        return omega_hat

class StageB:
    """pll_stageB: e = sign * u_eta * Vnorm; i1 = [clamp](i1 + ki ts e); w_pu = kp e + i1; theta = wrap(theta + ts omega_n w_pu);
    the outputs (phase, omega) are the values computed at the previous step (Unit Delay2, observer state)."""
    def __init__(self, p, kp, ki, sign=1.0, clamp=None, i1_init=0.0, theta_init=0.0):
        """clamp: None, a number c (|i1| <= c, as the dq_pll block) or a pair (lo, hi); i1_init: initial integrator
        value in pu of omega_n (the model starts at 0 = zero frequency)."""
        self.p, self.kp, self.ki, self.sign, self.clamp = p, kp, ki, sign, clamp
        self.i1 = i1_init; self.theta = theta_init; self.theta_out = theta_init; self.obs = DIObserver(p)
        self.obs.x[:] = [theta_init, i1_init * p.omega_n]
    def step(self, u_eta, vnorm, kp=None):
        kp = self.kp if kp is None else kp
        e = self.sign * u_eta * vnorm
        i1 = self.i1 + self.ki * self.p.ts * e
        if self.clamp is not None:
            lo, hi = (-self.clamp, self.clamp) if np.isscalar(self.clamp) else self.clamp
            i1 = min(max(i1, lo), hi)
        self.i1 = i1
        w_pu = kp * e + i1
        theta_new = wrap(self.theta + self.p.ts * self.p.omega_n * w_pu)
        # outputs of this step: the phase computed at the previous step and the observer speed from its delayed state
        phase_out = self.theta                      # Unit Delay2
        omega_out = self.obs.step(phase_out) / self.p.omega_n
        self.theta = theta_new
        return phase_out, omega_out, e, w_pu

def vnorm_of(al, be):
    return 1.0 / np.sqrt(max(al * al + be * be, 0.02))

def run_dq_pll(uabc, t, p, kp_sched=None, ki=None, kp=None, clamp=2.0, i1_init=0.0, theta_init=0.0):
    """uabc: (3, N) in pu at ts. kp_sched: optional array (N,) of the scheduled kp (pll_p or pll_p_frt)."""
    N = uabc.shape[1]
    sb = StageB(p, p.pll_p if kp is None else kp, p.pll_i1 if ki is None else ki, sign=+1.0, clamp=clamp, i1_init=i1_init, theta_init=theta_init)
    phase = np.zeros(N); omega = np.zeros(N); err = np.zeros(N); xi = np.zeros(N); eta = np.zeros(N)
    for k in range(N):
        al, be = clarke(*uabc[:, k])
        vn = vnorm_of(al, be)
        x, y = rot_pos(al, be, sb.theta)          # stage A uses the phase output of stage B (previous step)
        xi[k], eta[k] = x, y
        phase[k], omega[k], err[k], _ = sb.step(y, vn, kp=None if kp_sched is None else kp_sched[k])
    return dict(phase=phase, omega=omega, e=err, xi=xi, eta=eta)

def run_fht_pll(uabc, t, p, kp=None, ki=None, clamp=None, i1_init=0.0, theta_init=0.0, reset=True):
    N = uabc.shape[1]
    fa, fb = FHT(p), FHT(p)
    sb = StageB(p, p.pll_p_fht if kp is None else kp, p.pll_i1_fht if ki is None else ki, sign=-1.0, clamp=clamp, i1_init=i1_init, theta_init=theta_init)
    out = {k: np.zeros(N) for k in ('phase', 'omega', 'e', 'xi', 'eta', 'u_fht_a', 'v_fht_a', 'u_fht_b', 'v_fht_b', 'u_sr_a', 'u_sr_b')}
    for k in range(N):
        rst = reset and (t[k] < p.t_reset)
        al, be = clarke(*uabc[:, k])
        ua, da = fa.step(al, rst); ub, db = fb.step(be, rst)
        va, vb = da / p.omega_fht, db / p.omega_fht
        usa = -0.5 * (ua + vb); usb = -0.5 * (ub - va)
        vn = vnorm_of(usa, usb)
        x, y = rot_pos(usa, usb, sb.theta)
        out['phase'][k], out['omega'][k], out['e'][k], _ = sb.step(y, vn)
        out['xi'][k], out['eta'][k] = x, y
        out['u_fht_a'][k], out['v_fht_a'][k], out['u_fht_b'][k], out['v_fht_b'][k] = ua, va, ub, vb
        out['u_sr_a'][k], out['u_sr_b'][k] = usa, usb
    return out

class DDSRF:
    """sequences/ddsrf block: rotations at theta (pos) and -theta (neg), cross decoupling with the filtered other
    sequence rotated at 2 theta, four library LPFs (tau). Step k: y[k] of the LPFs from their states, decoupling
    terms from y[k], then the new LPF inputs are stored."""
    def __init__(self, p, tau=None, dc_tau=None):
        """dc_tau: time constant of an optional third (DC) branch in alpha-beta: the DC offset is estimated from the
        residual alpha-beta minus the reconstructed positive and negative sequences and removed from both frames."""
        tau = p.tau_ddsrf if tau is None else tau
        self.lp = [LPF(p.ts, tau) for _ in range(4)]
        self.dc = [LPF(p.ts, dc_tau) for _ in range(2)] if dc_tau else None
        self.dc_est = (0.0, 0.0)
    def step(self, al, be, th):
        th2 = wrap(2.0 * th)
        y = [f.output() for f in self.lp]                 # pos_xi, pos_eta, neg_xi, neg_eta at step k
        if self.dc is not None:
            dca, dcb = self.dc[0].output(), self.dc[1].output(); self.dc_est = (dca, dcb)
            ra = rot_neg(y[0], y[1], th); rb = rot_pos(y[2], y[3], th)        # sequences back to alpha-beta
            self.dc[0].push(al - ra[0] - rb[0]); self.dc[1].push(be - ra[1] - rb[1])
            al, be = al - dca, be - dcb
        pr = rot_pos(al, be, th); nr = rot_neg(al, be, th)
        dec_p = rot_pos(y[2], y[3], th2)                  # alphabeta2xieta_pos1(neg filtered, 2 theta)
        dec_n = rot_neg(y[0], y[1], th2)                  # alphabeta2xieta_neg1(pos filtered, 2 theta)
        self.lp[0].push(pr[0] - dec_p[0]); self.lp[1].push(pr[1] - dec_p[1])
        self.lp[2].push(nr[0] - dec_n[0]); self.lp[3].push(nr[1] - dec_n[1])
        return y[0], y[1], y[2], y[3]

def run_ddsrf_pll(uabc, t, p, feed='neg', sign=-1.0, kp=None, ki=None, prefilter=True, tau=None, clamp=None,
                  i1_init=0.0, theta_init=0.0, norm='total', reset=True, dc_tau=None):
    """feed='neg' and sign=-1: the model as it is (stage B fed with ddsrf outputs 3-4 through a gain -1);
    feed='pos', sign=+1: the loop on the decoupled positive sequence. norm='total': 1/|alpha+j beta| as the model,
    norm='pos': 1/|positive sequence output|."""
    N = uabc.shape[1]
    tau = p.tau_ddsrf if tau is None else tau
    f = [FHT(p) for _ in range(3)]
    sb = StageB(p, p.pll_p_ddsrt if kp is None else kp, p.pll_i1_ddsrt if ki is None else ki, sign=sign, clamp=clamp, i1_init=i1_init, theta_init=theta_init)
    out = {k: np.zeros(N) for k in ('phase', 'omega', 'e', 'pos_xi', 'pos_eta', 'neg_xi', 'neg_eta', 'alpha', 'beta')}
    dd = DDSRF(p, tau, dc_tau)
    for k in range(N):
        rst = reset and (t[k] < p.t_reset)
        if prefilter:
            a, b, c = (f[i].step(uabc[i, k], rst)[0] for i in range(3))
        else:
            a, b, c = uabc[:, k]
        al, be = clarke(a, b, c)
        p_xi, p_eta, n_xi, n_eta = dd.step(al, be, sb.theta)
        vn = vnorm_of(al, be) if norm == 'total' else vnorm_of(p_xi, p_eta)
        u_eta = n_eta if feed == 'neg' else p_eta
        out['phase'][k], out['omega'][k], out['e'][k], _ = sb.step(u_eta, vn)
        out['pos_xi'][k], out['pos_eta'][k], out['neg_xi'][k], out['neg_eta'][k] = p_xi, p_eta, n_xi, n_eta
        out['alpha'][k], out['beta'][k] = al, be
    return out

def three_phase(t, Vp=1.0, Vn=0.0, phi_n=0.0, f=50.0, phi_p=0.0, harmonics=()):
    """balanced set + negative sequence + harmonics [(order, amplitude, sequence +1/-1)], amplitude in pu of the phase peak"""
    w = TWO_PI * f
    ph = np.array([0.0, -TWO_PI / 3, TWO_PI / 3])
    u = np.zeros((3, t.size))
    for i in range(3):
        # a component of angle th(t) in the alpha-beta plane gives phase i = cos(th + ph[i]); negative sequence: th = -wt + phi_n
        u[i] = Vp * np.cos(w * t + phi_p + ph[i]) + Vn * np.cos(-(w * t) + phi_n + ph[i])
        for (h, amp, seq) in harmonics:
            u[i] += amp * np.cos(seq * h * w * t + ph[i])
    return u
