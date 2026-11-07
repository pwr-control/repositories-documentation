"""Single-phase PLLs of the model single_phase_pll_analysis (modelization-and-control,
foundations_of_power_electronics/single_phase_dqPLLs_analysis), written step by step as the Simulink blocks
compute them: control period ts, unit delays, saturations, wrapping of the angles.

    park_pll   block single_phase_pll: quadrature signal generator made of a direct and an inverse Park
               transform with two low-pass filters (the SOGI of the inverse Park), P controller on eta with the
               nominal frequency as feed-forward, double-integrator observer on the output phase
    fht_pll    block FHT-PLL: first harmonic tracker (FHT) as quadrature generator, a second FHT on alpha and on
               beta, positive-sequence calculator, normalization, PI loop (stage B)
    ddsrf_pll  block DDSRF-PLL: the same FHT as quadrature generator, decoupled double synchronous reference frame
               (DDSRF) with four low-pass filters, normalization, PI loop with the frequency integrator clamped

Parameters: the dict P of default_params(), the same numbers as init_model.m of the model (ctrl_afe_setup of the
library for the FHT, the observer and the stage B gains).
"""
import numpy as np

TWO_PI = 2.0 * np.pi


def wrap(x):
    """bounder of the model: -pi + mod(x + 3 pi, 2 pi), angle in (-pi, pi]."""
    return -np.pi + np.mod(x + 3.0 * np.pi, TWO_PI)


def acker(A, C, poles):
    """Observer gain L (column) such that eig(A - L C) = poles (Ackermann on the dual pair, 2 states)."""
    A = np.asarray(A, float)
    C = np.asarray(C, float).reshape(1, -1)
    At, Bt = A.T, C.T
    n = A.shape[0]
    ctrb = np.hstack([np.linalg.matrix_power(At, k) @ Bt for k in range(n)])
    coeffs = np.real(np.poly(poles))                    # s^n + c1 s^(n-1) + ...
    phi = np.zeros_like(At)
    for k, c in enumerate(coeffs):
        phi = phi + c * np.linalg.matrix_power(At, n - k)
    e_last = np.zeros((1, n))
    e_last[0, -1] = 1.0
    K = e_last @ np.linalg.inv(ctrb) @ phi
    return K.T.reshape(-1)                              # L = K'


def default_params(ts=125e-6, f0=50.0):
    """Numbers of init_model.m: ctrl_afe_setup(ts, 2 pi f0) and the local parameters of the Park PLL."""
    w0 = TWO_PI * f0
    P = dict(ts=ts, w0=w0)
    # first harmonic tracker (ctrl_afe_setup.first_harmonic_tracker): damped oscillator, forward Euler model,
    # discrete poles exp(ts * [-1 -4] w0)
    delta = 0.05
    Afht = np.array([[0.0, 1.0], [-w0 ** 2, -delta * w0]])
    Ad = np.eye(2) + Afht * ts
    Cf = np.array([1.0, 0.0])
    Ld = acker(Ad, Cf, np.exp(ts * np.array([-1.0, -4.0]) * w0))
    P.update(fht_delta=delta, Afht=Afht, Ad_fht=Ad, C_fht=Cf, Ld_fht=Ld, w_fht=w0)
    # double-integrator observer (ctrl_afe_setup.double_integrator_observer)
    Ado = np.array([[1.0, ts], [0.0, 1.0]])
    Co = np.array([1.0, 0.0])
    Ldo = acker(Ado, Co, np.exp(np.array([-1.0, -4.0]) * w0 * ts))
    P.update(Adso=Ado, Cso=Co, Ldso=Ldo)
    # stage B gains (FHT-PLL: library defaults; DDSRF-PLL: as in frt_dqpll)
    P.update(kp_fht=1.0, ki_fht=80.0, kp_ddsrf=0.32, ki_ddsrf=8.0, tau_ddsrf=1.0 / w0,
             clamp_ddsrf=(0.5, 1.5), i1_init_ddsrf=0.0)
    # Park PLL (block single_phase_pll): low-pass filters of the QSG, P controller, frequency feed-forward 1 pu
    P.update(tau_qsg=2.0 / w0, kp_park=1.0, ki_park=0.0, sat_park=(-2.0, 2.0))
    return P


def fht_variant(P, mode='class', poles=(-1.0, -4.0), pole_dc=-2.0):
    """Copy of P with the first harmonic tracker of another design.
    mode 'class'     damped oscillator (delta 0.05), forward Euler: the model (ctrl_afe_setup)
         'undamped'  undamped oscillator, exact discretization Ad = expm(A ts): option fht_undamped of init_model.m
         'dc'        'undamped' plus a third state for a DC offset of the input (u = x1 + x3), not in the model
    poles: observer poles in units of w0 (discrete exp(ts p w0)), pole_dc the third one for mode 'dc'."""
    from scipy.linalg import expm
    Q = dict(P)
    ts, w = P['ts'], P['w_fht']
    p = np.asarray(poles, complex) * w
    if mode == 'class':
        A = np.array([[0.0, 1.0], [-w ** 2, -P['fht_delta'] * w]])
        Ad, C = np.eye(2) + A * ts, np.array([1.0, 0.0])
    elif mode == 'undamped':
        A = np.array([[0.0, 1.0], [-w ** 2, 0.0]])
        Ad, C = expm(A * ts), np.array([1.0, 0.0])
    elif mode == 'dc':
        A = np.array([[0.0, 1.0], [-w ** 2, 0.0]])
        Ad = np.eye(3)
        Ad[:2, :2] = expm(A * ts)
        C = np.array([1.0, 0.0, 1.0])
        p = np.append(p, pole_dc * w)
    else:
        raise ValueError(mode)
    Q.update(Ad_fht=Ad, C_fht=C, Ld_fht=acker(Ad, C, np.exp(ts * p)))
    return Q


class LPF:
    """discrete-time-low-pass-filter of the library: y[k] = ts/tau x[k-1] + (1 - ts/tau) y[k-1] (no feedthrough)."""

    def __init__(self, ts, tau):
        self.a = ts / tau
        self.x = 0.0
        self.y = 0.0

    def output(self):
        return self.a * self.x + (1.0 - self.a) * self.y

    def push(self, x, y):
        self.x, self.y = x, y


class FHT:
    """first harmonic tracker in Simulink blocks: x[k+1] = Ad x[k] + Ld (u[k] - C x[k]); outputs of step k are the
    state x[k] (a priori estimate): u_hat = x1, quadrature q = sgn * x2 / w_fht (a DC state x3, if any, is left out)."""

    def __init__(self, P, sgn):
        self.Ad, self.Ld, self.w = P['Ad_fht'], P['Ld_fht'], P['w_fht']
        self.C = P['C_fht']
        self.sgn = sgn
        self.x = np.zeros(self.Ad.shape[0])

    def step(self, u):
        x1, x2 = self.x[0], self.x[1]
        out = (x1, self.sgn * x2 / self.w)
        self.x = self.Ad @ self.x + self.Ld * (u - self.C @ self.x)
        return out


class DIObserver:
    """double_integrator_observer of stage B: theta_hat = wrap(x1), x+ = Ad x + Ld wrap(theta - theta_hat);
    omega_hat = x2 (rad/s)."""

    def __init__(self, P):
        self.A, self.L = P['Adso'], P['Ldso']
        self.x = np.zeros(2)

    def step(self, theta):
        th_hat = wrap(self.x[0])
        w_hat = self.x[1]
        e = wrap(theta - th_hat)
        self.x = self.A @ self.x + self.L * e
        return th_hat, w_hat


def park_pll(u, P, kp=None, ki=None, tau=None, eta_filtered=False, normalize=False, vmin=0.1, dc_gain=0.0):
    """block single_phase_pll. Output phase = the observer estimate (phase_pll_hat), omega in pu (omega_pll_hat).
    The phase detector uses the raw input as alpha and the generator output as beta.
    Variants not in the model: eta_filtered, the PI fed by the filtered eta; normalize, eta divided by the amplitude
    sqrt(xf^2 + ef^2) of the filtered xi and eta (at least vmin); dc_gain > 0, a DC estimate
    d[k+1] = d[k] + ts dc_gain (u[k] - d[k] - alpha_hat[k]) subtracted from the input (alpha_hat = inverse Park of
    the filtered xi and eta: the third integrator of a SOGI with DC rejection)."""
    ts, w0 = P['ts'], P['w0']
    kp = P['kp_park'] if kp is None else kp
    ki = P['ki_park'] if ki is None else ki
    tau = P['tau_qsg'] if tau is None else tau
    lo, hi = P['sat_park']
    fx, fe = LPF(ts, tau), LPF(ts, tau)
    obs = DIObserver(P)
    th = 0.0           # Unit Delay1
    I = 0.0            # Unit Delay4 of pi-ctrl
    N = len(u)
    out = {k: np.zeros(N) for k in ('phase', 'omega', 'theta', 'eta', 'beta', 'freq')}
    dc = 0.0
    for k in range(N):
        xf, ef = fx.output(), fe.output()
        c, s = np.cos(th), np.sin(th)
        beta = xf * s + ef * c                  # xieta2alphabeta, output beta
        uk = u[k] - dc
        if dc_gain > 0.0:
            dc = dc + ts * dc_gain * (uk - (xf * c - ef * s))
        xi = uk * c + beta * s                  # alphabeta2xieta
        eta = -uk * s + beta * c
        err = ef if eta_filtered else eta       # ueta_pll (-1) and pi-ctrl (xref - x)
        if normalize:
            err = err / max(np.hypot(xf, ef), vmin)
        p_term = kp * err
        v = p_term + ki * ts * err + I
        sat = min(max(v, lo), hi)
        I = sat - p_term                        # back-calculation of the block pi-ctrl
        f = sat + 1.0                           # Constant 1: nominal frequency as feed-forward
        th_hat, w_hat = obs.step(th)
        out['phase'][k], out['omega'][k] = th_hat, w_hat / w0
        out['theta'][k], out['eta'][k], out['beta'][k], out['freq'][k] = th, eta, beta, f
        fx.push(xi, xf)
        fe.push(eta, ef)
        th = wrap(th + w0 * ts * f)
    return out


def stage_b(e, th_prev, I, kp, ki, P, clamp=None):
    """pll_stageB: i1 = ki ts e + I (clamped for the DDSRF), omega_pu = kp e + i1, theta+ = wrap(theta + w0 ts omega)."""
    i1 = ki * P['ts'] * e + I
    if clamp is not None:
        i1 = min(max(i1, clamp[0]), clamp[1])
    w_pu = kp * e + i1
    return wrap(th_prev + P['w0'] * P['ts'] * w_pu), i1, w_pu


def fht_pll(u, P, kp=None, ki=None, one_stage=False):
    """block FHT-PLL: FHTalpha1 (quadrature generator), FHTalpha2/3, u_sr = -1/2 (u - j v), normalization,
    stage B with the gain -1. Output phase = Unit Delay2 (theta), omega = observer x2 in pu.
    one_stage: variant without FHTalpha2/3, stage B on the outputs of FHTalpha1 (gain +1), not in the model."""
    kp = P['kp_fht'] if kp is None else kp
    ki = P['ki_fht'] if ki is None else ki
    f1, f2, f3 = FHT(P, -1.0), FHT(P, +1.0), FHT(P, +1.0)
    obs = DIObserver(P)
    th, I = 0.0, 0.0
    N = len(u)
    out = {k: np.zeros(N) for k in ('phase', 'omega', 'e', 'alpha', 'beta', 'wpu', 'xi', 'eta')}
    for k in range(N):
        a1, b1 = f1.step(u[k])
        if one_stage:
            al, be, sg = a1, b1, 1.0
        else:
            ua, va = f2.step(a1)
            ub, vb = f3.step(b1)
            al = -0.5 * (ua + vb)
            be = -0.5 * (-va + ub)
            sg = -1.0
        vn = 1.0 / np.sqrt(max(al * al + be * be, 0.02))
        c, s = np.cos(th), np.sin(th)
        xi = al * c + be * s
        eta = -al * s + be * c
        e = sg * eta * vn
        _, w_hat = obs.step(th)
        out['phase'][k], out['omega'][k] = th, w_hat / P['w0']
        out['e'][k], out['alpha'][k], out['beta'][k], out['xi'][k], out['eta'][k] = e, a1, b1, xi, eta
        th, I, w_pu = stage_b(e, th, I, kp, ki, P)
        out['wpu'][k] = w_pu
    return out


def ddsrf_pll(u, P, kp=None, ki=None, tau=None, clamp='default', i1_init=None):
    """block DDSRF-PLL: FHTalpha1 (quadrature generator), normalization on its outputs, DDSRF with four low-pass
    filters (tau), stage B on the positive-sequence outputs (gain +1), frequency integrator clamped to [0.5 1.5].
    Output phase = Unit Delay2 (theta), omega = observer x2 in pu."""
    kp = P['kp_ddsrf'] if kp is None else kp
    ki = P['ki_ddsrf'] if ki is None else ki
    tau = P['tau_ddsrf'] if tau is None else tau
    clamp = P['clamp_ddsrf'] if clamp == 'default' else clamp
    ts = P['ts']
    f1 = FHT(P, -1.0)
    lp = [LPF(ts, tau) for _ in range(4)]          # pos xi, pos eta, neg xi, neg eta
    obs = DIObserver(P)
    th = 0.0
    I = P['i1_init_ddsrf'] if i1_init is None else i1_init
    N = len(u)
    out = {k: np.zeros(N) for k in ('phase', 'omega', 'e', 'xi_p', 'eta_p', 'xi_n', 'eta_n', 'wpu')}
    for k in range(N):
        al, be = f1.step(u[k])
        vn = 1.0 / np.sqrt(max(al * al + be * be, 0.02))
        c, s = np.cos(th), np.sin(th)
        xp, ep = al * c + be * s, -al * s + be * c          # positive frame
        xn, en = al * c - be * s, al * s + be * c           # negative frame
        th2 = wrap(2.0 * th)
        c2, s2 = np.cos(th2), np.sin(th2)
        yxp, yep, yxn, yen = (f.output() for f in lp)
        a1 = yxn * c2 + yen * s2                            # alphabeta2xieta_pos1 (neg -> pos frame)
        b1 = -yxn * s2 + yen * c2
        c1 = yxp * c2 - yep * s2                            # alphabeta2xieta_neg1 (pos -> neg frame)
        d1 = yxp * s2 + yep * c2
        e = yep * vn
        _, w_hat = obs.step(th)
        out['phase'][k], out['omega'][k], out['e'][k] = th, w_hat / P['w0'], e
        out['xi_p'][k], out['eta_p'][k], out['xi_n'][k], out['eta_n'][k] = yxp, yep, yxn, yen
        lp[0].push(xp - a1, yxp)
        lp[1].push(ep - b1, yep)
        lp[2].push(xn - c1, yxn)
        lp[3].push(en - d1, yen)
        th, I, w_pu = stage_b(e, th, I, kp, ki, P, clamp)
        out['wpu'][k] = w_pu
    return out


def test_signal(t, T, w0):
    """block test_signal of the model (MATLAB Function), vectorized: u, theta_g (wrapped), omega_g (pu).
    T: dict with the fields of param.test (prv.test_param)."""
    t = np.asarray(t, float)
    f = np.where(t < T['t_fstep'], 1.0, T['f_step'])
    tf = min(T['t_fstep'], t[-1] + 1.0) if t.size else 0.0      # finite stand-in for a disabled step (Inf)
    phi = np.where(t < tf, w0 * t, w0 * tf + w0 * T['f_step'] * (t - tf))
    phi = phi + np.where(t >= T['t_jump'], T['dtheta'], 0.0)
    A = np.where(t >= T['t_rec'], T['a_rec'], np.where(t >= T['t_sag'], T['a_sag'], T['a0']))
    u = np.cos(phi + T['theta0'])
    for h, a, ph in zip(T['harm_order'], T['harm_amp'], T['harm_phase']):
        u = u + a * np.sin(h * phi + ph)
    u = A * u + np.where(t >= T['t_dc'], T['dc'], 0.0)
    return u, wrap(phi + T['theta0']), f


def test_param(case):
    """the six cases of prv.test_param (init_model.m), same numbers."""
    inf = np.inf
    T = dict(theta0=0.0, a0=1.0, t_sag=inf, a_sag=1.0, t_rec=inf, a_rec=1.0, harm_order=[3, 5, 7],
             harm_amp=[0.0, 0.0, 0.0], harm_phase=[0.0, 0.0, 0.0], t_jump=inf, dtheta=0.0, t_fstep=inf,
             f_step=1.0, t_dc=inf, dc=0.0)
    if case == 1:
        T.update(theta0=-np.pi, harm_order=[4, 5, 7], harm_amp=[0.1, 0.1, 0.0], t_sag=0.25, a_sag=0.2, t_rec=0.5,
                 a_rec=1.0)
        name = '2021 model: 4th and 5th harmonic 10 %, sag to 0.2 pu 0.25-0.5 s, start 180 deg away'
    elif case == 2:
        T.update(t_jump=0.4, dtheta=np.pi / 6)
        name = 'phase jump of +30 deg at 0.4 s'
    elif case == 3:
        T.update(t_fstep=0.4, f_step=49.0 / 50.0)
        name = 'frequency step 50 -> 49 Hz at 0.4 s'
    elif case == 4:
        T.update(t_dc=0.4, dc=0.02)
        name = 'DC offset of 0.02 pu from 0.4 s'
    elif case == 5:
        T.update(harm_amp=[0.05, 0.06, 0.05])
        name = 'odd harmonics 3rd 5 %, 5th 6 %, 7th 5 % from 0 s'
    elif case == 6:
        T.update(t_sag=0.25, a_sag=0.2, t_rec=0.5, a_rec=1.0)
        name = 'sag to 0.2 pu 0.25-0.5 s, no harmonics'
    else:
        raise ValueError(case)
    return T, name


def metrics(t, theta_g, omega_g, phase, omega, T):
    """pll_metrics.m of the model: e = phase - theta_g wrapped [deg]; lock_time (|e| < 5 deg until the first event),
    t_event, pre_mean/pre_pp (0.1 s before the event), peak (largest |e - pre_mean| after it), settle (within 1 deg of
    the final mean), end_mean/end_pp (last 0.2 s), w_end_mean/w_end_pp (omega - omega_g in the last 0.2 s, pu)."""
    t = np.asarray(t, float)
    e = np.degrees(wrap(np.asarray(phase) - np.asarray(theta_g)))
    r = dict(err=e)
    t_end = t[-1]
    r['t_event'] = min(T['t_sag'], T['t_jump'], T['t_fstep'], T['t_dc'])
    before = np.where(t < r['t_event'])[0]
    out = np.where(np.abs(e[before]) >= 5)[0]
    if out.size == 0:
        r['lock_time'] = t[0]
    elif out[-1] < before.size - 1:
        r['lock_time'] = t[out[-1] + 1]
    else:
        r['lock_time'] = np.nan
    pre = (t >= r['t_event'] - 0.1) & (t < r['t_event']) if np.isfinite(r['t_event']) else t >= t_end - 0.2
    r['pre_mean'], r['pre_pp'] = np.mean(e[pre]), np.ptp(e[pre])
    fin = t >= t_end - 0.2
    r['end_mean'], r['end_pp'] = np.mean(e[fin]), np.ptp(e[fin])
    dw = np.asarray(omega) - np.asarray(omega_g)
    r['w_end_mean'], r['w_end_pp'] = np.mean(dw[fin]), np.ptp(dw[fin])
    if np.isfinite(r['t_event']):
        post = t >= r['t_event']
        r['peak'] = np.max(np.abs(e[post] - r['pre_mean']))
        away = np.where(post & (np.abs(e - r['end_mean']) > 1))[0]
        if away.size == 0:
            r['settle'] = 0.0
        elif away[-1] < t.size - 1 and t[away[-1]] < t_end - 0.2:
            r['settle'] = t[away[-1] + 1] - r['t_event']
        else:
            r['settle'] = np.nan
    else:
        r['peak'] = r['settle'] = np.nan
    return r


def run_all(u, P):
    return dict(park=park_pll(u, P), fht=fht_pll(u, P), ddsrf=ddsrf_pll(u, P))
