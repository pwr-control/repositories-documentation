"""Reverse recovery of a high-voltage rectifier diode at the current slope of the circuit.
Companion model of diode_stack_design_guidelines.tex (Sec. 6); used by diode_stack_calc.py.

A datasheet gives the stored charge QS and the peak reverse current Irr at a few A/us and at a forward
current of about 1 kA. In a line-commutated bridge fed by a transformer the current crosses zero at about
0.1 to 0.2 A/us, so the datasheet point has to be moved by more than a decade. The charge-control model of
Lauritzen and Ma (charge q_M, transit time T_M, lifetime tau) scales linearly with the slope on a long
ramp and cannot follow the flat QS(di/dt) curve of a datasheet. Here it is extended with a quadratic
recombination term (end regions and Auger recombination, effective only at high plasma density):

    dq_M/dt = i - q_M/tau_b - k*q_M^2,     i = (q_E - q_M)/T_M

  conducting: the circuit imposes i(t) = IF - alpha*t, starting from the steady state at IF;
  the junction blocks when q_E = q_M + T_M*i reaches zero (peak reverse current Irrm);
  blocked:    i = -q_M/T_M (tail), q_M decays with 1/tau_b + 1/T_M and the quadratic term.

With k = 0 the model is the one of Lauritzen and Ma and the long-ramp formulas apply:
    t_tail = tau*T_M/(tau + T_M),  Irrm = alpha*(tau - t_tail),  Qrr = alpha*(tau^2 - t_tail^2)/2.

Units of this module: us, A, uC (so that uC/us = A); gb = 1/tau_b in 1/us (gb = 0: no bulk recombination),
k in 1/(uC us). Only numpy and scipy are needed.
Run:  python diode_stack_recovery.py     (prints the scenarios of the worked example; about 20 s)"""
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


def q_steady(IF, gb, k):
    """Stored charge in steady state at the forward current IF."""
    if k <= 0:
        return IF/gb
    return (-gb + math.sqrt(gb*gb + 4*k*IF))/(2*k)


def recovery(alpha, IF, gb, TM, k):
    """Current ramp IF - alpha*t from the steady state at IF. Returns a dict with Qrr, Irrm, t_a (zero crossing
    to peak), t_tail of the linear part of the tail, the triangle charge Qa and the softness s of the equivalent
    triangle (Qrr = Irrm^2*(1 + s)/(2*alpha))."""
    q0, t0 = q_steady(IF, gb, k), IF/alpha

    def fun(t, y):
        return [IF - alpha*t - gb*y[0] - k*y[0]**2]

    def ev(t, y):
        return y[0] + TM*(IF - alpha*t)
    ev.terminal, ev.direction = True, -1
    tmax = t0 + 40*TM + 20*(1/gb if gb > 0 else 100.0) + 50
    sol = solve_ivp(fun, [0, tmax], [q0], events=ev, rtol=1e-10, atol=1e-10, method="LSODA")
    if len(sol.t_events[0]) == 0:
        raise RuntimeError("no recovery found before tmax")
    ts, qs = sol.t_events[0][0], sol.y_events[0][0][0]
    Irrm, ta = qs/TM, ts - t0
    tt = TM/(1 + gb*TM)                                     # = tau_b*TM/(tau_b + TM)
    Qa = 0.5*alpha*ta**2
    Qtail = (math.log(1 + k*tt*qs)/k if k > 0 else tt*qs)/TM
    Qrr = Qa + Qtail
    return dict(Qrr=Qrr, Irrm=Irrm, ta=ta, tt=tt, Qa=Qa, s=2*alpha*Qrr/Irrm**2 - 1)


def lm_pair(alpha, tau, TM):
    """Long-ramp formulas of the Lauritzen-Ma model (k = 0): (Qrr, Irrm)."""
    tt = tau*TM/(tau + TM)
    return 0.5*alpha*(tau**2 - tt**2), alpha*(tau - tt)


def lm_from_point(alpha, Q, I):
    """Lauritzen-Ma parameters from one point on a long ramp: (tau, TM, t_tail)."""
    tt = Q/I - I/(2*alpha)
    tau = tt + I/alpha
    return tau, tau*tt*alpha/I, tt


def _best(res, starts):
    best = None
    for p0 in starts:
        try:
            s = least_squares(res, p0, x_scale=1.0, xtol=1e-7, ftol=1e-9)
        except (RuntimeError, ValueError, OverflowError):
            continue
        if best is None or s.cost < best.cost:
            best = s
    if best is None:
        raise RuntimeError("no fit converged")
    return best


def fit_curves(points, IF, starts=None):
    """Three-parameter fit (tau_b, TM, k) on points (alpha, Q, I) spread over a wide range of slopes.
    Returns (tau_b, TM, k, rms of the logarithmic residuals)."""
    def res(p):
        gb, TM, k = 1/math.exp(p[0]), math.exp(p[1]), math.exp(p[2])
        r = []
        for a, Q, I in points:
            rr = recovery(a, IF, gb, TM, k)
            r += [math.log(rr["Qrr"]/Q), math.log(rr["Irrm"]/I)]
        return np.array(r)
    starts = starts or ([math.log(100), math.log(30), math.log(1e-6)], [math.log(300), math.log(60), math.log(1e-6)],
                        [math.log(60), math.log(20), math.log(1e-7)])
    s = _best(res, starts)
    return math.exp(s.x[0]), math.exp(s.x[1]), math.exp(s.x[2]), float(np.sqrt(np.mean(s.fun**2)))


def fit_point(gb, q_ref, i_ref, IF, start=None):
    """T_M and k for an assumed bulk lifetime (gb = 1/tau_b): the charge follows q_ref = [(alpha, Q), ...]
    (the table point of the datasheet with the local slope of its QS curve) and the peak current matches
    i_ref = (alpha, I). start = (TM, k) of a neighbouring fit, if any. Returns (TM, k)."""
    def res(p):
        TM, k = math.exp(p[0]), math.exp(p[1])
        r = [math.log(recovery(a, IF, gb, TM, k)["Qrr"]/Q) for a, Q in q_ref]
        r.append(math.log(recovery(i_ref[0], IF, gb, TM, k)["Irrm"]/i_ref[1]))
        return np.array(r)
    if start is not None:                                   # warm start (parameter scans)
        starts = [[math.log(start[0]), math.log(start[1])]]
    else:
        starts = [[math.log(25), math.log(1e-6)], [math.log(30), math.log(3e-6)], [math.log(20), math.log(3e-7)]]
    s = _best(res, starts)
    return math.exp(s.x[0]), math.exp(s.x[1])


def t_scale(T_from, T_to, m=1.0):
    """Temperature factor of the lifetime, tau ~ T^m with T in kelvin (Qrr ~ tau^2, Irrm ~ tau on a long ramp)."""
    return ((T_to + 273.15)/(T_from + 273.15))**m


def power_law(p1, p2):
    """Power law Q = c*alpha^e through two points (alpha, Q): returns (c, e)."""
    e = math.log(p2[1]/p1[1])/math.log(p2[0]/p1[0])
    return p1[1]/p1[0]**e, e


if __name__ == "__main__":
    # worked example of the note: rectifier diode with the datasheet point QS 3200 uC / Irr 125 A at 5 A/us, 1000 A,
    # 150 C, and a QS(di/dt) curve of exponent 0.30; circuit: 0.176 A/us, 100 A
    QS, IRR, A_DS, IF_DS, E_QS, A_CIRC, IF_CIRC = 3200.0, 125.0, 5.0, 1000.0, 0.30, 0.176, 100.0
    q_ref = [(a, QS*(a/A_DS)**E_QS) for a in (A_DS/1.2, A_DS, 1.2*A_DS)]
    tau0, TM0, tt0 = lm_from_point(A_DS, QS, IRR)
    q, i = lm_pair(A_CIRC, tau0, TM0)
    print(f"floor   (Lauritzen-Ma through the table point): tau {tau0:5.1f} us, TM {TM0:4.1f} us -> "
          f"{q:5.0f} uC / {i:5.2f} A at {A_CIRC} A/us, 150 C")
    for name, tb in (("central", 108.0), ("upper", 200.0), ("ceiling", None)):
        gb = 0.0 if tb is None else 1/tb
        TM, k = fit_point(gb, q_ref, (A_DS, IRR), IF_DS)
        r = recovery(A_CIRC, IF_CIRC, gb, TM, k)
        print(f"{name:8s}(tau_b {'inf' if tb is None else f'{tb:.0f}':>3s} us): TM {TM:4.1f} us, k {k:.2e} 1/(uC us) -> "
              f"{r['Qrr']:5.0f} uC / {r['Irrm']:5.2f} A at {A_CIRC} A/us, 150 C (softness {r['s']:.2f})")
