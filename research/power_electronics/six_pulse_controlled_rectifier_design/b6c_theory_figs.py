"""Figure data of b6c_sizing_theory.tex (theory companion of b6c_sizing_formulas.tex).
Prints pgfplots coordinate lists, pasted into the .tex:
  bridge   ideal B6C with overlap: rail voltages, output voltage, phase current, voltage across T1
  snapoff  normalised snap-off transient of the RC snubber (Sec. snubber), v_T/V_com and v_C/V_com
  step     forward voltage step through 2Lk on a blocking device with RC snubber (dv/dt criterion)
Run:  python b6c_theory_figs.py bridge|snapoff|step|all   [alpha_deg mu_deg]"""
import sys
import numpy as np

pi = np.pi
d2r = np.deg2rad


# ---------------------------------------------------------------- ideal bridge with overlap
def bridge_waveforms(alpha_deg=30.0, mu_deg=15.0, n=1441):
    """Phase EMFs with unit peak, equal commutation inductances, constant Id, no resistance.
    Returns theta(deg), v_p, v_n, v_d, i_a/Id, v_T1 = v_a,term - v_p, (e_a, e_b, e_c)."""
    a, mu = d2r(alpha_deg), d2r(mu_deg)
    k = np.cos(a) - np.cos(a + mu)                       # = 2 w Lk Id / V_LL,pk  (eq. mu of the sheet)
    th = np.linspace(0.0, 360.0, n)
    t = d2r(th)
    e = {'a': np.sin(t), 'b': np.sin(t - 2*pi/3), 'c': np.sin(t + 2*pi/3)}
    # natural commutation angles (deg) and (outgoing, incoming) phases
    upper = [(30.0, 'c', 'a'), (150.0, 'a', 'b'), (270.0, 'b', 'c')]
    lower = [(90.0, 'b', 'c'), (210.0, 'c', 'a'), (330.0, 'a', 'b')]

    def group(events):
        v_rail = np.zeros_like(t)
        cur = {p: np.zeros_like(t) for p in 'abc'}      # current of each phase in this group, p.u. of Id
        vterm = {p: e[p].copy() for p in 'abc'}         # terminal voltage of each phase (changed only in overlap)
        for i, thi in enumerate(th):
            # latest event started at or before thi (with wrap)
            best = None
            for nat, out, inc in events:
                start = nat + alpha_deg
                dth = (thi - start) % 360.0
                if best is None or dth < best[0]:
                    best = (dth, nat, out, inc)
            dth, nat, out, inc = best
            if dth < mu_deg:                              # overlap: both conduct, terminals at the mean voltage
                x = (np.cos(a) - np.cos(d2r(dth) + a))/k  # incoming current fraction
                x = min(max(x, 0.0), 1.0)
                vm = 0.5*(e[out][i] + e[inc][i])
                v_rail[i] = vm
                vterm[out][i] = vterm[inc][i] = vm
                cur[inc][i], cur[out][i] = x, 1.0 - x
            else:
                v_rail[i] = e[inc][i]
                cur[inc][i] = 1.0
        return v_rail, cur, vterm

    v_p, cu, vt_u = group(upper)
    v_n, cl, vt_l = group(lower)
    v_d = v_p - v_n
    i_a = cu['a'] - cl['a']
    # terminal voltage of phase a: changed by whichever group commutates on it
    v_a = e['a'].copy()
    for arr in (vt_u['a'], vt_l['a']):
        m = arr != e['a']
        v_a[m] = arr[m]
    v_T1 = v_a - v_p
    return th, v_p, v_n, v_d, i_a, v_T1, e, k


def coords(x, y, step=1, fmt="({:.1f},{:.4f})"):
    return " ".join(fmt.format(xi, yi) for xi, yi in zip(x[::step], y[::step]))


# ---------------------------------------------------------------- snap-off transient (sheet, Sec. 5)
def snap_off_wave(zeta, chi, tmax=12.0, n=481):
    """x = i Z0/Vcom, tau = w0 t: x'' + 2 zeta x' + x = 0, x(0) = chi, x'(0) = 1 - 2 zeta chi.
    v_T/Vcom = 1 - x',  v_C/Vcom = v_T/Vcom - 2 zeta x."""
    if abs(zeta - 1.0) < 1e-6:
        zeta += 1e-6
    s = np.roots([1.0, 2*zeta, 1.0]).astype(complex)
    x0, dx0 = chi, 1.0 - 2*zeta*chi
    c2 = (dx0 - s[0]*x0)/(s[1] - s[0])
    c1 = x0 - c2
    tau = np.linspace(0.0, tmax, n)
    e1, e2 = np.exp(s[0]*tau), np.exp(s[1]*tau)
    x = (c1*e1 + c2*e2).real
    dx = (c1*s[0]*e1 + c2*s[1]*e2).real
    vT = 1.0 - dx
    vC = vT - 2*zeta*x
    return tau, x, vT, vC


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    al = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    mu = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    if what in ('bridge', 'all'):
        th, v_p, v_n, v_d, i_a, v_T1, e, k = bridge_waveforms(al, mu)
        print(f"% bridge waveforms: alpha={al} deg, mu={mu} deg, k = cos a - cos(a+mu) = {k:.4f}; "
              f"phase EMF peak = 1, line peak = sqrt(3); mean v_d = {v_d.mean():.4f} "
              f"(Vd0 cos a = {3*np.sqrt(3)/pi*np.cos(d2r(al)):.4f}, Vd0 (cos a + cos(a+mu))/2 = "
              f"{3*np.sqrt(3)/pi*0.5*(np.cos(d2r(al))+np.cos(d2r(al+mu))):.4f})")
        for name, y in (('v_p', v_p), ('v_n', v_n), ('v_d', v_d), ('i_a', i_a), ('v_T1', v_T1)):
            print(f"% {name}")
            print(coords(th, y, step=2))
    if what in ('snapoff', 'all'):
        chi = 0.85
        for z in (0.3, 0.66, 1.2):
            tau, x, vT, vC = snap_off_wave(z, chi)
            print(f"% snap-off chi={chi} zeta={z}: F={vT.max():.3f} at tau={tau[vT.argmax()]:.2f}; vT(0+)={vT[0]:.3f}")
            print("% v_T/V_com"); print(coords(tau, vT, step=2, fmt="({:.3f},{:.4f})"))
            print("% v_C/V_com"); print(coords(tau, vC, step=2, fmt="({:.3f},{:.4f})"))
            print("% i Z0/V_com"); print(coords(tau, x, step=2, fmt="({:.3f},{:.4f})"))
        tau, x, vT, vC = snap_off_wave(0.0, chi)
        print(f"% undamped chi={chi}: F={vT.max():.3f} = 1+sqrt(1+chi^2) = {1+np.sqrt(1+chi**2):.3f}")
        print(coords(tau, vT, step=2, fmt="({:.3f},{:.4f})"))
    if what in ('step', 'all'):
        for z in (0.3, 0.5, 0.7, 1.0):
            tau, x, vT, vC = snap_off_wave(z, 0.0, tmax=8.0, n=801)
            dv = np.gradient(vT, tau)
            print(f"% step chi=0 zeta={z}: max dv/dtau = {dv.max():.3f} at tau={tau[dv.argmax()]:.2f} "
                  f"(2 zeta = {2*z:.2f}); peak {vT.max():.3f}")
            print(coords(tau, vT, step=4, fmt="({:.3f},{:.4f})"))
