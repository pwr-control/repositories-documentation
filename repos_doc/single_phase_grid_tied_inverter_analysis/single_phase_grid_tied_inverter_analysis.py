"""Linear analysis and sampled-data simulation of the single-phase T-type AFE controller afe_sp_ttype_ctrl_pwm
(library block afe_single_phase_ttype of pwr-control/library, used by the SST bench sst_cllc_ttype_inv.slx of
pwr-control/solid-state-transformers).

Companion script of single_phase_grid_tied_inverter_analysis.tex: reproduces every number quoted in the note and
draws its figures (python3, numpy, scipy, matplotlib).

    python3 single_phase_grid_tied_inverter_analysis.py [figures_dir]

Model. The converter is single phase: the only physical current is i_s (alpha). The controller builds the
quadrature component with a first harmonic tracker (FHT), rotates the pair into the dq frame of the PLL, applies a
vector PI and projects the modulation vector back on alpha. With a locked PLL (theta = w0 t) the whole path from
the acquired current to the modulation index is linear time-invariant: the rotation by -theta, the PI and the
rotation by +theta are the frequency-shifted PI C(s - j w0), and the real part of a complex-coefficient filter
acting on a real signal is the real filter (G + conj G)/2. The loop is therefore analysed exactly in the
stationary frame as a SISO loop with the equivalent controller
    C_eq(s) = kp H_u(s) + ki [s H_u(s) - w0 H_v(s)] / (s^2 + w0^2),
a proportional-resonant controller shaped by the FHT (H_u, H_v: in-phase and quadrature outputs of the FHT).
The same holds in discrete time with C(z e^{-j w0 ts}): the resonant denominator is z^2 - 2 cos(w0 ts) z + 1.
The plant is the LC filter of the module with the line inductance to the grid EMF (pu of the converter base),
discretized with a zero-order hold at the control period ts = 1/(2 fpwm); the control uses the samples of the
previous trigger (one-step acquisition delay) and the FHT is written as in the code (forward Euler internal model,
gains from the closed-form pole placement of afe_single_phase_ttype_param_example.m).
Plant data: afe_single_phase_ttype_param_example.m (50 kW module, 800 V DC link, 4 kHz double update, 2 x 250 uH +
50 uF / 50 mOhm, 0.5 mH + 10 mOhm of line per module, one phase of a 690 V system) = private/sst_cllc_ttype_param.m
of the bench.
"""
import os
import sys
import numpy as np
from numpy.polynomial import polynomial as P
from scipy.linalg import expm
from scipy import signal
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


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight')
    plt.close(fig)


# =============================================================================================== 1. data
pr('=' * 100)
pr('1. MODULE DATA (afe_single_phase_ttype_param_example.m / sst_cllc_ttype_param.m)')
pwr_nom = 50e3
udc_nom = 800.0
ubez = 0.9*udc_nom                       # [V] peak, end of the linear modulation range
u_ll_ref = 690.0
ug_rms = u_ll_ref/np.sqrt(3)             # [V] one phase of the 690 V system
ug_pk = ug_rms*np.sqrt(2)
is_nom = pwr_nom/ug_rms                  # [A] rms
ibez = is_nom*np.sqrt(2)                 # [A] peak
zbez = ubez/ibez
f0 = 50.0
w0 = 2*np.pi*f0
fpwm = 4e3
ts = 1/(2*fpwm)
fs = 1/ts
k_mod = udc_nom/ubez                     # converter voltage [pu of ubez] per unit of PI output (pu of udc_nom)
L1 = L2 = 250e-6
R1 = R2 = 157*0.05*250e-6
Lc, Rc = L1 + L2, R1 + R2                # converter-side inductance (both lines)
Cf, Rcf = 50e-6, 50e-3
Lg, Rg = 0.5e-3, 10e-3                   # line inductance seen by one module (one of the two inductors of the bench)
ug_nom = ug_pk/ubez                      # [pu] grid amplitude
Cdc = 3e-3                               # [F] series of the two 6 mF banks
pr(f'  ubez = {ubez:.1f} V, ibez = {ibez:.1f} A ({is_nom:.1f} A rms), zbez = {zbez:.3f} Ohm, k_mod = udc_nom/ubez = {k_mod:.4f}')
pr(f'  grid {ug_rms:.1f} V rms, {ug_pk:.1f} V peak = {ug_nom:.4f} pu; ts = {ts*1e6:.0f} us, fs = {fs:.0f} Hz, fs/6 = {fs/6:.0f} Hz')
Lcp, Rcp, Cfp, Rcfp, Lgp, Rgp = Lc/zbez, Rc/zbez, Cf*zbez, Rcf/zbez, Lg/zbez, Rg/zbez
pr(f"  per unit: Lc' = {Lcp*1e6:.1f} us, Rc' = {Rcp*1e3:.3f} mpu, Cf' = {Cfp*1e6:.1f} us, Rcf' = {Rcfp*1e3:.2f} mpu, Lg' = {Lgp*1e6:.1f} us, Rg' = {Rgp*1e3:.2f} mpu")
f_lc = 1/(2*np.pi*np.sqrt(Lc*Cf))
Lpar = Lc*Lg/(Lc + Lg)
f_res = 1/(2*np.pi*np.sqrt(Lpar*Cf))
zres = np.sqrt(Lpar/Cf)
pr(f'  LC resonance (open line) {f_lc:.0f} Hz; LCL resonance with Lg = {Lg*1e3:.1f} mH: {f_res:.0f} Hz (Lpar = {Lpar*1e6:.0f} uH, '
   f'Z_res = {zres:.2f} Ohm, zeta_RCf ~ {Rcf/(2*zres):.3f})')
Lg_crit = 1/((2*np.pi*fs/6)**2*Cf)
Lg_crit = Lg_crit*Lc/(Lc - Lg_crit)
pr(f'  resonance at fs/6 for Lg = {Lg_crit*1e6:.0f} uH (above it the resonance is below fs/6)')
pr(f'  plant gain at the crossover (capacitor open): k_mod/(Lc\'+Lg\') = {k_mod/(Lcp+Lgp):.0f} pu/s per pu; with Lc only {k_mod/Lcp:.0f} pu/s')
dU = pwr_nom/(w0*Cdc*udc_nom)
pr(f'  DC link: 100 Hz ripple P/(w0 C U) = {dU:.1f} V peak-peak ({dU/udc_nom*100:.1f} %), stored energy swing P/w0 = {pwr_nom/w0:.0f} J')

# control parameters (param_example)
kp_i, ki_i = 0.1, 20.0
kp_old, ki_old = 0.5, 45.0
delta = 0.05
pll_kp, pll_ki, pll_tau = 1.0, 0.0, 2/w0
kp_udc, ki_udc, udc_tau = 2.0, 20.0, 20e-3
i_lim = 1.2


def obs_gains(omega, ts):
    z1, z2 = np.exp(-omega*ts), np.exp(-4*omega*ts)
    l1 = 2 - (z1 + z2)
    l2 = (z1*z2 - (1 - l1))/ts
    return l1, l2


def fht_gains(omega, delta, ts):
    z1, z2 = np.exp(-omega*ts), np.exp(-4*omega*ts)
    d = delta*omega*ts
    l1 = 2 - d - (z1 + z2)
    l2 = (z1*z2 - (1 - l1)*(1 - d) - ts**2*omega**2)/ts
    return l1, l2


obs_l1, obs_l2 = obs_gains(w0, ts)
fht_l1, fht_l2 = fht_gains(w0, delta, ts)
l1c, l2c = (5 - delta)*w0, (3 - 5*delta + delta**2)*w0**2          # continuous-time placement at -w0, -4 w0
pr(f'  observer gains (discrete) l1 = {obs_l1:.4f}, l2 = {obs_l2:.2f} 1/s; FHT gains l1 = {fht_l1:.4f}, l2 = {fht_l2:.2f} 1/s '
   f'(continuous equivalents l1/ts = {fht_l1/ts:.0f} 1/s, l2/ts = {fht_l2/ts:.3e} 1/s^2; continuous placement {l1c:.0f}, {l2c:.3e})')

# =============================================================================================== 2. plant
pr('=' * 100)
pr('2. PLANT')


def plant_ss(Lg_, Rg_):
    """Continuous state space in pu: x = [i_c, u_cf, i_g], u = [u_s, e_g], y = [i_c, u_ac, i_g]."""
    Lgp_, Rgp_ = Lg_/zbez, Rg_/zbez
    A = np.array([[-(Rcp + Rcfp)/Lcp, -1/Lcp, Rcfp/Lcp],
                  [1/Cfp, 0.0, -1/Cfp],
                  [Rcfp/Lgp_, 1/Lgp_, -(Rgp_ + Rcfp)/Lgp_]])
    B = np.array([[1/Lcp, 0.0], [0.0, 0.0], [0.0, -1/Lgp_]])
    C = np.array([[1.0, 0.0, 0.0], [Rcfp, 1.0, -Rcfp], [0.0, 0.0, 1.0]])
    return A, B, C


def tf_eval(A, B, C, s, iu, iy):
    """Transfer function y_iy / u_iu of (A, B, C) at the complex points s (vectorized)."""
    n = A.shape[0]
    out = np.empty(len(s), dtype=complex)
    for k, sk in enumerate(s):
        out[k] = (C[iy] @ np.linalg.solve(sk*np.eye(n) - A, B[:, iu]))
    return out


def zoh(A, B, T):
    n, m = B.shape
    M = expm(np.block([[A, B], [np.zeros((m, n + m))]])*T)
    return M[:n, :n], M[:n, n:]


A, B, C = plant_ss(Lg, Rg)
Ad, Bd = zoh(A, B, ts)
f = np.logspace(0, np.log10(fs/2), 2000)
w = 2*np.pi*f
s = 1j*w
z = np.exp(s*ts)
Pc = k_mod*tf_eval(A, B, C, s, 0, 0)                       # PI output -> i_c, continuous
Pd = k_mod*tf_eval(Ad, Bd, C, z, 0, 0)                     # ZOH plant, m(k) -> i_c(k) (strictly proper)
# stiff grid (Lg -> 0): first order
Pc_stiff = k_mod/(Lcp*s + Rcp)
eig_ol = np.linalg.eigvals(A)
pr('  open-loop poles of the plant [1/s]:', np.array2string(eig_ol, precision=1))
for e in eig_ol:
    if abs(e.imag) > 1:
        pr(f'    LCL mode: {abs(e)/(2*np.pi):.0f} Hz, sigma = {e.real:.0f} 1/s, zeta = {-e.real/abs(e):.4f}')

# =============================================================================================== 3. FHT and the equivalent controller
pr('=' * 100)
pr('3. FHT AND THE EQUIVALENT STATIONARY-FRAME CONTROLLER')


def fht_tf_c(s, l1, l2, delta=delta):
    """Continuous FHT: in-phase H_u = x1/i and quadrature H_v = (-x2/w0)/i."""
    D = s**2 + (l1 + delta*w0)*s + w0**2 + l2 + l1*delta*w0
    Hu = (l1*s + l1*delta*w0 + l2)/D
    Hv = (l1*w0**2 - l2*s)/(w0*D)
    return Hu, Hv


def fht_tf_d(z, l1=fht_l1, l2=fht_l2, delta=delta):
    """Discrete FHT as coded: x(k+1) = Ad x(k) + L (w(k) - x1(k)), Ad = I + [0 1; -w0^2 -delta w0] ts, outputs x1(k), -x2(k)/w0
    (the state before the update: one step from the input w to the outputs). w(k) is the acquired sample (previous trigger)."""
    a11, a12 = 1.0, ts
    a21, a22 = -w0**2*ts, 1 - delta*w0*ts
    # (zI - Ad + L C) x = L w, C = [1 0]
    m11, m12 = z - a11 + l1, -a12
    m21, m22 = -a21 + l2, z - a22
    det = m11*m22 - m12*m21
    x1 = (m22*l1 - m12*l2)/det
    x2 = (-m21*l1 + m11*l2)/det
    return x1, -x2/w0


def fht_polys(l1=fht_l1, l2=fht_l2, delta=delta):
    """Polynomials (ascending coefficients) of the discrete FHT: H_u = nu/den, H_v = nv/den, from
    (zI - Ad + L C) x = L w: x1 = ((z - a22) l1 + ts l2)/det, x2 = (-(l2 - a21) l1 + (z - 1 + l1) l2)/det, H_v = -x2/w0."""
    a21, a22 = -w0**2*ts, 1 - delta*w0*ts
    den = P.polyadd(P.polymul([-(1 - l1), 1.0], [-a22, 1.0]), [ts*(l2 - a21)])
    nu = np.array([-a22*l1 + ts*l2, l1])
    nv = np.array([(l2 - a21)*l1 + (1 - l1)*l2, -l2])/w0
    return nu, nv, den


# check of fht_polys against fht_tf_d at a few points
nu_, nv_, den_ = fht_polys()
zt = np.exp(1j*np.array([0.01, 0.3, 1.0, 2.5]))
Hu_t, Hv_t = fht_tf_d(zt)
err_u = np.max(np.abs(P.polyval(zt, nu_)/P.polyval(zt, den_) - Hu_t))
err_v = np.max(np.abs(P.polyval(zt, nv_)/P.polyval(zt, den_) - Hv_t))
assert err_u < 1e-9 and err_v < 1e-9, (err_u, err_v)

Hu_c, Hv_c = fht_tf_c(s, l1c, l2c)
Hu_d, Hv_d = fht_tf_d(z)
Hu_c0, Hv_c0 = fht_tf_c(np.array([0j, 1j*w0]), l1c, l2c)
pr(f'  continuous FHT (delta = {delta}): H_u(0) = {Hu_c0[0].real:.4f}, H_v(0) = {Hv_c0[0].real:.4f}; '
   f'H_u(j w0) = {abs(Hu_c0[1]):.4f} /_ {np.degrees(np.angle(Hu_c0[1])):.2f} deg, H_v(j w0) = {abs(Hv_c0[1]):.4f} /_ {np.degrees(np.angle(Hv_c0[1])):.2f} deg')
Hu_c00, Hv_c00 = fht_tf_c(np.array([1j*w0]), 5*w0, 3*w0**2, 0.0)
pr(f'  with delta = 0: H_u(j w0) = {Hu_c00[0]:.4f}, H_v(j w0) = {Hv_c00[0]:.4f}')
zz0 = np.exp(1j*w0*ts)
Hu_d0, Hv_d0 = fht_tf_d(np.array([1.0 + 0j, zz0]))
pr(f'  discrete FHT (code): H_u(1) = {Hu_d0[0].real:.4f}, H_v(1) = {Hv_d0[0].real:.4f}; at w0: H_u = {abs(Hu_d0[1]):.4f} /_ {np.degrees(np.angle(Hu_d0[1])):.2f} deg '
   f'(one step of delay included: {np.degrees(w0*ts):.2f} deg), H_v = {abs(Hv_d0[1]):.4f} /_ {np.degrees(np.angle(Hv_d0[1])):.2f} deg')
# the same, delayed by one step less (the estimate as an estimate of the sample it is compared with)
pr(f'  ... removing the one-step delay: H_u = {abs(Hu_d0[1]):.4f} /_ {np.degrees(np.angle(Hu_d0[1]*zz0)):.2f} deg, '
   f'H_v = {abs(Hv_d0[1]):.4f} /_ {np.degrees(np.angle(Hv_d0[1]*zz0)):.2f} deg')
f_fht = np.array([100, 150, 250, 500, 1000, 1423, 2000, 4000.0])
Hu_f, _ = fht_tf_c(1j*2*np.pi*f_fht, l1c, l2c)
pr('  |H_u| at', ' '.join(f'{ff:.0f} Hz: {abs(h):.3f}' for ff, h in zip(f_fht, Hu_f)))
pr(f'  high-frequency equivalent corner l1 = {l1c:.0f} rad/s ({l1c/(2*np.pi):.0f} Hz)')


def ceq_c(s, kp, ki, l1=l1c, l2=l2c):
    Hu, Hv = fht_tf_c(s, l1, l2)
    return kp*Hu + ki*(s*Hu - w0*Hv)/(s**2 + w0**2)


def ceq_d(z, kp, ki):
    """Discrete equivalent controller from the acquired sample w(k) (previous trigger) to the modulation index m(k):
    C_eq(z) = kp H_u + ki ts z [(z - cos th) H_u - sin th H_v]/(z^2 - 2 cos th z + 1), th = w0 ts
    (PI with the updated state in the output, C(z) = kp + ki ts z/(z - 1), frequency shifted by +-th)."""
    Hu, Hv = fht_tf_d(z)
    th = w0*ts
    return kp*Hu + ki*ts*z*((z - np.cos(th))*Hu - np.sin(th)*Hv)/(z**2 - 2*np.cos(th)*z + 1)


def ceq_polys(kp, ki):
    """Numerator and denominator (ascending) of the discrete C_eq(z)."""
    nu, nv, den = fht_polys()
    th = w0*ts
    res = np.array([1.0, -2*np.cos(th), 1.0])
    num = P.polyadd(P.polymul(P.polymul([kp], nu), res),
                    P.polymul([0.0, ki*ts], P.polysub(P.polymul([-np.cos(th), 1.0], nu), P.polymul([np.sin(th)], nv))))
    return num, P.polymul(den, res)


# check ceq_polys against ceq_d
nn, dd = ceq_polys(kp_i, ki_i)
err = np.max(np.abs(P.polyval(zt, nn)/P.polyval(zt, dd) - ceq_d(zt, kp_i, ki_i)))
assert err < 1e-9, err

for kp, ki, lab in [(kp_i, ki_i, 'library'), (kp_old, ki_old, 'bench (old modules)')]:
    c0 = ceq_c(np.array([0j]), kp, ki)[0].real
    c0d = ceq_d(np.array([1.0 + 0j]), kp, ki)[0].real
    pr(f'  C_eq {lab} kp = {kp}, ki = {ki}: DC gain continuous {c0:+.4f} (kp H_u(0) = {kp*Hu_c0[0].real:.4f}, -ki H_v(0)/w0 = {-ki*Hv_c0[0].real/w0:.4f}), discrete {c0d:+.4f}')

# =============================================================================================== 4. current loop
pr('=' * 100)
pr('4. CURRENT LOOP')
kp_rec, ki_rec = 0.15, 20.0            # recommended gains (see the note)


def margins(Lw, w):
    """First crossover of |L| = 1 (design crossover), phase margin there, modulus margin min|1 + L| and L(0)."""
    mag = np.abs(Lw)
    ph = np.unwrap(np.angle(Lw))
    ic = np.where((mag[:-1] >= 1) & (mag[1:] < 1))[0]
    wc = pm = np.nan
    if len(ic):
        i = ic[0]
        wc = np.interp(0.0, [np.log(mag[i + 1]), np.log(mag[i])], [w[i + 1], w[i]])
        phc = np.interp(wc, w, ph)
        pm = 180 + np.degrees(phc)
        pm = (pm + 180) % 360 - 180
    mm = np.min(np.abs(1 + Lw))
    return wc, pm, mm


def plant_d(Lg_=Lg, Rg_=Rg):
    A_, B_, C_ = plant_ss(Lg_, Rg_)
    Ad_, Bd_ = zoh(A_, B_, ts)
    return k_mod*tf_eval(Ad_, Bd_, C_, z, 0, 0)


def loop_d(kp, ki, Lg_=Lg, Rg_=Rg):
    Pd_ = plant_d(Lg_, Rg_)
    return ceq_d(z, kp, ki)/z*Pd_, Pd_


def plant_polys(Lg_=Lg, Rg_=Rg):
    A_, B_, C_ = plant_ss(Lg_, Rg_)
    Ad_, Bd_ = zoh(A_, B_, ts)
    num_p, den_p = signal.ss2tf(Ad_, Bd_[:, [0]], C_[[0], :], np.zeros((1, 1)))
    return k_mod*num_p[0][::-1], den_p[::-1]


def closed_loop_poles(kp, ki, Lg_=Lg, Rg_=Rg, pp=None):
    """Roots of 1 + C_eq(z) z^-1 P_zoh(z) = 0: z den_c den_p + num_c num_p = 0."""
    num_p, den_p = pp if pp is not None else plant_polys(Lg_, Rg_)
    num_c, den_c = ceq_polys(kp, ki)
    char = P.polyadd(P.polymul(P.polymul([0.0, 1.0], den_c), den_p), P.polymul(num_c, num_p))
    return P.polyroots(char)


def s_of_z(zz):
    return np.log(zz)/ts


def describe_poles(poles, fres_):
    """DC (real) pole, LCL mode (pair nearest to fres_) and the slowest low-frequency pair."""
    sp = s_of_z(poles)
    real = [p for p in sp if abs(p.imag) < 1.0]
    dc = max(real, key=lambda p: p.real) if real else np.nan
    cpx = [p for p in sp if p.imag > 1.0]
    lcl = min(cpx, key=lambda p: abs(abs(p.imag)/(2*np.pi) - fres_)) if cpx else np.nan
    low = [p for p in cpx if p.imag/(2*np.pi) < 400]
    slow = max(low, key=lambda p: p.real) if low else np.nan
    return dc, lcl, slow


R_loop = Rcp + Rgp
pr(f"  loop resistance of the model R' = Rc' + Rg' = {R_loop*1e3:.2f} mpu ({(Rc + Rg)*1e3:.1f} mOhm); DC plant gain k_mod/R' = {k_mod/R_loop:.0f}")
cases = [(kp_i, ki_i, 'library 0.1 / 20'), (kp_rec, ki_rec, 'recommended 0.15 / 20'), (0.2, 20.0, '0.2 / 20'), (0.15, 30.0, '0.15 / 30'), (kp_old, ki_old, 'bench 0.5 / 45')]
for kp, ki, lab in cases:
    Lw, _ = loop_d(kp, ki)
    wc, pm, mm = margins(Lw, w)
    Lres = np.interp(f_res, f, np.abs(Lw))
    L0 = ceq_d(np.array([1.0 + 0j]), kp, ki)[0].real*k_mod/R_loop
    poles = closed_loop_poles(kp, ki)
    dc, lcl, slow = describe_poles(poles, f_res)
    unstable = np.any(np.abs(poles) > 1 + 1e-9)
    pr(f'  {lab}: fc {wc/(2*np.pi):.0f} Hz ({wc:.0f} rad/s), PM {pm:.1f} deg, min|1+L| {mm:.3f}, |L(f_res)| {Lres:.2f}, L(0) = {L0:+.2f}; '
       f'{"UNSTABLE" if unstable else "stable"}: DC pole {dc.real:+.1f} 1/s, LCL mode sigma {lcl.real:.0f} 1/s at {lcl.imag/(2*np.pi):.0f} Hz, '
       f'slowest LF pair sigma {slow.real:.0f} 1/s at {slow.imag/(2*np.pi):.0f} Hz')

# DC condition in closed form: C_eq(0) = kp (ab-1)/ab - ki/w0 (a+b-delta)/ab  (continuous), stability L(0) > -1
a_, b_ = 1.0, 4.0
ab = a_*b_
c_kp = (ab - 1)/ab
c_ki = (a_ + b_ - delta)/ab
pr(f'  DC gain of C_eq (continuous): {c_kp:.4f} kp - {c_ki:.4f} ki/w0; discrete as coded: {Hu_d0[0].real:.4f} kp - {Hv_d0[0].real:.4f} ki/w0 (+ ki ts/2 term)')
ratio_lim = c_kp/c_ki*w0
pr(f'  C_eq(0) > 0 requires ki/kp < (ab-1)/(a+b-delta) w0 = {ratio_lim:.0f} rad/s; library 0.1/20: ki/kp = {ki_i/kp_i:.0f}, recommended: {ki_rec/kp_rec:.0f}, bench 0.5/45: {ki_old/kp_old:.0f}')
kp_min_dc = c_ki*ki_i/w0/c_kp
pr(f'  ... i.e. kp > {c_ki/c_kp:.3f} ki/w0 = {kp_min_dc:.4f} for ki = 20 (continuous); with the loop resistance the limit is kp > {kp_min_dc - R_loop/k_mod/c_kp:.4f} '
   f'(R\' = {R_loop*1e3:.2f} mpu) and {kp_min_dc - 2*R_loop/k_mod/c_kp:.4f} with twice that resistance')
# exact discrete DC limit for ki = 20 by bisection on the DC pole
def dc_pole(kp, ki, Lg_=Lg, Rg_=Rg):
    dc, _, _ = describe_poles(closed_loop_poles(kp, ki, Lg_, Rg_), f_res)
    return dc.real
lo, hi = 0.05, 0.2
for _ in range(40):
    mid = 0.5*(lo + hi)
    if dc_pole(mid, ki_i) > 0:
        lo = mid
    else:
        hi = mid
pr(f'  exact (discrete model, R\' of the model): the DC pole crosses zero at kp = {hi:.4f} for ki = 20')
for Rx, lab in [(2*(Rc + Rg), 'with 2 x the resistance (devices)'), ((Rc + Rg) + 18e-3, 'with 18 mOhm more (two T-type legs, 1.5 devices each)')]:
    Rg_x = Rx - Rc
    lo, hi = 0.05, 0.2
    for _ in range(40):
        mid = 0.5*(lo + hi)
        if dc_pole(mid, ki_i, Lg, Rg_x) > 0:
            lo = mid
        else:
            hi = mid
    pr(f'  ... {lab} ({Rx*1e3:.1f} mOhm): kp limit {hi:.4f}; DC pole with kp = 0.1: {dc_pole(0.1, ki_i, Lg, Rg_x):+.2f} 1/s')

# design formulas at the crossover (recommended gains)
Lc_tot = Lcp + Lgp
Lw_r, _ = loop_d(kp_rec, ki_rec)
Cw_r = ceq_d(z, kp_rec, ki_rec)
wc_r, pm_r, mm_r = margins(Lw_r, w)
i_c = np.argmin(np.abs(w - wc_r))
pr(f'  recommended gains: kp k_mod/(Lc\'+Lg\') = {kp_rec*k_mod/Lc_tot:.0f} rad/s; exact crossover {wc_r:.0f} rad/s: |C_eq| = {abs(Cw_r[i_c]):.3f} /_ {np.degrees(np.angle(Cw_r[i_c])):.1f} deg '
   f'(kp H_u = {kp_rec*abs(fht_tf_d(np.array([z[i_c]]))[0][0]):.3f}, resonant part 2 ki w/(w^2 - w0^2) = {2*ki_rec*wc_r/(wc_r**2 - w0**2):.3f}), '
   f'|P_zoh| = {abs(Pd[i_c]):.2f} /_ {np.degrees(np.angle(Pd[i_c])):.1f} deg, acquisition delay {np.degrees(wc_r*ts):.1f} deg')
Hu_cw, Hv_cw = fht_tf_d(np.array([z[i_c]]))
pr(f'  FHT at the crossover: H_u = {abs(Hu_cw[0]):.3f} /_ {np.degrees(np.angle(Hu_cw[0])):.1f} deg (incl. one step), H_v = {abs(Hv_cw[0]):.3f} /_ {np.degrees(np.angle(Hv_cw[0])):.1f} deg')
# the ideal PR with the same gains for comparison: kp + 2 ki s/(s^2 + w0^2), delay 1.5 ts, continuous plant
PRc = kp_rec + 2*ki_rec*s/(s**2 + w0**2)
L_pr = PRc*np.exp(-1.5j*w*ts)*Pc
wc_p, pm_p, mm_p = margins(L_pr, w)
pr(f'  ideal PR (no FHT) with the same gains: fc {wc_p/(2*np.pi):.0f} Hz, PM {pm_p:.1f} deg, |L(f_res)| = {np.interp(f_res, f, np.abs(L_pr)):.2f} (the FHT attenuation is what keeps the resonance below 1)')

# ---- candidate gain sets: crossover, margins, DC pole, weak-grid limit
pr('  candidate gain sets (Lg = 0.5 mH unless stated):')
pr('    kp    ki   fc[Hz]  PM[deg]  |L(fres)|  min|1+L|  DC pole[1/s]  LF pair sigma[1/s]  Lg limit of the LCL mode [mH]')
Lg_scan = np.linspace(0.3e-3, 2.5e-3, 45)
cand_rows = []
for kp, ki in [(0.1, 20.0), (0.1, 15.0), (0.12, 15.0), (0.12, 20.0), (0.15, 20.0), (0.15, 15.0), (0.2, 20.0), (0.5, 45.0)]:
    Lw, _ = loop_d(kp, ki)
    wc, pm, mm = margins(Lw, w)
    poles = closed_loop_poles(kp, ki)
    dc, lcl, slow = describe_poles(poles, f_res)
    lg_lim = np.nan
    for Lg_ in Lg_scan:
        fres_ = 1/(2*np.pi*np.sqrt(Lc*Lg_/(Lc + Lg_)*Cf))
        _, lcl_, _ = describe_poles(closed_loop_poles(kp, ki, Lg_, Rg), fres_)
        if lcl_.real > 0:
            lg_lim = Lg_
            break
    cand_rows.append((kp, ki, wc/(2*np.pi), pm, np.interp(f_res, f, np.abs(Lw)), mm, dc.real, slow.real, lg_lim*1e3))
    pr(f'    {kp:<5} {ki:<5} {wc/(2*np.pi):6.0f}  {pm:7.1f}  {np.interp(f_res, f, np.abs(Lw)):8.2f}  {mm:8.3f}  {dc.real:+11.1f}  {slow.real:+16.0f}  {lg_lim*1e3:8.2f}')

# ---- FHT variants at the fundamental: forward Euler with delta = 0.05 (as coded), forward Euler with delta = 0, exact rotation with delta = 0
pr('  FHT internal model variants (gains placed at exp([-1 -4] w0 ts) in each case), response from the acquired sample to the estimate used in the step:')


def fht_variant(Ad_, z):
    """Pole placement of the observer x(k+1) = Ad x + L (w - C x) at z1, z2, then H_u, H_v (quadrature = -x2/w0) at the points z."""
    z1, z2 = np.exp(-w0*ts), np.exp(-4*w0*ts)
    # det(zI - Ad + L C) = z^2 - (tr(Ad) - l1) z + det(Ad) - l1 Ad[1,1] + l2 Ad[0,1]  ->  match z^2 - (z1+z2) z + z1 z2
    l1 = np.trace(Ad_) - (z1 + z2)
    l2 = (z1*z2 - np.linalg.det(Ad_) + l1*Ad_[1, 1])/Ad_[0, 1]
    out = []
    for zz in z:
        M_ = zz*np.eye(2) - Ad_ + np.outer([l1, l2], [1.0, 0.0])
        x = np.linalg.solve(M_, [l1, l2])
        out.append((x[0], -x[1]/w0))
    out = np.array(out)
    return l1, l2, out[:, 0], out[:, 1]


zpts = np.array([1.0 + 0j, np.exp(1j*w0*ts), np.exp(1j*2*np.pi*f_res*ts)])
for nm, Ad_ in [('forward Euler, delta = 0.05 (as coded)', np.array([[1.0, ts], [-w0**2*ts, 1 - delta*w0*ts]])),
                ('forward Euler, delta = 0', np.array([[1.0, ts], [-w0**2*ts, 1.0]])),
                ('exact rotation, delta = 0', np.array([[np.cos(w0*ts), np.sin(w0*ts)/w0], [-w0*np.sin(w0*ts), np.cos(w0*ts)]]))]:
    l1v, l2v, Huv, Hvv = fht_variant(Ad_, zpts)
    pr(f'    {nm}: l1 = {l1v:.4f}, l2 = {l2v:.2f}; H_u(w0) = {abs(Huv[1]):.4f} /_ {np.degrees(np.angle(Huv[1])):+.2f} deg, H_v(w0) = {abs(Hvv[1]):.4f} /_ {np.degrees(np.angle(Hvv[1])):+.2f} deg; '
       f'DC: H_u(1) = {Huv[0].real:.4f}, H_v(1) = {Hvv[0].real:.4f}; |H_u(f_res)| = {abs(Huv[2]):.3f}')

# ---- figure: plant
fig, ax = plt.subplots(2, 1, figsize=(FIGW, 4.2), sharex=True)
for Lg_, lab, ls in [(0.5e-3, r'$L_g = 0.5$ mH (bench)', '-'), (1.0e-3, r'$L_g = 1$ mH', '--'), (0.2e-3, r'$L_g = 0.2$ mH', ':')]:
    A_, B_, C_ = plant_ss(Lg_, Rg)
    Pw = k_mod*tf_eval(A_, B_, C_, s, 0, 0)
    ax[0].semilogx(f, 20*np.log10(np.abs(Pw)), ls, label=lab)
    ax[1].semilogx(f, np.degrees(np.unwrap(np.angle(Pw))), ls)
ax[0].semilogx(f, 20*np.log10(np.abs(Pc_stiff)), '-.', color='gray', label=r'stiff grid at the terminals ($L_g \to 0$)')
for a in ax:
    a.axvline(fs/6, color='k', lw=0.6, alpha=0.5)
    a.axvline(fs/2, color='k', lw=0.6, alpha=0.5)
ax[0].text(fs/6*1.03, 45, r'$f_s/6$', fontsize=7)
ax[0].text(fs/2*0.78, 45, r'$f_s/2$', fontsize=7)
ax[0].set_ylabel(r'$|P|$ [dB]'); ax[1].set_ylabel(r'$\angle P$ [deg]'); ax[1].set_xlabel('f [Hz]')
ax[0].legend(loc='lower left'); ax[0].set_title(r'Plant $P(s) = i_s / y$: converter current per unit of PI output (pu/pu), continuous time')
savefig(fig, 'fig_plant_bode.pdf')

# ---- figure: FHT
fig, ax = plt.subplots(2, 2, figsize=(FIGW, 4.4), sharex=True)
ax[0, 0].semilogx(f, 20*np.log10(np.abs(Hu_c)), label='continuous, poles $-\\omega_0, -4\\omega_0$')
ax[0, 0].semilogx(f, 20*np.log10(np.abs(Hu_d*z)), '--', label='discrete as coded (acquisition delay removed)')
ax[1, 0].semilogx(f, np.degrees(np.unwrap(np.angle(Hu_c))))
ax[1, 0].semilogx(f, np.degrees(np.unwrap(np.angle(Hu_d*z))), '--')
ax[0, 1].semilogx(f, 20*np.log10(np.abs(Hv_c)))
ax[0, 1].semilogx(f, 20*np.log10(np.abs(Hv_d*z)), '--')
ax[1, 1].semilogx(f, np.degrees(np.unwrap(np.angle(Hv_c))))
ax[1, 1].semilogx(f, np.degrees(np.unwrap(np.angle(Hv_d*z))), '--')
ax[0, 0].semilogx(f, 20*np.log10(np.abs(l1c/(s + l1c))), ':', color='gray', label=r'$l_1/(s + l_1)$')
ax[0, 0].set_title(r'in-phase output $H_u = \hat{\imath}_\alpha / i_s$'); ax[0, 1].set_title(r'quadrature output $H_v = \hat{\imath}_\beta / i_s$')
ax[0, 0].set_ylabel('[dB]'); ax[1, 0].set_ylabel('[deg]'); ax[1, 0].set_xlabel('f [Hz]'); ax[1, 1].set_xlabel('f [Hz]')
ax[0, 0].set_ylim(-40, 10); ax[0, 1].set_ylim(-40, 10)
for a in ax.flat:
    a.axvline(f0, color='k', lw=0.6, alpha=0.5)
ax[0, 0].legend(loc='lower left', fontsize=6.5, framealpha=0.9)
savefig(fig, 'fig_fht.pdf')

# ---- figure: equivalent controller
fig, ax = plt.subplots(2, 1, figsize=(FIGW, 4.4), sharex=True)
for kp, ki, lab, col in [(kp_i, ki_i, r'$k_{p,i} = 0.1$, $k_{i,i} = 20$ s$^{-1}$ (library default)', 'C0'), (kp_rec, ki_rec, r'$k_{p,i} = 0.15$, $k_{i,i} = 20$ s$^{-1}$ (recommended)', 'C2'), (kp_old, ki_old, r'$k_{p,i} = 0.5$, $k_{i,i} = 45$ s$^{-1}$ (bench, old modules)', 'C3')]:
    Cw_ = ceq_d(z, kp, ki)
    ax[0].semilogx(f, 20*np.log10(np.abs(Cw_)), color=col, label=lab)
    ax[1].semilogx(f, np.degrees(np.angle(Cw_)), color=col)
    PR_ = kp + 2*ki*s/(s**2 + w0**2)
    ax[0].semilogx(f, 20*np.log10(np.abs(PR_)), '--', color=col, lw=0.7, alpha=0.6)
    ax[1].semilogx(f, np.degrees(np.angle(PR_)), '--', color=col, lw=0.7, alpha=0.6)
ax[0].set_ylabel(r'$|C_{eq}|$ [dB]'); ax[1].set_ylabel(r'$\angle C_{eq}$ [deg]'); ax[1].set_xlabel('f [Hz]')
ax[0].set_title(r'Equivalent stationary-frame controller $C_{eq}(z)$ (solid) and ideal PR $k_p + 2k_i s/(s^2+\omega_0^2)$ with the same gains (dashed)')
ax[0].legend(loc='upper right'); ax[0].set_ylim(-40, 40)
for a in ax:
    a.axvline(f0, color='k', lw=0.6, alpha=0.5)
savefig(fig, 'fig_ceq.pdf')

# ---- figure: loop Bode
fig, ax = plt.subplots(2, 1, figsize=(FIGW, 4.6), sharex=True)
for kp, ki, lab, col in [(kp_i, ki_i, 'library 0.1 / 20', 'C0'), (kp_rec, ki_rec, 'recommended 0.15 / 20', 'C2'), (kp_old, ki_old, 'bench 0.5 / 45', 'C3')]:
    Lw, _ = loop_d(kp, ki)
    wc, pm, mm = margins(Lw, w)
    ax[0].semilogx(f, 20*np.log10(np.abs(Lw)), color=col, label=f'{lab}: $f_c$ = {wc/(2*np.pi):.0f} Hz, PM {pm:.0f} deg, $|L(f_{{res}})|$ = {np.interp(f_res, f, np.abs(Lw)):.2f}')
    ax[1].semilogx(f, np.degrees(np.unwrap(np.angle(Lw))), color=col)
ax[0].axhline(0, color='k', lw=0.6); ax[1].axhline(-180, color='k', lw=0.6)
for a in ax:
    a.axvline(f_res, color='k', lw=0.6, alpha=0.5); a.axvline(fs/6, color='k', lw=0.6, alpha=0.5, ls='--')
ax[0].text(f_res*1.05, 25, r'$f_{res}$', fontsize=7); ax[0].text(fs/6*0.72, 25, r'$f_s/6$', fontsize=7)
ax[0].set_ylabel(r'$|L|$ [dB]'); ax[1].set_ylabel(r'$\angle L$ [deg]'); ax[1].set_xlabel('f [Hz]')
ax[1].set_ylim(-540, 0); ax[1].set_yticks([-540, -450, -360, -270, -180, -90, 0])
ax[0].set_title(r'Open-loop gain $L(z) = C_{eq}(z)\,z^{-1}\,P_{zoh}(z)$ of the current loop, $L_g = 0.5$ mH')
ax[0].legend(loc='lower left')
savefig(fig, 'fig_loop_bode.pdf')

# ---- figure: Nyquist
fig, ax = plt.subplots(1, 3, figsize=(FIGW, 2.5))
for k, (kp, ki, lab, col) in enumerate([(kp_i, ki_i, 'library 0.1 / 20', 'C0'), (kp_rec, ki_rec, 'recommended 0.15 / 20', 'C2'), (kp_old, ki_old, 'bench 0.5 / 45', 'C3')]):
    Lw, _ = loop_d(kp, ki)
    ax[k].plot(Lw.real, Lw.imag, color=col, lw=1)
    ax[k].plot(Lw.real, -Lw.imag, color=col, lw=0.6, alpha=0.4)
    ax[k].plot(-1, 0, 'k+', ms=8)
    th = np.linspace(0, 2*np.pi, 200)
    ax[k].plot(-1 + np.cos(th), np.sin(th), ':', color='gray', lw=0.6)
    ax[k].set_aspect('equal'); ax[k].set_title(lab, fontsize=8)
    ax[k].set_xlabel('Re')
    ax[k].set_xlim(-3.2, 1.2); ax[k].set_ylim(-2.2, 2.2)
    i_r = np.argmin(np.abs(f - f_res))
    ax[k].plot(Lw[i_r].real, Lw[i_r].imag, 'o', color=col, ms=4)
    ax[k].annotate(r'$f_{res}$', (Lw[i_r].real, Lw[i_r].imag), textcoords='offset points', xytext=(4, 4), fontsize=7)
    L0 = ceq_d(np.array([1.0 + 0j]), kp, ki)[0].real*k_mod/R_loop
    ax[k].plot(L0, 0, 's', color=col, ms=4)
    ax[k].annotate(r'$L(0)$', (L0, 0), textcoords='offset points', xytext=(4, -10), fontsize=7)
ax[0].set_ylabel('Im')
fig.suptitle('Nyquist plot of the current loop (unit circle around $-1$ dotted; $f > 0$ solid, $f < 0$ faint)', fontsize=9)
savefig(fig, 'fig_nyquist.pdf')

# ---- stability map in (kp, ki) for Lg = 0.5 mH
pr('  stability map (kp, ki) at Lg = 0.5 mH ...')
kps = np.linspace(0.02, 0.6, 59)
kis = np.linspace(0.0, 80.0, 41)
pp = plant_polys()
stab = np.zeros((len(kis), len(kps)))
dcmap = np.zeros_like(stab)
lclmap = np.zeros_like(stab)
for i, ki in enumerate(kis):
    for j, kp in enumerate(kps):
        poles = closed_loop_poles(kp, ki, pp=pp)
        dc, lcl, slow = describe_poles(poles, f_res)
        stab[i, j] = np.max(np.abs(poles))
        dcmap[i, j] = dc.real if not np.isnan(dc) else 0.0
        lclmap[i, j] = lcl.real
fig, ax = plt.subplots(1, 1, figsize=(FIGW*0.75, 3.4))
KP, KI = np.meshgrid(kps, kis)
ax.contourf(KP, KI, (stab < 1).astype(float), levels=[-0.5, 0.5, 1.5], colors=['#f4c7c3', '#c8e6c9'])
cs1 = ax.contour(KP, KI, dcmap, levels=[0.0], colors='C0', linewidths=1.2)
cs2 = ax.contour(KP, KI, lclmap, levels=[0.0], colors='C3', linewidths=1.2)
ax.clabel(cs1, fmt={0.0: 'DC pole = 0'}, fontsize=7)
ax.clabel(cs2, fmt={0.0: 'LCL mode = 0'}, fontsize=7)
ax.plot(c_ki/c_kp*kis/w0 - R_loop/k_mod/c_kp, kis, 'k:', lw=0.8, label=r'$C_{eq}(0) = -R^\prime/k_{mod}$ (closed form)')
for kp, ki, lab, mk in [(kp_i, ki_i, 'library 0.1 / 20', 'o'), (kp_rec, ki_rec, 'recommended 0.15 / 20', 's'), (kp_old, ki_old, 'bench 0.5 / 45', '^')]:
    ax.plot(kp, ki, mk, color='k', ms=5, label=lab)
ax.set_xlabel(r'$k_{p,i}$'); ax.set_ylabel(r'$k_{i,i}$ [s$^{-1}$]'); ax.legend(fontsize=7, loc='upper left')
ax.set_title(r'Stable region (green) of the current loop, $L_g = 0.5$ mH, $R^\prime$ of the model', fontsize=9)
savefig(fig, 'fig_stability_map.pdf')

# ---- robustness vs line inductance (recommended gains and library gains)
pr('  robustness vs the line inductance:')
Lg_sw = np.linspace(0.05e-3, 2.0e-3, 60)
res = {}
for kp, ki, lab in [(kp_rec, ki_rec, 'rec'), (kp_i, ki_i, 'lib')]:
    rows = []
    for Lg_ in Lg_sw:
        Lw, _ = loop_d(kp, ki, Lg_, Rg)
        wc, pm, mm = margins(Lw, w)
        fres_ = 1/(2*np.pi*np.sqrt(Lc*Lg_/(Lc + Lg_)*Cf))
        poles = closed_loop_poles(kp, ki, Lg_, Rg)
        dc, lcl, slow = describe_poles(poles, fres_)
        rows.append((Lg_, wc, pm, mm, fres_, lcl.real, lcl.imag/(2*np.pi), dc.real, np.interp(fres_, f, np.abs(Lw))))
    res[lab] = np.array(rows)
    pr(f'   {lab}:')
    for row in res[lab][::10]:
        pr(f'    Lg = {row[0]*1e3:.2f} mH: f_res {row[4]:.0f} Hz, fc {row[1]/(2*np.pi):.0f} Hz, PM {row[2]:.0f} deg, min|1+L| {row[3]:.2f}, |L(f_res)| {row[8]:.2f}, LCL mode sigma {row[5]:.0f} 1/s at {row[6]:.0f} Hz, DC pole {row[7]:+.1f} 1/s')
r_ = res['rec']
unst = r_[r_[:, 5] > 0]
if len(unst):
    pr(f'   recommended gains: LCL mode unstable from Lg = {unst[0, 0]*1e3:.2f} mH (f_res {unst[0, 4]:.0f} Hz)')
fig, ax = plt.subplots(1, 3, figsize=(FIGW, 2.6))
for lab, col, nm in [('rec', 'C2', '0.15 / 20'), ('lib', 'C0', '0.1 / 20')]:
    r_ = res[lab]
    ax[0].plot(r_[:, 0]*1e3, r_[:, 2], color=col, label=f'PM [deg], {nm}')
    ax[0].plot(r_[:, 0]*1e3, r_[:, 1]/(2*np.pi)/4, '--', color=col, label=f'$f_c$/4 [Hz], {nm}')
    ax[1].plot(r_[:, 0]*1e3, r_[:, 5], color=col, label=f'LCL mode, {nm}')
    ax[1].plot(r_[:, 0]*1e3, r_[:, 7]*10, ':', color=col, label=f'DC pole x10, {nm}')
    ax[2].plot(r_[:, 0]*1e3, r_[:, 8], color=col, label=f'$|L(f_{{res}})|$, {nm}')
ax[0].set_xlabel(r'$L_g$ [mH]'); ax[0].legend(fontsize=6); ax[0].set_title('phase margin and crossover')
ax[1].set_xlabel(r'$L_g$ [mH]'); ax[1].set_title(r'closed-loop $\sigma$ [s$^{-1}$]'); ax[1].axhline(0, color='k', lw=0.6); ax[1].legend(fontsize=6); ax[1].set_ylim(-200, 100)
ax[2].set_xlabel(r'$L_g$ [mH]'); ax[2].set_title('loop gain at the resonance'); ax[2].axhline(1, color='k', lw=0.6); ax[2].legend(fontsize=6)
ax2b = ax[2].twinx(); ax2b.plot(r_[:, 0]*1e3, r_[:, 4], color='gray', lw=0.7); ax2b.axhline(fs/6, color='gray', lw=0.6, ls=':'); ax2b.set_ylabel(r'$f_{res}$ [Hz]', color='gray', fontsize=7); ax2b.grid(False)
savefig(fig, 'fig_robust_lg.pdf')

# =============================================================================================== 5. PLL and the virtual quadrature
pr('=' * 100)
pr('5. PLL: VIRTUAL QUADRATURE (SOGI EQUIVALENCE), LOOP AND OBSERVER')
k_sogi = 1/(pll_tau*w0)
pr(f'  low pass tau = 2/w0 = {pll_tau*1e3:.2f} ms in the PLL frame, fed back as beta: SOGI-QSG with k = 1/(tau w0) = {k_sogi:.2f}, '
   f'poles s^2 + k w0 s + w0^2: zeta = {k_sogi/2:.3f}, sigma = -{k_sogi*w0/2:.1f} 1/s (time constant {2/(k_sogi*w0)*1e3:.1f} ms), DC gain of the beta path {k_sogi:.2f}')
pr(f'  PLL loop (type 1, ki = 0): pole at -kp w_nom = -{pll_kp*w0:.0f} 1/s ({1/(pll_kp*w0)*1e3:.2f} ms); static phase error per Hz of offset {np.degrees(2*np.pi/(pll_kp*w0)):.2f} deg')
pr(f'  angle observer poles exp([-1 -4] w0 ts): continuous -{w0:.0f}, -{4*w0:.0f} 1/s; l1 = {obs_l1:.4f}, l2 = {obs_l2:.2f}')


def Hbeta_c(s):
    return (w0/pll_tau)/(s**2 + s/pll_tau + w0**2)


Hb = Hbeta_c(s)
pr(f'  H_beta(j w0) = {Hbeta_c(np.array([1j*w0]))[0]:.4f}; |H_beta| at 150 Hz {abs(Hbeta_c(np.array([3j*w0]))[0]):.3f}, 250 Hz {abs(Hbeta_c(np.array([5j*w0]))[0]):.3f}')

# =============================================================================================== 6. DC-link loop
pr('=' * 100)
pr('6. DC-LINK VOLTAGE LOOP (mode 1)')
K_dc = (ug_pk*ibez/2)/(Cdc*udc_nom**2)
pr(f'  plant: d(udc_pu)/dt = -K i_d, K = (ug_pk ibez/2)/(Cdc udc_nom^2) = {K_dc:.1f} 1/s per pu')
N_ma = int(round(2*np.pi/(w0*ts)))
pr(f'  moving average over N = {N_ma} samples ({N_ma*ts*1e3:.0f} ms), group delay (N-1) ts/2 = {(N_ma - 1)*ts/2*1e3:.2f} ms')
wc_dc = kp_udc*K_dc
pr(f'  crossover ~ kp_udc K = {wc_dc:.1f} rad/s ({wc_dc/(2*np.pi):.1f} Hz); PI zero at ki/kp = {ki_udc/kp_udc:.0f} rad/s')
w_dc = np.logspace(-1, 3, 1000)
s_dc = 1j*w_dc
z_dc = np.exp(s_dc*ts)
H_ma = (1 - z_dc**(-N_ma))/(N_ma*(1 - z_dc**(-1)))
C_dc = kp_udc + ki_udc*ts*z_dc/(z_dc - 1)
wc_i = wc_r                               # crossover of the (recommended) current loop
T_i = 1/(1 + s_dc/wc_i)
L_dc = C_dc*T_i*K_dc/s_dc*H_ma*np.exp(-1j*w_dc*ts)
wc_d, pm_d, mm_d = margins(L_dc, w_dc)
pr(f'  with MA, inner loop as a lag at {wc_i:.0f} rad/s: crossover {wc_d:.1f} rad/s, PM {pm_d:.1f} deg (MA lag at the crossover {np.degrees(wc_d*(N_ma - 1)*ts/2):.1f} deg, '
   f'PI zero {np.degrees(np.arctan(ki_udc/(kp_udc*wc_d))):.1f} deg, inner loop {np.degrees(np.arctan(wc_d/wc_i)):.1f} deg)')
L_dc_nf = C_dc*T_i*K_dc/s_dc*np.exp(-1j*w_dc*ts)
wc_n, pm_n, _ = margins(L_dc_nf, w_dc)
pr(f'  without the average: crossover {wc_n:.1f} rad/s, PM {pm_n:.1f} deg; |L_dc| at 50 Hz without filter {np.interp(w0, w_dc, np.abs(L_dc_nf)):.3f}, at 100 Hz {np.interp(2*w0, w_dc, np.abs(L_dc_nf)):.3f}; with MA {np.interp(2*w0, w_dc, np.abs(L_dc)):.1e}')
tau_lp = 5e-3
L_dc_lp = L_dc_nf/(1 + s_dc*tau_lp)
wc_l, pm_l, _ = margins(L_dc_lp, w_dc)
pr(f'  with a first-order low pass of {tau_lp*1e3:.0f} ms instead: crossover {wc_l:.1f} rad/s, PM {pm_l:.1f} deg, |L_dc| at 50 Hz {np.interp(w0, w_dc, np.abs(L_dc_lp)):.3f}, at 100 Hz {np.interp(2*w0, w_dc, np.abs(L_dc_lp)):.3f}')
# closed-loop pole of the DC-link loop (dominant): 1 + L = 0 near the crossover -> from a dense root search on the discrete model is overkill; report the 2 % settling from the simulation below

fig, ax = plt.subplots(2, 1, figsize=(FIGW, 4.2), sharex=True)
for Lw_, lab, col in [(L_dc, 'moving average over one period', 'C0'), (L_dc_lp, f'first-order low pass {tau_lp*1e3:.0f} ms', 'C3'), (L_dc_nf, 'no measurement filter', 'gray')]:
    ax[0].semilogx(w_dc/(2*np.pi), 20*np.log10(np.abs(Lw_)), color=col, label=lab)
    ax[1].semilogx(w_dc/(2*np.pi), np.degrees(np.unwrap(np.angle(Lw_))), color=col)
ax[0].axhline(0, color='k', lw=0.6); ax[1].axhline(-180, color='k', lw=0.6)
for a in ax:
    a.axvline(f0, color='k', lw=0.6, alpha=0.5); a.axvline(2*f0, color='k', lw=0.6, alpha=0.5)
ax[0].set_ylabel(r'$|L_{dc}|$ [dB]'); ax[1].set_ylabel(r'$\angle L_{dc}$ [deg]'); ax[1].set_xlabel('f [Hz]')
ax[1].set_ylim(-400, 0); ax[0].set_ylim(-80, 60)
ax[0].set_title(r'DC-link voltage loop, $k_{p,u} = 2$, $k_{i,u} = 20$ s$^{-1}$, inner loop as a lag at $\omega_{c,i}$')
ax[0].legend(loc='lower left')
savefig(fig, 'fig_dclink_loop.pdf')

# =============================================================================================== 7. sampled-data simulation
pr('=' * 100)
pr('7. SAMPLED-DATA SIMULATION (controller as coded on the average model of the power stage)')


class Ctrl:
    """Port of the control step of afe_sp_ttype_ctrl_pwm.m (acquisitions, PLL, FHT, references, current loop); the
    carrier and the gates are replaced by the average bridge voltage m udc held over the control period."""

    def __init__(self, p):
        self.p = p
        self.is_z = self.uac_z = self.udc_z = 0.0
        self.th = 0.0
        self.xi_in_z = self.eta_in_z = self.xi_f = self.eta_f = 0.0
        self.x_pll = 0.0
        self.x_obs = np.zeros(2)
        self.fht_x = np.zeros(2)
        self.ref_z = 0.0
        self.udc_ref_f = -1.0
        self.udc_f = -1.0
        self.N = int(min(max(round(2*np.pi/(p['omega_nom']*ts)), 1), 1024))
        self.udc_buf = np.zeros(self.N)
        self.buf_k = 0
        self.x_udc = 0.0
        self.xd_i = self.xq_i = 0.0
        self.clip_z = 0
        self.enable_z = False
        self.m = 0.0
        self.log = {}

    @staticmethod
    def wrap(u):
        return -np.pi + np.mod(u + 3*np.pi, 2*np.pi)

    def step(self, t, i_s, u_ac, udc, sync=None, rng=None):
        p = self.p
        q = p['adc_q']
        enable = t >= p['t_enable']
        isd, uacd, udcd = self.is_z, self.uac_z, self.udc_z
        i_raw = i_s + (p['i_noise_std']*rng.standard_normal() if (rng is not None and p['i_noise_std'] > 0) else 0.0)
        self.is_z = q*np.round(i_raw/q)
        self.uac_z = q*np.round(u_ac/q)
        self.udc_z = q*np.round(udc/q)
        if self.udc_ref_f < 0:
            self.udc_ref_f = self.udc_f = self.udc_z
            self.udc_buf[:] = self.udc_z
        # PLL
        ug = uacd/p['ug_nom']
        theta = self.th
        a = ts/p['pll_tau']
        self.xi_f = a*self.xi_in_z + (1 - a)*self.xi_f
        self.eta_f = a*self.eta_in_z + (1 - a)*self.eta_f
        ubeta = self.xi_f*np.sin(theta) + self.eta_f*np.cos(theta)
        u_xi = ug*np.cos(theta) + ubeta*np.sin(theta)
        u_eta = -ug*np.sin(theta) + ubeta*np.cos(theta)
        self.xi_in_z, self.eta_in_z = u_xi, u_eta
        P_pll = p['pll_kp']*u_eta
        y_pll = P_pll + p['pll_ki']*ts*u_eta + self.x_pll
        y_pll = min(max(y_pll, -p['pll_lim']), p['pll_lim'])
        self.x_pll = y_pll - P_pll
        omega_pll_pu = 1 + y_pll
        self.th = self.wrap(theta + p['omega_nom']*ts*omega_pll_pu)
        theta_hat = self.wrap(self.x_obs[0])
        e_obs = self.wrap(theta - theta_hat)
        omega_hat = self.x_obs[1]
        x1n = self.wrap(self.x_obs[0] + ts*self.x_obs[1] + p['obs_l1']*e_obs)
        x2n = self.x_obs[1] + p['obs_l2']*e_obs
        self.x_obs = np.array([x1n, x2n])
        # FHT
        wf, dl = p['fht_omega'], p['fht_delta']
        i_alpha = self.fht_x[0]
        i_beta = -self.fht_x[1]/wf
        e_f = isd - self.fht_x[0]
        f1n = self.fht_x[0] + ts*self.fht_x[1] + p['fht_l1']*e_f
        f2n = -wf*wf*ts*self.fht_x[0] + (1 - dl*wf*ts)*self.fht_x[1] + p['fht_l2']*e_f
        self.fht_x = np.array([f1n, f2n])
        i_d = i_alpha*np.cos(theta_hat) + i_beta*np.sin(theta_hat)
        i_q = -i_alpha*np.sin(theta_hat) + i_beta*np.cos(theta_hat)
        # references
        i_q_ref = p['iq_ref']
        udc_ref = e_udc = 0.0
        if p['mode'] == 1:
            a_r = ts/p['udc_tau']
            self.udc_ref_f += a_r*(p['udc_ref'] - self.udc_ref_f)
            udc_ref = self.udc_ref_f
            udc_m = udcd
            if p['udc_ma']:
                self.buf_k = self.buf_k + 1 if self.buf_k < self.N else 1
                self.udc_buf[self.buf_k - 1] = udcd
                udc_m = np.mean(self.udc_buf)
            if p['udc_meas_tau'] > 0:
                self.udc_f += ts/p['udc_meas_tau']*(udc_m - self.udc_f)
            else:
                self.udc_f = udc_m
            if enable and not p['slave']:
                e_udc = udc_ref - self.udc_f
                P_u = p['kp_udc']*e_udc
                y_u = P_u + p['ki_udc']*ts*e_udc + self.x_udc
                y_s = min(max(y_u, -p['i_lim']), p['i_lim'])
                self.x_udc = y_s - P_u
                i_d_ref = -y_s
            else:
                self.x_udc = 0.0
                i_d_ref = 0.0
            self.ref_z = i_d_ref
        else:
            target = p['i_ref'] if t >= p['t_ref'] else 0.0
            dref = min(max(target - self.ref_z, -p['i_ref_slew']*ts), p['i_ref_slew']*ts)
            self.ref_z += dref
            i_d_ref = self.ref_z
            if not enable:
                self.ref_z = 0.0
                i_d_ref = 0.0
        # current loop
        clip = 0
        if not enable:
            self.xd_i = self.xq_i = 0.0
            m_d = m_q = 0.0
        elif p['slave']:
            m_d, m_q = sync[1], sync[2]
        else:
            if not self.enable_z:
                k_ff = p['ug_nom']*p['ubez']/p['udc_nom']
                dth = 1.5*p['omega_nom']*omega_pll_pu*ts
                self.xd_i = k_ff*(u_xi*np.cos(dth) - u_eta*np.sin(dth))
                self.xq_i = k_ff*(u_xi*np.sin(dth) + u_eta*np.cos(dth))
            udc_ff = max(self.udc_z if p.get('ff_no_delay', 0) else udcd, p['udc_min'])
            ed, eq = i_d_ref - i_d, i_q_ref - i_q
            if self.clip_z > 0.5:
                xd_new = self.xd_i - p['clip_release']*ts*self.xd_i
                md = p['kp_i']*ed + self.xd_i
                xq_new = self.xq_i - p['clip_release']*ts*self.xq_i
                mq = p['kp_i']*eq + self.xq_i
            else:
                xd_new = self.xd_i + p['ki_i']*ts*ed
                md = p['kp_i']*ed + xd_new
                xq_new = self.xq_i + p['ki_i']*ts*eq
                mq = p['kp_i']*eq + xq_new
            self.xd_i, self.xq_i = xd_new, xq_new
            md, mq = md/udc_ff, mq/udc_ff
            mag = np.hypot(md, mq)
            if mag >= p['u_lim']:
                md, mq = md/mag*p['u_lim'], mq/mag*p['u_lim']
                clip = 1
            m_d, m_q = md, mq
        self.clip_z = clip
        self.m = m_d*np.cos(theta_hat) - m_q*np.sin(theta_hat)
        self.enable_z = enable
        self.log = dict(theta=theta, theta_hat=theta_hat, omega_hat_pu=omega_hat/p['omega_nom'], u_xi=u_xi, u_eta=u_eta, ubeta=ubeta,
                        i_alpha=i_alpha, i_beta=i_beta, i_d=i_d, i_q=i_q, i_d_ref=i_d_ref, i_q_ref=i_q_ref, m_d=m_d, m_q=m_q,
                        clip=clip, udc_ref=udc_ref, e_udc=e_udc, udc_f=self.udc_f, enable=float(enable))
        return (self.m if enable else 0.0), np.array([theta_hat, m_d, m_q])


def default_p(**kw):
    p = dict(ubez=ubez, udc_nom=udc_nom, t_enable=0.1, ts=ts, adc_q=1/2**11, i_noise_std=0.0, omega_nom=w0, ug_nom=ug_nom,
             pll_kp=pll_kp, pll_ki=pll_ki, pll_lim=2.0, pll_tau=pll_tau, obs_l1=obs_l1, obs_l2=obs_l2,
             fht_omega=w0, fht_delta=delta, fht_l1=fht_l1, fht_l2=fht_l2, mode=0, slave=0, i_ref=1.0, t_ref=0.35, i_ref_slew=2.0,
             iq_ref=0.0, udc_ref=1.0, kp_udc=kp_udc, ki_udc=ki_udc, i_lim=i_lim, udc_tau=udc_tau, udc_ma=1, udc_meas_tau=0.0,
             kp_i=kp_rec, ki_i=ki_rec, u_lim=1.0, clip_release=0.001, udc_min=0.25)
    p.update(kw)
    return p


def simulate(p, t_end, grid=None, dc=None, nsub=10, seed=None, u_offset=0.0, t_offset=0.6, i_q_step=None, Lg_=Lg, Rg_=Rg, log_fields=()):
    """Sampled-data simulation. grid(t) -> EMF [pu] (default: nominal sinusoid); dc = None: stiff DC link at udc_nom;
    dc = dict(C=..., p_src(t) [W]): DC link C with a power source, udc from the power balance (mode 1).
    u_offset: DC voltage [pu of ubez] added to the bridge voltage from t_offset (disturbance)."""
    A_, B_, C_ = plant_ss(Lg_, Rg_)
    Ad_, Bd_ = zoh(A_, B_, ts/nsub)
    # gates off (before the enable): the bridge is an open rectifier whose diodes never conduct (udc > the AC peak), i_c = 0:
    # the same network with the converter current frozen at zero (its row of A and B zeroed)
    A_o, B_o = A_.copy(), B_.copy()
    A_o[0, :] = 0.0
    B_o[0, :] = 0.0
    Ad_o, Bd_o = zoh(A_o, B_o, ts/nsub)
    rng = np.random.default_rng(seed) if seed is not None else None
    if grid is None:
        grid = lambda t: ug_nom*np.cos(w0*t)
    ctrl = Ctrl(p)
    n = int(round(t_end/ts))
    x = np.array([0.0, grid(0.0), 0.0])
    udc = 1.0
    m = 0.0
    out = {k: np.zeros(n) for k in ('t', 'i_s', 'u_ac', 'i_g', 'm', 'udc', 'e_g') + tuple(log_fields)}
    for k in range(n):
        t = k*ts
        y = C_ @ x
        out['t'][k] = t
        out['i_s'][k], out['u_ac'][k], out['i_g'][k] = y
        out['udc'][k] = udc
        out['e_g'][k] = grid(t)
        if i_q_step is not None and t >= i_q_step[0]:
            ctrl.p['iq_ref'] = i_q_step[1]
        m, sync = ctrl.step(t, y[0], y[1], udc, rng=rng)
        for kf in log_fields:
            out[kf][k] = ctrl.log[kf]
        out['m'][k] = m
        gates_on = ctrl.enable_z
        for j in range(nsub):
            tj = t + j*ts/nsub
            u_s = m*udc*k_mod + (u_offset if tj >= t_offset else 0.0)
            if dc is not None:
                p_ac = u_s*ubez*x[0]*ibez
                udc += (dc['p_src'](tj) - p_ac)/(dc['C']*udc_nom**2)*ts/nsub
            if gates_on:
                x = Ad_ @ x + Bd_ @ np.array([u_s, grid(tj)])
            else:
                x = Ad_o @ x + Bd_o @ np.array([0.0, grid(tj)])
                x[0] = 0.0
    return out


def fundamental(t, x, w=w0):
    M_ = np.column_stack([np.cos(w*t), np.sin(w*t), np.ones(len(t))])
    c = np.linalg.lstsq(M_, x, rcond=None)[0]
    return np.hypot(c[0], c[1]), np.degrees(np.arctan2(-c[1], c[0])), c[2]


def one_period_mean(x):
    N1 = int(round(1/f0/ts))
    return np.convolve(x, np.ones(N1)/N1, mode='same')


LOGF = ('theta', 'theta_hat', 'omega_hat_pu', 'i_d', 'i_q', 'i_d_ref', 'i_q_ref', 'm_d', 'm_q', 'ubeta', 'u_eta', 'i_alpha', 'i_beta', 'udc_f', 'udc_ref', 'clip')

# ---- 7.1 start-up and reference ramp, recommended gains
pr('  7.1 recommended gains: enable at 0.1 s (bumpless), reference ramp 0 -> 1 pu from 0.35 s with 2 pu/s, i_q step -0.3 pu at 1.0 s')
sim = simulate(default_p(), 1.3, log_fields=LOGF, i_q_step=(1.0, -0.3))
t = sim['t']
i_pk_enable = np.max(np.abs(sim['i_s'][(t > 0.1) & (t < 0.16)]))
pr(f'    peak current in the first 60 ms after the enable: {i_pk_enable:.3f} pu ({i_pk_enable*ibez:.1f} A); DC component 0.1 s after the enable {np.interp(0.2, t, one_period_mean(sim["i_s"])):.4f} pu')
seg = (t > 0.9) & (t < 1.0)
amp, ph, dc_ = fundamental(t[seg], sim['i_s'][seg])
pr(f'    at 1 pu: FHT estimates i_d {np.mean(sim["i_d"][seg]):.4f}, i_q {np.mean(sim["i_q"][seg]):.4f}; true fundamental of i_s {amp:.4f} pu /_ {ph:.2f} deg vs the EMF, DC {dc_:+.5f} pu; '
   f'm_d {np.mean(sim["m_d"][seg]):.4f}, m_q {np.mean(sim["m_q"][seg]):.4f}')
ampu, phu, _ = fundamental(t[seg], sim['u_ac'][seg])
pr(f'    terminal voltage u_ac: {ampu:.4f} pu /_ {phu:.2f} deg (EMF {ug_nom:.4f} pu at 0 deg): the current is in phase with the terminal voltage within {ph - phu:.2f} deg')
seg_q = (t >= 1.0) & (t < 1.3)
iq = sim['i_q'][seg_q]
tq = t[seg_q] - 1.0
err_q = np.abs(iq - (-0.3))
idx_set = np.where(err_q > 0.05*0.3)[0]
t_set_q = tq[idx_set[-1]] if len(idx_set) else 0.0
pr(f'    i_q step -0.3 pu: 5 % settling of the FHT estimate {t_set_q*1e3:.1f} ms, overshoot {(np.min(iq) + 0.3)/0.3*100:.1f} %; '
   f'residual ripple of i_q at steady state {np.std(iq[tq > 0.2])*1e3:.2f} mpu rms')
# envelope settling of the ramp end: i_d after 0.85 s
seg_r = (t >= 0.85) & (t < 1.0)
e_r = np.abs(sim['i_d'][seg_r] - 1.0)
idx_r = np.where(e_r > 0.02)[0]
pr(f'    end of the ramp (0.85 s): i_d within 2 % after {(t[seg_r][idx_r[-1]] - 0.85)*1e3 if len(idx_r) else 0:.1f} ms, overshoot {(np.max(sim["i_d"][seg_r]) - 1)*100:.1f} %')

fig, ax = plt.subplots(3, 1, figsize=(FIGW, 5.6), sharex=True)
ax[0].plot(t, sim['i_s'], lw=0.5, label=r'$i_s$')
ax[0].plot(t, sim['i_d'], label=r'$i_d$ (FHT, dq)'); ax[0].plot(t, sim['i_d_ref'], '--', label=r'$i_d^*$')
ax[0].plot(t, sim['i_q'], label=r'$i_q$'); ax[0].plot(t, sim['i_q_ref'], '--', label=r'$i_q^*$')
ax[0].set_ylabel('[pu]'); ax[0].legend(loc='upper left', ncol=5); ax[0].set_title(r'converter current, references and dq estimates ($k_{p,i} = 0.15$, $k_{i,i} = 20$ s$^{-1}$)')
ax[1].plot(t, sim['u_ac'], lw=0.5, label=r'$u_{ac}$'); ax[1].plot(t, sim['m']*k_mod, lw=0.5, label=r'$m\,u_{dc}/u_{bez}$ (bridge voltage, average)')
ax[1].set_ylabel('[pu]'); ax[1].legend(loc='upper left', ncol=2); ax[1].set_title('terminal and bridge voltage')
ax[2].plot(t, sim['m_d'], label=r'$m_d$'); ax[2].plot(t, sim['m_q'], label=r'$m_q$')
ax[2].plot(t, np.degrees(Ctrl.wrap(sim['theta_hat'] - w0*t))/10, label=r'$(\hat\theta - \omega_0 t)$ [deg/10]')
ax[2].set_ylabel('[pu]'); ax[2].set_xlabel('t [s]'); ax[2].legend(loc='upper left', ncol=3); ax[2].set_title('modulation vector and PLL angle error')
savefig(fig, 'fig_sim_startup.pdf')

fig, ax = plt.subplots(1, 2, figsize=(FIGW, 2.4))
zm = (t > 0.08) & (t < 0.2)
ax[0].plot(t[zm]*1e3, sim['i_s'][zm], lw=0.7, label=r'$i_s$'); ax[0].plot(t[zm]*1e3, sim['i_d'][zm], label=r'$i_d$'); ax[0].plot(t[zm]*1e3, sim['i_q'][zm], label=r'$i_q$')
ax[0].set_xlabel('t [ms]'); ax[0].set_ylabel('[pu]'); ax[0].legend(); ax[0].set_title('enable at 100 ms: bumpless start')
zm = (t > 0.98) & (t < 1.1)
ax[1].plot(t[zm]*1e3, sim['i_s'][zm], lw=0.7, label=r'$i_s$'); ax[1].plot(t[zm]*1e3, sim['i_q'][zm], label=r'$i_q$'); ax[1].plot(t[zm]*1e3, sim['i_q_ref'][zm], '--', label=r'$i_q^*$')
ax[1].plot(t[zm]*1e3, sim['i_d'][zm], label=r'$i_d$')
ax[1].set_xlabel('t [ms]'); ax[1].legend(); ax[1].set_title(r'$i_q^*$ step $0 \to -0.3$ pu at 1 pu of $i_d$')
savefig(fig, 'fig_sim_zoom.pdf')

# ---- 7.2 the three gain sets on the same run
pr('  7.2 the three gain sets on the same run (enable 0.1 s, ramp from 0.35 s)')
sims = {}
for kp, ki, lab in [(kp_i, ki_i, 'lib'), (kp_rec, ki_rec, 'rec'), (kp_old, ki_old, 'old')]:
    sims[lab] = simulate(default_p(kp_i=kp, ki_i=ki), 1.3, log_fields=LOGF)
    s_ = sims[lab]
    dcc = one_period_mean(s_['i_s'])
    seg = (s_['t'] > 1.2)
    amp, ph, dc_ = fundamental(s_['t'][seg], s_['i_s'][seg])
    pr(f'    {lab} {kp}/{ki}: peak |i_s| {np.max(np.abs(s_["i_s"])):.2f} pu, clipped steps after the enable {np.mean(s_["clip"][s_["t"] > 0.1])*100:.0f} %, '
       f'DC component at 0.3 / 0.6 / 0.9 / 1.2 s: {np.interp(0.3, s_["t"], dcc):+.4f} / {np.interp(0.6, s_["t"], dcc):+.4f} / {np.interp(0.9, s_["t"], dcc):+.4f} / {np.interp(1.2, s_["t"], dcc):+.4f} pu; '
       f'fundamental at 1.2 s {amp:.3f} pu')
# growth rate of the DC component with the library gains between 0.4 and 1.0 s
s_ = sims['lib']
dcc = one_period_mean(s_['i_s'])
d1, d2 = np.interp(0.5, s_['t'], dcc), np.interp(0.9, s_['t'], dcc)
if d1*d2 > 0:
    pr(f'    library gains: DC component grows from {d1:+.4f} (0.5 s) to {d2:+.4f} pu (0.9 s): rate {np.log(abs(d2/d1))/0.4:+.1f} 1/s (DC pole of the linear model {dc_pole(kp_i, ki_i):+.1f} 1/s)')
fig, ax = plt.subplots(2, 1, figsize=(FIGW, 4.6), sharex=False)
for lab, col, nm in [('old', 'C3', '0.5 / 45 (bench, old modules)'), ('lib', 'C0', '0.1 / 20 (library default)'), ('rec', 'C2', '0.15 / 20 (recommended)')]:
    s_ = sims[lab]
    ax[0].plot(s_['t'], s_['i_s'], lw=0.5, color=col, alpha=0.8, label=nm)
    ax[1].plot(s_['t'], one_period_mean(s_['i_s']), color=col, label=nm)
ax[0].set_ylabel(r'$i_s$ [pu]'); ax[0].legend(loc='upper left', ncol=3, fontsize=7); ax[0].set_title('converter current with the three gain sets (enable at 0.1 s, reference ramp from 0.35 s)')
ax[0].set_ylim(-4, 4); ax[0].set_xlim(0, 1.3)
ax[1].set_ylabel('one-period average of $i_s$ [pu]'); ax[1].set_xlabel('t [s]'); ax[1].legend(fontsize=7); ax[1].set_title('DC component of the converter current')
ax[1].set_xlim(0, 1.3); ax[1].set_ylim(-0.5, 2.5)
savefig(fig, 'fig_sim_gains.pdf')
fig, ax = plt.subplots(1, 2, figsize=(FIGW, 2.4))
s_ = sims['old']
zm = (s_['t'] > 0.095) & (s_['t'] < 0.14)
ax[0].plot(s_['t'][zm]*1e3, s_['i_s'][zm], lw=0.7, color='C3', label='0.5 / 45')
s2 = sims['rec']
ax[0].plot(s2['t'][zm]*1e3, s2['i_s'][zm], lw=0.7, color='C2', label='0.15 / 20')
ax[0].set_xlabel('t [ms]'); ax[0].set_ylabel(r'$i_s$ [pu]'); ax[0].legend(fontsize=7); ax[0].set_title('after the enable: the 1.4 kHz mode with 0.5 / 45')
zm = (s_['t'] > 0.6) & (s_['t'] < 0.66)
ax[1].plot(s_['t'][zm]*1e3, s_['i_s'][zm], lw=0.7, color='C3', label='0.5 / 45')
ax[1].plot(s2['t'][zm]*1e3, s2['i_s'][zm], lw=0.7, color='C2', label='0.15 / 20')
ax[1].set_xlabel('t [ms]'); ax[1].legend(fontsize=7); ax[1].set_title('during the ramp')
savefig(fig, 'fig_sim_gains_zoom.pdf')

# ---- 7.3 DC voltage disturbance with the recommended gains (and the library gains)
pr('  7.3 DC voltage disturbance 0.01 pu of ubez (7.2 V) in the bridge voltage from 0.6 s, no current reference')
for kp, ki, lab in [(kp_rec, ki_rec, 'rec'), (kp_i, ki_i, 'lib')]:
    sd = simulate(default_p(kp_i=kp, ki_i=ki, i_ref=0.0), 1.6, log_fields=LOGF, u_offset=0.01)
    sims['dc_' + lab] = sd
    dcc = one_period_mean(sd['i_s'])
    L0 = ceq_d(np.array([1.0 + 0j]), kp, ki)[0].real*k_mod/R_loop
    pr(f'    {lab} {kp}/{ki}: DC component at 0.7 / 0.9 / 1.2 / 1.5 s: {np.interp(0.7, sd["t"], dcc):+.4f} / {np.interp(0.9, sd["t"], dcc):+.4f} / {np.interp(1.2, sd["t"], dcc):+.4f} / {np.interp(1.5, sd["t"], dcc):+.4f} pu; '
       f'open loop 0.01/R\' = {0.01/R_loop:.2f} pu, closed loop 0.01/R\' /(1 + L(0)) = {0.01/R_loop/(1 + L0):+.2f} pu (L(0) = {L0:+.2f})')
pr(f'    open-loop L/R time constant (Lc + Lg)/(Rc + Rg) = {(Lc + Lg)/(Rc + Rg)*1e3:.0f} ms')
fig, ax = plt.subplots(1, 2, figsize=(FIGW, 2.4))
for lab, col, nm in [('dc_rec', 'C2', '0.15 / 20'), ('dc_lib', 'C0', '0.1 / 20')]:
    sd = sims[lab]
    ax[0].plot(sd['t'], one_period_mean(sd['i_s']), color=col, label=nm)
    ax[1].plot(sd['t'], sd['m_d'], color=col, label=rf'$m_d$, {nm}'); ax[1].plot(sd['t'], sd['m_q'], '--', color=col, label=rf'$m_q$, {nm}')
ax[0].set_xlim(0.5, 1.6); ax[0].set_xlabel('t [s]'); ax[0].set_ylabel('[pu]'); ax[0].legend(fontsize=7); ax[0].set_title('DC component of $i_s$ (0.01 pu DC step at 0.6 s)')
ax[1].set_xlim(0.5, 1.6); ax[1].set_xlabel('t [s]'); ax[1].legend(fontsize=7, ncol=2); ax[1].set_title('modulation vector $m_d$, $m_q$')
savefig(fig, 'fig_sim_dcoffset.pdf')

# ---- 7.4 PLL: phase jump and frequency step
pr('  7.4 PLL: phase jump of +30 deg at 0.4 s, frequency step +1 Hz at 0.7 s (no current reference)')


def grid_angle(t):
    ph = np.radians(30.0) if t >= 0.4 else 0.0
    dw = 2*np.pi*1.0 if t >= 0.7 else 0.0
    return w0*t + ph + dw*(t - 0.7 if t >= 0.7 else 0.0)


def grid_pll(t):
    return ug_nom*np.cos(grid_angle(t))


sim_pll = simulate(default_p(i_ref=0.0), 1.1, grid=grid_pll, log_fields=LOGF)
t4 = sim_pll['t']
ga = np.array([grid_angle(tt_) for tt_ in t4])
err_th = np.degrees(Ctrl.wrap(sim_pll['theta_hat'] - ga))
err_pll = np.degrees(Ctrl.wrap(sim_pll['theta'] - ga))
seg = (t4 > 0.35) & (t4 < 0.4)
pr(f'    locked (before the jump): angle error of theta_hat {np.mean(err_th[seg]):.3f} deg (theta: {np.mean(err_pll[seg]):.3f} deg); one control step is {np.degrees(w0*ts):.2f} deg')
seg = (t4 > 0.4) & (t4 < 0.7)
e = np.abs(err_th[seg] - np.mean(err_th[(t4 > 0.65) & (t4 < 0.7)]))
i2 = np.where(e > 0.02*30)[0]
pr(f'    phase jump 30 deg: theta_hat error within 2 % (0.6 deg) after {(t4[seg][i2[-1]] - 0.4)*1e3:.0f} ms; peak error {np.max(np.abs(err_th[seg])):.1f} deg')
seg = (t4 > 1.0)
pr(f'    frequency +1 Hz: static angle error {np.mean(err_th[seg]):.2f} deg (type-1 PLL: 2 pi/(kp w_nom) rad per Hz = {np.degrees(2*np.pi/(pll_kp*w0)):.2f} deg), '
   f'omega_hat {np.mean(sim_pll["omega_hat_pu"][seg]):.5f} pu (expected {51/50:.5f}); u_eta ripple at steady state {np.std(sim_pll["u_eta"][seg]):.4f}')
fig, ax = plt.subplots(1, 2, figsize=(FIGW, 2.4))
ax[0].plot(t4, err_pll, lw=0.7, label=r'$\theta - \theta_g$ (PLL)'); ax[0].plot(t4, err_th, label=r'$\hat\theta - \theta_g$ (observer)')
ax[0].set_xlim(0.35, 1.1); ax[0].set_xlabel('t [s]'); ax[0].set_ylabel('[deg]'); ax[0].legend(fontsize=7); ax[0].set_title('angle error: +30 deg at 0.4 s, +1 Hz at 0.7 s')
ax[1].plot(t4, sim_pll['omega_hat_pu'], label=r'$\hat\omega/\omega_{nom}$'); ax[1].plot(t4, sim_pll['u_eta'], lw=0.6, label=r'$u_\eta$ (PLL error signal)')
ax[1].set_xlim(0.35, 1.1); ax[1].set_xlabel('t [s]'); ax[1].legend(fontsize=7); ax[1].set_title('frequency estimate and PLL error')
savefig(fig, 'fig_sim_pll.pdf')

fig, ax = plt.subplots(1, 2, figsize=(FIGW, 2.4))
zm = (t4 > 0.38) & (t4 < 0.5)
ax[0].plot(t4[zm]*1e3, sim_pll['u_ac'][zm]/ug_nom, lw=0.7, label=r'$u_g$ ($\alpha$)'); ax[0].plot(t4[zm]*1e3, sim_pll['ubeta'][zm], label=r'$u_\beta$ (virtual quadrature)')
ax[0].plot(t4[zm]*1e3, np.sin(ga[zm] - w0*ts), 'k:', lw=0.7, label=r'exact quadrature of the acquired $u_g$')
ax[0].set_xlabel('t [ms]'); ax[0].set_ylim(-1.3, 1.9); ax[0].legend(fontsize=6, loc='upper center', ncol=3); ax[0].set_title('virtual quadrature at the phase jump')
ax[1].semilogx(f, 20*np.log10(np.abs(Hb)), label=r'$|H_\beta|$, $k = 0.5$ ($\tau = 2/\omega_0$)')
ax[1].semilogx(f, 20*np.log10(np.abs((w0*1.0)/(s**2 + s*1.0*w0 + w0**2))), '--', label=r'$k = 1$')
ax[1].semilogx(f, 20*np.log10(np.abs((w0*np.sqrt(2))/(s**2 + s*np.sqrt(2)*w0 + w0**2))), ':', label=r'$k = \sqrt{2}$')
ax[1].set_xlabel('f [Hz]'); ax[1].set_ylabel('[dB]'); ax[1].legend(fontsize=7); ax[1].set_title(r'$H_\beta(s) = k\omega_0^2/(s^2 + k\omega_0 s + \omega_0^2)$')
ax[1].set_ylim(-40, 10)
savefig(fig, 'fig_sim_sogi.pdf')

# ---- 7.5 mode 1: DC-link voltage loop
pr('  7.5 mode 1: DC link 3 mF fed by a 20 kW source (40 %) from 0.4 s (0 before), udc_ref 1 pu; moving average vs first-order low pass')
P_STEP = 0.4*pwr_nom
P_SRC = lambda t: P_STEP if t >= 0.4 else 0.0
dc = dict(C=Cdc, p_src=P_SRC)
sim_m1 = simulate(default_p(mode=1, t_enable=0.1), 1.4, dc=dc, log_fields=LOGF)
t5 = sim_m1['t']
seg = (t5 > 0.4) & (t5 < 0.9)
seg9 = t5 > 1.2
pr(f'    MA: udc max {np.max(sim_m1["udc"][seg]):.4f} pu (+{(np.max(sim_m1["udc"][seg]) - 1)*udc_nom:.1f} V) at the 20 kW step, '
   f'i_d_ref final {np.mean(sim_m1["i_d_ref"][seg9]):.4f} pu; udc ripple at steady state {(np.max(sim_m1["udc"][seg9]) - np.min(sim_m1["udc"][seg9]))*udc_nom:.1f} V pk-pk; '
   f'ripple of i_d_ref {np.std(sim_m1["i_d_ref"][seg9])*1e3:.3f} mpu rms')
# settling of udc after the step (2 % of the excursion)
e_u = np.abs(sim_m1['udc'][seg] - 1.0)*udc_nom
exc = np.max(e_u)
idx_u = np.where(one_period_mean(sim_m1['udc'])[seg]*udc_nom - udc_nom > 0.02*exc)[0]
pr(f'    MA: average udc back within 2 % of the excursion {(t5[seg][idx_u[-1]] - 0.4)*1e3 if len(idx_u) else 0:.0f} ms after the step')
sim_lp = simulate(default_p(mode=1, t_enable=0.1, udc_ma=0, udc_meas_tau=tau_lp), 1.4, dc=dc, log_fields=LOGF)
M3 = np.column_stack([np.cos(w0*t5[seg9]), np.sin(w0*t5[seg9]), np.cos(3*w0*t5[seg9]), np.sin(3*w0*t5[seg9]), np.ones(seg9.sum())])
c_lp = np.linalg.lstsq(M3, sim_lp['i_s'][seg9], rcond=None)[0]
c_ma = np.linalg.lstsq(M3, sim_m1['i_s'][seg9], rcond=None)[0]
pr(f'    LP {tau_lp*1e3:.0f} ms: udc max {np.max(sim_lp["udc"][seg]):.4f}; ripple of i_d_ref at steady state {np.std(sim_lp["i_d_ref"][seg9])*1e3:.3f} mpu rms; '
   f'third harmonic of i_s {np.hypot(c_lp[2], c_lp[3])/np.hypot(c_lp[0], c_lp[1])*100:.2f} % (MA: {np.hypot(c_ma[2], c_ma[3])/np.hypot(c_ma[0], c_ma[1])*100:.2f} %), '
   f'DC of i_s {c_lp[4]:+.4f} (MA {c_ma[4]:+.4f}) pu')
# origin of the third harmonic with the moving average: the one-step delay of the DC-link feed-forward on the 100 Hz ripple
sim_ff = simulate(default_p(mode=1, t_enable=0.1, ff_no_delay=1), 1.4, dc=dc, log_fields=LOGF)
c_ff = np.linalg.lstsq(M3, sim_ff['i_s'][seg9], rcond=None)[0]
rip = (np.max(sim_m1['udc'][seg9]) - np.min(sim_m1['udc'][seg9]))/2
pr(f'    MA with the feed-forward on the sample of the same trigger (experiment): third harmonic of i_s {np.hypot(c_ff[2], c_ff[3])/np.hypot(c_ff[0], c_ff[1])*100:.2f} %; '
   f'estimate of the residual: ripple {rip*100:.2f} % x 2 sin(w_ripple ts/2) = {rip*2*np.sin(2*w0*ts/2)*100:.3f} % of the bridge voltage at 150 Hz -> '
   f'{rip*2*np.sin(2*w0*ts/2)*np.mean(sim_m1["m_d"][seg9])*k_mod/(3*w0*(Lcp + Lgp)):.4f} pu of third-harmonic current ({np.hypot(c_ma[2], c_ma[3]):.4f} pu measured)')
fig, ax = plt.subplots(2, 1, figsize=(FIGW, 4.2), sharex=True)
ax[0].plot(t5, sim_m1['udc']*udc_nom, label=r'$u_{dc}$, moving average'); ax[0].plot(t5, sim_lp['udc']*udc_nom, alpha=0.7, label=rf'$u_{{dc}}$, low pass {tau_lp*1e3:.0f} ms')
ax[0].plot(t5, sim_m1['udc_f']*udc_nom, 'k', lw=0.8, label=r'$u_{dc,f}$ (MA)'); ax[0].plot(t5, sim_m1['udc_ref']*udc_nom, 'k--', lw=0.7, label=r'$u_{dc}^*$')
ax[0].set_ylabel('[V]'); ax[0].legend(ncol=4, fontsize=7); ax[0].set_title('DC-link voltage: 20 kW (40 %) source step at 0.4 s, loop enabled at 0.1 s')
ax[1].plot(t5, sim_m1['i_d_ref'], label=r'$i_d^*$, MA'); ax[1].plot(t5, sim_lp['i_d_ref'], alpha=0.7, label=rf'$i_d^*$, LP {tau_lp*1e3:.0f} ms')
ax[1].plot(t5, sim_m1['i_s'], lw=0.4, alpha=0.5, label=r'$i_s$, MA')
ax[1].set_ylabel('[pu]'); ax[1].set_xlabel('t [s]'); ax[1].legend(ncol=3, fontsize=7); ax[1].set_title('current reference of the DC-link loop (negative: current into the bridge)')
savefig(fig, 'fig_sim_dclink.pdf')
fig, ax = plt.subplots(1, 2, figsize=(FIGW, 2.4))
zm = (t5 > 1.3) & (t5 < 1.36)
ax[0].plot(t5[zm]*1e3, sim_m1['udc'][zm]*udc_nom, label='MA'); ax[0].plot(t5[zm]*1e3, sim_lp['udc'][zm]*udc_nom, alpha=0.7, label=f'LP {tau_lp*1e3:.0f} ms')
ax[0].plot(t5[zm]*1e3, sim_m1['udc_f'][zm]*udc_nom, 'k', lw=0.8, label=r'$u_{dc,f}$ (MA)'); ax[0].plot(t5[zm]*1e3, sim_lp['udc_f'][zm]*udc_nom, 'k--', lw=0.8, label=r'$u_{dc,f}$ (LP)')
ax[0].set_xlabel('t [ms]'); ax[0].set_ylabel('[V]'); ax[0].legend(fontsize=7, ncol=2); ax[0].set_title('steady state: $u_{dc}$ and its filtered value')
ax[1].plot(t5[zm]*1e3, sim_m1['i_d_ref'][zm], label=r'$i_d^*$, MA'); ax[1].plot(t5[zm]*1e3, sim_lp['i_d_ref'][zm], alpha=0.7, label=rf'$i_d^*$, LP')
ax[1].set_xlabel('t [ms]'); ax[1].set_ylabel('[pu]'); ax[1].legend(fontsize=7); ax[1].set_title('current reference $i_d^*$')
savefig(fig, 'fig_sim_dclink_zoom.pdf')

# ---- 7.6 three-level modulation of one leg (gate function as coded)
pr('  7.6 modulator figure')


def leg_gates(mk, ramp, dtf):
    G = np.zeros(4, dtype=bool)
    if mk >= 0:
        G[1] = True
        G[0] = ramp < mk - dtf
        G[2] = ramp >= mk + dtf
    else:
        G[2] = True
        G[3] = ramp > 1 + mk + dtf
        G[1] = ramp <= 1 + mk - dtf
    return G


tt = np.linspace(0, 2/fpwm, 4001)
ramp = np.arcsin(np.sin(2*np.pi*fpwm*tt - np.pi/2))/np.pi + 0.5
dtf = 300e-9*fpwm*40            # dead time exaggerated 40 x for visibility
mref = 0.6
G = np.array([leg_gates(mref, r, dtf) for r in ramp])
Gv = np.array([leg_gates(-mref, r, dtf) for r in ramp])
v_leg = np.where(G[:, 0], 0.5, np.where(G[:, 3], -0.5, 0.0))
v_legv = np.where(Gv[:, 0], 0.5, np.where(Gv[:, 3], -0.5, 0.0))
fig, ax = plt.subplots(3, 1, figsize=(FIGW, 4.0), sharex=True)
ax[0].plot(tt*1e6, ramp, 'k', lw=0.8, label='carrier'); ax[0].axhline(mref, color='C0', lw=0.8, ls='--', label=r'$m = 0.6$ (leg u)'); ax[0].axhline(1 + (-mref), color='C3', lw=0.8, ls='--', label=r'$1 + (-m)$ (leg v, reference $-m$)')
for k_ in [0, 0.5, 1.0, 1.5, 2.0]:
    ax[0].axvline(k_/fpwm*1e6, color='gray', lw=0.5, alpha=0.5)
ax[0].legend(fontsize=7, ncol=3); ax[0].set_title('carrier, references and TRGO instants (grey, double update)')
for k_, nm in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    ax[1].step(tt*1e6, G[:, k_]*0.8 + 3 - k_, where='post', lw=0.8)
    ax[1].text(-25, 3 - k_ + 0.3, nm + 'u', fontsize=7)
ax[1].set_yticks([]); ax[1].set_title('gate commands of leg u (dead time exaggerated 40x)')
ax[2].step(tt*1e6, v_leg, where='post', label=r'$v_u - v_{np}$ (leg u)'); ax[2].step(tt*1e6, v_legv, where='post', label=r'$v_v - v_{np}$ (leg v)'); ax[2].step(tt*1e6, v_leg - v_legv, 'k', where='post', lw=0.8, label=r'$u_s$ (bridge)')
ax[2].set_xlabel(r't [$\mu$s]'); ax[2].set_ylabel(r'[$u_{dc}$]'); ax[2].legend(fontsize=7, ncol=3, loc='lower right'); ax[2].set_title('leg voltages to the neutral point and bridge voltage $u_s$')
ax[2].set_ylim(-1.2, 1.2)
savefig(fig, 'fig_modulator.pdf')

di_max = (udc_nom/2)*(1/(2*fpwm))/4/Lc
pr(f'  current ripple: max peak-peak (udc/2) (1/(2 fpwm))/4 / Lc = {di_max:.1f} A ({di_max/ibez*100:.0f} % of ibez)')
pr('=' * 100)
pr('done; figures in', OUT)
