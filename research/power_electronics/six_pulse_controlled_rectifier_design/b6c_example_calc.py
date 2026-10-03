"""Worked example of b6c_sizing_formulas.tex -- six-pulse thyristor bridge with arm fuses, RC snubbers
and DC LC filter. Step numbers S1-S14 and section numbers refer to the document.
Representative thyristor data, not a specific part.
Run:  python b6c_example_calc.py          (example)
      python b6c_example_calc.py table    (also the snubber design table and the pgfplots coordinates)"""
import sys
import numpy as np
pi = np.pi

# ---------------- specification ----------------
Vd, Id, Idmin = 440.0, 1000.0, 100.0        # V, A, A
kol, tol = 1.5, 60.0                         # overload factor, s
V1, dVm, dVp, f = 400.0, 0.10, 0.10, 50.0    # grid V, -tol, +tol, Hz
uk, ur = 0.06, 0.01                          # transformer p.u.
i0 = 0.015                                   # transformer magnetizing current, p.u. (assumed)
amin = np.deg2rad(10.0)
rpp = 0.03                                   # pk-pk ripple, p.u. of Vd
# representative thyristor
VT0, rT = 0.85, 0.35e-3
RthJC, RthCH, Ths, Ta = 0.030, 0.010, 70.0, 40.0
ITSM = 15e3
VRRM = 1600.0                                # voltage class chosen in S5
Qrr_ref, didt_ref = 1.0e-3, 10e6             # recovered charge at Tj,max and the di/dt it is given at
S_soft = 0.3                                 # softness factor tb/ta
dvdt_cr, didt_cr = 500e6, 150e6              # V/s, A/s
w, w6, w12 = 2*pi*f, 12*pi*f, 24*pi*f


# ---------------- RC snubber: normalised snap-off transient (Sec. 5) ----------------
def snap_off(zeta, chi, tmax=20.0, n=4001):
    """Series V_com - 2Lk - Rs - Cs with i(0) = I_RM, v_C(0) = 0 (ideal snap-off).
    x = i*Z0/V_com, tau = w0*t:  x'' + 2 zeta x' + x = 0,  x(0) = chi,  x'(0) = 1 - 2 zeta chi,
    v_T/V_com = 1 - x'.  Returns (V_RM/V_com, max d(v_T/V_com)/dtau)."""
    if abs(zeta - 1.0) < 1e-6:
        zeta += 1e-6
    s = np.roots([1.0, 2*zeta, 1.0]).astype(complex)
    x0, dx0 = chi, 1.0 - 2*zeta*chi
    c2 = (dx0 - s[0]*x0)/(s[1] - s[0])
    c1 = x0 - c2
    tau = np.linspace(0.0, tmax, n)
    e1, e2 = np.exp(s[0]*tau), np.exp(s[1]*tau)
    dx = (c1*s[0]*e1 + c2*s[1]*e2).real
    ddx = (c1*s[0]**2*e1 + c2*s[1]**2*e2).real
    return 1.0 - dx.min(), (-ddx).max()


def zeta_opt(chi, zmax=2.0):
    """Damping that minimises V_RM/V_com for a given chi (coarse grid, then refined)."""
    zc = np.arange(0.20, zmax + 1e-9, 0.05)
    k = int(np.argmin([snap_off(z, chi)[0] for z in zc]))
    zf = np.arange(max(0.20, zc[k] - 0.05), min(zmax, zc[k] + 0.05) + 1e-9, 0.005)
    F = [snap_off(z, chi)[0] for z in zf]
    k = int(np.argmin(F))
    return zf[k], F[k]


def chi_max(rho):
    """Largest chi whose optimally damped peak F_min(chi) does not exceed rho (F_min grows with chi)."""
    lo, hi = 0.0, 5.0
    if zeta_opt(lo)[1] > rho:
        return 0.0
    for _ in range(30):
        mid = 0.5*(lo + hi)
        lo, hi = (mid, hi) if zeta_opt(mid)[1] <= rho else (lo, mid)
    return lo


E12 = (1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2)


def e12_up(x):
    """Smallest E12 value >= x."""
    dec = 10.0**np.floor(np.log10(x))
    return min(m*dec for m in E12 + (10.0,) if m*dec >= x*(1 - 1e-9))


def e12_near(x):
    dec = 10.0**np.floor(np.log10(x))
    return min((m*dec for m in E12 + (10.0,)), key=lambda v: abs(np.log(v/x)))


def recovery(Vcom, Lk):
    didt = Vcom/(2*Lk)
    Qrr = Qrr_ref*np.sqrt(didt/didt_ref)
    IRM = np.sqrt(2*Qrr*didt/(1 + S_soft))
    return didt, Qrr, IRM, Qrr*Vcom/(1 + S_soft)


# ---------------- S2-S3: Vdo iteration (Sec. 2) ----------------
Vdo = Vd/np.cos(amin)
for it in range(50):
    V2 = Vdo*pi/(3*np.sqrt(2))
    I2 = np.sqrt(2/3)*Id
    ST = np.sqrt(3)*V2*I2
    Xk, Rk = uk*V2**2/ST, ur*V2**2/ST
    Lk = Xk/w
    Dx, Dr, Dt = 3/pi*Xk*Id, 2*Rk*Id, 2*(VT0 + rT*Id)
    Vdo_new = (Vd + Dx + Dr + Dt)/np.cos(amin)/(1 - dVm)
    if abs(Vdo_new - Vdo) < 1e-3:
        break
    Vdo = Vdo_new
print(f"S2-S3: {it+1} iterations  Vdo={Vdo:.1f} V  V2={V2:.1f} V  n={V1/V2:.3f}  I2={I2:.0f} A  ST={ST/1e3:.0f} kVA")
print(f"       Xk={Xk*1e3:.2f} mOhm  Lk={Lk*1e6:.1f} uH  Rk={Rk*1e3:.2f} mOhm  Dx={Dx:.1f} V  Dr={Dr:.1f} V  Dt={Dt:.2f} V")

# ---------------- S4: operating points (Sec. 3) ----------------
A6max = 0.0
ops = []
for g, lab in [(1-dVm, 'min'), (1.0, 'nom'), (1+dVp, 'max')]:
    Vdog, V2g = Vdo*g, V2*g
    a = np.arccos((Vd + Dx + Dr + Dt)/Vdog)
    mu = np.arccos(np.cos(a) - np.sqrt(2)*Xk*Id/V2g) - a
    cosphi = (np.cos(a) + np.cos(a+mu))/2
    Q = Vd*Id*np.tan(np.arccos(cosphi))
    A6 = 2*Vdog*np.sqrt(1 + 35*np.sin(a)**2)/35
    A12 = 2*Vdog*np.sqrt(1 + 143*np.sin(a)**2)/143
    didt = np.sqrt(2)*V2g*np.sin(a)/(2*Lk)
    ops.append((lab, V2g, a, mu))
    print(f"S4 grid {lab}: alpha={np.rad2deg(a):.1f} deg  mu={np.rad2deg(mu):.1f} deg  cosphi1={cosphi:.3f}  Q1={Q/1e3:.0f} kvar  A6={A6:.1f} V  A12={A12:.1f} V  di/dt_on={didt/1e6:.1f} A/us")
    if lab == 'max':
        amax, Vdomax, A6max, A12max = a, Vdog, A6, A12

# ---------------- S5-S6: thyristor (Sec. 4) ----------------
Vpk = np.sqrt(2)*V2*(1+dVp)
ITav, ITrms = Id/3, Id/np.sqrt(3)
PT = VT0*ITav + rT*ITrms**2
Pol = VT0*kol*ITav + rT*(kol*ITrms)**2
print(f"S5: Vpk_max={Vpk:.0f} V  VRRM >= {2*Vpk:.0f} (kv=2) ... {2.5*Vpk:.0f} V (kv=2.5)  -> class {VRRM:.0f} V (kv={VRRM/Vpk:.2f})")
print(f"S6: ITav={ITav:.0f} A  ITrms={ITrms:.0f} A  PT={PT:.0f} W  overload {kol*ITav:.0f}/{kol*ITrms:.0f} A  Pol={Pol:.0f} W")
print(f"    Tj={Ths + PT*(RthJC+RthCH):.1f} C  Tj_ol={Ths + Pol*(RthJC+RthCH):.1f} C (Ths const)  RthHA <= {(125-15-Ta)/PT - RthJC - RthCH:.3f} K/W")

# ---------------- S7: dynamic data (Sec. 4) ----------------
print(f"S7: Qrr={Qrr_ref*1e6:.0f} uC at {didt_ref/1e6:.0f} A/us, S={S_soft}, (dv/dt)cr={dvdt_cr/1e6:.0f} V/us, (di/dt)cr={didt_cr/1e6:.0f} A/us")

# ---------------- S8: RC snubber (Sec. 5) ----------------
k_rep, k_dv = 0.55, 2.0


def snubber_design(VRRM_, Vcom):
    didt, Qrr, IRM, EL = recovery(Vcom, Lk)
    rho = k_rep*VRRM_/Vcom
    cm = chi_max(rho)
    Cmin = 2*Lk*(IRM/(cm*Vcom))**2
    Cs = e12_up(Cmin)
    Z0 = np.sqrt(2*Lk/Cs)
    chi = IRM*Z0/Vcom
    zo, _ = zeta_opt(chi)
    Rs = e12_near(2*zo*Z0)
    zeta = Rs/(2*Z0)
    F, _ = snap_off(zeta, chi)
    return dict(didt=didt, Qrr=Qrr, IRM=IRM, EL=EL, rho=rho, chimax=cm, Cmin=Cmin, Cs=Cs, Z0=Z0, chi=chi,
                zopt=zo, Rs=Rs, zeta=zeta, F=F, VRM=F*Vcom)


def snubber_loss(Cs, V2pk, a, mu, EL):
    """P_Rs: turn-on discharge, turn-off step, three commutations of the other arms (two half-steps each),
    plus the recovery energy. Each step dV through Rs dissipates Cs*dV^2/2."""
    return f*(7/8*Cs*V2pk**2*(np.sin(a)**2 + np.sin(a+mu)**2) + EL)


Vcom = Vpk                                      # alpha + mu = 90 deg (soft start, current limit)
sd = snubber_design(VRRM, Vcom)
Cs, Rs, Z0, zeta = sd['Cs'], sd['Rs'], sd['Z0'], sd['zeta']
w0 = 1/np.sqrt(2*Lk*Cs)
dVC = k_rep*VRRM - Vcom
print(f"S8: design point alpha+mu=90 deg: Vcom={Vcom:.0f} V  (di/dt)off={sd['didt']/1e6:.2f} A/us  Qrr={sd['Qrr']*1e6:.0f} uC"
      f"  IRM={sd['IRM']:.0f} A  EL={sd['EL']:.2f} J")
print(f"    VRM,adm={k_rep*VRRM:.0f} V  rho={sd['rho']:.3f}  chi_max={sd['chimax']:.2f}  Cs >= {sd['Cmin']*1e6:.2f} uF"
      f"  (energy bound 2Lk(IRM/dVC)^2 = {2*Lk*(sd['IRM']/dVC)**2*1e6:.1f} uF with dVC={dVC:.0f} V)")
print(f"    Cs={Cs*1e6:.2f} uF  Z0={Z0:.2f} Ohm  f0={w0/2/pi/1e3:.1f} kHz  chi={sd['chi']:.2f}  zeta_opt={sd['zopt']:.2f}"
      f"  Rs={Rs:.1f} Ohm  zeta={zeta:.2f}  VRM={sd['VRM']:.0f} V ({sd['VRM']/VRRM:.2f} VRRM)  Rs*IRM={Rs*sd['IRM']:.0f} V")
_, g0 = snap_off(zeta, 0.0)
print(f"    dv/dt (forward step Vpk, chi=0) = {g0*w0*Vpk/1e6:.0f} V/us  (Rs*Vpk/2Lk = {Rs*Vpk/(2*Lk)/1e6:.0f} V/us)"
      f"  limit (dv/dt)cr/k_dv = {dvdt_cr/k_dv/1e6:.0f} V/us  -> Rs <= {2*Lk*dvdt_cr/(k_dv*Vpk):.0f} Ohm")
print(f"    turn-on discharge Idis={Vpk/Rs:.0f} A  Edis={0.5*Cs*Vpk**2:.2f} J")
for lab, V2g, a, mu in ops:
    Vc = np.sqrt(2)*V2g*np.sin(a+mu)
    didt, Qrr, IRM, EL = recovery(Vc, Lk)
    F, _ = snap_off(zeta, IRM*Z0/Vc)
    P = snubber_loss(Cs, np.sqrt(2)*V2g, a, mu, EL)
    print(f"    grid {lab}: alpha+mu={np.rad2deg(a+mu):.1f} deg  Vcom={Vc:.0f} V  IRM={IRM:.0f} A  chi={IRM*Z0/Vc:.2f}"
          f"  VRM={F*Vc:.0f} V  P_Rs={P:.0f} W")
P90 = snubber_loss(Cs, Vpk, pi/2, 0.0, sd['EL'])
Pmax = snubber_loss(Cs, Vpk, ops[2][2], ops[2][3], recovery(Vpk*np.sin(ops[2][2]+ops[2][3]), Lk)[3])
print(f"    P_Rs at alpha=90 deg: {P90:.0f} W ; steady state (alpha_max): {Pmax:.0f} W -> I_Cs,rms={np.sqrt(Pmax/Rs):.2f} A"
      f" ({np.sqrt(P90/Rs):.2f} A at 90 deg)  pulse energy {max(0.5*Cs*Vpk**2, sd['EL'] + 0.5*Cs*Vcom**2):.2f} J"
      f"  Ipk={max(sd['IRM'], Vpk/Rs):.0f} A  six snubbers {6*Pmax:.0f} W")
for V_, lab in [(1600.0, 'class 1600 V'), (1800.0, 'class 1800 V')]:
    d = snubber_design(V_, Vcom)
    print(f"    trade-off {lab}: VRM,adm={k_rep*V_:.0f} V  Cs={d['Cs']*1e6:.2f} uF  Rs={d['Rs']:.1f} Ohm  VRM={d['VRM']:.0f} V"
          f"  P_Rs(alpha_max)={snubber_loss(d['Cs'], Vpk, ops[2][2], ops[2][3], recovery(Vpk*np.sin(ops[2][2]+ops[2][3]), Lk)[3]):.0f} W"
          f"  P_Rs(90)={snubber_loss(d['Cs'], Vpk, pi/2, 0.0, d['EL']):.0f} W")
# bridge level: magnetizing energy at primary switch-off, upper bound (all energy into the capacitors,
# worst distribution among the phases): Vmax^2 = Vpk^2 + 4 Wm/CY
Wm = i0*ST/(2*w)
CY_arm = 2*Cs
Vmax_arm = np.sqrt(Vpk**2 + 4*Wm/CY_arm)
Vmax_ac = 0.75*VRRM
CY_req = 4*Wm/(Vmax_ac**2 - Vpk**2)
print(f"    bridge level: Wm={Wm:.1f} J  arm snubbers alone (CY=2Cs={CY_arm*1e6:.1f} uF) -> {Vmax_arm:.0f} V ;"
      f" CY >= {CY_req*1e6:.0f} uF per phase for {Vmax_ac:.0f} V (Q_C={V2**2*w*CY_req/1e3:.1f} kvar) or MOV >= {Wm:.0f} J")

# ---------------- S9: fault currents (Sec. 6) ----------------
V2max = V2*(1+dVp)
Zk = np.hypot(Rk, Xk)
Ik3, Ik2 = V2max/(np.sqrt(3)*Zk), V2max/(2*Zk)
kappa = 1.02 + 0.98*np.exp(-3*Rk/Xk)
print(f"S9: Zk={Zk*1e3:.2f} mOhm  X/R={Xk/Rk:.1f}  Ik3''={Ik3/1e3:.1f} kA  Ik2''={Ik2/1e3:.1f} kA  kappa={kappa:.2f}  ip={kappa*np.sqrt(2)*Ik2/1e3:.1f} kA")

# ---------------- S10-S11: fuse (Sec. 6) ----------------
I2tT = ITSM**2*0.01/2
print(f"S10: IFn >= {1.3*ITrms:.0f} A  UFn >= {V2max:.0f} V  Icn >= {Ik3/1e3:.0f} kA")
print(f"S11: I2t_T={I2tT:.3e} A2s -> fuse I2t_tot(U2={V2max:.0f} V, {Ik2/1e3:.1f} kA) <= {0.8*I2tT:.3e} A2s ; t_m({kol*ITrms:.0f} A) >= {3*tol/60:.0f}-{5*tol/60:.0f} min")

# ---------------- S12: filter (Sec. 7) ----------------
dv = rpp*Vd/2
LC = (A6max/dv + 1)/w6**2
Lmin = A6max/(w6*Idmin)
print(f"S12: alpha_max={np.rad2deg(amax):.1f} deg  A6={A6max:.1f} V  A12={A12max:.1f} V  target amp={dv:.1f} V  LC >= {LC:.3e} s2 (f0 <= {1/(2*pi*np.sqrt(LC)):.1f} Hz)  Lmin={Lmin*1e3:.2f} mH")
L, C = 1.0e-3, 5.6e-3
R = Vd/Id
H = lambda wx, R: 1/abs(1 - wx**2*L*C + 1j*wx*L/R)
f0 = 1/(2*pi*np.sqrt(L*C))
print(f"     L={L*1e3} mH  C={C*1e3} mF  f0={f0:.1f} Hz  |H(6f)|={H(w6,R):.4f} -> ripple pp={2*A6max*H(w6,R):.1f} V ({2*A6max*H(w6,R)/Vd*100:.2f} %)  12th amp={A12max*H(w12,R):.2f} V")
zeta_f = lambda R: 0.5/R*np.sqrt(L/C)
print(f"     zeta full={zeta_f(R):.3f}  zeta 10%={zeta_f(10*R):.3f}  |H(f)|={H(w,R):.2f} ({H(w,10*R):.2f} @10%)  |H(2f)|={H(2*w,R):.2f} ({H(2*w,10*R):.2f} @10%)")
for n in (3, 4, 5):
    Rd = np.sqrt(L/C)*np.sqrt((2+n)*(4+3*n)/(2*n**2*(4+n)))
    print(f"     damping n=Cd/C={n}: Rd_opt={Rd*1e3:.0f} mOhm  |H|max={np.sqrt(2*(2+n)/n):.2f}")
# ---------------- S13: ratings (Sec. 7) ----------------
ILamp = A6max/abs(w6*L - 1/(w6*C))
print(f"S13: dIL={ILamp:.0f} A  ICrms~{ILamp/np.sqrt(2):.0f} A  VC >= {1.1*Vdomax:.0f} V  hard start Ihat={Vdomax*np.sqrt(C/L)+Id:.0f} A  load rejection dv={Id*np.sqrt(L/C):.0f} V")
print(f"     DC-side short di/dt={np.sqrt(2)*V2max/(2*Lk+L)/1e6:.2f} A/us  device fault di/dt={np.sqrt(2)*V2max/(2*Lk)/1e6:.2f} A/us")
# ---------------- S14: grid (Sec. 8) ----------------
I1 = np.sqrt(6)/pi*Id
print(f"S14: I1={I1:.0f} A  I5={I1/5:.0f}  I7={I1/7:.0f}  I11={I1/11:.0f} A  THDi={np.sqrt((I2/I1)**2-1)*100:.1f} %")

# ---------------- snubber design table and chart (Sec. 5) ----------------
if len(sys.argv) > 1 and sys.argv[1] == 'table':
    chis = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
    print("\nV_RM/V_com = F(zeta, chi)")
    print("zeta  " + " ".join(f"{c:6.2f}" for c in chis))
    for z in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2):
        print(f"{z:4.2f}  " + " ".join(f"{snap_off(z, c)[0]:6.3f}" for c in chis))
    print("chi    " + " ".join(f"{c:6.2f}" for c in chis))
    print("z_opt  " + " ".join(f"{zeta_opt(c)[0]:6.2f}" for c in chis) + "   (chi < 0.5: no interior optimum)")
    print("F_min  " + " ".join(f"{zeta_opt(c)[1]:6.3f}" for c in chis))
    print("g(zeta) = max dv/dt/(w0 dV), chi = 0: " +
          " ".join(f"{z}:{snap_off(z, 0.0)[1]:.2f}" for z in (0.3, 0.4, 0.5, 0.6, 0.7, 1.0)))
    print("\npgfplots coordinates (zeta, F):")
    for c in (0.0, 0.5, 1.0, 1.5, 2.0):
        print(f"chi={c}: " + " ".join(f"({z:.2f},{snap_off(z, c)[0]:.3f})" for z in np.arange(0.2, 1.501, 0.05)))
    print("optimum locus: " + " ".join(f"({zeta_opt(c)[0]:.2f},{zeta_opt(c)[1]:.3f})" for c in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)))
