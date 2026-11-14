"""Single six-pulse thyristor bridge, 400/17 V transformer, 16.4 V / 1100 A output.
Line reactor + tuned traps 5-7-11 on the 400 V side, DC LC filter. Tables -> ./tables/"""
import numpy as np, os
os.makedirs("tables", exist_ok=True)

F = 50.0; W = 2*np.pi*F
V_NOM = 400.0; V_MIN = 360.0; V_MAX = 440.0
S_N = 1.0e6; UK_PCC = 0.06; SSC = S_N/UK_PCC; X_S = V_NOM**2/SSC

V_OUT = 16.4; ID_R = 1100.0; I_CASE = 1050.041
E_X = 0.06; E_R = 0.027; V_T = 1.3; A_MIN = 10.0   # e_r: relative resistive drop (winding resistance)

# ---- transformer secondary voltage ----
VD0_MIN = (V_OUT + 2*V_T)/(np.cos(np.radians(A_MIN)) - E_X/2 - E_R)   # at 360 V primary, rated current
E2_360 = VD0_MIN/(3*np.sqrt(2)/np.pi)
E2_NOM = 17.0                                                   # chosen rated secondary at 400 V
RATIO = V_NOM/E2_NOM
I2 = np.sqrt(2/3)*ID_R; S2 = np.sqrt(3)*E2_NOM*I2
I1_NOM = np.sqrt(6)/np.pi*ID_R/RATIO                              # fundamental at 400 V (approx)

def vd0(vline): return 3*np.sqrt(2)/np.pi*vline/RATIO
def dx(vline, idc): return vd0(vline)*E_X/2*idc/ID_R
def dr(vline, idc): return vd0(vline)*E_R*idc/ID_R
def alpha_for(vline, idc, vout=V_OUT):
    c = (vout + 2*V_T + dx(vline,idc) + dr(vline,idc))/vd0(vline)
    return np.degrees(np.arccos(c))
def overlap(vline, alpha_deg, idc):
    a = np.radians(alpha_deg)
    return np.degrees(np.arccos(np.clip(np.cos(a) - 2*dx(vline,idc)/vd0(vline),-1,1))) - alpha_deg
def harmonics(alpha_deg, u_deg, hmax=49):
    a, u = np.radians(alpha_deg), np.radians(u_deg); out = {}
    for h in range(5, hmax+1):
        if h % 6 not in (1,5): continue
        A = np.sin((h-1)*u/2)/(h-1); B = np.sin((h+1)*u/2)/(h+1)
        out[h] = (1/h)*np.sqrt(A*A+B*B-2*A*B*np.cos(2*a+u))/(np.cos(a)-np.cos(a+u))
    return out

# ---- AC filter: series reactor on the 400 V side + traps 5, 7, 11 ----
S_CONV = vd0(V_NOM)*ID_R          # ~ S1 at 400 V
X_SER_PU = 0.10
X_SER = X_SER_PU*V_NOM**2/S_CONV; L_SER = X_SER/W
TRAPS = [dict(name="5th", q=5.5e3, h=4.85, Q=40),
         dict(name="7th", q=3.0e3, h=6.8,  Q=40),
         dict(name="11th", q=2.5e3, h=10.7, Q=40)]
def params(t):
    xc = V_NOM**2/t["q"]; C = 1/(W*xc); L = xc/(t["h"]**2*W); R = t["h"]*W*L/t["Q"]
    return C, L, R
def z_branch(t, h):
    C, L, R = params(t); return R + 1j*h*W*L + 1/(1j*h*W*C)
def z_filter(h): return 1/sum(1/z_branch(t,h) for t in TRAPS)
def share(h):
    zs = 1j*h*(X_S + X_SER); zf = z_filter(h); return abs(zf/(zf+zs))
QF = sum(t["q"] for t in TRAPS)

# ---- DC filter ----
L_DC = 0.3e-3; C_DC = 5.0e-3
def ripple(vline, alpha_deg, idc):
    v6 = vd0(vline)*np.sqrt(2)/35*np.sqrt(1+35*np.sin(np.radians(alpha_deg))**2)
    R = V_OUT/idc; w6 = 6*W
    H = 1/(1 - w6**2*L_DC*C_DC + 1j*w6*L_DC/R)
    v6o = v6*H; i6 = (v6 - v6o)/(1j*w6*L_DC)
    return v6, abs(v6o), abs(i6)

# ---- operating points ----
rows = []
for vline in (V_MIN, V_NOM, V_MAX):
    for idc in (I_CASE, ID_R):
        a = alpha_for(vline, idc); u = overlap(vline, a, idc)
        cphi = (np.cos(np.radians(a))+np.cos(np.radians(a+u)))/2
        s1 = vd0(vline)*idc; P = V_OUT*idc; Q = s1*np.sqrt(1-cphi**2)
        i1 = np.sqrt(6)/np.pi*idc*(vline/RATIO)/vline * (vline/V_NOM)  # A at the line voltage, referred: keep simple -> at primary
        i1 = np.sqrt(6)/np.pi*idc/RATIO
        hs = harmonics(a,u)
        thd_c = np.sqrt(sum(r*r for r in hs.values()))
        thd_p = np.sqrt(sum((r*share(h))**2 for h,r in hs.items()))
        v6, v6o, i6 = ripple(vline, a, idc)
        rows.append((vline, idc, vd0(vline), a, u, P/1e3, Q/1e3, P/np.hypot(P,Q), i1, 100*thd_c, 100*thd_p, v6, v6o, 100*2*np.sqrt(2)*v6o/V_OUT, 100*2*np.sqrt(2)*i6/idc, hs))

with open("tables/op.tex","w") as fo:
    fo.write("\\begin{tabular}{rrrrrrrrrrr}\n\\toprule\n$V_{line}$ & $I_d$ & $V_{d0}$ & $\\alpha$ & $u$ & $P$ & $Q$ & PF & $I_1$ & THD$_{conv}$ & THD$_{PCC}$\\\\\n(V) & (A) & (V) & (deg) & (deg) & (kW) & (kvar) & & (A) & (\\%) & (\\%)\\\\\n\\midrule\n")
    for r in rows:
        fo.write(f"{r[0]:.0f} & {r[1]:.0f} & {r[2]:.1f} & {r[3]:.1f} & {r[4]:.1f} & {r[5]:.1f} & {r[6]:.1f} & {r[7]:.2f} & {r[8]:.1f} & {r[9]:.1f} & {r[10]:.1f}\\\\\n")
    fo.write("\\bottomrule\n\\end{tabular}\n")

with open("tables/dc.tex","w") as fo:
    fo.write("\\begin{tabular}{rrrrrrr}\n\\toprule\n$V_{line}$ & $I_d$ & $\\alpha$ & $V_6$ bridge & $V_6$ load & $\\Delta V_{pp}/V_{out}$ & $\\Delta I_{pp}/I_d$ reactor\\\\\n(V) & (A) & (deg) & (V rms) & (V rms) & (\\%) & (\\%)\\\\\n\\midrule\n")
    for r in rows:
        fo.write(f"{r[0]:.0f} & {r[1]:.0f} & {r[3]:.1f} & {r[11]:.2f} & {r[12]:.3f} & {r[13]:.2f} & {r[14]:.1f}\\\\\n")
    fo.write("\\bottomrule\n\\end{tabular}\n")

with open("tables/traps.tex","w") as fo:
    fo.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\nBranch & $Q_C$ (kvar) & $h_t$ & $f_t$ (Hz) & $C$ ($\\mu$F) & $L$ (mH) & $R$ (m$\\Omega$) & $I_{C1}$ (A)\\\\\n\\midrule\n")
    for t in TRAPS:
        C,L,R = params(t)
        fo.write(f"{t['name']} & {t['q']/1e3:.1f} & {t['h']:.2f} & {t['h']*F:.0f} & {C*1e6:.0f} & {L*1e3:.2f} & {R*1e3:.0f} & {t['q']/(np.sqrt(3)*V_NOM):.1f}\\\\\n")
    fo.write("\\bottomrule\n\\end{tabular}\n")

HH = [5,7,11,13,17,19,23,25,35,37]
with open("tables/share.tex","w") as fo:
    fo.write("\\begin{tabular}{l"+"r"*len(HH)+"}\n\\toprule\n$h$ & "+" & ".join(map(str,HH))+"\\\\\n\\midrule\n")
    fo.write("$|Z_F|$ ($\\Omega$) & "+" & ".join(f"{abs(z_filter(h)):.2f}" for h in HH)+"\\\\\n")
    fo.write("$h(X_S+X_s)$ ($\\Omega$) & "+" & ".join(f"{h*(X_S+X_SER):.2f}" for h in HH)+"\\\\\n")
    fo.write("$I_{h,PCC}/I_{h,conv}$ & "+" & ".join(f"{share(h):.3f}" for h in HH)+"\\\\\n\\bottomrule\n\\end{tabular}\n")

worst = max(rows, key=lambda r: r[10])
with open("tables/spectrum.tex","w") as fo:
    hh = [5,7,11,13,17,19,23,25]
    fo.write("\\begin{tabular}{l"+"r"*len(hh)+"}\n\\toprule\n$h$ & "+" & ".join(map(str,hh))+"\\\\\n\\midrule\n")
    fo.write("$I_h/I_1$ converter (\\%) & "+" & ".join(f"{100*worst[15][h]:.1f}" for h in hh)+"\\\\\n")
    fo.write("$I_h/I_1$ at PCC (\\%) & "+" & ".join(f"{100*worst[15][h]*share(h):.2f}" for h in hh)+"\\\\\n")
    fo.write("$I_h$ at PCC (A) & "+" & ".join(f"{worst[8]*worst[15][h]*share(h):.2f}" for h in hh)+"\\\\\n\\bottomrule\n\\end{tabular}\n")

# parallel resonance
hh = np.arange(1.5, 12, 0.02); zp = []
for h in hh:
    zs = 1j*h*(X_S+X_SER); zf = z_filter(h); zp.append(abs(zs*zf/(zs+zf)))
zp = np.array(zp)
for lo,hi in [(1.5,4.5),(5.2,6.5),(7.2,10.4)]:
    m = (hh>=lo)&(hh<=hi); i = np.argmax(zp[m]); print("peak h=%.2f |Z|=%.1f ohm" % (hh[m][i], zp[m][i]))
print("Z at h=3: %.2f ohm" % zp[np.argmin(abs(hh-3))])

print("VD0_MIN=%.2f E2_360=%.2f ratio=%.2f E2_NOM=%.1f I2=%.0f S2=%.1f kVA I1=%.1f A" % (VD0_MIN, E2_360, RATIO, E2_NOM, I2, S2/1e3, I1_NOM))
print("S_CONV=%.1f kVA X_SER=%.3f ohm L_SER=%.2f mH  QF=%.1f kvar  bus rise no-load %.2f %%" % (S_CONV/1e3, X_SER, L_SER*1e3, QF/1e3, 100*QF*X_SER/V_NOM**2))
print("shares", {h: round(share(h),3) for h in HH})
for r in rows: print("%.0f V %.0f A: Vd0=%.1f a=%.1f u=%.1f Q=%.1f thd=%.1f/%.1f v6=%.2f v6o=%.3f rip=%.2f%% i6=%.1f%%" % (r[0],r[1],r[2],r[3],r[4],r[6],r[9],r[10],r[11],r[12],r[13],r[14]))
print("f_r dc = %.0f Hz, E_L = %.0f J, zeta = %.2f" % (1/(2*np.pi*np.sqrt(L_DC*C_DC)), 0.5*L_DC*ID_R**2, 0.5/(V_OUT/ID_R)*np.sqrt(L_DC/C_DC)))
# fault current at secondary and thyristor data
print("I2 sc ~ %.0f A (e_x + reactor)" % (I2/np.hypot(E_X + X_SER_PU*S2/S_CONV, E_R)))
# ---- leakage and resistance estimates ----
ZB2 = E2_NOM**2/S2; XK = E_X*ZB2; LK = XK/W
print("Zb2=%.2f mOhm Xk=%.3f mOhm Lk=%.2f uH (sec, star eq.)  -> primary: %.3f ohm, %.2f mH" % (ZB2*1e3, XK*1e3, LK*1e6, XK*RATIO**2, LK*RATIO**2*1e3))
# geometry-based resistance: V/turn ~ 4.44 f B Afe
B=1.5; AFE=60e-4; VPT=4.44*F*B*AFE
N2 = (E2_NOM/np.sqrt(3))/VPT; N1 = V_NOM/VPT   # star secondary, delta primary
LM=0.5; J=2.5e6; RHO20=1.72e-8; K75=1.216
A2 = I2/J; A1 = (I1_NOM/np.sqrt(3))/J
R2 = RHO20*N2*LM/A2*K75; R1d = RHO20*N1*LM/A1*K75; R1y = R1d/3; R1s = R1y/RATIO**2
RK = R2 + R1s; PCU = 3*I2**2*RK; ER = np.sqrt(3)*RK*I2/E2_NOM
print("V/turn=%.2f N2=%.1f N1=%.0f A2=%.0f mm2 A1=%.1f mm2" % (VPT, N2, N1, A2*1e6, A1*1e6))
print("R2=%.3f mOhm  R1(delta)=%.3f ohm  R1(star eq)=%.3f ohm  R1->sec=%.3f mOhm  Rk=%.3f mOhm  Pcu=%.0f W  e_r=%.2f %%  X/R=%.2f" % (R2*1e3, R1d, R1y, R1s*1e3, RK*1e3, PCU, 100*ER, XK/RK))
print("dc drop 2 Rk Id = %.2f V ; d_r = %.3f Vd0 at 400 V -> %.2f V" % (2*RK*ID_R, E_R, E_R*vd0(V_NOM)))
print("alpha 360/400/440 at 1100 A:", [round(alpha_for(v,ID_R),1) for v in (360,400,440)])
print("Vd0 400 = %.2f, S2=%.1f kVA I1=%.1f" % (vd0(400), S2/1e3, I1_NOM))

print("thyristor: Iavg=%.0f Irms=%.0f Pcond=%.0f W" % (ID_R/3, ID_R/np.sqrt(3), 0.9*ID_R/3 + 0.35e-3*(ID_R/np.sqrt(3))**2))
# background distortion current into 5th trap: 3% V5
print("I5 from 3%% grid V5: %.1f A" % (0.03*V_NOM/np.sqrt(3)/abs(z_branch(TRAPS[0],5) + 1j*5*(X_S+X_SER))))
