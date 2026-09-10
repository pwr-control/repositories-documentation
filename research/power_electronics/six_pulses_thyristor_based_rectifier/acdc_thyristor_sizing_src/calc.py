"""Sizing calculations for the 400 V / 1 MW thyristor AC/DC chain.
Produces LaTeX table fragments in ./tables/."""
import numpy as np, os
os.makedirs("tables", exist_ok=True)

# ---------------- data ----------------
V_LINE = 400.0          # V, PCC
S_N    = 1.0e6          # VA, PCC rating (Pn)
F      = 50.0
W      = 2*np.pi*F
UK_PCC = 0.06           # short-circuit impedance of the PCC feeding transformer
SSC    = S_N/UK_PCC     # ~16.7 MVA
X_S    = V_LINE**2/SSC  # ohm, network reactance at 400 V

ID_R   = 1100.0         # A, rated dc current
V_UNIT = 16.4           # V, load voltage quantum
CASES  = [196.8,196.8,196.8,147.6,196.8,164,131.2,98.4,98.4,49.2,98.4,82,65.6,65.6,32.8,16.4]
I_CASE = 1050.041
REF    = [(20.0,1100.0),(200.0,1100.0)]

# design margins
K_LINE = 1.05   # line under-voltage (primary +-2x2.5 % off-load taps compensate the rest)
E_X    = 0.06   # relative commutation reactance of each secondary (transformer + reactor)
V_T    = 1.3    # V, thyristor on-state drop (each, two in series)
R_DC   = 0.4e-3 # ohm, dc-side resistive drops (reactor, busbars, bridge)
A_MIN  = 10.0   # deg, minimum firing angle of the 'full' bridges

WEIGHTS = [1,2,4,8]                     # binary-weighted bridges
# no-load voltage needed to deliver the step at low line, alpha_min and rated current:
# Vd0 = (V_step + 2 V_T) / (cos(a_min) - e_x/2) / (1/K_LINE)
K_HEAD  = K_LINE/(np.cos(np.radians(A_MIN)) - E_X/2)
VD0     = [(w*V_UNIT + 2*V_T)*K_HEAD for w in WEIGHTS]   # ideal no-load dc voltage of each bridge
ELL     = [v/(3*np.sqrt(2)/np.pi) for v in VD0]  # secondary line-to-line voltage

def dx_drop(vd0, idc):
    """commutation voltage drop, d_x = e_x/2 * Id/Id_r (IEC 60146-1-1)"""
    return vd0*E_X/2*idc/ID_R

def bridge_voltage(vd0, alpha_deg, idc):
    return vd0*np.cos(np.radians(alpha_deg)) - dx_drop(vd0,idc) - 2*V_T

def overlap(vd0, alpha_deg, idc):
    a = np.radians(alpha_deg)
    c = np.cos(a) - 2*dx_drop(vd0,idc)/vd0     # cos a - cos(a+u) = 2 d_x/Vd0 ... (d_x = X_c Id 3/pi -> 2dx/Vd0 = sqrt2 Xc Id/E)
    return np.degrees(np.arccos(np.clip(c,-1,1))) - alpha_deg

def harmonics(alpha_deg, u_deg, hmax=49):
    """I_h/I_1(ideal) for a 6-pulse bridge with overlap (IEC 60146-1-1 / Kimbark)"""
    a, u = np.radians(alpha_deg), np.radians(u_deg)
    out = {}
    for h in range(5, hmax+1):
        if h % 6 not in (1,5): continue
        A = np.sin((h-1)*u/2)/(h-1); B = np.sin((h+1)*u/2)/(h+1)
        num = np.sqrt(A*A + B*B - 2*A*B*np.cos(2*a+u))
        den = np.cos(a) - np.cos(a+u)
        out[h] = (1/h)*num/den if den > 1e-9 else 1/h
    return out

def dispatch(v_req, idc):
    """brute force over the 15 bridge subsets; in each subset the bridges are kept at
    A_MIN and the excess voltage is shed by the smallest bridge(s) (sequential control);
    the subset with the lowest reactive power is returned as list of (weight, vd0, alpha)"""
    best = None
    for n in range(1,16):
        sel = [i for i in range(4) if n & (1 << i)]
        active = [(WEIGHTS[i], VD0[i], A_MIN) for i in sel]
        v_tot = sum(bridge_voltage(vd0, A_MIN, idc) for _,vd0,_ in active)
        if v_tot < v_req: continue
        excess = v_tot - v_req
        res = []
        for w, vd0, a in active:                          # smallest first
            if excess <= 1e-9:
                res.append((w,vd0,a)); continue
            v_full = bridge_voltage(vd0, A_MIN, idc)
            v_min  = bridge_voltage(vd0, 90.0, idc)       # alpha limited to 90 deg
            if v_full - excess >= v_min:
                cosa = (v_full - excess + dx_drop(vd0,idc) + 2*V_T)/vd0
                res.append((w,vd0,np.degrees(np.arccos(cosa)))); excess = 0
            else:
                res.append((w,vd0,90.0)); excess -= (v_full - v_min)
        q = sum(vd0*idc*np.sin(np.radians(a)) for _,vd0,a in res)
        if best is None or q < best[0]: best = (q, res)
    return best[1]

def operating_point(v_req, idc):
    act = dispatch(v_req, idc)
    P = v_req*idc
    Qtot = 0; S1 = 0; I1p = 0
    hsum = {}
    for w, vd0, a in act:
        u = overlap(vd0, a, idc)
        # displacement angle with overlap: cos(phi) ~ (cos a + cos(a+u))/2
        cphi = (np.cos(np.radians(a)) + np.cos(np.radians(a+u)))/2
        s1 = vd0*idc                              # fundamental apparent power
        i1 = np.sqrt(6)/np.pi*idc * (vd0/ (3*np.sqrt(2)/np.pi) ) / V_LINE  # referred to 400 V
        I1p += i1
        Qtot += s1*np.sqrt(max(0,1-cphi**2))
        for h, r in harmonics(a,u).items():
            hsum[h] = hsum.get(h,0) + r*i1
    thd = np.sqrt(sum(v**2 for v in hsum.values()))/I1p
    return dict(v=v_req, i=idc, act=act, P=P, Q=Qtot, I1=I1p, harm=hsum, thd=thd)

# --------------- AC filter -------------
# series reactor between PCC and filter bus
S_CONV = sum(VD0)*ID_R*np.pi/3   # transformer rating basis
X_SER  = 0.10*V_LINE**2/S_CONV   # 10 % on converter base
# tuned branches (Qc at fundamental, tuning harmonic) + 2nd-order high pass
FILTERS = [dict(name="5th", q=75e3, h=4.85, R=None),
           dict(name="7th", q=40e3, h=6.8, R=None),
           dict(name="HP",  q=35e3, h=11.0, R="q2")]
def branch_params(f):
    xc = V_LINE**2/f["q"]                 # at fundamental
    C  = 1/(W*xc)
    L  = xc/(f["h"]**2*W)                 # (1/(w0 sqrt(LC)) = h)
    R  = None
    if f["R"] == "q2":
        R = 2.0*np.sqrt(L/C)               # quality factor ~2 for a high-pass
    return C, L, R
def z_branch(f, h):
    C, L, R = branch_params(f)
    zc = 1/(1j*h*W*C); zl = 1j*h*W*L
    if R is None:
        return zc + zl + 0.02*zl.imag      # small series resistance (Q~50)
    return zc + 1/(1/zl + 1/R)
def z_filter(h):
    y = sum(1/z_branch(f,h) for f in FILTERS)
    return 1/y
def grid_share(h):
    zs = 1j*h*(X_S + X_SER)
    zf = z_filter(h)
    return abs(zf/(zf+zs))

def thd_pcc(op):
    hs = {h: v*grid_share(h) for h,v in op["harm"].items()}
    return np.sqrt(sum(v**2 for v in hs.values()))/op["I1"], hs

# --------------- DC filter --------------
L_DC = 0.5e-3; C_DC = 10.0e-3
def dc_ripple(op):
    """6th-harmonic voltage (rms) at bridge output, arithmetic sum of the active bridges;
    LC filter loaded by R = V/I. returns V6 bridge, V6 load, V6 load/V, I6 reactor/I, I6 load/I"""
    v6 = 0
    for w,vd0,a in op["act"]:
        v6 += vd0*np.sqrt(2)/35*np.sqrt(1+35*np.sin(np.radians(a))**2)
    R = op["v"]/op["i"]
    w6 = 6*W
    H = 1/(1 - w6**2*L_DC*C_DC + 1j*w6*L_DC/R)
    v6o = v6*H
    i6L = (v6 - v6o)/(1j*w6*L_DC)
    return v6, abs(v6o), abs(v6o)/op["v"], abs(i6L)/op["i"], abs(v6o)/R/op["i"]

# ---------------- tables ----------------
def cfg_str(act):
    return "+".join(str(w) for w,_,_ in act)
def alpha_str(act):
    return "/".join(f"{a:.0f}" for _,_,a in act)

allcases = [(v, I_CASE) for v in CASES] + REF
rows = []
for v,i in allcases:
    op = operating_point(v,i)
    thd2, hs = thd_pcc(op)
    v6, v6o, rv, ri, ril = dc_ripple(op)
    n = int(np.ceil(v/V_UNIT - 1e-9))
    rows.append((v,i,n,cfg_str(op["act"]),alpha_str(op["act"]),op["P"]/1e3,op["Q"]/1e3,
                 op["P"]/np.hypot(op["P"],op["Q"]),op["I1"],op["thd"]*100,thd2*100, v6, 100*rv, 100*ri, 100*ril, v6o))

with open("tables/cases.tex","w") as fo:
    fo.write("\\begin{tabular}{rrrlrrrrrrr}\n\\toprule\n")
    fo.write("$V_{out}$ & $I_{out}$ & $n$ & bridges & $\\alpha$ & $P$ & $Q$ & PF & $I_1$ & THD$_{conv}$ & THD$_{PCC}$\\\\\n")
    fo.write("(V) & (A) & & (units) & (deg) & (kW) & (kvar) & & (A) & (\\%) & (\\%)\\\\\n\\midrule\n")
    for r in rows:
        fo.write(f"{r[0]:.1f} & {r[1]:.0f} & {r[2]} & {r[3]} & {r[4]} & {r[5]:.1f} & {r[6]:.1f} & {r[7]:.2f} & {r[8]:.0f} & {r[9]:.1f} & {r[10]:.1f}\\\\\n")
    fo.write("\\bottomrule\n\\end{tabular}\n")

with open("tables/dcripple.tex","w") as fo:
    fo.write("\\begin{tabular}{rrlrrrrr}\n\\toprule\n")
    fo.write("$V_{out}$ & $I_{out}$ & bridges & $\\alpha$ & $V_6$ bridge & $V_6$ load & $\\Delta V_{pp}/V_{out}$ & $\\Delta I_{pp}/I_{out}$ reactor\\\\\n")
    fo.write("(V) & (A) & (units) & (deg) & (V rms) & (V rms) & (\\%) & (\\%)\\\\\n\\midrule\n")
    for r in rows:
        fo.write(f"{r[0]:.1f} & {r[1]:.0f} & {r[3]} & {r[4]} & {r[11]:.1f} & {r[15]:.2f} & {2*np.sqrt(2)*r[12]:.2f} & {2*np.sqrt(2)*r[13]:.1f}\\\\\n")
    fo.write("\\bottomrule\n\\end{tabular}\n")

# baseline: single bridge sized for 200 V
VD0_1 = 200*K_HEAD
with open("tables/baseline.tex","w") as fo:
    fo.write("\\begin{tabular}{rrrrrr}\n\\toprule\n$V_{out}$ (V) & $\\alpha$ (deg) & $P$ (kW) & $Q$ (kvar) & PF & $I_1$ (A)\\\\\n\\midrule\n")
    for v in [16.4,32.8,65.6,98.4,131.2,164,196.8]:
        cosa = (v + dx_drop(VD0_1,I_CASE) + 2*V_T)/VD0_1
        a = np.degrees(np.arccos(cosa)); u = overlap(VD0_1,a,I_CASE)
        cphi = (np.cos(np.radians(a))+np.cos(np.radians(a+u)))/2
        s1 = VD0_1*I_CASE; P = v*I_CASE; Q = s1*np.sqrt(1-cphi**2)
        i1 = np.sqrt(6)/np.pi*I_CASE*(VD0_1/(3*np.sqrt(2)/np.pi))/V_LINE
        fo.write(f"{v:.1f} & {a:.1f} & {P/1e3:.1f} & {Q/1e3:.1f} & {P/np.hypot(P,Q):.2f} & {i1:.0f}\\\\\n")
    fo.write("\\bottomrule\n\\end{tabular}\n")

# bridge / transformer data
with open("tables/bridges.tex","w") as fo:
    fo.write("\\begin{tabular}{lrrrrrr}\n\\toprule\nBridge & weight & $V_{step}$ (V) & $V_{d0}$ (V) & $E_{2}$ (V) & $I_{2}$ (A) & $S_{2}$ (kVA)\\\\\n\\midrule\n")
    for k,(w,vd0,e) in enumerate(zip(WEIGHTS,VD0,ELL)):
        i2 = np.sqrt(2/3)*ID_R; s2 = np.sqrt(3)*e*i2
        fo.write(f"B{k+1} & {w} & {w*V_UNIT:.1f} & {vd0:.1f} & {e:.1f} & {i2:.0f} & {s2/1e3:.1f}\\\\\n")
    fo.write("\\bottomrule\n\\end{tabular}\n")

# filter data
with open("tables/filter.tex","w") as fo:
    fo.write("\\begin{tabular}{lrrrrrr}\n\\toprule\nBranch & $Q_C$ (kvar) & $h_t$ & $f_t$ (Hz) & $C$ ($\\mu$F) & $L$ ($\\mu$H) & $R$ ($\\Omega$)\\\\\n\\midrule\n")
    for f in FILTERS:
        C,L,R = branch_params(f)
        fo.write(f"{f['name']} & {f['q']/1e3:.0f} & {f['h']:.2f} & {f['h']*F:.0f} & {C*1e6:.0f} & {L*1e6:.0f} & {('%.2f'%R) if R else '--'}\\\\\n")
    fo.write("\\bottomrule\n\\end{tabular}\n")

with open("tables/share.tex","w") as fo:
    fo.write("\\begin{tabular}{l"+"r"*10+"}\n\\toprule\n$h$ & "+" & ".join(str(h) for h in [5,7,11,13,17,19,23,25,35,37])+"\\\\\n\\midrule\n")
    fo.write("$|Z_F|$ (m$\\Omega$) & "+" & ".join(f"{abs(z_filter(h))*1e3:.1f}" for h in [5,7,11,13,17,19,23,25,35,37])+"\\\\\n")
    fo.write("$|Z_S+Z_{ser}|$ (m$\\Omega$) & "+" & ".join(f"{h*(X_S+X_SER)*1e3:.1f}" for h in [5,7,11,13,17,19,23,25,35,37])+"\\\\\n")
    fo.write("$I_{h,PCC}/I_{h,conv}$ & "+" & ".join(f"{grid_share(h):.3f}" for h in [5,7,11,13,17,19,23,25,35,37])+"\\\\\n")
    fo.write("\\bottomrule\n\\end{tabular}\n")

# worst-case spectrum (largest THD_pcc) details
worst = max(allcases, key=lambda c: thd_pcc(operating_point(*c))[0])
op = operating_point(*worst); t2, hs = thd_pcc(op)
with open("tables/spectrum.tex","w") as fo:
    hh = [5,7,11,13,17,19,23,25]
    fo.write("\\begin{tabular}{l"+"r"*len(hh)+"}\n\\toprule\n$h$ & "+" & ".join(str(h) for h in hh)+"\\\\\n\\midrule\n")
    fo.write("$I_h/I_1$ converter (\\%) & "+" & ".join(f"{100*op['harm'][h]/op['I1']:.1f}" for h in hh)+"\\\\\n")
    fo.write("$I_h/I_1$ at PCC (\\%) & "+" & ".join(f"{100*hs[h]/op['I1']:.2f}" for h in hh)+"\\\\\n")
    fo.write("\\bottomrule\n\\end{tabular}\n")

# parallel resonance check: |Z| of filter // network seen from converter, h = 1..12 step 0.1
hh = np.arange(1.5,12,0.05)
zpar = []
for h in hh:
    zs = 1j*h*(X_S+X_SER); zf = z_filter(h)
    zpar.append(abs(zs*zf/(zs+zf)))
zpar = np.array(zpar)
idx = np.argmax(zpar[hh<4.5])
print("parallel resonance near h=%.2f, |Z|=%.1f mOhm" % (hh[idx], zpar[idx]*1e3))

# summary numbers for the text
S1max = max(r[8] for r in rows)*V_LINE*np.sqrt(3)
print("X_S = %.2f mOhm, Ssc = %.1f MVA, X_ser = %.2f mOhm, S_conv = %.0f kVA" % (X_S*1e3, SSC/1e6, X_SER*1e3, S_CONV/1e3))
print("VD0:", [round(v,1) for v in VD0], "ELL:", [round(e,1) for e in ELL])
print("Qmax = %.1f kvar, THDconv max = %.1f %%, THDpcc max = %.1f %% (case %s)" % (max(r[6] for r in rows), max(r[9] for r in rows), max(r[10] for r in rows), worst))
print("Filter Qc total = %.0f kvar" % (sum(f['q'] for f in FILTERS)/1e3))
for r in rows: print(r[0], r[3], r[4], "Q=%.1f" % r[6], "thd=%.1f/%.1f" % (r[9], r[10]), "rip=%.2f %.1f %.2f" % (2.83*r[12], 2.83*r[13], 2.83*r[14]))
L_DC_E = 0.5*L_DC*ID_R**2
print("f_r dc = %.1f Hz, E_L = %.0f J" % (1/(2*np.pi*np.sqrt(L_DC*C_DC)), L_DC_E))

print("shares:", {h: round(grid_share(h),3) for h in [5,7,11,13,17,19,23,25]})
QF = sum(f['q'] for f in FILTERS)
print("bus rise at no load: %.2f %%" % (100*QF*X_SER/V_LINE**2))
print("I_base conv = %.0f A, X_ser = %.2f mOhm, L_ser = %.0f uH" % (S_CONV/np.sqrt(3)/V_LINE, X_SER*1e3, X_SER/W*1e6))

print("K_HEAD = %.3f" % K_HEAD)
for h in [2.5,2.8,2.9,3.0,3.1,3.2,3.4,3.6,4.0,4.4]:
    zs = 1j*h*(X_S+X_SER); zf = z_filter(h); print("h=%.1f |Zpar|=%.2f ohm |Zf|=%.3f" % (h, abs(zs*zf/(zs+zf)), abs(zf)))
# ripple with 0.5 mH

# ---- pgfplots data: Q vs Vout, baseline vs proposed ----
with open("tables/qplot.tex","w") as fo:
    fo.write("\\addplot[thick,color=red,mark=*] coordinates {")
    for v in np.arange(16.4, 200.1, 8.2):
        cosa = (v + dx_drop(VD0_1,I_CASE) + 2*V_T)/VD0_1
        a = np.degrees(np.arccos(cosa)); u = overlap(VD0_1,a,I_CASE)
        cphi = (np.cos(np.radians(a))+np.cos(np.radians(a+u)))/2
        fo.write(f"({v:.1f},{VD0_1*I_CASE*np.sqrt(1-cphi**2)/1e3:.1f}) ")
    fo.write("};\n\\addlegendentry{single bridge, $V_{d0}=240$ V}\n")
    fo.write("\\addplot[thick,color=blue,mark=square*] coordinates {")
    for v in np.arange(16.4, 200.1, 4.1):
        op = operating_point(v, I_CASE); fo.write(f"({v:.1f},{op['Q']/1e3:.1f}) ")
    fo.write("};\n\\addlegendentry{4 series bridges 1:2:4:8, sequential control}\n")
    fo.write("\\addplot[thick,color=black,dashed] coordinates {(16.4,150) (200,150)};\n\\addlegendentry{AC filter $Q_C$}\n")
