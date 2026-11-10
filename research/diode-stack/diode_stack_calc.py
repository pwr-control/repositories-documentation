"""Worked example of diode_stack_design_guidelines.tex: series-connected diode stack of a six-pulse bridge.

The script follows the design path of the note and writes every number quoted in it:
  1. operating point of the bridge (overlap, commutation voltage, current slope) and ideal arm voltage;
  2. static sharing (leakage, balancing resistor);
  3. reverse recovery at the current slope of the circuit (diode_stack_recovery.py);
  4. commutation transient in the bridge: two-mode model of the snubbed arms (closed form);
  5. switched simulations of the bridge (diode_stack_sim.py): reference cases with a resistive load (R) and
     with a smoothing reactor (L), tolerances, sweeps of Qrr, unbalance, Csn, Rsn, Rb, Ld and n;
  6. hand formulas against the simulations;
  7. losses and thermal design, insulation voltages and distances, faults.

The example is generic: grid, transformer, load, diode and resistor are round values typical of this kind of
rectifier, not the data of a plant or of a product. To design a real stack, replace the parameters below with the
data of the installation and of the datasheets.

Outputs, next to the script: diode_stack_results.txt (log), diode_stack_values.tex (macros \\ds... of the note),
figures/fig_*.pdf. Only numpy, scipy and matplotlib are needed (numpy 1.x and 2.x). About three minutes.
Set the environment variable DIODE_STACK_CACHE to a folder to store the simulation results and reuse them.
Run:  python diode_stack_calc.py"""
import hashlib, json, math, os, pickle, re, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diode_stack_sim import simulate, lm_params, lm_recovery, level_array
import diode_stack_recovery as rcv

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")

# ---------------------------------------------------------------- parameters of the worked example
# grid and transformer
U_N, K_GRID, F = 10e3, 1.10, 50.0       # V rms line-to-line, sustained overvoltage factor, Hz
S_T, UK, UK_MIN = 1.6e6, 0.07, 0.05     # VA, short-circuit voltage of the transformer (rated and lowest)
I_D = 100.0                             # A, DC current at the rated grid voltage
# DC side: (R) resistive load behind a small inductance, (L) smoothing reactor
LD_R, LD_L = 3e-3, 0.3                  # H
LD_STIFF = 3.0                          # H, current-stiff limit used to check the analytical model
LD_MORE = (10e-3, 0.1, 0.6, 1.5)        # H, further values for the sweep of the DC inductance
# stack and sharing network of a level: diode || Rb || (Rsn + Csn)
N = 6                                   # diodes in series per arm
RB, RSN, CSN = 470e3, 150.0, 150e-9     # Ohm, Ohm, F
TOL_RB, TOL_C = 0.05, 0.05              # tolerances of Rb and Csn
K_U = 0.80                              # highest repetitive peak on a level, referred to VRRM
# diode: assumed datasheet of a 4.5 kV press-pack rectifier diode (round values typical of the class)
VRRM, VRSM = 4500.0, 4600.0             # V
TVJ_MAX = 150.0                         # C, maximum junction temperature (blocking)
IRM = 50e-3                             # A, at VRRM and TVJ_MAX
VT0, RT = 0.85, 0.60e-3                 # V, Ohm, at TVJ_MAX
RTH_JC, RTH_CH = 0.025, 0.010           # K/W, double side cooled
IFSM, I2T = 12e3, 720e3                 # A, A2s (half sine of 10 ms)
QS_TAB, IRR_TAB, A_TAB, IF_TAB, T_TAB = 3200.0, 125.0, 5.0, 1000.0, 150.0   # uC, A, A/us, A, C
QS_EXP, QS_RANGE = 0.30, (0.5, 50.0)    # QS(di/dt) curve of the datasheet: exponent, range of the slope in A/us
QS_CURVE = tuple((a, QS_TAB*(a/A_TAB)**QS_EXP) for a in QS_RANGE)           # its two ends (A/us, uC)
# normalised leakage current against voltage and temperature: typical curves, read on the graphs of the
# application note AN6531 (Dynex), Fig. 4; the 150 C point is extrapolated
KV_PCT = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
KV_VAL = [0, 0.04, 0.10, 0.16, 0.21, 0.28, 0.33, 0.40, 0.52, 0.72, 1.00]
KT_T = [25, 50, 70, 85, 100, 110, 120, 125, 130, 140, 150]
KT_VAL = [0.03, 0.05, 0.085, 0.136, 0.255, 0.44, 0.75, 1.0, 1.4, 2.7, 4.2]
# class reference for the bulk lifetime: published curves of a 4.2 kV thyristor, application note AN5950 (Dynex),
# Figs. 12 and 13 (Tj 125 C, 1 to 30 A/us): QS = cq*(di/dt)^nq [uC], IRR = ci*(di/dt)^ni [A]
AN5950 = {"max": (3397.4, 0.5061, 48.236, 0.7553), "min": (1357.3, 0.6271, 29.853, 0.8222)}
AN5950_T, AN5950_IF, AN5950_RANGE = 125.0, 1000.0, (1.0, 30.0)
TAU_UPPER = 200.0                       # us at 150 C, long-lifetime scenario
# design values of the recovery, given at the current slope of the rated operating point (A_REF, computed below)
# and at the design junction temperature
TJ_DES = 100.0                          # C
Q_DES, I_DES, U_UNB = 0.50e-3, 11.3, 0.25  # C, A, unbalance: the weak diode has Qrr*(1 - u)
T_SNAP = 1e-6                           # s, tail time constant of the snappy diode used to check the model
# sweeps of the simulation
Q_SWEEP = (0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.2)                   # recovered charge, multiples of Q_DES
U_SWEEP = (0.0, 0.10, 0.25, 0.40)                               # unbalance (U_UNB has to be one of them)
C_SWEEP = (68e-9, 100e-9, 150e-9, 220e-9, 330e-9, 470e-9)       # F (CSN has to be one of them)
R_SWEEP = (68.0, 100.0, 150.0, 220.0, 330.0, 470.0)             # Ohm (RSN has to be one of them)
OPTIONS = ((220e-9, 220.0), (330e-9, 180.0))                    # (Csn, Rsn) of the larger snubbers tried in case (L)
DAMP = ((680.0, 150e-9), (680.0, 300e-9), (330.0, 300e-9))      # (Rx, Cx) of a damper across the DC terminals
# ambient and resistors: assumed thick-film high-voltage element, linear derating from R_PN at R_TN to zero at R_TMAX
T_A, ALTITUDE = 55.0, 1000.0            # C (air inside the stack), m
R_PN, R_TN, R_TMAX = 100.0, 25.0, 225.0 # W, C, C
K_P = 0.60                              # highest power of a resistor, referred to its rating at T_A
# insulation
U_IMP_STD = (2.5e3, 4e3, 6e3, 8e3, 12e3)                        # V, standard impulse withstand voltages
U_IMP_LEVEL = min(u for u in U_IMP_STD if u >= 1.3*VRSM)        # V, impulse withstand voltage across one level
CJ = (0.3e-9, 0.6e-9)                   # F, junction capacitance of the blocking diode at the working voltage (estimate)
KAPPA = 1.8                             # peak factor of the fault current
# simulation
DT, DT_FINE, PERIODS = 0.5e-6, 0.25e-6, 2.5
ARM = 1                                 # arm used for the results: upper arm of phase b (all the arms are equal)
R_NOLEAK = 1e12

# IEC 60664-1 tables as used here (values transcribed, to be checked against the standard before use)
# Table F.2, case A (inhomogeneous field), up to 2000 m: (impulse withstand voltage kV, clearance mm)
F2 = [(2.5, 1.5), (4.0, 3.0), (6.0, 5.5), (8.0, 8.0), (12, 14), (15, 18), (20, 25), (25, 33), (30, 40), (40, 60),
      (50, 75), (60, 90), (80, 130), (100, 170)]
# creepage distances (Table F.5 of the 2020 edition, F.4 of the 2007 edition), pollution degree 2:
# (V rms, material group I, II, III) mm
F5 = [(1000, 5.0, 7.1, 10.0), (1250, 6.3, 9.0, 12.5), (1600, 8.0, 11, 16), (2000, 10, 14, 20), (2500, 12.5, 18, 25),
      (3200, 16, 22, 32), (4000, 20, 28, 40), (5000, 25, 36, 50), (6300, 32, 45, 63), (8000, 40, 56, 80),
      (10000, 50, 71, 100), (12500, 63, 90, 125), (16000, 80, 110, 160), (20000, 100, 140, 200),
      (25000, 125, 180, 250), (32000, 160, 220, 320)]

# ---------------------------------------------------------------- style of the figures
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_NOM, C_WEAK, C_3, C_4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"       # categorical colours, fixed order
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]  # ordinal ramp, light to dark
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.3, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03})

W = 2*math.pi*F
LOG, MAC = [], {}


def line(s=""):
    print(s)
    LOG.append(s)


def put(name, value, d=1):
    """macro \\ds<name> of the note (letters only)."""
    assert name.isalpha(), name
    MAC[name] = value if isinstance(value, str) else f"{value:.{d}f}"


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, float)**2)))


# ---------------------------------------------------------------- 1. operating point of the bridge
def op_point(U, Id, Lc):
    """ideal six-pulse bridge with constant DC current Id: overlap, DC voltage, commutation voltage V0 at the
    end of the overlap and current slope alpha = V0/(2 Lc) at the current zero."""
    Vhat = math.sqrt(2)*U
    mu = math.acos(1 - 2*W*Lc*Id/Vhat)
    V0 = Vhat*math.sin(mu)
    return dict(U=U, Id=Id, Lc=Lc, Vhat=Vhat, mu=mu, Vdc=3/math.pi*Vhat - 3*W*Lc/math.pi*Id, V0=V0,
                alpha=V0/(2*Lc), tramp=Id/(V0/(2*Lc)))


def id_resistive(U, Lc, R):
    """DC current of the ideal bridge on a resistance R."""
    return 3/math.pi*math.sqrt(2)*U/(R + 3*W*Lc/math.pi)


def arm_voltage_ideal(theta, mu, Vhat):
    """reverse voltage of an arm of the ideal bridge; theta is the angle from the natural commutation instant at
    which the arm hands its current over."""
    th = np.mod(theta, 2*math.pi)
    s3, p3 = math.sqrt(3)/2, math.pi/3
    conds = [th < mu, th < p3, th < p3 + mu, th < 2*p3, th < 2*p3 + mu, th < 3*p3, th < 3*p3 + mu, th < 4*p3]
    vals = [0*th, Vhat*np.sin(th), s3*Vhat*np.cos(th - p3), Vhat*np.sin(th), s3*Vhat*np.cos(th - 2*p3),
            Vhat*np.sin(th - p3), -s3*Vhat*np.cos(th), Vhat*np.sin(th - p3)]
    return np.select(conds, vals, 0.0)


LC = UK*U_N**2/(W*S_T)
LC_MIN = LC*UK_MIN/UK
I_TN = S_T/(math.sqrt(3)*U_N)
OP_N = op_point(U_N, I_D, LC)
R_LOAD = OP_N["Vdc"]/I_D
A_REF = OP_N["alpha"]                             # A/s, slope at which the design set of the recovery is given
Q_SNAP = I_DES*(I_DES/(2*A_REF) + T_SNAP)         # C, snappy diode with the peak current of the design set
OP = {"Nom": OP_N,
      "Max": op_point(K_GRID*U_N, id_resistive(K_GRID*U_N, LC, R_LOAD), LC),
      "Ukmin": op_point(K_GRID*U_N, id_resistive(K_GRID*U_N, LC_MIN, R_LOAD), LC_MIN),
      "Half": op_point(U_N, id_resistive(U_N, LC, 2*R_LOAD), LC)}


def section_operating_point():
    line("1. Operating point of the ideal bridge")
    line(f"   Lc {LC*1e3:.2f} mH (uk {UK*100:.0f} % on {S_T/1e6:.2f} MVA), {LC_MIN*1e3:.2f} mH (uk {UK_MIN*100:.0f} %); "
         f"transformer rated current {I_TN:.2f} A; load resistance {R_LOAD:.1f} Ohm")
    names = {"Nom": "rated grid, rated load", "Max": f"grid +{(K_GRID-1)*100:.0f} %, same load resistance",
             "Ukmin": f"grid +{(K_GRID-1)*100:.0f} %, uk {UK_MIN*100:.0f} %", "Half": "rated grid, half load"}
    th = np.linspace(0, 2*math.pi, 72000, endpoint=False)
    for k, o in OP.items():
        v = arm_voltage_ideal(th, o["mu"], o["Vhat"])
        o["Vs_rms"], o["Vs_mean"] = rms(v), float(np.mean(v))
        line(f"   {names[k]:32s}: Id {o['Id']:5.1f} A, mu {math.degrees(o['mu']):5.2f} deg, Vdc {o['Vdc']/1e3:6.2f} kV, "
             f"V0 {o['V0']/1e3:5.2f} kV, alpha {o['alpha']/1e6:.4f} A/us, Id/alpha {o['tramp']*1e6:4.0f} us; "
             f"arm: crest {o['Vhat']/1e3:.2f} kV, rms {o['Vs_rms']/1e3:.2f} kV, mean {o['Vs_mean']/1e3:.2f} kV")
        put(f"Id{k}", o["Id"], 1); put(f"Mu{k}", math.degrees(o["mu"]), 1); put(f"Vdc{k}", o["Vdc"]/1e3, 2)
        put(f"Vzero{k}", o["V0"]/1e3, 2); put(f"Alpha{k}", o["alpha"]/1e6, 3); put(f"Tramp{k}", o["tramp"]*1e6, 0)
        put(f"Vhat{k}", o["Vhat"]/1e3, 2); put(f"VsRms{k}", o["Vs_rms"]/1e3, 2); put(f"VsMean{k}", o["Vs_mean"]/1e3, 2)
        put(f"Tov{k}", o["mu"]/W*1e3, 2)
    v0 = arm_voltage_ideal(th, 1e-9, 1.0)
    put("KrmsZero", rms(v0), 3); put("KrmsUZero", rms(v0)*math.sqrt(2), 3)
    put("KrmsNom", OP_N["Vs_rms"]/OP_N["Vhat"], 3)
    put("Lc", LC*1e3, 2); put("LcMin", LC_MIN*1e3, 2); put("Itn", I_TN, 1); put("Rload", R_LOAD, 1)
    put("St", f"{S_T/1e6:g}"); put("Un", f"{U_N/1e3:g}"); put("Umax", f"{K_GRID*U_N/1e3:g}")
    put("Uk", UK*100, 0); put("UkMin", UK_MIN*100, 0); put("IdRated", I_D, 0); put("Kgrid", (K_GRID - 1)*100, 0)
    put("AlphaRatioUkmin", OP["Ukmin"]["alpha"]/OP["Max"]["alpha"], 2)
    al, tr = [o["alpha"]/1e6 for o in OP.values()], [o["tramp"]*1e6 for o in OP.values()]
    put("AlphaLo", min(al), 2); put("AlphaHi", max(al), 2)
    put("TrampLo", math.floor(min(tr)/10)*10, 0); put("TrampHi", math.ceil(max(tr)/10)*10, 0)


# ---------------------------------------------------------------- 2. static sharing
def leak(v, Tj):
    """leakage current of a device at the datasheet limit, at the reverse voltage v [V] and at Tj [C]."""
    kv = np.interp(np.clip(v, 0, VRRM)/VRRM*100, KV_PCT, KV_VAL)
    kt = np.exp(np.interp(Tj, KT_T, np.log(KT_VAL)))
    return IRM*kv*kt/KT_VAL[-1]


V_LEV = K_GRID*math.sqrt(2)*U_N/N                 # balanced level voltage at the crest, highest grid voltage
ROFF_LIM = V_LEV/float(leak(V_LEV, TJ_DES))       # equivalent resistance of a device at the leakage limit


def rb_max(VD, Ilk, a=TOL_RB, b=K_GRID - 1, n=N):
    """AN6531 Eq. (2): highest balancing resistance for the leakage current Ilk."""
    return (n*(1 - a)*VD - (1 + a)*(1 + b)*math.sqrt(2)*U_N)/((n - 1)*(1 - a*a)*Ilk)


def section_static():
    line("\n2. Static sharing")
    Ilk = float(leak(V_LEV, TJ_DES))
    line(f"   balanced level voltage at the crest (grid +{(K_GRID-1)*100:.0f} %): {V_LEV/1e3:.3f} kV; leakage of a device at the "
         f"limit at {TJ_DES:.0f} C: {Ilk*1e3:.2f} mA -> equivalent resistance {ROFF_LIM/1e6:.2f} MOhm")
    put("Vlev", V_LEV/1e3, 2); put("IlkDes", Ilk*1e3, 2); put("RoffLim", ROFF_LIM/1e6, 2)
    put("IlkRatio", V_LEV/RB/Ilk, 1); put("IRb", V_LEV/RB*1e3, 1)
    Ts = np.linspace(25, 150, 2501)
    Il = leak(V_LEV, Ts)
    for nm, VD in (("Des", K_U*VRRM), ("Rated", VRRM)):
        Rmax = rb_max(VD, Ilk)
        Iadm = rb_max(VD, 1.0)/RB
        Tcov = float(Ts[np.argmax(Il > Iadm)]) if np.any(Il > Iadm) else 150.0
        line(f"   AN6531 (2), V_D = {VD/1e3:.1f} kV, a = {TOL_RB:.2f}, b = {K_GRID-1:.2f}: Rb <= {rb_max(VD, 1.0):.0f} V / I_lk = "
             f"{Rmax/1e3:.0f} kOhm at {TJ_DES:.0f} C; with {RB/1e3:.0f} kOhm I_lk <= {Iadm*1e3:.2f} mA, covered up to {Tcov:.0f} C")
        put(f"RbNum{nm}", rb_max(VD, 1.0), 0); put(f"RbMax{nm}", Rmax/1e3, 0); put(f"IlkAdm{nm}", Iadm*1e3, 2)
        put(f"Tcov{nm}", Tcov, 0)
    for T, nm in ((85, "EightyFive"), (100, "Hundred"), (125, "HundredTwentyFive"), (150, "HundredFifty")):
        put(f"Ilk{nm}", float(leak(V_LEV, T))*1e3, 2)
    tb = RB*CSN
    put("TauB", tb*1e3, 0); put("Tblock", 2/(3*F)*1e3, 1); put("DecayBlock", math.exp(-2/(3*F)/tb)*100, 0)
    line(f"   Rb*Csn = {tb*1e3:.0f} ms against a blocking interval of {2/(3*F)*1e3:.1f} ms: "
         f"{math.exp(-2/(3*F)/tb)*100:.0f} % of an offset is left at the end of the interval")
    put("KvRows", " & ".join(f"{x:g}" for x in KV_VAL)); put("KvPct", " & ".join(f"{x:g}" for x in KV_PCT))
    put("KtRows", " & ".join(f"{x:g}" for x in KT_VAL)); put("KtT", " & ".join(f"{x:g}" for x in KT_T))


# ---------------------------------------------------------------- 3. recovery at the slope of the circuit
def section_recovery():
    line("\n3. Reverse recovery at the current slope of the circuit (units: us, A, uC)")
    out = {}
    # class reference: bulk lifetime from curves that span a wide range of slopes (AN5950)
    A = np.geomspace(*AN5950_RANGE, 7)
    for nm, (cq, nq, ci, ni) in AN5950.items():
        tb, TM, k, r = rcv.fit_curves([(a, cq*a**nq, ci*a**ni) for a in A], AN5950_IF)
        wp = rcv.recovery(0.1, I_D, 1/tb, TM, k)
        out[f"cls_{nm}"] = dict(tau=tb, TM=TM, k=k, rms=r)
        line(f"   AN5950 {nm}: tau_b {tb:.0f} us at {AN5950_T:.0f} C, T_M {TM:.1f} us, k {k:.2e} 1/(uC us), rms of the fit "
             f"{r*100:.1f} %; at 0.1 A/us the model gives {wp['Qrr']:.0f} uC / {wp['Irrm']:.2f} A, the power law "
             f"{cq*0.1**nq:.0f} uC / {ci*0.1**ni:.2f} A")
        cap = nm.capitalize()
        put(f"ClsTau{cap}", tb, 0); put(f"ClsRms{cap}", r*100, 0); put(f"ClsQ{cap}", wp["Qrr"], 0)
        put(f"ClsQlaw{cap}", cq*0.1**nq, 0); put(f"ClsNq{cap}", nq, 2); put(f"ClsNi{cap}", ni, 2)
        put(f"ClsQratio{cap}", cq*0.1**nq/wp["Qrr"], 1)
    tau_class = rcv.t_scale(AN5950_T, T_TAB)*out["cls_max"]["tau"]
    put("TauClass", tau_class, 0); put("TauUpper", TAU_UPPER, 0)
    line(f"   class bulk lifetime at {T_TAB:.0f} C (tau ~ T): {tau_class:.0f} us")
    # device: table point with the local slope of the QS curve of the datasheet
    cq, eq = rcv.power_law(*QS_CURVE)
    q_ref = [(a, QS_TAB*(a/A_TAB)**eq) for a in (A_TAB/1.2, A_TAB, 1.2*A_TAB)]
    i_ref = (A_TAB, IRR_TAB)
    put("QsSlope", eq, 2); put("QsCurveAtTab", cq*A_TAB**eq, 0)
    put("QsLo", QS_CURVE[0][1], 0); put("QsHi", QS_CURVE[1][1], 0); put("AQsLo", f"{QS_CURVE[0][0]:g}")
    put("AQsHi", f"{QS_CURVE[1][0]:g}")
    put("QsTab", QS_TAB, 0); put("IrrTab", IRR_TAB, 0); put("ATab", f"{A_TAB:g}"); put("IfTab", IF_TAB, 0)
    put("TTab", T_TAB, 0); put("SoftTab", 2*A_TAB*QS_TAB/IRR_TAB**2 - 1, 2); put("TaTab", IRR_TAB/A_TAB, 0)
    line(f"   datasheet curve: QS ~ (di/dt)^{eq:.3f}, {cq*A_TAB**eq:.0f} uC at {A_TAB:.0f} A/us (table: {QS_TAB:.0f} uC max, "
         f"{IRR_TAB:.0f} A)")
    fq, fi = rcv.t_scale(T_TAB, TJ_DES, 2.0), rcv.t_scale(T_TAB, TJ_DES, 1.0)
    put("FacQ", fq, 2); put("FacI", fi, 2)
    tau0, TM0, tt0 = rcv.lm_from_point(A_TAB, QS_TAB, IRR_TAB)
    sc = {"floor": dict(gb=1/tau0, TM=TM0, k=0.0, tau=tau0, label="Lauritzen-Ma through the table point")}
    for key, tb, lab in (("central", tau_class, "class lifetime"), ("upper", TAU_UPPER, "long lifetime"),
                         ("ceiling", None, "no bulk recombination")):
        gb = 0.0 if tb is None else 1/tb
        TM, k = rcv.fit_point(gb, q_ref, i_ref, IF_TAB)
        sc[key] = dict(gb=gb, TM=TM, k=k, tau=tb, label=lab)
    pts = {"Nom": (OP["Nom"]["alpha"]/1e6, OP["Nom"]["Id"]), "Max": (OP["Max"]["alpha"]/1e6, OP["Max"]["Id"]),
           "Ukmin": (OP["Ukmin"]["alpha"]/1e6, OP["Ukmin"]["Id"])}
    line(f"   from {T_TAB:.0f} C to {TJ_DES:.0f} C: Qrr x{fq:.2f}, Irrm x{fi:.2f} (tau ~ T)")
    line("   scenario  tau_b    T_M      k         point  alpha   Qrr150 Irrm150  Qrr(Tj) Irrm(Tj)   t_a    s")
    for key, s in sc.items():
        s["wp"] = {}
        for pk, (a, IF) in pts.items():
            r = rcv.recovery(a, IF, s["gb"], s["TM"], s["k"])
            s["wp"][pk] = dict(Q150=r["Qrr"], I150=r["Irrm"], Q=r["Qrr"]*fq, I=r["Irrm"]*fi, ta=r["ta"], s=r["s"])
            tb_txt = "inf" if s["tau"] is None else f"{s['tau']:.0f}"
            line(f"   {key:8s} {tb_txt:>5s} {s['TM']:6.1f} {s['k']:10.2e}   {pk:5s} {a:.3f}  {r['Qrr']:7.0f} {r['Irrm']:6.2f}  "
                 f"{r['Qrr']*fq:7.0f} {r['Irrm']*fi:6.2f}  {r['ta']:5.0f} {r['s']:5.2f}")
            cap = key.capitalize()
            put(f"Q{cap}{pk}", r["Qrr"]*fq, 0); put(f"I{cap}{pk}", r["Irrm"]*fi, 1)
            put(f"Qhot{cap}{pk}", r["Qrr"], 0); put(f"Ihot{cap}{pk}", r["Irrm"], 1)
        put(f"Tm{key.capitalize()}", s["TM"], 1); put(f"K{key.capitalize()}", s["k"]*1e6, 2)
        put(f"Soft{key.capitalize()}", s["wp"]["Nom"]["s"], 2); put(f"Ta{key.capitalize()}", s["wp"]["Nom"]["ta"], 0)
        # check on the whole datasheet curve
        s["dev"] = [rcv.recovery(a, IF_TAB, s["gb"], s["TM"], s["k"])["Qrr"]/q - 1 for a, q in QS_CURVE]
        put(f"DevLo{key.capitalize()}", s["dev"][0]*100, 0); put(f"DevHi{key.capitalize()}", s["dev"][1]*100, 0)
    put("TauFloor", tau0, 1); put("TtFloor", tt0, 1)
    line("   deviation from the datasheet curve at its two ends: " + ", ".join(
        f"{k} {s['dev'][0]*100:+.0f} % / {s['dev'][1]*100:+.0f} %" for k, s in sc.items()))
    # naive scalings of the table point to the slope of the circuit (150 C)
    a0 = pts["Nom"][0]
    put("QlinTab", QS_TAB*a0/A_TAB, 0); put("IlinTab", IRR_TAB*a0/A_TAB, 1); put("QpowTab", QS_TAB*(a0/A_TAB)**eq, 0)
    line(f"   naive scaling of the table point to {a0:.3f} A/us at {T_TAB:.0f} C: linear {QS_TAB*a0/A_TAB:.0f} uC / "
         f"{IRR_TAB*a0/A_TAB:.1f} A, with the slope of the curve {QS_TAB*(a0/A_TAB)**eq:.0f} uC")
    # scan of the bulk lifetime
    tscan = np.array([math.ceil(tau0)] + [t for t in (50, 60, 80, 100, 130, 160, 200, 250, 300, 400, 500, 700, 1000,
                                                      2000) if t > 1.15*tau0], float)
    qscan, start = [], None
    for tb in tscan:
        start = rcv.fit_point(1/tb, q_ref, i_ref, IF_TAB, start=start)
        qscan.append(rcv.recovery(a0, pts["Nom"][1], 1/tb, *start)["Qrr"])
    out["scan"] = dict(tau=tscan, Q=np.array(qscan)*fq)
    line("   Qrr at the circuit slope and Tj against tau_b: " + ", ".join(f"{t:.0f} us {q*fq:.0f} uC" for t, q in zip(tscan, qscan)))
    put("TauDes", float(np.interp(Q_DES*1e6, out["scan"]["Q"], tscan)), 0)
    # amplification of a QS band from the datasheet point to the circuit slope (central scenario)
    s = sc["central"]
    q_at = lambda a, IF, gb: rcv.recovery(a, IF, gb, s["TM"], s["k"])["Qrr"]
    d_wp = math.log(q_at(a0, pts["Nom"][1], s["gb"]/1.05)/q_at(a0, pts["Nom"][1], s["gb"]))/math.log(1.05)
    d_ds = math.log(q_at(A_TAB, IF_TAB, s["gb"]/1.05)/q_at(A_TAB, IF_TAB, s["gb"]))/math.log(1.05)
    put("SensWp", d_wp, 2); put("SensDs", d_ds, 2); put("BandFac", d_wp/d_ds, 1); put("BandTen", 10*d_wp/d_ds, 0)
    line(f"   central scenario, dlnQ/dln(tau_b): {d_wp:.2f} at the circuit slope, {d_ds:.2f} at the datasheet point: "
         f"a band of 10 % on QS becomes {10*d_wp/d_ds:.0f} % in the circuit")
    # design set and its Lauritzen-Ma parameters
    tau, TM = lm_params(Q_DES, I_DES, A_REF)
    tt = tau*TM/(tau + TM)
    out["lm"] = dict(tau=float(tau), TM=float(TM), tt=float(tt))
    s_des = 2*A_REF*Q_DES/I_DES**2 - 1
    line(f"   design set: {Q_DES*1e6:.0f} uC / {I_DES:.1f} A at {A_REF/1e6:.3f} A/us and {TJ_DES:.0f} C (softness {s_des:.2f}) -> "
         f"tau {tau*1e6:.1f} us, T_M {TM*1e6:.1f} us, t_tail {tt*1e6:.1f} us, t_a {I_DES/A_REF*1e6:.1f} us")
    put("Qdes", Q_DES*1e6, 0); put("Ides", I_DES, 1); put("Aref", A_REF/1e6, 3); put("Sdes", s_des, 2)
    put("LmTau", tau*1e6, 1); put("LmTm", TM*1e6, 1); put("LmTt", tt*1e6, 1); put("LmTa", I_DES/A_REF*1e6, 1)
    put("Uunb", U_UNB, 2); put("UunbPct", U_UNB*100, 0); put("TjDes", TJ_DES, 0)
    for pk in ("Max", "Ukmin"):
        q, i, ta, _ = lm_recovery(tau, TM, OP[pk]["alpha"])
        out[f"lm_{pk}"] = dict(Q=float(q), I=float(i))
        put(f"QlmAt{pk}", q*1e6, 0); put(f"IlmAt{pk}", i, 2)
        line(f"   design set at the slope of case {pk} ({OP[pk]['alpha']/1e6:.3f} A/us, long ramp): {q*1e6:.0f} uC / {i:.2f} A")
    put("RampCorr", W*float(tau)/math.tan(OP["Max"]["mu"])*100, 0)
    out["sc"], out["q_ref"], out["eq"], out["cq"], out["pts"] = sc, q_ref, eq, cq, pts
    return out


# ---------------------------------------------------------------- 4. commutation transient: two-mode model
def mcm(t, zeta, chi, w0, tt=0.0):
    """normalised voltage f = v/V across the R-C branch of a series R-L-C loop after the turn-off of a diode in
    parallel with the branch: step V, inductor current I0 at the peak of the reverse current, diode current decaying
    as I0*exp(-t/tt). zeta = R/(2 Z0), chi = I0 Z0/V, Z0 = sqrt(L/C), w0 = 1/sqrt(L C), theta = w0*tt.
    With the normalised snubber current j over x = w0*t:  j'' + 2 zeta j' + j = -(chi/theta^2) exp(-x/theta),
    j(0) = 0, j'(0) = 1 + chi/theta,  f = 1 - j' + (chi/theta) exp(-x/theta).  tt = 0: ideal snap-off (McMurray),
    f(0) = 2 zeta chi."""
    z = zeta if abs(zeta - 1.0) > 1e-9 else zeta + 1e-9
    wd = np.sqrt(complex(1 - z*z))
    x = w0*np.asarray(t, float)
    th = w0*tt
    if th < 1e-6:
        A, Bc = chi, (1 - z*chi)/wd
        dx = np.exp(-z*x)*((-z*A + wd*Bc)*np.cos(wd*x) + (-z*Bc - wd*A)*np.sin(wd*x))
        return 1 - dx.real
    K = -chi/(1 - 2*z*th + th*th)
    A = -K
    Bc = (1 + chi/th + K/th + z*A)/wd
    e = np.exp(-x/th)
    dx = np.exp(-z*x)*((-z*A + wd*Bc)*np.cos(wd*x) + (-z*Bc - wd*A)*np.sin(wd*x)) - K/th*e
    return (1 - dx + chi/th*e).real


def f_closed(x, zeta, chi, theta):
    """the same voltage in the form printed in the note (zeta < 1, theta > 0), with y = v_C/V:
    y = 1 + K exp(-x/theta) + exp(-zeta x)(A cos wd x + B sin wd x),  f = y + 2 zeta y'."""
    wd = math.sqrt(1 - zeta*zeta)
    K = chi*theta/(1 - 2*zeta*theta + theta*theta)
    A = -(1 + K)
    B = (zeta*A + K/theta)/wd
    x = np.asarray(x, float)
    e, d, c, s = np.exp(-x/theta), np.exp(-zeta*x), np.cos(wd*x), np.sin(wd*x)
    y = 1 + K*e + d*(A*c + B*s)
    dy = -K/theta*e + d*((wd*B - zeta*A)*c - (wd*A + zeta*B)*s)
    return y + 2*zeta*dy


def snap_off(zeta, chi, theta=0.0):
    """peak F of the normalised voltage and its normalised time w0*t; theta = w0*t_tail."""
    x = np.linspace(0, 4*math.pi, 20001)
    f = mcm(x, zeta, chi, 1.0, theta)
    j = int(np.argmax(f))
    return float(f[j]), float(x[j])


def turn_off(o, tau, TM):
    """turn-off of the Lauritzen-Ma diode on the ramp of the operating point o: peak reverse current, recovered
    charge, time t_a from the current zero to the peak, tail time constant and commutation voltage at the peak."""
    Q, Irr, ta, tt = (float(x) for x in lm_recovery(tau, TM, o["alpha"]))
    return dict(Q=Q, Irr=Irr, ta=ta, tt=tt, V0s=o["Vhat"]*math.sin(o["mu"] + W*ta))


def modes(n=N, Rsn=RSN, Csn=CSN, Lc=LC):
    """two modes of the snubbed bridge with a current-stiff DC side; string values Rs = n Rsn, Cs = Csn/n.
    mode 1: recovering phase against the other two, Zs + 3 s Lc = 0; mode 2: DC terminals, Zs + 5 s Lc = 0."""
    Rs, Cs = n*Rsn, Csn/n
    m = {}
    for j, k in ((1, 3.0), (2, 5.0)):
        Z0 = math.sqrt(Lc/(k*Cs))
        m[j] = dict(Z0=Z0, zeta=Rs/k/(2*Z0), w0=1/math.sqrt(k*Lc*Cs))
    Z0 = math.sqrt(2*Lc/(5/3*Cs))          # single loop of the 5/3 rule: 2 Lc, 3/5 Rs, 5/3 Cs
    m["rule"] = dict(Z0=Z0, zeta=0.6*Rs/(2*Z0), w0=1/math.sqrt(2*Lc*5/3*Cs))
    return m


def transient(V0, Irr, m, t, tt=0.0):
    """voltages after the turn-off of an arm (peak reverse current Irr, commutation voltage V0, tail time constant
    tt), referred to V0: own = recovering arm, dc = excess on the arms across the DC terminals, rule = single loop
    of the 5/3 rule."""
    f1 = mcm(t, m[1]["zeta"], 2*Irr*m[1]["Z0"]/V0, m[1]["w0"], tt)
    f2 = mcm(t, m[2]["zeta"], 2*Irr*m[2]["Z0"]/V0, m[2]["w0"], tt)
    fr = mcm(t, m["rule"]["zeta"], Irr*m["rule"]["Z0"]/V0, m["rule"]["w0"], tt)
    return dict(own=0.75*f1 + 0.25*f2, dc=0.5*(f2 - 1), rule=fr, f1=f1, f2=f2)


def stack_peak_stiff(o, h, m, tail=True):
    """highest arm voltage with a current-stiff DC side: step +V0s/2 at theta = 60 deg + mu + w t_a, near the
    crest; h = turn_off(...). Returns the peak and its delay from the step."""
    t = np.linspace(0, 1.5e-3, 6001)
    v = (o["Vhat"]*np.sin(math.pi/3 + o["mu"] + W*(h["ta"] + t))
         + h["V0s"]*transient(h["V0s"], h["Irr"], m, t, h["tt"] if tail else 0.0)["dc"])
    j = int(np.argmax(v))
    return float(v[j]), float(t[j])


def section_modes(rec):
    line("\n4. Commutation transient: two-mode model of the snubbed bridge (current-stiff DC side)")
    o, m = OP["Max"], modes()
    h = turn_off(o, rec["lm"]["tau"], rec["lm"]["TM"])
    Irr, V0s = h["Irr"], h["V0s"]
    line(f"   turn-off at grid +{(K_GRID-1)*100:.0f} %: Irr {Irr:.2f} A, t_a {h['ta']*1e6:.1f} us, t_tail {h['tt']*1e6:.1f} us, "
         f"commutation voltage at the peak V0s {V0s/1e3:.2f} kV (V0 {o['V0']/1e3:.2f} kV at the end of the ideal overlap)")
    put("Rs", N*RSN, 0); put("Cs", f"{CSN/N*1e9:.4g}"); put("VzeroS", V0s/1e3, 2); put("VzeroSRatio", V0s/o["V0"], 2)
    put("IrrHand", Irr, 2); put("TaHand", h["ta"]*1e6, 0); put("QrrHand", h["Q"]*1e6, 0)
    # the weak diode (charge Qrr (1 - u), current Irr sqrt(1 - u)) blocks earlier by the difference of the times t_a
    dta = h["ta"]*(1 - math.sqrt(1 - U_UNB))
    put("TaDiff", dta*1e6, 0); put("IrrWeakHand", Irr*math.sqrt(1 - U_UNB), 1)
    # until the others block, the string current moves from the weak diode (tail) to its snubber
    i_step = Irr*(1 - math.sqrt(1 - U_UNB)*math.exp(-dta/h["tt"]))
    h["i_step"] = i_step
    put("IsnWeakHand", i_step, 1); put("IsnWeakShare", i_step/Irr, 2); put("VstepHand", RSN*i_step/1e3, 2)
    line(f"   the weak diode (u = {U_UNB}) blocks {dta*1e6:.1f} us before the others, at {Irr*math.sqrt(1 - U_UNB):.2f} A; "
         f"its snubber then takes {i_step:.2f} A ({i_step/Irr:.2f} Irr): resistive step {RSN*i_step/1e3:.2f} kV")
    for j, nm in ((1, "One"), (2, "Two"), ("rule", "Rule")):
        chi = (1 if j == "rule" else 2)*Irr*m[j]["Z0"]/V0s
        th = m[j]["w0"]*h["tt"]
        F0, _ = snap_off(m[j]["zeta"], chi)
        F_, x = snap_off(m[j]["zeta"], chi, th)
        m[j].update(chi=chi, F=F_, F0=F0, tpk=x/m[j]["w0"], theta=th)
        line(f"   mode {j}: Z0 {m[j]['Z0']:.0f} Ohm, f0 {m[j]['w0']/2/math.pi/1e3:.2f} kHz, zeta {m[j]['zeta']:.3f}, chi {chi:.2f}, "
             f"theta {th:.2f}: F {F_:.3f} after {x/m[j]['w0']*1e6:.0f} us (ideal snap-off {F0:.3f})")
        put(f"Zz{nm}", m[j]["Z0"], 0); put(f"Fz{nm}", m[j]["w0"]/2/math.pi/1e3, 2); put(f"Zeta{nm}", m[j]["zeta"], 3)
        put(f"Chi{nm}", chi, 2); put(f"F{nm}", F_, 2); put(f"Fideal{nm}", F0, 2); put(f"Tpk{nm}", x/m[j]["w0"]*1e6, 0)
        put(f"Theta{nm}", th, 2)
    t = np.linspace(0, 1.5e-3, 6001)
    Fown = float(transient(V0s, Irr, m, t, h["tt"])["own"].max())
    Fown0 = float(transient(V0s, Irr, m, t)["own"].max())
    Vs, tpk = stack_peak_stiff(o, h, m)
    Vs0, _ = stack_peak_stiff(o, h, m, tail=False)
    line(f"   recovering arm: peak {Fown:.3f} V0s = {Fown*V0s/1e3:.2f} kV (ideal snap-off {Fown0:.3f}, 5/3 rule "
         f"{m['rule']['F']:.3f}); arms across the DC terminals: +{(m[2]['F']-1)/2*V0s/1e3:.2f} kV; highest arm voltage "
         f"{Vs/1e3:.2f} kV = {Vs/o['Vhat']:.3f} of the crest, {tpk*1e6:.0f} us after the step (ideal snap-off {Vs0/1e3:.2f} kV)")
    put("Fown", Fown, 2); put("FownIdeal", Fown0, 2); put("VownPk", Fown*V0s/1e3, 1)
    put("DvDc", (m[2]["F"] - 1)/2*V0s/1e3, 2); put("VsStiffHand", Vs/1e3, 2); put("KtrHand", Vs/o["Vhat"], 2)
    put("VsStiffIdeal", Vs0/1e3, 2)
    rise = sorted({f"{(m[j]['F']/m[j]['F0'] - 1)*100:.0f}" for j in (1, 2)}, key=float)
    put("TailRise", " to ".join(rise))
    # damping of mode 2 against Rsn: resistance of the lowest highest-arm voltage and of zeta2 = 0.5
    R_half = 10*0.5*m[2]["Z0"]/N
    Rl = np.geomspace(0.2*R_half, 6*R_half, 141)
    pk = np.array([stack_peak_stiff(o, h, modes(Rsn=r))[0] for r in Rl])
    j = int(np.argmin(pk))
    near = Rl[pk <= 1.01*pk[j]]
    put("RsnOptTwo", Rl[j], 0); put("RsnOptLo", near.min(), 0); put("RsnOptHi", near.max(), 0)
    put("VsOptTwo", pk[j]/1e3, 2); put("RsnHalfZeta", R_half, 0)
    put("RsnStep", RSN*Irr/1e3, 2)
    line(f"   lowest highest-arm voltage {pk[j]/1e3:.2f} kV with Rsn = {Rl[j]:.0f} Ohm (within 1 % from {near.min():.0f} to "
         f"{near.max():.0f} Ohm); zeta2 = 0.5 with {R_half:.0f} Ohm; Rsn*Irr = {RSN*Irr/1e3:.2f} kV")
    put("LdStiff", 2*LC*1e3, 0); put("LdStiffSim", LD_STIFF, 0)
    put("VsSimple", (o["Vhat"] + 0.5*V0s*(m[2]["F"] - 1))/1e3, 2)
    # peak factor F(zeta, chi, theta) of the single loop: table of the note
    ZL, XL, TH = (0.2, 0.3, 0.4, 0.5, 0.7, 1.0), (0.5, 1.0, 1.5, 2.0, 3.0), 0.4
    rows = []
    line(f"   peak factor F of the loop (rows zeta; columns chi = {', '.join(f'{c:g}' for c in XL)}; snap-off | theta = {TH})")
    for z in ZL:
        a0 = [snap_off(z, c)[0] for c in XL]
        a1 = [snap_off(z, c, TH)[0] for c in XL]
        rows.append(f"{z:.1f} & " + " & ".join(f"{v:.2f}" for v in a0 + a1) + r"\\")
        line(f"      {z:.1f}: " + " ".join(f"{v:5.2f}" for v in a0) + "  | " + " ".join(f"{v:5.2f}" for v in a1))
    put("FtabRows", "\n".join(rows))
    x = np.linspace(0, 4*math.pi, 4001)
    err = max(float(np.max(np.abs(f_closed(x, z, c, th_) - mcm(x, z, c, 1.0, th_))))
              for z in (0.2, 0.3, 0.5, 0.7) for c in XL for th_ in (0.1, 0.4, 1.0))
    line(f"   closed form printed in the note against mcm(): largest difference {err:.1e}")
    assert err < 1e-9
    return m, h


# ---------------------------------------------------------------- 5. simulations
def case(side="R", **kw):
    c = dict(Vll=K_GRID*U_N, Lc=LC, Rload=R_LOAD, Ld=LD_R if side == "R" else LD_L, Q=Q_DES, I=I_DES, u=U_UNB,
             Rb=RB, Rsn=RSN, Csn=CSN, Roff_w=R_NOLEAK, Roff_n=ROFF_LIM, cw=0.0, cn=0.0, rw=0.0, rn=0.0, short=0,
             Rx=1.0, Cx=0.0)
    c.update(kw)
    return {k: (float(v) if not isinstance(v, int) else v) for k, v in c.items()}


def run(name, cases, n=N, dt=DT, periods=PERIODS, rec=(), rec_every=2):
    """simulate a list of cases in one batch; the weak diode is level 0 of every arm."""
    key = hashlib.md5(json.dumps([cases, n, dt, periods, list(rec), rec_every], sort_keys=True).encode()).hexdigest()[:12]
    cache = os.environ.get("DIODE_STACK_CACHE")
    path = os.path.join(cache, f"{name}_{key}.pkl") if cache else None
    if path and os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    B = len(cases)
    get = lambda k: np.array([c[k] for c in cases], float)
    Q, I, u = get("Q"), get("I"), get("u")
    tau_n, TM_n = lm_params(Q, I, A_REF)
    tau_w, TM_w = lm_params(Q*(1 - u), I*np.sqrt(1 - u), A_REF)
    Roff = level_array(B, n, get("Roff_w"), get("Roff_n"))
    for j, c in enumerate(cases):
        if c["short"]:
            Roff[j, 0, 1] = 1e-3                    # one nominal level of arm 0 failed short
    t0 = time.time()
    r = simulate(get("Vll"), F, get("Lc"), get("Rload"), get("Ld"), level_array(B, n, tau_w, tau_n),
                 level_array(B, n, TM_w, TM_n), Roff, level_array(B, n, get("Rb")*(1 + get("rw")), get("Rb")*(1 + get("rn"))),
                 get("Rsn")[:, None, None], level_array(B, n, get("Csn")*(1 + get("cw")), get("Csn")*(1 + get("cn"))),
                 VT0=VT0, rT=RT, iD0=[id_resistive(c["Vll"], c["Lc"], c["Rload"]) for c in cases], Rx=get("Rx"),
                 Cx=get("Cx"), periods=periods, dt=dt, rec=rec, rec_every=rec_every)
    print(f"   [batch {name}: {B} cases, n = {n}, dt = {dt*1e6:.2f} us, {time.time() - t0:.0f} s]")
    if path:
        os.makedirs(cache, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(r, fh)
    return r


def summary(r, c, arm=ARM, nom=1):
    """key figures of case c: weak level (0) and a nominal level of the arm."""
    s = dict(Vdc=float(r["Vdc"][c]), Idc=float(r["Idc"][c]), Iline=float(r["Iline_rms"][c, 0]),
             Vs=float(r["Vstack_pk"][c, arm]), P_res=float((r["P_Rsn"][c, arm] + r["P_Rb"][c, arm]).sum()),
             P_x=float(r["P_x"][c]))
    for nm, j in (("w", 0), ("n", nom)):
        for k in ("Vpk", "Vrms", "Vmean", "P_Rsn", "P_Rb", "Isn_rms", "Isn_pk", "Irrm", "Qrr", "IFav", "IFrms", "IFpk",
                  "P_rec"):
            s[k + "_" + nm] = float(r[k][c, arm, j])
    s["dQ"] = s["Qrr_n"] - s["Qrr_w"]
    # time of the peak of the weak level, counted from the natural commutation at which the arm hands over
    s["tpk_w"] = float(np.mod(r["Tpk"][c, arm, 0] - t_nat(arm), 1.0/F))
    return s


def roll(Wv, t0):
    """recorded period as a continuous window starting at the time t0 (periodic steady state)."""
    t = Wv["t"]
    T = 1.0/F
    j = int(np.argmin(np.abs(np.mod(t - t0, T))))
    out = {k: np.roll(v, -j, axis=0) for k, v in Wv.items()}
    out["t"] = np.mod(out["t"] - t0 + 1e-12, T) - 1e-12
    return out


T_NAT = {0: 150.0, 1: 270.0, 2: 30.0, 3: 330.0, 4: 90.0, 5: 210.0}    # natural commutation instants, electrical deg


def t_nat(arm):
    """time of the natural commutation at which the arm hands over, within a period (e_a = sin(w t))."""
    return math.radians(T_NAT[arm])/W


def section_simulations(rec):
    line(f"\n5. Simulations (grid +{(K_GRID-1)*100:.0f} % unless stated; weak level / nominal level)")
    A = {"R": case("R"), "L": case("L"),
         "R_ctol": case("R", cw=-TOL_C, cn=TOL_C), "L_ctol": case("L", cw=-TOL_C, cn=TOL_C),
         "R_tol": case("R", cw=-TOL_C, cn=TOL_C, rw=TOL_RB, rn=-TOL_RB),
         "L_tol": case("L", cw=-TOL_C, cn=TOL_C, rw=TOL_RB, rn=-TOL_RB),
         "R_bal": case("R", u=0.0, Roff_w=ROFF_LIM), "L_bal": case("L", u=0.0, Roff_w=ROFF_LIM),
         "R_ukmin": case("R", Lc=LC_MIN), "L_ukmin": case("L", Lc=LC_MIN),
         "R_nom": case("R", Vll=U_N), "L_nom": case("L", Vll=U_N),
         "R_short": case("R", short=1), "L_short": case("L", short=1),
         "R_short_ctol": case("R", short=1, cw=-TOL_C, cn=TOL_C), "L_short_ctol": case("L", short=1, cw=-TOL_C, cn=TOL_C),
         "R_half": case("R", Rload=2*R_LOAD), "L_half": case("L", Rload=2*R_LOAD),
         "R_noleak": case("R", Roff_n=R_NOLEAK), "R_leakw": case("R", Roff_w=ROFF_LIM, Roff_n=R_NOLEAK)}
    names = list(A)
    recs = [names.index(k) for k in ("R", "L", "R_nom", "L_nom")]
    rA = run("A", [A[k] for k in names], rec=recs)
    S = {}
    for c, k in enumerate(names):
        short = "short" in k
        S[k] = summary(rA, c, arm=0 if short else ARM, nom=2 if short else 1)
        S[k]["wave"] = rA["wave"].get(c)
    # balanced stack in the current-stiff limit, without Rb and leakage, for the check of the two-mode model:
    # design diode and a snappy diode with the same peak reverse current
    V = {"S_snap": case("L", Ld=LD_STIFF, Q=Q_SNAP, u=0.0, Rb=1e12, Roff_n=R_NOLEAK),
         "S_soft": case("L", Ld=LD_STIFF, u=0.0, Rb=1e12, Roff_n=R_NOLEAK)}
    vn = list(V)
    rV = run("V", [V[k] for k in vn], dt=DT_FINE, rec=[0, 1], rec_every=2)
    for c, k in enumerate(vn):
        S[k] = summary(rV, c)
        S[k]["wave"] = rV["wave"].get(c)
    line("   case          Vdc[kV] Idc[A]  Irrm[A]      Qrr[uC]      Vpk[kV]      Vrms[kV]     P_Rsn[W]    P_Rb[W]   "
         "stack pk[kV]  resistors/stack[W]")
    for k in names + vn:
        s = S[k]
        line(f"   {k:12s} {s['Vdc']/1e3:7.2f} {s['Idc']:6.1f}  {s['Irrm_w']:5.2f}/{s['Irrm_n']:5.2f}  {s['Qrr_w']*1e6:6.1f}/{s['Qrr_n']*1e6:6.1f} "
             f"{s['Vpk_w']/1e3:6.3f}/{s['Vpk_n']/1e3:5.3f} {s['Vrms_w']/1e3:6.3f}/{s['Vrms_n']/1e3:5.3f} "
             f"{s['P_Rsn_w']:5.1f}/{s['P_Rsn_n']:5.1f} {s['P_Rb_w']:5.1f}/{s['P_Rb_n']:5.1f}   {s['Vs']/1e3:6.2f}     {s['P_res']:6.0f}")
    tags = {"R": "R", "L": "L", "R_ctol": "RCtol", "L_ctol": "LCtol", "R_tol": "RTol", "L_tol": "LTol", "R_bal": "RBal",
            "L_bal": "LBal", "R_ukmin": "RUkmin", "L_ukmin": "LUkmin", "R_nom": "RNom", "L_nom": "LNom",
            "R_short": "RShort", "L_short": "LShort", "R_short_ctol": "RShortCtol", "L_short_ctol": "LShortCtol",
            "R_half": "RHalf", "L_half": "LHalf", "R_noleak": "RNoleak",
            "R_leakw": "RLeakw", "S_snap": "StiffSnap", "S_soft": "StiffSoft"}
    for k, tg in tags.items():
        s = S[k]
        put(f"{tg}Vdc", s["Vdc"]/1e3, 2); put(f"{tg}Idc", s["Idc"], 1); put(f"{tg}Iline", s["Iline"], 1)
        put(f"{tg}Vs", s["Vs"]/1e3, 2); put(f"{tg}Pres", s["P_res"], 0); put(f"{tg}Dq", s["dQ"]*1e6, 0)
        put(f"{tg}KtrSim", s["Vs"]/(math.sqrt(2)*A.get(k, V.get(k))["Vll"]), 3)
        for nm, cap in (("w", "W"), ("n", "N")):
            put(f"{tg}Irrm{cap}", s["Irrm_" + nm], 2); put(f"{tg}Qrr{cap}", s["Qrr_" + nm]*1e6, 0)
            put(f"{tg}Vpk{cap}", s["Vpk_" + nm]/1e3, 2); put(f"{tg}Vrms{cap}", s["Vrms_" + nm]/1e3, 2)
            put(f"{tg}Vmean{cap}", s["Vmean_" + nm]/1e3, 2)
            put(f"{tg}Util{cap}", s["Vpk_" + nm]/VRRM*100, 0); put(f"{tg}Prsn{cap}", s["P_Rsn_" + nm], 1)
            put(f"{tg}Prb{cap}", s["P_Rb_" + nm], 1); put(f"{tg}Isnpk{cap}", s["Isn_pk_" + nm], 2)
            put(f"{tg}Isnrms{cap}", s["Isn_rms_" + nm], 2); put(f"{tg}Prec{cap}", s["P_rec_" + nm], 1)
        put(f"{tg}Ifav", s["IFav_n"], 1); put(f"{tg}Ifrms", s["IFrms_n"], 1); put(f"{tg}Ifpk", s["IFpk_n"], 1)
    # sweeps with the reference number of devices, in one batch
    sw, idx = [], {}

    def add(tag, **kw):
        idx[tag] = len(sw)
        sw.append(case(kw.pop("side"), **kw))
    QL, UL, CL, RL_ = tuple(x*Q_DES for x in Q_SWEEP), U_SWEEP, C_SWEEP, R_SWEEP
    for sd in "RL":
        for q in QL:
            for u in UL:
                add(("Q", sd, q, u), side=sd, Q=q, I=math.sqrt(2*A_REF*q/(1 + rec["s_des"])), u=u)
            add(("QT", sd, q), side=sd, Q=q, I=math.sqrt(2*A_REF*q/(1 + rec["s_des"])), cw=-TOL_C, cn=TOL_C)
    for sd in "RL":
        for c_ in CL:
            for r_ in RL_:
                add(("N", sd, c_, r_), side=sd, Csn=c_, Rsn=r_)
        for rb in (RB/2, 2*RB):
            add(("B", sd, rb), side=sd, Rb=rb)
    LDL = tuple(sorted({LD_R, 2*LC, LD_L, LD_STIFF} | set(LD_MORE)))
    for ld in LDL:
        add(("D", ld), side="R", Ld=ld)
    for rx, cx in DAMP:
        add(("X", rx, cx), side="L", Rx=rx, Cx=cx)
    for c_, r_ in OPTIONS:
        add(("O", c_, r_), side="L", Csn=c_, Rsn=r_, cw=-TOL_C, cn=TOL_C)
    rS = run("S", sw)
    G = lambda tag: summary(rS, idx[tag])
    sweeps = dict(QL=QL, UL=UL, CL=CL, RL=RL_, LDL=LDL, DAMP=DAMP, G=G)
    # balanced stack with a current-stiff DC side against Qrr (validation of the two-mode model)
    rT = run("T", [case("L", Ld=LD_STIFF, Q=q, I=math.sqrt(2*A_REF*q/(1 + rec["s_des"])), u=0.0, Roff_w=ROFF_LIM)
                   for q in QL])
    sweeps["stiff"] = [summary(rT, c) for c in range(len(QL))]
    # other numbers of diodes in series
    sweeps["n"] = {N: {"R": S["R"], "L": S["L"], "R_ctol": S["R_ctol"], "L_ctol": S["L_ctol"]}}
    for n in (N - 1, N + 1, N + 2):
        ks = ("R", "L", "R_ctol", "L_ctol")
        cs = [A[k] for k in ks]
        if n == N + 1:                                      # option of the note: one more level and a larger snubber
            cs.append(case("L", Csn=OPTIONS[0][0], Rsn=OPTIONS[0][1], cw=-TOL_C, cn=TOL_C))
        r = run(f"n{n}", cs, n=n)
        sweeps["n"][n] = {k: summary(r, c) for c, k in enumerate(ks)}
        if n == N + 1:
            sweeps["n"][n]["L_opt"] = summary(r, len(ks))
    return S, sweeps


# ---------------------------------------------------------------- 6. hand formulas against the simulations
def weak_peak(Vs, dQ, t_pk, n=N, Csn=CSN, Rb=RB, c=0.0, dIlk=0.0):
    """charge model of the weak level: the others share the string voltage through equal capacitors, the weak one
    carries the offset charge dQ, which decays with Rb*Csn; c = capacitor tolerance (weak -c, others +c)."""
    tb = Rb*Csn
    dq = dQ*math.exp(-t_pk/tb) + tb*dIlk*(1 - math.exp(-t_pk/tb))
    return ((1 + c)*Vs + (n - 1)*dq/Csn)/(n - (n - 2)*c)


def p_rsn_hand(o, Irr, n=N, Csn=CSN, Rsn=RSN, Vrms=None, stiff=True):
    """mean power of the snubber resistor of a nominal level: energy of the four steps of a period (7/8 Cs V0s^2 per
    arm with a current-stiff DC side, the share 9/16 of mode 1 with a resistive load), energy trapped in the
    commutation inductance (Lc Irr^2, share 3/4 of mode 1) and line-frequency current."""
    kc, kl = (7/8, 1.0) if stiff else (9/16, 0.75)
    V0s = o["Vhat"]*math.sin(o["mu"] + W*Irr/o["alpha"])
    p = F*(kc*Csn*(V0s/n)**2 + kl*o["Lc"]*Irr**2/n)
    if Vrms is not None:
        p += Rsn*(W*Csn*Vrms)**2
    return p


def section_hand(S, sweeps, rec, m, h):
    line("\n6. Hand formulas against the simulations")
    o = OP["Max"]
    Irr, Q = h["Irr"], h["Q"]
    dQ = U_UNB*Q
    tb = RB*CSN
    tpkR = (math.pi/2 - o["mu"])/W
    VsL, tF = stack_peak_stiff(o, h, m)
    tpkL = (math.pi/3)/W + tF
    dI = float(leak(V_LEV, TJ_DES))
    hand = {"R": dict(Vs=o["Vhat"], tpk=tpkR), "L": dict(Vs=VsL, tpk=tpkL)}
    put("DqHand", dQ*1e6, 0); put("DvHand", dQ/CSN/1e3, 2); put("TpkR", tpkR*1e3, 1); put("TpkL", tpkL*1e3, 1)
    put("DecayR", math.exp(-tpkR/tb), 2); put("DecayL", math.exp(-tpkL/tb), 2)
    put("LeakTerm", (N - 1)/N*RB*dI*(1 - math.exp(-tpkR/tb)), 0)
    put("CapShare", (1 + TOL_C)/(N - (N - 2)*TOL_C), 3)
    put("CapShareRel", ((1 + TOL_C)*N/(N - (N - 2)*TOL_C) - 1)*100, 1)
    dq_pk = dQ*math.exp(-tpkR/tb) + tb*dI*(1 - math.exp(-tpkR/tb))
    cmin = (N - 1)*dq_pk/((N - (N - 2)*TOL_C)*K_U*VRRM - (1 + TOL_C)*o["Vhat"])
    put("CsnMinHandR", cmin*1e9, 0)
    line(f"   smallest Csn of the charge model with the tolerance (R): {cmin*1e9:.0f} nF for a charge difference of "
         f"{dq_pk*1e6:.0f} uC at the crest")
    for sd in "RL":
        hd = hand[sd]
        hd["Vw"] = weak_peak(hd["Vs"], dQ, hd["tpk"])
        hd["Vw_lk"] = weak_peak(hd["Vs"], dQ, hd["tpk"], dIlk=dI)
        hd["Vw_tol"] = weak_peak(hd["Vs"], dQ, hd["tpk"], c=TOL_C, dIlk=dI)
        hd["Vn"] = (hd["Vs"] - hd["Vw_lk"])/(N - 1)
        hd["P_n"] = p_rsn_hand(o, Irr, Vrms=S[sd]["Vrms_n"], stiff=sd == "L")
        hd["P_w"] = hd["P_n"] + F*RSN*Irr*dQ
        s, st = S[sd], S[sd + "_ctol"]
        line(f"   {sd}: stack peak hand {hd['Vs']/1e3:.2f} kV / sim {s['Vs']/1e3:.2f} kV; weak level hand {hd['Vw_lk']/1e3:.2f} kV "
             f"(without leakage {hd['Vw']/1e3:.2f}) / sim {s['Vpk_w']/1e3:.2f} kV; with Csn -+{TOL_C*100:.0f} % hand "
             f"{hd['Vw_tol']/1e3:.2f} kV / sim {st['Vpk_w']/1e3:.2f} kV; P_Rsn nominal hand {hd['P_n']:.1f} W / sim "
             f"{s['P_Rsn_n']:.1f} W, weak hand {hd['P_w']:.1f} W / sim {s['P_Rsn_w']:.1f} W")
        put(f"Hand{sd}Vs", hd["Vs"]/1e3, 2); put(f"Hand{sd}Vw", hd["Vw_lk"]/1e3, 2); put(f"Hand{sd}VwNoLk", hd["Vw"]/1e3, 2)
        put(f"Hand{sd}VwTol", hd["Vw_tol"]/1e3, 2); put(f"Hand{sd}Pn", hd["P_n"], 1); put(f"Hand{sd}Pw", hd["P_w"], 1)
        put(f"Hand{sd}UtilTol", hd["Vw_tol"]/VRRM*100, 0); put(f"Hand{sd}Vn", hd["Vn"]/1e3, 2)
        put(f"Hand{sd}DevVw", (hd["Vw_lk"]/s["Vpk_w"] - 1)*100, 1); put(f"Hand{sd}DevTol", (hd["Vw_tol"]/st["Vpk_w"] - 1)*100, 1)
    put("PwExtra", F*RSN*Irr*dQ, 1); put("EstepLevel", 7/8*CSN*(h["V0s"]/N)**2*1e3, 0)
    put("EindLevel", LC*Irr**2/N*1e3, 0)
    for sd in "RL":
        put(f"PwExtraSim{sd}", S[sd]["P_Rsn_w"] - S[sd]["P_Rsn_n"], 1)
    # largest deviations of the hand values from the simulation (table of the note)
    dv = [hand[sd]["Vs"]/S[sd]["Vs"] - 1 for sd in "RL"] + [hand[sd]["Vw_lk"]/S[sd]["Vpk_w"] - 1 for sd in "RL"] \
        + [hand[sd]["Vw_tol"]/S[sd + "_ctol"]["Vpk_w"] - 1 for sd in "RL"]
    dp = [hand[sd]["P_n"]/S[sd]["P_Rsn_n"] - 1 for sd in "RL"] + [hand[sd]["P_w"]/S[sd]["P_Rsn_w"] - 1 for sd in "RL"]
    put("HandVdevLo", min(dv)*100, 0); put("HandVdevHi", max(dv)*100, 0)
    put("HandPdevLo", min(dp)*100, 0); put("HandPdevHi", max(dp)*100, 0)
    put("IrrHandRatio", (Irr/S["L"]["Irrm_n"] - 1)*100, 0)
    line(f"   hand values against the simulation: voltages from {min(dv)*100:+.1f} % to {max(dv)*100:+.1f} %, powers from "
         f"{min(dp)*100:+.0f} % to {max(dp)*100:+.0f} %; long-ramp Irr {Irr:.2f} A against {S['R']['Irrm_n']:.2f} A (R) and "
         f"{S['L']['Irrm_n']:.2f} A (L) in the simulation")
    # the same formulas with the simulated dQ, stack peak and Irrm over the Qrr sweep (structure of the model)
    G, dev = sweeps["G"], {"R": [], "L": []}
    for sd in "RL":
        for q in sweeps["QL"]:
            for u in sweeps["UL"]:
                s = G(("Q", sd, q, u))
                v = weak_peak(s["Vs"], s["dQ"], hand[sd]["tpk"], dIlk=dI if u > 0 else 0.0)
                dev[sd].append(v/s["Vpk_w"] - 1)
        line(f"   {sd}: charge model with the simulated stack peak and dQ over the Qrr-u sweep: deviation of the weak-level "
             f"peak from {min(dev[sd])*100:+.1f} % to {max(dev[sd])*100:+.1f} %")
        put(f"ChargeDevLo{sd}", min(dev[sd])*100, 1); put(f"ChargeDevHi{sd}", max(dev[sd])*100, 1)
    # snubber power over the network sweep: formula with the simulated Irrm
    CLi = sweeps["CL"][1:]                                  # without the smallest capacitor
    RLi = [r_ for r_ in sweeps["RL"] if r_ <= 1.5*RSN]      # resistors up to 1.5 times the reference value
    put("PrsnCLo", CLi[0]*1e9, 0); put("PrsnRHi", RLi[-1], 0)
    d_all, dz_all = [], []
    for sd in "RL":
        d, dz = [], []
        line(f"   {sd}: P_Rsn of a nominal level, formula with the simulated Irrm / simulation - 1 [%] (rows Csn, columns Rsn)")
        for c_ in sweeps["CL"]:
            row = []
            for r_ in sweeps["RL"]:
                s = G(("N", sd, c_, r_))
                x = p_rsn_hand(o, s["Irrm_n"], Csn=c_, Rsn=r_, Vrms=s["Vrms_n"], stiff=sd == "L")/s["P_Rsn_n"] - 1
                row.append(x); d.append(x)
                if c_ in CLi and r_ in RLi:
                    dz.append(x)
            line(f"      {c_*1e9:4.0f} nF: " + " ".join(f"{x*100:+5.0f}" for x in row))
        line(f"      whole sweep from {min(d)*100:+.0f} % to {max(d)*100:+.0f} %; from {CLi[0]*1e9:.0f} nF and up to "
             f"{RLi[-1]:.0f} Ohm from {min(dz)*100:+.0f} % to {max(dz)*100:+.0f} %")
        d_all += d; dz_all += dz
    put("PrsnDevLo", min(dz_all)*100, 0); put("PrsnDevHi", max(dz_all)*100, 0); put("PrsnDevTop", max(d_all)*100, 0)
    # AN6531 Eq. (3): smallest snubber capacitor for the charge difference
    for nm, VD in (("Des", K_U*VRRM), ("Rated", VRRM)):
        den = (1 - TOL_C)*(N*VD - K_GRID*math.sqrt(2)*U_N)
        put(f"CsMin{nm}", (N - 1)*dQ/den*1e9, 0); put(f"DqAdm{nm}", CSN*den/(N - 1)*1e6, 0)
        line(f"   AN6531 (3), V_D = {VD/1e3:.1f} kV: Cs >= {(N-1)*dQ/den*1e9:.0f} nF for dQ = {dQ*1e6:.0f} uC; with "
             f"{CSN*1e9:.0f} nF dQ <= {CSN*den/(N-1)*1e6:.0f} uC")
    # validation of the two-mode model in the stiff limit
    res = {}
    for k, tg in (("S_snap", "Snap"), ("S_soft", "Soft")):
        Wv = S[k]["wave"]
        ia = Wv["ia"][:, ARM]
        j = int(np.argmin(ia))
        ts, Irr_s = Wv["t"][j], -float(ia[j])
        e = math.sqrt(2/3)*K_GRID*U_N*np.sin(W*Wv["t"][:, None] - np.array([0, 2*math.pi/3, 4*math.pi/3]))
        V0s = float(e[j, 2] - e[j, 1])
        vown = -Wv["v"][:, ARM].sum(-1)
        vdc = -Wv["v"][:, 5].sum(-1)
        sl = slice(j, j + int(1.2e-3/(Wv["t"][1] - Wv["t"][0])))
        t = Wv["t"][sl] - ts
        tt_s = rec["lm"]["tt"] if k == "S_soft" else float(np.prod(lm_params(Q_SNAP, I_DES, A_REF))/np.sum(lm_params(Q_SNAP, I_DES, A_REF)))
        tr = transient(V0s, Irr_s, m, t, tt_s)
        tr0 = transient(V0s, Irr_s, m, t)
        mod_own = (e[sl, 2] - e[sl, 1]) + V0s*(tr["own"] - 1)
        mod_dc = (e[sl, 2] - e[sl, 0]) + V0s*tr["dc"]
        mod_rule = (e[sl, 2] - e[sl, 1]) + V0s*(tr["rule"] - 1)
        id_own = (e[sl, 2] - e[sl, 1]) + V0s*(tr0["own"] - 1)
        id_dc = (e[sl, 2] - e[sl, 0]) + V0s*tr0["dc"]
        res[k] = dict(t=t, own=vown[sl], dc=vdc[sl], mod_own=mod_own, mod_dc=mod_dc, mod_rule=mod_rule, id_own=id_own,
                      id_dc=id_dc, Irr=Irr_s, V0=V0s)
        line(f"   stiff limit, {k}: Irr {Irr_s:.2f} A, V0s {V0s/1e3:.2f} kV, t_tail {tt_s*1e6:.1f} us; recovering arm peak sim "
             f"{vown[sl].max()/1e3:.2f} kV / model {mod_own.max()/1e3:.2f} kV (ideal snap-off {id_own.max()/1e3:.2f}, 5/3 rule "
             f"{mod_rule.max()/1e3:.2f}); arm across the DC terminals sim {vdc[sl].max()/1e3:.2f} kV / model "
             f"{mod_dc.max()/1e3:.2f} kV (ideal snap-off {id_dc.max()/1e3:.2f})")
        put(f"Mod{tg}OwnIdeal", id_own.max()/1e3, 2); put(f"Mod{tg}DcIdeal", id_dc.max()/1e3, 2)
        put(f"Mod{tg}Tt", tt_s*1e6, 1)
        put(f"Mod{tg}Irr", Irr_s, 2); put(f"Mod{tg}Vzero", V0s/1e3, 2)
        put(f"Mod{tg}OwnSim", vown[sl].max()/1e3, 2); put(f"Mod{tg}OwnMod", mod_own.max()/1e3, 2)
        put(f"Mod{tg}OwnRule", mod_rule.max()/1e3, 2)
        put(f"Mod{tg}DcSim", vdc[sl].max()/1e3, 2); put(f"Mod{tg}DcMod", mod_dc.max()/1e3, 2)
        put(f"Mod{tg}OwnDev", (mod_own.max()/vown[sl].max() - 1)*100, 1)
        put(f"Mod{tg}DcDev", (mod_dc.max()/vdc[sl].max() - 1)*100, 1)
        put(f"Mod{tg}RuleDev", (mod_rule.max()/vown[sl].max() - 1)*100, 1)
        put(f"Mod{tg}OwnIdealDev", (1 - id_own.max()/vown[sl].max())*100, 0)
        res[k]["dev"] = max(abs(mod_own.max()/vown[sl].max() - 1), abs(mod_dc.max()/vdc[sl].max() - 1))
        # energy dissipated in the snubber resistors of an arm per period (line-frequency loss removed)
        sk = S[k]
        e_sim = N*(sk["P_Rsn_n"] - RSN*(W*CSN*sk["Vrms_n"])**2)/F
        e_step, e_ind, e_src, e_rec = 7/8*CSN/N*V0s**2, LC*Irr_s**2, V0s*Irr_s*tt_s, N*sk["P_rec_n"]/F
        e_mod = e_step + e_ind + e_src - e_rec
        line(f"      energy in the snubber resistors of an arm per period: sim {e_sim:.2f} J; model {e_mod:.2f} J = steps "
             f"{e_step:.2f} + inductance {e_ind:.2f} + source during the tail {e_src:.2f} - diodes {e_rec:.2f}")
        put(f"Ebal{tg}Sim", e_sim, 2); put(f"Ebal{tg}Mod", e_mod, 2); put(f"Ebal{tg}Dev", (e_mod/e_sim - 1)*100, 0)
        put(f"Ebal{tg}Step", e_step, 2); put(f"Ebal{tg}Ind", e_ind, 2); put(f"Ebal{tg}Src", e_src, 2)
        put(f"Ebal{tg}Rec", e_rec, 2)
    put("ModDevMax", math.ceil(max(res[k]["dev"] for k in res)*1000)/10, 1)
    return hand, res


# ---------------------------------------------------------------- 7. sweeps: tables and limits
def crossing(x, y, lim):
    """first x at which y(x) crosses lim upwards (linear interpolation); None if it never does."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if y[0] >= lim:
        return float(x[0])
    for j in range(1, len(x)):
        if y[j] >= lim:
            return float(x[j - 1] + (lim - y[j - 1])*(x[j] - x[j - 1])/(y[j] - y[j - 1]))
    return None


def section_sweeps(S, sweeps, rec, h):
    line("\n7. Sweeps")
    G, QL, UL, CL, RL_ = sweeps["G"], sweeps["QL"], sweeps["UL"], sweeps["CL"], sweeps["RL"]
    P_rat = R_PN*(R_TMAX - T_A)/(R_TMAX - R_TN)             # rating of the resistor in air at T_A
    P_lim = K_P*P_rat                                       # highest power allowed on a resistor
    put("Pmax", f"{P_rat:.3g}"); put("Plim", P_lim, 1)
    put("QswLo", QL[0]*1e6, 0); put("QswHi", QL[-1]*1e6, 0); put("UswHi", f"{UL[-1]:g}")
    put("UswList", ", ".join(f"{u:.2f}" for u in UL[1:-1]) + f" and {UL[-1]:.2f}")
    put("QsweepHead", " & ".join([f"$u = {UL[1]:.2f}$"] + [f"{u:.2f}" for u in UL[2:]]))
    u_top = f"{UL[-1]:.2f}"
    q_ceil = rec["sc"]["ceiling"]["wp"]["Nom"]["Q"]*1e-6
    # limits on the reference Qrr (at the reference slope) for the design unbalance
    for sd in "RL":
        v = [G(("Q", sd, q, U_UNB))["Vpk_w"] for q in QL]
        p = [G(("Q", sd, q, U_UNB))["P_Rsn_w"] for q in QL]
        qv, qp = crossing(QL, v, K_U*VRRM), crossing(QL, p, P_lim)
        assert qv is not None, "the sweep of Qrr does not reach K_U*VRRM: extend Q_SWEEP"
        sweeps[f"qlim_{sd}"] = (qv, qp)
        line(f"   {sd}, u = {U_UNB}: weak level at {K_U:.1f} VRRM for Qrr = {qv*1e6:.0f} uC ({qv/q_ceil*100:.0f} % of the ceiling); "
             f"its Rsn at {P_lim:.1f} W for Qrr " + (f"= {qp*1e6:.0f} uC" if qp is not None else f"> {QL[-1]*1e6:.0f} uC"))
        put(f"QlimV{sd}", qv*1e6, 0); put(f"QlimVRel{sd}", qv/Q_DES, 2); put(f"QlimVCeil{sd}", qv/q_ceil*100, 0)
        put(f"QlimP{sd}", f"{qp*1e6:.0f}" if qp is not None else f"more than {QL[-1]*1e6:.0f}")
        put(f"TauLimV{sd}", float(np.interp(qv*1e6, rec["scan"]["Q"], rec["scan"]["tau"])), 0)
        qt = crossing(QL, [G(("QT", sd, q))["Vpk_w"] for q in QL], K_U*VRRM)
        assert qt is not None and qt > QL[0], "the sweep of Qrr with the capacitor tolerance does not bracket K_U*VRRM"
        line(f"   {sd}, u = {U_UNB}, Csn -+{TOL_C*100:.0f} %: weak level at {K_U:.1f} VRRM for Qrr = {qt*1e6:.0f} uC "
             f"({qt/q_ceil*100:.0f} % of the ceiling)")
        put(f"QlimVTol{sd}", qt*1e6, 0); put(f"QlimVTolRel{sd}", qt/Q_DES, 2)
        put(f"TauLimVTol{sd}", float(np.interp(qt*1e6, rec["scan"]["Q"], rec["scan"]["tau"])), 0)
        ulim = [crossing(UL, [G(("Q", sd, q, u))["Vpk_w"] for u in UL], K_U*VRRM) for q in QL]
        line(f"   {sd}: unbalance that brings the weak level to {K_U:.1f} VRRM, by Qrr: " + ", ".join(
            f"{q*1e6:.0f} uC: " + (f"> {u_top}" if ul is None else f"{ul:.2f}") for q, ul in zip(QL, ulim)))
        ud = ulim[int(np.argmin(np.abs(np.array(QL) - Q_DES)))]
        put(f"UlimDes{sd}", f"above {u_top}" if ud is None else f"{ud:.2f}")
    rows = []
    for q in QL:
        cells = []
        for sd in "RL":
            for u in UL[1:]:
                cells.append(f"{G(('Q', sd, q, u))['Vpk_w']/1e3:.2f}")
            cells.append(f"{G(('Q', sd, q, U_UNB))['P_Rsn_w']:.0f}")
        rows.append(f"{q*1e6:.0f} & {math.sqrt(2*A_REF*q/(1 + rec['s_des'])):.1f} & " + " & ".join(cells) + r"\\")
    put("QsweepRows", "\n".join(rows))
    # network sweep
    t_off = OP["Max"]["mu"]/W + 0.5e-3                      # a peak earlier than this belongs to the turn-off of the arm
    line("   Csn-Rsn sweep, weak-level peak [kV] / P_Rsn of the weak level [W] / resistors per stack [W]; "
         "* = the peak occurs at the turn-off of the arm")
    for sd in "RL":
        for c_ in CL:
            line(f"   {sd} {c_*1e9:4.0f} nF: " + "  ".join(
                f"{r_:.0f}: {G(('N', sd, c_, r_))['Vpk_w']/1e3:.2f}{'*' if G(('N', sd, c_, r_))['tpk_w'] < t_off else ''}/"
                f"{G(('N', sd, c_, r_))['P_Rsn_w']:.0f}/{G(('N', sd, c_, r_))['P_res']:.0f}" for r_ in RL_))
    RLt = (RL_[0], RSN, RL_[-2], RL_[-1])                   # columns of the table of the note
    assert RL_[0] < RSN < RL_[-2], "R_SWEEP: RSN has to lie between the smallest and the second largest value"
    rows = []
    for c_ in CL:
        cells = []
        for key, fmt in (("Vpk_w", lambda x: f"{x/1e3:.2f}"), ("P_Rsn_w", lambda x: f"{x:.0f}")):
            for sd in "RL":
                for r_ in RLt:
                    cells.append(fmt(G(("N", sd, c_, r_))[key]))
        rows.append(f"{c_*1e9:.0f} & " + " & ".join(cells) + r"\\")
    put("NetRows", "\n".join(rows)); put("NetHead", " & ".join(f"{r_:.0f}" for r_ in RLt))
    put("NetRNext", RL_[-2], 0); put("NetCCount", ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight")[len(CL)])
    put("NetCMin", CL[0]*1e9, 0); put("NetCMax", CL[-1]*1e9, 0); put("NetRMin", RL_[0], 0); put("NetRMax", RL_[-1], 0)
    # resistive load: the capacitor sets the peak; effect of the resistor below the largest value of the sweep
    vR = lambda c_, r_: G(("N", "R", c_, r_))["Vpk_w"]
    put("NetRVwCLo", vR(CL[0], RSN)/1e3, 2); put("NetRVwCHi", vR(CL[-1], RSN)/1e3, 2)
    spread = {c_: max(abs(vR(c_, r_)/vR(c_, RL_[-2]) - 1) for r_ in RL_[:-1]) for c_ in CL}
    small = [c_ for c_ in CL if spread[c_] > 0.005]
    put("NetRSpreadHi", max(spread.values())*100, 0)
    put("NetRSpreadCHi", (max(small) if small else CL[0])*1e9, 0)
    put("NetRSpreadRef", max(spread[c_] for c_ in CL if c_ >= CSN)*100, 1)
    line("   R: largest change of the weak-level peak along Rsn (largest value excluded), by Csn: " + ", ".join(
        f"{c_*1e9:.0f} nF {spread[c_]*100:.1f} %" for c_ in CL))
    # largest resistance of the sweep: the peak moves to the turn-off (resistive step on the weak level)
    s_top = G(("N", "R", CSN, RL_[-1]))
    put("NetRTopVw", s_top["Vpk_w"]/1e3, 2); put("NetRTopRel", (s_top["Vpk_w"]/vR(CSN, RSN) - 1)*100, 0)
    put("NetRTopIsn", s_top["Isn_pk_w"], 1); put("NetRTopStep", RL_[-1]*s_top["Isn_pk_w"]/1e3, 2)
    put("NetRTopUtil", s_top["Vpk_w"]/VRRM*100, 0); put("NetRTopStepHand", RL_[-1]*h["i_step"]/1e3, 2)
    put("NetRTopCap", (s_top["Vpk_w"] - RL_[-1]*s_top["Isn_pk_w"])/1e3, 2)
    put("NetLTopVw", G(("N", "L", CSN, RL_[-1]))["Vpk_w"]/1e3, 2)
    put("NetRCornerVw", vR(CL[0], RL_[0])/1e3, 2); put("NetRCornerRef", vR(CL[0], RSN)/1e3, 2)
    mc = modes(Rsn=RL_[0], Csn=CL[0])[1]                    # mode 1 with the smallest capacitor and resistor
    put("CornerZetaOne", mc["zeta"], 2); put("CornerChiOne", 2*h["Irr"]*mc["Z0"]/h["V0s"], 1)
    # share of the commutation energy between the snubber resistor and the diode of a nominal level, against Rsn
    s_lo, s_hi = G(("N", "R", CSN, RL_[0])), s_top
    put("NetRPrecLo", s_lo["P_rec_n"], 1); put("NetRPrecHi", s_hi["P_rec_n"], 1)
    put("NetRPnLo", s_lo["P_Rsn_n"], 1); put("NetRPnHi", s_hi["P_Rsn_n"], 1)
    line(f"   R, {CSN*1e9:.0f} nF, nominal level, {RL_[0]:.0f} -> {RL_[-1]:.0f} Ohm: snubber resistor {s_lo['P_Rsn_n']:.1f} -> "
         f"{s_hi['P_Rsn_n']:.1f} W, turn-off loss of the diode {s_lo['P_rec_n']:.1f} -> {s_hi['P_rec_n']:.1f} W")
    line(f"   R, {CSN*1e9:.0f} nF and {RL_[-1]:.0f} Ohm: weak level {s_top['Vpk_w']/1e3:.2f} kV, {s_top['tpk_w']*1e3:.2f} ms after the "
         f"natural commutation (overlap {OP['Max']['mu']/W*1e3:.2f} ms), snubber current {s_top['Isn_pk_w']:.2f} A peak, "
         f"Rsn x current = {RL_[-1]*s_top['Isn_pk_w']/1e3:.2f} kV")
    # smoothing reactor: best resistor of the sweep for the reference capacitor
    vL = {r_: G(("N", "L", CSN, r_)) for r_ in RL_}
    r_best = min(RL_, key=lambda r_: vL[r_]["Vpk_w"])
    put("NetLBestR", r_best, 0); put("NetLBestVw", vL[r_best]["Vpk_w"]/1e3, 2)
    put("NetLBestRel", (1 - vL[r_best]["Vpk_w"]/vL[RSN]["Vpk_w"])*100, 0)
    put("NetLBestPRel", (vL[r_best]["P_Rsn_w"]/vL[RSN]["P_Rsn_w"] - 1)*100, 0)
    line(f"   L, {CSN*1e9:.0f} nF: lowest weak-level peak {vL[r_best]['Vpk_w']/1e3:.2f} kV at {r_best:.0f} Ohm "
         f"({(vL[r_best]['Vpk_w']/vL[RSN]['Vpk_w'] - 1)*100:+.1f} % against {RSN:.0f} Ohm), P_Rsn of the weak level "
         f"{(vL[r_best]['P_Rsn_w']/vL[RSN]['P_Rsn_w'] - 1)*100:+.0f} %")
    for sd in "RL":
        for rb, nm in ((RB/2, "Half"), (2*RB, "Double")):
            s = G(("B", sd, rb))
            line(f"   {sd}, Rb = {rb/1e3:.0f} kOhm: weak {s['Vpk_w']/1e3:.3f} kV, P_Rb {s['P_Rb_w']:.1f}/{s['P_Rb_n']:.1f} W")
            put(f"Rb{nm}{sd}Vw", s["Vpk_w"]/1e3, 2); put(f"Rb{nm}{sd}Pw", s["P_Rb_w"], 1); put(f"Rb{nm}{sd}Pn", s["P_Rb_n"], 1)
    put("RbHalf", RB/2e3, 0); put("RbDouble", 2*RB/1e3, 0)
    # DC-side inductance
    rows = []
    for ld in sweeps["LDL"]:
        s = G(("D", ld))
        line(f"   Ld = {ld*1e3:6.0f} mH ({ld/(2*LC):5.1f} x 2Lc): stack {s['Vs']/1e3:.2f} kV, weak {s['Vpk_w']/1e3:.3f} kV, "
             f"P_Rsn {s['P_Rsn_w']:.1f}/{s['P_Rsn_n']:.1f} W, resistors per stack {s['P_res']:.0f} W")
        rows.append(f"{ld*1e3:.0f} & {ld/(2*LC):.1f} & {s['Vs']/1e3:.2f} & {s['Vs']/OP['Max']['Vhat']:.2f} & "
                    f"{s['Vpk_w']/1e3:.2f} & {s['P_Rsn_w']:.1f} & {s['P_Rsn_n']:.1f} & {s['P_res']:.0f}" + r"\\")
    put("LdRows", "\n".join(rows))
    sst = G(("D", LD_STIFF))
    vs_ld = [G(("D", ld))["Vs"] for ld in sweeps["LDL"]]
    flat = [ld for ld, v in zip(sweeps["LDL"], vs_ld) if v <= 1.005*OP["Max"]["Vhat"]]
    near = [ld for ld, v in zip(sweeps["LDL"], vs_ld) if v >= 0.98*sst["Vs"]]
    put("LdFlatRatio", max(flat)/(2*LC), 1); put("LdNearStiffRatio", min(near)/(2*LC), 0)
    put("StiffVs", sst["Vs"]/1e3, 2); put("StiffVw", sst["Vpk_w"]/1e3, 2); put("StiffKtr", sst["Vs"]/OP["Max"]["Vhat"], 2)
    put("StiffIrr", sst["Irrm_n"], 2)
    # options for the current-stiff case
    rows = []
    net = lambda c_, r_: f"$C_\\mathrm{{sn}}$ {c_*1e9:.0f} nF, $R_\\mathrm{{sn}}$ {r_:.0f} $\\Omega$"
    ref = S["L_ctol"]
    opt = [("reference network", ref, "reference network")]
    for c_, r_ in OPTIONS:
        opt.append((net(c_, r_), G(("O", c_, r_)), f"Csn {c_*1e9:.0f} nF, Rsn {r_:.0f} Ohm"))
    opt.append((f"$n = {N + 1}$, reference network", sweeps["n"][N + 1]["L_ctol"], f"n = {N + 1}, reference network"))
    opt.append((f"$n = {N + 1}$, " + net(*OPTIONS[0]), sweeps["n"][N + 1]["L_opt"],
                f"n = {N + 1}, Csn {OPTIONS[0][0]*1e9:.0f} nF, Rsn {OPTIONS[0][1]:.0f} Ohm"))
    for (lab, s, txt), tg in zip(opt, ("Ref", "A", "B", "Plus", "PlusA")):
        rows.append(f"{lab} & {s['Vpk_w']/1e3:.2f} & {s['Vpk_w']/VRRM*100:.0f} & {s['Vs']/1e3:.1f} & {s['P_Rsn_w']:.0f} & "
                    f"{s['P_Rsn_n']:.0f} & {s['P_res']:.0f} & --" + r"\\")
        line(f"   option {txt}: weak {s['Vpk_w']/1e3:.2f} kV with Csn tolerance ({s['Vpk_w']/VRRM*100:.0f} %), P_Rsn "
             f"{s['P_Rsn_w']:.0f}/{s['P_Rsn_n']:.0f} W, stack {s['P_res']:.0f} W ({(s['P_res']/ref['P_res'] - 1)*100:+.0f} %)")
        put(f"Opt{tg}Vw", s["Vpk_w"]/1e3, 2); put(f"Opt{tg}Util", s["Vpk_w"]/VRRM*100, 0); put(f"Opt{tg}Pw", s["P_Rsn_w"], 0)
        put(f"Opt{tg}Pres", s["P_res"], 0); put(f"Opt{tg}PresRel", abs(s["P_res"]/ref["P_res"] - 1)*100, 0)
    for j, (c_, r_) in enumerate(OPTIONS):
        put(f"Opt{'AB'[j]}C", c_*1e9, 0); put(f"Opt{'AB'[j]}R", r_, 0)
        put(f"Opt{'AB'[j]}Rrule", math.sqrt(5*LC/(N*c_)), 0)
    px, ux = [], []
    for rx, cx in sweeps["DAMP"]:
        s = G(("X", rx, cx))
        rows.append(f"DC damper {rx:.0f} $\\Omega$ + {cx*1e9:.0f} nF (no tolerance) & {s['Vpk_w']/1e3:.2f} & "
                    f"{s['Vpk_w']/VRRM*100:.0f} & {s['Vs']/1e3:.1f} & {s['P_Rsn_w']:.0f} & {s['P_Rsn_n']:.0f} & {s['P_res']:.0f} & "
                    f"{s['P_x']:.0f}" + r"\\")
        line(f"   DC damper {rx:.0f} Ohm + {cx*1e9:.0f} nF: weak {s['Vpk_w']/1e3:.3f} kV, stack {s['Vs']/1e3:.2f} kV, damper "
             f"{s['P_x']:.0f} W")
        px.append(s["P_x"]); ux.append(s["Vs"]/OP["Max"]["Vhat"])
    put("OptRows", "\n".join(rows))
    put("DampPLo", round(min(px)/10)*10, 0); put("DampPHi", round(max(px)/10)*10, 0)
    put("DampKtrLo", min(ux), 2); put("DampKtrHi", max(ux), 2)
    # number of diodes in series
    rows = []
    for n in sorted(sweeps["n"]):
        d = sweeps["n"][n]
        cells = []
        for sd in "RL":
            s, st = d[sd], d[sd + "_ctol"]
            cells += [f"{s['Vpk_w']/1e3:.2f}", f"{st['Vpk_w']/1e3:.2f}", f"{st['Vpk_w']/VRRM*100:.0f}", f"{s['P_res']:.0f}"]
            tg = {N - 1: "Minus", N: "Ref", N + 1: "Plus", N + 2: "PlusTwo"}[n]
            put(f"Nsw{sd}{tg}", st["Vpk_w"]/1e3, 2); put(f"NswUtil{sd}{tg}", st["Vpk_w"]/VRRM*100, 0)
        rows.append(f"{n} & " + " & ".join(cells) + r"\\")
        line(f"   n = {n}: " + "; ".join(f"{sd}: weak {d[sd]['Vpk_w']/1e3:.2f} kV, with Csn tolerance "
                                         f"{d[sd + '_ctol']['Vpk_w']/1e3:.2f} kV, resistors {d[sd]['P_res']:.0f} W" for sd in "RL"))
    put("NRows", "\n".join(rows))
    # first-cut number of devices: n >= k_tr k_sh Vhat/(K_U VRRM)
    for sd in "RL":
        s = S[sd + "_ctol"]
        ktr, ksh = S[sd]["Vs"]/OP["Max"]["Vhat"], s["Vpk_w"]/(S[sd]["Vs"]/N)
        put(f"Ktr{sd}", ktr, 2); put(f"Ksh{sd}", ksh, 2); put(f"Nmin{sd}", ktr*ksh*OP["Max"]["Vhat"]/(K_U*VRRM), 1)
        line(f"   {sd}: k_tr {ktr:.2f}, k_sh {ksh:.2f} (with Csn tolerance) -> n >= {ktr*ksh*OP['Max']['Vhat']/(K_U*VRRM):.1f}")
    put("NVrrm", N*VRRM/1e3, 0); put("NVrsm", N*VRSM/1e3, 1); put("KuVrrm", K_U*VRRM/1e3, 1)
    put("StackUtil", OP["Max"]["Vhat"]/(N*VRRM)*100, 0)
    # ratings of the snubber components of the weak level
    line("   ratings of the snubber of the weak level (peak voltage with the tolerances of Csn and Rb)")
    for sd in "RL":
        s, st = S[sd], S[sd + "_tol"]
        un, dvdt, esn, vr = st["Vpk_w"]/0.7, s["Isn_pk_w"]/CSN, s["P_Rsn_w"]/F, RSN*s["Isn_pk_w"]
        line(f"      {sd}: capacitor rated voltage >= {un/1e3:.1f} kV (70 % rule on {st['Vpk_w']/1e3:.2f} kV), "
             f"{s['Isn_rms_w']:.2f} A rms, {s['Isn_pk_w']:.2f} A peak, {dvdt/1e6:.0f} V/us; resistor {esn:.2f} J per period, "
             f"peak {vr/1e3:.2f} kV")
        put(f"CapUn{sd}", un/1e3, 1); put(f"Dvdt{sd}", dvdt/1e6, 0); put(f"Esn{sd}", esn, 2); put(f"VrsnPk{sd}", vr/1e3, 2)
    put("Iimp", U_IMP_LEVEL/RSN, 0)
    put("LsLimLo", RSN**2*CJ[0]/4*1e6, 1); put("LsLimHi", RSN**2*CJ[1]/4*1e6, 1)
    put("CjLo", CJ[0]*1e9, 1); put("CjHi", CJ[1]*1e9, 1)
    line(f"      impulse of {U_IMP_LEVEL/1e3:.0f} kV on a level: {U_IMP_LEVEL/RSN:.0f} A in Rsn; loop inductance of the snubber "
         f"<= Rsn^2 Cj/4 = {RSN**2*CJ[0]/4*1e6:.1f} to {RSN**2*CJ[1]/4*1e6:.1f} uH for Cj = {CJ[0]*1e9:.1f} to {CJ[1]*1e9:.1f} nF")


# ---------------------------------------------------------------- 8. losses and thermal design
def solve_tj(PF, PR_of_T, Rth):
    """junction temperature at the thermal equilibrium; nan if there is none below Tvj max (thermal runaway)."""
    T = T_A + PF*Rth
    for _ in range(400):
        Tn = T_A + (PF + PR_of_T(T))*Rth
        if Tn > TVJ_MAX:
            return float("nan")
        if abs(Tn - T) < 1e-4:
            return Tn
        T = Tn
    return T


RHA_TAB = (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8)    # K/W, heat sink to ambient, per diode


def section_thermal(S):
    line(f"\n8. Losses and thermal design (grid +{(K_GRID-1)*100:.0f} %)")
    th = {}
    for sd in "RL":
        s, Wv = S[sd], S[sd]["wave"]
        vw, vn = np.clip(-Wv["v"][:, ARM, 0], 0, None), np.clip(-Wv["v"][:, ARM, 1], 0, None)
        PF = VT0*s["IFav_n"] + RT*s["IFrms_n"]**2
        PRw = lambda T, v=vw: float(np.mean(v*leak(v, T)))
        PRn = lambda T, v=vn: float(np.mean(v*leak(v, T)))
        th[sd] = dict(PF=PF, PRw=PRw, PRn=PRn, Prec=s["P_rec_n"])
        line(f"   {sd}: I_F(AV) {s['IFav_n']:.1f} A, I_F(rms) {s['IFrms_n']:.1f} A, peak {s['IFpk_n']:.1f} A -> P_F {PF:.1f} W; "
             f"turn-off {s['P_rec_n']:.1f} W; reverse loss at the leakage limit, weak-level voltage: "
             + ", ".join(f"{T} C {PRw(T):.1f} W" for T in (90, 100, 110, 125)))
        put(f"Pf{sd}", PF, 1); put(f"PrW{sd}Hundred", PRw(100), 1); put(f"PrN{sd}Hundred", PRn(100), 1)
        put(f"PrW{sd}HundredTwentyFive", PRw(125), 1); put(f"PrW{sd}Ninety", PRw(90), 1)
        P0 = PF + s["P_rec_n"]
        th[sd]["tj"] = []
        for Rha in RHA_TAB:
            Rth = RTH_JC + RTH_CH + Rha
            T0, Tw, Tn = T_A + P0*Rth, solve_tj(P0, PRw, Rth), solve_tj(P0, PRn, Rth)
            f = lambda x: "runaway" if not np.isfinite(x) else f"{x:.0f}"
            th[sd]["tj"].append(f"{T0:.0f} & {f(Tn)} & {f(Tw)}")
            line(f"      Rth(h-a) {Rha:.1f} K/W: Tj {T0:.1f} C without leakage, nominal level {f(Tn)}, weak level {f(Tw)}")
        res = {}
        for nm, Tlim in (("Des", TJ_DES), ("Run", TVJ_MAX - 1e-3)):
            lo, hi = 0.02, 10.0
            for _ in range(60):
                mid = 0.5*(lo + hi)
                T = solve_tj(P0, PRw, RTH_JC + RTH_CH + mid)
                lo, hi = (mid, hi) if (np.isfinite(T) and T <= Tlim) else (lo, mid)
            res[nm] = lo
        th[sd].update(P0=P0, Rdes=res["Des"], Rrun=res["Run"])
        Pd = P0 + PRw(TJ_DES)
        th[sd]["Pd"] = Pd
        Rcond = (TVJ_MAX - T_A)/P0 - RTH_JC - RTH_CH        # heat sink for the forward and turn-off loss alone at Tvj max
        line(f"      Rth(h-a) <= {res['Des']:.2f} K/W for Tj = {TJ_DES:.0f} C on the weak level; thermal runaway above "
             f"{res['Run']:.2f} K/W (ratio {res['Run']/res['Des']:.2f}); diode loss at {TJ_DES:.0f} C {Pd:.1f} W; without "
             f"reverse loss the heat sink could have {Rcond:.1f} K/W at Tvj max")
        put(f"Rha{sd}Des", res["Des"], 2); put(f"Rha{sd}Run", res["Run"], 2); put(f"Pd{sd}", Pd, 1)
        put(f"Rha{sd}Ratio", res["Run"]/res["Des"], 1); put(f"Rha{sd}Cond", Rcond, 1)
        # resistors
        for nm, cap in (("w", "W"), ("n", "N")):
            for kind in ("Rsn", "Rb"):
                P = s[f"P_{kind}_{nm}"]
                put(f"Ths{kind}{sd}{cap}", T_A + (R_TMAX - R_TN)*P/R_PN, 0)
        Pst = N*Pd + s["P_res"]
        rho = 1.204*293.15/(T_A + 273.15)*math.exp(-ALTITUDE/8434.0)
        flow = lambda P, dT: P/(rho*1007.0*dT)*3600
        th[sd].update(Pst=Pst, rho=rho)
        line(f"      resistors per stack {s['P_res']:.0f} W, diodes {N*Pd:.0f} W, stack {Pst:.0f} W, bridge {6*Pst/1e3:.2f} kW; air at "
             f"{T_A:.0f} C and {ALTITUDE:.0f} m: {rho:.3f} kg/m3, {flow(6*Pst, 10):.0f} m3/h for a rise of 10 K")
        put(f"Pstack{sd}", Pst, 0); put(f"Pbridge{sd}", 6*Pst/1e3, 2); put(f"Flow{sd}", flow(6*Pst, 10), 0)
        put(f"Pdiodes{sd}", N*Pd, 0); put(f"PresShare{sd}", s["P_res"]/Pst*100, 0)
    put("TjRows", "\n".join(f"{Rha:.1f} & {a} & {b}" + r"\\" for Rha, a, b in zip(RHA_TAB, th["R"]["tj"], th["L"]["tj"])))
    P_rat = R_PN*(R_TMAX - T_A)/(R_TMAX - R_TN)
    put("PrsnSixty", K_P*P_rat, 0); put("Kp", K_P*100, 0); put("TresLim", T_A + (R_TMAX - R_TN)*K_P*P_rat/R_PN, 0)
    put("PrsnUse", max(S[sd]["P_Rsn_w"] for sd in "RL")/P_rat*100, 0)
    for sd in "RL":
        put(f"PrsnUse{sd}", S[sd]["P_Rsn_w"]/P_rat*100, 0)
    put("PrsnUseShort", max(S[sd + "_short_ctol"]["P_Rsn_w"] for sd in "RL")/P_rat*100, 0)
    line(f"   resistor: {R_PN:.0f} W at {R_TN:.0f} C, zero at {R_TMAX:.0f} C -> {P_rat:.1f} W in air at {T_A:.0f} C, "
         f"{K_P*100:.0f} % of it {K_P*P_rat:.1f} W ({T_A + (R_TMAX - R_TN)*K_P*P_rat/R_PN:.0f} C on the element)")
    put("Rho", th["R"]["rho"], 3); put("FlowK", 3600/(th["R"]["rho"]*1007.0), 2)
    put("Ta", T_A, 0); put("Altitude", ALTITUDE, 0); put("RthJc", RTH_JC*1e3, 0); put("RthCh", RTH_CH*1e3, 0)
    put("RthJh", RTH_JC + RTH_CH, 3)
    return th


# ---------------------------------------------------------------- 9. insulation
def clearance(Uimp):
    for u, d in F2:
        if Uimp/1e3 <= u + 1e-9:
            return d
    raise ValueError(Uimp)


def creepage(Urms, group):
    col = {"I": 1, "II": 2, "III": 3}[group]
    xs, ys = [r[0] for r in F5], [r[col] for r in F5]
    if Urms <= xs[0]:
        return ys[0]*max(Urms, 0.0)/xs[0]
    return float(np.interp(Urms, xs, ys))


def node_potentials(Wv):
    """potentials of the n + 1 nodes of every arm referred to the source neutral, (samples, 6, n + 1);
    node 0 is on the AC terminal, level 0 (the weak one) is next to it."""
    v, uk = Wv["v"], Wv["uk"]
    cs = np.cumsum(v, axis=2)
    pot = np.zeros(v.shape[:2] + (v.shape[2] + 1,))
    pot[:, :3, 0], pot[:, 3:, 0] = uk, uk
    pot[:, :3, 1:] = uk[:, :, None] - cs[:, :3]
    pot[:, 3:, 1:] = uk[:, :, None] + cs[:, 3:]
    return pot


def section_insulation(S):
    line(f"\n9. Insulation (rms at the rated grid voltage, peak at +{(K_GRID-1)*100:.0f} %; table values to be checked on "
         "the standard)")
    put("UimpLevel", U_IMP_LEVEL/1e3, 0); put("UimpRatio", U_IMP_LEVEL/VRSM, 2)
    earthing = (("A", "AC neutral earthed"), ("B", "one DC pole earthed"), ("C", "DC midpoint earthed"),
                ("D", "one AC phase earthed (first fault of an unearthed system)"))
    lev, earth = {}, {}
    for sd in "RL":
        Wn, Wx = S[sd + "_nom"]["wave"], S[sd]["wave"]
        lev[sd] = []
        for mlev in range(1, N + 1):
            Urms = rms(Wn["v"][:, ARM, :mlev].sum(-1))
            Upk = float(np.max(-Wx["v"][:, ARM, :mlev].sum(-1)))
            lev[sd].append((Urms, Upk))
            line(f"   {sd}, {mlev} level(s) from the weak one: {Urms/1e3:.2f} kV rms, {Upk/1e3:.2f} kV peak, impulse "
                 f"{mlev*U_IMP_LEVEL/1e3:.0f} kV -> clearance {clearance(mlev*U_IMP_LEVEL):g} mm, creepage "
                 f"{creepage(Urms, 'I'):.0f} mm (group I) / {creepage(Urms, 'III'):.0f} mm (III)")
            if mlev in (1, N):
                tg = "One" if mlev == 1 else "All"
                put(f"Ins{sd}{tg}Rms", Urms/1e3, 2); put(f"Ins{sd}{tg}Pk", Upk/1e3, 2)
        earth[sd] = {}
        for tag, lab in earthing:
            out = []
            for Wv in (Wn, Wx):
                pot = node_potentials(Wv)
                P, Nn = pot[:, 0, -1], pot[:, 3, -1]
                Vph = math.sqrt(2/3)*(U_N if Wv is Wn else K_GRID*U_N)
                ref = {"A": 0*P, "B": Nn, "C": 0.5*(P + Nn), "D": Vph*np.sin(W*Wv["t"] - 4*math.pi/3)}[tag]
                d = pot - ref[:, None, None]
                out.append((float(np.sqrt(np.mean(d**2, axis=0)).max()), float(np.abs(d).max())))
            earth[sd][tag] = (out[0][0], out[1][1])
            line(f"   {sd}, {lab}: highest node voltage to earth {out[0][0]/1e3:.2f} kV rms, {out[1][1]/1e3:.2f} kV peak")
            put(f"Earth{sd}{tag}Rms", out[0][0]/1e3, 1); put(f"Earth{sd}{tag}Pk", out[1][1]/1e3, 1)
    # tables of the note: the rms values of the two DC sides differ by less than 1 %, the larger one is used
    rows = []
    for mlev in range(1, N + 1):
        Urms = max(lev[sd][mlev - 1][0] for sd in "RL")
        rows.append(f"{mlev} & {Urms/1e3:.2f} & {lev['R'][mlev - 1][1]/1e3:.2f} & {lev['L'][mlev - 1][1]/1e3:.2f} & "
                    f"{mlev*U_IMP_LEVEL/1e3:.0f} & {clearance(mlev*U_IMP_LEVEL):g} & {creepage(Urms, 'I'):.0f} & "
                    f"{creepage(Urms, 'III'):.0f}" + r"\\")
    put("InsRows", "\n".join(rows))
    for tg, mlev in (("One", 1), ("All", N)):
        Urms = max(lev[sd][mlev - 1][0] for sd in "RL")
        put(f"Clear{tg}", f"{clearance(mlev*U_IMP_LEVEL):g}"); put(f"CreepI{tg}", creepage(Urms, "I"), 0)
        put(f"CreepIII{tg}", creepage(Urms, "III"), 0)
    rise = [lev["L"][j][1]/lev["R"][j][1] - 1 for j in range(N)]
    put("InsRiseLo", min(rise)*100, 0); put("InsRiseHi", max(rise)*100, 0)
    line(f"   repetitive peaks of case L above those of case R: {min(rise)*100:.0f} to {max(rise)*100:.0f} %")
    rows = []
    for tag, lab in earthing:
        Urms = max(earth[sd][tag][0] for sd in "RL")
        rows.append(f"{tag} & {lab} & {Urms/1e3:.1f} & {earth['R'][tag][1]/1e3:.1f} & {earth['L'][tag][1]/1e3:.1f} & "
                    f"{creepage(Urms, 'I'):.0f} & {creepage(Urms, 'III'):.0f}" + r"\\")
    put("EarthRows", "\n".join(rows))
    ratio = max(earth[sd]["B"][0] for sd in "RL")/max(earth[sd]["A"][0] for sd in "RL")
    put("EarthRatioBA", ratio, 1)
    line(f"   rms voltage to earth, one DC pole earthed against earthed neutral: ratio {ratio:.2f}")
    put("FtwoRows", "\n".join(f"{u:g} & {d:g}" + r"\\" for u, d in F2))
    put("FfiveRows", "\n".join(f"{u:g} & {a:g} & {b:g} & {c:g}" + r"\\" for u, a, b, c in F5))


# ---------------------------------------------------------------- 10. faults
def section_faults(S):
    line("\n10. Faults")
    for nm, uk in (("", UK), ("Min", UK_MIN)):
        Ik = I_TN/uk
        ipk = KAPPA*math.sqrt(2)*Ik
        hw = (math.sqrt(2)*Ik)**2/(4*F)             # I2t of a symmetrical half wave
        line(f"   short circuit on the DC terminals, uk {uk*100:.0f} %: {Ik:.0f} A rms, peak {ipk/1e3:.2f} kA (kappa {KAPPA}), "
             f"I2t {hw/1e3:.1f}e3 A2s per half wave, {5*hw/1e3:.0f}e3 A2s in 100 ms; diode: IFSM {IFSM/1e3:.1f} kA, "
             f"I2t {I2T/1e3:.0f}e3 A2s")
        put(f"Ik{nm}", Ik, 0); put(f"IkPk{nm}", ipk/1e3, 2); put(f"ItwoHw{nm}", hw/1e3, 1)
        put(f"ItwoHundred{nm}", 5*hw/1e3, 0)
    put("Ifsm", IFSM/1e3, 1); put("Itwot", I2T/1e3, 0); put("ItwoMargin", I2T/(5*(math.sqrt(2)*I_TN/UK_MIN)**2/(4*F)), 0)
    put("IkPkShare", KAPPA*math.sqrt(2)*I_TN/UK_MIN/IFSM*100, 0)
    for sd in "RL":
        s, sc = S[sd + "_short"], S[sd + "_short_ctol"]
        line(f"   {sd}, one level failed short (n - 1 = {N-1}): weak {s['Vpk_w']/1e3:.2f} kV = {s['Vpk_w']/VRRM*100:.0f} % of VRRM, "
             f"nominal {s['Vpk_n']/1e3:.2f} kV, P_Rsn {s['P_Rsn_w']:.1f}/{s['P_Rsn_n']:.1f} W; with the capacitor tolerance "
             f"{sc['Vpk_w']/1e3:.2f} kV = {sc['Vpk_w']/VRRM*100:.0f} % of VRRM (VRSM {VRSM/1e3:.1f} kV)")


# ---------------------------------------------------------------- figures
BOX = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9)
SIDE_TITLE = {"R": f"resistive load, $L_\\mathrm{{d}}$ = {LD_R*1e3:.0f} mH", "L": f"smoothing reactor, $L_\\mathrm{{d}}$ = {LD_L:g} H"}
REF_V = ((K_U*VRRM/1e3, f"{K_U:g} $V_\\mathrm{{RRM}}$"), (VRRM/1e3, "$V_\\mathrm{RRM}$"))
P_RAT = R_PN*(R_TMAX - T_A)/(R_TMAX - R_TN)                 # rating of the resistor in air at T_A
REF_P = ((K_P*P_RAT, f"{K_P*100:.0f} % of the\nrating"),)


def ref_lines(ax, x, items, ha="left"):
    """horizontal reference lines, each with a label on a white box."""
    for y, lab in items:
        ax.axhline(y, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.annotate(lab, (x, y), xytext=(0, 2.5), textcoords="offset points", color=INK2, fontsize=7, va="bottom", ha=ha,
                    bbox=BOX, zorder=4)


def ref_margin(axes, items):
    """horizontal reference lines on a row of panels with a shared axis, labelled in the right margin."""
    for y, lab in items:
        for a in axes:
            a.axhline(y, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
        axes[-1].annotate(lab, (1.0, y), xycoords=("axes fraction", "data"), xytext=(6, 0), textcoords="offset points",
                          color=INK2, fontsize=7, va="center", ha="left", annotation_clip=False)


def end_labels(ax, x, items, gap=0.07):
    """labels at the right end of the lines of a panel, moved apart where they would overlap; items = [(y, text)],
    gap = smallest distance between two labels as a fraction of the height of the axis."""
    lo, hi = ax.get_ylim()
    d = gap*(hi - lo)
    items = sorted(items)
    ys = [y for y, _ in items]
    for _ in range(200):
        moved = False
        for j in range(1, len(ys)):
            e = d - (ys[j] - ys[j - 1])
            if e > 1e-9:
                ys[j - 1] -= e/2
                ys[j] += e/2
                moved = True
        if not moved:
            break
    for (_, txt), y in zip(items, ys):
        ax.annotate(txt, (x, y), xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK, va="center",
                    annotation_clip=False)


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, name + ".pdf"))
    plt.close(fig)


def fig_bridge(S):
    """arm and level voltages over a period, both DC sides."""
    fig, ax = plt.subplots(2, 2, figsize=(6.5, 4.0), sharex=True, sharey="row")
    o = OP["Max"]
    for c, sd in enumerate("RL"):
        Wv = roll(S[sd]["wave"], t_nat(ARM) - 1e-3)
        t = np.concatenate([Wv["t"], Wv["t"] + 1/F])*1e3 - 1.0          # one period and the start of the next one
        Tms = 1e3/F
        keep = t <= Tms + 0.9
        t = t[keep]
        two = lambda x: np.concatenate([x, x])[keep]
        vs = two(-Wv["v"][:, ARM].sum(-1)/1e3)
        a = ax[0, c]
        a.plot(t, arm_voltage_ideal(W*(t*1e-3), o["mu"], o["Vhat"])/1e3, color=AXIS, lw=3.2, solid_capstyle="butt",
               label="ideal bridge")
        a.plot(t, vs, color=INK, lw=0.9, label="simulation")
        j = int(np.argmax(vs))
        a.annotate(f"{vs[j]:.1f} kV", (t[j], vs[j]), xytext=(0, 3), textcoords="offset points", fontsize=7, color=INK,
                   ha="center", va="bottom")
        a.set_title(SIDE_TITLE[sd])
        a.set_ylim(0, 1.36*o["Vhat"]/1e3)
        a.legend(loc="center right", handlelength=1.6, borderaxespad=0.1, labelspacing=0.4)
        b = ax[1, c]
        for j_, col, lab in ((0, C_WEAK, "weak level"), (1, C_NOM, "nominal level")):
            v = two(-Wv["v"][:, ARM, j_]/1e3)
            b.plot(t, v, color=col, label=f"{lab}\npeak {v.max():.2f} kV", zorder=3 - j_)
        ref_lines(b, Tms + 0.8, REF_V, ha="right")
        b.set_ylim(0, 1.16*VRRM/1e3)
        b.set_xlim(-1, Tms + 1)
        b.set_xticks(range(0, int(Tms) + 1, 4))
        b.set_xlabel("time from the natural commutation [ms]")
        b.legend(loc="lower right", bbox_to_anchor=(1.0, 0.04), handlelength=1.6, borderaxespad=0.1, labelspacing=0.6)
    ax[0, 0].set_ylabel("arm reverse voltage [kV]")
    ax[1, 0].set_ylabel("level reverse voltage [kV]")
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    save(fig, "fig_bridge_waveforms")


def fig_turn_off(S):
    """diode currents, level voltages and snubber currents around the turn-off."""
    fig, ax = plt.subplots(3, 2, figsize=(6.5, 4.6), sharex=True, sharey="row")
    ilim = 5*math.ceil(1.1*max(S[sd]["Irrm_n"] for sd in "RL")/5)
    for c, sd in enumerate("RL"):
        Wv = roll(S[sd]["wave"], t_nat(ARM))
        ia = Wv["ia"][:, ARM]
        j0 = int(np.argmax(ia[:len(ia)//4] < 0))            # current zero of the arm
        sl = slice(max(j0 - 300, 0), j0 + 700)
        t = (Wv["t"][sl] - Wv["t"][j0])*1e6
        for j_, col, lab in ((1, C_NOM, "nominal level"), (0, C_WEAK, "weak level")):
            ax[0, c].plot(t, Wv["idio"][sl, ARM, j_], color=col, label=lab)
            ax[1, c].plot(t, -Wv["v"][sl, ARM, j_]/1e3, color=col, label=lab)
            ax[2, c].plot(t, -Wv["isn"][sl, ARM, j_], color=col, label=lab)
        ax[0, c].set_title(SIDE_TITLE[sd])
        ax[0, c].set_ylim(-ilim, ilim)
        ax[0, c].axhline(0, color=AXIS, lw=0.6, zorder=1)
        ax[2, c].axhline(0, color=AXIS, lw=0.6, zorder=1)
        ax[2, c].set_xlabel("time from the current zero [µs]")
        ax[2, c].set_xlim(-150, 650)
    ax[0, 0].set_ylabel("diode current [A]")
    ax[1, 0].set_ylabel("level reverse voltage [kV]")
    ax[2, 0].set_ylabel("snubber current [A]")
    ax[0, 1].legend(loc="upper right", handlelength=1.8, borderaxespad=0.2)
    fig.tight_layout(h_pad=0.5, w_pad=0.8)
    save(fig, "fig_turn_off")


def fig_recovery(rec, sweeps):
    """stored charge against the slope (datasheet temperature) and recovered charge at the circuit slope against the
    bulk lifetime."""
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.9))
    a = ax[0]
    al = np.geomspace(0.05, 60, 40)
    IFs = np.minimum(IF_TAB, 2000*al)                       # long ramp: the forward current does not matter
    sty = (("floor", C_NOM, "floor: linear scaling"), ("central", C_WEAK, "central: class lifetime"),
           ("upper", C_3, f"upper: {TAU_UPPER:.0f} µs"), ("ceiling", C_4, "ceiling: no bulk recombination"))
    for key, col, lab in sty:
        s = rec["sc"][key]
        q = [rcv.recovery(x, i, s["gb"], s["TM"], s["k"])["Qrr"] for x, i in zip(al, IFs)]
        a.loglog(al, q, color=col, label=lab)
    xs = np.array([QS_CURVE[0][0], QS_CURVE[1][0]])
    a.loglog(xs, rec["cq"]*xs**rec["eq"], color=INK, lw=2.0, label="datasheet curve")
    a.loglog([A_TAB], [QS_TAB], "o", color=INK, ms=4.5, mfc="white", mew=1.2, label="datasheet table (max)")
    a0 = rec["pts"]["Nom"][0]
    a.axvline(a0, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    a.text(a0*1.08, 2.2*QS_TAB, f"circuit\n{a0:.3f} A/µs", fontsize=7, color=INK2, va="top")
    a.set_xlim(0.05, 60); a.set_ylim(0.008*QS_TAB, 3*QS_TAB)
    a.set_xlabel("current slope [A/µs]"); a.set_ylabel(f"stored charge at {T_TAB:.0f} °C [µC]")
    a.legend(loc="lower right", handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
    b = ax[1]
    sc = rec["scan"]
    x0, x1 = 0.94*sc["tau"][0], sc["tau"][-1]
    b.semilogx(sc["tau"], sc["Q"], color=INK)
    for (key, col, lab), off, ha, va in zip(sty[1:3], ((6, -3), (-5, 4)), ("left", "right"), ("top", "bottom")):
        s = rec["sc"][key]
        b.plot([s["tau"]], [s["wp"]["Nom"]["Q"]], "o", color=col, ms=5, zorder=5)
        b.annotate(f"{key} {s['wp']['Nom']['Q']:.0f} µC", (s["tau"], s["wp"]["Nom"]["Q"]), xytext=off,
                   textcoords="offset points", fontsize=7, color=INK, ha=ha, va=va, bbox=BOX, zorder=4)
    qc = rec["sc"]["ceiling"]["wp"]["Nom"]["Q"]
    b.axhline(qc, color=C_4, lw=1.0)
    b.annotate(f"ceiling {qc:.0f} µC", (0.95*x1, qc), xytext=(0, 2.5), textcoords="offset points", fontsize=7, color=INK,
               va="bottom", ha="right", bbox=BOX)
    # horizontal reference lines; a label goes to the left where the curve is below it, to the right otherwise
    items = [(Q_DES*1e6, f"design value {Q_DES*1e6:.0f} µC")]
    for sd, lab in (("R", "limit, resistive load"), ("L", "limit, smoothing reactor")):
        qv = sweeps[f"qlim_{sd}"][0]
        if qv is not None:
            items.append((qv*1e6, f"{lab} {qv*1e6:.0f} µC"))
    q_left = float(np.interp(4*x0, sc["tau"], sc["Q"]))     # the curve under a label that starts at the left edge
    for y, lab in items:
        left = y > q_left + 0.12*qc
        under = 0 < qc - y < 0.12*qc                        # a label above this line would cover the ceiling line
        b.axhline(y, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
        b.annotate(lab, (1.03*x0 if left else 0.95*x1, y), xytext=(0, -2.5 if under else 2.5),
                   textcoords="offset points", color=INK2, fontsize=7, va="top" if under else "bottom",
                   ha="left" if left else "right", bbox=BOX, zorder=4)
    b.set_xlim(x0, x1); b.set_ylim(0, 1.16*qc)
    b.set_xlabel(f"bulk lifetime at {T_TAB:.0f} °C [µs]")
    b.set_ylabel(f"$Q_\\mathrm{{rr}}$ at {a0:.3f} A/µs, {TJ_DES:.0f} °C [µC]")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_recovery_slope")


def fig_modes(res):
    """arm voltages after the turn-off with a current-stiff DC side: simulation against the two-mode model."""
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.7), sharex=True)
    d = res["S_soft"]
    t = d["t"]*1e6
    for a, key, ttl in ((ax[0], "own", "recovering arm"), (ax[1], "dc", "arm across the DC terminals")):
        a.plot(t, d[key]/1e3, color=C_NOM, lw=1.6, label="simulation")
        a.plot(t, d["mod_" + key]/1e3, color=C_WEAK, lw=1.1, ls=(0, (4, 2)), label="model with the recovery tail")
        a.plot(t, d["id_" + key]/1e3, color=C_3, lw=1.1, ls=(0, (1, 1.5)), label="model, ideal snap-off")
        a.set_title(ttl)
        a.set_xlabel("time from the peak of the reverse current [µs]")
        a.set_xlim(0, 1200)
    ax[0].set_ylabel("arm reverse voltage [kV]")
    ax[0].set_ylim(0, 2*math.ceil(1.25*d["own"].max()/2e3))
    lo = math.floor(d["dc"].min()/1e3/2)*2
    ax[1].set_ylim(lo, lo + 2*math.ceil(1.5*(d["dc"].max()/1e3 - lo)/2))
    ax[1].legend(loc="upper right", handlelength=2.2, borderaxespad=0.2)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_commutation_modes")


def fig_sharing(sweeps, hand, rec, m):
    """highest arm voltage against the recovered charge and offset of the weak level against the charge difference."""
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.9))
    G, o, QL = sweeps["G"], OP["Max"], np.array(sweeps["QL"])
    mk = dict(ls="none", ms=4, mec=INK, mew=0.9, zorder=3)
    a = ax[0]
    qf = np.linspace(0.6*QL[0], 1.04*QL[-1], 24)
    vm = []
    for q in qf:
        tau, TM = lm_params(q, math.sqrt(2*A_REF*q/(1 + rec["s_des"])), A_REF)
        vm.append(stack_peak_stiff(o, turn_off(o, float(tau), float(TM)), m)[0]/1e3)
    l1, = a.plot(qf*1e6, vm, color=INK, lw=0.9, label="two-mode model, current-stiff limit")
    l2 = a.axhline(o["Vhat"]/1e3, color=INK, lw=0.9, ls=(0, (4, 2)), label="crest of the line voltage")
    p1, = a.plot(QL*1e6, [G(("Q", "R", q, 0.0))["Vs"]/1e3 for q in QL], marker="o", mfc="white",
                 label=f"$L_\\mathrm{{d}}$ = {LD_R*1e3:.0f} mH", **mk)
    p2, = a.plot(QL*1e6, [G(("Q", "L", q, 0.0))["Vs"]/1e3 for q in QL], marker="o", mfc=INK,
                 label=f"$L_\\mathrm{{d}}$ = {LD_L:g} H", **mk)
    p3, = a.plot(QL*1e6, [x["Vs"]/1e3 for x in sweeps["stiff"]], marker="s", mfc=MUTED,
                 label=f"$L_\\mathrm{{d}}$ = {LD_STIFF:g} H", **mk)
    a.set_xlim(0.6*QL[0]*1e6, 1.04*QL[-1]*1e6); a.set_ylim(round(0.70*o["Vhat"]/1e3), round(1.44*o["Vhat"]/1e3))
    a.set_xlabel(f"$Q_\\mathrm{{rr}}$ at {A_REF/1e6:.3f} A/µs [µC]")
    a.set_ylabel("highest arm voltage [kV]")
    leg = a.legend(handles=[l1, l2], loc="upper left", handlelength=2.0, borderaxespad=0.2, labelspacing=0.35)
    a.add_artist(leg)
    a.legend(handles=[p1, p2, p3], loc="lower right", title="simulation", ncol=3, handletextpad=0.2, columnspacing=1.0,
             borderaxespad=0.2)
    b = ax[1]
    dI = float(leak(V_LEV, TJ_DES))
    tpk = 0.5*(hand["R"]["tpk"] + hand["L"]["tpk"])
    pts = {}
    for sd in "RL":
        pts[sd] = [(s["dQ"]*1e6, (s["Vpk_w"] - s["Vs"]/N)/1e3)
                   for s in (G(("Q", sd, q, u)) for q in sweeps["QL"] for u in sweeps["UL"][1:])]
    xmax = 100*math.ceil(max(x for sd in "RL" for x, _ in pts[sd])/100)
    ymax = math.ceil(1.3*max(y for sd in "RL" for _, y in pts[sd]))
    xs = np.linspace(0, xmax, 20)
    b.plot(xs, [weak_peak(0.0, q*1e-6, tpk, dIlk=dI)/1e3 for q in xs], color=INK, lw=0.9, label="charge model")
    for sd, mfc, lab in (("R", "white", "simulation, resistive load"), ("L", INK, "simulation, smoothing reactor")):
        b.plot([x for x, _ in pts[sd]], [y for _, y in pts[sd]], marker="o", mfc=mfc, label=lab, **mk)
    b.set_xlim(0, xmax); b.set_ylim(0, ymax)
    b.set_xlabel("charge difference $\\Delta Q$ [µC]")
    b.set_ylabel("weak-level peak above $\\hat V_\\mathrm{s}/n$ [kV]")
    b.legend(loc="upper left", handlelength=2.0, borderaxespad=0.2, labelspacing=0.35)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_dynamic_sharing")


def fig_qrr(sweeps):
    """weak-level peak and snubber power against the recovered charge, for several unbalances."""
    fig, ax = plt.subplots(2, 2, figsize=(6.5, 4.3), sharex=True, sharey="row")
    G, QL, UL = sweeps["G"], np.array(sweeps["QL"]), sweeps["UL"]
    cols = [RAMP[0], RAMP[2], RAMP[3], RAMP[5]]
    vall = [G(("Q", sd, q, u))["Vpk_w"]/1e3 for sd in "RL" for q in QL for u in UL]
    pall = [G(("Q", sd, q, u))["P_Rsn_w"] for sd in "RL" for q in QL for u in UL]
    for c, sd in enumerate("RL"):
        for u, col in zip(UL, cols):
            v = [G(("Q", sd, q, u))["Vpk_w"]/1e3 for q in QL]
            p = [G(("Q", sd, q, u))["P_Rsn_w"] for q in QL]
            ax[0, c].plot(QL*1e6, v, "-o", color=col, ms=3, label=f"$u$ = {u:.2f}")
            ax[1, c].plot(QL*1e6, p, "-o", color=col, ms=3, label=f"$u$ = {u:.2f}")
        ax[0, c].set_title(SIDE_TITLE[sd])
        ax[0, c].set_ylim(math.floor(min(vall)), math.ceil(max(max(vall), VRRM/1e3) + 0.2))
        end_labels(ax[0, c], QL[-1]*1e6, [(G(("Q", sd, QL[-1], u))["Vpk_w"]/1e3, f"{u:.2f}") for u in UL])
        ax[1, c].set_ylim(0, 10*math.ceil(max(max(pall), K_P*P_RAT)/10 + 0.3))
        ax[1, c].set_xlabel(f"$Q_\\mathrm{{rr}}$ at {A_REF/1e6:.3f} A/µs [µC]")
        ax[1, c].set_xlim(0.6*QL[0]*1e6, 1.11*QL[-1]*1e6)
    ref_margin(ax[0], REF_V)
    ref_margin(ax[1], REF_P)
    ax[0, 0].set_ylabel("weak-level peak [kV]")
    ax[1, 0].set_ylabel("$R_\\mathrm{sn}$ of the weak level [W]")
    ax[1, 0].legend(loc="upper left", title="unbalance", ncol=2, handlelength=1.6, columnspacing=1.0, borderaxespad=0.2)
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    save(fig, "fig_qrr_sensitivity")


def fig_network(sweeps):
    """weak-level peak and snubber power against Rsn for several Csn."""
    fig, ax = plt.subplots(2, 2, figsize=(6.5, 4.3), sharex=True, sharey="row")
    G, CL, RL_ = sweeps["G"], sweeps["CL"], np.array(sweeps["RL"])
    vall = [G(("N", sd, c_, r_))["Vpk_w"]/1e3 for sd in "RL" for c_ in CL for r_ in RL_]
    pall = [G(("N", sd, c_, r_))["P_Rsn_w"] for sd in "RL" for c_ in CL for r_ in RL_]
    for c, sd in enumerate("RL"):
        for c_, col in zip(CL, RAMP):
            v = [G(("N", sd, c_, r_))["Vpk_w"]/1e3 for r_ in RL_]
            p = [G(("N", sd, c_, r_))["P_Rsn_w"] for r_ in RL_]
            ax[0, c].semilogx(RL_, v, "-o", color=col, ms=3, label=f"{c_*1e9:.0f} nF")
            ax[1, c].semilogx(RL_, p, "-o", color=col, ms=3, label=f"{c_*1e9:.0f} nF")
        ax[0, c].set_title(SIDE_TITLE[sd])
        ax[0, c].set_ylim(0.5*math.floor(2*min(vall) - 0.2), 0.5*math.ceil(2*max(max(vall), VRRM/1e3) + 0.4))
        end_labels(ax[0, c], RL_[-1], [(G(("N", sd, c_, RL_[-1]))["Vpk_w"]/1e3, f"{c_*1e9:.0f}") for c_ in CL])
        ax[1, c].set_ylim(0, 10*math.ceil(max(max(pall), K_P*P_RAT)/10 + 2.2))
        ax[1, c].set_xlabel("$R_\\mathrm{sn}$ [Ω]")
        ax[1, c].set_xlim(0.95*RL_[0], 1.22*RL_[-1])
        ax[1, c].set_xticks(RL_)
        ax[1, c].set_xticklabels([f"{r_:.0f}" for r_ in RL_])
        ax[1, c].minorticks_off()
    ref_margin(ax[0], REF_V)
    ref_margin(ax[1], REF_P)
    ax[0, 0].set_ylabel("weak-level peak [kV]")
    ax[1, 0].set_ylabel("$R_\\mathrm{sn}$ of the weak level [W]")
    ax[1, 0].legend(loc="upper left", title="$C_\\mathrm{sn}$", ncol=3, handlelength=1.6, columnspacing=1.0,
                    borderaxespad=0.2)
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    save(fig, "fig_network_sweep")


def fig_thermal(th):
    """thermal stability of a blocking diode: generated power against the power removed by the heat sink."""
    fig, a = plt.subplots(1, 1, figsize=(4.9, 2.9))
    d = th["L"]
    T = np.linspace(T_A, TVJ_MAX, 161)
    a.plot(T, [d["P0"] + d["PRw"](x) for x in T], color=C_WEAK, label="generated, weak level")
    a.plot(T, [d["P0"] + d["PRn"](x) for x in T], color=C_NOM, label="generated, nominal level")
    for Rha, lab in ((d["Rdes"], f"removed,\n{d['Rdes']:.2f} K/W"), (d["Rrun"], f"removed,\n{d['Rrun']:.2f} K/W")):
        Rth = RTH_JC + RTH_CH + Rha
        a.plot(T, (T - T_A)/Rth, color=INK2, lw=0.9, ls=(0, (4, 3)))
        a.annotate(lab, (TVJ_MAX, (TVJ_MAX - T_A)/Rth), xytext=(4, 0), textcoords="offset points", fontsize=7,
                   color=INK2, ha="left", va="center", annotation_clip=False)
    y = (TJ_DES - T_A)/(RTH_JC + RTH_CH + d["Rdes"])
    a.plot([TJ_DES], [y], "o", ms=4.5, mfc="white", mec=INK, mew=1.0, zorder=5)
    a.annotate(f"design point\n{TJ_DES:.0f} °C, {y:.0f} W", (TJ_DES, y), xytext=(-6, 5), textcoords="offset points", fontsize=7,
               color=INK, ha="right", va="bottom")
    a.set_xlim(T_A, TVJ_MAX)
    a.set_ylim(0, 10*math.ceil(1.25*(TVJ_MAX - T_A)/(RTH_JC + RTH_CH + d["Rdes"])/10))
    a.set_xlabel("junction temperature [°C]"); a.set_ylabel("power of a diode [W]")
    a.legend(loc="upper left", handlelength=1.8, borderaxespad=0.2)
    fig.tight_layout()
    save(fig, "fig_thermal_stability")


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    section_operating_point()
    section_static()
    rec = section_recovery()
    rec["s_des"] = 2*A_REF*Q_DES/I_DES**2 - 1
    m, h = section_modes(rec)
    S, sweeps = section_simulations(rec)
    hand, res = section_hand(S, sweeps, rec, m, h)
    section_sweeps(S, sweeps, rec, h)
    th = section_thermal(S)
    section_insulation(S)
    section_faults(S)
    words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")
    ordinals = ("", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
                "eleventh", "twelfth")
    put("N", str(N)); put("Nminus", str(N - 1)); put("Nplus", str(N + 1))
    put("Nword", words[N]); put("NwordCap", words[N].capitalize()); put("NminusWord", words[N - 1])
    put("NminusWordCap", words[N - 1].capitalize()); put("NplusWord", words[N + 1]); put("NplusOrd", ordinals[N + 1])
    put("Rb", RB/1e3, 0); put("Rsn", RSN, 0); put("Csn", CSN*1e9, 0)
    put("TolRb", TOL_RB*100, 0); put("TolC", TOL_C*100, 0); put("Vrrm", VRRM/1e3, 1); put("Vrsm", VRSM/1e3, 1)
    put("VrrmV", VRRM, 0); put("VrsmV", VRSM, 0); put("VrsmExtra", VRSM - VRRM, 0); put("TvjMax", TVJ_MAX, 0)
    put("Irm", IRM*1e3, 0); put("Vto", VT0, 2); put("Rt", RT*1e3, 2); put("LdR", LD_R*1e3, 0); put("LdL", LD_L, 1)
    put("Ku", K_U*100, 0); put("Rpn", R_PN, 0); put("Rtn", R_TN, 0); put("Rtmax", R_TMAX, 0)
    put("RthJcVal", RTH_JC, 3); put("RthChVal", RTH_CH, 3); put("F", F, 0)
    with open(os.path.join(HERE, "diode_stack_results.txt"), "w", encoding="utf-8") as fo:
        fo.write("\n".join(LOG) + "\n")
    with open(os.path.join(HERE, "diode_stack_values.tex"), "w", encoding="utf-8") as fo:
        fo.write("% generated by diode_stack_calc.py: do not edit by hand\n")
        for k, v in MAC.items():
            fo.write(f"\\newcommand{{\\ds{k}}}{{{v}}}\n")
    note = os.path.join(HERE, "diode_stack_design_guidelines.tex")
    if os.path.exists(note):                                # every macro of the note has to be written above
        with open(note, encoding="utf-8") as fi:
            missing = sorted(set(re.findall(r"\\ds([A-Za-z]+)", fi.read())) - set(MAC))
        if missing:
            print("WARNING: macros used in the note and not written by this script: " + ", ".join(missing))
    fig_bridge(S)
    fig_turn_off(S)
    fig_recovery(rec, sweeps)
    fig_modes(res)
    fig_sharing(sweeps, hand, rec, m)
    fig_qrr(sweeps)
    fig_network(sweeps)
    fig_thermal(th)
    print(f"done in {time.time() - t0:.0f} s: {len(MAC)} macros, figures in {FIGDIR}")


if __name__ == "__main__":
    main()
