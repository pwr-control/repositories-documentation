#!/usr/bin/env python3
"""
bod_sim.py - Python replica of the Simscape component dtt.crowbar.bod
(break-over diode, functional model with switching dynamics).

The device equations are the same as in bod.ssc:
  i        = i_dev(v, mode) + C_j dv/dt
  Blocking : i_dev = I_D(T) v/(0.8 V_BO(T)) + I_BO (v/V_BO(T))^m    (v >= 0)
             reverse breakdown beyond -V_RB with resistance R_RB
  Triggered: i_dev = leakage + max(0, v - v_cl)/r_av, v_cl = V_T0 + xb (V_BO(T) - V_T0)
  On       : i_dev = (v - V_T0)/r_T
  xb       : blocking capability, d xb/dt = (1 - xb)/tau_q (Blocking),
             -xb/tau_d (Triggered, On)
  V_blk    = V_rq + xb (V_BO(T) - V_rq)
  events   : Blocking -> Triggered  when v >= V_blk
             Blocking -> TriggeredDV when v > 0 and v dv/dt >= 0.9 V_blk dvdt_cr(T)
             Triggered -> On when xb <= x_on
             On -> Blocking when i_dev < I_H(T)

The external circuits of the verification scenarios are small ODE systems
integrated with scipy (Radau, stiff) piecewise between events.

Figures are produced in grey tones (printable in B/W).

Rev. 00, 2026-09-22
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "doc", "fig")
os.makedirs(FIGDIR, exist_ok=True)

# ---------------------------------------------------------------- plotting
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "lines.linewidth": 1.0, "axes.grid": True, "grid.color": "0.85",
    "grid.linewidth": 0.5, "savefig.bbox": "tight",
})
GREYS = ["0.0", "0.45", "0.7"]
STYLES = ["-", "--", ":", "-."]

# ---------------------------------------------------------------- parameters
class BOD:
    """Parameter set (defaults: IXBOD2-11) and device functions."""
    def __init__(self, **kw):
        self.V_BO = 1100.0      # V
        self.I_BO = 20e-3       # A
        self.m_av = 20.0
        self.I_D25 = 0.2e-6     # A at 0.8 V_BO, 25 degC
        self.I_D125 = 200e-6    # A at 0.8 V_BO, 125 degC
        self.V_T0 = 0.9         # V
        self.r_T = 80e-3        # Ohm
        self.I_H25 = 40e-3      # A
        self.I_H125 = 20e-3     # A
        self.K_T = 1e-3         # 1/K
        self.dvdt_cr25 = 3000e6 # V/s
        self.dvdt_cr150 = 1000e6
        self.V_RB = 10.0        # V
        self.R_RB = 1.0         # Ohm
        self.C_j = 20e-12       # F
        self.t_d = 0.3e-6       # s
        self.r_av = 1.0         # Ohm
        self.t_q = 200e-6       # s
        self.V_rq = 50.0        # V
        self.T_j = 25.0         # degC
        self.x_on = 1e-3
        for k, val in kw.items():
            setattr(self, k, val)
        self.k_H = np.log(self.I_H25/self.I_H125)/100.0
        self.k_D = np.log(self.I_D125/self.I_D25)/100.0
        self.k_dv = np.log(self.dvdt_cr25/self.dvdt_cr150)/125.0
        self.tau_d = self.t_d/np.log(1/self.x_on)
        self.tau_q = self.t_q/np.log(100.0)

    # temperature-dependent quantities
    def V_BOT(self, T=None):
        T = self.T_j if T is None else T
        return self.V_BO*(1 + self.K_T*(T - 25.0))
    def I_HT(self, T=None):
        T = self.T_j if T is None else T
        return self.I_H25*np.exp(-self.k_H*(T - 25.0))
    def I_DT(self, T=None):
        T = self.T_j if T is None else T
        return self.I_D25*np.exp(self.k_D*(T - 25.0))
    def dvdt_crT(self, T=None):
        T = self.T_j if T is None else T
        return self.dvdt_cr25*np.exp(-self.k_dv*(T - 25.0))

    def V_blk(self, xb):
        return self.V_rq + xb*(self.V_BOT() - self.V_rq)
    def v_cl(self, xb):
        return self.V_T0 + xb*(self.V_BOT() - self.V_T0)

    def i_dev(self, v, xb, mode):
        leak = self.I_DT()*v/(0.8*self.V_BOT())
        if mode == 0:  # Blocking
            if v >= 0:
                return leak + self.I_BO*(v/self.V_BOT())**self.m_av
            elif v >= -self.V_RB:
                return leak
            else:
                return leak + (v + self.V_RB)/self.R_RB
        elif mode in (1, 2):  # Triggered
            vc = self.v_cl(xb)
            return leak + (max(0.0, v - vc))/self.r_av
        else:  # On
            return (v - self.V_T0)/self.r_T

    def dxb(self, xb, mode):
        return (1 - xb)/self.tau_q if mode == 0 else -xb/self.tau_d


# ---------------------------------------------------------------- generic simulator
def simulate(bod, circuit, y0, t_end, mode0=0, max_step=None, extra_events=None):
    """
    Piecewise integration with BOD mode switching.
    circuit(t, y, bod, mode) -> (dy/dt, i_bod, aux dict) where y[0] = v_bod, y[1] = xb.
    extra_events: list of (fun(t,y,mode)->float, direction, callback(t,y,state)->None);
    each extra event is one-shot (deactivated after its callback has run).
    """
    state = {"mode": mode0, "log": []}
    t0 = 0.0
    y = np.array(y0, dtype=float)
    T, Y, MODE, IB = [], [], [], []
    extra_events = list(extra_events or [])

    def rhs(t, y):
        dy, _, _ = circuit(t, y, bod, state["mode"])
        return dy

    def ev_trig(t, y):      # Blocking -> Triggered
        return y[0] - bod.V_blk(y[1])
    ev_trig.direction = 1; ev_trig.terminal = True
    def ev_dvdt(t, y):      # Blocking -> TriggeredDV
        dy, _, _ = circuit(t, y, bod, state["mode"])
        if y[0] <= 0: return -1.0
        return y[0]*dy[0] - 0.9*bod.V_blk(y[1])*bod.dvdt_crT()
    ev_dvdt.direction = 1; ev_dvdt.terminal = True
    def ev_on(t, y):        # Triggered -> On
        return y[1] - bod.x_on
    ev_on.direction = -1; ev_on.terminal = True
    def ev_off(t, y):       # On -> Blocking
        return bod.i_dev(y[0], y[1], 3) - bod.I_HT()
    ev_off.direction = -1; ev_off.terminal = True

    while t0 < t_end:
        m = state["mode"]
        if m == 0:
            evs = [ev_trig, ev_dvdt]
        elif m in (1, 2):
            evs = [ev_on]
        else:
            evs = [ev_off]
        xevs = []
        for f, d, cb in extra_events:
            g = (lambda f: (lambda t, y: f(t, y, state["mode"])))(f)
            g.direction = d; g.terminal = True
            xevs.append(g)
        sol = solve_ivp(rhs, (t0, t_end), y, method="Radau", events=evs + xevs,
                        max_step=max_step if max_step else np.inf, rtol=1e-6, atol=1e-9,
                        dense_output=False)
        for k in range(len(sol.t)):
            _, ib, _ = circuit(sol.t[k], sol.y[:, k], bod, m)
            T.append(sol.t[k]); Y.append(sol.y[:, k].copy()); MODE.append(m); IB.append(ib)
        t0 = sol.t[-1]; y = sol.y[:, -1].copy()
        if sol.status == 1:  # event
            fired = [j for j, te in enumerate(sol.t_events) if len(te) > 0]
            j = fired[0]
            if j < len(evs):
                if m == 0 and evs[j] is ev_trig:
                    state["mode"] = 1
                elif m == 0 and evs[j] is ev_dvdt:
                    state["mode"] = 2
                elif m in (1, 2):
                    state["mode"] = 3
                else:
                    state["mode"] = 0
                state["log"].append((t0, m, state["mode"]))
            else:
                ev = extra_events.pop(j - len(evs))
                ev[2](t0, y, state)
            # nudge to avoid re-detecting the same event
            t0 = t0*(1 + 1e-12) + 1e-15
        elif sol.status < 0:
            raise RuntimeError(sol.message)
    return np.array(T), np.array(Y), np.array(MODE), np.array(IB), state


# ================================================================ scenario A: static I-V (current source sweep)
def scenario_A():
    """Curve-tracer sweep: voltage source 0 -> 1300 V -> 0 through R_L = 1 kOhm (load line)."""
    bod = BOD()
    V_pk = 1300.0; T_sw = 4e-3; R_L = 1e3

    def v_src(t):
        return V_pk*(1 - abs(2*t/T_sw - 1))

    def circuit(t, y, b, mode):
        v, xb = y
        i = (v_src(t) - v)/R_L
        dv = (i - b.i_dev(v, xb, mode))/b.C_j
        return np.array([dv, b.dxb(xb, mode)]), i, {}

    T, Y, MODE, IB, st = simulate(bod, circuit, [0.0, 1.0], T_sw, max_step=2e-6)
    v = Y[:, 0]
    # conduction current i_dev (static characteristic, without the C_j displacement current)
    IDEV = np.array([bod.i_dev(Y[k, 0], Y[k, 1], MODE[k]) for k in range(len(T))])
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    up = T <= T_sw/2
    ax.semilogy(v[up], np.maximum(IDEV[up], 1e-8), "-", color=GREYS[0], label="rising source voltage")
    ax.semilogy(v[~up], np.maximum(IDEV[~up], 1e-8), "--", color=GREYS[1], label="falling source voltage")
    ax.axhline(bod.I_BO, color="0.6", lw=0.6); ax.text(250, bod.I_BO*0.45, r"$I_{BO}$", fontsize=8)
    ax.axhline(bod.I_HT(), color="0.6", lw=0.6); ax.text(250, bod.I_HT()*1.3, r"$I_H$", fontsize=8)
    ax.axvline(bod.V_BOT(), color="0.6", lw=0.6); ax.text(bod.V_BOT()-160, 2e-7, r"$V_{BO}$", fontsize=8)
    ax.set_xlabel("$v$ (V)"); ax.set_ylabel("$i_{dev}$ (A)"); ax.set_xlim(-20, 1200); ax.set_ylim(1e-8, 3)
    ax.legend(loc="center right")
    fig.savefig(os.path.join(FIGDIR, "bod_static_iv.pdf"))
    plt.close(fig)
    print("A: mode log", st["log"][:6])
    # export the curve for pgfplots-free inspection
    np.savetxt(os.path.join(FIGDIR, "bod_static_iv.csv"), np.c_[T, v, IB, IDEV, MODE], delimiter=",",
               header="t_s,v_V,i_A,i_dev_A,mode", comments="")
    return st


# ================================================================ scenario B: thyristor protection
def scenario_B(T_j=25.0, fname="bod_protection.pdf"):
    bod = BOD(T_j=T_j)
    # external circuit
    R_s = 2.0; S = 200e6; V_s_max = 1600.0        # source ramp 200 V/us up to 1600 V
    R1 = 150.0; R2 = 220.0; C2 = 22e-9             # BOD series resistor, gate network
    V_GT = 2.5; r_g = 1.0; I_GT = 0.3; t_gd = 3e-6 # thyristor gate/turn-on data
    V_TT = 1.5; r_TT = 0.5e-3                      # thyristor on-state
    thy = {"t_trig": None, "on": False}

    def v_src(t):
        return min(S*t, V_s_max)

    def i_gk(vg):
        return (vg - V_GT)/r_g if vg > V_GT else 0.0

    def circuit(t, y, b, mode):
        v, xb, vg = y
        vs = v_src(t)
        if not thy["on"]:
            # thyristor off: anode node = vs - R_s*i, BOD branch through D1 (ideal), R1
            i = (vs - v - vg)/(R1 + R_s)
            i = max(i, 0.0)          # D1 blocks
            vA = vs - R_s*i
            iA = i
        else:
            # thyristor on: vA = V_TT + r_TT*iA, iA = (vs - vA)/R_s (BOD current negligible)
            vA = (V_TT + r_TT*vs/R_s)/(1 + r_TT/R_s)
            iA = (vs - vA)/R_s
            i = max((vA - v - vg)/R1, 0.0)
        dv = (i - b.i_dev(v, xb, mode))/b.C_j
        dvg = (i - vg/R2 - i_gk(vg))/C2
        return np.array([dv, b.dxb(xb, mode), dvg]), i, {"vA": vA, "iA": iA, "ig": i_gk(vg)}

    def ev_gate(t, y, mode):   # gate current reaches I_GT
        return i_gk(y[2]) - I_GT
    def cb_gate(t, y, state):
        if thy["t_trig"] is None:
            thy["t_trig"] = t
    def ev_fire(t, y, mode):
        return (t - thy["t_trig"] - t_gd) if thy["t_trig"] is not None else -1.0
    def cb_fire(t, y, state):
        thy["on"] = True

    T, Y, MODE, IB, st = simulate(bod, circuit, [0.0, 1.0, 0.0], 25e-6, max_step=20e-9,
                                  extra_events=[(ev_gate, 1, cb_gate), (ev_fire, 1, cb_fire)])
    # reconstruct node quantities
    vA = np.zeros_like(T); ig = np.zeros_like(T); iA = np.zeros_like(T)
    on_flag = np.zeros_like(T)
    thy2 = {"t_trig": thy["t_trig"], "on": False}
    for k in range(len(T)):
        if thy["t_trig"] is not None and T[k] >= thy["t_trig"] + t_gd:
            on = True
        else:
            on = False
        thy_saved = thy["on"]; thy["on"] = on
        _, _, aux = circuit(T[k], Y[k], bod, MODE[k])
        thy["on"] = thy_saved
        vA[k] = aux["vA"]; ig[k] = aux["ig"]; iA[k] = aux["iA"]; on_flag[k] = on

    tu = T*1e6
    fig, axs = plt.subplots(3, 1, figsize=(5.2, 5.6), sharex=True)
    axs[0].plot(tu, vA, "-", color=GREYS[0], label="$v_{AK}$ thyristor")
    axs[0].plot(tu, Y[:, 0], "--", color=GREYS[1], label="$v$ BOD")
    axs[0].axhline(bod.V_BOT(), color="0.6", lw=0.6)
    axs[0].text(0.3, bod.V_BOT()+40, r"$V_{BO}(T_j)$", fontsize=8)
    axs[0].set_ylabel("voltage (V)"); axs[0].legend(loc="upper right")
    axs[1].plot(tu, IB, "-", color=GREYS[0], label="$i$ BOD")
    axs[1].plot(tu, ig, "--", color=GREYS[1], label="$i_G$ thyristor gate")
    axs[1].set_ylabel("current (A)"); axs[1].legend(loc="upper right")
    axs[2].plot(tu, MODE, "-", color=GREYS[0], label="BOD mode $M$")
    axs[2].plot(tu, 3*on_flag, "--", color=GREYS[1], label="thyristor on (3)")
    axs[2].set_ylabel("mode"); axs[2].set_yticks([0, 1, 2, 3]); axs[2].set_xlabel(r"$t$ ($\mu$s)")
    axs[2].legend(loc="center right")
    fig.savefig(os.path.join(FIGDIR, fname)); plt.close(fig)
    print("B: BOD log", st["log"], " thy t_trig", thy["t_trig"], "on at", (thy["t_trig"] or 0)+t_gd)
    return st, thy


# ================================================================ scenario C: dv/dt triggering
def scenario_C():
    cases = [(25.0, 1500e6, "-", GREYS[0]), (25.0, 4000e6, "--", GREYS[1]), (150.0, 1500e6, ":", GREYS[0])]
    R_s = 10.0
    fig, axs = plt.subplots(2, 1, figsize=(5.0, 4.2), sharex=True)
    for Tj, S, ls, col in cases:
        bod = BOD(T_j=Tj)
        V_end = 0.95*1100.0
        def v_src(t, S=S):
            return min(S*t, V_end)
        def circuit(t, y, b, mode):
            v, xb = y
            i = (v_src(t) - v)/R_s
            dv = (i - b.i_dev(v, xb, mode))/b.C_j
            return np.array([dv, b.dxb(xb, mode)]), i, {}
        T, Y, MODE, IB, st = simulate(bod, circuit, [0.0, 1.0], 2.0e-6, max_step=2e-9)
        lab = r"$T_j$ = %d °C, d$v$/d$t$ = %d V/$\mu$s" % (Tj, S/1e6)
        axs[0].plot(T*1e9, Y[:, 0], ls, color=col, label=lab)
        axs[1].plot(T*1e9, MODE, ls, color=col)
        print("C:", lab, "log", st["log"])
    axs[0].set_ylabel("$v$ (V)"); axs[0].legend(loc="lower right")
    axs[1].set_ylabel("mode $M$"); axs[1].set_yticks([0, 1, 2, 3]); axs[1].set_xlabel("$t$ (ns)")
    fig.savefig(os.path.join(FIGDIR, "bod_dvdt.pdf")); plt.close(fig)


# ================================================================ scenario D: recovery (tq surrogate)
def scenario_D():
    bod = BOD()
    R_s = 20.0
    fig, axs = plt.subplots(2, 1, figsize=(5.0, 4.2), sharex=False)
    for k, (dt_re, ls, col) in enumerate([(50e-6, "-", GREYS[0]), (500e-6, "--", GREYS[1])]):
        # pulse: source 1300 V for 20 us (BOD conducts), then 0 V, then a ramp re-applied after dt_re
        t_off = 20e-6
        def v_src(t, dt_re=dt_re):
            if t < t_off:
                return min(t*400e6, 1300.0)
            elif t < t_off + dt_re:
                return 0.0
            else:
                return min((t - t_off - dt_re)*200e6, 1300.0)
        def circuit(t, y, b, mode):
            v, xb = y
            i = (v_src(t) - v)/R_s
            dv = (i - b.i_dev(v, xb, mode))/b.C_j
            return np.array([dv, b.dxb(xb, mode)]), i, {}
        t_end = t_off + dt_re + 12e-6
        T, Y, MODE, IB, st = simulate(bod, circuit, [0.0, 1.0], t_end, max_step=50e-9)
        # plot only the re-application window
        sel = T >= t_off + dt_re - 2e-6
        tt = (T[sel] - (t_off + dt_re))*1e6
        axs[0].plot(tt, Y[sel, 0], ls, color=col, label=r"re-applied after %d $\mu$s" % (dt_re*1e6))
        axs[0].plot(tt, bod.V_blk(Y[sel, 1]), ls, color="0.6", lw=0.7)
        axs[1].plot(tt, Y[sel, 1], ls, color=col)
        print("D: dt_re", dt_re, "log", st["log"])
    axs[0].set_ylabel("$v$ (V)"); axs[0].legend(loc="center right")
    axs[0].text(6.5, 300, r"thin grey: $V_{blk}(x_b)$", fontsize=8)
    axs[1].set_ylabel("$x_b$"); axs[1].set_xlabel(r"$t$ after re-application ($\mu$s)")
    fig.savefig(os.path.join(FIGDIR, "bod_recovery.pdf")); plt.close(fig)


# ================================================================ scenario E: temperature dependence
def scenario_E():
    bod = BOD()
    T = np.linspace(-40, 150, 200)
    fig, axs = plt.subplots(2, 2, figsize=(5.4, 3.8))
    axs[0, 0].plot(T, bod.V_BOT(T), color=GREYS[0]); axs[0, 0].set_ylabel("$V_{BO}$ (V)")
    axs[0, 1].semilogy(T, bod.I_HT(T)*1e3, color=GREYS[0]); axs[0, 1].set_ylabel("$I_H$ (mA)")
    axs[0, 1].set_yticks([20, 30, 40, 60]); axs[0, 1].set_yticklabels(["20", "30", "40", "60"])
    axs[0, 1].minorticks_off()
    axs[1, 0].semilogy(T, bod.I_DT(T)*1e6, color=GREYS[0]); axs[1, 0].set_ylabel(r"$I_D$ at $0.8V_{BO}$ ($\mu$A)")
    axs[1, 1].plot(T, bod.dvdt_crT(T)/1e6, color=GREYS[0]); axs[1, 1].set_ylabel(r"$(dv/dt)_{cr}$ (V/$\mu$s)")
    for ax in axs[1, :]:
        ax.set_xlabel("$T_j$ (°C)")
    # datasheet points
    axs[0, 1].plot([25, 125], [40, 20], "o", color=GREYS[0], ms=3)
    axs[1, 0].plot([25, 125], [0.2, 200], "o", color=GREYS[0], ms=3)
    axs[1, 1].plot([25, 150], [3000, 1000], "o", color=GREYS[0], ms=3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "bod_temperature.pdf")); plt.close(fig)


if __name__ == "__main__":
    scenario_A()
    scenario_B()
    scenario_C()
    scenario_D()
    scenario_E()
    print("figures written to", FIGDIR)
