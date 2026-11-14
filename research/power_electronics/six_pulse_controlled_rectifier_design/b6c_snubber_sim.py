"""Switched simulation of the B6C of b6c_sizing_formulas.tex with one RC snubber per thyristor.
Check of the snubber loss estimate (eq. PRs, Sec. 5), of the dv/dt bound (eq. dvdt) and of the step overshoot
(Table 1, chi = 0). Ideal thyristors (on/off resistance, turn-off at current zero, no reverse recovery: the
recovery energy E_L of eq. EL is not included). Data of the worked example (V1 = 440 V, Cs = 2.7 uF, Rs = 8.2 Ohm).
Modified nodal analysis with trapezoidal companion models.
Run:  python b6c_snubber_sim.py [alpha_deg] [V2_V]      e.g.  python b6c_snubber_sim.py 80"""
import sys
import numpy as np
from scipy.linalg import lu_factor, lu_solve
pi = np.pi

f = 50.0
w = 2*pi*f
alpha = np.deg2rad(float(sys.argv[1]) if len(sys.argv) > 1 else 36.3)
V2 = float(sys.argv[2]) if len(sys.argv) > 2 else 426.1     # secondary line voltage at V1,max
Lk, Rk = 52.3e-6, 2.74e-3                                     # commutation inductance, resistance per phase
Cs, Rs = 2.7e-6, 8.2                                          # arm snubber
Ld, Cd = 1.0e-3, 5.6e-3                                       # DC filter
Id0 = 1000.0                                                  # DC current: load resistance chosen for it
Vdo = 3*np.sqrt(2)/pi*V2
Vd0 = max(Vdo*np.cos(alpha) - 3/pi*w*Lk*Id0, 20.0)
Rl = Vd0/Id0
dt, T = 0.25e-6, 0.10                                         # step, simulated time (last cycle evaluated)
Ron, Roff = 1e-3, 1e8

# nodes: 0 a, 1 b, 2 c (bridge AC terminals), 3 p, 4 n, 5 x (filter capacitor), 6..11 snubber midpoints
NN = 12
TH = [(0, 3), (4, 2), (1, 3), (4, 0), (2, 3), (4, 1)]       # (anode, cathode) of T1..T6
fire = [np.deg2rad(30 + 60*k) + alpha for k in range(6)]     # firing angles of T1..T6
Vph = np.sqrt(2)*V2/np.sqrt(3)


def esrc(t):
    return Vph*np.array([np.sin(w*t), np.sin(w*t - 2*pi/3), np.sin(w*t + 2*pi/3)])


GL = (dt/2)/(Lk + Rk*dt/2)                                    # series Rk-Lk branch
kL = (Lk - Rk*dt/2)/(Lk + Rk*dt/2)
GLd, GC, GCd = dt/(2*Ld), 2*Cs/dt, 2*Cd/dt


def stamp(G, i, j, g):
    G[i, i] += g
    G[j, j] += g
    G[i, j] -= g
    G[j, i] -= g


cache = {}


def factor(state):
    if state not in cache:
        G = np.zeros((NN, NN))
        for k in range(3):
            G[k, k] += GL
        stamp(G, 3, 5, GLd)
        stamp(G, 5, 4, GCd)
        stamp(G, 5, 4, 1/Rl)
        for k, (an, ca) in enumerate(TH):
            stamp(G, an, ca, 1/Ron if state[k] else 1/Roff)
            stamp(G, an, 6 + k, 1/Rs)
            stamp(G, 6 + k, ca, GC)
        G[4, 4] += 1e-6                                       # weak reference of the floating DC side
        cache[state] = lu_factor(G)
    return cache[state]


iL = np.zeros(3)
iLd, iCd = Id0, 0.0
iCs = np.zeros(6)
v = np.zeros(NN)
v[3], v[4] = Vd0/2, -Vd0/2
v[5] = v[4] + Vd0
state = [False]*6
ER = np.zeros(6)
dvmax = vfmax = vrmax = 0.0
t_last = T - 1/f
e_old = esrc(0.0)
for s in range(1, int(T/dt) + 1):
    t = s*dt
    e = esrc(t)
    th = (w*t) % (2*pi)
    for k, (an, ca) in enumerate(TH):                         # 120 deg gate pulse
        if not state[k] and (th - fire[k]) % (2*pi) < np.deg2rad(120) and v[an] > v[ca]:
            state[k] = True
    rhs = np.zeros(NN)
    hL = kL*iL + GL*(e_old - v[:3])
    rhs[:3] += GL*e + hL
    hLd = iLd + GLd*(v[3] - v[5])
    rhs[3] -= hLd
    rhs[5] += hLd
    hCd = GCd*(v[5] - v[4]) + iCd
    rhs[5] += hCd
    rhs[4] -= hCd
    for k, (an, ca) in enumerate(TH):
        h = GC*(v[6 + k] - v[ca]) + iCs[k]
        rhs[6 + k] += h
        rhs[ca] -= h
    vn = lu_solve(factor(tuple(state)), rhs)
    iL = GL*(e - vn[:3]) + hL
    iLd = iLd + GLd*((vn[3] - vn[5]) + (v[3] - v[5]))
    iCd = GCd*((vn[5] - vn[4]) - (v[5] - v[4])) - iCd
    for k, (an, ca) in enumerate(TH):
        iCs[k] = GC*((vn[6 + k] - vn[ca]) - (v[6 + k] - v[ca])) - iCs[k]
        if t > t_last:
            ir = (vn[an] - vn[6 + k])/Rs
            ER[k] += ir*ir*Rs*dt
            vt = vn[an] - vn[ca]
            if not state[k]:
                if vt > 0:
                    dvmax = max(dvmax, (vt - (v[an] - v[ca]))/dt)
                vfmax, vrmax = max(vfmax, vt), max(vrmax, -vt)
    for k, (an, ca) in enumerate(TH):                         # turn-off at current zero
        if state[k] and vn[an] < vn[ca]:
            state[k] = False
    v, e_old = vn, e

P = ER*f
mu = np.arccos(np.cos(alpha) - np.sqrt(2)*w*Lk*iLd/V2) - alpha
Vpk = np.sqrt(2)*V2
Pform = f*7/8*Cs*Vpk**2*(np.sin(alpha)**2 + np.sin(alpha + mu)**2)
print(f"alpha={np.rad2deg(alpha):.1f} deg  Id={iLd:.0f} A  mu={np.rad2deg(mu):.1f} deg  P_Rs per arm {P.mean():.1f} W"
      f"  eq. PRs without E_L {Pform:.1f} W  (ratio {P.mean()/Pform:.2f})")
print(f"   forward dv/dt max {dvmax/1e6:.1f} V/us (bound Rs*Vpk/2Lk = {Rs*Vpk/(2*Lk)/1e6:.1f} V/us);"
      f" peak forward {vfmax:.0f} V, reverse {vrmax:.0f} V = {vrmax/Vpk:.2f} Vpk")
