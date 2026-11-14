#!/usr/bin/env python3
"""Numbers and figures of the note isop_rail_auxiliary_supply.tex (pwr-control/repositories-documentation).

The system is the model ps_isop_rail of pwr-control/modelization-and-control
(foundations_of_power_electronics/ps_isop_rail): four 250 kW phase-shift full bridges in series at the input
(3 kV catenary, emulated by a 4.5 kV battery) and in parallel at the output (750 V); the transformers of
modules 1+4 and 2+3 are paralleled on the AC side, one diode rectifier per pair.

Part 1 (analytic, instant): averaged phase-shift full bridge, operating point, nominal catenary, turns ratio,
input-voltage sharing of modules on separate rectifiers (linearized, and steady state with one module
different), voltage loop margins, input filter, gains of the balancing controls (rev. 01).
Part 2 (switched simulation with isop_sim.py, numpy + numba, about one minute): start and steady state,
sharing between and inside the pairs, mismatch cases, start from 0 V, emulator at 3 kV, DC magnetizing
current.
Part 3 (rev. 01, switched simulation, about one minute more): balancing controls switched on at 0.5 s (pair
loops on the relative phase, pair-to-pair trim of the phase shift, classic per-module loop).

Usage: python3 isop_rail_auxiliary_supply_analysis.py
       (prints the numbers, writes figures/*.pdf; the simulation results are cached in _cache/)
"""
import os
import sys

import numpy as np
from scipy.optimize import fsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import isop_sim  # noqa: E402  (switched simulator, same circuit and values as init_model.m of the model)

FIG = os.path.join(HERE, 'figures')
CACHE = os.path.join(HERE, '_cache')
os.makedirs(FIG, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
isop_sim.HERE = CACHE                         # run_case writes case_<name>.npz there

S = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']          # modules 1-4 (fixed order)
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.2,
                     'legend.frameon': False, 'axes.titlesize': 8, 'axes.titleweight': 'bold'})

P = isop_sim.P
N = 4


def section(title):
    print('\n' + '=' * 100 + '\n' + title + '\n' + '=' * 100)


# ============================================================================================ part 1
section('1. Data and operating point (values of init_model.m)')
n = P['n']
f = P['fpwm']
Ls = P['Ld1'] + P['Ld2']/n**2                  # total leakage referred to the primary
Req = 4*n**2*Ls*f                              # output resistance of the duty-cycle loss
vo = 750.0
R = P['Rload']
io = vo/R
imod = io/N
u_ocv = isop_sim.ocv_pack(P, P['soc0'])
i_string = vo*io/u_ocv
u_in = (u_ocv - i_string*(P['R0'] + P['R1'] + P['RLin']))/N
u_drop = 2*P['Vf_r'] + 2*P['Ron_sw']*n*imod
D = (vo + Req*imod + u_drop)/(n*u_in)
print(f'module: n = {n:.2f}, f = {f:.0f} Hz, Ls = {Ls*1e6:.2f} uH (Ld1 = Ld2 = {P["Ld1"]*1e6:.2f} uH), Lm = {P["Lm"]*1e3:.2f} mH, '
      f'Cin = {P["Cin"]*1e3:.2f} mF')
print(f'Req = 4 n^2 Ls f = {Req:.4f} Ohm, duty-cycle loss at {imod:.0f} A: {Req*imod:.1f} V')
print(f'load {R} Ohm: {vo:.0f} V, {io:.0f} A, {vo*io/1e6:.4f} MW = 4 x {vo*io/4e3:.1f} kW ({vo*io/4/250e3*100:.0f} % of 250 kW)')
print(f'emulator: open circuit {u_ocv:.1f} V at soc {P["soc0"]}, string current {i_string:.1f} A, module input {u_in:.1f} V')
print(f'phase shift needed D = {D:.4f} (rectifier diodes and switches: {u_drop:.2f} V)')
u_out_3kV = n*3000/N - Req*imod - u_drop
print(f'nominal catenary 3000 V: {3000/N:.0f} V per module, largest output (D = 1) at {imod:.0f} A: {u_out_3kV:.1f} V')
for vcat in (3600.0, 3900.0):
    print(f'  catenary {vcat:.0f} V: {vcat/N:.0f} V per module')

section('2. Turns ratio and leakage for a phase-shift bridge (design rule, D_max = 0.9, duty-cycle loss 10 %)')
Dmax, loss = 0.9, 0.10
for vcat in (3000.0, 2000.0):
    vmod = vcat/N
    n_req = vo/(vmod*Dmax*(1 - loss))
    Req_req = loss*n_req*vmod*Dmax/imod
    Ls_req = Req_req/(4*n_req**2*f)
    print(f'750 V at {vcat:.0f} V ({vmod:.0f} V per module): n = {n_req:.3f}, Req = {Req_req:.3f} Ohm, '
          f'Ls = {Ls_req*1e6:.2f} uH referred to the primary')

section('3. Input-voltage sharing of modules on separate rectifiers (common phase shift)')
g = vo/u_in*(n*D/Req - imod/u_in)
print(f'restoring conductance g = (vo/v)(nD/Req - i/v) = {g:.4f} S -> tau = Cin/g = {P["Cin"]/g*1e3:.3f} ms '
      f'(first order, output inductors left out)')
print(f'  term nD/Req = {n*D/Req:.3f} S, term i/v = {imod/u_in:.3f} S, factor vo/v = {vo/u_in:.4f}')

Vs = N*1170.9          # string voltage of the switched simulation at the operating point


def classic(req, nn):
    """steady state of four modules on separate rectifiers: v_o = n D v_k - Req_k i_k, equal input currents"""
    def eqs(x):
        v, i, d = x[0:4], x[4:8], x[8]
        r = [nn[k]*d*v[k] - req[k]*i[k] - vo for k in range(4)]
        iin = [vo*i[k]/v[k] for k in range(4)]
        r += [iin[0] - iin[1], iin[1] - iin[2], iin[2] - iin[3], sum(v) - Vs, sum(i) - io]
        return r
    x = fsolve(eqs, np.r_[np.full(4, Vs/4), np.full(4, imod), 0.78])
    return x[0:4], x[4:8], x[8]


CASES = [('equal modules', [Req]*4, [n]*4),
         ('module 1 turns +2 %', [Req]*4, [1.02*n, n, n, n]),
         ('module 2 leakage +5 %', [Req, 1.05*Req, Req, Req], [n]*4),
         ('module 2 leakage +20 %', [Req, 1.2*Req, Req, Req], [n]*4)]
classic_v = {}
for name, req, nn in CASES:
    v, i, d = classic(req, nn)
    classic_v[name] = v
    print(f'{name:24s}: v = {", ".join(f"{x:7.1f}" for x in v)} V, i = {", ".join(f"{x:6.1f}" for x in i)} A, D = {d:.4f}')

section('4. Voltage loop (averaged plant, PI as continuous, 1.5 samples of delay) and input filter')
ts = P['ts']
L2 = P['Lout']/2
C2 = P['Cout']
Rs = (Req/2 + P['RLout'])/2
fr = np.logspace(-1, 3.5, 20000)
s = 1j*2*np.pi*fr
Zl = R/(1 + s*R*C2)
G = n*u_in/vo*Zl/(Zl + Rs + s*L2)*np.exp(-1.5*ts*s)
print(f'output filter (two 250 uH branches on 1.2 mF): {1/(2*np.pi*np.sqrt(L2*C2)):.0f} Hz; plant gain {abs(G[0]):.3f} pu/pu')
for kp, ki in ((0.5, 18.0), (0.6, 100.0), (0.8, 60.0)):
    Lv = (kp + ki/s)*G
    k = np.where(np.abs(Lv) < 1)[0][0]
    ph = np.unwrap(np.angle(Lv))
    j = np.where(ph < -np.pi)[0]
    gm = -20*np.log10(abs(Lv[j[0]])) if len(j) else np.inf
    pole = -ki*abs(G[0])/(1 + kp*abs(G[0]))
    print(f'kp {kp:.1f} ki {ki:5.1f}: crossover {fr[k]:6.1f} Hz, phase margin {180 + np.degrees(np.angle(Lv[k])):5.0f} deg, '
          f'gain margin {gm:4.1f} dB, closed-loop pole (static plant) {pole:6.1f} rad/s')
Lf = P['L_line'] + P['Lin']
Cf = P['Cin']/N
print(f'input filter: {Lf*1e3:.1f} mH, {Cf*1e3:.2f} mF -> {1/(2*np.pi*np.sqrt(Lf*Cf)):.0f} Hz, Z0 = {np.sqrt(Lf/Cf):.2f} Ohm; '
      f'converter input -V^2/P = {-Vs**2/(vo*io):.1f} Ohm')

section('5. Balancing controls: gains of the two handles of a bridge (fundamental-harmonic model of a pair)')
X = 2*np.pi*f*Ls*n**2                          # leakage reactance referred to the secondary
Vb = 4/np.pi*u_in*np.sin(np.pi*D/2)            # fundamental of a bridge voltage (peak)
Vr = 4/np.pi*vo                                # fundamental of the rectifier input voltage (peak)
Pm = vo*imod
th = np.arcsin(2*X*Pm/(n*Vb*Vr))               # load angle: P = n Vb Vr sin(theta) / (2 X)
Kphi = n*Vb*Vr*np.cos(th)/(2*X*u_in)           # d(i_in,a - i_in,b)/d(phi), the two bridges moved by +-phi/2
Kamp = i_string*np.pi/2/np.tan(np.pi*D/2)      # d(i_in,k)/dD_k with the angle fixed (amplitude only)
dI20 = 2*i_string*0.2/2.2                      # input-current difference, 20 % more leakage in one bridge
print(f'X = {X:.4f} Ohm, bridge fundamental {Vb:.0f} V peak, rectifier fundamental {Vr:.0f} V peak, '
      f'load angle {np.degrees(th):.1f} deg')
print(f'relative phase of the two bridges: K_phi = {Kphi:.0f} A/rad; with 20 % more leakage in one bridge the input '
      f'currents differ by {dI20:.1f} A -> phi = {np.degrees(dI20/Kphi):.2f} deg (time shift {dI20/Kphi/(2*np.pi*f)*1e6:.2f} us)')
print(f'amplitude (phase shift of one bridge, angle fixed): K_amp = {Kamp:.0f} A per unit; the same case would need '
      f'+-{dI20/(2*Kamp):.3f} on the two bridges')
B = isop_sim.BAL
wcp = Kphi*B['kp_phi']/P['Cin']
print(f'pair loop: kp {B["kp_phi"]:.1e} rad/V, ki {B["ki_phi"]:.4f} rad/(V s), limit {np.degrees(B["phi_max"]):.1f} deg '
      f'-> crossover {wcp:.0f} rad/s ({wcp/2/np.pi:.0f} Hz), PI zero {B["ki_phi"]/B["kp_phi"]:.0f} rad/s')
Kd = n*vo/Req                                  # d(i_in,k)/dD with separate rectifiers, from (avg)
poles = np.roots([P['Cin'], g + 4*Kd*B['kp_d'], 4*Kd*B['ki_d']])
print(f'pair-to-pair trim: d(i_in)/dD = n vo/Req = {Kd:.0f} A per unit and module; kp {B["kp_d"]:.1e} 1/V, '
      f'ki {B["ki_d"]:.3f} 1/(V s), limit {B["d_max"]:.2f} -> poles {", ".join(f"{p_.real:.0f}" for p_ in poles)} rad/s')
print(f'classic per-module loop, trims +-x on the two bridges of a pair: modulator of the model (leg B moved, the '
      f'pulse centre moves by x T/4): d(i_a - i_b)/dx = -pi K_phi + 2 K_amp = {-np.pi*Kphi + 2*Kamp:.0f} A (wrong sign); '
      f'symmetric modulator: +2 K_amp = {2*Kamp:.0f} A (right sign, {np.pi*Kphi/(2*Kamp):.1f} times smaller than the '
      f'phase effect)')


# ============================================================================================ part 2
def case(name, run):
    path = os.path.join(CACHE, f'case_{name}.npz')
    if not os.path.exists(path):
        run()
    d = np.load(path)
    return d['rec'], d['win']


v0 = isop_sim.ocv_pack(P, P['soc0'])/4
MM = {'mm_n1': (0, dict(n=1.02)), 'mm_L2s': (1, dict(Ld1=1.05*P['Ld1'], Ld2=1.05*P['Ld2'])),
      'mm_L2': (1, dict(Ld1=1.2*P['Ld1'], Ld2=1.2*P['Ld2']))}


def run_mm(w):
    mod = [{} for _ in range(4)]
    mod[MM[w][0]] = MM[w][1]
    isop_sim.run_case(w, 1.0, mod=mod, win_t=0.996, win_ms=4.0)


def run_3kV():
    p = dict(P)
    p['N'] = int(round(P['N']*3000.0/isop_sim.ocv_pack(P, P['soc0'])))
    isop_sim.run_case('catenary_3kV', 0.8, p=p, win_t=0.796, win_ms=4.0)


section('6. Switched simulation (isop_sim.py)')
rec, win = case('nominal', lambda: isop_sim.run_case('nominal', 1.75, win_t=1.746, win_ms=4.0))
rz, wz = case('zero_init', lambda: isop_sim.run_case('zero_init', 0.5, precharged=False, win_t=0.0, win_ms=12.0))
rc, wc = case('catenary_3kV', run_3kV)
rp, _ = case('split_pairs_ss', lambda: isop_sim.run_case('split_pairs_ss', 0.1, vcin0=[v0 + 80, v0 - 80, v0 - 80, v0 + 80],
                                                          pd_fixed=0.7823, win_t=0.096, win_ms=4.0))
ri, _ = case('split_in_pair_ss', lambda: isop_sim.run_case('split_in_pair_ss', 0.3, vcin0=[v0, v0 + 80, v0 - 80, v0],
                                                            pd_fixed=0.7823, win_t=0.296, win_ms=4.0))
mm = {w: case(w, lambda w=w: run_mm(w)) for w in MM}

t = rec[:, 0]
vout = rec[:, 1]
k95 = np.argmax(vout >= 0.95*750)
k99 = np.argmax(vout >= 0.99*750)
print(f'start (DoDC on): 95 % at {t[k95]*1e3:.0f} ms, 99 % at {t[k99]*1e3:.0f} ms, max {vout.max():.1f} V; '
      f'max module input voltage {rec[:, 25].max():.0f} V; max |iLm| {rec[:, 29].max():.1f} A')
for tt in (0.005, 0.2, 0.5, 0.75, 1.0):
    kk = np.argmin(abs(t - tt))
    print(f'  t = {tt*1e3:6.0f} ms: mean magnetizing current {rec[kk-8:kk+8, 11].mean():6.2f} A, output {vout[kk-8:kk+8].mean():6.1f} V')
w = win[win[:, 0] > 0]
nper = int(round(1/f/P['h']))
w = w[:(len(w)//nper)*nper]
print(f'steady state: output mean {w[:, 1].mean():.1f} V (sampled {vout[-16:].mean():.1f} V), ripple {np.ptp(w[:, 1]):.2f} V p-p, '
      f'phase shift {w[:, 25].mean():.4f}, module input {w[:, 2].mean():.1f} V, primary {np.sqrt(np.mean(w[:, 6]**2)):.1f} A rms / '
      f'{np.abs(w[:, 6]).max():.1f} A peak, magnetizing peak {np.abs(w[:, 18]).max():.1f} A, rectifiers {w[:, 22].mean():.1f} / '
      f'{w[:, 23].mean():.1f} A, string current {rec[-16:, 3].mean():.1f} A')
mz = rz[:, 0] <= 0.012
print(f'start from 0 V: module input up to {wz[:, 2].max():.0f} V, string current up to {rz[mz, 24].max():.0f} A')
wcc = wc[wc[:, 0] > 0]
print(f'emulator at 3 kV: output {wcc[:, 1].mean():.1f} V, phase shift {wcc[:, 25].mean():.3f}, module input {wcc[:, 2].mean():.1f} V')
dp = np.abs((rp[:, 7] + rp[:, 10] - rp[:, 8] - rp[:, 9])/2)
di = np.abs((ri[:, 8] - ri[:, 9])/2)


def envelope_tau(tv, dv, t1, t2):
    """time constant of a decaying envelope between t1 and t2 (log-linear fit of the local maxima)"""
    m = (tv >= t1) & (tv <= t2)
    x, y = tv[m], dv[m]
    pk = [j for j in range(1, len(y) - 1) if y[j] >= y[j - 1] and y[j] >= y[j + 1] and y[j] > 1e-3]
    if len(pk) < 3:
        pk = list(range(len(y)))
    c = np.polyfit(x[pk], np.log(y[pk]), 1)
    return -1/c[0]


print(f'sharing, 80 V initial deviation at the operating point (fixed phase shift): between the pairs envelope tau '
      f'{envelope_tau(rp[:, 0], dp, 0.0, 0.02)*1e3:.2f} ms, inside a pair tau {envelope_tau(ri[:, 0], di, 0.05, 0.3)*1e3:.1f} ms')
sim_v = {}
for wname, label in (('mm_n1', 'module 1 turns +2 %'), ('mm_L2s', 'module 2 leakage +5 %'), ('mm_L2', 'module 2 leakage +20 %')):
    ww = mm[wname][1]
    ww = ww[ww[:, 0] > 0]
    v = [ww[:, 2 + k].mean() for k in range(4)]
    sim_v[label] = v
    print(f'{label:24s}: pairs (simulation) v = {", ".join(f"{x:7.1f}" for x in v)} V; primary rms '
          f'{", ".join(f"{np.sqrt(np.mean(ww[:, 6 + k]**2)):5.0f}" for k in range(4))} A, peak '
          f'{", ".join(f"{np.abs(ww[:, 6 + k]).max():5.0f}" for k in range(4))} A')
sim_v['equal modules'] = [w[:, 2 + k].mean() for k in range(4)]

# ---------------------------------------------------------------------------------------- figures
# 1. steady-state waveforms
n2 = int(round(2/f/P['h']))
ww = w[-n2:]
tt = (ww[:, 0] - ww[0, 0])*1e6
fig, ax = plt.subplots(3, 1, figsize=(6.3, 4.6), sharex=True)
ax[0].plot(tt, ww[:, 14], color=S[0])
ax[0].set_ylabel('V')
ax[0].set_title(f'(a) bridge voltage of module 1 (phase shift {w[:, 25].mean():.2f})', loc='left')
ax[1].plot(tt, ww[:, 6], color=S[0], label='primary current')
ax[1].plot(tt, ww[:, 18], color=S[1], label='magnetizing current')
ax[1].set_ylabel('A')
ax[1].set_title('(b) transformer of module 1', loc='left')
ax[1].set_ylim(-560, 800)
ax[1].legend(loc='upper center', ncol=2)
ax[2].plot(tt, ww[:, 10], color=S[0], label='secondary, module 1')
ax[2].plot(tt, ww[:, 13], color=S[3], ls='--', label='secondary, module 4')
ax[2].plot(tt, ww[:, 22], color=S[2], label='output of rectifier 1')
ax[2].set_ylabel('A')
ax[2].set_title('(c) pair 1+4 on rectifier 1', loc='left')
ax[2].legend(loc='lower right', ncol=3)
ax[2].set_ylim(-900, 1100)
ax[2].set_xlabel(r'time [$\mu$s]')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_waveforms.pdf'))
plt.close(fig)

# 2. input-voltage sharing
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.6), gridspec_kw={'width_ratios': [1.15, 1, 1]})
ax[0].semilogy(rp[:, 0]*1e3, np.maximum(dp, 1e-2), color=S[0], label='between the pairs')
ax[0].semilogy(ri[:, 0]*1e3, np.maximum(di, 1e-2), color=S[1], label='inside a pair')
ax[0].set_xlim(0, 300)
ax[0].set_ylim(0.1, 200)
ax[0].set_xlabel('time [ms]')
ax[0].set_ylabel('|deviation| [V]')
ax[0].set_title('(a) 80 V initial deviation', loc='left')
ax[0].legend(loc='lower right')
labels = ['equal modules', 'module 1 turns +2 %', 'module 2 leakage +5 %', 'module 2 leakage +20 %']
ticks = ['equal', '$n_1$\n+2 %', '$L_{s2}$\n+5 %', '$L_{s2}$\n+20 %']
x = np.arange(4)
bw = 0.19
for j, (data, title) in enumerate(((sim_v, '(b) pairs on one rectifier'),
                                   (classic_v, '(c) one rectifier per module'))):
    a = ax[1 + j]
    for k in range(4):
        a.bar(x + (k - 1.5)*(bw + 0.01), [data[lb][k] for lb in labels], bw, color=S[k], label=f'module {k + 1}')
    a.axhline(Vs/4, color=INK2, lw=0.8, ls='--')
    a.set_xticks(x)
    a.set_xticklabels(ticks, fontsize=7)
    a.set_ylim(800, 1550)
    a.grid(axis='x', visible=False)
    a.set_title(title, loc='left')
ax[1].set_ylabel('module input voltage [V]')
ax[1].legend(loc='upper left', ncol=2, fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_sharing.pdf'))
plt.close(fig)

# 3. start, nominal catenary, start from 0 V, DC magnetizing current
fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.4))
m = t <= 0.8
ax[0, 0].plot(t[m], vout[m], color=S[0], label='emulator 4.7 kV (the model)')
ax[0, 0].plot(rc[:, 0], rc[:, 1], color=S[1], label='emulator 3 kV')
ax[0, 0].axhline(750, color=INK2, lw=0.8, ls='--')
ax[0, 0].set_ylim(0, 820)
ax[0, 0].set_ylabel('V')
ax[0, 0].set_title('(a) output voltage', loc='left')
ax[0, 0].legend(loc='lower right')
ax[0, 1].plot(t[m], rec[m, 2], color=S[0])
ax[0, 1].plot(rc[:, 0], rc[:, 2], color=S[1])
ax[0, 1].set_ylim(0, 1.05)
ax[0, 1].set_ylabel('pu of $\\pi$')
ax[0, 1].set_title('(b) phase shift (PI output)', loc='left')
wz2 = wz[wz[:, 0] > 0]
mp = t <= 0.012
ax[1, 0].plot(wz2[:, 0]*1e3, wz2[:, 2], color=S[1], label='string from 0 V')
ax[1, 0].plot(t[mp]*1e3, rec[mp, 7], color=S[0], label='string precharged')
ax[1, 0].axhline(Vs/4, color=INK2, lw=0.8, ls='--')
ax[1, 0].set_xlabel('time [ms]')
ax[1, 0].set_ylabel('V')
ax[1, 0].set_title('(c) input voltage of a module at the start', loc='left')
ax[1, 0].legend(loc='lower right')
pm = np.convolve(rec[:, 11], [0.5, 0.5], mode='same')
ax[1, 1].plot(t, pm, color=S[0])
ax[1, 1].set_xlim(0, 1.75)
ax[1, 1].set_xlabel('time [s]')
ax[1, 1].set_ylabel('A')
ax[1, 1].set_title('(d) mean magnetizing current (one period)', loc='left')
ax[0, 0].set_xlabel('time [s]')
ax[0, 1].set_xlabel('time [s]')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_start.pdf'))
plt.close(fig)

# ============================================================================================ part 3
section('7. Balancing controls in the switched simulation (module 2 leakage +20 % unless named, loops on at 0.5 s)')
BC = ['bal_phase_d', 'bal_phase', 'bal_ivs', 'bal_ivs_sym', 'bal_phase_d_L2s', 'bal_phase_d_n1', 'bal_phase_d_nom',
      'bal_phase_d_n1_raw']
bal = {nm: case(nm, lambda nm=nm: isop_sim.run_bal(nm)) for nm in BC}


def deviation(r):
    vc = r[:, 7:11]
    return np.abs(vc - vc.mean(1, keepdims=True)).max(1)/vc.mean(1)


def settling(r, lim=0.02):
    """time after 0.5 s from which the largest module deviation stays below lim (nan: never)"""
    dev = deviation(r)
    bad = np.where((r[:, 0] >= 0.5) & (dev >= lim))[0]
    if bad.size == 0:
        return 0.0
    return np.nan if bad[-1] == len(dev) - 1 else r[bad[-1] + 1, 0] - 0.5


for nm in BC:
    r, ww = bal[nm]
    ww = ww[ww[:, 0] > 0]
    dev = deviation(r)
    v = [ww[:, 2 + k].mean() for k in range(4)]
    print(f'{nm:19s}: v = {", ".join(f"{x:7.1f}" for x in v)} V; largest deviation {dev[np.argmin(abs(r[:, 0] - 0.49))]*100:4.1f} % '
          f'before, {dev[-1]*100:5.2f} % at the end, below 2 % after {settling(r)*1e3:5.1f} ms')
    print(f'{"":19s}  phi14 {np.degrees(r[-1, 30]):5.2f} deg, phi23 {np.degrees(r[-1, 31]):5.2f} deg (largest '
          f'{np.degrees(np.abs(r[:, 30:32]).max()):5.2f} deg), trims {", ".join(f"{x:+.4f}" for x in r[-1, 32:36])}; '
          f'primary rms {", ".join(f"{np.sqrt(np.mean(ww[:, 6 + k]**2)):4.0f}" for k in range(4))} A, peak '
          f'{", ".join(f"{np.abs(ww[:, 6 + k]).max():4.0f}" for k in range(4))} A; mean magnetizing current '
          f'{", ".join(f"{r[-16:, 11 + k].mean():6.1f}" for k in range(4))} A')

# 4. balancing controls
fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.4), sharex=True)
panels = [('bal_phase_d', '(a) pair loops (phase) + pair-to-pair trim', (900, 1500)),
          ('bal_phase', '(b) pair loops (phase) only', (900, 1500)),
          ('bal_ivs', "(c) classic loop per module, model's modulator", (0, 2500)),
          ('bal_ivs_sym', '(d) classic loop per module, symmetric modulator', (900, 1500))]
for a, (nm, title, yl) in zip(ax.flat, panels):
    r = bal[nm][0]
    m = (r[:, 0] >= 0.45) & (r[:, 0] <= 0.75)
    for k in range(4):
        a.plot(r[m, 0], r[m, 7 + k], color=S[k], ls='--' if k == 3 else '-', label=f'module {k + 1}')
    a.axhline(Vs/4, color=INK2, lw=0.8, ls=':')
    a.axvline(0.5, color=INK2, lw=0.8, ls=':')
    a.set_ylim(*yl)
    a.set_xlim(0.45, 0.75)
    a.set_title(title, loc='left')
for a in ax[:, 0]:
    a.set_ylabel('module input voltage [V]')
for a in ax[1, :]:
    a.set_xlabel('time [s]')
ax[0, 0].legend(loc='upper right', ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_balancing.pdf'))
plt.close(fig)
print('\nfigures:', sorted(os.listdir(FIG)))
