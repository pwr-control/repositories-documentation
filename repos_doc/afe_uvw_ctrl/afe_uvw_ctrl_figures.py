#!/usr/bin/env python3
"""Figures and numbers of the note afe_uvw_ctrl (pwr-control/repositories-documentation).

Run from this folder:   python3 afe_uvw_ctrl_figures.py
Writes into figures/:   res_pi_bode.pdf, res_pi_poles.pdf, res_pi_closed_loop.pdf, afe_uvw_ctrl_numbers.txt
Needs numpy and matplotlib only. The numbers are those of the model afe_uvw_three_phase (control period
125 us, 50 Hz, resonant PI kp = 0.6, ki = 35 1/s; delta_res = 0.015 in the class ctrl_afe_setup, whose
resonator is forward Euler) and of the library block current_ctrl_uvw (rev. 03: forward-backward Euler
resonator, delta_res = 0.005 in the example structure and in the model init), on the plant of the test
bench of the package (RL branches x_l = 0.1 pu, r = 0.01 pu, one step of output delay) and on the plant
of the model.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'lines.linewidth': 1.1,
                     'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 120})

# ------------------------------------------------------------------ data
ts = 125e-6                 # [s] control period
f0 = 50.0                   # [Hz] grid frequency = resonance
w = 2*np.pi*f0              # [rad/s]
kp, ki = 0.6, 35.0          # resonant PI of the model (LVRT setting of init_model.m)
D_MODEL = 0.015             # delta_res of the model's class (forward Euler resonator)
D_BLOCK = 0.005             # delta_res of the block (forward-backward Euler resonator, library rev. 03)
# test bench plant (afe_uvw_ctrl_test_param.m) and model plant (690 V, 250 kW, 500 uH, 5 % of X at 25 Hz)
plants = {'bench': dict(x_l=0.1, r=0.01),
          'model 690 V': dict(x_l=w*500e-6/(690*np.sqrt(2)/np.sqrt(3)/(250e3/np.sqrt(3)/690/0.9*np.sqrt(2))),
                              r=157*0.05*500e-6/(690*np.sqrt(2)/np.sqrt(3)/(250e3/np.sqrt(3)/690/0.9*np.sqrt(2))))}
out_delay = 1
n_per = 1/f0/ts             # samples per grid period

lines = []
def say(s=''):
    print(s)
    lines.append(s)

# ------------------------------------------------------------------ controllers
def c_cont(s, d):
    """continuous resonant PI C(s) = kp + ki s/(s^2 + 2 d w s + w^2)"""
    return kp + ki*s/(s**2 + 2*d*w*s + w**2)

def polys_euler(d):
    """forward Euler (the model: Aresd = I + A ts): den = (z-1)(z-1+2 d w ts) + (w ts)^2, num = kp den + ki ts (z-1)"""
    a = 1 - 2*d*w*ts
    den = np.array([1.0, -(1 + a), a + (w*ts)**2])
    num = kp*den + np.array([0.0, ki*ts, -ki*ts])
    return num, den

def polys_fb(d):
    """forward Euler on x1, backward (with the new x1) on x2 (the block since library rev. 03):
    den = z^2 - (2 - (w ts)^2 - 2 d w ts) z + 1 - 2 d w ts, num = kp den + ki ts (z-1)"""
    den = np.array([1.0, -(2 - (w*ts)**2 - 2*d*w*ts), 1 - 2*d*w*ts])
    num = kp*den + np.array([0.0, ki*ts, -ki*ts])
    return num, den

def resp(num, den, f):
    z = np.exp(1j*2*np.pi*f*ts)
    return np.polyval(num, z)/np.polyval(den, z)

def free_response(d, fb, t_end=0.3):
    """free response of x2 from x2(0) = 1 with e = 0; fb = True: forward-backward Euler, False: forward Euler"""
    n = int(t_end/ts)
    t = np.arange(n)*ts
    x = np.array([0.0, 1.0]); y = np.zeros(n)
    for k in range(n):
        y[k] = x[1]
        x1 = x[0] + ts*x[1]
        x2 = x[1] + ts*(-w**2*(x1 if fb else x[0]) - 2*d*w*x[1])
        x = np.array([x1, x2])
    return t, y

# the three resonators of the note
CASES = [('forward Euler, delta 0.015 (the model)',          polys_euler(D_MODEL), 'C3', '--'),
         ('forward-backward Euler, delta 0.015',             polys_fb(D_MODEL),    'C0', ':'),
         ('forward-backward Euler, delta 0.005 (the block)', polys_fb(D_BLOCK),    'C2', '-')]

# ------------------------------------------------------------------ numbers
say('afe_uvw_ctrl_figures.py: numbers of the note')
say(f'ts = {ts*1e6:.0f} us, f0 = {f0:.0f} Hz, w = {w:.4f} rad/s, w ts = {w*ts:.6f}')
say(f'resonant PI: kp = {kp}, ki = {ki} 1/s; delta = {D_MODEL} in the model (forward Euler), {D_BLOCK} in the block (forward-backward Euler)')
say()
say('--- continuous controller')
for d in (D_MODEL, D_BLOCK):
    sc = -d*w + 1j*w*np.sqrt(1 - d**2)
    say(f'delta = {d}: gain of the resonator at f0 ki/(2 delta w) = {ki/(2*d*w):.3f} -> |C(j w)| = {abs(c_cont(1j*w, d)):.3f} '
        f'({20*np.log10(abs(c_cont(1j*w, d))):.1f} dB), -3 dB width 2 delta w = {2*d*w:.3f} rad/s = {2*d*w/2/np.pi:.3f} Hz, '
        f'poles s = {sc.real:.3f} +- j{sc.imag:.3f} rad/s, decay per grid period exp(-2 pi delta) = {np.exp(-d*2*np.pi):.4f}')
say()
say('--- forward Euler (the model, ctrl_afe_setup: x1+ = x1 + ts x2, x2+ = x2 + ts(-w^2 x1 - 2 delta w x2 + ki e)), delta 0.015')
num_e, den_e = polys_euler(D_MODEL)
pe = np.roots(den_e)
rad_e = np.sqrt(1 - 2*D_MODEL*w*ts + (w*ts)**2)
say(f'poles z = {pe[0].real:.6f} +- j{abs(pe[0].imag):.6f}, |z| = {abs(pe[0]):.6f} (formula sqrt(1 - 2 delta w ts + (w ts)^2) = {rad_e:.6f})')
say(f'angle of the poles {np.angle(pe[0]):.6f} rad -> resonance at {np.angle(pe[0])/ts/2/np.pi:.3f} Hz')
d_lim = w*ts/2
say(f'stable (|z| < 1) only for delta > w ts/2 = {d_lim:.5f}')
sigma = np.log(abs(pe[0]))/ts
say(f'equivalent continuous damping: ln|z|/ts = +{sigma:.3f} rad/s -> delta_eff = {-sigma/w:.5f}')
say(f'growth per grid period |z|^{n_per:.0f} = {abs(pe[0])**n_per:.4f}')
g_e = abs(resp(num_e, den_e, f0))
say(f'|C_d(e^(j w ts))| at f0 = {g_e:.3f} ({20*np.log10(g_e):.1f} dB), resonator alone {g_e - kp:.3f}')
say()
say('--- forward-backward Euler (the block, library rev. 03: x1+ = x1 + ts x2, x2+ = x2 + ts(-w^2 x1+ - 2 delta w x2 + ki e))')
for d in (D_MODEL, D_BLOCK, 0.0):
    num_f, den_f = polys_fb(d)
    pf = np.roots(den_f)
    g_f = abs(resp(num_f, den_f, f0))
    say(f'delta = {d}: poles z = {pf[0].real:.6f} +- j{abs(pf[0].imag):.6f}, |z| = {abs(pf[0]):.6f} '
        f'(formula sqrt(1 - 2 delta w ts) = {np.sqrt(1 - 2*d*w*ts):.6f}), resonance at {np.angle(pf[0])/ts/2/np.pi:.3f} Hz, '
        f'|C_d| at f0 = {g_f:.3f} ({20*np.log10(g_f):.1f} dB), resonator alone {g_f - kp:.3f}, '
        f'decay per grid period |z|^{n_per:.0f} = {abs(pf[0])**n_per:.4f}')
g_res = g_e - kp
say(f'damping of the forward-backward resonator with the same gain at f0 as the model\'s: ki/(2 delta_eq w) = {g_res:.3f} '
    f'-> delta_eq = {ki/(2*g_res*w):.5f}')
num_f, den_f = polys_fb(D_MODEL)
num_b, den_b = polys_fb(D_BLOCK)
pb = np.roots(den_b)
say()

# closed loop with an RL plant: P(z) = K/(z - e_p), K = (1 - e_p)/r, loop L = C z^-out_delay P / scale
def closed_loop(num, den, x_l, r, scale=1):
    a = r*w/x_l
    e_p = np.exp(-a*ts)
    K = (1 - e_p)/r/scale
    den_ol = np.polymul(den, np.polymul(np.array([1.0] + [0.0]*out_delay), np.array([1.0, -e_p])))
    num_ol = K*num
    char = np.polyadd(den_ol, num_ol)
    poles = np.roots(char)
    f = np.logspace(0, np.log10(1/ts/2), 20000)
    z = np.exp(1j*2*np.pi*f*ts)
    L = np.polyval(num_ol, z)/np.polyval(den_ol, z)
    S = 1/(1 + L)
    # crossover and phase margin
    i = np.where(np.diff(np.sign(abs(L) - 1)) != 0)[0]
    fc = f[i[-1]] if len(i) else np.nan
    pm = 180 + np.degrees(np.angle(L[i[-1]])) if len(i) else np.nan
    return dict(poles=poles, f=f, L=L, S=S, fc=fc, pm=pm, e_p=e_p, K=K)

cl = {}
for name, pl in plants.items():
    say(f'--- closed loop, {name} plant: x_l = {pl["x_l"]:.4f} pu, r = {pl["r"]:.4f} pu, output delay {out_delay} step')
    for lab, (num, den), _, _ in CASES:
        c = closed_loop(num, den, **pl)
        cl[(name, lab)] = c
        s50 = abs(np.interp(f0, c['f'], abs(c['S'])))
        zr = max(abs(c['poles']))
        say(f'  {lab:48s}: max |z| = {zr:.6f} (time constant {-ts/np.log(zr)*1e3:.1f} ms), crossover {c["fc"]:.0f} Hz, '
            f'phase margin {c["pm"]:.1f} deg, |S| at f0 = {s50:.4f} ({20*np.log10(s50):.1f} dB)')
        say(f'  {"":48s}  poles: ' + ', '.join(f'{p.real:.5f}{p.imag:+.5f}j (|z| {abs(p):.5f})' for p in c['poles']))
say()
say('--- closed loop (bench plant) when the vector limiter divides the output by scale (loop gain / scale): largest |z|')
for scale in [1, 2, 5, 10]:
    say(f'  scale {scale:2d}: ' + ', '.join(f'{lab} {max(abs(closed_loop(num, den, scale=scale, **plants["bench"])["poles"])):.6f}'
                                           for lab, (num, den), _, _ in CASES))
say()

# ------------------------------------------------------------------ more numbers
def obs_gains(poles, ts):
    z = np.exp(np.array(poles)*ts)
    l1 = 2 - (z[0] + z[1])
    l2 = (z[0]*z[1] - 1 + l1)/ts
    return l1/ts, l2/ts
say('--- double-integrator observers (discrete gains, poles at exp(p ts); continuous equivalents kx = -(p1+p2), kv = p1 p2)')
for name, poles in {'FRT voltage rate [-0.1 -0.4] w': [-0.1*w, -0.4*w], 'islanding ROCOF [-20 -80] rad/s': [-20.0, -80.0]}.items():
    kx, kv = obs_gains(poles, ts)
    say(f'  {name}: kx = {kx:.3f} 1/s, kv = {kv:.2f} 1/s^2 (continuous {-(poles[0]+poles[1]):.3f}, {poles[0]*poles[1]:.2f})')
with open(os.path.join(FIG, 'afe_uvw_ctrl_numbers.txt'), 'w') as fh:
    fh.write('\n'.join(lines) + '\n')

# ------------------------------------------------------------------ figure 1: Bode of the resonant PI
LBL = {'forward Euler, delta 0.015 (the model)': r'forward Euler, $\delta = 0.015$ (the model)',
       'forward-backward Euler, delta 0.015': r'forward-backward Euler, $\delta = 0.015$',
       'forward-backward Euler, delta 0.005 (the block)': r'forward-backward Euler, $\delta = 0.005$ (the block)'}
f = np.logspace(0, np.log10(2000), 4000)
fz = np.linspace(40, 60, 4001)
fig, ax = plt.subplots(1, 3, figsize=(6.7, 2.5), constrained_layout=True)
ax[0].semilogx(f, 20*np.log10(abs(c_cont(1j*2*np.pi*f, D_MODEL))), 'k', label=r'continuous $C(s)$, $\delta = 0.015$')
ax[1].semilogx(f, np.degrees(np.angle(c_cont(1j*2*np.pi*f, D_MODEL))), 'k')
ax[2].plot(fz, abs(c_cont(1j*2*np.pi*fz, D_MODEL)), 'k')
for lab, (num, den), col, ls in CASES:
    ax[0].semilogx(f, 20*np.log10(abs(resp(num, den, f))), color=col, ls=ls, label=LBL[lab])
    ax[1].semilogx(f, np.degrees(np.angle(resp(num, den, f))), color=col, ls=ls)
    ax[2].plot(fz, abs(resp(num, den, fz)), color=col, ls=ls)
ax[0].set_xlabel('f [Hz]'); ax[0].set_ylabel('|C| [dB]'); ax[0].set_title('magnitude', fontsize=8)
ax[0].legend(loc='upper right', fontsize=6)
ax[1].set_xlabel('f [Hz]'); ax[1].set_ylabel('phase [deg]'); ax[1].set_title('phase', fontsize=8)
ax[1].set_yticks([-180, -90, 0, 90, 180])
ax[2].set_xlabel('f [Hz]'); ax[2].set_ylabel('|C| [pu/pu]'); ax[2].set_title('magnitude around $f_0$', fontsize=8)
fig.savefig(os.path.join(FIG, 'res_pi_bode.pdf'))

# ------------------------------------------------------------------ figure 2: pole radius and free response
dd = np.linspace(0, 0.05, 501)
fig, ax = plt.subplots(1, 2, figsize=(6.7, 2.8), constrained_layout=True)
ax[0].plot(dd, np.exp(-dd*w*ts), 'k', label=r'exact $e^{-\delta\omega t_s}$')
ax[0].plot(dd, np.sqrt(1 - 2*dd*w*ts + (w*ts)**2), 'C3--', label='forward Euler (the model)')
ax[0].plot(dd, np.sqrt(1 - 2*dd*w*ts), 'C2-', label='forward-backward Euler (the block)')
ax[0].axhline(1, color='gray', lw=0.8)
ax[0].axvline(D_MODEL, color='gray', lw=0.8, ls='-.')
ax[0].axvline(D_BLOCK, color='gray', lw=0.8, ls='-.')
ax[0].axvline(d_lim, color='C3', lw=0.8, ls='-.')
ax[0].annotate(r'$0.015$ (model)', (D_MODEL, 1.00078), xytext=(0.0155, 1.00078), fontsize=6)
ax[0].annotate(r'$\delta = 0.005$ (block)', (D_BLOCK, 1.00078), xytext=(0.0055, 1.00078), fontsize=6)
ax[0].annotate(r'$\delta = \omega t_s/2$: Euler poles on the circle', (d_lim, 1.0004), xytext=(0.021, 1.0005), fontsize=6.5, color='C3')
ax[0].set_xlabel(r'damping $\delta$'); ax[0].set_ylabel('|z| of the resonator poles')
ax[0].set_ylim(0.998, 1.001); ax[0].legend(loc='lower left', fontsize=6)
ax[0].set_title('pole radius against the damping', fontsize=8)
# free response x2 from x2(0) = 1, e = 0
t, ye = free_response(D_MODEL, fb=False)
_, yb = free_response(D_BLOCK, fb=True)
yc = np.exp(-D_MODEL*w*t)*np.cos(w*np.sqrt(1 - D_MODEL**2)*t)
ax[1].plot(t, yc, 'k', lw=0.9, label=r'continuous, $\delta = 0.015$')
ax[1].plot(t, ye, 'C3--', lw=0.9, label=r'forward Euler, $\delta = 0.015$ (the model)')
ax[1].plot(t, yb, 'C2-', lw=0.9, label=r'forward-backward Euler, $\delta = 0.005$ (the block)')
ax[1].plot(t, np.exp(-D_MODEL*w*t), 'k', lw=0.6); ax[1].plot(t, -np.exp(-D_MODEL*w*t), 'k', lw=0.6)
ax[1].plot(t, rad_e**(t/ts), 'C3', lw=0.6); ax[1].plot(t, -rad_e**(t/ts), 'C3', lw=0.6)
ax[1].plot(t, abs(pb[0])**(t/ts), 'C2', lw=0.6); ax[1].plot(t, -abs(pb[0])**(t/ts), 'C2', lw=0.6)
ax[1].set_xlabel('t [s]'); ax[1].set_ylabel('$x_2$ (free response, $x_2(0)=1$)')
ax[1].set_title('free response of the resonator', fontsize=8)
ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=1, fontsize=6)
fig.savefig(os.path.join(FIG, 'res_pi_poles.pdf'))

# ------------------------------------------------------------------ figure 3: closed loop with the bench plant
fig, ax = plt.subplots(1, 2, figsize=(6.7, 2.6), constrained_layout=True)
th = np.linspace(-0.08, 0.08, 400)
ax[0].plot(np.cos(th), np.sin(th), 'gray', lw=0.8, label='unit circle')
ax[0].plot(pe.real, pe.imag, 'C3x', ms=7, mew=1.5, label='resonator poles, open loop (the model)')
ax[0].plot(pb.real, pb.imag, 'C2+', ms=8, mew=1.5, label='resonator poles, open loop (the block)')
MK = {'forward Euler, delta 0.015 (the model)': 'o', 'forward-backward Euler, delta 0.015': '^',
      'forward-backward Euler, delta 0.005 (the block)': 's'}
for lab, (num, den), col, ls in CASES:
    c = cl[('bench', lab)]
    ax[0].plot(c['poles'].real, c['poles'].imag, color=col, marker=MK[lab], ls='none', ms=5, mfc='none',
               label='closed loop, ' + LBL[lab])
    ax[1].semilogx(c['f'], 20*np.log10(abs(c['S'])), color=col, ls=ls, label=LBL[lab])
ax[0].set_xlim(0.994, 1.0015); ax[0].set_ylim(-0.05, 0.05)
ax[0].set_xlabel('Re z'); ax[0].set_ylabel('Im z'); ax[0].set_title('poles near z = 1 (zoom)', fontsize=8)
ax[0].legend(loc='center left', fontsize=5.5)
ax[1].set_xlabel('f [Hz]'); ax[1].set_ylabel('|S| = |1/(1+L)| [dB]'); ax[1].set_xlim(1, 4000)
ax[1].set_title('sensitivity of the current loop, bench plant', fontsize=8); ax[1].legend(loc='lower right', fontsize=6)
fig.savefig(os.path.join(FIG, 'res_pi_closed_loop.pdf'))
print('figures written to', FIG)
