#!/usr/bin/env python3
"""Figures and numbers of the note afe_uvw_ctrl (pwr-control/repositories-documentation).

Run from this folder:   python3 afe_uvw_ctrl_figures.py
Writes into figures/:   res_pi_bode.pdf, res_pi_poles.pdf, res_pi_closed_loop.pdf, afe_uvw_ctrl_numbers.txt
Needs numpy and matplotlib only. The numbers are those of the model afe_uvw_three_phase (control period
125 us, 50 Hz, resonant PI kp = 0.6, ki = 35 1/s, delta = 0.015) and of the test bench of the library
package afe_uvw_ctrl (RL branches x_l = 0.1 pu, r = 0.01 pu, one step of output delay).
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
kp, ki, delta = 0.6, 35.0, 0.015   # resonant PI of the model (LVRT setting of init_model.m)
# test bench plant (afe_uvw_ctrl_test_param.m) and model plant (690 V, 250 kW, 500 uH, 5 % of X at 25 Hz)
plants = {'bench': dict(x_l=0.1, r=0.01),
          'model 690 V': dict(x_l=w*500e-6/(690*np.sqrt(2)/np.sqrt(3)/(250e3/np.sqrt(3)/690/0.9*np.sqrt(2))),
                              r=157*0.05*500e-6/(690*np.sqrt(2)/np.sqrt(3)/(250e3/np.sqrt(3)/690/0.9*np.sqrt(2))))}
out_delay = 1

lines = []
def say(s=''):
    print(s)
    lines.append(s)

# ------------------------------------------------------------------ controllers
def c_cont(s, d=delta):
    """continuous resonant PI C(s) = kp + ki s/(s^2 + 2 d w s + w^2)"""
    return kp + ki*s/(s**2 + 2*d*w*s + w**2)

def polys_euler(d=delta):
    """forward Euler (as implemented): den = (z-1)(z-1+2 d w ts) + (w ts)^2, num = kp den + ki ts (z-1)"""
    a = 1 - 2*d*w*ts
    den = np.array([1.0, -(1 + a), a + (w*ts)**2])
    num = kp*den + np.array([0.0, ki*ts, -ki*ts])
    return num, den

def polys_fb(d=delta):
    """forward Euler on x1, backward (with the new x1) on x2: den = z^2 - (2 - (w ts)^2 - 2 d w ts) z + 1 - 2 d w ts"""
    den = np.array([1.0, -(2 - (w*ts)**2 - 2*d*w*ts), 1 - 2*d*w*ts])
    num = kp*den + np.array([0.0, ki*ts, -ki*ts])
    return num, den

def resp(num, den, f):
    z = np.exp(1j*2*np.pi*f*ts)
    return np.polyval(num, z)/np.polyval(den, z)

def c_tustin_pw(f, d=delta):
    z = np.exp(1j*2*np.pi*f*ts)
    k = w/np.tan(w*ts/2)
    return c_cont(k*(z - 1)/(z + 1), d)

# ------------------------------------------------------------------ numbers
say('afe_uvw_ctrl_figures.py: numbers of the note')
say(f'ts = {ts*1e6:.0f} us, f0 = {f0:.0f} Hz, w = {w:.4f} rad/s, w ts = {w*ts:.6f}')
say(f'resonant PI: kp = {kp}, ki = {ki} 1/s, delta = {delta}')
say()
say('--- continuous controller')
say(f'gain of the resonator at f0: ki/(2 delta w) = {ki/(2*delta*w):.3f}  -> |C(j w)| = {abs(c_cont(1j*w)):.3f} '
    f'({20*np.log10(abs(c_cont(1j*w))):.1f} dB)')
say(f'-3 dB width of the resonant peak: 2 delta w = {2*delta*w:.3f} rad/s = {2*delta*w/2/np.pi:.3f} Hz')
sc = -delta*w + 1j*w*np.sqrt(1 - delta**2)
say(f'poles s = {sc.real:.3f} +- j{sc.imag:.3f} rad/s, decay per grid period exp(-delta 2 pi) = {np.exp(-delta*2*np.pi):.4f}')
say()
say('--- forward Euler (as implemented: x1+ = x1 + ts x2, x2+ = x2 + ts(-w^2 x1 - 2 delta w x2 + ki e))')
num_e, den_e = polys_euler()
pe = np.roots(den_e)
rad_e = np.sqrt(1 - 2*delta*w*ts + (w*ts)**2)
say(f'poles z = {pe[0].real:.6f} +- j{abs(pe[0].imag):.6f}, |z| = {abs(pe[0]):.6f} (formula {rad_e:.6f})')
say(f'angle of the poles {np.angle(pe[0]):.6f} rad -> resonance at {np.angle(pe[0])/ts/2/np.pi:.3f} Hz')
d_lim = w*ts/2
say(f'stable (|z| < 1) only for delta > w ts/2 = {d_lim:.5f}')
sigma = np.log(abs(pe[0]))/ts
say(f'equivalent continuous damping: ln|z|/ts = +{sigma:.3f} rad/s -> delta_eff = {-sigma/w:.5f}')
n_per = 1/f0/ts
say(f'growth per grid period |z|^{n_per:.0f} = {abs(pe[0])**n_per:.4f}')
g_e = abs(resp(num_e, den_e, f0))
say(f'|C_d(e^(j w ts))| at f0 = {g_e:.3f} ({20*np.log10(g_e):.1f} dB), resonator alone {g_e - kp:.3f}')
say()
say('--- forward-backward Euler (x1+ = x1 + ts x2, x2+ = x2 + ts(-w^2 x1+ - 2 delta w x2 + ki e)), not implemented')
num_f, den_f = polys_fb()
pf = np.roots(den_f)
say(f'poles |z| = {abs(pf[0]):.6f} (formula sqrt(1 - 2 delta w ts) = {np.sqrt(1 - 2*delta*w*ts):.6f}), '
    f'resonance at {np.angle(pf[0])/ts/2/np.pi:.3f} Hz, |C_d| at f0 = {abs(resp(num_f, den_f, f0)):.3f}')
say(f'with delta = 0: |z| = {abs(np.roots(polys_fb(0)[1])[0]):.6f} (on the unit circle)')
say()

# closed loop with an RL plant: P(z) = K/(z - e_p), K = (1 - e_p)/r, loop L = C z^-out_delay P
def closed_loop(num, den, x_l, r):
    a = r*w/x_l
    e_p = np.exp(-a*ts)
    K = (1 - e_p)/r
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
    for lab, (num, den) in {'forward Euler': (num_e, den_e), 'forward-backward': (num_f, den_f)}.items():
        c = closed_loop(num, den, **pl)
        cl[(name, lab)] = c
        s50 = abs(np.interp(f0, c['f'], abs(c['S'])))
        say(f'  {lab:17s}: max |z| = {max(abs(c["poles"])):.6f}, crossover {c["fc"]:.0f} Hz, phase margin {c["pm"]:.1f} deg, '
            f'|S| at f0 = {s50:.4f} ({20*np.log10(s50):.1f} dB)')
        say(f'  {"":17s}  poles: ' + ', '.join(f'{p.real:.5f}{p.imag:+.5f}j (|z| {abs(p):.5f})' for p in c['poles']))
say()
with open(os.path.join(FIG, 'afe_uvw_ctrl_numbers.txt'), 'w') as fh:
    fh.write('\n'.join(lines) + '\n')


# ------------------------------------------------------------------ more numbers
def obs_gains(poles, ts):
    z = np.exp(np.array(poles)*ts)
    l1 = 2 - (z[0] + z[1])
    l2 = (z[0]*z[1] - 1 + l1)/ts
    return l1/ts, l2/ts
lines2 = []
def say2(s=''):
    print(s); lines2.append(s)
say2('--- double-integrator observers (discrete gains, poles at exp(p ts); continuous equivalents kx = -(p1+p2), kv = p1 p2)')
for name, poles in {'FRT voltage rate [-0.1 -0.4] w': [-0.1*w, -0.4*w], 'islanding ROCOF [-20 -80] rad/s': [-20.0, -80.0]}.items():
    kx, kv = obs_gains(poles, ts)
    say2(f'  {name}: kx = {kx:.3f} 1/s, kv = {kv:.2f} 1/s^2 (continuous {-(poles[0]+poles[1]):.3f}, {poles[0]*poles[1]:.2f})')
say2('--- closed loop (bench plant) when the vector limiter divides the output by scale (loop gain / scale), forward Euler')
for scale in [1, 2, 5, 10]:
    pl = dict(plants['bench']); 
    a = pl['r']*w/pl['x_l']; e_p = np.exp(-a*ts); K = (1 - e_p)/pl['r']/scale
    den_ol = np.polymul(den_e, np.polymul(np.array([1.0, 0.0]), np.array([1.0, -e_p])))
    poles = np.roots(np.polyadd(den_ol, K*num_e))
    say2(f'  scale {scale:2d}: max |z| = {max(abs(poles)):.6f}')
say2('--- forward-backward Euler: damping that gives the same gain at f0 as the implemented resonator')
g_res = g_e - kp
say2(f'  ki/(2 delta_eq w) = {g_res:.3f} -> delta_eq = {ki/(2*g_res*w):.5f}')
with open(os.path.join(FIG, 'afe_uvw_ctrl_numbers.txt'), 'a') as fh:
    fh.write('\n'.join(lines2) + '\n')

# ------------------------------------------------------------------ figure 1: Bode of the resonant PI
f = np.logspace(0, np.log10(2000), 4000)
Cc = c_cont(1j*2*np.pi*f)
Ce = resp(num_e, den_e, f)
Cf = resp(num_f, den_f, f)
fz = np.linspace(40, 60, 4001)
fig, ax = plt.subplots(1, 3, figsize=(6.7, 2.4), constrained_layout=True)
ax[0].semilogx(f, 20*np.log10(abs(Cc)), 'k', label='continuous $C(s)$')
ax[0].semilogx(f, 20*np.log10(abs(Ce)), 'C3--', label='forward Euler (implemented)')
ax[0].semilogx(f, 20*np.log10(abs(Cf)), 'C0:', label='forward-backward Euler')
ax[0].set_xlabel('f [Hz]'); ax[0].set_ylabel('|C| [dB]'); ax[0].set_title('magnitude', fontsize=8)
ax[0].legend(loc='upper right')
ax[1].semilogx(f, np.degrees(np.angle(Cc)), 'k')
ax[1].semilogx(f, np.degrees(np.angle(Ce)), 'C3--')
ax[1].semilogx(f, np.degrees(np.angle(Cf)), 'C0:')
ax[1].set_xlabel('f [Hz]'); ax[1].set_ylabel('phase [deg]'); ax[1].set_title('phase', fontsize=8)
ax[1].set_yticks([-180, -90, 0, 90, 180])
ax[2].plot(fz, abs(c_cont(1j*2*np.pi*fz)), 'k')
ax[2].plot(fz, abs(resp(num_e, den_e, fz)), 'C3--')
ax[2].plot(fz, abs(resp(num_f, den_f, fz)), 'C0:')
ax[2].set_xlabel('f [Hz]'); ax[2].set_ylabel('|C| [pu/pu]'); ax[2].set_title('magnitude around $f_0$', fontsize=8)
fig.savefig(os.path.join(FIG, 'res_pi_bode.pdf'))

# ------------------------------------------------------------------ figure 2: pole radius and free response
dd = np.linspace(0, 0.05, 501)
fig, ax = plt.subplots(1, 2, figsize=(6.7, 2.4), constrained_layout=True)
ax[0].plot(dd, np.exp(-dd*w*ts), 'k', label=r'exact $e^{-\delta\omega t_s}$')
ax[0].plot(dd, np.sqrt(1 - 2*dd*w*ts + (w*ts)**2), 'C3--', label='forward Euler (implemented)')
ax[0].plot(dd, np.sqrt(1 - 2*dd*w*ts), 'C0:', label='forward-backward Euler')
ax[0].axhline(1, color='gray', lw=0.8)
ax[0].axvline(delta, color='gray', lw=0.8, ls='-.')
ax[0].axvline(d_lim, color='C3', lw=0.8, ls='-.')
ax[0].annotate(r'$\delta = 0.015$ (model)', (delta, 0.99965), xytext=(0.001, 0.99963), fontsize=7)
ax[0].annotate(r'$\delta = \omega t_s/2$: Euler poles on the circle', (d_lim, 1.0004), xytext=(0.021, 1.0005), fontsize=7, color='C3')
ax[0].set_xlabel(r'damping $\delta$'); ax[0].set_ylabel('|z| of the resonator poles')
ax[0].set_ylim(0.998, 1.001); ax[0].legend(loc='lower left')
ax[0].set_title('pole radius against the damping', fontsize=8)
# free response x2 from x2(0) = 1, e = 0
n = int(0.3/ts)
t = np.arange(n)*ts
x = np.array([0.0, 1.0]); ye = np.zeros(n)
for k in range(n):
    ye[k] = x[1]
    x = np.array([x[0] + ts*x[1], x[1] + ts*(-w**2*x[0] - 2*delta*w*x[1])])
yc = np.exp(-delta*w*t)*np.cos(w*np.sqrt(1 - delta**2)*t)
ax[1].plot(t, yc, 'k', label='continuous')
ax[1].plot(t, ye, 'C3--', label='forward Euler (implemented)')
ax[1].plot(t, np.exp(-delta*w*t), 'k', lw=0.6); ax[1].plot(t, -np.exp(-delta*w*t), 'k', lw=0.6)
ax[1].plot(t, rad_e**(t/ts), 'C3', lw=0.6); ax[1].plot(t, -rad_e**(t/ts), 'C3', lw=0.6)
ax[1].set_xlabel('t [s]'); ax[1].set_ylabel('$x_2$ (free response, $x_2(0)=1$)')
ax[1].set_title(r'free response of the resonator, $\delta = 0.015$', fontsize=8); ax[1].legend(loc='lower center', ncol=2, fontsize=6.5)
fig.savefig(os.path.join(FIG, 'res_pi_poles.pdf'))

# ------------------------------------------------------------------ figure 3: closed loop with the bench plant
cb_e = cl[('bench', 'forward Euler')]
cb_f = cl[('bench', 'forward-backward')]
fig, ax = plt.subplots(1, 2, figsize=(6.7, 2.5), constrained_layout=True)
th = np.linspace(-0.08, 0.08, 400)
ax[0].plot(np.cos(th), np.sin(th), 'gray', lw=0.8, label='unit circle')
ax[0].plot(pe.real, pe.imag, 'C3x', ms=7, mew=1.5, label='resonator poles, open loop (forward Euler)')
ax[0].plot(cb_e['poles'].real, cb_e['poles'].imag, 'C3o', ms=5, mfc='none', label='closed loop, forward Euler')
ax[0].plot(cb_f['poles'].real, cb_f['poles'].imag, 'C0s', ms=5, mfc='none', label='closed loop, forward-backward')
ax[0].set_xlim(0.994, 1.0015); ax[0].set_ylim(-0.05, 0.05)
ax[0].set_xlabel('Re z'); ax[0].set_ylabel('Im z'); ax[0].set_title('poles near z = 1 (zoom)', fontsize=8)
ax[0].legend(loc='center left', fontsize=6)
ax[1].semilogx(cb_e['f'], 20*np.log10(abs(cb_e['S'])), 'C3--', label='forward Euler (implemented)')
ax[1].semilogx(cb_f['f'], 20*np.log10(abs(cb_f['S'])), 'C0:', label='forward-backward Euler')
ax[1].set_xlabel('f [Hz]'); ax[1].set_ylabel('|S| = |1/(1+L)| [dB]'); ax[1].set_xlim(1, 4000)
ax[1].set_title('sensitivity of the current loop, bench plant', fontsize=8); ax[1].legend(loc='lower right')
fig.savefig(os.path.join(FIG, 'res_pi_closed_loop.pdf'))
print('figures written to', FIG)
