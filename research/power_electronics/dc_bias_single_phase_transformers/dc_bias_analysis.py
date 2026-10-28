"""Componente continua nei trasformatori monofase alimentati da inverter V/f in anello aperto.

Script di accompagnamento della nota dc_bias_single_phase_transformers.tex (research/power_electronics):
riproduce ogni numero citato nella nota e disegna le sue figure (python3, numpy, scipy, matplotlib).

    python3 dc_bias_analysis.py [cartella_figure]

Contenuto:
  1. dati assunti (trasformatore 1.7 MVA monofase, inverter 2 livelli 1070 V / 1.25 kHz) e grandezze derivate
     (I_n, I_0, L_m, flusso concatenato, soglia di saturazione, L_sat, resistenza dell'anello in continua R)
  2. bilancio delle sorgenti di errore in continua dell'inverter e effetto del tempo morto
  3. caratteristica magnetizzante mediata di un nucleo senza traferro con offset di flusso: corrente media I_dc,
     picco dell'impulso magnetizzante, coefficiente di seconda armonica, induttanza incrementale L_inc
  4. impianto in continua visto dall'anello alpha-beta: R + s L_inc, accoppiamento per squilibrio delle resistenze
  5. progetto dell'anello: PI sulla componente continua alpha-beta con media mobile su un periodo; Bode e margini
     di fase con nucleo lineare e con nucleo saturo; tabella di progetto
  6. simulazione nel tempo (modello medio dell'inverter, tre trasformatori monofase a stella con neutro isolato, a
     vuoto): senza compensazione, anello inserito su nucleo saturo, anello attivo dall'inizio, offset dei sensori,
     regolazione della seconda armonica
  7. indicatore di seconda armonica (alternativa insensibile all'offset dei sensori)
Durata: circa 2 minuti (simulazioni).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 9, 'axes.grid': True,
                     'grid.alpha': 0.35, 'legend.fontsize': 7.5, 'axes.titlesize': 9, 'figure.dpi': 100})
FIGW = 6.3     # [in] larghezza del testo della nota
MU0 = 4e-7*np.pi


def pr(*a):
    print(*a)


# =============================================================================================== 1. dati
pr('=' * 100)
pr('1. DATI ASSUNTI E GRANDEZZE DERIVATE')
S_n = 1.7e6            # [VA] potenza del singolo trasformatore monofase
U_1 = 690.0            # [V] tensione nominale (efficace) dell'avvolgimento primario
f_1 = 50.0             # [Hz] frequenza di uscita
w_1 = 2*np.pi*f_1
T_1 = 1/f_1
i0_pct = 0.3           # [%] corrente a vuoto in percento di I_n
pcu_pct = 1.0          # [%] perdite nel rame a carico nominale (entrambi gli avvolgimenti)
B_max, B_sat = 1.7, 2.0    # [T] induzione di picco di progetto e di saturazione (lamierino a grani orientati)
mu_r, l_m = 30e3, 3.5      # permeabilita' relativa iniziale e lunghezza del circuito magnetico [m]
L_sat_pu = 0.2         # [pu] induttanza satura (nucleo d'aria) dell'avvolgimento, base del trasformatore
w_knee_rel = 0.03      # arrotondamento del ginocchio (in frazione del flusso di picco)
U_dc = 1070.0          # [V] tensione del DC link
f_sw = 1250.0          # [Hz] frequenza di commutazione
T_s = 1/f_sw           # [s] periodo di commutazione = periodo di controllo (un aggiornamento per periodo PWM)
t_d = 5e-6             # [s] tempo morto
R_cab, R_sw = 1.0e-3, 0.5e-3   # [ohm] cavi/sbarre e resistenza differenziale IGBT+diodo

I_n = S_n/U_1
I_0 = i0_pct/100*I_n
I_0pk = np.sqrt(2)*I_0
U_pk = np.sqrt(2)*U_1
lam_hat = U_pk/w_1                        # [Wb] ampiezza del flusso concatenato
lam_s = lam_hat*B_sat/B_max               # [Wb] flusso concatenato di saturazione
lam_knee = lam_s - lam_hat                # [Wb] offset di flusso che porta il picco in saturazione
L_m = U_pk/(w_1*I_0pk)                    # [H] induttanza magnetizzante (zona lineare)
Z_b = U_1**2/S_n
L_b = Z_b/w_1
L_sat = L_sat_pu*L_b
w_knee = w_knee_rel*lam_hat
R_tr = pcu_pct/100*U_1/I_n                # [ohm] resistenza totale del trasformatore (entrambi gli avvolgimenti)
R_1 = R_tr/2                              # [ohm] avvolgimento primario
R = R_1 + R_cab + R_sw                    # [ohm] resistenza dell'anello in continua
tau_m = L_m/R
NI_knee = (B_sat - B_max)/(MU0*mu_r)*l_m  # [A spire] amperspire continue che esauriscono il margine (modello lineare)
U_ll = 690.0                              # [V] tensione concatenata di uscita dell'inverter
m_idx = np.sqrt(2)*U_ll/U_dc              # indice di modulazione (SVM: limite lineare = U_dc concatenato di picco)
N_1 = int(round(T_1/T_s))                 # campioni per periodo fondamentale
pr(f'  S_n = {S_n/1e6:.1f} MVA, U_1 = {U_1:.0f} V, f_1 = {f_1:.0f} Hz: I_n = {I_n:.0f} A, I_0 = {I_0:.2f} A ({i0_pct} %), '
   f'picco {I_0pk:.1f} A')
pr(f'  U_dc = {U_dc:.0f} V, f_sw = {f_sw:.0f} Hz, T_s = {T_s*1e6:.0f} us, rapporto di impulsi {N_1}, m = {m_idx:.2f} per '
   f'{U_ll:.0f} V concatenati')
pr(f'  flusso concatenato: lam_hat = {lam_hat:.3f} Wb, lam_s = {lam_s:.3f} Wb (B_sat/B_max = {B_sat/B_max:.3f}), '
   f'margine lam_s - lam_hat = {lam_knee:.3f} Wb ({lam_knee/lam_hat*100:.0f} % di lam_hat)')
pr(f'  L_m = {L_m:.3f} H, Z_b = {Z_b:.3f} ohm, L_b = {L_b*1e3:.3f} mH, L_sat = {L_sat_pu} pu = {L_sat*1e3:.3f} mH, '
   f'ginocchio w = {w_knee:.3f} Wb')
pr(f'  R_tr = {R_tr*1e3:.2f} mohm (entrambi gli avvolgimenti), R_1 = {R_1*1e3:.2f} mohm, cavi {R_cab*1e3:.1f} mohm, '
   f'IGBT+diodo {R_sw*1e3:.1f} mohm: R = {R*1e3:.2f} mohm, tau_m = L_m/R = {tau_m:.0f} s')
pr(f'  10 mV di continua fra due rami -> {10e-3/R:.1f} A; corrente continua che porta il picco al ginocchio '
   f'(modello lineare): lam_knee/L_m = {lam_knee/L_m:.2f} A = {lam_knee/L_m/I_0:.2f} I_0')
pr(f'  amperspire continue che esauriscono il margine (mu_r = {mu_r:.0f}, l_m = {l_m} m): N I_dc = {NI_knee:.0f} A spire')

# riscalatura se l'avvolgimento e' da 398 V (primari a stella su 690 V concatenati) con la stessa potenza
for U1b in (U_1, U_ll/np.sqrt(3)):
    Inb = S_n/U1b
    I0b = i0_pct/100*Inb
    Lmb = U1b/(w_1*I0b)
    Rb = pcu_pct/100*U1b/Inb/2 + R_cab + R_sw
    pr(f'  avvolgimento da {U1b:.0f} V, {S_n/1e6:.1f} MVA: I_n = {Inb:.0f} A, I_0 = {I0b:.1f} A, L_m = {Lmb:.3f} H, '
       f'R = {Rb*1e3:.2f} mohm, tau_m = {Lmb/Rb:.0f} s, 10 mV -> {10e-3/Rb:.1f} A, L_sat = {L_sat_pu*U1b**2/S_n/w_1*1e3:.3f} mH')

# ===================================================================================== 2. sorgenti di errore
pr('=' * 100)
pr('2. SORGENTI DI ERRORE IN CONTINUA DELL\'INVERTER E TEMPO MORTO')
dV_per_us = U_dc/T_s*1e-6
pr(f'  asimmetria temporale: dV/dt_asym = U_dc/T_s = {dV_per_us:.3f} V/us; 50 ns -> {dV_per_us*0.05*1e3:.0f} mV -> '
   f'{dV_per_us*0.05/R:.0f} A; 200 ns -> {dV_per_us*0.2:.2f} V -> {dV_per_us*0.2/R:.0f} A')
for dv in (50e-3, 100e-3, 150e-3):
    pr(f'  errore differenziale {dv*1e3:.0f} mV -> I_dc = {dv/R:.0f} A = {dv/R/I_0:.1f} I_0')
V_dt = t_d/T_s*U_dc
I_pk_load = np.sqrt(2)*I_n
R_eq_dt = 2*V_dt/(np.pi*I_pk_load)
pr(f'  tempo morto t_d = {t_d*1e6:.0f} us: V_dt = {V_dt:.2f} V; R_eq = 2 V_dt/(pi I_pk) = {R_eq_dt*1e3:.2f} mohm a '
   f'I_pk = {I_pk_load:.0f} A; a meta\' carico {2*R_eq_dt*1e3:.2f} mohm')
pr(f'  a vuoto (I_pk = {I_0pk:.1f} A) il tempo morto agisce come rele\': con E = 67 mV la continua residua sarebbe '
   f'circa I_0pk*pi/2*E/V_dt = {I_0pk*np.pi/2*0.067/V_dt:.2f} A (commutazione ancora dura a questa corrente)')


# ============================================================================= 3. caratteristica magnetizzante
def i_mag(lam):
    """Corrente magnetizzante [A] di un trasformatore dal flusso concatenato lam [Wb]: due pendenze (L_m, L_sat)
    con ginocchio arrotondato (softplus di larghezza w_knee)."""
    exc = w_knee*np.logaddexp(0.0, (np.abs(lam) - lam_s)/w_knee)
    return lam/L_m + np.sign(lam)*exc/L_sat


def L_inc_inst(lam):
    """Induttanza incrementale istantanea d lam / d i [H]."""
    sig = 1/(1 + np.exp(-(np.abs(lam) - lam_s)/w_knee))
    return 1/(1/L_m + sig/L_sat)


pr('=' * 100)
pr('3. CARATTERISTICA MAGNETIZZANTE MEDIATA CON OFFSET DI FLUSSO')
theta = np.linspace(0, 2*np.pi, 20001, endpoint=False)
lam_dc_grid = np.linspace(0, 0.30*lam_hat, 1201)
I_mean = np.zeros_like(lam_dc_grid)
I_peak = np.zeros_like(lam_dc_grid)
a2 = np.zeros_like(lam_dc_grid)      # coefficiente coseno della seconda armonica (rispetto all'angolo della tensione)
b2 = np.zeros_like(lam_dc_grid)
for k, ldc in enumerate(lam_dc_grid):
    i = i_mag(ldc + lam_hat*np.sin(theta))
    I_mean[k] = i.mean()
    I_peak[k] = i.max()
    a2[k] = 2*np.mean(i*np.cos(2*theta))
    b2[k] = 2*np.mean(i*np.sin(2*theta))
L_inc_avg = np.gradient(lam_dc_grid, I_mean)      # induttanza incrementale mediata d lam_dc / d I_dc


def lam_of_Idc(I):
    return np.interp(I, I_mean, lam_dc_grid)


def interp_at_I(y, I):
    return np.interp(I, I_mean, y)


pr(f'  {"I_dc/I_0":>9} {"I_dc [A]":>9} {"lam_dc [Wb]":>12} {"lam_dc/lam_hat":>15} {"I_pk [A]":>9} {"I_pk/I_0pk":>11} '
   f'{"a_2 [A]":>9} {"a_2/I_dc":>9} {"L_inc [mH]":>11}')
for r in (0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10):
    I = r*I_0
    pr(f'  {r:9.2f} {I:9.2f} {lam_of_Idc(I):12.3f} {lam_of_Idc(I)/lam_hat:15.3f} {interp_at_I(I_peak, I):9.1f} '
       f'{interp_at_I(I_peak, I)/I_0pk:11.1f} {interp_at_I(a2, I):9.2f} {interp_at_I(a2, I)/I:9.2f} '
       f'{interp_at_I(L_inc_avg, I)*1e3:11.2f}')
pr(f'  massimo |b_2| sulla griglia: {np.abs(b2).max():.2e} A (la seconda armonica e\' in quadratura con la tensione, '
   f'cioe\' un termine -a_2 cos 2 theta con a_2 > 0 per offset positivo)')
L_inc_1 = interp_at_I(L_inc_avg, I_0)
L_inc_3 = interp_at_I(L_inc_avg, 3*I_0)
L_inc_10 = interp_at_I(L_inc_avg, 10*I_0)
pr(f'  L_inc(I_0) = {L_inc_1*1e3:.2f} mH, L_inc(3 I_0) = {L_inc_3*1e3:.2f} mH, L_inc(10 I_0) = {L_inc_10*1e3:.2f} mH, '
   f'L_m = {L_m*1e3:.0f} mH: rapporto L_m/L_inc(3 I_0) = {L_m/L_inc_3:.0f}')
# zona morta dell'indicatore di seconda armonica: offset a cui |a_2| supera 0.1 I_0
k_dz = np.argmax(np.abs(a2) > 0.1*I_0)
pr(f'  indicatore di seconda armonica: |a_2| > 0.1 I_0 per I_dc > {I_mean[k_dz]:.2f} A = {I_mean[k_dz]/I_0:.2f} I_0 '
   f'(zona morta); pendenza a_2/I_dc in saturazione piena circa {interp_at_I(a2, 3*I_0)/(3*I_0 - lam_of_Idc(3*I_0)/L_m):.2f}')
# tempo per raggiungere il ginocchio con un errore E, modello lineare dlam/dt = E - R lam/L_m
E_ref = 67e-3
t_knee = -tau_m*np.log(1 - lam_knee/(E_ref*tau_m))
pr(f'  con E = {E_ref*1e3:.0f} mV il flusso raggiunge il ginocchio in t = {t_knee:.1f} s (regime finale {E_ref/R:.1f} A '
   f'= {E_ref/R/I_0:.1f} I_0, raggiunto in pochi secondi oltre il ginocchio)')

# ---- figura 1: caratteristica e forme d'onda
fig, ax = plt.subplots(2, 2, figsize=(FIGW, 5.0))
lam_plot = np.linspace(-1.3*lam_s, 1.3*lam_s, 2001)
ax[0, 0].plot(lam_plot/lam_hat, i_mag(lam_plot)/I_0pk, 'k')
ax[0, 0].axvline(1, color='C3', ls='--', lw=0.8)
ax[0, 0].axvline(-1, color='C3', ls='--', lw=0.8)
ax[0, 0].axvline(lam_s/lam_hat, color='C0', ls=':', lw=0.8)
ax[0, 0].axvline(-lam_s/lam_hat, color='C0', ls=':', lw=0.8)
ax[0, 0].set_xlabel(r'$\lambda/\hat\lambda$')
ax[0, 0].set_ylabel(r'$i_m/\hat I_0$')
ax[0, 0].set_ylim(-40, 40)
ax[0, 0].set_title(r'(a) $i_m(\lambda)$; $\pm\hat\lambda$ (tratteggio), $\pm\lambda_s$ (punti)')
th = np.linspace(0, 2*np.pi, 2001)
for r, c in ((0, 'k'), (0.25, 'C2'), (1, 'C0'), (3, 'C3')):
    ldc = lam_of_Idc(r*I_0)
    ax[0, 1].plot(th/np.pi*180, i_mag(ldc + lam_hat*np.sin(th))/I_0pk, c, lw=1.2,
                  label=rf'$I_{{dc}} = {r:g}\,I_0$')
ax[0, 1].set_xlabel(r'$\theta$ [$^\circ$]')
ax[0, 1].set_ylabel(r'$i_m/\hat I_0$')
ax[0, 1].set_xlim(0, 360)
ax[0, 1].legend(loc='upper right', handlelength=2.5)
ax[0, 1].set_title(r'(b) corrente magnetizzante, $\lambda = \lambda_{dc} + \hat\lambda\sin\theta$')
ax[1, 0].plot(lam_dc_grid/lam_hat, I_mean/I_0, 'C0', label=r'$I_{dc}/I_0$')
ax[1, 0].plot(lam_dc_grid/lam_hat, -a2/I_0, 'C2', label=r'$-a_2/I_0$ (2$^a$ arm.)')
ax[1, 0].axvline(lam_knee/lam_hat, color='k', ls='--', lw=0.8)
ax[1, 0].set_xlabel(r'$\lambda_{dc}/\hat\lambda$')
ax[1, 0].set_xlim(0, 0.2)
ax[1, 0].set_ylim(0, 12)
ax[1, 0].set_ylabel(r'$I_{dc}/I_0$, $-a_2/I_0$')
axr = ax[1, 0].twinx()
axr.plot(lam_dc_grid/lam_hat, I_peak/I_0pk, 'C3', label=r'$I_{pk}/\hat I_0$ (destra)')
axr.set_ylim(0, 60)
axr.set_ylabel(r'$I_{pk}/\hat I_0$')
axr.grid(False)
h1, l1 = ax[1, 0].get_legend_handles_labels()
h2, l2 = axr.get_legend_handles_labels()
ax[1, 0].legend(h1 + h2, l1 + l2, loc='upper left')
ax[1, 0].set_title(r'(c) media, picco e 2$^a$ armonica vs offset')
ax[1, 1].semilogy(I_mean/I_0, L_inc_avg*1e3, 'k')
ax[1, 1].axhline(L_m*1e3, color='C0', ls=':', lw=0.8)
ax[1, 1].axhline(L_sat*1e3, color='C3', ls=':', lw=0.8)
ax[1, 1].set_xlabel(r'$I_{dc}/I_0$')
ax[1, 1].set_ylabel(r'$L_{inc}$ [mH]')
ax[1, 1].set_xlim(0, 10)
ax[1, 1].set_title(r'(d) $L_{inc} = d\lambda_{dc}/dI_{dc}$; $L_m$, $L_{sat}$ (punti)')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_magnetizing_characteristic.pdf'))
plt.close(fig)

# ================================================================= 4. impianto alpha-beta e squilibrio delle R
pr('=' * 100)
pr('4. IMPIANTO IN CONTINUA NEL RIFERIMENTO ALPHA-BETA')
T_cl = (2/3)*np.array([[1, -0.5, -0.5], [0, np.sqrt(3)/2, -np.sqrt(3)/2]])        # Clarke (invariante in ampiezza)
T_inv = np.array([[1, 0], [-0.5, np.sqrt(3)/2], [-0.5, -np.sqrt(3)/2]])            # inversa (senza omopolare)


def G_abc(Rk):
    """Matrice di conduttanza 3x3 i_abc = G v_abc di tre resistenze a stella con neutro isolato."""
    g = 1/np.asarray(Rk, float)
    return np.diag(g) - np.outer(g, g)/g.sum()


for unb in (0.0, 0.1, 0.2):
    Rk = R*np.array([1 + unb, 1 - unb, 1.0])
    G_ab = T_cl @ G_abc(Rk) @ T_inv
    ev = np.linalg.eigvals(G_ab)
    pr(f'  squilibrio {unb*100:.0f} % (R_a, R_b, R_c) = ({Rk[0]*1e3:.2f}, {Rk[1]*1e3:.2f}, {Rk[2]*1e3:.2f}) mohm: '
       f'G_ab*R = [[{G_ab[0,0]*R:.3f}, {G_ab[0,1]*R:.3f}], [{G_ab[1,0]*R:.3f}, {G_ab[1,1]*R:.3f}]], '
       f'autovalori*R = {np.sort(ev.real)*R}')
E_abc = np.array([66.9e-3, -40.1e-3, 53.5e-3])      # esempio della nota: dt = (+50, -30, +40) ns
E_ab = T_cl @ E_abc
E_cm = E_abc.mean()
I_unc = G_abc([R, R, R]) @ E_abc
pr(f'  esempio: dt = (+50, -30, +40) ns -> E_abc = {E_abc*1e3} mV, modo comune {E_cm*1e3:.1f} mV, '
   f'E_ab = ({E_ab[0]*1e3:.1f}, {E_ab[1]*1e3:.1f}) mV, |E_ab| = {np.hypot(*E_ab)*1e3:.1f} mV')
pr(f'  correnti continue senza compensazione: I_abc = {np.round(I_unc, 1)} A = {np.round(I_unc/I_0, 2)} I_0; '
   f'I_ab = E_ab/R = ({E_ab[0]/R:.1f}, {E_ab[1]/R:.1f}) A')

# ============================================================================== 5. progetto dell'anello PI
pr('=' * 100)
pr('5. PROGETTO DELL\'ANELLO PI SULLA CONTINUA ALPHA-BETA')
t_delay = 1.5*T_s                     # calcolo + ritenuta della PWM


def H_ma(w):
    """Media mobile discreta su N_1 campioni a passo T_s, risposta in frequenza."""
    z = np.exp(1j*w*T_s)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = (1 - z**(-N_1))/(N_1*(1 - z**(-1)))
    return np.where(np.abs(w) < 1e-12, 1.0, h)


def loop(w, kp, ki, L):
    C = kp + ki/(1j*w)
    P = 1/(R + 1j*w*L)
    return C*H_ma(w)*np.exp(-1j*w*t_delay)*P


def margins(kp, ki, L):
    w = np.logspace(-3, 3.5, 20000)
    Lw = loop(w, kp, ki, L)
    mag = np.abs(Lw)
    idx = np.where((mag[:-1] >= 1) & (mag[1:] < 1))[0]
    if len(idx) == 0:
        return np.nan, np.nan
    k = idx[-1]
    wc = w[k]
    ph = np.unwrap(np.angle(Lw))
    pm = 180 + np.degrees(ph[k])
    pm = (pm + 180) % 360 - 180
    return wc, pm


zeta_lin = 0.8
pr(f'  misura: media mobile su N_1 = {N_1} campioni (un periodo, {T_1*1e3:.0f} ms, ritardo {T_1/2*1e3:.0f} ms) + '
   f'calcolo e PWM {t_delay*1e3:.2f} ms')
pr(f'  regola: k_i = (R + k_p)^2/(4 zeta^2 L_m) con zeta = {zeta_lin} nella zona lineare')
pr(f'  {"k_p [mohm]":>11} {"k_p/R":>6} {"k_i [V/(A s)]":>14} {"w_n [rad/s]":>12} {"t_set lin [s]":>14} '
   f'{"I_pk/E [A/V]":>13} {"w_c lin":>8} {"PM lin":>7} {"w_c(I_0)":>9} {"PM(I_0)":>8} {"w_c(3I_0)":>10} {"PM(3I_0)":>9} '
   f'{"w_c(10I_0)":>11} {"PM(10I_0)":>10}')
design = {}
for kp in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2):
    if kp == 0.0:
        ki = R**2/(4*zeta_lin**2*L_m)
    else:
        ki = (R + kp)**2/(4*zeta_lin**2*L_m)
    wn = np.sqrt(ki/L_m)
    t_set = 4/(zeta_lin*wn)
    ipk_per_E = 1/(R + kp)      # limite superiore del picco di corrente per gradino di errore E (zona lineare)
    row = [kp*1e3, kp/R, ki, wn, t_set, ipk_per_E]
    for L in (L_m, L_inc_1, L_inc_3, L_inc_10):
        wc, pm = margins(kp, ki, L)
        row += [wc, pm]
    design[kp] = (ki, row)
    pr(f'  {row[0]:11.0f} {row[1]:6.1f} {row[2]:14.2e} {row[3]:12.4f} {row[4]:14.0f} {row[5]:13.1f} '
       f'{row[6]:8.3f} {row[7]:7.0f} {row[8]:9.2f} {row[9]:8.0f} {row[10]:10.1f} {row[11]:9.0f} {row[12]:11.1f} {row[13]:10.0f}')
k_p = 0.05
k_i = design[k_p][0]
V_lim = 2.0
pr(f'  scelta: k_p = {k_p*1e3:.0f} mohm = {k_p/R:.0f} R, k_i = {k_i:.2e} V/(A s), limite dell\'uscita +-{V_lim} V')
pr(f'  zona lineare: w_n = {np.sqrt(k_i/L_m):.3f} rad/s ({np.sqrt(k_i/L_m)/2/np.pi*1e3:.0f} mHz), '
   f'picco di corrente per E = 67 mV circa {2*0.067/(np.e*(R + k_p)):.2f} A (zeta = 1) .. {0.067/(R + k_p):.2f} A, '
   f'contro {0.067/R:.0f} A senza anello')
pr(f'  k_p max per PM >= 60 gradi con L_inc(3 I_0): circa {0.45/(T_1/2 + t_delay)*L_inc_3*1e3:.0f} mohm '
   f'(w_c T_d <= 0.45)')
# discreto
pr(f'  forma discreta a T_s = {T_s*1e6:.0f} us: x[k+1] = x[k] + k_i T_s e[k], k_i T_s = {k_i*T_s:.2e} V/A per passo; '
   f'k_p = {k_p} V/A')

# ---- figura 2: Bode dell'anello
w = np.logspace(-2, 3, 4000)
fig, ax = plt.subplots(2, 1, figsize=(FIGW, 4.6), sharex=True)
for L, lab, c in ((L_m, r'nucleo lineare, $L = L_m$', 'C0'), (L_inc_1, r'$L = L_{inc}(I_0)$', 'C2'),
                  (L_inc_3, r'$L = L_{inc}(3 I_0)$', 'C3'), (L_inc_10, r'$L = L_{inc}(10 I_0)$', 'C1')):
    Lw = loop(w, k_p, k_i, L)
    wc, pm = margins(k_p, k_i, L)
    fc = wc/2/np.pi
    fct = rf'{fc*1e3:.0f} mHz' if fc < 0.1 else rf'{fc:.2f} Hz'
    ax[0].semilogx(w/2/np.pi, 20*np.log10(np.abs(Lw)), c, label=lab + rf' ($f_c$ = {fct}, PM = {pm:.0f}$^\circ$)')
    ph = np.degrees(np.unwrap(np.angle(Lw)))
    ax[1].semilogx(w/2/np.pi, ph, c)
ax[0].axhline(0, color='k', lw=0.6)
ax[0].set_ylabel('|L| [dB]')
ax[0].set_ylim(-40, 60)
ax[0].legend(loc='upper right')
ax[0].set_title(rf'anello PI sulla continua $\alpha\beta$: $k_p$ = {k_p*1e3:.0f} m$\Omega$, $k_i$ = {k_i:.1e} V/(A s), '
                rf'media mobile su {T_1*1e3:.0f} ms')
ax[1].axhline(-180, color='k', lw=0.6)
ax[1].set_ylabel('fase [$^\\circ$]')
ax[1].set_xlabel('f [Hz]')
ax[1].set_ylim(-360, 0)
ax[1].set_yticks([-360, -270, -180, -90, 0])
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_loop_bode.pdf'))
plt.close(fig)

# ============================================================================== 6. simulazione nel tempo
pr('=' * 100)
pr('6. SIMULAZIONE NEL TEMPO (modello medio, tre trasformatori a stella con neutro isolato, a vuoto)')
dt = 100e-6
n_sub = int(round(T_s/dt))
dt = T_s/n_sub
t_end = 60.0
n_ctrl = int(round(t_end/T_s))
t_on = 20.0                               # inserzione dell'anello nello scenario A e D
k_p2, k_i2 = k_p, k_i                     # anello di seconda armonica (scenario D): stessi guadagni del PI
g_2 = 2.0                                 # [A/A] |a_2| per ampere di corrente continua oltre la zona morta
delta_sens = np.array([5.0, -2.0, 4.0])   # [A] offset dei sensori nello scenario C
SCEN = ['A: anello inserito a t = 15 s', 'B: anello attivo da t = 0', 'C: come B con offset dei sensori',
        'D: anello di 2a armonica inserito a t = 15 s']
n_sc = len(SCEN)
phase_shift = np.array([0.0, -2*np.pi/3, 2*np.pi/3])
lam = np.tile(lam_hat*np.sin(phase_shift), (n_sc, 1))   # flusso di regime a t = 0 (fine della rampa di avviamento)
x_ab = np.zeros((n_sc, 2))                # stato integrale del PI (V)
v_c_ab = np.zeros((n_sc, 2))              # uscita del compensatore (V)
buf_i = np.zeros((N_1, n_sc, 3))          # campioni delle correnti di fase dell'ultimo periodo
buf_c2 = np.zeros((N_1, n_sc, 3))         # campioni di i_k cos(2 theta_k)
ptr = 0
# registrazione a ogni periodo fondamentale
n_rec = int(round(t_end/T_1))
rec_t = np.zeros(n_rec)
rec_idc = np.zeros((n_rec, n_sc, 3))
rec_a2 = np.zeros((n_rec, n_sc, 3))
rec_vc = np.zeros((n_rec, n_sc, 2))
rec_lam = np.zeros((n_rec, n_sc, 3))
rec_ipk = np.zeros((n_rec, n_sc, 3))
# forme d'onda istantanee in due finestre di due periodi
win = [(t_on - 2*T_1, t_on), (t_end - 2*T_1, t_end)]
wave_t = [[] for _ in win]
wave_i = [[] for _ in win]


def deriv(lam_, v_leg):
    """dlam/dt per n_sc x 3 trasformatori a stella con neutro isolato: v_k - v_n = R i_k + dlam_k/dt, sum i_k = 0."""
    i = i_mag(lam_)
    Li = L_inc_inst(lam_)
    g = 1/Li
    v_n = ((v_leg - R*i)*g).sum(axis=1, keepdims=True)/g.sum(axis=1, keepdims=True)
    return v_leg - v_n - R*i


t = 0.0
rec_k = 0
rec_acc_i = np.zeros((n_sc, 3))
rec_acc_c2 = np.zeros((n_sc, 3))
rec_acc_lam = np.zeros((n_sc, 3))
rec_pk = np.zeros((n_sc, 3))
rec_cnt = 0
for k in range(n_ctrl):
    # ---- controllo (inizio del periodo k): campiona, media mobile, PI, nuova uscita valida per il periodo
    th_k = w_1*t + phase_shift
    i_s = i_mag(lam)                                     # correnti campionate (modello medio: senza ripple)
    i_meas = i_s.copy()
    i_meas[2] += delta_sens                              # scenario C: offset dei sensori
    buf_i[ptr] = i_meas
    buf_c2[ptr] = i_meas*np.cos(2*th_k)
    ptr = (ptr + 1) % N_1
    i_dc_abc = buf_i.mean(axis=0)                        # media su un periodo
    a2_abc = 2*buf_c2.mean(axis=0)                       # coefficiente coseno della 2a armonica
    i_dc_ab = i_dc_abc @ T_cl.T
    ind2_ab = (-a2_abc/g_2) @ T_cl.T                     # indicatore equivalente in ampere (metodo B)
    en = np.array([t >= t_on, True, True, t >= t_on])
    for s in range(n_sc):
        if not en[s]:
            continue
        if s < 3:
            e = -i_dc_ab[s]
            x_ab[s] = np.clip(x_ab[s] + k_i*T_s*e, -V_lim, V_lim)
            v_c_ab[s] = np.clip(k_p*e + x_ab[s], -V_lim, V_lim)
        else:
            e = -ind2_ab[s]
            x_ab[s] = np.clip(x_ab[s] + k_i2*T_s*e, -V_lim, V_lim)
            v_c_ab[s] = np.clip(k_p2*e + x_ab[s], -V_lim, V_lim)
    dv_abc = v_c_ab @ T_inv.T                            # offset sulle tre gambe (somma nulla)
    # ---- impianto: n_sub passi RK4 entro il periodo di controllo
    for j in range(n_sub):
        def vleg(tt):
            return U_pk*np.cos(w_1*tt + phase_shift)[None, :] + E_abc[None, :] + dv_abc
        k1 = deriv(lam, vleg(t))
        k2 = deriv(lam + 0.5*dt*k1, vleg(t + 0.5*dt))
        k3 = deriv(lam + 0.5*dt*k2, vleg(t + 0.5*dt))
        k4 = deriv(lam + dt*k3, vleg(t + dt))
        lam = lam + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        # proiezione sul vincolo sum i_k = 0 (spostamento omopolare del flusso, due passi di Newton)
        for _ in range(2):
            lam = lam - (i_mag(lam).sum(axis=1)/(1/L_inc_inst(lam)).sum(axis=1))[:, None]
        t += dt
        i_now = i_mag(lam)
        rec_acc_i += i_now
        rec_acc_c2 += i_now*np.cos(2*(w_1*t + phase_shift))
        rec_acc_lam += lam
        rec_pk = np.maximum(rec_pk, np.abs(i_now))
        rec_cnt += 1
        for m, (ta, tb) in enumerate(win):
            if ta <= t < tb:
                wave_t[m].append(t)
                wave_i[m].append(i_now.copy())
    if (k + 1) % N_1 == 0 and rec_k < n_rec:
        rec_t[rec_k] = t
        rec_idc[rec_k] = rec_acc_i/rec_cnt
        rec_a2[rec_k] = 2*rec_acc_c2/rec_cnt
        rec_lam[rec_k] = rec_acc_lam/rec_cnt
        rec_ipk[rec_k] = rec_pk
        rec_vc[rec_k] = v_c_ab
        rec_acc_i[:] = 0
        rec_acc_c2[:] = 0
        rec_acc_lam[:] = 0
        rec_pk[:] = 0
        rec_cnt = 0
        rec_k += 1

rec_t = rec_t[:rec_k]
rec_idc, rec_a2, rec_vc, rec_lam, rec_ipk = rec_idc[:rec_k], rec_a2[:rec_k], rec_vc[:rec_k], rec_lam[:rec_k], rec_ipk[:rec_k]


def at(tq, arr):
    return arr[np.argmin(np.abs(rec_t - tq))]


pr(f'  passo {dt*1e6:.0f} us (RK4), controllo a T_s = {T_s*1e6:.0f} us, {t_end:.0f} s simulati; errori E_abc = {E_abc*1e3} mV')
pr(f'  scenario A, senza compensazione (t < {t_on:.0f} s): I_dc a 5 s = {np.round(at(5.0, rec_idc)[0], 2)} A, '
   f'a 10 s = {np.round(at(10.0, rec_idc)[0], 2)} A, a 15 s = {np.round(at(15.0, rec_idc)[0], 2)} A, '
   f'a {t_on:.0f} s = {np.round(at(t_on - T_1, rec_idc)[0], 2)} A (regime E_diff/R = {np.round(I_unc, 1)} A); '
   f'somma delle tre a {t_on:.0f} s = {at(t_on - T_1, rec_idc)[0].sum():.2e} A')
k_knee = np.argmax(np.abs(rec_idc[:, 0, 1]) > lam_knee/L_m)
pr(f'    la fase b supera il ginocchio ({lam_knee/L_m:.2f} A) a t = {rec_t[k_knee]:.1f} s; offset di flusso a {t_on:.0f} s = '
   f'{np.round(at(t_on - T_1, rec_lam)[0], 3)} Wb')
pr(f'    picchi delle correnti a 15 s: {np.round(at(t_on - T_1, rec_ipk)[0], 0)} A = '
   f'{np.round(at(t_on - T_1, rec_ipk)[0]/I_0pk, 1)} I_0pk; a_2 = {np.round(at(t_on - T_1, rec_a2)[0], 1)} A')
for s, tq in ((0, t_on + 1), (0, t_on + 2), (0, t_on + 5), (0, t_on + 10), (0, t_on + 20), (0, t_end - T_1)):
    pr(f'    anello inserito: t = {tq:.0f} s: I_dc = {np.round(at(tq, rec_idc)[s], 2)} A, v_c = {np.round(at(tq, rec_vc)[s]*1e3, 1)} mV, '
       f'picchi {np.round(at(tq, rec_ipk)[s], 0)} A')
sel = rec_t > t_on + 1
pr(f'    massimo |I_dc| oltre 1 s dall\'inserzione: {np.abs(rec_idc[sel, 0]).max():.2f} A; |I_dc| < 0.5 A da t = '
   f'{rec_t[np.argmax(np.all(np.abs(rec_idc[:, 0]) < 0.5, axis=1) & sel)]:.0f} s; -E_ab = ({-E_ab[0]*1e3:.1f}, {-E_ab[1]*1e3:.1f}) mV')
mB = rec_idc[:, 1]
pr(f'  scenario B (anello da t = 0): massimo |I_dc| = {np.abs(mB).max():.2f} A = {np.abs(mB).max()/I_0:.2f} I_0 a '
   f't = {rec_t[np.argmax(np.abs(mB).max(axis=1))]:.1f} s; |I_dc| < 0.5 A da t = '
   f'{rec_t[np.argmax(np.all(np.abs(mB) < 0.5, axis=1) & (rec_t > 3))]:.0f} s, < 0.2 A da t = '
   f'{rec_t[np.argmax(np.all(np.abs(mB) < 0.2, axis=1) & (rec_t > 3))]:.0f} s; '
   f'a {t_end:.0f} s I_dc = {np.round(at(t_end - T_1, rec_idc)[1], 3)} A, v_c = {np.round(at(t_end - T_1, rec_vc)[1]*1e3, 1)} mV')
pr(f'    offset massimo di flusso: {np.abs(rec_lam[:, 1]).max():.3f} Wb = {np.abs(rec_lam[:, 1]).max()/lam_knee:.2f} '
   f'del margine al ginocchio')
d_cm = delta_sens.mean()
d_diff = delta_sens - d_cm
pr(f'  scenario C (offset sensori {delta_sens} A, modo comune {d_cm:.2f} A, differenziale {np.round(d_diff, 2)} A): '
   f'I_dc a 40 s = {np.round(at(t_end - T_1, rec_idc)[2], 2)} A (atteso -delta_diff = {np.round(-d_diff, 2)} A)')
pr(f'  scenario D (anello di 2a armonica PI, k_p2 = {k_p2}, k_i2 = {k_i2:.2e} V/(A s), g_2 = {g_2}):')
for tq in (t_on + 2, t_on + 5, t_on + 10, t_on + 20, t_end - T_1):
    pr(f'    t = {tq:.0f} s: I_dc = {np.round(at(tq, rec_idc)[3], 2)} A, -a_2 = {np.round(-at(tq, rec_a2)[3], 2)} A, '
       f'v_c = {np.round(at(tq, rec_vc)[3]*1e3, 1)} mV, picchi {np.round(at(tq, rec_ipk)[3], 0)} A')

# ---- figura 3: scenario A (senza compensazione, poi anello inserito) e D
fig, ax = plt.subplots(3, 1, figsize=(FIGW, 6.2), sharex=True)
for p, c in enumerate('C0 C3 C2'.split()):
    ax[0].plot(rec_t, rec_idc[:, 0, p], c, label=f'fase {"abc"[p]}')
    ax[0].plot(rec_t, rec_idc[:, 3, p], c, ls='--', lw=0.8)
for p in range(3):
    ax[0].axhline(I_unc[p], color='k', ls=':', lw=0.6)
ax[0].axvline(t_on, color='k', lw=0.6)
ax[0].set_ylabel(r'$I_{dc}$ [A]')
ax[0].legend(loc='upper right')
ax[0].set_title(rf'A: anello PI $\alpha\beta$ inserito a $t$ = {t_on:.0f} s (continua); D: anello di 2$^a$ armonica (tratteggio)')
for p, c in enumerate('C0 C3 C2'.split()):
    ax[1].plot(rec_t, -rec_a2[:, 0, p], c)
    ax[1].plot(rec_t, -rec_a2[:, 3, p], c, ls='--', lw=0.8)
ax[1].axvline(t_on, color='k', lw=0.6)
ax[1].set_ylabel(r'$-a_2$ [A]')
ax[1].set_title(r'coefficiente di 2$^a$ armonica $-a_2$ (indicatore di saturazione a semionda)')
ax[2].plot(rec_t, rec_vc[:, 0, 0]*1e3, 'C0', label=r'$\Delta v_\alpha$ (A)')
ax[2].plot(rec_t, rec_vc[:, 0, 1]*1e3, 'C3', label=r'$\Delta v_\beta$ (A)')
ax[2].plot(rec_t, rec_vc[:, 3, 0]*1e3, 'C0', ls='--', lw=0.8, label=r'$\Delta v_\alpha$ (D)')
ax[2].plot(rec_t, rec_vc[:, 3, 1]*1e3, 'C3', ls='--', lw=0.8, label=r'$\Delta v_\beta$ (D)')
ax[2].axhline(-E_ab[0]*1e3, color='C0', ls=':', lw=0.6)
ax[2].axhline(-E_ab[1]*1e3, color='C3', ls=':', lw=0.6)
ax[2].axvline(t_on, color='k', lw=0.6)
ax[2].set_ylabel(r'$\Delta v_{\alpha\beta}$ [mV]')
ax[2].set_xlabel('t [s]')
ax[2].set_ylim(-250, 250)
ax[2].legend(loc='upper right', ncol=2)
ax[2].set_title(r'uscita del compensatore, scala tagliata a $\pm$250 mV (punti: $-E_{\alpha\beta}$)')
ax[2].set_xlim(0, t_end)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_sim_loop_insertion.pdf'))
plt.close(fig)

# ---- figura 4: scenari B e C
fig, ax = plt.subplots(2, 1, figsize=(FIGW, 4.4), sharex=True)
for p, c in enumerate('C0 C3 C2'.split()):
    ax[0].plot(rec_t, rec_idc[:, 1, p], c, label=f'fase {"abc"[p]} (B)')
    ax[0].plot(rec_t, rec_idc[:, 2, p], c, ls='--', lw=0.8, label=f'fase {"abc"[p]} (C)')
for p in range(3):
    ax[0].axhline(-d_diff[p], color='k', ls=':', lw=0.6)
ax[0].set_ylabel(r'$I_{dc}$ [A]')
ax[0].legend(loc='upper right', ncol=2)
ax[0].set_title(r'B: anello attivo da $t$ = 0; C: come B con offset dei sensori (+5, $-$2, +4) A '
                r'(punti: $-\delta_{diff}$)')
ax[1].plot(rec_t, rec_vc[:, 1, 0]*1e3, 'C0', label=r'$\Delta v_\alpha$ (B)')
ax[1].plot(rec_t, rec_vc[:, 1, 1]*1e3, 'C3', label=r'$\Delta v_\beta$ (B)')
ax[1].plot(rec_t, rec_vc[:, 2, 0]*1e3, 'C0', ls='--', lw=0.8, label=r'$\Delta v_\alpha$ (C)')
ax[1].plot(rec_t, rec_vc[:, 2, 1]*1e3, 'C3', ls='--', lw=0.8, label=r'$\Delta v_\beta$ (C)')
ax[1].axhline(-E_ab[0]*1e3, color='C0', ls=':', lw=0.6)
ax[1].axhline(-E_ab[1]*1e3, color='C3', ls=':', lw=0.6)
ax[1].set_ylabel(r'$\Delta v_{\alpha\beta}$ [mV]')
ax[1].set_xlabel('t [s]')
ax[1].legend(loc='right', ncol=2)
ax[1].set_xlim(0, t_end)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_sim_compensated.pdf'))
plt.close(fig)

# ---- figura 5: forme d'onda istantanee
fig, ax = plt.subplots(1, 2, figsize=(FIGW, 2.8))
for m, ttl in enumerate((rf'senza compensazione, $t \to$ {t_on:.0f} s', rf'anello PI $\alpha\beta$ attivo, $t \to$ {t_end:.0f} s')):
    tt = np.array(wave_t[m])
    ii = np.array(wave_i[m])          # (n, n_sc, 3)
    s = 0 if m == 0 else 1
    for p, c in enumerate('C0 C3 C2'.split()):
        ax[m].plot((tt - tt[0])*1e3, ii[:, s, p]/I_0pk, c, lw=1, label=f'fase {"abc"[p]}')
    ax[m].set_xlabel('t [ms]')
    ax[m].set_ylabel(r'$i/\hat I_0$')
    ax[m].set_title(ttl)
ax[0].legend(loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_sim_waveforms.pdf'))
plt.close(fig)

pr('=' * 100)
pr(f'figure scritte in {OUT}')
