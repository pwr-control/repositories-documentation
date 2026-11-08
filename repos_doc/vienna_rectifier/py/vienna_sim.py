"""Switched model of vienna_rectifier.slx (pwr-control/modelization-and-control, foundations_of_power_electronics/
vienna_rectifier), written step by step as the Simulink model and the library C code do it. numpy + numba.

Plant (step dt = tc = ts/200 = 0.625 us, semi-implicit Euler):
  grid emulator: three voltage sources, amplitude Step 0 -> 1 at t = 0, rate limiter 1/t_ramp (20 ms), second order
  filter 1/(tau s + 1)^2 at 50 Hz; Dyn11 transformer reduced to its leakage and copper referred to the star secondary
  (L2, R2, no-load emf E_pk at +30 deg); LCL: CFu 200 uF in star (star floating) with RCFu in series, LFu 500 uH +
  1 uH on the converter side (R1); the star of the grid is grounded, the DC link is floating (np to ground 1e5 Ohm),
  so the converter currents sum to zero.
  Vienna leg (library vienna_rectifier_ideal_switch_based_model): D1 phase -> udc+, D4 udc- -> phase, Q3 (collector
  at the phase) and Q2 (collector at np) in common emitter. Positive phase current (rectifier convention, into the
  pole): pole at np when Q3 is on, else at +vC1 (D1). Negative current: pole at np when Q2 is on, else at -vC2 (D4).
  A phase whose path for the sign it would take is blocked stays at zero current (open phase) while the voltage it
  sees lies between the two candidate pole voltages.
  Split DC link 2 x 14.4 mF, resistor load across udc, measuring resistors Rudc_measure across each half.
Modulator: gates of the T-type pattern of the library three-level PWM (carriers in phase disposition, triangle 0..1
  at 4 kHz, double update): Q1 = m > c, Q4 = m < c - 1, Q3 = not Q1, Q2 = not Q4 (Q1 and Q4 unused by the Vienna
  leg), dead time as turn-on delay of every gate (dead_time_generator_cmos_simulink).
Control at every carrier peak and valley (ts = 125 us), as afe_dq_mod1/AFE_control/afe_control:
  acquisitions quantized (1/2^11) and one step late; DDSRF-PLL (first harmonic tracker per phase, reset for 12.5 ms,
  decoupling LPFs tau = 1/omega, loop on the positive sequence, kp 0.32, ki 8, integrator clamp [0.5 1.5] pu,
  integrator initial value option); converter current in xi-eta with the PLL phase; udc PI 4/45 with
  back-calculation, reference Step at 1 ms + library LPF 20 ms (zero initial state), output limits option; dq
  vector PI 0.25/35 with |u| <= 1 and integrator decay 1e-3 while clipped; inverse Park with the PLL phase; SVM of
  sv_pwm_cm (k_offset 0.5, 1 pu = udc/sqrt(3) of phase voltage); the modulator reads the C output as high fraction,
  so the converter voltage is -u_ctrl (the inversion of the current acquisition, gain -1, closes the loop with the
  right sign).
Options of run() not in the model: pll_legacy (DDSRF-PLL as saved, on the negative sequence), pll_i1_init, dead_time,
  vienna_cm (2: zero sequence on the current sign), i_ref_min (lower limit of the udc PI), udc_aw (1: conditional
  integration of the udc PI in place of the back-calculation), ref_init (initial state of the udc reference filter),
  udc_init (precharge), R_load_step_t / R_load_step (load step), v_ff (1: feed-forward of the capacitor voltage on the
  output of the current PI, sample advanced by 1.5 ts).
Not modelled: clock alignment, power meter, conduction drops (diode 2 V, switch 1.5 mOhm) and RC snubbers of the
  devices.
"""
import math
import numpy as np
from numba import njit

TWO_PI = 2.0 * math.pi


class Data:
    """numbers of init_model.m (param.*), library classes and model blocks"""
    def __init__(self):
        self.f_nom = 50.0; self.w = TWO_PI * self.f_nom
        self.fpwm = 4e3; self.ts = 1.0 / (2.0 * self.fpwm); self.tc = self.ts / 200.0; self.dead_time = 3e-6
        self.V_ll = 690.0; self.pwr_nom = 250e3
        self.is_nom = self.pwr_nom / math.sqrt(3) / self.V_ll / 0.9
        self.ubez = self.V_ll * math.sqrt(2.0 / 3.0); self.ibez = self.is_nom * math.sqrt(2.0); self.udc_nom = 1070.0
        # LCL filter and DC link (afe_hw_param)
        self.L1 = 500e-6 + 1e-6; self.R1 = 157 * 0.05 * 500e-6; self.Cf = 200e-6; self.Rc = 50e-3
        self.C1 = self.C2 = 14.4e-3; self.R_meas = 107e3; self.R_load = 5.0
        # transformer Dyn11 of the emulator (grid_param + three_phase_transformer_setup, delta factors of 2026-11-04)
        pwr_tr = 1e6; ucc = 5.0; eta = 95.0; n2 = 14; n1 = math.floor(n2 * math.sqrt(3))
        i1_nom = pwr_tr / math.sqrt(3) / self.V_ll
        Lsigma = self.V_ll * ucc / 100 / math.sqrt(3) / i1_nom / self.w
        Ld1 = 1.5 * Lsigma; Ld2 = Lsigma / 2; Rd_star = 0.5 * ((1 - eta / 100) * pwr_tr / 3 / i1_nom ** 2)
        Rd1 = 3 * Rd_star; Rd2 = Rd_star
        k21 = (n2 / n1) ** 2
        self.L2 = Ld2 + Ld1 * k21; self.R2 = Rd2 + Rd1 * k21
        self.Ld1, self.Ld2, self.Rd1, self.Rd2, self.n1, self.n2 = Ld1, Ld2, Rd1, Rd2, n1, n2
        self.E_pk = self.V_ll * n2 / n1 * math.sqrt(2)       # no-load phase peak at the secondary
        self.phi_tr = math.radians(30.0)                       # Dyn11: secondary 30 deg ahead
        # control (ctrl_afe_setup + init)
        self.kp_i, self.ki_i, self.clip_release = 0.25, 35.0, 1e-3
        self.kp_u, self.ki_u, self.i_lim = 4.0, 45.0, 1.65
        self.tau_ref, self.t_ref = 20e-3, 1e-3
        self.adc_q = 1.0 / 2 ** 11
        self.pll_kp, self.pll_ki = 0.32, 8.0
        # grid emulator ramp
        self.t_ramp = 20e-3; self.tau_sof = 1.0 / (TWO_PI * 50.0)
        # first harmonic tracker (ctrl_afe_setup.first_harmonic_tracker: delta 0.05, poles -1 -4 omega, forward Euler)
        om = self.w; ts = self.ts; dl = 0.05
        A = np.array([[0.0, 1.0], [-om ** 2, -dl * om]]); C = np.array([[1.0, 0.0]])
        Ad = np.eye(2) + A * ts
        pd = np.exp(ts * np.array([-1.0, -4.0]) * om)
        At, Bt = Ad.T, C.T
        ctrb = np.hstack([Bt, At @ Bt])
        phi = (At - pd[0] * np.eye(2)) @ (At - pd[1] * np.eye(2))
        K = np.array([[0.0, 1.0]]) @ np.linalg.inv(ctrb) @ phi
        self.l_fht = K.ravel()
        a = -om ** 2; b = -dl * om
        self.a_fht = np.array([1.0 + 0.5 * a * ts * ts, ts + 0.5 * b * ts * ts,
                               a * ts + 0.5 * a * b * ts * ts, 1.0 + b * ts + 0.5 * (a + b * b) * ts * ts])
        self.t_fht_reset = 0.0125

    def pack(self):
        return np.array([self.w, self.ts, self.tc, self.dead_time, self.ubez, self.ibez, self.udc_nom,
                         self.L1, self.R1, self.Cf, self.Rc, self.C1, self.C2, self.R_meas, self.R_load,
                         self.L2, self.R2, self.E_pk, self.phi_tr,
                         self.kp_i, self.ki_i, self.clip_release, self.kp_u, self.ki_u, self.i_lim,
                         self.tau_ref, self.t_ref, self.adc_q, self.pll_kp, self.pll_ki, self.t_ramp, self.tau_sof,
                         self.l_fht[0], self.l_fht[1], self.a_fht[0], self.a_fht[1], self.a_fht[2], self.a_fht[3],
                         self.t_fht_reset])


@njit(cache=True)
def wrap(x):
    return -math.pi + ((x + 3.0 * math.pi) % TWO_PI)


@njit(cache=True)
def quant(x, q):
    return q * math.floor(x / q + 0.5)


@njit(cache=True)
def simulate(p, T_end, i_ref_min, pll_i1_init, udc_init, use_dead_time, vienna_cm, log_from, R_load_step_t, R_load_step, ref_init,
             pll_legacy, v_ff, udc_aw):
    """returns per-control-step logs and full-resolution waveforms from log_from to T_end.
    i_ref_min: lower limit of the active current reference (model: -i_lim); pll_i1_init: initial value of the PLL
    integrator [pu] (init_model.m: 1; frt_dqpll starts from 0, held at 0.5 by the clamp); udc_init: initial DC-link
    voltage [V]; vienna_cm: 0 the SVM zero sequence of sv_pwm_cm, 1 or 2 a zero sequence that gives each pole voltage
    the sign of its phase current where possible (1: sign of the measured current, 2: sign of the reference current
    1.5 ts ahead; options for the comparison, not in the model); R_load_step_t / R_load_step: time and new value of
    the load; ref_init: initial value of the udc reference (model: 0); pll_legacy: 1 the DDSRF-PLL as saved (loop on
    the negative sequence through a gain -1, no clamp of the frequency integrator); v_ff: gain of a feed-forward of
    the measured capacitor voltage added to the output of the current PI (option, not in the model: 0), the sample
    advanced by 1.5 ts with the PLL frequency; udc_aw: anti-windup of the udc PI, 0 as the model (velocity form, state =
    output - proportional part, also while clamped), 1 conditional integration (option: the state holds while the output
    is clamped)."""
    w = p[0]; ts = p[1]; dt = p[2]; dead = p[3]; ubez = p[4]; ibez = p[5]; udc_nom = p[6]
    L1 = p[7]; R1 = p[8]; Cf = p[9]; Rc = p[10]; C1 = p[11]; C2 = p[12]; Rm = p[13]; R_load = p[14]
    L2 = p[15]; R2 = p[16]; E_pk = p[17]; phi_tr = p[18]
    kp_i = p[19]; ki_i = p[20]; clip_rel = p[21]; kp_u = p[22]; ki_u = p[23]; i_lim = p[24]
    tau_ref = p[25]; t_ref = p[26]; q = p[27]; pll_kp = p[28]; pll_ki = p[29]; t_ramp = p[30]; tau_sof = p[31]
    l1f = p[32]; l2f = p[33]; a11 = p[34]; a12 = p[35]; a21 = p[36]; a22 = p[37]; t_fht_reset = p[38]

    nsub = int(round(ts / dt))
    nctrl = int(round(T_end / ts))
    nd = int(round(dead / dt)) if use_dead_time else 0
    s3 = math.sqrt(3.0)

    # plant states
    i1 = np.zeros(3); vc = np.zeros(3); i2 = np.zeros(3)
    dirn = np.zeros(3)                      # +1 / -1 conducting with that sign, 0 open
    vC1 = udc_init / 2.0; vC2 = udc_init / 2.0
    sof1 = 0.0; sof2 = 0.0
    # gates after the dead time: counters of the raw on time
    cnt3 = np.zeros(3); cnt2 = np.zeros(3)
    Q2 = np.ones(3); Q3 = np.ones(3)
    m_conv = np.zeros(3)
    # control states
    meas_i = np.zeros(3); meas_u = np.zeros(3); meas_udc = udc_init / udc_nom
    fs1 = np.zeros(3); fs2 = np.zeros(3); fout = np.zeros(3)
    lp_x = np.zeros(4); lp_y = np.zeros(4)
    th = 0.0; pll_i1 = pll_i1_init
    g_ref = ts / tau_ref; ref_f = ref_init; ref_in_prev = ref_init
    x_u = 0.0
    x_id = 0.0; x_iq = 0.0; clip_prev = 0
    # logs
    L_t = np.zeros(nctrl); L_udc = np.zeros(nctrl); L_unb = np.zeros(nctrl); L_ref = np.zeros(nctrl)
    L_iref = np.zeros(nctrl); L_id = np.zeros(nctrl); L_iq = np.zeros(nctrl); L_m = np.zeros(nctrl)
    L_perr = np.zeros(nctrl); L_open = np.zeros(nctrl); L_ipk = np.zeros(nctrl); L_E = np.zeros(nctrl)
    L_clip = np.zeros(nctrl); L_P = np.zeros(nctrl)
    k0 = int(round(log_from / dt))
    nw = nctrl * nsub - k0
    if nw < 0:
        nw = 0
    W_i1 = np.zeros((3, nw)); W_i2 = np.zeros((3, nw)); W_vf = np.zeros((3, nw)); W_pole = np.zeros((3, nw))
    W_vnp = np.zeros(nw); W_vC1 = np.zeros(nw); W_vC2 = np.zeros(nw); W_e = np.zeros((3, nw)); W_m = np.zeros((3, nw))
    kstep = 0
    pole = np.zeros(3)
    vf = np.zeros(3)
    e = np.zeros(3)
    Pp = np.zeros(3); Pm = np.zeros(3)
    ipk = 0.0
    p_acc = 0.0
    for k in range(nctrl):
        t = k * ts
        # ===================== control at the trigger k =====================
        # LPF reference: y[k] = g x[k-1] + (1 - g) y[k-1]
        ref_in = 1.0 if t >= t_ref else ref_init
        ref_f = g_ref * ref_in_prev + (1.0 - g_ref) * ref_f
        ref_in_prev = ref_in
        # PLL (DDSRF) on the delayed voltage samples
        rst = t < t_fht_reset
        uf = np.zeros(3)
        for x in range(3):
            if rst:
                fs1[x] = 0.0; fs2[x] = 0.0; fout[x] = 0.0
            err = meas_u[x] - fout[x]
            s1n = l1f * err + a11 * fs1[x] + a12 * fs2[x]
            s2n = l2f * err + a21 * fs1[x] + a22 * fs2[x]
            fout[x] = fs1[x]
            fs1[x] = s1n; fs2[x] = s2n
            uf[x] = fout[x]
        al = 2.0 / 3.0 * (uf[0] - 0.5 * uf[1] - 0.5 * uf[2]); be = (uf[1] - uf[2]) / s3
        # ddsrf
        gl = ts * w      # LPF gain ts/tau with tau = 1/omega
        for j in range(4):
            lp_y[j] = gl * lp_x[j] + (1.0 - gl) * lp_y[j]
        th2 = wrap(2.0 * th)
        c1 = math.cos(th); s1 = math.sin(th); c2 = math.cos(th2); s2 = math.sin(th2)
        pr_x = al * c1 + be * s1; pr_y = -al * s1 + be * c1
        nr_x = al * c1 - be * s1; nr_y = al * s1 + be * c1
        dpx = lp_y[2] * c2 + lp_y[3] * s2; dpy = -lp_y[2] * s2 + lp_y[3] * c2
        dnx = lp_y[0] * c2 - lp_y[1] * s2; dny = lp_y[0] * s2 + lp_y[1] * c2
        lp_x[0] = pr_x - dpx; lp_x[1] = pr_y - dpy; lp_x[2] = nr_x - dnx; lp_x[3] = nr_y - dny
        vn = 1.0 / math.sqrt(max(al * al + be * be, 0.02))
        if pll_legacy == 0:
            epll = lp_y[1] * vn             # filtered decoupled positive sequence (ddsrf out 2)
        else:
            epll = -lp_y[3] * vn            # as saved: negative sequence (ddsrf out 4) through the gain -1
        pll_i1 = pll_i1 + pll_ki * ts * epll
        if pll_legacy == 0:                 # clamp of the frequency integrator (added, as in frt_dqpll)
            if pll_i1 > 1.5:
                pll_i1 = 1.5
            if pll_i1 < 0.5:
                pll_i1 = 0.5
        wpu = pll_kp * epll + pll_i1
        th_out = th
        th = wrap(th + ts * w * wpu)
        # currents in xi-eta with the PLL phase output
        ia = meas_i[0]; ib = meas_i[1]; ic = meas_i[2]
        ial = 2.0 / 3.0 * (ia - 0.5 * ib - 0.5 * ic); ibe = (ib - ic) / s3
        co = math.cos(th_out); so = math.sin(th_out)
        i_d = ial * co + ibe * so; i_q = -ial * so + ibe * co
        # udc PI (velocity form with back-calculation)
        e_u = ref_f - meas_udc
        Pu = kp_u * e_u
        out = Pu + ki_u * ts * e_u + x_u
        clamped = False
        if out > i_lim:
            out = i_lim; clamped = True
        if out < i_ref_min:
            out = i_ref_min; clamped = True
        if udc_aw == 0:
            x_u = out - Pu
        elif not clamped:
            x_u = x_u + ki_u * ts * e_u
        # dq vector PI
        ed = out - i_d; eq = 0.0 - i_q
        if clip_prev == 1:
            cd = x_id; cq = x_iq
            nxd = x_id * (1.0 - clip_rel * ts); nxq = x_iq * (1.0 - clip_rel * ts)
        else:
            cd = x_id + ki_i * ts * ed; cq = x_iq + ki_i * ts * eq
            nxd = cd; nxq = cq
        ud = kp_i * ed + cd; uq = kp_i * eq + cq
        if v_ff != 0.0:
            # feed-forward of the capacitor voltage (option): u_pole = -u_ctrl udc/sqrt(3) must cancel vf
            ua_m = 2.0 / 3.0 * (meas_u[0] - 0.5 * meas_u[1] - 0.5 * meas_u[2]); ub_m = (meas_u[1] - meas_u[2]) / s3
            adv = 1.5 * ts * w * wpu
            ca = math.cos(adv); sa = math.sin(adv)
            ua_p = ua_m * ca - ub_m * sa; ub_p = ua_m * sa + ub_m * ca
            kff = v_ff * s3 * ubez / max(meas_udc * udc_nom, 100.0)
            ud -= kff * (ua_p * co + ub_p * so); uq -= kff * (-ua_p * so + ub_p * co)
        mag = math.sqrt(ud * ud + uq * uq)
        clip = 1 if mag >= 1.0 else 0
        if clip == 1:
            ud = ud / mag; uq = uq / mag
        x_id = nxd; x_iq = nxq; clip_prev = clip
        # inverse Park, SVM (sv_pwm_cm with k_offset 0.5 = centred zero sequence), m = -(2 d - 1) of the requested
        ua_c = ud * co - uq * so; ub_c = ud * so + uq * co
        ux0 = ua_c; ux1 = -0.5 * ua_c + 0.5 * s3 * ub_c; ux2 = -0.5 * ua_c - 0.5 * s3 * ub_c
        mr0 = 2.0 / s3 * ux0; mr1 = 2.0 / s3 * ux1; mr2 = 2.0 / s3 * ux2
        mx = max(mr0, max(mr1, mr2)); mn = min(mr0, min(mr1, mr2))
        v0 = -0.5 * (mx + mn)
        mc0 = -(mr0 + v0); mc1 = -(mr1 + v0); mc2 = -(mr2 + v0)
        if vienna_cm >= 1:
            # zero sequence that gives each pole the sign of its current (current direction from the last sample)
            lo = -1.0 - min(-mr0, min(-mr1, -mr2)); hi = 1.0 - max(-mr0, max(-mr1, -mr2))
            # want sign(-mr_x + z) = sign(i_x): -mr_x + z >= 0 for i_x > 0, <= 0 for i_x < 0
            zl = lo; zh = hi
            mm = (-mr0, -mr1, -mr2)
            # current direction: measured sample (option 1) or reference current advanced by 1.5 ts (option 2)
            thr = th_out + 1.5 * w * ts
            ir0 = out * math.cos(thr); ir1 = out * math.cos(thr - TWO_PI / 3.0); ir2 = out * math.cos(thr + TWO_PI / 3.0)
            irr = (ir0, ir1, ir2)
            for x in range(3):
                ii = meas_i[x] if vienna_cm == 1 else irr[x]
                if ii > 0.0:
                    zl = max(zl, -mm[x])
                elif ii < 0.0:
                    zh = min(zh, -mm[x])
            if zl <= zh:
                z = min(max(-v0, zl), zh)
                mc0 = mm[0] + z; mc1 = mm[1] + z; mc2 = mm[2] + z
        m_conv[0] = min(max(mc0, -1.0), 1.0); m_conv[1] = min(max(mc1, -1.0), 1.0); m_conv[2] = min(max(mc2, -1.0), 1.0)
        # ---------------- logs ----------------
        L_t[k] = t; L_udc[k] = vC1 + vC2; L_unb[k] = vC2 - vC1; L_ref[k] = ref_f; L_iref[k] = out
        L_id[k] = i_d; L_iq[k] = i_q; L_m[k] = mag; L_clip[k] = clip
        L_perr[k] = wrap(th_out - (w * t + phi_tr))
        L_E[k] = math.sqrt(al * al + be * be)
        L_ipk[k] = ipk; ipk = 0.0
        L_P[k] = p_acc / nsub; p_acc = 0.0
        # samples for the next trigger (acquisition unit delay)
        nopen = 0
        for x in range(3):
            meas_i[x] = quant(i1[x] / ibez, q)
            meas_u[x] = quant(vf[x] / ubez, q)
            if dirn[x] == 0:
                nopen += 1
        meas_udc = quant((vC1 + vC2) / udc_nom, q)
        L_open[k] = nopen
        # ===================== plant over [k ts, (k+1) ts] =====================
        for j in range(nsub):
            tt = t + j * dt
            if tt >= R_load_step_t:
                R_load = R_load_step
            # grid emulator amplitude: ramp + second order filter (zero state)
            ramp = min(tt / t_ramp, 1.0)
            d1 = (ramp - sof1) / tau_sof; d2 = (sof1 - sof2) / tau_sof
            sof1 += dt * d1; sof2 += dt * d2
            amp = sof2
            thg = w * tt + phi_tr
            for x in range(3):
                e[x] = E_pk * amp * math.cos(thg - x * TWO_PI / 3.0)
            # carrier (0 at k ts, 1 at (k+1) ts on rising slopes, double update: the slope alternates)
            if k % 2 == 0:
                cr = j / nsub
            else:
                cr = 1.0 - j / nsub
            # raw T-type gates and turn-on delay
            for x in range(3):
                g1 = m_conv[x] > cr
                g4 = m_conv[x] < cr - 1.0
                r3 = not g1
                r2 = not g4
                if r3:
                    cnt3[x] += 1
                else:
                    cnt3[x] = 0
                if r2:
                    cnt2[x] += 1
                else:
                    cnt2[x] = 0
                Q3[x] = 1.0 if cnt3[x] > nd else 0.0
                Q2[x] = 1.0 if cnt2[x] > nd else 0.0
            # node voltages of the capacitors (star of the capacitors at the mean of the grid emf)
            em = (e[0] + e[1] + e[2]) / 3.0
            for x in range(3):
                vf[x] = em + vc[x] + Rc * (i2[x] - i1[x])
                Pp[x] = 0.0 if Q3[x] > 0.5 else vC1
                Pm[x] = 0.0 if Q2[x] > 0.5 else -vC2
            # conduction set and np potential
            vnp = 0.0
            for it in range(4):
                ncond = 0; acc = 0.0
                for x in range(3):
                    if dirn[x] != 0:
                        pole[x] = Pp[x] if dirn[x] > 0 else Pm[x]
                        acc += vf[x] - R1 * i1[x] - pole[x]
                        ncond += 1
                if ncond >= 2:
                    vnp = acc / ncond
                else:
                    # zero or one conducting phase: no current can flow; np floats at the mean of the open voltages
                    acc2 = 0.0
                    for x in range(3):
                        acc2 += vf[x]
                    vnp = acc2 / 3.0
                changed = False
                for x in range(3):
                    if dirn[x] == 0:
                        vo = vf[x] - vnp
                        if vo > Pp[x] + 1e-9:
                            dirn[x] = 1.0; changed = True
                        elif vo < Pm[x] - 1e-9:
                            dirn[x] = -1.0; changed = True
                        elif Q2[x] > 0.5 and Q3[x] > 0.5:
                            dirn[x] = 1.0; changed = True
                if not changed:
                    break
            ncond = 0
            for x in range(3):
                if dirn[x] != 0:
                    ncond += 1
            if ncond < 2:
                for x in range(3):
                    dirn[x] = 0.0; i1[x] = 0.0
                    pole[x] = vf[x] - vnp
            else:
                for x in range(3):
                    if dirn[x] != 0:
                        pole[x] = Pp[x] if dirn[x] > 0 else Pm[x]
                    else:
                        pole[x] = vf[x] - vnp
            # integrate the converter currents
            ipos = 0.0; ineg = 0.0
            for x in range(3):
                if dirn[x] != 0:
                    i1[x] += dt / L1 * (vf[x] - R1 * i1[x] - pole[x] - vnp)
            # zero crossings
            for x in range(3):
                both = Q2[x] > 0.5 and Q3[x] > 0.5
                if dirn[x] > 0 and i1[x] < 0.0:
                    if both:
                        dirn[x] = -1.0
                    else:
                        dirn[x] = 0.0
                elif dirn[x] < 0 and i1[x] > 0.0:
                    if both:
                        dirn[x] = 1.0
                    else:
                        dirn[x] = 0.0
            nopen = 0; resid = 0.0
            for x in range(3):
                if dirn[x] == 0:
                    resid += i1[x]; i1[x] = 0.0; nopen += 1
            if nopen == 1:
                for x in range(3):
                    if dirn[x] != 0:
                        i1[x] += resid / 2.0
            elif nopen >= 2:
                for x in range(3):
                    i1[x] = 0.0; dirn[x] = 0.0
            # DC link currents
            for x in range(3):
                if dirn[x] > 0 and Q3[x] < 0.5:
                    ipos += i1[x]
                elif dirn[x] < 0 and Q2[x] < 0.5:
                    ineg += i1[x]
                ai = abs(i1[x])
                if ai > ipk:
                    ipk = ai
            iload = (vC1 + vC2) / R_load
            p_acc += (vC1 + vC2) * iload
            # grid currents and capacitors (semi-implicit: new currents)
            for x in range(3):
                i2[x] += dt / L2 * (e[x] - R2 * i2[x] - vf[x])
            m2 = (i2[0] + i2[1] + i2[2]) / 3.0
            for x in range(3):
                i2[x] -= m2
                vc[x] += dt / Cf * (i2[x] - i1[x])
            vC1 += dt / C1 * (ipos - iload - vC1 / Rm)
            vC2 += dt / C2 * (-ineg - iload - vC2 / Rm)
            # waveforms
            kk = kstep - k0
            if kk >= 0 and kk < nw:
                for x in range(3):
                    W_i1[x, kk] = i1[x]; W_i2[x, kk] = i2[x]; W_vf[x, kk] = vf[x]; W_pole[x, kk] = pole[x]
                    W_e[x, kk] = e[x]; W_m[x, kk] = m_conv[x]
                W_vnp[kk] = vnp; W_vC1[kk] = vC1; W_vC2[kk] = vC2
            kstep += 1
    return (L_t, L_udc, L_unb, L_ref, L_iref, L_id, L_iq, L_m, L_open, L_ipk, L_clip, L_P, L_perr, L_E,
            W_i1, W_i2, W_vf, W_pole, W_vnp, W_vC1, W_vC2, W_e, W_m)


def run(d=None, T_end=0.4, i_ref_min=None, pll_i1_init=1.0, udc_init=None, dead_time=True, vienna_cm=0,
        log_from=None, R_load_step_t=1e9, R_load_step=None, ref_init=0.0, pll_legacy=False, v_ff=0.0, udc_aw=0):
    d = Data() if d is None else d
    i_ref_min = -d.i_lim if i_ref_min is None else i_ref_min
    udc_init = d.udc_nom if udc_init is None else udc_init
    log_from = T_end - 0.04 if log_from is None else log_from
    R_load_step = d.R_load if R_load_step is None else R_load_step
    r = simulate(d.pack(), T_end, i_ref_min, pll_i1_init, udc_init, 1 if dead_time else 0, vienna_cm, log_from,
                 R_load_step_t, R_load_step, ref_init, 1 if pll_legacy else 0, v_ff, udc_aw)
    keys = ['t', 'udc', 'unb', 'ref', 'iref', 'id', 'iq', 'm', 'open', 'ipk', 'clip', 'P', 'perr', 'uabs',
            'i1', 'i2', 'vf', 'pole', 'vnp', 'vC1', 'vC2', 'e', 'mc']
    out = dict(zip(keys, r))
    out['dt'] = d.tc; out['log_from'] = log_from; out['data'] = d
    return out


def thd(x, dt, f0=50.0, nper=2, nh=40):
    """THD of the last nper periods of x (sampled at dt): harmonics 2..nh over the fundamental; also the spectrum"""
    n = int(round(nper / f0 / dt))
    seg = x[-n:]
    X = np.fft.rfft(seg) / n * 2
    k1 = nper
    fund = abs(X[k1])
    h = np.array([abs(X[k1 * hh]) for hh in range(2, nh + 1)])
    return math.sqrt(np.sum(h ** 2)) / fund, fund, X
