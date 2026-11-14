"""Switched simulation of a six-pulse diode bridge with n series-connected diodes per arm (diode stack).
Companion model of diode_stack_design_guidelines.tex (App. A); used by diode_stack_calc.py.

Circuit: three-phase source (line-to-line Vll) with the commutation inductance Lc (and Rc) per phase, six arms,
each arm = n levels in series, each level = diode || Rb || (Rsn + Csn), DC side = Ld (+ Rd) in series with the
load resistance Rload (optionally Cdc across Rload); optional damping network Rx + Cx across the DC terminals.

Diode: charge-control model of Lauritzen and Ma (charge q_M, transit time T_M, lifetime tau), reduced to two
piecewise-linear modes; the junction capacitance is neglected (Cj << Csn):
  conducting (q_E > 0):  v = VT0 + rT*i_d,              dq_M/dt = i_d - q_M/tau
  blocked    (q_E = 0):  i_d = -q_M/T_M + v/Roff,       dq_M/dt = -q_M*(1/tau + 1/T_M)
  conducting -> blocked when q_E = q_M + T_M*i_d reaches zero (peak of the reverse current),
  blocked -> conducting when v exceeds VT0.
In both modes a level is a Thevenin source v = E + R*i_arm, so the network is solved in closed form at every step
(no matrix); the states are the line currents, the DC current, and (v_C, q_M) of every level (plus the voltages
of Cdc and Cx when present).
Explicit trapezoidal (Heun) integration with a fixed step; all the cases of a batch are advanced together
(leading array axis), so that a parameter sweep costs about as much as a single run.

Arms: 0, 1, 2 = upper arms of phases a, b, c (anode on the AC terminal); 3, 4, 5 = lower arms (cathode on the AC
terminal). Arrays per level have shape (cases, 6, n). Only numpy is needed (numpy 1.x and 2.x).
Run:  python diode_stack_sim.py      (reference case of the note, prints the key figures; about 20 s)"""
import numpy as np

PI = np.pi
PH = np.array([0.0, 2*PI/3, 4*PI/3])


def lm_params(Qrr, Irrm, a):
    """Lauritzen-Ma parameters (tau, TM) from one recovery point (Qrr [C], Irrm [A]) on a long current ramp of
    slope a [A/s]:  t_tail = Qrr/Irrm - Irrm/(2a),  tau = t_tail + Irrm/a,  TM = tau*t_tail*a/Irrm."""
    Qrr, Irrm = np.asarray(Qrr, float), np.asarray(Irrm, float)
    t_tail = Qrr/Irrm - Irrm/(2*a)
    if np.any(t_tail <= 0):
        raise ValueError("Irrm >= sqrt(2*a*Qrr): no Lauritzen-Ma fit (t_tail <= 0)")
    tau = t_tail + Irrm/a
    return tau, tau*t_tail*a/Irrm


def lm_recovery(tau, TM, a):
    """Long-ramp recovery of the Lauritzen-Ma diode at the slope a: (Qrr, Irrm, t_a, t_tail)."""
    t_tail = tau*TM/(tau + TM)
    return 0.5*a*(tau**2 - t_tail**2), a*(tau - t_tail), tau - t_tail, t_tail


def level_array(cases, n, weak, nominal):
    """(cases, 6, n) array with `weak` on level 0 of every arm and `nominal` on the other levels;
    weak and nominal are scalars or arrays (cases,)."""
    a = np.empty((cases, 6, n))
    a[:, :, 0] = np.broadcast_to(np.asarray(weak, float), (cases,))[:, None]
    a[:, :, 1:] = np.broadcast_to(np.asarray(nominal, float), (cases,))[:, None, None]
    return a


def simulate(Vll, f, Lc, Rload, Ld, tau, TM, Roff, Rb, Rsn, Csn, VT0=0.9, rT=1e-3, Rc=0.0, Rd=0.0, Cdc=0.0,
             vcap0=0.0, iD0=0.0, Rx=1.0, Cx=0.0, periods=2.5, dt=0.25e-6, rec=(), rec_every=2, verbose=False):
    """Simulate `periods` grid periods and evaluate the last one.
    Vll, Lc, Rc, Rload, Ld, Rd, Cdc, vcap0, iD0, Rx, Cx: scalars or arrays (cases,). tau, TM, Roff, Rb, Rsn, Csn:
    arrays broadcastable to (cases, 6, n). Cdc = 0: no DC capacitor (R-L load); Cx = 0: no damping network on the
    DC terminals; iD0: initial DC current (it shortens the settling when Ld/Rload is longer than a period).
    rec: indices of the cases whose waveforms are stored every rec_every steps (last period).
    Returns a dict of arrays; quantities per level have shape (cases, 6, n), reverse voltages are positive
    (Vpk: highest reverse voltage, Tpk: time at which it occurs).
    out["wave"][case] holds the waveforms: t, v (level voltage, anode - cathode), idio, isn, ia (arm currents),
    Vdc, iD, iL, uk (AC terminals to the source neutral; P = uk[:, k] - sum of v of arm k, N = P - Vdc)."""
    tau, TM, Roff, Rb, Rsn, Csn = np.broadcast_arrays(*[np.asarray(x, float) for x in (tau, TM, Roff, Rb, Rsn, Csn)])
    if tau.ndim != 3 or tau.shape[1] != 6:
        raise ValueError("level arrays must have shape (cases, 6, n)")
    B, _, n = tau.shape
    col = lambda x: np.broadcast_to(np.asarray(x, float), (B,)).copy()
    Vll, Lc, Rc, Rload, Ld, Rd, Cdc, vcap0, iD0, Rx, Cx = (col(x) for x in (Vll, Lc, Rc, Rload, Ld, Rd, Cdc, vcap0, iD0,
                                                                         Rx, Cx))
    w, T = 2*PI*f, 1.0/f
    Vph = (np.sqrt(2.0/3.0)*Vll)[:, None]
    hascap = Cdc > 0
    iCdc = np.where(hascap, 1.0/np.where(hascap, Cdc, 1.0), 0.0)
    Rser = np.where(hascap, Rd, Rd + Rload)                 # resistance in series with Ld
    hasx = Cx > 0
    iRx = np.where(hasx, 1.0/Rx, 0.0)                       # damping network Rx + Cx across the DC terminals
    iRxCx = np.where(hasx, 1.0/(Rx*np.where(hasx, Cx, 1.0)), 0.0)
    # constants per level
    iRsn, iTM, itau = 1.0/Rsn, 1.0/TM, 1.0/tau
    g = 1.0/Rb + iRsn
    R_blk = 1.0/(1.0/Roff + g)
    den = 1.0 + rT*g
    R_con, E0c, kEc = rT/den, VT0/den, rT*iRsn/den
    iRC, kb, iRoff = iRsn/Csn, itau + iTM, 1.0/Roff
    iLc, iLd, Rc2 = (1.0/Lc)[:, None], 1.0/Ld, Rc[:, None]

    # states
    iL, iD, vcap, vx = np.zeros((B, 3)), iD0.copy(), vcap0.copy(), np.zeros(B)
    vC, qM = np.zeros((B, 6, n)), np.zeros((B, 6, n))
    cond = np.zeros((B, 6, n), bool)
    ia = np.zeros((B, 6))

    def network():
        R = np.where(cond, R_con, R_blk)
        Ra = R.sum(-1)
        Ru, Rl = Ra[:, :3], Ra[:, 3:]
        iRk = 1.0/(Ru + Rl)
        return R, Ru, Rl, iRk, 1.0/(iRk.sum(-1) + iRx)

    R, Ru, Rl, iRk, Rpar = network()

    def rhs(t, iL, iD, vcap, vx, vC, qM):
        E = np.where(cond, E0c + kEc*vC, (qM*iTM + vC*iRsn)*R_blk)
        Ea = E.sum(-1)
        Ek = Ea[:, :3] + Ea[:, 3:] - Rl*iL
        Vdc = -(iD - vx*iRx + (Ek*iRk).sum(-1))*Rpar       # KCL on the positive DC terminal
        iu = -(Vdc[:, None] + Ek)*iRk
        ia[:, :3] = iu
        ia[:, 3:] = iu - iL
        v = E + R*ia[:, :, None]                            # level voltage, anode - cathode
        vu = Ea[:, :3] + Ru*iu                              # upper arm voltage = AC terminal - P
        uk = vu - vu.mean(-1, keepdims=True)                # AC terminals referred to the source neutral
        e = Vph*np.sin(w*t - PH)
        diL = (e - uk - Rc2*iL)*iLc
        diD = (Vdc - Rser*iD - np.where(hascap, vcap, 0.0))*iLd
        dvcap = (iD - vcap/Rload)*iCdc
        idc = ia[:, :, None] - v*g + vC*iRsn                # diode current of a conducting level
        dqM = np.where(cond, idc - qM*itau, -qM*kb)
        return diL, diD, dvcap, (Vdc - vx)*iRxCx, (v - vC)*iRC, dqM, v, idc, Vdc, uk

    N = int(round(periods*T/dt))
    Nm = int(round(T/dt))                                   # steps of the evaluated (last) period
    s0 = N - Nm
    z = lambda: np.zeros((B, 6, n))
    acc = dict(Vpk=z(), tpk=z(), v1=z(), v2=z(), isn2=z(), Isn_pk=z(), Irrm=z(), Qrr=z(), ifw=z(), if2=z(), plk=z(),
               erec=z(), ifpk=z())
    acc_vdc, acc_id, acc_iL2, vs_pk = np.zeros(B), np.zeros(B), np.zeros((B, 3)), np.zeros((B, 6))
    acc_px, vdc_pk = np.zeros(B), np.zeros(B)
    rec = [rec] if np.isscalar(rec) else list(rec)
    Nr = Nm//rec_every
    waves = {c: dict(t=np.zeros(Nr), v=np.zeros((Nr, 6, n)), idio=np.zeros((Nr, 6, n)), isn=np.zeros((Nr, 6, n)),
                     ia=np.zeros((Nr, 6)), Vdc=np.zeros(Nr), iD=np.zeros(Nr), iL=np.zeros((Nr, 3)),
                     uk=np.zeros((Nr, 3))) for c in rec}
    nflip = 0
    for s in range(N):
        t = s*dt
        k1 = rhs(t, iL, iD, vcap, vx, vC, qM)
        # mode changes, evaluated on the solution at the beginning of the step
        locked = None
        for _ in range(8):
            v, idc = k1[6], k1[7]
            flip = (cond & (qM + TM*idc <= 0.0)) | (~cond & (v > VT0))
            if locked is not None:
                flip &= ~locked                             # a level changes mode at most once per step
            if not flip.any():
                break
            cond ^= flip
            locked = flip if locked is None else (locked | flip)
            nflip += int(flip.sum())
            R, Ru, Rl, iRk, Rpar = network()
            k1 = rhs(t, iL, iD, vcap, vx, vC, qM)
        diL, diD, dvcap, dvx, dvC, dqM, v, idc, Vdc, uk = k1
        if s >= s0:
            vr = -v
            isn = (v - vC)*iRsn
            irev = np.where(cond, -idc, qM*iTM)             # reverse current without the leakage term
            acc["tpk"][vr > acc["Vpk"]] = t                 # time of the highest reverse voltage of each level
            np.maximum(acc["Vpk"], vr, out=acc["Vpk"])
            acc["v1"] += vr
            acc["v2"] += v*v
            acc["isn2"] += isn*isn
            np.maximum(acc["Isn_pk"], np.abs(isn), out=acc["Isn_pk"])
            np.maximum(acc["Irrm"], irev, out=acc["Irrm"])
            acc["Qrr"] += np.maximum(irev, 0.0)
            ifw = np.where(cond, np.maximum(idc, 0.0), 0.0)
            acc["ifw"] += ifw
            acc["if2"] += ifw*ifw
            np.maximum(acc["ifpk"], ifw, out=acc["ifpk"])
            acc["plk"] += np.where(cond, 0.0, v*v*iRoff)
            acc["erec"] += np.where(cond, 0.0, vr*qM*iTM)   # turn-off loss in the diode (tail current x voltage)
            acc_vdc += Vdc
            acc_id += iD
            acc_iL2 += iL*iL
            acc_px += (Vdc - vx)**2*iRx
            np.maximum(vdc_pk, Vdc, out=vdc_pk)
            np.maximum(vs_pk, vr.sum(-1), out=vs_pk)
            if rec and (s - s0) % rec_every == 0 and (s - s0)//rec_every < Nr:
                j = (s - s0)//rec_every
                for c, W in waves.items():
                    W["t"][j] = t
                    W["v"][j], W["isn"][j] = v[c], isn[c]
                    W["idio"][j] = np.where(cond[c], idc[c], v[c]*iRoff[c] - qM[c]*iTM[c])
                    W["ia"][j], W["Vdc"][j], W["iD"][j], W["iL"][j], W["uk"][j] = ia[c], Vdc[c], iD[c], iL[c], uk[c]
        # Heun step with the modes frozen
        k2 = rhs(t + dt, iL + dt*diL, iD + dt*diD, vcap + dt*dvcap, vx + dt*dvx, vC + dt*dvC, qM + dt*dqM)
        h = 0.5*dt
        iL = iL + h*(diL + k2[0])
        iD = iD + h*(diD + k2[1])
        vcap = vcap + h*(dvcap + k2[2])
        vx = vx + h*(dvx + k2[3])
        vC = vC + h*(dvC + k2[4])
        qM = qM + h*(dqM + k2[5])
        if verbose and s % (N//10) == 0:
            print(f"  t = {t*1e3:6.2f} ms, mode changes so far {nflip}")
    out = dict(
        Vpk=acc["Vpk"], Tpk=acc["tpk"], Vmean=acc["v1"]/Nm, Vrms=np.sqrt(acc["v2"]/Nm),
        P_Rb=acc["v2"]/Nm/Rb, P_Rsn=acc["isn2"]/Nm*Rsn, Isn_rms=np.sqrt(acc["isn2"]/Nm), Isn_pk=acc["Isn_pk"],
        Irrm=acc["Irrm"], Qrr=acc["Qrr"]*dt, IFav=acc["ifw"]/Nm, IFrms=np.sqrt(acc["if2"]/Nm), IFpk=acc["ifpk"],
        P_lk=acc["plk"]/Nm, P_rec=acc["erec"]*dt*f, Vdc=acc_vdc/Nm, Idc=acc_id/Nm, Iline_rms=np.sqrt(acc_iL2/Nm),
        Vstack_pk=vs_pk, Vdc_pk=vdc_pk, P_x=acc_px/Nm, n=n, dt=dt, mode_changes=nflip, wave=waves)
    return out


if __name__ == "__main__":
    import time
    # reference case of the note: 10 kV + 10 %, 50 Hz, uk = 7 % on 1.6 MVA, resistive load (100 A at 10 kV), n = 6
    n, u, U, Id, w = 6, 0.25, 10e3, 100.0, 2*PI*50.0
    Lc = 0.07*U**2/(w*1.6e6)
    Vhat = np.sqrt(2)*U
    mu = np.arccos(1 - 2*w*Lc*Id/Vhat)                      # overlap of the ideal bridge at the rated point
    a_ref = Vhat*np.sin(mu)/(2*Lc)                          # current slope at the current zero (recovery set given here)
    Rload = 3/PI*Vhat/Id - 3*w*Lc/PI
    tau_n, TM_n = lm_params(0.50e-3, 11.3, a_ref)
    tau_w, TM_w = lm_params(0.50e-3*(1 - u), 11.3*np.sqrt(1 - u), a_ref)
    t0 = time.time()
    r = simulate(1.1*U, 50.0, Lc, Rload, 3e-3, level_array(1, n, tau_w, tau_n), level_array(1, n, TM_w, TM_n),
                 level_array(1, n, 1e12, 2.7e6), 470e3, 150.0, 150e-9, VT0=0.85, rT=0.60e-3, verbose=True)
    print(f"elapsed {time.time() - t0:.1f} s, {r['mode_changes']} mode changes")
    print(f"Vdc {r['Vdc'][0]/1e3:.2f} kV, Idc {r['Idc'][0]:.1f} A, line {r['Iline_rms'][0, 0]:.1f} A rms")
    for j, name in ((0, "weak"), (1, "nominal")):
        print(f"{name:8s}: Irrm {r['Irrm'][0, 0, j]:.2f} A, Qrr {r['Qrr'][0, 0, j]*1e6:.1f} uC, "
              f"Vpk {r['Vpk'][0, 0, j]/1e3:.3f} kV, Vrms {r['Vrms'][0, 0, j]/1e3:.3f} kV, "
              f"P_Rsn {r['P_Rsn'][0, 0, j]:.2f} W, P_Rb {r['P_Rb'][0, 0, j]:.2f} W, "
              f"Isn {r['Isn_pk'][0, 0, j]:.2f} A pk / {r['Isn_rms'][0, 0, j]:.3f} A rms")
