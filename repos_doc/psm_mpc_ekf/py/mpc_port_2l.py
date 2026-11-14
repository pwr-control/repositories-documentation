"""Python copies of the MATLAB Function mp control/mpc_pu_d of mpc_psm.slx (two-level bridge).

MPCNew: line-by-line copy of the fixed function (fix/mpc.m, the new chart script), checked against Octave by
        check_port.py.
MPCOld: copy of the function of the model before the fix (7 vectors, zero vector = all upper switches on,
        only the neighbour vectors and the zero vector allowed after an active vector, voltage in pu of udc,
        no delay compensation, k_sw*|u|^2).
States: [leg a, leg b, leg c], 1 = upper switch on, 0 = lower switch on.
"""
import numpy as np

STATES = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]])
INV_SQRT3 = 0.577350269189626


class MPCNew:
    def __init__(self):
        self.u_last = np.zeros(2)
        self.s_last = np.array([0, 0, 0])

    def step(self, vdc, ialpha_ref, ibeta_ref, ialpha, ibeta, omega_hat, psi_alpha_hat, psi_beta_hat, u_scale,
             L_norm, omega_bez, R_norm, k_kalman, k_sw, ts_inv):
        """Returns (ualpha, ubeta, gates[6], best_idx)."""
        ts_pu = ts_inv * omega_bez
        g = ts_pu / L_norm
        ealpha = -omega_hat * psi_beta_hat
        ebeta = omega_hat * psi_alpha_hat
        ialpha0 = ialpha + g * (self.u_last[0] - R_norm * ialpha - ealpha)
        ibeta0 = ibeta + g * (self.u_last[1] - R_norm * ibeta - ebeta)
        minJ = 1e18
        best = 0
        ua_best = ub_best = 0.0
        for n in range(8):
            sa, sb, sc = STATES[n]
            ua = u_scale * vdc * (sa - 0.5 * sb - 0.5 * sc) * 2 / 3
            ub = u_scale * vdc * (sb - sc) * INV_SQRT3
            ia_p = ialpha0 + g * (ua - R_norm * ialpha0 - ealpha)
            ib_p = ibeta0 + g * (ub - R_norm * ibeta0 - ebeta)
            nsw = abs(sa - self.s_last[0]) + abs(sb - self.s_last[1]) + abs(sc - self.s_last[2])
            J = k_kalman * ((ialpha_ref - ia_p) ** 2 + (ibeta_ref - ib_p) ** 2) + k_sw * nsw
            if J < minJ:
                minJ = J
                best = n
                ua_best, ub_best = ua, ub
        ualpha, ubeta = self.u_last[0], self.u_last[1]
        self.u_last = np.array([ua_best, ub_best])
        self.s_last = STATES[best].copy()
        ga, gb, gc = STATES[best]
        gates = 15.0 * np.array([ga, 1 - ga, gb, 1 - gb, gc, 1 - gc])
        return ualpha, ubeta, gates, best


# vectors of the old function: index 0 = zero vector (all upper switches on), 1..6 = active vectors
_OLD_UA = np.array([0.0, 2 / 3, 1 / 3, -1 / 3, -2 / 3, -1 / 3, 1 / 3])
_OLD_UB = np.array([0.0, 0.0, 2 / 3 * np.sqrt(3) / 2, 2 / 3 * np.sqrt(3) / 2, 0.0, -2 / 3 * np.sqrt(3) / 2,
                    -2 / 3 * np.sqrt(3) / 2])
# allowed vectors after each vector, in the order of the min() of the old code (first index wins a tie)
_OLD_ALLOWED = {0: [0, 1, 2, 3, 4, 5, 6], 1: [0, 1, 2, 6], 2: [0, 1, 2, 3], 3: [0, 2, 3, 4], 4: [0, 3, 4, 5],
                5: [0, 4, 5, 6], 6: [0, 5, 6, 1]}
# bridge state of each old vector (gate decoding of the old code)
OLD_TO_STATE = [7, 1, 2, 3, 4, 5, 6]        # u0 -> [1 1 1], u1 -> [1 0 0], ..., u6 -> [1 0 1]


class MPCOld:
    def __init__(self):
        self.prev = 0                       # persistent ualpha_z, ubeta_z (index of the old vector)

    def step(self, vdc, ialpha_ref, ibeta_ref, ialpha, ibeta, omega_hat, psi_alpha_hat, psi_beta_hat, Lbeta_norm,
             Lalpha_norm, omega_bez, Rs_norm, k_kalman, k_sw, ts_inv):
        """Returns (ualpha, ubeta, state index of STATES). ualpha, ubeta in pu of udc (2/3 vdc for u1)."""
        ts = ts_inv * omega_bez
        ua = _OLD_UA * vdc
        ub = _OLD_UB * vdc
        ia_p = (1 - Rs_norm / Lalpha_norm * ts) * ialpha + omega_hat / Lalpha_norm * psi_beta_hat * ts + ua * ts / Lalpha_norm
        ib_p = (1 - Rs_norm / Lbeta_norm * ts) * ibeta - omega_hat / Lbeta_norm * psi_alpha_hat * ts + ub * ts / Lbeta_norm
        J = k_kalman * ((ialpha_ref - ia_p) ** 2 + (ibeta_ref - ib_p) ** 2) + k_sw * (ua ** 2 + ub ** 2)
        allowed = _OLD_ALLOWED[self.prev]
        n = allowed[int(np.argmin([J[a] for a in allowed]))]
        self.prev = n
        return ua[n], ub[n], OLD_TO_STATE[n]
