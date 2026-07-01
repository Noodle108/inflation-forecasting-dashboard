"""Sims (2002) `gensys` linear rational-expectations solver.

Solves systems of the form

    g0 @ y_t = g1 @ y_{t-1} + c + psi @ z_t + pi @ eta_t

where `z_t` are exogenous i.i.d. innovations and `eta_t` are endogenous
expectational errors (eta_t = y_t - E_{t-1} y_t for the forward-looking variables).
Returns the reduced form

    y_t = G1 @ y_{t-1} + C + impact @ z_t

together with `eu = [existence, uniqueness]` (both 1 ⇒ a unique stable solution).

This is a faithful NumPy translation of Sims' `gensys.m`, using SciPy's ordered
generalized Schur (QZ) decomposition. It is validated in the test suite against the
closed-form solution of the small New Keynesian model.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import ordqz

_REALSMALL = 1e-7


def gensys(g0, g1, c, psi, pi, div: float = 1.0 + 1e-8):
    g0 = np.asarray(g0, dtype=complex)
    g1 = np.asarray(g1, dtype=complex)
    c = np.asarray(c, dtype=complex).reshape(-1)
    psi = np.asarray(psi, dtype=complex)
    pi = np.asarray(pi, dtype=complex)
    n = g0.shape[0]
    eu = [0, 0]

    def _sort_stable(alpha, beta):
        return np.abs(beta) <= div * np.abs(alpha)

    S, T, alpha, beta, Q, Z = ordqz(g0, g1, sort=_sort_stable, output="complex")
    # SciPy convention: g0 = Q S Z^H, g1 = Q T Z^H  ⇒  Q^H g0 Z = S, Q^H g1 Z = T
    Qh = Q.conj().T

    unstab = np.abs(beta) > div * np.abs(alpha)
    nunstab = int(np.sum(unstab))
    nstab = n - nunstab

    q1, q2 = Qh[:nstab, :], Qh[nstab:, :]
    neta = pi.shape[1]

    def _trunc_svd(M):
        u, d, vh = np.linalg.svd(M, full_matrices=False)
        keep = d > _REALSMALL
        return u[:, keep], d[keep], vh.conj().T[:, keep]

    # unstable block: expectational-error and shock loadings
    ueta, deta, veta = _trunc_svd(q2 @ pi)
    zwt = q2 @ psi

    # existence: the explosive block's shock loadings must be spanned by pi's
    if zwt.size == 0 or ueta.shape[1] == 0:
        exist = np.linalg.norm(zwt) < _REALSMALL * n
        if ueta.shape[1] > 0:
            exist = True
    else:
        exist = np.linalg.norm(zwt - ueta @ (ueta.conj().T @ zwt)) < _REALSMALL * n
    eu[0] = int(bool(exist))

    # uniqueness: stable block adds no further free expectational directions
    ueta1, deta1, veta1 = _trunc_svd(q1 @ pi)
    if veta1.shape[1] == 0:
        eu[1] = 1
    else:
        loose = veta1 - veta @ (veta.conj().T @ veta1)
        eu[1] = int(np.linalg.norm(loose) < _REALSMALL * n)

    # ---- construct the reduced form (Sims' determinate construction) ----
    if deta.size:
        M = ueta @ np.diag(1.0 / deta) @ veta.conj().T @ veta1 @ np.diag(deta1) @ ueta1.conj().T
    else:
        M = np.zeros((nunstab, nstab), dtype=complex)
    tmat = np.hstack([np.eye(nstab, dtype=complex), -M.conj().T])  # nstab x n

    G0 = np.vstack([tmat @ S,
                    np.hstack([np.zeros((nunstab, nstab)), np.eye(nunstab)])])
    G1m = np.vstack([tmat @ T, np.zeros((nunstab, n))])
    G0I = np.linalg.inv(G0)
    G1m = G0I @ G1m

    usix = slice(nstab, n)
    Suu, Tuu = S[usix, usix], T[usix, usix]
    if nunstab > 0:
        c_low = np.linalg.solve(Suu - Tuu, q2 @ c)
    else:
        c_low = np.zeros(0, dtype=complex)
    Cvec = G0I @ np.concatenate([tmat @ (Qh @ c), c_low])
    impact = G0I @ np.vstack([tmat @ (Qh @ psi),
                              np.zeros((nunstab, psi.shape[1]), dtype=complex)])

    # rotate back to the original variable space
    G1_out = np.real(Z @ G1m @ Z.conj().T)
    C_out = np.real(Z @ Cvec)
    impact_out = np.real(Z @ impact)
    return G1_out, C_out, impact_out, eu
