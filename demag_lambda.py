# -*- coding: utf-8 -*-
"""
Chantier 2 de l'article Q : CALCULER lambda_flux au lieu de l'encadrer à dire
d'expert. lambda_flux relie le flux motionnel réel au flux "circuit fermé" :
  b_reel = lambda * d33m * T.

Physique : la couche magnétostrictive (pavé Lm x w x t aimanté selon Lm) est
en CIRCUIT MAGNÉTIQUE OUVERT ; l'aimantation motionnelle M* = d.T/mu0 créée
par la contrainte dynamique subit le champ démagnétisant H_in = -N.M_tot :
  M_tot = M* + chi.H_in  =>  M_tot = M*/(1 + chi.N)
  b = mu0 (H_in + M_tot) = mu0 (1-N) M_tot  =>  lambda = (1-N)/(1+chi.N)
avec chi = mu_r - 1 (susceptibilité différentielle au biais) et N le facteur
démagnétisant MAGNÉTOMÉTRIQUE (moyenne volumique) du pavé selon Lm.

N est calculé EXACTEMENT (aimantation uniforme) : champ des deux faces
chargées (+-M) par la formule fermée du rectangle chargé, moyenné sur le
volume par quadrature de Gauss. Auto-tests : cube -> N = 1/3 exactement,
et N_a + N_b + N_c = 1 pour tout pavé.
"""
import numpy as np


def _hx_rect(x, y, z, b, c):
    """H_x au point (x,y,z) d'une plaque rectangulaire uniformément chargée
    (densité de charge magnétique sigma_m = 1 [A/m]) occupant le plan x=0,
    |y|<b/2, |z|<c/2. Formule fermée classique (somme d'arctangentes)."""
    hx = 0.0
    for sy, yy in ((1, b / 2 - y), (-1, -b / 2 - y)):
        for sz, zz in ((1, c / 2 - z), (-1, -c / 2 - z)):
            r = np.sqrt(x * x + yy * yy + zz * zz)
            hx += sy * sz * np.arctan(yy * zz / (x * r))   # impair en x
    return hx / (4 * np.pi)


def demag_N(a, b, c, ng=24):
    """Facteur démagnétisant magnétométrique N_a d'un pavé a x b x c aimanté
    uniformément selon a : N = -<H_x>/M, M = 1. Champ = 2 faces chargées
    (+1 en x=a/2, -1 en x=-a/2), moyenne par quadrature de Gauss-Legendre."""
    xg, wx = np.polynomial.legendre.leggauss(ng)
    X = a / 2 * xg
    Y = b / 2 * xg
    Z = c / 2 * xg
    W = wx / 2                       # poids normalisés (somme = 1)
    N = 0.0
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            hz_line = 0.0
            for k, z in enumerate(Z):
                hx = _hx_rect(x - a / 2, y, z, b, c) \
                    - _hx_rect(x + a / 2, y, z, b, c)
                hz_line += W[k] * hx
            N += -W[i] * W[j] * hz_line
    return float(N)


def lam_flux(a=14e-3, b=10e-3, c=1e-3, chi=7.4, ng=24):
    """lambda_flux = (1-N)/(1+chi.N) pour le pavé magnétostrictif."""
    N = demag_N(a, b, c, ng)
    return (1.0 - N) / (1.0 + chi * N), N


def lam_flux_interval(a=14e-3, b=10e-3, c=1e-3, chis=(7.4, 4.0, 9.0)):
    """(nominal, borne_inf, borne_sup) : chi = mu_r - 1 varie dans son
    intervalle DEAM ; lambda décroît avec chi -> bornes aux extrêmes."""
    lam_nom, N = lam_flux(a, b, c, chis[0])
    lam_hi, _ = lam_flux(a, b, c, chis[1])       # chi min -> lambda max
    lam_lo, _ = lam_flux(a, b, c, chis[2])       # chi max -> lambda min
    return (lam_nom, lam_lo, lam_hi), N


if __name__ == "__main__":
    # auto-tests
    Nc = demag_N(1, 1, 1)
    print(f"cube : N = {Nc:.5f} (attendu 1/3)")
    Ns = [demag_N(*dims) for dims in ((14, 10, 1), (10, 14, 1), (1, 10, 14))]
    print(f"pavé 14x10x1 : N_a = {Ns[0]:.4f}, somme 3 axes = {sum(Ns):.5f}")
    (lam, lo, hi), N = lam_flux_interval()
    print(f"Terfenol 14x10x1, chi=7,4 : N = {N:.4f} -> "
          f"lambda = {lam:.3f} [{lo:.3f} ; {hi:.3f}]")
