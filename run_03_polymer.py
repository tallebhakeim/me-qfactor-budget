# -*- coding: utf-8 -*-
"""
Composite particulaire 0-3 polymère (Zeng et al. 2015 : PVDF 0,30 / PZT 0,63 /
Terfenol-D 0,07 en volume, disque Ø 15 x 1 mm, mode radial à 87 kHz, biais
1000 Oe). Le budget des laminés y est presque vide : Foucault dans des
particules de 20 µm (δ ~ 0,6 mm) et canal diélectrique (k² ~ 0,5 %) sont
négligeables ; la perte est portée par la MATRICE viscoélastique, canal absent
des laminés : 1/Q ~ f_m · η_PVDF, avec f_m la part d'énergie de déformation
de la matrice (bornes Voigt/Reuss) et η_PVDF = 1/Q_m du polymère.
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MU0 = 4e-7 * np.pi
# données de l'article (Zeng 2015) et propriétés de phases (littérature)
F_R, RADIUS, THICK = 87.2e3, 7.5e-3, 1e-3
RHO_REL, NU = 5.64, 0.30                     # densité relative (Table 2), Poisson
EPS_R, TAND, D33 = 165.0, 0.045, 20e-12      # fig. 6-7 : eps_r, tan delta, d33 (pC/N)
PHI = dict(pvdf=0.30, pzt=0.63, td=0.07)
E_PH = dict(pvdf=2.5e9, pzt=60e9, td=30e9)   # modules de phase (Pa)
ETA_PVDF = (0.06, 0.04, 0.10)                # 1/Q_m du PVDF : (nom, min, max)
A_TD, SIG_TD, MUR_TD = 10e-6, 1.67e6, 5.0    # rayon des particules, conductivité


def main(verbose=True):
    rho = RHO_REL * 1e3
    v = F_R * 2 * np.pi * RADIUS / (1.867 + 0.6054 * NU)         # éq. (2) de l'article
    E_c = rho * v**2 * (1 - NU**2)                                # module du composite
    eps = EPS_R * 8.854e-12
    k2 = D33**2 * E_c / eps
    inv_diel = TAND * k2
    # Foucault dans une sphère conductrice (petite devant delta) : P/V = sigma w² a² B²/15
    # -> facteur de perte de la phase TD pour un flux motionnel b* = d33m T : borne haute
    delta = np.sqrt(2 / (2 * np.pi * F_R * MUR_TD * MU0 * SIG_TD))
    d33m, T_over_E = 5e-9, 1.0                                     # eta = P/(w U), U = T²/(2E)
    w = 2 * np.pi * F_R
    eta_td_eddy = (SIG_TD * w**2 * A_TD**2 * (d33m * 1.0)**2 / 15) / (w * 0.5 / E_PH["td"])
    # parts d'énergie de la matrice : bornes Voigt (parallèle) et Reuss (série)
    voigt = PHI["pvdf"] * E_PH["pvdf"] / sum(PHI[p] * E_PH[p] for p in PHI)
    reuss = (PHI["pvdf"] / E_PH["pvdf"]) / sum(PHI[p] / E_PH[p] for p in PHI)
    E_voigt = sum(PHI[p] * E_PH[p] for p in PHI)
    E_reuss = 1 / sum(PHI[p] / E_PH[p] for p in PHI)
    # interpolation de la part matrice au module mesuré (log-linéaire entre les bornes)
    t = (np.log(E_c) - np.log(E_reuss)) / (np.log(E_voigt) - np.log(E_reuss))
    f_m = reuss + (voigt - reuss) * np.clip(t, 0, 1)
    Q = {c: 1 / (f_m * e + inv_diel) for c, e in zip(("nom", "min_loss", "max_loss"),
                                                      (ETA_PVDF[0], ETA_PVDF[1], ETA_PVDF[2]))}
    Q_bounds = (1 / (voigt * ETA_PVDF[1] + inv_diel), 1 / (reuss * ETA_PVDF[2] + inv_diel))
    z = np.load(HERE / "zeng2015_curve.npz")
    def q3(x, y):
        i = int(np.argmax(y)); h = y[i] / np.sqrt(2)
        lo = x[:i][y[:i] < h].max(); hi = x[i:][y[i:] < h].min(); return x[i] / (hi - lo)
    Qm = (q3(z["f_kHz"], z["alpha"]), q3(z["f_inset_kHz"], z["alpha_inset_rel"]))
    out = dict(E_c=E_c, v=v, k2=k2, inv_diel=inv_diel, delta_td=delta,
               eta_td_eddy=eta_td_eddy, f_m=f_m, voigt=voigt, reuss=reuss,
               E_voigt=E_voigt, E_reuss=E_reuss, Q=Q, Q_bounds=Q_bounds, Q_meas=Qm)
    if verbose:
        print(f"composite : v = {v:.0f} m/s, E_c = {E_c/1e9:.1f} GPa (Voigt {E_voigt/1e9:.0f}, Reuss {E_reuss/1e9:.1f})")
        print(f"k² = {k2:.4f} -> canal diélectrique 1/Q = {inv_diel:.1e}")
        print(f"Terfenol-D en particules : δ = {delta*1e3:.2f} mm >> a = {A_TD*1e6:.0f} µm ; η_Foucault(TD) < {eta_td_eddy:.1e}")
        print(f"part d'énergie de la matrice : Reuss {reuss:.2f}, Voigt {voigt:.3f}, au module mesuré {f_m:.2f}")
        print(f"Q prédit (matrice η = {ETA_PVDF[0]}) = {Q['nom']:.0f} ; intervalle [{Q['max_loss']:.0f} ; {Q['min_loss']:.0f}] ; bornes Voigt/Reuss extrêmes [{Q_bounds[1]:.0f} ; {Q_bounds[0]:.0f}]")
        print(f"Q mesuré (-3 dB, fig. 5) : {Qm[0]:.0f} (principal) / {Qm[1]:.0f} (encart)")
    return out


if __name__ == "__main__":
    main()
