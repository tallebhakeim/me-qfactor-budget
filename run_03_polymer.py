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
ETA_PVDF = (0.06, 0.02, 0.20)                # 1/Q_m du PVDF : (nom, min, max)
# Q_m(PVDF) pris entre 5 et 50 : HYPOTHÈSE d'intervalle large (polymère à bas Q ;
# Ohigashi 1976 = méthode de résonance et constantes complexes, valeur non relue).
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
    figure(out, z)
    return out


def figure(out, z):
    """fig_en_03.png : (a) alpha(f) digitalisée (Zeng 2015, fig. 5) contre les
    lorentziennes au Q mesuré et au Q nominal du budget ; (b) budget 1/Q."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    BLUE, RED, GREY, GREEN = "#4878a8", "#c1272d", "#666666", "#3a8f4a"
    f, al = z["f_kHz"], z["alpha"]
    i = int(np.argmax(al)); f0, ap = f[i], al[i]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.8, 3.5))
    a0.plot(f, al, ".", color=RED, ms=2.5, alpha=0.7,
            label="measured, digitized (Zeng 2015, fig. 5)")
    fl = np.linspace(60, 100, 600)
    lowf = al[(f > 65) & (f < 72)]
    base = float(np.median(lowf)) if lowf.size else float(al.min())
    for Q, c, ls, lab in ((out["Q_meas"][0], RED, "-", f"Lorentzian at measured Q = {out['Q_meas'][0]:.0f}"),
                          (out["Q"]["nom"], BLUE, "--", f"budget nominal Q = {out['Q']['nom']:.0f} (matrix share 0.37)"),
                          (out["Q_bounds"][1], GREEN, ":", f"budget, Reuss bound Q = {out['Q_bounds'][1]:.0f}")):
        a0.plot(fl, base + (ap - base) / np.sqrt(1 + (2 * Q * (fl - f0) / f0)**2), ls, color=c, lw=1.1, label=lab)
    a0.set_xlim(60, 100); a0.set_ylim(0, 95)
    a0.set_xlabel("frequency [kHz]"); a0.set_ylabel("α$_{ME}$ [mV cm$^{-1}$ Oe$^{-1}$]")
    a0.set_title("(a) 0-3 PVDF/PZT/Terfenol-D disk, 1000 Oe, 2 Oe drive", fontsize=9)
    a0.legend(fontsize=6.5, loc="upper left")
    labels = ["dielectric\n(tan δ k²)", "Terfenol-D\nparticle eddy", "matrix\n(Voigt share)", "matrix\n(share 0.37)", "matrix\n(Reuss share)"]
    vals = [out["inv_diel"], out["eta_td_eddy"] * 0.07, out["voigt"] * ETA_PVDF[0],
            out["f_m"] * ETA_PVDF[0], out["reuss"] * ETA_PVDF[0]]
    cols = [GREY, BLUE, GREEN, GREEN, GREEN]
    a1.bar(range(5), [max(v * 1e3, 1.2e-3) for v in vals], color=cols, width=0.7)
    a1.text(1, 2e-3, f"< {vals[1]*1e3:.0e}", ha="center", fontsize=7, color=BLUE)
    for k in (2, 3, 4):
        share = (out["voigt"], out["f_m"], out["reuss"])[k - 2]
        a1.plot([k, k], [share * ETA_PVDF[1] * 1e3, share * ETA_PVDF[2] * 1e3], "-", color="k", lw=0.8)
    a1.axhline(1e3 / out["Q_meas"][0], color=RED, lw=1.2, label=f"measured 1/Q (Q = {out['Q_meas'][0]:.0f}-{out['Q_meas'][1]:.0f})")
    a1.axhline(1e3 / out["Q_meas"][1], color=RED, lw=1.2)
    a1.set_yscale("log"); a1.set_ylim(1e-3, 300)
    a1.set_xticks(range(5)); a1.set_xticklabels(labels, fontsize=7)
    a1.set_ylabel("1/Q [10$^{-3}$]")
    a1.set_title("(b) channels: laminate channels empty, matrix loss decides", fontsize=9)
    a1.legend(handles=[Line2D([], [], color=RED, lw=1.2, label=f"measured 1/Q (Q = {out['Q_meas'][0]:.0f}-{out['Q_meas'][1]:.0f})"),
                       Line2D([], [], color="k", lw=0.8, label="Q$_m$(PVDF) 5-50")], fontsize=6.5, loc="upper left")
    fig.tight_layout(); fig.savefig(HERE / "fig_en_03.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
