# -*- coding: utf-8 -*-
"""
Extension DISQUE du budget de Q : mode radial de disques laminés
Terfenol-D / PIC181 (thèse G. Rizzo, C2N, 2020 — HT rapporteur).

Jeu de données : Table 2.3 de la thèse = paramètres de circuit identifiés
(Lm, Cm, Rm...) pour 6 échantillons x 6 biais, dont le FACTEUR DE QUALITÉ Q
et la fréquence de résonance série f_s = 1/(2π sqrt(Lm Cm)).
36 points de mesure indépendants (autre labo, autre géométrie, autre piézo).

Modèle :
 - mode radial d'un disque laminé (iso-déformation dans l'épaisseur) :
   u(r) = J1(kr), condition de bord libre  ka·J0(ka) = (1-ν̄)·J1(ka),
   raideur plane A = Σ E_i t_i/(1-ν_i²), f_s = k·v_p/2π (constante-E : la
   résonance identifiée par Lm-Cm est la résonance SÉRIE).
 - PIC181 (piézo dur) : v_p déduit du coefficient de fréquence datasheet
   Np = f·D = 2265 Hz·m et ν = 0.35 -> E = 80.6 GPa ; Qm = 2200,
   tanδ = 3e-3 : le PZT est quasi transparent, le Q du disque teste
   les canaux MAGNÉTIQUES presque seuls.
 - Terfenol : coefficients au biais INTERNE, auto-cohérents avec la
   démagnétisation dans le plan  H_int = H_app - N·M  (point fixe avec
   chi(H_int) du modèle énergie me_energy) ; N calculé par demag_lambda
   sur le pavé équivalent (D, D, t).
 - canaux : Foucault (plaque exacte, source b* = λ d33m T_plan),
   hystérésis de Rayleigh (ΔH piloté par la contrainte, auto-cohérent en
   amplitude), PZT mécanique + diélectrique (minuscules), colle négligée.

NB simplifications documentées : contrainte motrice = DÉVIATEUR en plan
(σ_r-σ_θ)/(1+ν), projeté sur l'axe de biais en RMS azimutale (une contrainte
équibiaxiale ne crée pas d'anisotropie en plan -> pas de dM) ; couplage
électrique non raidi (résonance série) ; pas de précontrainte de collage ;
PAS de canal « pertes excédentaires » (parois, Bertotti) -> le budget
sous-estime les pertes à bas biais où chi_int atteint 30-47.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))
try:
    from me import me_energy as en           # Terfenol énergie-moyennée
    HAVE_CORE = True
except ImportError:
    HAVE_CORE = False

import demag_lambda as dl
import qfactor_hysteresis as qh

OE = 1e3 / (4 * np.pi)
MU0 = 4e-7 * np.pi

# PIC181 (PI Ceramic, datasheet thèse Table 4.4) ; E dérivé de Np et nu
PIC = dict(rho=7850.0, nu=0.35, Np=2265.0, Qm=2200.0, tand=(3e-3, 1e-3, 5e-3))
# E du Terfenol POLARISÉ : l'effet Delta-E raidit le barreau de ~30 GPa
# (démagnétisé) vers 45-55 GPa sous biais (Engdahl) ; nominal 45 GPa.
TERF = dict(rho=9250.0, nu=0.30, E=45e9, sigma=1.67e6,
            c_rev=(0.15, 0.10, 0.20), eta_inf=(0.9, 0.7, 1.0),
            Ha0_oe=(18.0, 10.0, 40.0))

# échantillons (thèse Table 4.5) : (D [m], couches [(mat, t)], f_s Table 2.3)
SAMPLES = {
    "A (M-P, 16)":    dict(D=16e-3, tp=1e-3, tms=[1e-3],       f_meas=130.9e3),
    "B (M-P, 16)":    dict(D=16e-3, tp=2e-3, tms=[1e-3],       f_meas=132.5e3),
    "C (M-P-M, 16)":  dict(D=16e-3, tp=2e-3, tms=[1e-3, 1e-3], f_meas=130.2e3),
    "D (P-M-P, 16)":  dict(D=16e-3, tp=4e-3, tms=[1e-3],       f_meas=129.8e3),
    "E (M-P, 10)":    dict(D=10e-3, tp=1e-3, tms=[1e-3],       f_meas=180.4e3),
    "F (P-M-P, 10)":  dict(D=10e-3, tp=2e-3, tms=[1e-3],       f_meas=199.9e3),
}
# Q identifiés (Table 2.3), biais B_dc décroissants [0.1 .. 0.012 T]
BIAS_T = [0.1, 0.058, 0.036, 0.023, 0.016, 0.012]
Q_MEAS = {
    "A (M-P, 16)":   [117.0, 91.2, 94.3, 119.5, 127.0, 130.8],
    "B (M-P, 16)":   [139.4, 112.4, 92.5, 101.3, 111.8, 187.5],
    "C (M-P-M, 16)": [173.2, 172.4, 185.8, 211.5, 230.1, 239.3],
    "D (P-M-P, 16)": [429.5, 421.4, 422.8, 450.6, 481.4, 512.3],
    "E (M-P, 10)":   [139.9, 166.6, 186.2, 204.6, 304.7, 314.4],
    "F (P-M-P, 10)": [266.5, 287.5, 326.4, 363.5, 389.8, 394.6],
}


def pic_modulus():
    """E du PIC181 depuis Np (datasheet) : ka(nu)·v_p = 2π f a, f·D = Np."""
    nu = PIC["nu"]
    ka = brentq(lambda x: x * j0(x) - (1 - nu) * j1(x), 1.2, 3.0)
    v_p = 2 * np.pi * PIC["Np"] * 0.5 / ka          # f·D=Np -> ω a = ka v_p
    E = PIC["rho"] * (1 - nu**2) * v_p**2
    return E, ka


def radial_mode(sample, nr=400):
    """Mode radial fondamental du disque laminé : f_s, fractions d'énergie,
    profils de contrainte plane moyenne dans le Terfenol."""
    s = SAMPLES[sample]
    Ep, _ = pic_modulus()
    layers = [("pzt", Ep, PIC["nu"], PIC["rho"], s["tp"])] + \
             [("terf", TERF["E"], TERF["nu"], TERF["rho"], t)
              for t in s["tms"]]
    A = sum(E * t / (1 - nu**2) for _, E, nu, _, t in layers)
    rhoT = sum(rho * t for _, _, _, rho, t in layers)
    nub = sum(nu * E * t / (1 - nu**2) for _, E, nu, _, t in layers) / A
    ka = brentq(lambda x: x * j0(x) - (1 - nub) * j1(x), 1.2, 3.0)
    a = s["D"] / 2
    v_p = np.sqrt(A / rhoT)
    f = ka * v_p / (2 * np.pi * a)

    r = np.linspace(1e-6, a, nr)
    k = ka / a
    u = j1(k * r)
    Sr = k * (j0(k * r) - j1(k * r) / (k * r))
    St = j1(k * r) / r
    W = {}
    Tm_mean = None
    for nm, E, nu, _, t in layers:
        dens = E / (1 - nu**2) * (Sr**2 + St**2 + 2 * nu * Sr * St)
        w = float(np.trapz(dens * 2 * np.pi * r, r) * t)
        W[nm] = W.get(nm, 0.0) + w
        if nm == "terf" and Tm_mean is None:
            # contrainte MOTRICE = déviateur en plan (sigma_r - sigma_theta) :
            # une contrainte équibiaxiale ne cree pas d'anisotropie en plan,
            # donc pas de dM (nulle au centre du disque). Le facteur cos(2θ)
            # de la projection sur l'axe de biais est pris en RMS (1/racine2).
            Tm_mean = E / (1 + nu) * (Sr - St) / np.sqrt(2.0)
    Wtot = sum(W.values())
    return dict(f=f, ka=ka, r=r, Tm=Tm_mean, W=W, Wtot=Wtot, a=a,
                layers=layers, sample=s)


def bias_interne(H_app_oe, N):
    """Biais interne : H (1 + chi(H) N) = H_app. La susceptibilité chi(H)
    décroissante peut donner PLUSIEURS racines (bistabilité du point de
    polarisation) ; on prend la racine de PLUS HAUT champ, cohérente avec le
    protocole (aimants approchés depuis la saturation). Balayage en grille
    puis raffinement. Renvoie (H_int_oe, d33m, chi)."""
    Hs = np.linspace(1.0, H_app_oe, 240)
    chis = np.array([max(en.energy_coeffs("Terfenol-D", float(h), 0.0)["mu_r"]
                         - 1.0, 1e-3) for h in Hs])
    g = Hs * (1 + chis * N) - H_app_oe
    idx = np.where(np.diff(np.sign(g)) != 0)[0]
    if len(idx):
        i = idx[-1]                                  # racine de plus haut champ
        h1, h2 = Hs[i], Hs[i + 1]
        for _ in range(30):
            hm = 0.5 * (h1 + h2)
            chim = max(en.energy_coeffs("Terfenol-D", hm, 0.0)["mu_r"] - 1.0,
                       1e-3)
            if (hm * (1 + chim * N) - H_app_oe) * (g[i]) > 0:
                h1 = hm
            else:
                h2 = hm
        H = 0.5 * (h1 + h2)
    else:
        H = H_app_oe
    c = en.energy_coeffs("Terfenol-D", max(H, 1.0), 0.0)
    return H, c["d33m"], max(c["mu_r"] - 1.0, 1e-3)


def _pick(triple, corner, is_Q):
    if corner == "nom":
        return triple[0]
    if corner == "min_loss":
        return triple[2] if is_Q else triple[1]
    return triple[1] if is_Q else triple[2]


_N_CACHE = {}


def _N_of(D, t):
    key = (round(D, 6), round(t, 6))
    if key not in _N_CACHE:
        _N_CACHE[key] = dl.demag_N(D, D, t)     # pavé équivalent D x D x t
    return _N_CACHE[key]


def q_budget_disk(sample, B_dc_T, corner="nom", H_ac_oe=1.0, max_iter=60):
    """Q auto-cohérent du disque au biais donné. Échelle : le mode unitaire
    (u = J1) donne le profil de contrainte Tm ; la réponse réelle est mise à
    l'échelle T0 = E.Q.d33m.h_int (amplification modale), toutes les énergies
    suivant en (T0/max|Tm|)² -> le canal Foucault est indépendant de
    l'échelle, l'hystérésis de Rayleigh y est sensible (non-linéaire)."""
    md = radial_mode(sample)
    s, f = md["sample"], md["f"]
    w0 = 2 * np.pi * f
    r = md["r"]
    H_app = B_dc_T / MU0 / OE                       # Oe appliqués
    invQ_pzt = (1.0 / _pick((PIC["Qm"], 1500.0, 3000.0), corner, True)) \
        * md["W"]["pzt"] / md["Wtot"] \
        + _pick(PIC["tand"], corner, False) * 0.10 * md["W"]["pzt"] / md["Wtot"]
    # (0.10 = fraction électrique majorée à la résonance série ; minuscule)

    Tmax_u = float(np.max(np.abs(md["Tm"])))
    shape = np.abs(md["Tm"]) / Tmax_u
    lay = []
    for t in s["tms"]:
        N = _N_of(s["D"], t)
        H_int, d33m, chi = bias_interne(H_app, N)
        lam = (1.0 - N) / (1.0 + chi * N)
        p1 = _eddy(1.0, f, chi + 1.0, TERF["sigma"], d33m, lam, t)
        lay.append(dict(t=t, N=N, H_int=H_int, d33m=d33m, chi=chi,
                        lam=lam, p1=p1))
    Q = 200.0
    for it in range(max_iter):
        invQ = invQ_pzt
        for L in lay:
            h_int = H_ac_oe / (1.0 + L["chi"] * L["N"]) * OE
            T0 = TERF["E"] * Q * L["d33m"] * h_int  # échelle modale
            beta2 = (T0 / Tmax_u) ** 2
            Tprof = shape * T0
            P = L["p1"] * float(np.trapz(Tprof**2 * 2 * np.pi * r, r)) * L["t"]
            invQ += P / (w0 * 0.5 * md["Wtot"] * beta2)
            vol = 2 * np.pi * r * (r[1] - r[0]) * L["t"]
            invQ += qh.invQ_hyst_layer(
                Tprof, vol, md["Wtot"] * beta2, L["d33m"], L["chi"], L["N"],
                _pick(TERF["c_rev"], corner, True),
                _pick(TERF["eta_inf"], corner, False),
                _pick(TERF["Ha0_oe"], corner, True))
        Qn = 1.0 / invQ
        if abs(Qn - Q) < 1e-4 * Q:
            break
        Q = 0.5 * (Q + Qn)
    return dict(sample=sample, f=f, Q=Qn, iters=it + 1, H_app_oe=H_app,
                lay=lay)


def _eddy(T_amp, f, mu_r, sigma, d33m, lam, t, npts=200):
    """Puissance Foucault moyenne [W/m^3] pour une amplitude de contrainte
    T_amp (quadratique en T -> évaluée à 1 Pa puis mise à l'échelle)."""
    w0 = 2 * np.pi * f
    mu = mu_r * MU0
    b = lam * d33m * T_amp
    k = np.sqrt(1j * w0 * mu * sigma)
    A = (b / mu) / np.cosh(k * t / 2)
    y = np.linspace(-t / 2, t / 2, npts)
    J = A * k * np.sinh(k * y)
    return float(np.trapz(np.abs(J)**2, y) / (2 * sigma) / t)


def main():
    Ep, ka0 = pic_modulus()
    print(f"PIC181 : E = {Ep/1e9:.1f} GPa (déduit de Np = 2265), "
          f"ka(nu=0.35) = {ka0:.3f}")
    print(f"{'échantillon':16} {'f_s mod':>9} {'f_s mes':>9} {'écart':>7}")
    for sname, s in SAMPLES.items():
        md = radial_mode(sname)
        print(f"{sname:16} {md['f']/1e3:8.1f}k {s['f_meas']/1e3:8.1f}k "
              f"{(md['f']/s['f_meas']-1)*100:+6.1f}%")
    print()
    print(f"{'échantillon':16}" + "".join(f"  B={b:5.3f}T" for b in BIAS_T))
    for sname in SAMPLES:
        preds = [q_budget_disk(sname, b)["Q"] for b in BIAS_T]
        meas = Q_MEAS[sname]
        print(f"{sname:16}" + "".join(f" {p:8.0f}" for p in preds) + "  (préd)")
        print(f"{'':16}" + "".join(f" {m:8.0f}" for m in meas) + "  (mesuré)")


if __name__ == "__main__":
    main()
