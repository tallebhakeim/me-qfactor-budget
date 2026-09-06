# -*- coding: utf-8 -*-
"""
PoC v3 : PRÉDIRE le facteur de qualité Q d'un laminé ME par BILAN D'ÉNERGIE,
sans mesure sur la structure, et le VALIDER sur les 4 échantillons de la
thèse Kévin Malleron avec UNE SEULE table de matériaux (aucun recalage).

Mesures de référence (thèse, tableau 7.3 p. 126) :
  PZT/Met        : H=38 Oe,  a_stat=1,186, f_r=68,4 kHz, a_res=6,86
  PZT/Met/Terf   : H=682 Oe, a_stat=1,366, f_r=69,4 kHz, a_res=24,3
  PZT/Terf       : H=525 Oe, a_stat=1,796, f_r=70,4 kHz, a_res=19,8
  PZT/Terf/Met   : H=525 Oe, a_stat=1,91,  f_r=70,4 kHz, a_res=18,7
(Le Tab. 2 de l'article Microelectronics J. 2019 publié porte ces valeurs
 dynamiques x10 ; la thèse fait foi, vérifié le 06/09/2026.)

Principe : 1/Q_total = somme des 1/Q_i, 1/Q_i = P_i/(omega.U).
Canaux (v3 — TOUS calculés ou bornés par des propriétés MATÉRIAU) :
  1. mécanique PZT     : Q_m PZT-5H (fiche) x fraction d'énergie
  2. diélectrique PZT  : tan delta_eps x fraction électrique
  3. hystérésis magnétomécanique : loi de RAYLEIGH calculée (aire de boucle
     mineure pilotée par la contrainte, cf. qfactor_hysteresis) — canal NON
     LINÉAIRE -> résolution auto-cohérente Q(H_ac) ; couches "Qm" (Metglas) :
     intervalle Q_m x fraction.
  4. Foucault          : diffusion 1D exacte dans l'épaisseur, flux motionnel
     b* = lambda.d33m.T avec lambda = (1-N)/(1+chi.N) CALCULÉ (demag_lambda)
  5. colle époxy       : shear-lag aux discontinuités x = Lm (négligeable)
L'ancrage (support) n'est pas modélisé et sort en résidu.

Modèle structurel : barre 1D bi-segment (pile 0..Lm, PZT seul Lm..Lp), FEM P1
libre-libre, forçage d33m.H, raideur piézo circuit ouvert (rang-1 exact).
"""
import numpy as np

import demag_lambda as dl
import qfactor_hysteresis as qh

EPS0 = 8.8541878128e-12
OE = 1e3 / (4 * np.pi)          # 1 Oe en A/m = 79,577

# ---------------------------------------------------------------- géométrie
GEO = dict(Lp=20e-3, Lm=14e-3, w=10e-3, tp=1e-3)

# ------------------------------------------------- matériaux (cf. core/me)
# Triplets = (nominal, borne_inf, borne_sup) : dispersion documentée du
# paramètre matériau, PAS un réglage. PZT-5H : Berlincourt ; E_INSITU : la
# mesure "PZT seul" de la thèse (f_s = 67,5 kHz, Tab. 3) donne
# E = rho.(2.L.f_s)^2 = 54,7 GPa (caractérisation matériau, une fois).
# Terfenol : Engdahl + modèle énergie (Talleb&Ren 2022) au biais 525 Oe /
# -24 MPa -> d33m=18,7 nm/A, mu_r=8,4. Rayleigh (c_rev, eta_inf, Ha0) : forme
# q_ac_factor de core/me/me_deam (Aubert PRApplied 2018), à resserrer par une
# calibration VSM matériau. Metglas 2605 : E=110 GPa (thèse p. 119), rho=7180,
# resistivité 1,3 uOhm.m ; à 38 Oe il travaille (chi énorme, pertes
# magnétoélastiques Delta-E -> Qm bas, intervalle large) ; à >=525 Oe SATURÉ.
PZT = dict(E=60e9, rho=7500.0, d31=-274e-12, epsT33=3400 * EPS0,
           Qm=(65.0, 40.0, 80.0), tand_eps=(0.020, 0.015, 0.025))
E_PZT_DATASHEET, E_PZT_INSITU = 60e9, 54.7e9

MAGS = {
    "Terfenol-D@bias": dict(E=30e9, rho=9250.0, sigma=1.67e6,
                            chi=(7.4, 4.0, 9.0), d33m=18.7e-9,
                            hyst="rayleigh",
                            c_rev=(0.15, 0.10, 0.20),
                            eta_inf=(0.9, 0.7, 1.0),
                            Ha0_oe=(18.0, 10.0, 40.0)),
    "Metglas@travail": dict(E=110e9, rho=7180.0, sigma=7.7e5,
                            chi=(1000.0, 100.0, 5000.0), d33m=2.0e-8,
                            hyst="Qm", Qm=(15.0, 5.0, 100.0)),
    "Metglas@sature":  dict(E=110e9, rho=7180.0, sigma=7.7e5,
                            chi=(50.0, 10.0, 100.0), d33m=1.0e-9,
                            hyst="Qm", Qm=(200.0, 50.0, 1000.0)),
}
GLUE = dict(Gg=(1.3e9, 0.8e9, 2.0e9), tg=(20e-6, 10e-6, 50e-6),
            tand_g=(0.05, 0.02, 0.10))

SAMPLES = {
    "PZT/Terf":     dict(mags=[("Terfenol-D@bias", 1e-3)],
                         meas=dict(a_stat=1.796, a_res=19.8, f_r=70.4e3)),
    "PZT/Met":      dict(mags=[("Metglas@travail", 35e-6)],
                         meas=dict(a_stat=1.186, a_res=6.86, f_r=68.4e3)),
    "PZT/Met/Terf": dict(mags=[("Metglas@sature", 35e-6),
                               ("Terfenol-D@bias", 1e-3)],
                         meas=dict(a_stat=1.366, a_res=24.3, f_r=69.4e3)),
    "PZT/Terf/Met": dict(mags=[("Terfenol-D@bias", 1e-3),
                               ("Metglas@sature", 35e-6)],
                         meas=dict(a_stat=1.91, a_res=18.7, f_r=70.4e3)),
}
REF = "PZT/Terf"


def set_pzt_insitu(on=True):
    """Bascule E_p entre la fiche (60 GPa) et la valeur in situ (54,7 GPa,
    déduite du f_s mesuré du PZT seul) — caractérisation matériau, une fois."""
    PZT["E"] = E_PZT_INSITU if on else E_PZT_DATASHEET


def _pick(triple, corner, is_Q):
    """Valeur d'un triplet (nom, inf, sup) selon le coin de pertes demandé.
    is_Q : le paramètre agit comme un Q (pertes min <-> valeur sup)."""
    if corner == "nom":
        return triple[0]
    if corner == "min_loss":
        return triple[2] if is_Q else triple[1]
    return triple[1] if is_Q else triple[2]


def epsS():
    """Permittivité bloquée effective 31 : epsT33 (1 - k31^2)."""
    k31sq = PZT["d31"]**2 * PZT["E"] / PZT["epsT33"]
    return PZT["epsT33"] * (1.0 - k31sq)


# ------------------------------------------ lambda_flux et N par (t, chi)
_LAM_CACHE = {}


def lam_N(t, chi):
    """(lambda, N) démagnétisant CALCULÉS pour une couche Lm x w x t."""
    key = (round(t, 9), round(chi, 4))
    if key not in _LAM_CACHE:
        lam, N = dl.lam_flux(GEO["Lm"], GEO["w"], t, chi)
        _LAM_CACHE[key] = (lam, N)
    return _LAM_CACHE[key]


# ============================================================ FEM 1D barre
def assemble(sample=REF, nx=200, eta_p=0.0, eta_mag=0.0, stiffen=True,
             geo=GEO):
    """K, M, F (forçage magnétostrictif pour H = 1 A/m), maillage."""
    Lp, Lm, w, tp = geo["Lp"], geo["Lm"], geo["w"], geo["tp"]
    mags = [(MAGS[nm], t) for nm, t in SAMPLES[sample]["mags"]]
    x = np.linspace(0.0, Lp, nx + 1)
    h = np.diff(x)
    xc = 0.5 * (x[:-1] + x[1:])
    in1 = xc < Lm

    Ep = PZT["E"] * (1 + 1j * eta_p)
    EA1 = Ep * tp * w + sum(m["E"] * (1 + 1j * eta_mag) * t * w
                            for m, t in mags)
    rA1 = PZT["rho"] * tp * w + sum(m["rho"] * t * w for m, t in mags)
    EA = np.where(in1, EA1, Ep * tp * w)
    rA = np.where(in1, rA1, PZT["rho"] * tp * w)

    n = nx + 1
    K = np.zeros((n, n), complex)
    M = np.zeros((n, n))
    F = np.zeros(n, complex)
    sig_star = sum(m["E"] * m["d33m"] * t * w for m, t in mags)   # H = 1 A/m
    for e in range(nx):
        i, j = e, e + 1
        ke = EA[e] / h[e]
        K[i, i] += ke; K[j, j] += ke; K[i, j] -= ke; K[j, i] -= ke
        me = rA[e] * h[e] / 6.0
        M[i, i] += 2 * me; M[j, j] += 2 * me; M[i, j] += me; M[j, i] += me
        if in1[e]:
            F[i] -= sig_star; F[j] += sig_star

    if stiffen:
        # rang-1 circuit ouvert : énergie epsS'.E3^2.Vp = kap.(u_L-u_0)^2,
        # ressort SANS perte (électrode équipotentielle, charge totale nulle)
        kap = (PZT["E"] * PZT["d31"])**2 * tp * w / (epsS() * Lp)
        K[0, 0] += kap; K[-1, -1] += kap; K[0, -1] -= kap; K[-1, 0] -= kap
    return dict(K=K, M=M, F=F, x=x, h=h, in1=in1, nx=nx, mags=mags,
                sample=sample, geo=geo)


def solve_harm(asm, f, H_oe=1.0):
    """Réponse harmonique à f ; renvoie déformations, alpha, énergies W_i.
    Les énergies utilisent les modules RÉELS (les eta de l'assemblage ne
    servent qu'à fixer l'amplitude de la réponse)."""
    geo = asm["geo"]
    w0 = 2 * np.pi * f
    H = H_oe * OE
    u = np.linalg.solve(asm["K"] - w0**2 * asm["M"], asm["F"] * H)
    S = np.diff(u) / asm["h"]
    Tp = PZT["E"] * S
    Tm = [np.where(asm["in1"], m["E"] * (S - m["d33m"] * H), 0.0)
          for m, t in asm["mags"]]

    S_moy = np.sum(S.real * asm["h"]) / geo["Lp"] + \
        1j * np.sum(S.imag * asm["h"]) / geo["Lp"]
    E3 = -PZT["d31"] * PZT["E"] * S_moy / epsS()
    V = E3 * geo["tp"]
    t_tot = geo["tp"] + sum(t for m, t in asm["mags"])
    alpha = abs(V) / (t_tot * 100 * H_oe)                     # V/(cm.Oe)

    dVp = geo["tp"] * geo["w"] * asm["h"]
    Wp = float(np.sum(PZT["E"] * np.abs(S)**2 * dVp))
    Wm = [float(np.sum(m["E"] * np.abs(S - m["d33m"] * H)**2
                       * np.where(asm["in1"], t * geo["w"] * asm["h"], 0.0)))
          for m, t in asm["mags"]]
    We = float(epsS() * abs(E3)**2 * geo["tp"] * geo["w"] * geo["Lp"])
    return dict(u=u, S=S, Tp=Tp, Tm=Tm, E3=E3, alpha=alpha,
                Wp=Wp, Wm=Wm, We=We, Wtot=Wp + sum(Wm) + We, f=f)


def find_resonance(asm, f0=40e3, f1=110e3, n=400, refine=3):
    for _ in range(refine):
        fs = np.linspace(f0, f1, n)
        a = np.array([solve_harm(asm, f)["alpha"] for f in fs])
        i = int(np.argmax(a))
        f0, f1 = fs[max(i - 1, 0)], fs[min(i + 1, n - 1)]
    return fs[i], a[i]


def sweep(asm, fs):
    return np.array([solve_harm(asm, f)["alpha"] for f in fs])


def q_from_bandwidth(asm, fr):
    """Q extrait de la largeur à -3 dB du pic de |alpha| (référence)."""
    fs = np.linspace(0.90 * fr, 1.10 * fr, 4001)
    a = sweep(asm, fs)
    i = int(np.argmax(a))
    half = a[i] / np.sqrt(2)
    lo = np.where(a[:i] < half)[0]
    hi = np.where(a[i:] < half)[0]
    f_lo = np.interp(half, [a[lo[-1]], a[lo[-1] + 1]], [fs[lo[-1]], fs[lo[-1] + 1]])
    k = i + hi[0]
    f_hi = np.interp(half, [a[k], a[k - 1]], [fs[k], fs[k - 1]])
    return fs[i] / (f_hi - f_lo)


# ================================================== canal 4 : Foucault exact
def eddy_power_slab(T_amp, f, mat, lam=1.0, t=1e-3, npts=400, mu_r=None):
    """Puissance Foucault moyenne (W/m^3, moyennée sur l'épaisseur) — plaque
    conductrice, diffusion 1D exacte, source b* = lam.d33m.T (cf. v2)."""
    w0 = 2 * np.pi * f
    if mu_r is None:
        mu_r = mat["chi"][0] + 1.0
    mu = mu_r * 4e-7 * np.pi
    sig = mat["sigma"]
    bstar = lam * mat["d33m"] * T_amp
    k = np.sqrt(1j * w0 * mu * sig)
    A = (bstar / mu) / np.cosh(k * t / 2)
    y = np.linspace(-t / 2, t / 2, npts)
    J = A * k * np.sinh(k * y)
    return float(np.trapz(np.abs(J)**2, y) / (2 * sig) / t)


def q_eddy(asm, res, corner="nom"):
    """1/Q_Foucault, lambda = (1-N)/(1+chi.N) CALCULÉ par couche au chi du
    coin demandé (chi agit comme un Q : chi grand -> lambda petit -> pertes
    min ; cohérent avec le canal hystérésis)."""
    geo = asm["geo"]
    w0 = 2 * np.pi * res["f"]
    invQ = 0.0
    for il, (m, t) in enumerate(asm["mags"]):
        chi = _pick(m["chi"], corner, True)
        lam, _ = lam_N(t, chi)
        dV = t * geo["w"] * asm["h"]
        P = sum(eddy_power_slab(abs(res["Tm"][il][e]), res["f"], m, lam, t,
                                mu_r=chi + 1.0)
                * dV[e] for e in range(asm["nx"]) if asm["in1"][e])
        invQ += P / (w0 * 0.5 * res["Wtot"])
    return invQ


# ================================================ canal 5 : colle (shear-lag)
def q_glue(asm, res, Gg, tg, tand_g):
    """Shear-lag aux joints de x = Lm (cf. v2). Canal négligeable."""
    geo = asm["geo"]
    e_end = np.where(asm["in1"])[0][-1]
    invQ = 0.0
    for j in range(len(asm["mags"])):
        dP = sum(abs(res["Tm"][il][e_end]) * t
                 for il, (m, t) in enumerate(asm["mags"]) if il >= j) * geo["w"]
        Es = sum(m["E"] * t for il, (m, t) in enumerate(asm["mags"]) if il >= j)
        beta = np.sqrt((Gg / tg) * (1.0 / (PZT["E"] * geo["tp"]) + 1.0 / Es))
        tau0 = beta * dP / geo["w"]
        Wg = tau0**2 / Gg * tg * geo["w"] / (2 * beta)
        invQ += tand_g * Wg / res["Wtot"]
    return invQ


# ============================== canal 3 : magnétomécanique (Rayleigh ou Qm)
def q_mag(asm, res, corner="nom"):
    """1/Q magnétomécanique : couches 'rayleigh' -> aire de boucle mineure
    calculée aux amplitudes de contrainte de res (NON LINÉAIRE) ; couches
    'Qm' -> intervalle littérature x fraction d'énergie."""
    geo = asm["geo"]
    invQ = 0.0
    for il, (m, t) in enumerate(asm["mags"]):
        if m["hyst"] == "rayleigh":
            chi = _pick(m["chi"], corner, True)
            _, N = lam_N(t, chi)
            iin = asm["in1"]
            T = np.abs(res["Tm"][il][iin])
            vol = (t * geo["w"] * asm["h"])[iin]
            invQ += qh.invQ_hyst_layer(
                T, vol, res["Wtot"], m["d33m"], chi, N,
                _pick(m["c_rev"], corner, True),        # c_rev grand = pertes min
                _pick(m["eta_inf"], corner, False),     # eta_inf grand = pertes max
                _pick(m["Ha0_oe"], corner, True))       # Ha0 grand = pertes min
        else:
            invQ += (1.0 / _pick(m["Qm"], corner, True)) \
                * res["Wm"][il] / res["Wtot"]
    return invQ


# ================================= budget auto-cohérent (canal non linéaire)
def q_budget(sample=REF, corner="nom", nx=200, H_ac_oe=1.0, tol=1e-4,
             max_iter=40):
    """Budget 1/Q par canal, résolu en point fixe sur Q : les amplitudes de
    la réponse (donc le canal Rayleigh) dépendent du Q total."""
    Q = 25.0
    fr_lo, fr_hi = 40e3, 110e3
    for it in range(max_iter):
        eta = 1.0 / Q
        asm = assemble(sample, nx, eta_p=eta, eta_mag=eta)
        fr, _ = find_resonance(asm, fr_lo, fr_hi,
                               n=400 if it == 0 else 80,
                               refine=3 if it == 0 else 2)
        fr_lo, fr_hi = 0.97 * fr, 1.03 * fr
        res = solve_harm(asm, fr, H_oe=H_ac_oe)
        inv = dict(
            pzt_meca=(1.0 / _pick(PZT["Qm"], corner, True))
            * res["Wp"] / res["Wtot"],
            mag_hyst=q_mag(asm, res, corner),
            pzt_diel=_pick(PZT["tand_eps"], corner, False)
            * res["We"] / res["Wtot"],
            foucault=q_eddy(asm, res, corner),
            colle=q_glue(asm, res, _pick(GLUE["Gg"], corner, True),
                         _pick(GLUE["tg"], corner, False),
                         _pick(GLUE["tand_g"], corner, False)),
        )
        inv["total"] = sum(inv.values())
        Qn = 1.0 / inv["total"]
        if abs(Qn - Q) < tol * Q:
            break
        Q = 0.5 * (Q + Qn)
    return dict(sample=sample, fr=fr, res=res, inv=inv, Q=Qn,
                H_ac_oe=H_ac_oe, iters=it + 1)


def q_bracket(sample=REF, nx=200, H_ac_oe=1.0):
    """[Q_min, Q_max] par coins d'intervalles matériaux, à amplitude donnée."""
    return (q_budget(sample, "max_loss", nx, H_ac_oe),
            q_budget(sample, "min_loss", nx, H_ac_oe))


# ====================================== lien alpha_res/alpha_stat <-> Q_eff
def ratio_slope(sample=REF, nx=200, Q_ref=100.0):
    """alpha_res/alpha_stat = c * Q ; retourne c (eta uniforme 1/Q_ref)."""
    eta = 1.0 / Q_ref
    asm = assemble(sample, nx, eta_p=eta, eta_mag=eta)
    fr, a_res = find_resonance(asm)
    a_stat = solve_harm(asm, 1e3)["alpha"]
    return (a_res / a_stat) / Q_ref, fr, a_stat


def q_eff_measured(sample=REF, nx=200):
    """Q effectif mesuré : rapport a_res/a_stat de la thèse inversé à travers
    le modèle (pas de formule 8/pi^2 admise a priori)."""
    c, fr, a_stat = ratio_slope(sample, nx)
    ms = SAMPLES[sample]["meas"]
    return (ms["a_res"] / ms["a_stat"]) / c, c, fr, a_stat


# ============================================= bornes INVERSES
def inverse_bounds(Q_meas, sample=REF, nx=200):
    """1/Q_meas >= 1/Q_i pour CHAQUE canal : lam_flux <= lam_max
    (1/Q_Foucault ~ lam^2) ; Q_m PZT in situ >= Qm_min."""
    asm = assemble(sample, nx)
    fr, _ = find_resonance(asm)
    res = solve_harm(asm, fr)
    geo = asm["geo"]
    w0 = 2 * np.pi * fr
    m, t = asm["mags"][0]
    dV = t * geo["w"] * asm["h"]
    P1 = sum(eddy_power_slab(abs(res["Tm"][0][e]), fr, m, 1.0, t)
             * dV[e] for e in range(asm["nx"]) if asm["in1"][e])
    invQ_e1 = P1 / (w0 * 0.5 * res["Wtot"])
    lam_max = np.sqrt((1.0 / Q_meas) / invQ_e1)
    Qm_min = Q_meas * res["Wp"] / res["Wtot"]
    return dict(lam_max=lam_max, Qm_min=Qm_min, invQ_eddy_lam1=invQ_e1)
