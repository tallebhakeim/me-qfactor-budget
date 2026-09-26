# -*- coding: utf-8 -*-
"""
Section MULTI-ÉCHELLE « mêmes lames » : mesures U. Acevedo-Salas (GeePs,
ANR BIOMEN, sept. 2019, data_ulises/) — chaque lame caractérisée NUE, puis
les assemblages bilame et trilame mesurés avec les MÊMES lames.

Échelle des mesures :
  1. PZT-5H n°3 nue (20x10x1 mm) : |Z|(f) -> fit Butterworth-Van Dyke
     -> f_s, f_a, Q_m in situ, keff².
  2. Terfenol-D n°1 et n°4 nus (14x10x1 mm), biais 506 Oe, |Z| via bobine
     (bobine seule mesurée à part) : la bosse motionnelle Δ|Z| -> fit
     lorentzien -> f_0, Q de la lame magnétique seule = mesure DIRECTE
     des canaux magnétiques (Foucault + hystérésis) au niveau matériau.
  3. Bilame PZT3+TD4 et trilame TD1/PZT3/TD4 à 507 Oe : V_out(f) à
     H_ac = 1 Oe (documenté) -> alpha_res, Q(-3 dB et fit).

Échelle des prédictions (zéro recalage sur les assemblages) :
  A. lame TD nue : canaux Foucault + Rayleigh seuls (fraction d'énergie 1),
     barre libre demi-onde, E_TD déduit de f_0 mesuré.
  B. voie CALCULÉE : budget complet avec Q_m(PZT) = mesuré in situ et
     E_p = mesuré in situ (le reste inchangé).
  C. voie ANCRÉE : 1/Q = f_pzt/Q_pzt,mes + f_TD/Q_TD,mes (fractions du
     modèle structurel, canaux magnétiques REMPLACÉS par la mesure de la
     lame nue) -> teste l'ADDITIVITÉ seule.
Sortie : fig_en_multiscale.png (4 panneaux) + tableau console + JSON.
"""
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import least_squares

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import demag_lambda as dl
import qfactor_hysteresis as qh
import qfactor_model as qm

DATA = Path(__file__).parent / "data_ulises"
BLUE, RED, GREY, GREEN = "#4878a8", "#c1272d", "#666666", "#3c7a3c"
OE = qm.OE
MU0 = 4e-7 * np.pi

L_PZT, W, T1 = 20e-3, 10e-3, 1e-3
L_TD = 14e-3                       # cohérent avec f_0 mesuré et le protocole


def _one(dic):
    return np.ravel([dic[k] for k in dic if not k.startswith("__")][0])


# ----------------------------------------------------- 1. PZT nu : fit BVD
def fit_pzt():
    Z = _one(loadmat(DATA / "PZT5H_1mm_3_Z_65to80kHz.mat"))
    f = np.linspace(65e3, 80e3, Z.size)

    def model(p):
        R, L, C, C0 = np.exp(p)
        w = 2 * np.pi * f
        Zm = R + 1j * w * L + 1 / (1j * w * C)
        return np.abs(1 / (1j * w * C0 + 1 / Zm))

    s = int(np.argmin(Z))
    fs0 = f[s]
    C0g = 3760 * 8.85e-12 * L_PZT * W / T1
    Cg = C0g * 0.1
    Lg = 1 / (Cg * (2 * np.pi * fs0)**2)
    p0 = np.log([Z[s], Lg, Cg, C0g])
    r = least_squares(lambda p: model(p) - Z, p0, method="lm", max_nfev=20000)
    R, L, C, C0 = np.exp(r.x)
    fs = 1 / (2 * np.pi * np.sqrt(L * C))
    Q = np.sqrt(L / C) / R
    fa = fs * np.sqrt(1 + C / C0)
    k2 = 1 - (fs / fa)**2
    Ep = 7500 * (2 * L_PZT * fs)**2
    return dict(f=f, Z=Z, Zfit=model(r.x), fs=fs, fa=fa, Q=Q, k2=k2,
                Ep=Ep, R=R, C0=C0)


# ------------------------- 2. TD nus : bosse motionnelle via la bobine
def fit_td(name):
    Zt = _one(loadmat(DATA / f"{name}.mat"))
    ft = np.linspace(50e3, 80e3, Zt.size)
    Zc = _one(loadmat(DATA / "ZCoil_50to90kHz_Hdc507Oe.mat"))
    fc = np.linspace(50e3, 90e3, Zc.size)
    dZ = Zt - np.interp(ft, fc, Zc)
    # Δ|Z| = projection de l'impédance motionnelle sur la phase de la
    # bobine -> forme DISPERSIVE (S). Modèle : lorentzienne complexe
    # projetée, Re[C e^{j psi} L(f)] + fond affine, L = 1/(1+2jQ(f-f0)/f0).
    m = (ft > 52e3) & (ft < 79e3)
    x, y = ft[m], dZ[m]

    def model(p):
        C, psi, f0, Q, a, b = p
        Lz = 1.0 / (1 + 2j * Q * (x - f0) / f0)
        return C * np.real(np.exp(1j * psi) * Lz) + a + b * (x - 65e3) / 1e4

    # amorces depuis les extrema (séparation max->min = f0/Q)
    imax, imin = int(np.argmax(y)), int(np.argmin(y))
    f0g = 0.5 * (x[imax] + x[imin])
    Qg = max(f0g / abs(x[imin] - x[imax]), 3.0)
    p0 = [y[imax] - y[imin], np.pi / 2, f0g, Qg, np.median(y), 0.0]
    lo = [0.1, -np.pi, f0g - 4e3, 2.0, -np.inf, -np.inf]
    hi = [np.inf, np.pi, f0g + 4e3, 60.0, np.inf, np.inf]
    r = least_squares(lambda p: model(p) - y, p0, bounds=(lo, hi),
                      max_nfev=40000)
    C, psi, f0, Q, a, b = r.x
    return dict(f=x, dZ=y, fit=model(r.x), f0=f0, Q=abs(Q), A=C)


# -------------------------------- 3. assemblages : V_out(f) à H_ac = 1 Oe
def fit_sweep(fname, t_tot_cm):
    d = loadmat(DATA / fname)
    V = np.ravel(d["Vout_res"])
    f = np.ravel(d["FreqSweep"]) * 1e3

    def model(p):
        A, f0, Q, base = p
        return A / np.sqrt(1 + (2 * Q * (f - f0) / f0)**2) + base

    i = int(np.argmax(V))
    p0 = [V[i], f[i], 20.0, np.median(V[:40])]
    r = least_squares(lambda p: model(p) - V, p0, max_nfev=20000)
    A, f0, Q, base = r.x
    alpha = (A + base) / t_tot_cm
    return dict(f=f, V=V, fit=model(r.x), f0=f0, Q=abs(Q), alpha=alpha)


# ---------------------- A. prédiction de la lame TD NUE (barre libre)
def predict_td_bare(f0_meas, H_app_oe=506.0, H_ac_oe=(0.1, 1.0), corner="nom"):
    """Canaux magnétiques seuls, fraction d'énergie 1. Barre libre
    demi-onde : S(x) = S0 cos(pi x/L), T = E_TD S. E_TD déduit de f0."""
    E_td = 9250 * (2 * L_TD * f0_meas)**2
    N = dl.demag_N(L_TD, W, T1)
    # biais interne auto-cohérent avec chi(H) du modèle énergie
    import qfactor_disk as qd
    # lame NUE : aucune précontrainte de collage (la table des barreaux et
    # des disques collés porte -23,8 MPa)
    pre = qd.PRESTRESS_PA
    qd.PRESTRESS_PA = 0.0
    try:
        H_int, d33m, chi = qd.bias_interne(H_app_oe, N)
    finally:
        qd.PRESTRESS_PA = pre
    lam = (1 - N) / (1 + chi * N)
    x = np.linspace(0, L_TD, 200)
    dx = x[1] - x[0]
    prof = np.abs(np.cos(np.pi * x / L_TD))
    # p1 : densité de puissance Foucault pour T = 1 Pa (quadratique en T)
    p1 = qm.eddy_power_slab(1.0, f0_meas,
                            dict(sigma=1.67e6, chi=(chi,), d33m=d33m),
                            lam, T1, mu_r=chi + 1)
    out = {}
    for hac in H_ac_oe:
        Q = 10.0
        for _ in range(60):
            h_int = hac / (1 + chi * N) * OE
            T0 = E_td * Q * d33m * h_int
            Tprof = prof * T0
            # énergie élastique (carré d'amplitude) : int (T²/E) dV
            Wtot = float(np.trapz(Tprof**2, x)) * W * T1 / E_td
            # puissance Foucault : int p1.T² dV
            P = p1 * float(np.trapz(Tprof**2, x)) * W * T1
            invQ = P / (2 * np.pi * f0_meas * 0.5 * Wtot)
            vol = np.full(x.size, W * T1 * dx)
            invQ += qh.invQ_hyst_layer(Tprof, vol, Wtot, d33m, chi, N,
                                       0.15, 0.9, 18.0)
            Qn = 1 / invQ
            if abs(Qn - Q) < 1e-4 * Q:
                break
            Q = 0.5 * (Q + Qn)
        out[hac] = Qn
    return dict(E_td=E_td, chi=chi, d33m=d33m, lam=lam, N=N,
                H_int=H_int, Q=out)


def main():
    print("=" * 76)
    print("MULTI-ÉCHELLE « mêmes lames » — mesures U. Acevedo-Salas "
          "(GeePs/BIOMEN, 2019)")
    print("=" * 76)
    pzt = fit_pzt()
    print(f"\nPZT-5H n°3 NUE : f_s = {pzt['fs']/1e3:.2f} kHz, f_a = "
          f"{pzt['fa']/1e3:.2f} kHz, Q_m in situ = {pzt['Q']:.0f}, "
          f"keff² = {pzt['k2']:.3f}, E_p = {pzt['Ep']/1e9:.1f} GPa")
    td1 = fit_td("TD_1mm_1_Z_50to80kHz_Hdc506")
    td4 = fit_td("TD_1mm_4_Z_50to80kHz_Hdc506")
    for nm, td in (("TD n°1", td1), ("TD n°4", td4)):
        print(f"{nm} NU @506 Oe : f_0 = {td['f0']/1e3:.2f} kHz, "
              f"Q = {td['Q']:.1f} (fit lorentzien sur Δ|Z|)")
    bil = fit_sweep("Bilame_PZT3TD4_1_1_SweepFreq_60to80kHz_Hac1Oe_"
                    "Hdc507Oe.mat", 0.2)
    tri = fit_sweep("Trilame_TD1PZT3TD4_1_1_1_SweepFreq_60to80kHz_"
                    "Hdc507Oe_Hac1Oe.mat.mat", 0.3)
    print(f"BILAME @507 Oe, 1 Oe : f_r = {bil['f0']/1e3:.2f} kHz, "
          f"Q = {bil['Q']:.1f}, alpha = {bil['alpha']:.1f} V/cmOe")
    print(f"TRILAME @507 Oe, 1 Oe : f_r = {tri['f0']/1e3:.2f} kHz, "
          f"Q = {tri['Q']:.1f}, alpha = {tri['alpha']:.1f} V/cmOe")

    # ---------------- prédiction lame TD nue
    tdp = predict_td_bare(0.5 * (td1["f0"] + td4["f0"]))
    print(f"\nTD nu PRÉDIT (Foucault+Rayleigh, E_TD = {tdp['E_td']/1e9:.1f} "
          f"GPa mesuré, chi_int = {tdp['chi']:.1f}, lam = {tdp['lam']:.2f}) : "
          f"Q = {tdp['Q'][1.0]:.1f} @1 Oe, {tdp['Q'][0.1]:.1f} @0,1 Oe "
          f"— mesuré {td1['Q']:.1f}/{td4['Q']:.1f}")

    # ---------------- assemblages : voie calculée (budget, entrées mesurées)
    Qm_pzt_sauve, E_sauve = qm.PZT["Qm"], qm.PZT["E"]
    E_td_sauve = qm.MAGS["Terfenol-D@bias"]["E"]
    qm.PZT["Qm"] = (pzt["Q"], 0.8 * pzt["Q"], 1.2 * pzt["Q"])
    qm.PZT["E"] = pzt["Ep"]
    qm.MAGS["Terfenol-D@bias"]["E"] = tdp["E_td"]
    if "TD/PZT/TD" not in qm.SAMPLES:
        qm.SAMPLES["TD/PZT/TD"] = dict(
            mags=[("Terfenol-D@bias", 1e-3), ("Terfenol-D@bias", 1e-3)],
            meas=dict(a_stat=np.nan, a_res=np.nan, f_r=tri["f0"]))
    res_calc = {}
    for nm, smp, meas in (("bilame", "PZT/Terf", bil),
                          ("trilame", "TD/PZT/TD", tri)):
        b = qm.q_budget(smp, nx=200, H_ac_oe=1.0)
        lo, hi = qm.q_bracket(smp, 200, H_ac_oe=1.0)
        res_calc[nm] = dict(Q=b["Q"], lo=lo["Q"], hi=hi["Q"], fr=b["fr"],
                            fr_meas=meas["f0"], res=b["res"], inv=b["inv"])
        print(f"{nm} CALCULÉ : Q = {b['Q']:.1f} [{lo['Q']:.1f};{hi['Q']:.1f}]"
              f", f_r = {b['fr']/1e3:.1f} kHz (mesuré {meas['f0']/1e3:.1f})"
              f" — Q mesuré {meas['Q']:.1f}")

    # ---------------- voie ANCRÉE : fractions x mesures de lames
    Q_td_meas = 0.5 * (td1["Q"] + td4["Q"])
    res_anc = {}
    for nm, meas in (("bilame", bil), ("trilame", tri)):
        r = res_calc[nm]["res"]
        f_pzt = r["Wp"] / r["Wtot"]
        f_td = sum(r["Wm"]) / r["Wtot"]
        invQ = f_pzt / pzt["Q"] + f_td / Q_td_meas
        res_anc[nm] = 1 / invQ
        print(f"{nm} ANCRÉ (f_pzt = {f_pzt:.2f}, f_TD = {f_td:.2f}, "
              f"Q_TD,mes = {Q_td_meas:.1f}) : Q = {1/invQ:.1f} "
              f"— mesuré {meas['Q']:.1f}")
    qm.PZT["Qm"], qm.PZT["E"] = Qm_pzt_sauve, E_sauve
    qm.MAGS["Terfenol-D@bias"]["E"] = E_td_sauve

    # ================================================================ figure
    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(9.8, 6.8))
    a1.plot(pzt["f"] / 1e3, pzt["Z"], ".", ms=2.5, color=RED,
            label="measured |Z|")
    a1.plot(pzt["f"] / 1e3, pzt["Zfit"], "-", color=BLUE, lw=1.2,
            label=f"BVD fit: Q$_m$ = {pzt['Q']:.0f}, "
                  f"E$_p$ = {pzt['Ep']/1e9:.1f} GPa")
    a1.set_yscale("log")
    a1.set_xlabel("frequency [kHz]"); a1.set_ylabel("|Z| [Ω]")
    a1.set_title("(a) bare PZT-5H plate", fontsize=9.5)
    a1.legend(fontsize=7)
    for td, col, lab in ((td1, BLUE, "TD plate 1"), (td4, GREEN, "TD plate 4")):
        a2.plot(td["f"] / 1e3, td["dZ"], ".", ms=2.5, color=col, alpha=0.6)
        a2.plot(td["f"] / 1e3, td["fit"], "-", color=col, lw=1.2,
                label=f"{lab}: Q = {td['Q']:.1f}")
    a2.set_xlabel("frequency [kHz]")
    a2.set_ylabel("motional Δ|Z| [Ω]")
    a2.set_title("(b) bare Terfenol-D plates at 506 Oe (through coil)",
                 fontsize=9.5)
    a2.legend(fontsize=7)
    for sw, col, lab in ((bil, BLUE, "bilayer PZT3+TD4"),
                         (tri, GREEN, "trilayer TD1/PZT3/TD4")):
        a3.plot(sw["f"] / 1e3, sw["V"], ".", ms=2.5, color=col, alpha=0.6)
        a3.plot(sw["f"] / 1e3, sw["fit"], "-", color=col, lw=1.2,
                label=f"{lab}: Q = {sw['Q']:.1f}")
    a3.set_xlim(65, 80)
    a3.set_xlabel("frequency [kHz]"); a3.set_ylabel("V$_{out}$ [V]")
    a3.set_title("(c) assembled, measured at h$_{ac}$ = 1 Oe, 507 Oe",
                 fontsize=9.5)
    a3.legend(fontsize=7)
    # (d) l'échelle
    labels = ["bare TD plate", "bilayer", "trilayer"]
    meas_q = [Q_td_meas, bil["Q"], tri["Q"]]
    calc_q = [tdp["Q"][1.0], res_calc["bilame"]["Q"], res_calc["trilame"]["Q"]]
    anc_q = [np.nan, res_anc["bilame"], res_anc["trilame"]]
    xpos = np.arange(3)
    for i in (1, 2):
        nm = ("bilame", "trilame")[i - 1]
        a4.plot([xpos[i]] * 2, [res_calc[nm]["lo"], res_calc[nm]["hi"]],
                color=BLUE, lw=7, alpha=0.35, solid_capstyle="butt",
                label="a-priori interval" if i == 1 else None)
    a4.plot(xpos, calc_q, "o", color=BLUE, ms=8, label="computed budget")
    a4.plot(xpos[1:], anc_q[1:], "s", color=GREEN, ms=8,
            label="anchored on bare-plate Q")
    a4.plot(xpos, meas_q, "*", color=RED, ms=16, label="measured")
    a4.set_xticks(xpos); a4.set_xticklabels(labels, fontsize=8.5)
    a4.set_ylabel("quality factor Q"); a4.set_ylim(0, 32)
    a4.set_title("(d) the same-plates ladder", fontsize=9.5)
    a4.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("fig_en_multiscale.png", dpi=200)
    print("\nFigure : fig_en_multiscale.png")

    out = dict(pzt={k: float(pzt[k]) for k in ("fs", "fa", "Q", "k2", "Ep")},
               td1={k: float(td1[k]) for k in ("f0", "Q")},
               td4={k: float(td4[k]) for k in ("f0", "Q")},
               bilame={k: float(bil[k]) for k in ("f0", "Q", "alpha")},
               trilame={k: float(tri[k]) for k in ("f0", "Q", "alpha")},
               td_pred={str(k): float(v) for k, v in tdp["Q"].items()},
               calc={k: float(v["Q"]) for k, v in res_calc.items()},
               anchored={k: float(v) for k, v in res_anc.items()})
    (Path(__file__).parent / "multiscale_results.json").write_text(
        json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
