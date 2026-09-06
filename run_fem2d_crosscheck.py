# -*- coding: utf-8 -*-
"""
Chantier 4 de l'article Q : brancher le BUDGET DE PERTES PRÉDIT (par canal)
sur le FEM 2D VALIDÉ du core (me_fem), via les pertes élastiques PAR COUCHE
(layer_eta) — plus aucun Q global recalé.

Chaîne : budget qfactor_model (canaux, H_ac = 1 Oe)
  -> eta_PZT  = (1/Q_pzt_meca + 1/Q_diel) . W_tot/W_pzt
     eta_Terf = (1/Q_hyst + 1/Q_Foucault) . W_tot/W_terf
  -> me_fem.solve_harmonic_rect(..., layer_eta=[eta_Terf, eta_PZT])
  -> alpha(f) PRÉDIT, à comparer à la mesure (19,8 V/cmOe @ 70,4 kHz).

Sanity check inclus : layer_eta uniforme = comportement du Q global.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))
from me import me_fem as fe          # noqa: E402
from me import me_deam as dm         # noqa: E402

import qfactor_model as qm           # noqa: E402

W, TP, TM, LP = 10e-3, 1e-3, 1e-3, 20e-3
SIG_PRESTRESS = -23.8e6              # calibré sur H_opt=525 Oe (validate_malleron)


def layers():
    return [dict(kind="mag", name="Terfenol-D", t=TM),
            dict(kind="piezo", name="PZT-5H", t=TP)]


def main():
    coeffs = dm.deam_coeffs("Terfenol-D", 525.0, SIG_PRESTRESS)

    # ---- 1. sanity : layer_eta uniforme ~ Q global (au pic, tol qq %)
    Q0 = 12.0
    f = np.linspace(50e3, 80e3, 400)
    hQ = fe.solve_harmonic_rect(LP, W, layers(), f, mode="LT", Q=Q0,
                                H_oe=1.0, eddy=False, mag_coeffs=coeffs)
    hE = fe.solve_harmonic_rect(LP, W, layers(), f, mode="LT",
                                H_oe=1.0, eddy=False, mag_coeffs=coeffs,
                                layer_eta=[1.0 / Q0, 1.0 / Q0])
    print(f"sanity eta uniforme 1/{Q0:.0f} vs Q global : "
          f"alpha_pic {hE['alpha_r']:.1f} vs {hQ['alpha_r']:.1f} "
          f"({(hE['alpha_r']/hQ['alpha_r']-1)*100:+.1f} %), "
          f"f_r {hE['f_r']/1e3:.1f} vs {hQ['f_r']/1e3:.1f} kHz")

    # ---- 2. pertes par couche issues du BUDGET PRÉDIT (aucun recalage)
    bud = qm.q_budget("PZT/Terf", nx=200, H_ac_oe=1.0)
    res, inv = bud["res"], bud["inv"]
    eta_pzt = (inv["pzt_meca"] + inv["pzt_diel"]) * res["Wtot"] / res["Wp"]
    eta_terf = (inv["mag_hyst"] + inv["foucault"]) * res["Wtot"] / res["Wm"][0]
    print(f"\nbudget -> eta_PZT = {eta_pzt:.4f} (Q_pzt {1/eta_pzt:.0f}), "
          f"eta_Terf = {eta_terf:.4f} (Q_terf {1/eta_terf:.0f}) ; "
          f"Q_total prédit = {bud['Q']:.1f}")

    hP = fe.solve_harmonic_rect(LP, W, layers(), f, mode="LT",
                                H_oe=1.0, eddy=False, mag_coeffs=coeffs,
                                layer_eta=[eta_terf, eta_pzt])
    a_stat = fe.solve_static_rect(LP, W, layers(), 1.0, mode="LT",
                                  mag_coeffs=coeffs)
    a_stat_v = abs(a_stat["V_oc"]) / ((TP + TM) * 100)
    meas = qm.SAMPLES["PZT/Terf"]["meas"]
    print(f"\nFEM 2D + pertes par couche PRÉDITES (0 recalage) :")
    print(f"  alpha_stat = {a_stat_v:.2f} vs {meas['a_stat']} mesuré "
          f"({(a_stat_v/meas['a_stat']-1)*100:+.0f} %)")
    print(f"  alpha_res  = {hP['alpha_r']:.1f} vs {meas['a_res']} mesuré "
          f"({(hP['alpha_r']/meas['a_res']-1)*100:+.0f} %) "
          f"@ f_r {hP['f_r']/1e3:.1f} kHz (mes 70,4)")
    return hE, hQ, hP


if __name__ == "__main__":
    main()
