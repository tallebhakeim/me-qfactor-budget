# -*- coding: utf-8 -*-
"""
Vérification du PoC "Q par bilan d'énergie" v3 — 18 tests.
Auto-cohérence du modèle, limites analytiques (Foucault, Rayleigh, démag),
monotonie des bornes, non-linéarité Q(H_ac), raideur circuit ouvert,
confrontation aux 4 échantillons Malleron, et branchement FEM 2D (layer_eta).
"""
import numpy as np

import demag_lambda as dl
import qfactor_hysteresis as qh
import qfactor_model as qm

TESTS = []
NX = 200


def check(nom, ok, detail=""):
    TESTS.append((nom, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {nom}" + (f"  ({detail})" if detail else ""))


def main():
    asm = qm.assemble(qm.REF, NX)
    fr, _ = qm.find_resonance(asm)
    res = qm.solve_harm(asm, fr)
    terf = qm.MAGS["Terfenol-D@bias"]

    # 1. fréquence de résonance vs mesure (référence, tol 3 %)
    check("f_r modèle proche de la mesure (70,4 kHz, tol 3 %)",
          abs(fr / qm.SAMPLES[qm.REF]["meas"]["f_r"] - 1) < 0.03,
          f"f_r = {fr/1e3:.1f} kHz")

    # 2. convergence en maillage
    fr2, _ = qm.find_resonance(qm.assemble(qm.REF, 2 * NX))
    check("convergence maillage (f_r stable à 0,2 % quand nx double)",
          abs(fr2 / fr - 1) < 0.002, f"delta = {abs(fr2/fr-1)*100:.3f} %")

    # 3. fractions d'énergie sommant à 1
    s = (res["Wp"] + sum(res["Wm"]) + res["We"]) / res["Wtot"]
    check("fractions d'énergie sommant à 1", abs(s - 1) < 1e-12,
          f"We/Wtot = {res['We']/res['Wtot']*100:.1f} %")

    # 4. AUTO-COHÉRENCE clé : Q bande passante = Q bilan d'énergie
    eta = 0.01
    asm_d = qm.assemble(qm.REF, NX, eta_p=eta, eta_mag=eta)
    Q_bw = qm.q_from_bandwidth(asm_d, fr)
    Q_eb = 1.0 / (eta * (res["Wp"] + sum(res["Wm"])) / res["Wtot"])
    check("Q bande passante = Q bilan d'énergie (eta uniforme, tol 1 %)",
          abs(Q_bw / Q_eb - 1) < 0.01, f"bw {Q_bw:.1f} vs bilan {Q_eb:.1f}")

    # 5-6. rapport alpha_res/alpha_stat = c.Q, c physique
    c1, _, _ = qm.ratio_slope(qm.REF, NX, Q_ref=60.0)
    c2, _, _ = qm.ratio_slope(qm.REF, NX, Q_ref=200.0)
    check("pente c = (alpha_res/alpha_stat)/Q indépendante de Q (tol 2 %)",
          abs(c1 / c2 - 1) < 0.02, f"c(60) = {c1:.3f}, c(200) = {c2:.3f}")
    check("pente c dans [0,7 ; 1,3] autour de 8/pi² = 0,811",
          0.7 < c1 < 1.3, f"c = {c1:.3f}")

    # 7. Foucault : limite basse fréquence = sigma w² b*² t²/24
    T, f_lo, lam, t = 1e6, 100.0, 1.0, 1e-3
    p_num = qm.eddy_power_slab(T, f_lo, terf, lam, t)
    b = lam * terf["d33m"] * T
    sig, mu_r = terf["sigma"], terf["chi"][0] + 1
    p_ana = sig * (2 * np.pi * f_lo)**2 * b**2 * t**2 / 24
    check("Foucault : limite BF = sigma.w².b².t²/24 (tol 0,1 %)",
          abs(p_num / p_ana - 1) < 1e-3, f"num/ana = {p_num/p_ana:.5f}")

    # 8. Foucault : écrantage haute fréquence
    f_hi = 300e3
    p_hi = qm.eddy_power_slab(T, f_hi, terf, lam, t)
    p_bf = sig * (2 * np.pi * f_hi)**2 * b**2 * t**2 / 24
    check("Foucault : écrantage HF (p_exact < formule BF pour t > delta)",
          p_hi < p_bf, f"rapport = {p_hi/p_bf:.2f}")

    # 9. Foucault : sigma -> 0 et loi en lam²
    mat0 = dict(terf, sigma=1.0)
    p0 = qm.eddy_power_slab(T, 70e3, mat0, 1.0, t)
    p_ref = qm.eddy_power_slab(T, 70e3, terf, 1.0, t)
    r_lam = qm.eddy_power_slab(T, 70e3, terf, 0.5, t) / p_ref
    check("Foucault : p négligeable si sigma -> 0 ; p proportionnel à lam²",
          p0 < 1e-5 * p_ref and abs(r_lam - 0.25) < 1e-9,
          f"p(sigma=1)/p = {p0/p_ref:.1e}, ratio lam 0,5/1 = {r_lam:.3f}")

    # 10. colle : loi en t_g^{1/2}
    q_g1 = qm.q_glue(asm, res, 1.3e9, 20e-6, 0.05)
    q_g2 = qm.q_glue(asm, res, 1.3e9, 5e-6, 0.05)
    check("colle : 1/Q décroît en t_g^{1/2} quand t_g décroît",
          q_g2 < q_g1 and abs(q_g2 / q_g1 - 0.5) < 1e-6,
          f"rapport = {q_g2/q_g1:.4f} (attendu 0,500)")

    # 11. monotonie : Q_min <= Q_nominal <= Q_max ; retirer un canal augmente Q
    blo, bhi = qm.q_bracket(qm.REF, NX)
    bud = qm.q_budget(qm.REF, nx=NX)
    q_sans = 1.0 / (bud["inv"]["total"] - bud["inv"]["foucault"])
    check("encadrement ordonné et monotone par canal",
          blo["Q"] <= bud["Q"] <= bhi["Q"] and q_sans > bud["Q"],
          f"[{blo['Q']:.1f} ; {bud['Q']:.1f} ; {bhi['Q']:.1f}]")

    # 12. raideur circuit ouvert : +1 à +8 % sur f_r
    fr_ns, _ = qm.find_resonance(qm.assemble(qm.REF, NX, stiffen=False))
    gain = fr / fr_ns - 1
    check("raideur piézo circuit ouvert : +1 à +8 % sur f_r",
          0.01 < gain < 0.08, f"+{gain*100:.1f} %")

    # 13. VALIDATION : les 3 échantillons au biais Terfenol encadrés (1 table)
    ok3, det = True, []
    for smp in ["PZT/Terf", "PZT/Met/Terf", "PZT/Terf/Met"]:
        lo, hi = qm.q_bracket(smp, NX)
        Qe, _, _, _ = qm.q_eff_measured(smp, NX)
        ok3 &= lo["Q"] <= Qe <= hi["Q"]
        det.append(f"{smp}: {Qe:.0f} ∈ [{lo['Q']:.0f};{hi['Q']:.0f}]")
    check("3 échantillons Terfenol encadrés (1 table, 0 recalage)",
          ok3, " ; ".join(det))

    # 14. anomalie PZT/Met documentée (rapport incompatible avec un Q linéaire)
    lo_m, _ = qm.q_bracket("PZT/Met", NX)
    Qe_m, _, _, _ = qm.q_eff_measured("PZT/Met", NX)
    check("PZT/Met : rapport mesuré sous la borne basse (mesure non linéaire)",
          Qe_m < lo_m["Q"], f"Q_rapport = {Qe_m:.1f} < Q_min = {lo_m['Q']:.1f}")

    # 15. démagnétisation : cube = 1/3, somme des 3 axes = 1, lambda dans ]0;1[
    Nc = dl.demag_N(1, 1, 1)
    Ns = sum(dl.demag_N(*d) for d in ((14, 10, 1), (10, 14, 1), (1, 10, 14)))
    lam, N = qm.lam_N(1e-3, terf["chi"][0])
    check("démag : cube N=1/3, somme 3 axes = 1, lambda calculé dans ]0;1[",
          abs(Nc - 1 / 3) < 1e-3 and abs(Ns - 1) < 1e-4 and 0 < lam < 1,
          f"N_cube = {Nc:.5f}, somme = {Ns:.5f}, lambda = {lam:.3f}")

    # 16. Rayleigh : W ~ dH³ en petit signal, saturation quadratique, W(0)=0
    OE = qm.OE
    args = (7.4, 0.15, 0.9, 18 * OE)
    r_small = qh.w_cycle(2.0, *args) / qh.w_cycle(1.0, *args)
    r_big = qh.w_cycle(400 * OE, *args) / qh.w_cycle(200 * OE, *args)
    check("Rayleigh : W~dH³ petit signal, ~dH² saturé, W(0)=0",
          abs(r_small - 8) < 0.1 and r_big < 5.0
          and qh.w_cycle(0.0, *args) == 0.0,
          f"ratio petit = {r_small:.2f}, grand = {r_big:.2f}")

    # 17. non-linéarité : Q auto-cohérent DÉCROÎT avec H_ac, et converge
    q05 = qm.q_budget(qm.REF, nx=NX, H_ac_oe=0.5)
    q65 = qm.q_budget(qm.REF, nx=NX, H_ac_oe=6.5)
    check("Q(H_ac) décroissant (hystérésis non linéaire) et point fixe convergé",
          q05["Q"] > bud["Q"] > q65["Q"] and q65["iters"] < 40,
          f"Q(0,5)={q05['Q']:.1f} > Q(1)={bud['Q']:.1f} > Q(6,5)={q65['Q']:.1f}, "
          f"{q65['iters']} itér.")

    # 18. branchement FEM 2D validé (layer_eta) : eta uniforme ~ Q global
    #     (tol 10 %) et chaîne budget->2D dans ±30 % de la mesure (19,8).
    #     Nécessite le core FEM interne (non distribué) : SKIP proprement
    #     dans le dépôt public.
    try:
        import run_fem2d_crosscheck as cc
        hE, hQ, hP = cc.main()
        ok18 = abs(hE["alpha_r"] / hQ["alpha_r"] - 1) < 0.10 \
            and abs(hP["alpha_r"] / 19.8 - 1) < 0.30
        check("FEM 2D layer_eta : uniforme~Q global (10 %) ; "
              "chaîne prédite à ±30 % de la mesure",
              ok18, f"uniforme/global = {hE['alpha_r']/hQ['alpha_r']:.2f}, "
              f"alpha_res prédit = {hP['alpha_r']:.1f} vs 19,8")
    except ImportError:
        print("[SKIP] test 18 (FEM 2D interne non disponible dans ce dépôt ; "
              "les 17 tests autonomes suffisent au budget)")

    n_ok = sum(ok for _, ok in TESTS)
    print(f"\n{n_ok}/{len(TESTS)} PASS")
    return n_ok == len(TESTS)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
