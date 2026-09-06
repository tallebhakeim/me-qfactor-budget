# -*- coding: utf-8 -*-
"""
Budget de facteur de qualité par bilan d'énergie (v3) — les 4 échantillons de
la thèse Malleron avec UNE SEULE table de matériaux, canaux Foucault et
hystérésis CALCULÉS (lambda démagnétisant exact + loi de Rayleigh), résolution
auto-cohérente Q(H_ac). Figure fig_q_budget.png (3 panneaux).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qfactor_model as qm

NX = 200
NOMS = dict(pzt_meca="PZT mécanique (Q_m)",
            mag_hyst="hystérésis magnétoméca (Rayleigh)",
            pzt_diel="PZT diélectrique",
            foucault="Foucault (λ démag calculé)",
            colle="colle époxy (shear-lag)")


def main():
    print("=" * 76)
    print("BUDGET DE Q PAR BILAN D'ÉNERGIE v3 — canaux calculés, "
          "auto-cohérent Q(H_ac)")
    print("=" * 76)

    # ------------------------- budget détaillé de l'échantillon de référence
    bud = qm.q_budget(qm.REF, nx=NX, H_ac_oe=1.0)
    res = bud["res"]
    lam, N = qm.lam_N(1e-3, qm.MAGS["Terfenol-D@bias"]["chi"][0])
    print(f"\n--- Référence {qm.REF} (H_ac = 1 Oe) ---")
    print(f"f_r modèle = {bud['fr']/1e3:.1f} kHz (mesure 70,4) ; "
          f"N démag = {N:.4f} -> lambda_flux = {lam:.3f} (CALCULÉ)")
    print(f"{'canal':36} {'1/Q_i':>9} {'Q_i seul':>9}")
    for k, lbl in NOMS.items():
        v = bud["inv"][k]
        print(f"{lbl:36} {v:9.5f} {1/max(v,1e-12):9.0f}")
    print(f"{'TOTAL (auto-cohérent, %d itér.)' % bud['iters']:36} "
          f"{bud['inv']['total']:9.5f} {bud['Q']:9.1f}")

    Qe_ref, c_ref, _, a_stat_ref = qm.q_eff_measured(qm.REF, NX)
    ib = qm.inverse_bounds(Qe_ref, qm.REF, NX)
    print(f"\nBornes inverses (mesure Q = {Qe_ref:.0f}) : "
          f"lam_flux <= {ib['lam_max']:.2f} (calculé : {lam:.2f} OK) ; "
          f"Q_m PZT in situ >= {ib['Qm_min']:.0f}")

    # --------------------------------------- les 4 échantillons, même table
    print(f"\n--- Les 4 échantillons (1 table matériaux, 0 recalage, "
          f"H_ac = 1 Oe) ---")
    print(f"{'échantillon':14} {'f_r mod/mes kHz':>17} {'Q pred':>7} "
          f"{'[Q_min;Q_max]':>14} {'Q mesuré':>9}  verdict")
    rows = []
    for s in qm.SAMPLES:
        b = qm.q_budget(s, nx=NX)
        blo, bhi = qm.q_bracket(s, NX)
        Qe, c, _, a_stat = qm.q_eff_measured(s, NX)
        ms = qm.SAMPLES[s]["meas"]
        ok = blo["Q"] <= Qe <= bhi["Q"]
        rows.append(dict(s=s, b=b, blo=blo, bhi=bhi, Qe=Qe, c=c,
                         a_stat=a_stat, ok=ok))
        print(f"{s:14} {b['fr']/1e3:7.1f} / {ms['f_r']/1e3:5.1f} "
              f"{b['Q']:7.1f} [{blo['Q']:5.1f};{bhi['Q']:5.1f}] "
              f"{Qe:9.1f}  {'DANS' if ok else 'HORS'}")

    print("""
Lecture : 3/4 encadrés sans recalage ; pertes dominées par le TERFENOL
(Foucault calculé ~68 % + hystérésis Rayleigh ~13 %). PZT/Met (38 Oe) HORS
cadre = diagnostic protocole : alpha_stat mesuré à H_ac = 6,5 Oe = ±17 % du
biais (Rayleigh non linéaire) -> le rapport alpha_res/alpha_stat n'est une
mesure de Q qu'en régime linéaire.""")

    # ------------------------------------------- courbe Q(H_ac) : non-linéarité
    Hacs = np.array([0.3, 0.7, 1.0, 2.0, 3.0, 4.5, 6.5, 10.0])
    q_nom = [qm.q_budget(qm.REF, "nom", NX, h)["Q"] for h in Hacs]
    q_min = [qm.q_budget(qm.REF, "max_loss", NX, h)["Q"] for h in Hacs]
    q_max = [qm.q_budget(qm.REF, "min_loss", NX, h)["Q"] for h in Hacs]
    print("Q(H_ac) référence :",
          " ; ".join(f"{h:g} Oe -> {q:.1f}" for h, q in zip(Hacs, q_nom)))

    # -------------------------------- sensibilité E_p in situ (chantier 3)
    qm.set_pzt_insitu(True)
    b_ins = qm.q_budget(qm.REF, nx=NX)
    fmet = qm.q_budget("PZT/Met", nx=NX)["fr"]
    print(f"\nSensibilité E_p in situ (54,7 GPa, du f_s PZT seul 67,5 kHz) : "
          f"f_r réf {b_ins['fr']/1e3:.1f} kHz, Q {b_ins['Q']:.1f} "
          f"(vs {bud['Q']:.1f}) ; f_r PZT/Met {fmet/1e3:.1f} kHz "
          f"(mes 68,4 ; fiche : 76,5)")
    qm.set_pzt_insitu(False)

    # ------------------------------------------------ alpha_res prédits
    print(f"\n{'échantillon':14} {'alpha_res prédit [min;max]':>28} {'mesuré':>8}")
    for r in rows:
        ms = qm.SAMPLES[r["s"]]["meas"]
        lo = r["c"] * r["blo"]["Q"] * ms["a_stat"]
        hi = r["c"] * r["bhi"]["Q"] * ms["a_stat"]
        print(f"{r['s']:14} {'[%.0f ; %.0f]' % (lo, hi):>28} "
              f"{ms['a_res']:8.1f}")

    # ================================================================ figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4.2))

    keys = list(NOMS)
    y = np.arange(len(keys))
    ref = next(r for r in rows if r["s"] == qm.REF)
    v_nom = [bud["inv"][k] for k in keys]
    v_lo = [ref["bhi"]["inv"][k] for k in keys]
    v_hi = [ref["blo"]["inv"][k] for k in keys]
    ax1.barh(y, v_nom, color="#4878a8", alpha=0.85, label="nominal")
    ax1.errorbar(v_nom, y, xerr=[np.maximum(np.array(v_nom) - v_lo, 0),
                                 np.maximum(np.array(v_hi) - v_nom, 0)],
                 fmt="none", ecolor="k", capsize=4, lw=1.2, label="intervalle")
    ax1.axvline(1 / ref["Qe"], color="crimson", ls="--", lw=1.5,
                label=f"1/Q mesuré = 1/{ref['Qe']:.0f}")
    ax1.set_yticks(y)
    ax1.set_yticklabels([NOMS[k] for k in keys], fontsize=8.5)
    ax1.set_xlabel("contribution 1/Q_i")
    ax1.set_title(f"(a) Budget de pertes — {qm.REF}", fontsize=10)
    ax1.legend(fontsize=7.5, loc="lower right")
    ax1.invert_yaxis()

    for i, r in enumerate(rows):
        ax2.plot([i, i], [r["blo"]["Q"], r["bhi"]["Q"]], color="#4878a8",
                 lw=7, alpha=0.45, solid_capstyle="butt",
                 label="encadrement a priori" if i == 0 else None)
        ax2.plot(i, r["b"]["Q"], "o", color="#4878a8", ms=7,
                 label="nominal" if i == 0 else None)
        ax2.plot(i, r["Qe"], "r*", ms=15,
                 label="Q mesuré" if i == 0 else None)
    ax2.set_xticks(range(len(rows)))
    ax2.set_xticklabels([r["s"] for r in rows], fontsize=7.5)
    ax2.set_ylabel("facteur de qualité Q")
    ax2.set_title("(b) 4 échantillons, 1 table, 0 recalage", fontsize=10)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=7.5)
    ax2.annotate("mesure non linéaire\n(H_ac = ±17 % du biais)",
                 xy=(1, rows[1]["Qe"]), xytext=(1.1, 32), fontsize=7,
                 color="crimson", arrowprops=dict(arrowstyle="->",
                                                  color="crimson", lw=0.8))

    ax3.fill_between(Hacs, q_min, q_max, color="#4878a8", alpha=0.25,
                     label="encadrement matériaux")
    ax3.plot(Hacs, q_nom, "o-", color="#4878a8", label="nominal")
    ax3.axhline(ref["Qe"], color="crimson", ls="--", lw=1.5,
                label=f"Q mesuré = {ref['Qe']:.0f}")
    ax3.set_xlabel("amplitude d'excitation H_ac [Oe]")
    ax3.set_ylabel("Q auto-cohérent")
    ax3.set_title("(c) Non-linéarité Q(H_ac) — hystérésis Rayleigh",
                  fontsize=10)
    ax3.legend(fontsize=7.5)
    ax3.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig("fig_q_budget.png", dpi=160)
    print("\nFigure : fig_q_budget.png")


if __name__ == "__main__":
    main()
