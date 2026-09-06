# -*- coding: utf-8 -*-
"""
Rapport + figure du TRANSFERT DISQUE : le budget (matériaux seuls, aucun
recalage) confronté aux 36 points identifiés de la thèse Rizzo (C2N, 2020).
Produit fig_en_disk.png (2 panneaux) et le tableau f_s / Q.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qfactor_disk as qd

BLUE, RED, GREY = "#4878a8", "#c1272d", "#666666"


def main():
    print("=" * 72)
    print("TRANSFERT DISQUE — 6 échantillons Rizzo (C2N), 36 Q identifiés")
    print("=" * 72)
    Ep, ka0 = qd.pic_modulus()
    print(f"PIC181 : E = {Ep/1e9:.1f} GPa (déduit de Np), ka = {ka0:.3f}\n")

    print(f"{'échantillon':16} {'f_s mod [kHz]':>13} {'f_s mes':>8} {'écart':>7}")
    fs_err = []
    for sname, s in qd.SAMPLES.items():
        md = qd.radial_mode(sname)
        e = md["f"] / s["f_meas"] - 1
        fs_err.append(e)
        print(f"{sname:16} {md['f']/1e3:13.1f} {s['f_meas']/1e3:8.1f} "
              f"{e*100:+6.1f}%")
    print(f"écart moyen |f_s| : {np.mean(np.abs(fs_err))*100:.1f}%\n")

    preds = {s: [qd.q_budget_disk(s, b)["Q"] for b in qd.BIAS_T]
             for s in qd.SAMPLES}
    print(f"{'échantillon':16}" + "".join(f"  B={b:5.3f}" for b in qd.BIAS_T))
    for s in qd.SAMPLES:
        print(f"{s:16}" + "".join(f" {p:8.0f}" for p in preds[s]) + "  préd")
        print(f"{'':16}" + "".join(f" {m:8.0f}" for m in qd.Q_MEAS[s]) + "  mes")
    print("""
Lecture : aux DEUX PLUS HAUTS biais (Terfenol quasi linéaire, chi_int <= 2),
les 12 Q prédits sont à un facteur <= 2,8 des mesures avec l'ORDRE des
échantillons respecté (P-M-P > M-P ; Ø10 > Ø16 à fraction égale). À bas
biais, chi_int atteint 30-47 : les deux canaux calculés s'effondrent
((λd)² et écrantage) alors que la mesure sature vers 100-500 -> pertes
EXCÉDENTAIRES de parois (Bertotti), canal absent du budget, dominant
uniquement dans ce régime.""")

    # ---------------------------- figure : (a) scatter hauts biais, (b) Q(B)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    marks = dict(zip(qd.SAMPLES, "osD^vP"))
    for s in qd.SAMPLES:
        for j in (0, 1):                      # 0.1 et 0.058 T
            a1.plot(qd.Q_MEAS[s][j], preds[s][j], marks[s], color=BLUE,
                    ms=6, label=s if j == 0 else None)
    lim = [0, 600]
    a1.plot(lim, lim, "-", color=GREY, lw=1)
    a1.fill_between(lim, [l / 2 for l in lim], [l * 2 for l in lim],
                    color=GREY, alpha=0.15, label="factor-2 band")
    a1.set_xlim(lim); a1.set_ylim(lim)
    a1.set_xlabel("measured Q (identified)"); a1.set_ylabel("predicted Q")
    a1.set_title("(a) linear-bias regime (0.1 and 0.058 T)", fontsize=9.5)
    a1.legend(fontsize=6.5, loc="upper left")

    for s, c in (("A (M-P, 16)", RED), ("D (P-M-P, 16)", BLUE)):
        a2.semilogy(qd.BIAS_T, qd.Q_MEAS[s], "o-", color=c,
                    label=f"{s} measured")
        a2.semilogy(qd.BIAS_T, preds[s], "--", color=c, alpha=0.6,
                    label=f"{s} predicted")
    a2.axvspan(0.012, 0.04, color=GREY, alpha=0.12)
    a2.text(0.017, 800, "domain-wall\nexcess losses\n(χ = 30-47)",
            fontsize=7, color=GREY)
    a2.set_xlabel("bias field B$_{dc}$ [T]"); a2.set_ylabel("Q")
    a2.set_title("(b) bias dependence", fontsize=9.5)
    a2.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig("fig_en_disk.png", dpi=200)
    print("\nFigure : fig_en_disk.png")
    return preds


if __name__ == "__main__":
    main()
