# -*- coding: utf-8 -*-
"""
Rapport + figure du TRANSFERT DISQUE : le budget (matériaux seuls, aucun
recalage, même précontrainte que les barreaux) confronté aux 36 Q de la thèse
Rizzo (C2N, 2020) RENDUS COHÉRENTS (Q_m = Q_tab·phi_p², cf. qfactor_disk) et
aux trois courbes de résonance excitées magnétiquement (bench_disk_curves).
Produit fig_en_disk.png (4 panneaux) et le tableau f_s / Q.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qfactor_disk as qd

BLUE, RED, GREY = "#4878a8", "#c1272d", "#666666"


def main():
    print("=" * 72)
    print("TRANSFERT DISQUE — 6 échantillons Rizzo (C2N), 36 Q (cohérents)")
    print(f"identité Q_tab = sqrt(L/C)/(R phi²) : écart max "
          f"{qd.rizzo_q_identity()*100:.1f} % sur {len(qd.RIZZO_ROWS)} lignes")
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
        print(f"{'':16}" + "".join(f" {m:8.0f}" for m in qd.Q_MEAS[s]) + "  Q_m")
        print(f"{'':16}" + "".join(f" {m:8.0f}" for m in qd.Q_TAB[s]) + "  (publié)")
    # résidu = facteur de perte Terfenol manquant, par unité de fraction
    # d'énergie de la couche
    print("\nrésidu (1/Q_m - 1/Q_pred)/f_terf :")
    for s in qd.SAMPLES:
        md = qd.radial_mode(s)
        ft = md["W"].get("terf", 0.0) / md["Wtot"]
        eta = [(1 / qd.Q_MEAS[s][j] - 1 / preds[s][j]) / ft for j in range(6)]
        print(f"{s:16} f_terf={ft:.2f} " + " ".join(f"{e:6.3f}" for e in eta))
    print("""
Lecture : à 0,1 T (Terfenol au point de fonctionnement des barreaux,
chi_int ~ 5) les Q prédits sont à un facteur 1,0-2,2 des Q_m cohérents pour
les disques de 16 mm (2,2-3,9 pour 10 mm), ordre D > C > A respecté. Quand le
biais baisse, le budget MONTE (d33m² chute) alors que la mesure reste plate :
le résidu est un facteur de perte de la couche Terfenol de 0,03-0,06,
indépendant du biais et de l'amplitude, absent du budget.""")

    # ------- figure 2x2 : (a) schéma, (b) scatter, (c) Q(B), (d) courbe mesurée
    from matplotlib.patches import Rectangle, FancyArrowPatch
    fig, ((a0, a1), (a2, a3)) = plt.subplots(2, 2, figsize=(9.8, 7.0))

    # (a) schéma du disque tricouche M-P-M (échantillon C)
    a0.add_patch(Rectangle((0, 0.0), 16, 1.0, fc="#9fb4cc", ec="k", lw=0.8))
    a0.add_patch(Rectangle((0, 1.0), 16, 2.0, fc="#f0c674", ec="k", lw=0.8))
    a0.add_patch(Rectangle((0, 3.0), 16, 1.0, fc="#9fb4cc", ec="k", lw=0.8))
    a0.text(8, 3.5, "Terfenol-D (1 mm)", ha="center", va="center", fontsize=8)
    a0.text(8, 2.0, "PIC181 (2 mm), radial mode", ha="center", va="center",
            fontsize=8)
    a0.text(8, 0.5, "Terfenol-D (1 mm)", ha="center", va="center", fontsize=8)
    a0.annotate("", xy=(16, -0.7), xytext=(0, -0.7),
                arrowprops=dict(arrowstyle="<->", lw=0.8))
    a0.text(8, -1.25, "Ø 16 mm", ha="center", fontsize=8)
    a0.add_patch(FancyArrowPatch((17.0, 2.0), (20.5, 2.0),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color=RED))
    a0.text(18.7, 2.5, "B$_{dc}$ + b$_{ac}$", color=RED, ha="center",
            fontsize=8.5)
    a0.set_xlim(-1.5, 22); a0.set_ylim(-1.9, 4.7)
    a0.axis("off")
    a0.set_title("(a) sample C: M-P-M disk (thesis Table 4.5)", fontsize=9)
    marks = dict(zip(qd.SAMPLES, "osD^vP"))
    for s in qd.SAMPLES:
        for j in (0, 1):                      # 0.1 et 0.058 T
            a1.plot(qd.Q_MEAS[s][j], preds[s][j], marks[s], color=BLUE,
                    ms=6, label=s if j == 0 else None)
    lim = [0, 500]
    a1.plot(lim, lim, "-", color=GREY, lw=1)
    a1.fill_between(lim, [l / 2 for l in lim], [l * 2 for l in lim],
                    color=GREY, alpha=0.15, label="factor-2 band")
    a1.set_xlim(lim); a1.set_ylim(lim)
    a1.set_xlabel("measured Q$_m$ (identified, consistent definition)")
    a1.set_ylabel("predicted Q")
    a1.set_title("(b) linear-bias regime (0.1 and 0.058 T)", fontsize=9)
    a1.legend(fontsize=6.5, loc="upper left")

    for s, c in (("A (M-P, 16)", RED), ("D (P-M-P, 16)", BLUE)):
        a2.semilogy(qd.BIAS_T, qd.Q_MEAS[s], "o-", color=c,
                    label=f"{s} measured Q$_m$")
        a2.semilogy(qd.BIAS_T, qd.Q_TAB[s], "o:", color=c, alpha=0.4, ms=4,
                    label=f"{s} as tabulated (÷φ$_p^2$)")
        a2.semilogy(qd.BIAS_T, preds[s], "--", color=c, alpha=0.6,
                    label=f"{s} predicted")
    a2.text(0.013, 900, "budget rises as d$_{33,m}^2$ falls;\nmeasured Q flat:\nbias-independent Terfenol loss",
            fontsize=7, color=GREY)
    a2.set_xlabel("bias field B$_{dc}$ [T]"); a2.set_ylabel("Q")
    a2.set_title("(c) bias dependence", fontsize=9)
    a2.legend(fontsize=5.5, ncol=2, loc="lower right")
    # (d) résonances EXCITÉES MAGNÉTIQUEMENT (bench_disk_curves.npz) :
    #     largeurs de bande contre Q_m cohérent et Q publié
    bc = np.load("bench_disk_curves.npz")

    def q3(f, v):
        i = int(np.argmax(v)); h = v[i] / np.sqrt(2)
        lo = f[:i][v[:i] < h].max(); hi = f[i:][v[i:] < h].min()
        return f[i], v[i], f[i] / (hi - lo)
    curves = [("geeps_bilayer", "GeePs 2023, M-P disk, 1 Oe, 0.07 T", RED),
              ("geeps_trilayer", "GeePs 2023, sample C, 1 Oe", BLUE),
              ("rizzo_A", "Rizzo fig. 0.3, sample A, 8 Oe, 0.1 T", GREY)]
    for key, lab, c in curves:
        f, v = bc[key + "_f"], bc[key + "_V"]
        f0, vp, q = q3(f, v)
        a3.plot(f - f0, v / vp, ".", color=c, ms=2.2, alpha=0.7,
                label=f"{lab}: Q = {q:.0f}")
    fl = np.linspace(-8, 8, 400)
    a3.plot(fl, 1 / np.sqrt(1 + (2 * 61 * fl / 142.5)**2), "-", color="k",
            lw=1.0, label="Lorentzian, consistent Q$_m$ = 61 (A, 0.1 T)")
    a3.plot(fl, 1 / np.sqrt(1 + (2 * 117 * fl / 142.5)**2), "--", color="k",
            lw=0.9, label="Lorentzian, tabulated Q = 117 (A, 0.1 T)")
    a3.set_xlim(-8, 8); a3.set_ylim(0, 1.12)
    a3.set_xlabel("f − f$_p$ [kHz]"); a3.set_ylabel("normalized voltage")
    a3.set_title("(d) magnetically driven bandwidths vs identified Q",
                 fontsize=9)
    a3.legend(fontsize=6, loc="upper left")

    fig.tight_layout()
    fig.savefig("fig_en_disk.png", dpi=200)
    print("\nFigure : fig_en_disk.png")
    return preds


if __name__ == "__main__":
    main()
