# -*- coding: utf-8 -*-
"""
Provenance de malleron_measured_curve.npz : fenêtre 60-80 kHz de la courbe
de résonance MESURÉE du bilame PZT-5H/Terfenol-D, digitalisée depuis la
figure 9 de Malleron et al., Microelectronics J. 88 (2019) (extraction
vectorielle PyMuPDF, cf. studies/me3d_structure/compare_malleron_curves.py),
remise à l'échelle du pic tabulé de la thèse (19,8 V/cm.Oe — la figure
publiée porte l'échelle décalée d'une décimale, cf. manuscrit §2).
Le Q de bande passante à -3 dB de cette courbe vaut 19,8.
"""
import numpy as np
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "me3d_structure" / \
    "kevin_digitized" / "measured_curves.npz"


def main():
    if not SRC.exists():
        print("source digitalisée absente (dépôt public) : le fichier "
              "malleron_measured_curve.npz distribué fait foi")
        return
    d = np.load(SRC)
    f, a = d["f_kHz"], d["alpha_f"] * 0.1     # échelle du pic tabulé (19,8)
    m = (f > 60) & (f < 80)
    fs, av = f[m], a[m]
    o = np.argsort(fs)
    fs, av = fs[o], av[o]
    i = int(np.argmax(av))
    half = av[i] / np.sqrt(2)
    idx = np.where(av > half)[0]
    Qbw = fs[i] / (fs[idx[-1]] - fs[idx[0]])
    np.savez(Path(__file__).parent / "malleron_measured_curve.npz",
             f_kHz=fs, alpha=av)
    print(f"pic {av[i]:.2f} V/cmOe @ {fs[i]:.2f} kHz ; "
          f"Q(-3 dB) = {Qbw:.1f} ; fichier régénéré")


if __name__ == "__main__":
    main()
