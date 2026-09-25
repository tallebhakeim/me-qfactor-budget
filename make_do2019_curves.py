# -*- coding: utf-8 -*-
"""
Provenance de do2019_fig313_curves.npz et do2019_fig319c_curves.npz :
courbes digitalisées (raster 400 dpi, séparation par couleur, PyMuPDF) des
figures 3.13 et 3.19(c) de la thèse de T. A. Do (Sorbonne Université, 2019,
doi:10.70675/31d31b50z2e3dz41f7za1afzc7a0ca6faa6c).

Fig. 3.13 : mêmes lames que le bilame de référence (Terfenol-D 14x10x1 mm,
PZT 20x10x1 mm), tension à 1 Oe, 60-80 kHz : mesure (bleu) et modèle 3D
A-V-u (arêtes de Whitney, DEAM, courants de Foucault explicites,
amortissement de Rayleigh ajusté sur l'admittance mesurée) (orange).
Fig. 3.19(c) : barreau Terfenol-D/PZT-5A/Terfenol-D 12x1x1 mm, coefficient
ME (V/Oe) 40-100 kHz, modèle 3D avec (bleu) et sans (orange) courants de
Foucault.

Le PDF de la thèse n'est pas distribué : les .npz font foi dans le dépôt.
"""
import numpy as np
from pathlib import Path

PDF = Path.home() / ("Library/Mobile Documents/com~apple~CloudDocs/"
                     "Professionel/02_Recherche/RECHERCHE/Thèse Tuan Anh/"
                     "manuscrit de thèse/Manuscrit_Tuan_Anh_DO-1.pdf")
HERE = Path(__file__).resolve().parent
S = 72 / 110          # points par pixel du repérage à 110 dpi


def _render(doc, page, clip):
    import fitz
    pix = doc[page].get_pixmap(dpi=400, clip=fitz.Rect(*clip), alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3).astype(int)


def _frame(a, thr):
    dark = a.sum(2) < thr
    rows = np.where(dark.mean(1) > 0.5)[0]
    cols = np.where(dark.mean(0) > 0.5)[0]
    return rows.min(), rows.max(), cols.min(), cols.max()


def _curves(a, fr, xlim, ylim, legend):
    r0, r1, c0, c1 = fr
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    masks = dict(blue=(B > R + 40) & (R < 200),
                 orange=(R > B + 60) & (R > 150) & (B < 200))
    out = {}
    for name, m in masks.items():
        m = m.copy()
        m[:r0 + 3] = False; m[r1 - 3:] = False
        m[:, :c0 + 3] = False; m[:, c1 - 3:] = False
        lr0, lr1, lc0, lc1 = legend
        m[lr0:lr1, lc0:lc1] = False
        xs, ys = [], []
        for c in range(c0 + 3, c1 - 3):
            rr = np.where(m[:, c])[0]
            if len(rr) == 0:
                continue
            xs.append(xlim[0] + (c - c0) / (c1 - c0) * (xlim[1] - xlim[0]))
            ys.append(ylim[1] - (rr.mean() - r0) / (r1 - r0)
                      * (ylim[1] - ylim[0]))
        out[name] = (np.array(xs), np.array(ys))
    return out


def main():
    if not PDF.exists():
        print("PDF de la thèse absent : les .npz distribués font foi")
        return
    import fitz
    doc = fitz.open(str(PDF))
    # figure 3.13 (page 66 du PDF) : mesure = bleu, modèle 3D = orange
    a = _render(doc, 65, (255 * S, 140 * S, 670 * S, 480 * S))
    fr = _frame(a, 250)
    r0, r1, c0, c1 = fr
    leg = (r0, r0 + int(0.15 * (r1 - r0)), c0 + int(0.68 * (c1 - c0)), c1)
    cv = _curves(a, fr, (60, 80), (0, 2), leg)
    np.savez(HERE / "do2019_fig313_curves.npz",
             f_meas=cv["blue"][0], V_meas=cv["blue"][1],
             f_3d=cv["orange"][0], V_3d=cv["orange"][1])
    # figure 3.19(c) (page 72) : avec Foucault = bleu, sans = orange
    b = _render(doc, 71, (455 * S, 270 * S, 750 * S, 500 * S))
    fr = _frame(b, 450)
    r0, r1, c0, c1 = fr
    leg = (r0, r0 + int(0.22 * (r1 - r0)), c0, c0 + int(0.52 * (c1 - c0)))
    cv = _curves(b, fr, (40, 100), (0, 20), leg)
    np.savez(HERE / "do2019_fig319c_curves.npz",
             f_with=cv["blue"][0], a_with=cv["blue"][1],
             f_without=cv["orange"][0], a_without=cv["orange"][1])
    print("2 fichiers .npz régénérés")


if __name__ == "__main__":
    main()
