# -*- coding: utf-8 -*-
"""
Provenance de rizzo_measured_curve.npz : courbe supérieure (500 kOhm) de la
figure 2.19 de la thèse G. Rizzo (V_RMS(f) de l'échantillon C, tricouche
M-P-M Ø16, biais 0,1 T, drive 0,8 mT, 20 charges de 10 Ω à 500 kΩ).
Au voisinage du pic principal, l'enveloppe est la courbe 500 kΩ (la plus
proche du circuit ouvert, V_max = 35,7 V d'après le texte de la thèse).

Extraction : image MATLAB embarquée dans le PDF (page 120), cadre d'axes
détecté par les lignes noires, étalonnage cadre = [100;200] kHz x [0;40] V,
enveloppe = pixel coloré le plus haut par colonne.
"""
import numpy as np
from pathlib import Path

PDF = Path.home() / "Downloads" / "90503_RIZZO_2020_archivage.pdf"
OUT = Path(__file__).parent / "rizzo_measured_curve.npz"


def main():
    try:
        import fitz
    except ImportError:
        print("PyMuPDF absent : le fichier rizzo_measured_curve.npz "
              "distribué fait foi")
        return
    if not PDF.exists():
        print("PDF de la thèse absent (dépôt public) : le fichier "
              "rizzo_measured_curve.npz distribué fait foi")
        return
    d = fitz.open(PDF)
    page = d[119]                                    # page 120 : figure 2.19
    # la figure = la plus GRANDE image embarquée de la page
    best = max(page.get_images(),
               key=lambda im: fitz.Pixmap(d, im[0]).w
               * fitz.Pixmap(d, im[0]).h)
    pix = fitz.Pixmap(d, best[0])
    if pix.n > 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
    rgb = img[:, :, :3].astype(int)

    # cadre : lignes/colonnes sombres continues (frame MATLAB anticrénelé)
    dark = rgb.sum(axis=2) < 400
    rows = np.where(dark.sum(axis=1) > 0.7 * pix.w)[0]
    cols = np.where(dark.sum(axis=0) > 0.7 * pix.h)[0]
    top, bot = rows.min(), rows.max()
    left, right = cols.min(), cols.max()

    # pixels de courbe : teinte même pâle (le cyan 500 kOhm est à 47 du
    # blanc), en excluant fond blanc et cadre sombre. Le pixel le plus HAUT
    # par colonne = la courbe 500 kOhm (V croît avec la charge à f donnée).
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    s = rgb.sum(axis=2)
    colored = (mx - mn > 15) & (s < 735) & (s > 150)
    env_f, env_v = [], []
    for c in range(left + 2, right - 1):
        ys = np.where(colored[top + 2:bot - 1, c])[0]
        if len(ys):
            y = top + 2 + ys.min()                   # pixel le plus HAUT
            f = 100.0 + (c - left) / (right - left) * 100.0     # kHz
            v = 40.0 * (bot - y) / (bot - top)                  # V
            if 105.0 <= f <= 178.0:      # évite la boîte de légende (f>180)
                env_f.append(f)
                env_v.append(v)
    env_f, env_v = np.array(env_f), np.array(env_v)
    i = int(np.argmax(env_v))
    np.savez(OUT, f_kHz=env_f, V_rms=env_v)
    print(f"{len(env_f)} points ; pic {env_v[i]:.1f} V @ {env_f[i]:.1f} kHz "
          f"(thèse : 35,7 V) -> {OUT.name}")


if __name__ == "__main__":
    main()
