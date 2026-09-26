# -*- coding: utf-8 -*-
"""
Provenance de bench_disk_curves.npz : trois courbes de résonance de disques
ME du C2N (Terfenol-D/PIC181, échantillons de la thèse Rizzo 2020) EXCITÉS
MAGNÉTIQUEMENT et lus en tension quasi circuit ouvert :

  geeps_bilayer : banc GeePs (19/04/2023, fichier .mat, pas 0,1 kHz),
                  bicouche M-P, h_ac = 1 Oe, biais 706 Oe appliqué en plan
                  (V en V/Oe, RMS) ;
  geeps_trilayer: banc GeePs, trilame C (R = 8 mm, t_m = 1, t_p = 2 mm),
                  h_ac = 1 Oe, digitalisée (raster 400 dpi) de la figure 4.10
                  de la thèse Karimi 2024 (alpha_V en V/Oe) ;
  rizzo_A       : figure 0.3 (annexe III) de la thèse Rizzo 2020, échantillon
                  A (bicouche M-P, 16 mm), 0,1 T, 0,8 mT (8 Oe), enveloppe
                  (charge 300 kOhm, la plus proche du circuit ouvert), V_RMS.

Les sources (fichiers .mat, PDF des thèses) ne sont pas distribuées : le
fichier .npz fait foi dans le dépôt public.
"""
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
ICLOUD = Path.home() / ("Library/Mobile Documents/com~apple~CloudDocs/"
                        "Professionel/02_Recherche/RECHERCHE")
MAT = ICLOUD / ("Thèse de Sheno/mesures_C2N_samples/La premiere mesure disk "
                "bicouche No1/Sample_1_disk_19042023_1753_VfctFreq_Hac1Oe_"
                "HdcOpt706Oe.mat")
PDF_KARIMI = ICLOUD / "Thèse de Sheno/Manuscrit_Sheno.pdf"
PDF_RIZZO = Path.home() / "Downloads/90503_RIZZO_2020_archivage.pdf"


def _render(pdf, page0, clip_pt, dpi=400):
    import fitz
    doc = fitz.open(str(pdf))
    pix = doc[page0].get_pixmap(dpi=dpi, clip=fitz.Rect(*clip_pt), alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3).astype(int)


def _frame(a, thr, frac):
    dark = a.sum(2) < thr
    rows = np.where(dark.mean(1) > frac)[0]
    cols = np.where(dark.mean(0) > frac)[0]
    return rows.min(), rows.max(), cols.min(), cols.max()


def _trace(mask, fr, xlim, ylim, top=False):
    r0, r1, c0, c1 = fr
    xs, ys = [], []
    for c in range(c0 + 3, c1 - 3):
        rr = np.where(mask[:, c])[0]
        if len(rr) == 0:
            continue
        rv = rr.min() if top else rr.mean()
        xs.append(xlim[0] + (c - c0) / (c1 - c0) * (xlim[1] - xlim[0]))
        ys.append(ylim[1] - (rv - r0) / (r1 - r0) * (ylim[1] - ylim[0]))
    return np.array(xs), np.array(ys)


def main():
    out = {}
    if MAT.exists():
        import scipy.io as sio
        m = sio.loadmat(str(MAT))
        out["geeps_bilayer_f"] = m["FreqSweep"].ravel()
        out["geeps_bilayer_V"] = m["Vout_res"].ravel()
    if PDF_KARIMI.exists():
        s = 0.72                       # points par pixel du repérage à 100 dpi
        a = _render(PDF_KARIMI, 87, (150 * s, 105 * s, 535 * s, 375 * s))
        fr = _frame(a, 600, 0.4)
        r0, r1, c0, c1 = fr
        R, G, B = a[..., 0], a[..., 1], a[..., 2]
        red = (R > 140) & (G < 120) & (B < 140) & (R - G > 60)
        red[:r0 + 3] = False; red[r1 - 3:] = False
        red[:, :c0 + 3] = False; red[:, c1 - 3:] = False
        red[r0:r0 + int(0.2 * (r1 - r0)), c0 + int(0.55 * (c1 - c0)):] = False
        f, v = _trace(red, fr, (100, 200), (0, 3))
        out["geeps_trilayer_f"], out["geeps_trilayer_V"] = f, v
    if PDF_RIZZO.exists():
        s = 72 / 80
        a = _render(PDF_RIZZO, 170, (195 * s, 80 * s, 535 * s, 345 * s))
        fr = _frame(a, 450, 0.5)
        r0, r1, c0, c1 = fr
        R, G, B = a[..., 0], a[..., 1], a[..., 2]
        col = (np.abs(R - G) > 25) | (np.abs(G - B) > 25) | (np.abs(R - B) > 25)
        col[:r0 + 3] = False; col[r1 - 3:] = False
        col[:, :c0 + 3] = False; col[:, c1 - 3:] = False
        col[r0:r0 + int(0.85 * (r1 - r0)), c0 + int(0.72 * (c1 - c0)):] = False
        f, v = _trace(col, fr, (120, 160), (0, 16), top=True)
        out["rizzo_A_f"], out["rizzo_A_V"] = f, v
    if not out:
        print("sources absentes : le fichier bench_disk_curves.npz distribué "
              "fait foi")
        return
    np.savez(HERE / "bench_disk_curves.npz", **out)
    for k in ("geeps_bilayer", "geeps_trilayer", "rizzo_A"):
        if k + "_f" in out:
            f, v = out[k + "_f"], out[k + "_V"]
            i = int(np.argmax(v)); h = v[i] / np.sqrt(2)
            lo = f[:i][v[:i] < h].max(); hi = f[i:][v[i:] < h].min()
            print(f"{k:15s} pic {v[i]:.2f} @ {f[i]:.2f} kHz ; "
                  f"Q(-3 dB) = {f[i] / (hi - lo):.1f}")


if __name__ == "__main__":
    main()
