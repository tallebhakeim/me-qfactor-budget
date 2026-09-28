# -*- coding: utf-8 -*-
"""
Provenance de zeng2015_curve.npz : coefficient ME résonant alpha(f) du
composite particulaire 0-3 PVDF/PZT/Terfenol-D (0,30/0,63/0,07, disque
Ø 15 mm x 1 mm, biais 1000 Oe, h_ac = 2 Oe) de Zeng et al., J. Alloys
Compd. 630 (2015) 183-188, figure 5 (enveloppe des six courbes H_i, bitmap
natif 990 x 810 de l'article : graphe principal 0-100 kHz et encart 80-94 kHz).
Le PDF n'est pas distribué : le .npz fait foi.
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PDF = Path.home() / ("Library/Mobile Documents/com~apple~CloudDocs/Professionel/"
                     "02_Recherche/RECHERCHE/thèse Louka/Terfenol-D en poudre.pdf")


def _env(mask, c0, c1, r0, r1, xlim, ylim):
    xs, ys = [], []
    for c in range(c0 + 2, c1 - 2):
        rr = np.where(mask[r0 + 2:r1 - 2, c])[0]
        if len(rr) == 0:
            continue
        xs.append(xlim[0] + (c - c0) / (c1 - c0) * (xlim[1] - xlim[0]))
        ys.append(ylim[1] - (rr.min() + 2) / (r1 - r0) * (ylim[1] - ylim[0]))
    return np.array(xs), np.array(ys)


def main():
    if not PDF.exists():
        print("PDF absent : zeng2015_curve.npz distribué fait foi")
        return
    import fitz
    d = fitz.open(str(PDF))
    pix = fitz.Pixmap(d, 13)                      # bitmap natif de la figure 5
    if pix.n >= 4:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)[..., :3].astype(int)
    H, W, _ = a.shape
    dark = a.sum(2) < 300
    rows = [r for r in range(H) if dark[r].mean() > 0.5]
    cols = [c for c in range(W) if dark[:, c].mean() > 0.5]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    ir = [r for r in range(r0 + 5, r1 - 5) if dark[r, c0 + 5:c1 - 5].mean() > 0.3]
    ic = [c for c in range(c0 + 5, c1 - 5) if dark[r0 + 5:r1 - 5, c].mean() > 0.3]
    ir0, ir1, ic0, ic1 = min(ir), max(ir), min(ic), max(ic)
    col = (np.abs(a[..., 0] - a[..., 1]) > 40) | (np.abs(a[..., 1] - a[..., 2]) > 40) \
        | (np.abs(a[..., 0] - a[..., 2]) > 40)
    m = col.copy()
    m[:r0 + 3] = False; m[r1 - 3:] = False; m[:, :c0 + 3] = False; m[:, c1 - 3:] = False
    mm = m.copy(); mm[ir0 - 5:ir1 + 5, ic0 - 5:ic1 + 5] = False
    f, al = _env(mm, c0, c1, r0, r1, (0, 100), (0, 90))
    mi = m.copy()
    mi[:ir0 + 2] = False; mi[ir1 - 2:] = False; mi[:, :ic0 + 2] = False; mi[:, ic1 - 2:] = False
    fi, ai = _env(mi, ic0, ic1, ir0, ir1, (80, 94), (0, 1))     # encart : échelle relative
    np.savez(HERE / "zeng2015_curve.npz", f_kHz=f, alpha=al, f_inset_kHz=fi,
             alpha_inset_rel=ai / ai.max())
    for nm, (x, y) in (("principal", (f, al)), ("encart", (fi, ai))):
        i = int(np.argmax(y)); h = y[i] / np.sqrt(2)
        lo = x[:i][y[:i] < h].max(); hi = x[i:][y[i:] < h].min()
        print(f"{nm}: pic @ {x[i]:.1f} kHz, -3 dB {hi - lo:.2f} kHz, Q = {x[i] / (hi - lo):.0f}")


if __name__ == "__main__":
    main()
