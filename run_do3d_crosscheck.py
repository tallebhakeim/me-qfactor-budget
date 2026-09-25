# -*- coding: utf-8 -*-
"""
Confrontation du budget avec le modèle 3D A-V-u ajusté (thèse T. A. Do 2019,
figures 3.13 et 3.19(c) digitalisées, cf. make_do2019_curves.py).

Fig. 3.13 (mêmes lames que le bilame de référence) : le modèle 3D, courants
de Foucault explicites + amortissement de Rayleigh ajusté sur l'admittance,
reproduit la HAUTEUR du pic mesuré mais pas sa LARGEUR : Q(-3 dB) ~ 39
contre ~ 18 mesuré. Le budget (Q = 16,4 [6,7 ; 23,8]) comble l'écart.
Fig. 3.19(c) (barreau 12x1x1 mm) : dans ce modèle 3D, les courants de
Foucault divisent l'amplitude par ~2 (écran sur l'excitation, déjà 0,58 à
42-50 kHz loin de la résonance) sans élargir la résonance : ils y agissent
en écran, pas en amortisseur. La lame Terfenol-D NUE mesurée à Q ~ 9
(run_multiscale) impose pourtant une dissipation magnétique ; l'équation (5)
du manuscrit (flux motionnel) en est le mécanisme.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def q_3db(f, y):
    i = int(np.argmax(y))
    h = y[i] / np.sqrt(2)
    lo = f[:i][y[:i] < h].max()
    hi = f[i:][y[i:] < h].min()
    return dict(f0=float(f[i]), peak=float(y[i]), df=float(hi - lo),
                Q=float(f[i] / (hi - lo)))


def q_lorentz(f, y, fwin):
    from scipy.optimize import curve_fit

    def lor(x, A, f0, Q, a, b):
        return A / np.sqrt(1 + (2 * Q * (x - f0) / f0)**2) + a + b * (x - f0)
    m = (f > fwin[0]) & (f < fwin[1])
    p, _ = curve_fit(lor, f[m], y[m], p0=[y.max(), f[np.argmax(y)], 30, 0, 0],
                     maxfev=20000)
    return dict(A=float(p[0]), f0=float(p[1]), Q=float(abs(p[2])))


def main(verbose=True):
    d = np.load(HERE / "do2019_fig313_curves.npz")
    m313 = q_3db(d["f_meas"], d["V_meas"])
    s313 = q_3db(d["f_3d"], d["V_3d"])
    out = dict(fig313=dict(meas=m313, model3d=s313,
                           peak_ratio=s313["peak"] / m313["peak"]))
    try:
        out["fig313"]["meas_lorentz"] = q_lorentz(d["f_meas"], d["V_meas"],
                                                  (62, 78))
        out["fig313"]["model3d_lorentz"] = q_lorentz(d["f_3d"], d["V_3d"],
                                                     (62, 78))
    except ImportError:
        pass

    e = np.load(HERE / "do2019_fig319c_curves.npz")
    w = q_3db(e["f_with"], e["a_with"])
    wo = q_3db(e["f_without"], e["a_without"])
    lowf = lambda f, y: float(y[(f > 42) & (f < 50)].mean())
    out["fig319c"] = dict(
        with_eddy=w, without_eddy=wo,
        peak_ratio=w["peak"] / wo["peak"],
        lowf_ratio=lowf(e["f_with"], e["a_with"])
        / lowf(e["f_without"], e["a_without"]))
    try:
        out["fig319c"]["with_lorentz"] = q_lorentz(e["f_with"], e["a_with"],
                                                   (70, 95))
        out["fig319c"]["without_lorentz"] = q_lorentz(
            e["f_without"], e["a_without"], (70, 95))
    except ImportError:
        pass

    # budget de référence pour la comparaison
    import qfactor_model as qm
    bud = {c: qm.q_budget(qm.REF, corner=c)
           for c in ("nom", "min_loss", "max_loss")}
    out["budget"] = dict(Q=bud["nom"]["Q"], Q_lo=bud["max_loss"]["Q"],
                         Q_hi=bud["min_loss"]["Q"],
                         invQ_eddy=float(bud["nom"]["inv"]["foucault"]),
                         invQ_total=float(bud["nom"]["inv"]["total"]))
    msf = HERE / "multiscale_results.json"
    if msf.exists():
        ms = json.load(open(msf))
        try:
            out["bare_td_Q_meas"] = 0.5 * (ms["td1"]["Q"] + ms["td4"]["Q"])
        except (KeyError, TypeError):
            pass

    if verbose:
        print("=== Fig. 3.13 (memes lames, tension a 1 Oe) ===")
        print(f"  mesure    : pic {m313['peak']:.3f} V @ {m313['f0']:.2f} kHz,"
              f" -3 dB {m313['df']:.2f} kHz, Q = {m313['Q']:.1f}")
        print(f"  modele 3D : pic {s313['peak']:.3f} V @ {s313['f0']:.2f} kHz,"
              f" -3 dB {s313['df']:.2f} kHz, Q = {s313['Q']:.1f}")
        print(f"  rapport des pics 3D/mesure = {out['fig313']['peak_ratio']:.3f}")
        if "meas_lorentz" in out["fig313"]:
            print(f"  Lorentz : Q mesure {out['fig313']['meas_lorentz']['Q']:.1f}"
                  f", Q 3D {out['fig313']['model3d_lorentz']['Q']:.1f}")
        print(f"  1/Q : mesure {1/m313['Q']:.4f}, 3D {1/s313['Q']:.4f}, "
              f"budget {out['budget']['invQ_total']:.4f} "
              f"(Foucault seul {out['budget']['invQ_eddy']:.4f})")
        print("=== Fig. 3.19(c) (barreau 12x1x1 mm, avec/sans Foucault) ===")
        print(f"  avec : pic {w['peak']:.2f} V/Oe, -3 dB {w['df']:.2f} kHz, "
              f"Q = {w['Q']:.1f} ; sans : pic {wo['peak']:.2f}, "
              f"-3 dB {wo['df']:.2f} kHz, Q = {wo['Q']:.1f}")
        print(f"  rapport d'amplitude a la resonance {out['fig319c']['peak_ratio']:.3f}"
              f" ; a 42-50 kHz (ecran seul) {out['fig319c']['lowf_ratio']:.3f}")
        if "with_lorentz" in out["fig319c"]:
            print(f"  Lorentz : Q avec {out['fig319c']['with_lorentz']['Q']:.1f}"
                  f", sans {out['fig319c']['without_lorentz']['Q']:.1f}")
        print(f"=== budget : Q = {out['budget']['Q']:.1f} "
              f"[{out['budget']['Q_lo']:.1f} ; {out['budget']['Q_hi']:.1f}]")
    return out


if __name__ == "__main__":
    main()
