# -*- coding: utf-8 -*-
"""
v17 (21/10/2026) : trois compléments demandés par la relecture au filtre des
retours SMS-120911 (points 2, 3 et 4 de REVUE_RETOURS_21-10-2026.md).

(A) Convergence en maillage du FEM 2-D de la section 5.5 sur la RÉPONSE
    COUPLÉE (alpha_res, f_r, Q à -3 dB), pertes par couche prédites par le
    budget, moduli documentés (s = 1) et raidis (s = 1.24).
(B) Sensibilité du budget au facteur démagnétisant : N = 0.0724 (prisme
    uniformément aimanté, Aharoni) contre N_eff = 0.053 (démagnétisation 3-D
    non linéaire de la même plaque, article compagnon SMS-120911 R2).
(C) Résolution des courbes numérisées : pas d'échantillonnage et nombre de
    points dans la bande -3 dB de chaque courbe utilisée (sections 5.5, 5.7,
    6.1), pour dire ce que vaut chaque largeur lue.

Sortie : conv_fem2d.json (lu par build_manuscript.js) + impression.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "core"))
sys.path.insert(0, str(HERE))
from me import me_fem as fe          # noqa: E402
from me import me_deam as dm         # noqa: E402
import qfactor_model as qm           # noqa: E402
import run_fem2d_crosscheck as rc    # noqa: E402

OUT = {}


def q_3db(f, a):
    i = int(np.argmax(a)); h = a[i] / np.sqrt(2)
    lo = i
    while lo > 0 and a[lo] > h:
        lo -= 1
    hi = i
    while hi < len(a) - 1 and a[hi] > h:
        hi += 1
    # interpolation linéaire des deux croisements
    def x(j, k):
        return f[j] + (h - a[j]) * (f[k] - f[j]) / (a[k] - a[j])
    f1 = x(lo, lo + 1) if lo < i else f[lo]
    f2 = x(hi, hi - 1) if hi > i else f[hi]
    return f[i] / (f2 - f1), f[i], a[i], (f2 - f1)


# ------------------------------------------------------------------ (A)
def conv_mesh():
    coeffs = dm.deam_coeffs("Terfenol-D", 525.0, rc.SIG_PRESTRESS)
    bud = qm.q_budget("PZT/Terf", nx=200, H_ac_oe=1.0)
    res, inv = bud["res"], bud["inv"]
    eta_pzt = (inv["pzt_meca"] + inv["pzt_diel"]) * res["Wtot"] / res["Wp"]
    eta_terf = (inv["mag_hyst"] + inv["foucault"]) * res["Wtot"] / res["Wm"][0]
    rows = []
    for s in (1.0, 1.24):
        rc.set_epoxy_stiffening(s)
        for nx in (10, 20, 40, 80, 160):
            t0 = time.time()
            asm = fe._assemble_rect(rc.LP, rc.W, rc.layers(), mode="LT",
                                    nx=nx, mag_coeffs=coeffs)
            # balayage grossier puis fin autour du pic
            f = np.linspace(55e3, 80e3, 251)
            h = fe._harmonic_from_asm(asm, f, H_oe=1.0, eddy=False,
                                      layer_eta=[eta_terf, eta_pzt])
            fr0 = h["f_r"]
            f2 = np.linspace(fr0 - 6e3, fr0 + 6e3, 601)
            h2 = fe._harmonic_from_asm(asm, f2, H_oe=1.0, eddy=False,
                                       layer_eta=[eta_terf, eta_pzt])
            Q, fr, ar, bw = q_3db(f2, h2["alpha"])
            rows.append(dict(s=s, nx=nx, n_nodes=int(asm["n"]),
                             n_tri=int(len(asm["tris"])), f_r_kHz=fr / 1e3,
                             alpha_res=float(ar), Q_3dB=float(Q),
                             bw_kHz=bw / 1e3, cpu_s=time.time() - t0))
            print(f"s={s:.2f} nx={nx:4d} nodes={asm['n']:6d} tri={len(asm['tris']):6d} "
                  f"f_r={fr/1e3:7.3f} kHz alpha={ar:6.2f} Q={Q:6.2f} "
                  f"({time.time()-t0:.1f} s)")
        rc.set_epoxy_stiffening(1.0)
    # écarts par rapport au maillage le plus fin, par s
    for s in (1.0, 1.24):
        rr = [r for r in rows if r["s"] == s]; ref = rr[-1]
        for r in rr:
            r["d_alpha_pct"] = (r["alpha_res"] / ref["alpha_res"] - 1) * 100
            r["d_fr_pct"] = (r["f_r_kHz"] / ref["f_r_kHz"] - 1) * 100
            r["d_Q_pct"] = (r["Q_3dB"] / ref["Q_3dB"] - 1) * 100
    OUT["mesh"] = dict(eta_pzt=eta_pzt, eta_terf=eta_terf, rows=rows)


# ------------------------------------------------------------------ (B)
def sens_N():
    out = {}
    orig = qm.lam_N
    for label, N in (("aharoni", None), ("n3d", 0.053)):
        if N is None:
            qm.lam_N = orig
        else:
            def lam_fixed(t, chi, N=N):
                return (1.0 - N) / (1.0 + chi * N), N
            qm.lam_N = lam_fixed
        qm._LAM_CACHE.clear()
        b = qm.q_budget("PZT/Terf", nx=200, H_ac_oe=1.0)
        lam, Nv = qm.lam_N(1e-3, 7.4)
        out[label] = dict(N=Nv, lam=lam, Q=b["Q"],
                          inv={k: float(v) for k, v in b["inv"].items()})
        print(f"{label}: N={Nv:.4f} lam={lam:.3f} Q={b['Q']:.2f} "
              f"foucault 1/Q={b['inv']['foucault']:.4f}")
    qm.lam_N = orig; qm._LAM_CACHE.clear()
    OUT["N_sens"] = out


# ------------------------------------------------------------------ (C)
def digit_res():
    out = {}
    def rep(name, f, a, note=""):
        f = np.asarray(f, float); a = np.asarray(a, float)
        o = np.argsort(f); f, a = f[o], a[o]
        Q, fr, ar, bw = q_3db(f, a)
        step = float(np.median(np.diff(f)))
        npts = int(np.sum((f > fr - bw / 2) & (f < fr + bw / 2)))
        out[name] = dict(step_kHz=step, bw_kHz=bw, n_in_band=npts, Q=Q,
                         f_r_kHz=fr, note=note)
        print(f"{name:22s} step={step:.3f} kHz  bw={bw:.2f} kHz  "
              f"pts in band={npts:4d}  Q={Q:.1f}  {note}")
    d = np.load(HERE / "malleron_measured_curve.npz")
    rep("malleron_fig9", d["f_kHz"], d["alpha"], "fig. 9 of [20], 60-80 kHz window")
    d = np.load(HERE / "do2019_fig313_curves.npz")
    rep("do2019_meas", d["f_meas"] / 1e3, d["V_meas"], "fig. 3.13 of [21], measured")
    rep("do2019_3d", d["f_3d"] / 1e3, d["V_3d"], "fig. 3.13 of [21], 3-D model")
    d = np.load(HERE / "bench_disk_curves.npz")
    rep("rizzo_A_fig03", d["rizzo_A_f"], d["rizzo_A_V"], "fig. 0.3 of [41]")
    rep("geeps_bilayer_2023", d["geeps_bilayer_f"], d["geeps_bilayer_V"], "bench file, 0.1 kHz step")
    rep("geeps_trilayer_2023", d["geeps_trilayer_f"], d["geeps_trilayer_V"], "bench file, 0.1 kHz step")
    d = np.load(HERE / "zeng2015_curve.npz")
    rep("zeng2015_fig5", d["f_kHz"], d["alpha"], "fig. 5 of [46], main plot")
    OUT["digitized"] = out


if __name__ == "__main__":
    digit_res()
    sens_N()
    conv_mesh()
    (HERE / "conv_fem2d.json").write_text(json.dumps(OUT, indent=1))
    print("-> conv_fem2d.json")
