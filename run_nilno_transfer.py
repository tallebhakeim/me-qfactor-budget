# -*- coding: utf-8 -*-
"""
TRANSFERT à une troisième famille : films Ni pulvérisés (50 µm) sur LiNbO3
monocristallin (0,5 mm), plaques 10 x 10 mm auto-polarisées, SANS adhésif
(dispositifs BIO68/69/70, coupes 0°Y/36°Y/128°Y, GeePs 2026, cf.
make_nilno_curves.py). Le budget se réduit à la couche de Ni :
  - Foucault (éq. 5, plaque 1D, t/δ = 0,7-1,6) avec les propriétés
    incrémentales MESURÉES du Ni (d33m, mu_r) au biais (tables VSM/jauge,
    M. Touati, GeePs) ;
  - hystérésis magnétomécanique : boucle mineure calculée par un modèle de
    Jiles-Atherton calibré une fois sur le VSM du tricouche BIO69 (paramètres
    NI_JA de MEGA, core/me/me_hysteresis.py), balayage ΔH_eq piloté par la
    contrainte résonante (auto-cohérent en Q) ;
  - LiNbO3 : plancher Q_m >= 3000 (monocristal), tan δ <= 1e-3 : négligeable.
Cibles : Q_m du fit BVD de |Z| (cohérent : L, C, R même côté) et Q(-3 dB) de
V(f) au biais optimal et à champ nul (rémanence).
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MU0 = 4e-7 * np.pi
_TRAPZ = getattr(np, "trapezoid", np.trapz)
OE = 1e3 / (4 * np.pi)
SIG_NI, E_NI, RHO_NI = 1.4e7, 200e9, 8900.0
E_LNO, RHO_LNO = 170e9, 4640.0
T_NI, T_LNO, L, W = 50e-6, 0.5e-3, 10e-3, 10e-3
Q_LNO = (1e4, 3e3, 3e4)                 # (nom, plancher, plafond)
TAND_LNO = (3e-4, 1e-3, 1e-4)
# propriétés incrémentales MESURÉES du Ni pulvérisé au biais (d33m nm/A, mu_r),
# branches montante/descendante, sigma = 0 (tables VSM/jauge, G(H, sigma)) ;
# nominal = branche montante ; intervalle = les deux branches et sigma ±50 MPa
NI_MEAS = {
    0:  dict(up=(1.91, 30.4), down=(3.10, 53.7), span=((1.80, 29.5), (3.29, 55.2))),
    17: dict(up=(2.12, 59.2), down=(1.95, 24.1), span=((1.84, 23.4), (2.25, 61.0))),
    19: dict(up=(2.02, 63.7), down=(1.84, 22.4), span=((1.74, 21.7), (2.15, 65.6))),
    53: dict(up=(2.20, 55.0), down=(1.30, 14.0), span=((1.20, 13.0), (2.40, 58.0))),
    61: dict(up=(2.58, 42.1), down=(1.11, 11.2), span=((1.05, 10.9), (2.74, 43.3))),
}
# Jiles-Atherton du Ni de dispositif, calibré sur le VSM BIO69 (|d33|pic 4 nm/A,
# Hc 3,8 Oe) : MEGA core/me/me_hysteresis.NI_JA
NI_JA = dict(Ms=4.85e5, a=2.6e3, alpha_JA=1.0e-3, k=3.0e2, c=0.15, lam_s=-34e-6)
DEVICES = {
    "BIO68 (0°Y)":   dict(f=266.7e3, Hopt=19, Q_bvd=555, k2=0.016,
                          Q_V={19: 299, 0: 384}),
    "BIO69 (36°Y)":  dict(f=324.0e3, Hopt=61, Q_bvd=544, k2=0.052,
                          Q_V={61: 665, 53: 531, 0: 831}),
    "BIO70 (128°Y)": dict(f=252.8e3, Hopt=17, Q_bvd=422, k2=0.040,
                          Q_V={17: 258, 0: 343}),
}


def _man_chi(He, p):
    x = He / p["a"]; xs = np.where(np.abs(x) < 1e-6, 1e-6, x)
    Lg = 1.0 / np.tanh(xs) - 1.0 / xs
    dL = 1.0 - (1.0 / np.tanh(xs))**2 + 1.0 / xs**2
    return p["Ms"] * Lg, (p["Ms"] / p["a"]) * dL


def _ja_step(H_prev, M_prev, dH, p):
    delta = 1.0 if dH >= 0 else -1.0
    He = H_prev + p["alpha_JA"] * M_prev
    Man, chi = _man_chi(He, p)
    dMirr = (Man - M_prev) / (delta * p["k"] - p["alpha_JA"] * (Man - M_prev))
    num = (1 - p["c"]) * dMirr + p["c"] * chi
    den = 1.0 - p["alpha_JA"] * p["c"] * chi
    slope = num / den
    if delta * (Man - M_prev) < 0:
        slope = p["c"] * chi / den
    return M_prev + slope * dH


def ja_minor_loop(H_dc_oe, dH_oe, p=NI_JA, Hmax_oe=500.0, n=3000, ncyc=6):
    """Énergie dissipée par cycle W = ∮ mu0 M dH (J/m^3) d'une boucle mineure
    H_dc ± dH atteinte depuis la saturation par la branche DESCENDANTE
    (protocole aimants approchés / rémanence), et pente moyenne mu_r du cycle."""
    Hmax = Hmax_oe * OE
    H = 0.0; M = 0.0
    for h in np.linspace(0, Hmax, n)[1:]:               # montée à +Hmax
        M = _ja_step(H, M, h - H, p); H = h
    Hdc = H_dc_oe * OE
    for h in np.linspace(Hmax, Hdc, n)[1:]:             # descente au biais
        M = _ja_step(H, M, h - H, p); H = h
    dH = max(dH_oe, 1e-3) * OE
    seg = np.linspace(0, dH, 200)
    cyc = np.concatenate([Hdc + seg[1:], Hdc + seg[::-1][1:], Hdc - seg[1:],
                          Hdc - seg[::-1][1:]])
    Wc, mu = 0.0, 1.0
    for c in range(ncyc):
        Hs, Ms = [H], [M]
        for h in cyc:
            M = _ja_step(H, M, h - H, p); H = h
            Hs.append(H); Ms.append(M)
        Hs, Ms = np.array(Hs), np.array(Ms)
        Wc = abs(MU0 * _TRAPZ(Ms, Hs))                   # aire de la boucle
        mu = 1.0 + (Ms.max() - Ms.min()) / (Hs.max() - Hs.min())
    return Wc, mu


def eddy_eta(f, mu_r, d33m, lam=1.0, t=T_NI):
    """Facteur de perte Foucault de la couche (forme fermée = éq. 5 intégrée)."""
    w = 2 * np.pi * f; mu = mu_r * MU0
    delta = np.sqrt(2 / (w * mu * SIG_NI))
    x = (1 + 1j) * t / (2 * delta); F = np.tanh(x) / x
    k2 = lam**2 * d33m**2 * E_NI / mu; r = k2 / (1 - k2); one = 1 - F
    return float((r * one.imag) / (1 + r * one.real)), float(delta), float(k2)


def budget(dev, H_oe, corner="nom", h_ac_oe=1.0, N=0.0045):
    d = DEVICES[dev]; f = d["f"]
    fNi = 2 * E_NI * T_NI / (2 * E_NI * T_NI + E_LNO * T_LNO)
    fL = 1 - fNi
    m = NI_MEAS[H_oe]
    if corner == "nom":
        d33, mu_r = m["up"]
    elif corner == "min_loss":
        d33, mu_r = min(m["span"][0][0], m["span"][1][0]), max(m["span"][0][1], m["span"][1][1])
    else:
        d33, mu_r = max(m["span"][0][0], m["span"][1][0]), min(m["span"][0][1], m["span"][1][1])
    d33 *= 1e-9
    chi = mu_r - 1.0
    lam = (1 - N) / (1 + chi * N)
    eta_e, delta, k2 = eddy_eta(f, mu_r, d33, lam)
    inv_e = fNi * eta_e
    ql = {"nom": Q_LNO[0], "min_loss": Q_LNO[2], "max_loss": Q_LNO[1]}[corner]
    td = {"nom": TAND_LNO[0], "min_loss": TAND_LNO[2], "max_loss": TAND_LNO[1]}[corner]
    inv_l = fL / ql + td * d["k2"]
    # hystérésis auto-cohérente : T = E Q d33 h_int ; dH_eq = d33 T/(mu0 chi (1+chi N))
    h_int = h_ac_oe * OE / (1 + chi * N)
    Q = 500.0
    for it in range(60):
        T = E_NI * Q * d33 * h_int
        dH_eq = d33 * T / (MU0 * chi * (1 + chi * N))
        Wc, _ = ja_minor_loop(H_oe, dH_eq / OE)
        eta_h = Wc * E_NI / (np.pi * T**2)
        inv_h = fNi * eta_h
        Qn = 1.0 / (inv_e + inv_l + inv_h)
        if abs(Qn - Q) < 1e-3 * Q:
            break
        Q = 0.5 * (Q + Qn)
    return dict(dev=dev, H=H_oe, corner=corner, Q=Qn, inv_eddy=inv_e,
                inv_hyst=inv_h, inv_lno=inv_l, eta_e=eta_e, eta_h=eta_h,
                d33=d33, mu_r=mu_r, delta=delta, t_over_delta=T_NI / delta,
                k2=k2, T=T, dH_eq_oe=dH_eq / OE, fNi=fNi, iters=it + 1)


def figure(res):
    """fig_en_nilno.png : (a) V(f) mesurées au biais optimal, normalisées et
    centrées, avec la lorentzienne au Q nominal prédit ; (b) budget 1/Q par
    dispositif et biais (barres empilées) contre les 1/Q mesurés."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, RED, GREY, GREEN = "#4878a8", "#c1272d", "#666666", "#3a8f4a"
    nz = np.load(HERE / "nilno_devices.npz")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.8, 3.6))
    cols = {"BIO68 (0°Y)": BLUE, "BIO69 (36°Y)": RED, "BIO70 (128°Y)": GREEN}
    for dev, d in DEVICES.items():
        key = dev.split()[0]
        f = nz[f"{key}_Hopt{d['Hopt']}_f"]; V = nz[f"{key}_Hopt{d['Hopt']}_V"]
        i = int(np.argmax(V)); f0 = f[i]
        a0.plot((f - f0) / 1e3, V / V[i], ".", color=cols[dev], ms=2.5,
                alpha=0.7, label=f"{dev}, {d['Hopt']} Oe: Q = {d['Q_V'][d['Hopt']]}")
        Qn = res[(dev, d["Hopt"])]["nom"]["Q"]
        fl = np.linspace(-2.5, 2.5, 400)
        a0.plot(fl, 1 / np.sqrt(1 + (2 * Qn * fl * 1e3 / f0)**2), "-",
                color=cols[dev], lw=1.0, alpha=0.8,
                label=f"   budget Q = {Qn:.0f}")
    a0.set_xlim(-2.5, 2.5); a0.set_ylim(0, 1.1)
    a0.set_xlabel("f − f$_p$ [kHz]"); a0.set_ylabel("normalized open-circuit voltage")
    a0.set_title("(a) Ni/LiNbO$_3$/Ni plates, 1 Oe drive (measured)", fontsize=9)
    a0.legend(fontsize=6, loc="upper left")
    labels, x = [], 0
    for dev, d in DEVICES.items():
        for H in sorted(d["Q_V"], reverse=True):
            r = res[(dev, H)]; n = r["nom"]
            a1.bar(x, n["inv_eddy"] * 1e3, color=BLUE, width=0.7)
            a1.bar(x, n["inv_hyst"] * 1e3, bottom=n["inv_eddy"] * 1e3, color=RED, width=0.7)
            a1.bar(x, n["inv_lno"] * 1e3, bottom=(n["inv_eddy"] + n["inv_hyst"]) * 1e3,
                   color=GREY, width=0.7)
            lo, hi = 1e3 / r["min_loss"]["Q"], 1e3 / r["max_loss"]["Q"]
            a1.plot([x, x], [lo, hi], "-", color="k", lw=0.8)
            a1.plot(x + 0.22, 1e3 / d["Q_V"][H], "D", color="k", ms=5)
            if H == d["Hopt"]:
                a1.plot(x + 0.22, 1e3 / d["Q_bvd"], "s", color="k", ms=5, mfc="w")
            labels.append(f"{dev.split()[0]}\n{H} Oe"); x += 1
    a1.set_xticks(range(x)); a1.set_xticklabels(labels, fontsize=7)
    a1.set_ylabel("1/Q [10$^{-3}$]"); a1.set_yscale("log"); a1.set_ylim(0.2, 20)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    a1.legend(handles=[Patch(color=BLUE, label="Ni eddy currents (eq. 5)"),
                       Patch(color=RED, label="Ni magnetomechanical hysteresis (J-A)"),
                       Patch(color=GREY, label="LiNbO$_3$ (floor)"),
                       Line2D([], [], color="k", lw=0.8, label="material interval"),
                       Line2D([], [], marker="D", color="k", ls="", ms=5, label="measured, V(f) bandwidth"),
                       Line2D([], [], marker="s", color="k", ls="", ms=5, mfc="w", label="measured, BVD fit of |Z|")],
              fontsize=6, loc="upper right", ncol=2)
    a1.set_title("(b) budget against measured 1/Q", fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / "fig_en_nilno.png", dpi=200)
    plt.close(fig)


def main(verbose=True):
    res = {}
    for dev, d in DEVICES.items():
        for H in sorted(d["Q_V"], reverse=True):
            r = {c: budget(dev, H, c) for c in ("nom", "min_loss", "max_loss")}
            res[(dev, H)] = r
            if verbose:
                n = r["nom"]
                print(f"{dev:15s} H={H:3d} Oe : Q = {n['Q']:5.0f} [{r['max_loss']['Q']:.0f} ; "
                      f"{r['min_loss']['Q']:.0f}]  mesuré V(f) {d['Q_V'][H]}"
                      f"{'  BVD ' + str(d['Q_bvd']) if H == d['Hopt'] else ''}"
                      f"  | 1/Q : Foucault {n['inv_eddy']:.2e} hyst {n['inv_hyst']:.2e} "
                      f"LNO {n['inv_lno']:.2e} | t/δ {n['t_over_delta']:.2f} k² {n['k2']:.3f} "
                      f"T {n['T']/1e6:.1f} MPa ΔH_eq {n['dH_eq_oe']:.1f} Oe")
    figure(res)
    json.dump({f"{k[0]}|{k[1]}": {c: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                                     for kk, vv in r.items()} for c, r in v.items()}
               for k, v in res.items()}, open(HERE / "nilno_results.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    main()
