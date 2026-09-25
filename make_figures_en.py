# -*- coding: utf-8 -*-
"""
Figures (EN) du manuscrit SMS — générées depuis le modèle, aucune valeur
saisie à la main. Sorties : fig_en_schematic / budget / samples / qhac /
eddy / fem2d (PNG 200 dpi).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

import qfactor_model as qm
import qfactor_hysteresis as qh

NX = 200
BLUE, RED, GREY = "#4878a8", "#c1272d", "#666666"
GREEN = "#3a8f4a"
DPI = 200

NOMS = dict(pzt_meca="PZT mechanical (Q$_m$)",
            mag_hyst="magnetomechanical hysteresis",
            pzt_diel="PZT dielectric",
            foucault="eddy currents (computed $\\lambda$)",
            colle="epoxy bond (shear lag)")


def fig_schematic():
    fig = plt.figure(figsize=(10.2, 6.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.42)
    gtop = gs[0].subgridspec(1, 2, width_ratios=[1.15, 1], wspace=0.08)
    gbot = gs[1].subgridspec(1, 4, wspace=0.55)
    a1 = fig.add_subplot(gtop[0, 0])
    a2 = fig.add_subplot(gtop[0, 1])
    s1, s2, s3, s4 = (fig.add_subplot(gbot[0, k]) for k in range(4))
    # ---- (a) sample stack
    a1.add_patch(Rectangle((0, 0), 20, 1, fc="#f0c674", ec="k", lw=0.8))
    a1.add_patch(Rectangle((0, 1), 14, 1, fc="#9fb4cc", ec="k", lw=0.8))
    a1.text(10, 0.5, "PZT-5H  (20 × 10 × 1 mm)", ha="center", va="center",
            fontsize=9)
    a1.text(7, 1.5, "Terfenol-D  (14 × 10 × 1 mm)", ha="center", va="center",
            fontsize=9)
    a1.annotate("", xy=(20, -0.55), xytext=(0, -0.55),
                arrowprops=dict(arrowstyle="<->", lw=0.8))
    a1.text(10, -0.95, "$L_p$ = 20 mm", ha="center", fontsize=8)
    a1.annotate("", xy=(14, 2.45), xytext=(0, 2.45),
                arrowprops=dict(arrowstyle="<->", lw=0.8))
    a1.text(7, 2.62, "$L_m$ = 14 mm", ha="center", fontsize=8)
    a1.add_patch(FancyArrowPatch((15.2, 1.75), (19.4, 1.75),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=RED))
    a1.text(17.3, 2.0, "$H_{dc}+h_{ac}$", color=RED, ha="center", fontsize=9)
    # électrodes sur TOUTE la longueur du PZT (faces haut et bas), comme
    # sur les échantillons et dans les modèles (eq. (2) : <S> sur L_p)
    a1.plot([0.0, 20.0], [-0.05, -0.05], lw=2.5, color=GREY,
            solid_capstyle="butt")
    a1.text(17.0, -0.38, "full-length electrodes", fontsize=7.5, color=GREY,
            ha="center")
    a1.plot([0.0, 20.0], [1.05, 1.05], lw=2.5, color=GREY,
            solid_capstyle="butt")
    a1.set_xlim(-1.2, 21.5); a1.set_ylim(-1.5, 3.1)
    a1.axis("off"); a1.set_title("(a) reference bilayer (L–T mode)",
                                 fontsize=9.5)
    # ---- (b) loss channels
    a2.add_patch(Rectangle((0, 0), 10, 1.4, fc="#f0c674", ec="k", lw=0.8))
    a2.add_patch(Rectangle((0, 1.4), 7, 1.4, fc="#9fb4cc", ec="k", lw=0.8))
    a2.text(5, 0.6, "PZT:  1/Q$_m$ , tan$\\,\\delta_\\varepsilon$",
            ha="center", va="center", fontsize=9)
    a2.text(3.5, 1.78, "Terfenol-D:  eddy currents, Rayleigh hysteresis",
            ha="center", va="center", fontsize=8)
    a2.add_patch(Rectangle((0, 1.28), 7, 0.24, fc="#888888", ec="none"))
    a2.annotate("epoxy bond: shear lag", xy=(6.4, 1.40), xytext=(7.6, 0.45),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    for xc in (1.6, 3.5, 5.4):
        c = plt.Circle((xc, 2.42), 0.28, fill=False, color=RED, lw=1.0)
        a2.add_patch(c)
        a2.annotate("", xy=(xc + 0.26, 2.42), xytext=(xc - 0.26, 2.42),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    a2.text(3.5, 3.15, "induced eddy currents  ($b^* = \\lambda\\, d_{33,m} T$)",
            color=RED, ha="center", fontsize=8)
    a2.set_xlim(-0.6, 10.9); a2.set_ylim(-0.5, 3.8)
    a2.axis("off"); a2.set_title("(b) loss channels", fontsize=9.5)

    # ---- (c)-(f) Q SANS mesure sur le composite : frise calculée
    _strategy_strip(fig, s1, s2, s3, s4)
    fig.savefig("fig_en_schematic.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


CH_COL = dict(foucault="#c1272d", mag_hyst="#e39b3d", pzt_meca="#4878a8",
              pzt_diel="#8fb3d9", colle="#999999")
CH_LAB = dict(foucault="eddy", mag_hyst="hysteresis", pzt_meca="PZT mech.",
              pzt_diel="PZT diel.", colle="bond")


def _layer_etas(b):
    """Facteurs de perte PAR COUCHE issus d'un budget (pour tracer la
    réponse prédite) : canaux PZT -> couche PZT, canaux magnétiques et
    colle -> couche Terfenol."""
    r, inv = b["res"], b["inv"]
    eta_p = (inv["pzt_meca"] + inv["pzt_diel"]) * r["Wtot"] / r["Wp"]
    eta_m = (inv["mag_hyst"] + inv["foucault"] + inv["colle"]) \
        * r["Wtot"] / sum(r["Wm"])
    return eta_p, eta_m


def _strategy_strip(fig, s1, s2, s3, s4):
    import qfactor_disk as qd
    INK = "#43516b"

    # (c) données matériau seules
    H = np.linspace(20, 1500, 150)
    d33 = np.array([qd.en.energy_coeffs("Terfenol-D", h, -23.8e6)["d33m"]
                    for h in H]) * 1e9
    s1.plot(H, d33, "-", color=BLUE, lw=1.4)
    s1.plot(525, 18.7, "o", color=RED, ms=6)
    s1.annotate("bias\n525 Oe", xy=(525, 18.7), xytext=(950, 19.5),
                fontsize=6.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.7))
    s1.set_xlabel("H [Oe]", fontsize=7.5)
    s1.set_ylabel("Terfenol-D d$_{33,m}$ [nm/A]", fontsize=7.5)
    s1.tick_params(labelsize=6.5)
    s1.set_ylim(0, 34)
    s1.text(0.03, 0.97, "PZT-5H datasheet:\nQ$_m$ = 65, tan$\\,\\delta$ = 0.02\n"
            "Terfenol-D: $\\sigma$, $\\chi$(H),\nRayleigh triplet",
            transform=s1.transAxes, fontsize=6.0, va="top",
            bbox=dict(fc="white", ec=INK, lw=0.6, alpha=0.9))
    s1.set_title("(c) material data only", fontsize=8.5)

    # (d) où va l'énergie : mode structurel calculé
    asm = qm.assemble(qm.REF, 200)
    fr, _ = qm.find_resonance(asm)
    r = qm.solve_harm(asm, fr)
    xc = 0.5 * (asm["x"][:-1] + asm["x"][1:]) * 1e3
    ep = qm.PZT["E"] * np.abs(r["S"])**2
    m0 = qm.MAGS["Terfenol-D@bias"]
    em = np.where(asm["in1"], m0["E"] * np.abs(r["S"] - m0["d33m"] * qm.OE)**2,
                  np.nan)
    vmax = np.nanmax(np.concatenate([ep, em]))
    edges = asm["x"] * 1e3
    s2.pcolormesh(edges, [0, 1], (ep / vmax)[None, :], cmap="YlOrRd",
                  vmin=0, vmax=1, shading="flat")
    s2.pcolormesh(edges, [1, 2], (em / vmax)[None, :], cmap="YlOrRd",
                  vmin=0, vmax=1, shading="flat")
    s2.add_patch(Rectangle((0, 0), 20, 1, fill=False, ec="k", lw=0.8))
    s2.add_patch(Rectangle((0, 1), 14, 1, fill=False, ec="k", lw=0.8))
    s2.text(17, 0.5, "PZT", fontsize=7, ha="center", va="center")
    s2.text(7, 2.35, "Terfenol-D", fontsize=7, ha="center")
    fp = r["Wp"] / r["Wtot"] * 100
    fm = sum(r["Wm"]) / r["Wtot"] * 100
    fe = r["We"] / r["Wtot"] * 100
    s2.text(10, -0.75, f"energy: PZT {fp:.0f}%  ·  TD {fm:.0f}%  ·  "
            f"electric {fe:.0f}%", fontsize=6.5, ha="center")
    s2.text(10, -1.35, f"mode at f$_r$ = {fr/1e3:.1f} kHz (geometry only)",
            fontsize=6.3, ha="center", color=INK)
    s2.set_xlim(-0.5, 20.5); s2.set_ylim(-1.7, 2.8)
    s2.axis("off")
    s2.set_title("(d) where the energy goes", fontsize=8.5)

    # (e) chaque canal pondéré par sa fraction : 1/Q = somme
    bud = qm.q_budget(qm.REF, nx=200, H_ac_oe=1.0)
    blo, bhi = qm.q_bracket(qm.REF, 200, H_ac_oe=1.0)
    order = ["foucault", "mag_hyst", "pzt_meca", "pzt_diel", "colle"]
    left = 0.0
    for k in order:
        v = bud["inv"][k] * 100
        s3.barh(0, v, left=left, color=CH_COL[k], ec="white", lw=0.5,
                height=0.55)
        if v > 0.5:
            s3.text(left + v / 2, 0, CH_LAB[k], rotation=90, fontsize=6,
                    ha="center", va="center", color="white")
        left += v
    s3.annotate("", xy=(left, 0.52), xytext=(0, 0.52),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color=INK))
    s3.text(left / 2, 0.66, f"$\\Sigma_i$ 1/Q$_i$ = {left/100:.3f}",
            fontsize=6.8, ha="center", color=INK)
    s3.text(left / 2, -0.62, f"Q = {bud['Q']:.1f}   "
            f"[{blo['Q']:.1f} ; {bhi['Q']:.1f}]", fontsize=7.6,
            ha="center", weight="bold", color=INK)
    s3.set_xlim(0, left * 1.02); s3.set_ylim(-0.95, 0.95)
    s3.set_yticks([])
    s3.set_xlabel("1/Q$_i$ [×10$^{-2}$]", fontsize=7.5)
    s3.tick_params(labelsize=6.5)
    for sp in ("top", "right", "left"):
        s3.spines[sp].set_visible(False)
    s3.set_title("(e) weight each channel, sum", fontsize=8.5)

    # (f) résonance prédite (forme normalisée), mesure = validation seule
    fs = np.linspace(62e3, 79e3, 260)
    curves = {}
    for nm, b in (("nom", bud), ("lo", blo), ("hi", bhi)):
        ep_, em_ = _layer_etas(b)
        asb = qm.assemble(qm.REF, 120, eta_p=ep_, eta_mag=em_)
        a = qm.sweep(asb, fs)
        curves[nm] = a / a.max()
    s4.fill_between(fs / 1e3, np.minimum(curves["lo"], curves["hi"]),
                    np.maximum(curves["lo"], curves["hi"]),
                    color=BLUE, alpha=0.22, lw=0, label="material intervals")
    s4.plot(fs / 1e3, curves["nom"], "-", color=BLUE, lw=1.4,
            label="predicted")
    mc = np.load("malleron_measured_curve.npz")
    s4.plot(mc["f_kHz"], mc["alpha"] / mc["alpha"].max(), ".", ms=1.6,
            color=GREY, alpha=0.55, label="measured (validation)")
    s4.set_xlabel("frequency [kHz]", fontsize=7.5)
    s4.set_ylabel("normalized α$_E$", fontsize=7.5)
    s4.tick_params(labelsize=6.5)
    s4.set_ylim(0, 1.12)
    s4.legend(fontsize=5.8, loc="upper left", framealpha=0.9)
    s4.set_title("(f) predicted resonance", fontsize=8.5)

    # flèches entre étapes + bandeau
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for a, b in ((s1, s2), (s2, s3), (s3, s4)):
        pa = a.get_position()
        # la flèche s'arrête AVANT les étiquettes d'axe de la cible
        tb = b.get_tightbbox(rend).transformed(inv)
        y = 0.5 * (pa.y0 + pa.y1)
        x0 = a.get_tightbbox(rend).transformed(inv).x1 if a is s1 else pa.x1
        fig.add_artist(FancyArrowPatch(
            (x0 + 0.004, y), (tb.x0 - 0.004, y),
            transform=fig.transFigure, arrowstyle="-|>",
            mutation_scale=14, color=INK, lw=1.2))
    p1, p4 = s1.get_position(), s4.get_position()
    fig.text(0.5 * (p1.x0 + p4.x1), p1.y1 + 0.075,
             "Predicting Q with no measurement on the composite: material "
             "data  →  structural mode  →  weighted loss channels  →  "
             "resonance", ha="center", fontsize=8.8, color=INK,
             style="italic")


def main():
    fig_schematic()

    bud = qm.q_budget(qm.REF, nx=NX, H_ac_oe=1.0)
    rows = []
    for s in qm.SAMPLES:
        b = qm.q_budget(s, nx=NX)
        blo, bhi = qm.q_bracket(s, NX)
        Qe, c, _, a_stat = qm.q_eff_measured(s, NX)
        rows.append(dict(s=s, b=b, blo=blo, bhi=bhi, Qe=Qe, c=c))
    ref = next(r for r in rows if r["s"] == qm.REF)

    # ---- Fig 2 : budget bars
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    keys = list(NOMS)
    y = np.arange(len(keys))
    v_nom = np.array([bud["inv"][k] for k in keys])
    v_lo = np.array([ref["bhi"]["inv"][k] for k in keys])
    v_hi = np.array([ref["blo"]["inv"][k] for k in keys])
    ax.barh(y, v_nom, color=BLUE, alpha=0.85, label="nominal")
    ax.errorbar(v_nom, y, xerr=[np.maximum(v_nom - v_lo, 0),
                                np.maximum(v_hi - v_nom, 0)],
                fmt="none", ecolor="k", capsize=4, lw=1.2,
                label="material intervals")
    ax.axvline(1 / ref["Qe"], color=RED, ls="--", lw=1.5,
               label=f"measured 1/Q = 1/{ref['Qe']:.0f}")
    ax.set_yticks(y); ax.set_yticklabels([NOMS[k] for k in keys], fontsize=9)
    ax.set_xlabel("loss contribution 1/$Q_i$")
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig("fig_en_budget.png", dpi=DPI)
    plt.close(fig)

    # ---- Fig 3 : four samples
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for i, r in enumerate(rows):
        ax.plot([i, i], [r["blo"]["Q"], r["bhi"]["Q"]], color=BLUE, lw=8,
                alpha=0.45, solid_capstyle="butt",
                label="a-priori interval" if i == 0 else None)
        ax.plot(i, r["b"]["Q"], "o", color=BLUE, ms=7,
                label="nominal prediction" if i == 0 else None)
        ax.plot(i, r["Qe"], "*", color=RED, ms=15,
                label="measured $Q_{eff}$" if i == 0 else None)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["s"] for r in rows], fontsize=8)
    ax.set_ylabel("quality factor Q"); ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.annotate("static reading taken in the\nnonlinear regime (section 6.2)",
                xy=(1, rows[1]["Qe"]), xytext=(1.08, 34), fontsize=7.5,
                color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    fig.tight_layout(); fig.savefig("fig_en_samples.png", dpi=DPI)
    plt.close(fig)

    # ---- Fig 4 : (a) Q(H_ac), (b) charge R -> f_r et Q
    Hacs = np.array([0.3, 0.7, 1.0, 2.0, 3.0, 4.5, 6.5, 10.0])
    q_nom = [qm.q_budget(qm.REF, "nom", NX, h)["Q"] for h in Hacs]
    q_min = [qm.q_budget(qm.REF, "max_loss", NX, h)["Q"] for h in Hacs]
    q_max = [qm.q_budget(qm.REF, "min_loss", NX, h)["Q"] for h in Hacs]
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(9.8, 3.4))
    ax.fill_between(Hacs, q_min, q_max, color=BLUE, alpha=0.25,
                    label="material-interval envelope")
    ax.plot(Hacs, q_nom, "o-", color=BLUE, label="nominal (self-consistent)")
    ax.axhline(ref["Qe"], color=RED, ls="--", lw=1.5,
               label=f"measured $Q_{{eff}}$ = {ref['Qe']:.0f}")
    ax.set_xlabel("drive amplitude $h_{ac}$ [Oe]")
    ax.set_ylabel("self-consistent Q")
    ax.set_ylim(bottom=0); ax.legend(fontsize=8)
    ax.set_title("(a) drive amplitude", fontsize=9.5)

    Rs = np.logspace(1, 6, 21)
    sw = qm.load_sweep(Rs, nx=NX)
    frs = np.array([o["f_r"] for o in sw]) / 1e3
    qls = np.array([o["Q"] for o in sw])
    asm0 = qm.assemble(qm.REF, NX, stiffen=False)
    Ropt = 1 / (2 * np.pi * 70.3e3 * asm0["C0"])
    axb.semilogx(Rs, frs, "o-", color=BLUE, ms=4)
    axb.set_xlabel("load resistance R [Ω]")
    axb.set_ylabel("resonance $f_r$ [kHz]", color=BLUE)
    axb.tick_params(axis="y", labelcolor=BLUE)
    axb2 = axb.twinx()
    axb2.semilogx(Rs, qls, "s--", color=RED, ms=4)
    axb2.set_ylabel("loaded Q", color=RED)
    axb2.tick_params(axis="y", labelcolor=RED)
    axb2.set_ylim(0, 20)
    axb.axvline(Ropt, color=GREY, ls=":", lw=1.2)
    axb.text(Ropt * 1.3, 67.4, "R = 1/(ω C$_0$)\n= %d Ω" % round(Ropt),
             fontsize=7.5, color=GREY)
    axb.set_title("(b) electrical load ($f_s \\rightarrow f_p$, Q dip)",
                  fontsize=9.5)
    fig.tight_layout(); fig.savefig("fig_en_qhac.png", dpi=DPI)
    plt.close(fig)
    np.savez("fig_qhac_data.npz", Hacs=Hacs, q_nom=q_nom, q_min=q_min,
             q_max=q_max, Rs=Rs, frs=frs, qls=qls)

    # ---- Fig 5 : eddy physics in the slab
    terf = qm.MAGS["Terfenol-D@bias"]
    mu = (terf["chi"][0] + 1) * 4e-7 * np.pi
    sig = terf["sigma"]; t = 1e-3
    yv = np.linspace(-t / 2, t / 2, 400)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.2))
    for f, ls in ((7e3, ":"), (70.3e3, "-"), (700e3, "--")):
        k = np.sqrt(1j * 2 * np.pi * f * mu * sig)
        bprof = np.cosh(k * yv) / np.cosh(k * t / 2)
        delta = np.sqrt(2 / (2 * np.pi * f * mu * sig))
        a1.plot(yv * 1e3, np.abs(bprof), ls, color=BLUE,
                label=f"f = {f/1e3:.0f} kHz  (t/$\\delta$ = {t/delta:.1f})")
    a1.set_xlabel("position across thickness y [mm]")
    a1.set_ylabel("|b(y)| / b*")
    a1.legend(fontsize=8); a1.set_title("(a) motional flux profile",
                                        fontsize=9.5)
    freqs = np.logspace(2, 6, 120)
    p_ex = np.array([qm.eddy_power_slab(1e6, f, terf, 1.0, t) for f in freqs])
    b1 = terf["d33m"] * 1e6
    p_bf = sig * (2 * np.pi * freqs)**2 * b1**2 * t**2 / 24
    a2.loglog(freqs / 1e3, p_bf, "--", color=GREY,
              label="low-frequency law $\\sigma\\omega^2 b^{*2} t^2/24$")
    a2.loglog(freqs / 1e3, p_ex, "-", color=BLUE, label="exact 1-D diffusion")
    a2.axvline(70.3, color=RED, ls=":", lw=1.2)
    a2.text(78, p_ex.min() * 2.4, "$f_r$", color=RED, fontsize=9)
    a2.set_xlabel("frequency [kHz]")
    a2.set_ylabel("eddy loss density [W m$^{-3}$]")
    a2.legend(fontsize=8); a2.set_title("(b) screening above t $\\approx\\delta$",
                                        fontsize=9.5)
    fig.tight_layout(); fig.savefig("fig_en_eddy.png", dpi=DPI)
    plt.close(fig)

    # ---- Fig 6 : 2-D FEM with predicted per-layer losses
    import run_fem2d_crosscheck as cc
    import sys
    from io import StringIO
    old = sys.stdout; sys.stdout = StringIO()
    hE, hQ, hP, hS, s_ep = cc.main()
    sys.stdout = old
    ms = qm.SAMPLES[qm.REF]["meas"]
    fig, ax = plt.subplots(figsize=(5.9, 3.4))
    mc = np.load("malleron_measured_curve.npz")
    ax.plot(mc["f_kHz"], mc["alpha"], ".", color=RED, ms=2.5, alpha=0.6,
            label="measured resonance curve (digitized)")
    ax.plot(hP["freqs"] / 1e3, hP["alpha"], "--", color=GREY,
            label="2-D FEM, predicted per-layer losses")
    ax.plot(hS["freqs"] / 1e3, hS["alpha"], "-", color=BLUE,
            label=f"idem + epoxy stiffening (s = {s_ep:.2f})")
    # modèle 3D ajusté de la thèse Do 2019 (fig. 3.13), même échelle que la
    # mesure (pic tabulé 19,8) : hauteur reproduite, largeur non (Q ~ 39)
    d3 = np.load("do2019_fig313_curves.npz")
    sc = ms["a_res"] / d3["V_meas"].max()
    ax.plot(d3["f_3d"], d3["V_3d"] * sc, "-.", color=GREEN, lw=1.2,
            label="3-D FEM, fitted Rayleigh damping (Q = 39)")
    ax.plot(ms["f_r"] / 1e3, ms["a_res"], "*", color=RED, ms=13,
            label=f"tabulated peak ({ms['a_res']} V cm$^{{-1}}$ Oe$^{{-1}}$)")
    ax.axhline(ms["a_stat"], color=GREY, ls=":", lw=1.0)
    ax.text(51, ms["a_stat"] * 1.5, "measured static level", fontsize=7.5,
            color=GREY)
    ax.set_xlim(50, 85)
    ax.set_ylim(0, 36)                     # marge pour la légende
    ax.set_xlabel("frequency [kHz]")
    ax.set_ylabel("$\\alpha_E$ [V cm$^{-1}$ Oe$^{-1}$]")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.95)
    fig.tight_layout(); fig.savefig("fig_en_fem2d.png", dpi=DPI)
    plt.close(fig)

    print("6 figures EN générées.")


if __name__ == "__main__":
    main()
