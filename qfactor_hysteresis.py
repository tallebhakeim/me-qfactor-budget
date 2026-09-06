# -*- coding: utf-8 -*-
"""
Chantier 1 de l'article Q : remplacer l'intervalle de littérature
Q_m(Terfenol) = [3 ; 30] par un canal d'hystérésis magnétomécanique CALCULÉ,
en loi de Rayleigh, dépendant de l'amplitude — sur la même base que le
q_ac_factor déjà présent dans core/me/me_deam.py (Aubert PRApplied 2018 +
plancher de réversibilité Jiles-Atherton).

Physique :
 1. À la résonance, la couche magnétostrictive voit une oscillation de
    CONTRAINTE d'amplitude |T|. Par la relation constitutive
    M = chi.H + (d/mu0).T, cette oscillation équivaut à un champ
      Delta_H_eq = d33m.|T| / (mu0.chi)         [A/m]
    atténué par la réaction démagnétisante du circuit ouvert :
      Delta_H_eff = Delta_H_eq / (1 + chi.N)    (même N que demag_lambda).
 2. La boucle mineure parcourue autour du biais a une susceptibilité
      chi_ac(DH) = chi.[ c_rev + (eta_inf - c_rev).DH/(DH + Ha0) ]
    (forme q_ac_factor : plancher réversible c_rev.chi puis montée Rayleigh)
    et son AIRE (perte par cycle et par volume) est, en régime de Rayleigh
    généralisé :
      W(DH) = (4/3).mu0.[chi_ac(DH) - chi_rev].DH^2     [J/m^3/cycle]
    -> W ~ (4/3).mu0.nu.DH^3 pour DH << Ha0 (Rayleigh classique,
       nu = chi.(eta_inf - c_rev)/Ha0), saturation douce au-delà.
 3. 1/Q_hyst = somme_elements W(DH_e).V_e / (2.pi.U),  U = W_tot/2.
    Comme W ~ DH^3 et U ~ amplitude^2, 1/Q_hyst CROÎT avec l'amplitude :
    le Q du composite est NON LINÉAIRE (bien documenté pour le Terfenol),
    d'où la résolution AUTO-COHÉRENTE Q(H_ac) dans qfactor_model.q_budget.

Paramètres de Rayleigh du Terfenol (c_rev, eta_inf, Ha0) : triplets
(nominal, inf, sup) — à resserrer par UNE mesure matériau (barreau nu) ou
par calibration VSM ; ce sont des propriétés MATÉRIAU, pas structure.
"""
import numpy as np

MU0 = 4e-7 * np.pi
OE = 1e3 / (4 * np.pi)


def chi_ac(dH_am, chi, c_rev, eta_inf, Ha0_am):
    """Susceptibilité de boucle mineure (forme q_ac_factor de me_deam)."""
    return chi * (c_rev + (eta_inf - c_rev) * dH_am / (dH_am + Ha0_am))


def w_cycle(dH_am, chi, c_rev, eta_inf, Ha0_am):
    """Aire de la boucle mineure [J/m^3/cycle], Rayleigh généralisé :
    W = (4/3) mu0 (chi_ac - chi_rev) dH^2, chi_rev = c_rev.chi."""
    dchi = chi_ac(dH_am, chi, c_rev, eta_inf, Ha0_am) - c_rev * chi
    return (4.0 / 3.0) * MU0 * dchi * dH_am**2


def dH_equiv(T_amp, d33m, chi, N=0.0):
    """Champ équivalent [A/m] de l'oscillation de contrainte T_amp [Pa],
    atténué par la réaction démagnétisante (facteur 1/(1+chi.N))."""
    return d33m * T_amp / (MU0 * chi) / (1.0 + chi * N)


def invQ_hyst_layer(T_amps, vol_elems, Wtot, d33m, chi, N,
                    c_rev, eta_inf, Ha0_oe):
    """1/Q du canal hystérésis pour UNE couche : somme élémentaire de
    W(DH_eff) rapportée à l'énergie stockée U = Wtot/2."""
    Ha0 = Ha0_oe * OE
    dH = dH_equiv(np.asarray(T_amps, float), d33m, chi, N)
    W = w_cycle(dH, chi, c_rev, eta_inf, Ha0)
    return float(np.sum(W * np.asarray(vol_elems)) / (np.pi * Wtot))


if __name__ == "__main__":
    # auto-tests rapides
    chi, c_rev, eta_inf, Ha0 = 7.4, 0.15, 0.9, 18 * OE
    # 1) régime de Rayleigh : W ~ dH^3 (rapport 8 quand dH double, petit dH)
    w1 = w_cycle(1.0, chi, c_rev, eta_inf, Ha0)
    w2 = w_cycle(2.0, chi, c_rev, eta_inf, Ha0)
    print(f"W(2dH)/W(dH) petit signal = {w2/w1:.3f} (attendu ~8)")
    # 2) saturation : croissance < cubique à grand dH
    w3 = w_cycle(10 * Ha0, chi, c_rev, eta_inf, Ha0)
    w4 = w_cycle(20 * Ha0, chi, c_rev, eta_inf, Ha0)
    print(f"W(2dH)/W(dH) grand signal = {w4/w3:.3f} (attendu ~4 : quadratique)")
    # 3) ordre de grandeur au point de fonctionnement Malleron :
    #    T ~ 0,45 MPa (H_ac = 1 Oe, Q ~ 10) -> dH_eff, W, 1/Q estime
    dH = dH_equiv(4.5e5, 18.7e-9, chi, N=0.0724)
    print(f"dH_eff(T=0,45 MPa) = {dH/OE:.1f} Oe ; "
          f"W = {w_cycle(dH, chi, c_rev, eta_inf, Ha0):.2f} J/m^3/cycle")
