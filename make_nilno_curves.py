# -*- coding: utf-8 -*-
"""
Provenance de nilno_devices.npz : trois dispositifs Ni/LiNbO3/Ni auto-polarisés
(plaques 10 x 10 mm, LiNbO3 0,5 mm en coupes 0°Y (BIO68), 36°Y (BIO69),
128°Y (BIO70), Ni pulvérisé 50 µm sur les deux faces = électrodes, aucun
adhésif), mesurés au GeePs (janvier-mars 2026) : module d'impédance |Z|(f)
et tension en circuit ouvert V(f) à h_ac = 1 Oe, au biais optimal et à champ
appliqué nul (rémanence). Fichiers .mat dans
iCloud/.../conf_SGE/2025_Toulouse/Mesure/ (non distribués : le .npz fait foi).
"""
import glob
import os
import re
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = Path.home() / ("Library/Mobile Documents/com~apple~CloudDocs/Professionel/"
                     "02_Recherche/RECHERCHE/PUBLICATIONS&COMMUNICATIONS/"
                     "conf_SGE/2025_Toulouse/Mesure")
DEV = {"BIO68": ("BIO68_0ycutaxe_X", 0), "BIO69": ("BIO69_36ycut_axe_x", 36),
       "BIO70": ("BIO70_128ycutaxe_x", 128)}


def main():
    if not SRC.exists():
        print("sources absentes : nilno_devices.npz distribué fait foi")
        return
    from scipy.io import loadmat
    out = {}
    for dev, (d, cut) in DEV.items():
        impf = glob.glob(str(SRC / d / "imp_*.mat"))[0]
        m = loadmat(impf)
        key = "module" if "module" in m else [k for k in m if not k.startswith("__")][0]
        imp = m[key].flatten()
        mm = re.search(r"(\d+)_(\d+)kHz", impf)
        fZ = np.linspace(float(mm.group(1)), float(mm.group(2)), imp.size) * 1e3
        out[f"{dev}_fZ"], out[f"{dev}_Z"] = fZ, imp
        for vf in sorted(glob.glob(str(SRC / d / "*VfctFreq*.mat"))):
            v = loadmat(vf)
            V = v["Vout_res"].flatten(); F = v["FreqSweep"].flatten()
            F = F * 1e3 if F.max() < 1e4 else F
            tag = "H0" if "0Hdc" in os.path.basename(vf) or "hdc0" in os.path.basename(vf) else \
                "Hopt" + re.search(r"HdcOpt(\d+)Oe", vf).group(1)
            out[f"{dev}_{tag}_f"], out[f"{dev}_{tag}_V"] = F, V
    np.savez(HERE / "nilno_devices.npz", **out)
    print("nilno_devices.npz :", sorted(k for k in out))


if __name__ == "__main__":
    main()
