# -*- coding: utf-8 -*-
"""
Bibliographie de l'article Q — construite DEPUIS Crossref, jamais de mémoire
(cf. leçon verify-refs : 11 réfs fabriquées sur 45 en rédigeant de tête).

Chaque entrée est soit un DOI (vérifié tel quel), soit une requête
titre+auteur (le premier hit est proposé, À INSPECTER dans le rapport).
Sortie : refs_verified.json (métadonnées canoniques Crossref) que
build_manuscript.js consomme directement -> impossible de fabriquer.
Les entrées type="manual" (thèse) sont recopiées telles quelles.
"""
import json
import subprocess
import sys
import time
import urllib.parse

# ordre = ordre de première citation prévu dans le manuscrit (ajusté ensuite)
WANTED = [
    # --- contexte ME
    dict(key="nan2008", doi="10.1063/1.2836410"),
    dict(key="srinivasan2010", doi="10.1146/annurev-matsci-070909-104459"),
    dict(key="bichurin2003lf", doi="10.1103/PhysRevB.68.054402"),
    dict(key="bichurin2003res", doi="10.1103/PhysRevB.68.132408"),
    dict(key="dong2003", q="Longitudinal and transverse magnetoelectric voltage coefficients of magnetostrictive piezoelectric laminate composite theory", au="Dong"),
    dict(key="zhai2008", doi="10.1111/j.1551-2916.2008.02259.x"),
    dict(key="dong2003resonant", q="Enhanced magnetoelectric effects in laminate composites of Terfenol-D Pb(Zr,Ti)O3 under resonant drive", au="Dong"),
    dict(key="kopyl2021", doi="10.1016/j.mtbio.2021.100149"),
    # --- mesures & modèles de l'équipe
    dict(key="malleron2019", q="Experimental study of magnetoelectric transducers for power supply of small biomedical devices", au="Malleron"),
    dict(key="malleron2018these", type="manual", text=(
        "K. Malleron, Modelisation multiphysique, caracterisation et "
        "conception de transducteurs magnetoelectriques pour l'alimentation "
        "de capteurs biomedicaux autonomes, PhD thesis, Sorbonne Universite, "
        "Paris (2018), doi:10.70675/41bf3f16z0636z4a7bz830dz3f570115aabb.")),
    dict(key="do2019", doi="10.1109/TMAG.2019.2926237"),
    dict(key="talleb2015", q="Finite element modeling of a magnetoelectric energy transducer including the load effect", au="Talleb"),
    dict(key="talleb2022", q="Talleb magnetostrictive energy averaged model", au="Talleb", y=2022),
    # --- pertes piézo
    dict(key="holland1967", q="Representation of dielectric elastic and piezoelectric losses by complex coefficients", au="Holland"),
    dict(key="uchino2001", q="Loss mechanisms in piezoelectrics how to measure different losses separately", au="Uchino"),
    dict(key="ieee176", doi="10.1109/IEEESTD.1988.79638"),
    # --- magnétostrictifs : modèles, hystérésis, pertes
    dict(key="engdahl2000", type="manual", text=(
        "G. Engdahl (ed.), Handbook of Giant Magnetostrictive Materials "
        "(Academic Press, San Diego, 2000).")),
    dict(key="dapino2000", doi="10.1109/20.846217"),
    dict(key="jiles1986", doi="10.1016/0304-8853(86)90066-1"),
    dict(key="jiles1995", doi="10.1088/0022-3727/28/8/001"),
    dict(key="bertotti1988", doi="10.1109/20.43994"),
    dict(key="bertotti1998", type="manual", text=(
        "G. Bertotti, Hysteresis in Magnetism: for Physicists, Materials "
        "Scientists, and Engineers (Academic Press, San Diego, 1998).")),
    dict(key="rayleigh1887", q="On the behaviour of iron and steel under the operation of feeble magnetic forces", au="Rayleigh"),
    dict(key="aubert2018", doi="10.1103/PhysRevApplied.9.044035"),
    dict(key="squire1990", q="Phenomenological model for magnetization magnetostriction and Delta E effect in field annealed amorphous ribbons", au="Squire"),
    # --- démagnétisation & shear lag
    dict(key="aharoni1998", doi="10.1063/1.367113"),
    dict(key="chang2007", q="Modeling shear lag and demagnetization effects in magneto-electric laminate composites", au="Chang"),
    # --- non-linéarité ME
    dict(key="burdin2014", q="Nonlinear magneto-electric effects in ferromagnetic piezoelectric composites", au="Burdin"),
    dict(key="fetisov2018", q="Nonlinear magnetoelectric effects at high magnetic field amplitudes in composite multiferroics", au="Fetisov"),
    # --- récupération d'énergie / applications
    dict(key="choi2022", doi="10.1039/D2SE00445C"),
    dict(key="chu2019", q="Review of multi-layered magnetoelectric composite materials and devices applications", au="Chu"),
    # --- circuits équivalents & amortissement en FEM ME
    dict(key="dong2004circuit", q="Equivalent circuit method for static and dynamic analysis of magnetoelectric laminated composites", au="Dong"),
    dict(key="talleb2014", doi="10.1016/j.jallcom.2014.06.121"),
    dict(key="caughey1965", doi="10.1115/1.3627262"),
    dict(key="dong2006metglas", q="Giant magnetoelectric effect in Metglas long-type PZT fiber laminates", au="Dong"),
]


def crossref(url):
    out = subprocess.run(["curl", "-s", "-m", "30", url],
                         capture_output=True, text=True)
    try:
        return json.loads(out.stdout)
    except Exception:
        return None


def fetch(entry):
    if entry.get("type") == "manual":
        return dict(key=entry["key"], type="manual", text=entry["text"])
    if "doi" in entry:
        j = crossref(f"https://api.crossref.org/works/{entry['doi']}")
        if not j or "message" not in j:
            return dict(key=entry["key"], type="FAILED", query=entry["doi"])
        return canon(entry["key"], j["message"], via="doi")
    q = urllib.parse.quote(entry["q"])
    au = urllib.parse.quote(entry.get("au", ""))
    url = (f"https://api.crossref.org/works?query.bibliographic={q}"
           f"&query.author={au}&rows=3")
    j = crossref(url)
    if not j or not j.get("message", {}).get("items"):
        return dict(key=entry["key"], type="FAILED", query=entry["q"])
    # premier hit + alternatives pour inspection
    items = j["message"]["items"]
    best = canon(entry["key"], items[0], via="query")
    best["alternatives"] = [short(it) for it in items[1:3]]
    return best


def short(m):
    a = m.get("author", [{}])
    return dict(title=(m.get("title") or [""])[0][:90],
                first=a[0].get("family", "?") if a else "?",
                cont=(m.get("container-title") or [""])[0],
                year=(m.get("issued", {}).get("date-parts", [[0]])[0][0]),
                doi=m.get("DOI", ""))


def canon(key, m, via):
    auth = [f"{a.get('given','')} {a.get('family','')}".strip()
            for a in m.get("author", [])]
    year = m.get("issued", {}).get("date-parts", [[None]])[0][0]
    return dict(key=key, type="crossref", via=via,
                authors=auth,
                title=(m.get("title") or [""])[0],
                container=(m.get("container-title") or [""])[0],
                short_container=(m.get("short-container-title") or [""])[:1],
                volume=m.get("volume", ""), issue=m.get("issue", ""),
                page=m.get("page", ""),
                article_number=m.get("article-number", ""),
                year=year, doi=m.get("DOI", ""))


def main():
    out = []
    for e in WANTED:
        r = fetch(e)
        out.append(r)
        tag = r["type"]
        if tag == "crossref":
            print(f"[{r['via']:5}] {r['key']:16} {r['authors'][0] if r['authors'] else '?'} "
                  f"et al., {r['title'][:70]} | {r['container'][:40]} "
                  f"{r['volume']}, {r['page'] or r['article_number']} ({r['year']}) "
                  f"doi:{r['doi']}")
        else:
            print(f"[{tag:5}] {r['key']:16} {r.get('text', r.get('query',''))[:80]}")
        time.sleep(0.4)
    with open("refs_verified.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    n_fail = sum(1 for r in out if r["type"] == "FAILED")
    print(f"\n{len(out)} entrées, {n_fail} échecs -> refs_verified.json")


if __name__ == "__main__":
    sys.exit(main())
