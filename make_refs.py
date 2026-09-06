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
    dict(key="dong2003", doi="10.1109/tuffc.2003.1244741"),
    dict(key="zhai2008", doi="10.1111/j.1551-2916.2008.02259.x"),
    dict(key="dong2003resonant", doi="10.1063/1.1631756"),
    dict(key="kopyl2021", doi="10.1016/j.mtbio.2021.100149"),
    # --- mesures & modèles de l'équipe
    dict(key="malleron2019", doi="10.1016/j.mejo.2018.01.013"),
    dict(key="malleron2018these", type="manual", text=(
        "K. Malleron, Modelisation multiphysique, caracterisation et "
        "conception de transducteurs magnetoelectriques pour l'alimentation "
        "de capteurs biomedicaux autonomes, PhD thesis, Sorbonne Universite, "
        "Paris (2018), doi:10.70675/41bf3f16z0636z4a7bz830dz3f570115aabb.")),
    dict(key="do2019", doi="10.1109/TMAG.2019.2926237"),
    dict(key="talleb2015", doi="10.1109/tmag.2014.2357492"),
    dict(key="talleb2022", doi="10.1016/j.compstruct.2022.116260"),
    # --- pertes piézo
    dict(key="holland1967", doi="10.1109/t-su.1967.29405"),
    dict(key="uchino2001", doi="10.1109/58.896144"),
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
    dict(key="rayleigh1887", doi="10.1080/14786448708628000"),
    dict(key="aubert2018", doi="10.1103/PhysRevApplied.9.044035"),
    dict(key="squire1990", doi="10.1016/0304-8853(90)90764-h"),
    # --- démagnétisation & shear lag
    dict(key="aharoni1998", doi="10.1063/1.367113"),
    dict(key="chang2007", doi="10.1103/physrevb.76.134116"),
    # --- non-linéarité ME
    dict(key="burdin2014", doi="10.1016/j.jmmm.2014.01.062"),
    dict(key="fetisov2018", doi="10.1088/1361-6463/aab384"),
    # --- récupération d'énergie / applications
    dict(key="choi2022", doi="10.1039/D2SE00445C"),
    dict(key="chu2019", doi="10.1088/1361-6463/aac29b"),
    # --- circuits équivalents & amortissement en FEM ME
    dict(key="dong2004circuit", doi="10.1007/s11434-008-0304-7"),
    dict(key="talleb2014", doi="10.1016/j.jallcom.2014.06.121"),
    dict(key="caughey1965", doi="10.1115/1.3627262"),
    dict(key="dong2006metglas", doi="10.1063/1.2337996"),
    dict(key="rizzo2020these", doi="10.70675/66fce602z26b0z40c1z83d1z08d20353c521"),
    dict(key="rizzo2019", doi="10.1109/ismict.2019.8743873"),
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
