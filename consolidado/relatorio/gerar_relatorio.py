"""Gera o relatório HTML a partir de consolidado/scores_consolidados.csv.

Calcula os agregados (cobertura, histogramas, correlações, médias por partido por
legislatura, séries por legislatura, extremos por legislatura) em dados.json e injeta
o JSON em relatorio_template.html -> relatorio.html.

Uso: python3 consolidado/relatorio/gerar_relatorio.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IDX = {"ipp": "ipp_final_v4", "ird": "ird_peso", "co2": "co2_index", "ipipe": "ipipe_score_exposicao", "ifs": "ifs", "irr": "irr"}
MIN_PARTIDO = {"ipp": 8, "ird": 8, "co2": 8, "ipipe": 5, "ifs": 8, "irr": 8}


def s(v):
    """Texto seguro para JSON (NaN -> '–')."""
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def hist(series: pd.Series, bins, clip: tuple | None = None) -> dict:
    """Histograma; `clip` agrupa a cauda nos intervalos extremos, mas as estatísticas usam os valores originais."""
    x = series.dropna()
    cnt, edges = np.histogram(x.clip(*clip) if clip else x, bins=bins)
    return {
        "edges": [round(float(e), 4) for e in edges],
        "counts": [int(c) for c in cnt],
        "n": int(len(x)),
        "median": round(float(x.median()), 4),
        "mean": round(float(x.mean()), 4),
        "min": round(float(x.min()), 4),
        "max": round(float(x.max()), 4),
    }


def corr(df: pd.DataFrame, cols: list[tuple[str, str]]) -> dict:
    d = df[[c for c, _ in cols]]
    r = d.corr(method="spearman")
    present = d.notna().astype(int)
    n = present.T.dot(present)
    return {
        "labels": [l for _, l in cols],
        "r": [[None if pd.isna(v) else round(float(v), 3) for v in row] for row in r.values],
        "n": [[int(v) for v in row] for row in n.values],
    }


def by_party(df: pd.DataFrame, col: str, minn: int) -> list[dict]:
    g = df.dropna(subset=[col]).groupby("siglaPartido")[col].agg(["count", "mean", "median"]).reset_index()
    g = g[g["count"] >= minn].sort_values("mean", ascending=False)
    return [
        {"partido": s(r.siglaPartido), "n": int(r["count"]), "mean": round(float(r["mean"]), 3), "median": round(float(r["median"]), 3)}
        for _, r in g.iterrows()
    ]


def rank(df: pd.DataFrame, col: str, asc: bool, k: int = 10) -> list[dict]:
    x = df.dropna(subset=[col]).sort_values(col, ascending=asc).head(k)
    return [{"nome": s(r.nome), "partido": s(r.siglaPartido), "uf": s(r.siglaUf), "v": round(float(r[col]), 3)} for _, r in x.iterrows()]


def main() -> None:
    f = pd.read_csv(ROOT / "consolidado" / "scores_consolidados.csv")
    nc = pd.read_csv(ROOT / "consolidado" / "nao_casados.csv")
    agg = pd.read_csv(ROOT / "consolidado" / "scores" / "IRD" / "df_ird_agg_legislatura.csv")
    out: dict = {}

    out["meta"] = {
        "rows": int(len(f)),
        "deputados": int(f.id_camara.nunique()),
        "legs": [int(x) for x in sorted(f.legislatura.unique())],
        "ifs_profiles": int(f.ifs.notna().sum() + (nc.indice == "IFS").sum()),
        "ifs_matched": int(f.ifs.notna().sum()),
        "ifs_senators": int((nc.cargo == "senador").sum()),
        "co2": int(f.co2_index.notna().sum()),
        "ipipe": int(f.ipipe_id_parlamentar.notna().sum()),
        "ipipe_completo": int((f.ipipe_status == "SCORE_COMPLETO").sum()),
    }

    cov = f.groupby("legislatura")[list(IDX.values())].count()
    out["coverage"] = {
        "legs": [int(l) for l in cov.index],
        "base": [int(x) for x in f.groupby("legislatura").size()],
        "idx": {k: [int(x) for x in cov[c]] for k, c in IDX.items()},
    }

    out["hist"] = {
        "ipp": hist(f.ipp_final_v4, np.linspace(0, 1, 21)),
        "ird": hist(f.ird_peso, np.linspace(0, 7, 29)),
        "co2": hist(f.co2_index, np.linspace(-6, 9, 31), clip=(-6, 9)),
        "ipipe": hist(f.ipipe_score_exposicao, np.linspace(0, 300, 31), clip=(0, 300)),
        "ifs": hist(f.ifs, np.linspace(0.3, 1, 29)),
        "irr": hist(f.irr, np.linspace(0, 1, 21)),
    }
    out["hist"]["co2"]["note"] = "valores abaixo de -6 agrupados no primeiro intervalo"
    out["hist"]["ipipe"]["note"] = "valores acima de 300 agrupados no último intervalo"

    out["corr56"] = corr(f[f.legislatura == 56], [("ipp_final_v4", "IPP"), ("ird_peso", "IRD"), ("ipipe_score_exposicao", "IPIPE")])
    out["corr57"] = corr(f[f.legislatura == 57], [("ipp_final_v4", "IPP"), ("ird_peso", "IRD"), ("ipipe_score_exposicao", "IPIPE"), ("ifs", "IFS"), ("irr", "IRR")])

    out["party"] = {}
    out["rank"] = {}
    for k, c in IDX.items():
        out["party"][k] = {"min": MIN_PARTIDO[k], "legs": {}}
        out["rank"][k] = {}
        for leg, g in f.groupby("legislatura"):
            n = int(g[c].notna().sum())
            if not n:
                continue
            out["party"][k]["legs"][str(leg)] = {"n": n, "data": by_party(g, c, MIN_PARTIDO[k])}
            out["rank"][k][str(leg)] = {"n": n, "top": rank(g, c, False), "bottom": rank(g, c, True)}

    t = f.groupby("legislatura").agg(ipp_mean=("ipp_final_v4", "mean"), ipp_median=("ipp_final_v4", "median"), ird=("ird_ajuste_legislatura", "first")).reset_index()
    t = t.merge(agg[["legislatura", "%_F", "%_n_branca"]], on="legislatura", how="left")
    num = lambda v: None if pd.isna(v) else round(float(v), 4)
    out["time"] = [
        {"leg": int(r.legislatura), "ipp_mean": num(r.ipp_mean), "ipp_median": num(r.ipp_median), "ird": num(r.ird), "pF": num(r["%_F"]), "pNB": num(r["%_n_branca"])}
        for _, r in t.iterrows()
    ]

    # ---- Índice de Qualidade (indice_qualidade.py)
    iq = pd.read_csv(ROOT / "consolidado" / "indice_qualidade.csv")
    resumo = json.loads((ROOT / "consolidado" / "indice_qualidade_resumo.json").read_text(encoding="utf-8"))
    comp = iq.dropna(subset=["iq"])
    pts = comp.dropna(subset=["co2_index"])
    rob = pd.read_csv(ROOT / "consolidado" / "indice_qualidade_por_legislatura.csv")
    rob = rob[(rob.legislatura == 56)].dropna(subset=["co2_index", "iq_disponivel"])
    out["iq"] = {
        "resumo": resumo,
        "scatter2": [{"nome": s(r.nome), "partido": s(r.siglaPartido), "uf": s(r.siglaUf), "iq": round(float(r.iq_disponivel), 3), "co2": round(float(r.co2_index), 3), "nc": int(r.n_componentes)} for _, r in rob.iterrows()],
        "hist": hist(comp.iq, np.linspace(0, 1, 21)),
        "scatter": [{"nome": s(r.nome), "partido": s(r.siglaPartido), "uf": s(r.siglaUf), "iq": round(float(r.iq), 3), "co2": round(float(r.co2_index), 3)} for _, r in pts.iterrows()],
        "top": [{"nome": s(r.nome), "partido": s(r.siglaPartido), "uf": s(r.siglaUf), "v": round(float(r.iq), 3), "ipp": round(float(r.ipp_norm), 2), "ird": round(float(r.ird_norm), 2), "ipipe": round(float(r.ipipe_norm), 2), "ifs": round(float(r.ifs_norm), 2), "irr": round(float(r.irr_norm), 2)} for _, r in comp.head(10).iterrows()],
        "bottom": [{"nome": s(r.nome), "partido": s(r.siglaPartido), "uf": s(r.siglaUf), "v": round(float(r.iq), 3), "ipp": round(float(r.ipp_norm), 2), "ird": round(float(r.ird_norm), 2), "ipipe": round(float(r.ipipe_norm), 2), "ifs": round(float(r.ifs_norm), 2), "irr": round(float(r.irr_norm), 2)} for _, r in comp.tail(10).iloc[::-1].iterrows()],
    }

    # IQ por partido (57ª, mínimo de 8 deputados com IQ completo)
    out["party"]["iqc"] = {"min": 8, "legs": {"57": {"n": int(len(comp)), "data": by_party(comp, "iq", 8)}}}

    data = json.dumps(out, ensure_ascii=False, allow_nan=False)  # NaN quebraria o JSON.parse da página
    (HERE / "dados.json").write_text(data, encoding="utf-8")
    template = (HERE / "relatorio_template.html").read_text(encoding="utf-8")
    assert "__DATA__" in template
    (HERE / "relatorio.html").write_text(template.replace("__DATA__", data), encoding="utf-8")
    # cópia do banco consolidado publicada junto da página (botão de download)
    (HERE / "scores_consolidados.csv").write_bytes((ROOT / "consolidado" / "scores_consolidados.csv").read_bytes())
    print(f"relatorio.html gerado ({len(data):,} bytes de dados)")


if __name__ == "__main__":
    main()
