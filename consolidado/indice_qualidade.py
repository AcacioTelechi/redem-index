"""Índice de Qualidade (IQ) a partir dos scores consolidados.

Componentes: IPP, IRD (peso), IPIPE (exposição, invertido) e IFS. CO2 não entra; é o alvo
da correlação. Preparado para receber o IRS como quinto componente (basta adicionar em COMPONENTES).

Cada componente vira uma posição percentual (0 = pior, 1 = melhor) dentro da legislatura em
que foi medido, e o IQ é a média simples dos componentes.

Saídas:
  consolidado/indice_qualidade.csv                  - IQ dos deputados da 57ª legislatura (exige os 4
                                                      componentes), com o CO2-Index da 56ª dos reeleitos
  consolidado/indice_qualidade_por_legislatura.csv  - robustez: IQ por parlamentar-legislatura com os
                                                      componentes disponíveis na linha
  consolidado/indice_qualidade_resumo.json          - correlações e contagens usadas no relatório

Uso: python3 consolidado/indice_qualidade.py  (depois de consolidar_scores.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "consolidado"

# nome do componente -> (coluna em scores_consolidados.csv, maior é melhor?)
COMPONENTES = {
    "ipp": ("ipp_final_v4", True),
    "ird": ("ird_peso", True),  # peso maior = perfil sub-representado, contribui mais à representação descritiva
    "ipipe": ("ipipe_score_exposicao", False),  # mais exposição a processos = pior
    "ifs": ("ifs", True),
    # "irs": ("irs", True),  # quando houver dados
}
LEG_IQ = 57  # legislatura com os quatro componentes
LEG_CO2 = 56  # o CO2-Index só existe para a 56ª


def rank01(s: pd.Series, maior_melhor: bool) -> pd.Series:
    """Posição percentual em [0, 1] dentro da série (NaN fica NaN); 1 = melhor."""
    n = int(s.notna().sum())
    if n == 0:
        return s
    r = s.rank(pct=True, method="average")
    return r if maior_melhor else 1 - r + 1 / n


def normalizar_por_legislatura(f: pd.DataFrame) -> pd.DataFrame:
    """Adiciona colunas <comp>_norm, ranqueadas dentro de cada legislatura."""
    f = f.copy()
    for nome, (col, maior) in COMPONENTES.items():
        f[f"{nome}_norm"] = f.groupby("legislatura")[col].transform(lambda s: rank01(s, maior))
    return f


def corr_table(df: pd.DataFrame, x: str, cols: list[str]) -> list[dict]:
    rows = []
    for c in cols:
        d = df[[x, c]].dropna()
        if len(d) < 3:
            rows.append({"var": c, "n": len(d), "spearman": None, "p_spearman": None, "pearson": None, "p_pearson": None})
            continue
        sp = stats.spearmanr(d[x], d[c])
        pe = stats.pearsonr(d[x], d[c])
        rows.append({"var": c, "n": int(len(d)), "spearman": round(float(sp.statistic), 3), "p_spearman": round(float(sp.pvalue), 4),
                     "pearson": round(float(pe.statistic), 3), "p_pearson": round(float(pe.pvalue), 4)})
    return rows


def regressao(df: pd.DataFrame, x: str, y: str) -> dict:
    """OLS de y sobre x, com IC95% da inclinação, mais Pearson e Spearman."""
    d = df[[x, y]].dropna()
    n = len(d)
    lr = stats.linregress(d[x], d[y])
    tcrit = stats.t.ppf(0.975, n - 2)
    sp = stats.spearmanr(d[x], d[y])
    r = lambda v, k=4: round(float(v), k)
    return {"n": int(n), "slope": r(lr.slope), "intercept": r(lr.intercept), "stderr": r(lr.stderr),
            "ci_low": r(lr.slope - tcrit * lr.stderr), "ci_high": r(lr.slope + tcrit * lr.stderr),
            "r2": r(lr.rvalue ** 2), "p": r(lr.pvalue, 6), "pearson": r(lr.rvalue, 3),
            "spearman": r(sp.statistic, 3), "p_spearman": r(sp.pvalue),
            "x_min": r(d[x].min()), "x_max": r(d[x].max())}


def analise_fatorial(X: pd.DataFrame) -> dict:
    """Análise fatorial de 1 fator sobre os componentes normalizados (sensibilidade).

    Devolve os diagnósticos (Bartlett, KMO, autovalores), as cargas por componentes principais
    e por eixos principais, e os pesos = carga² normalizada (sem sinal, para não inverter a
    direção conceitual de nenhum componente).
    """
    R = X.corr().values
    n, p = len(X), R.shape[0]
    chi = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R))
    gl = p * (p - 1) / 2
    inv = np.linalg.inv(R)
    parcial = -inv / np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
    np.fill_diagonal(parcial, 0)
    r0 = R - np.eye(p)
    kmo_var = (r0**2).sum(0) / ((r0**2).sum(0) + (parcial**2).sum(0))
    kmo = (r0**2).sum() / ((r0**2).sum() + (parcial**2).sum())
    w, v = np.linalg.eigh(R)
    cp = v[:, -1] * np.sqrt(w[-1])
    cp *= np.sign(cp.sum())
    h = 1 - 1 / np.diag(inv)  # eixos principais, comunalidades iterativas
    for _ in range(1000):
        Rr = R.copy()
        np.fill_diagonal(Rr, h)
        w2, v2 = np.linalg.eigh(Rr)
        ep = v2[:, -1] * np.sqrt(max(w2[-1], 0))
        if np.abs(ep**2 - h).max() < 1e-9:
            break
        h = ep**2
    ep *= np.sign(ep.sum())
    pesos = cp**2 / (cp**2).sum()
    r = lambda a: [round(float(x), 4) for x in a]
    return {"n": int(n), "vars": list(X.columns), "corr": [r(row) for row in R],
            "bartlett_chi2": round(float(chi), 3), "bartlett_gl": int(gl), "bartlett_p": round(float(stats.chi2.sf(chi, gl)), 4),
            "kmo": round(float(kmo), 3), "kmo_var": r(kmo_var), "autovalores": r(w[::-1]),
            "var_explicada": round(float(w[-1] / p), 4), "cargas_cp": r(cp), "cargas_ep": r(ep), "pesos": r(pesos)}


CO2_OUTLIER = -10  # sensibilidade: sem os deputados com CO2-Index abaixo disso


def main() -> None:
    f = normalizar_por_legislatura(pd.read_csv(OUT / "scores_consolidados.csv"))
    normcols = [f"{n}_norm" for n in COMPONENTES]

    # ---- IQ principal: 57ª legislatura, os quatro componentes medidos na mesma legislatura.
    # O CO2-Index só existe para a 56ª; ele é trazido pelo id do deputado (reeleitos).
    base = f[f.legislatura == LEG_IQ].set_index("id_camara")
    co2 = f[f.legislatura == LEG_CO2].set_index("id_camara")["co2_index"].dropna()
    iq = base[["nome", "siglaPartido", "siglaUf"] + normcols].copy()
    iq["co2_index"] = iq.index.map(co2)
    iq["n_componentes"] = iq[normcols].notna().sum(axis=1)
    iq["iq_parcial"] = iq[normcols].mean(axis=1)  # média do que existe
    iq["iq"] = iq["iq_parcial"].where(iq["n_componentes"] == len(COMPONENTES))
    # sensibilidade: pesos derivados da análise fatorial (carga² normalizada)
    fat = analise_fatorial(iq.loc[iq["iq"].notna(), normcols])
    iq["iq_fatorial"] = (iq[normcols] * fat["pesos"]).sum(axis=1).where(iq["iq"].notna())
    iq = iq.reset_index().sort_values("iq", ascending=False)
    iq.to_csv(OUT / "indice_qualidade.csv", index=False)

    # ---- opção 2: IQ por parlamentar-legislatura com o que existe
    f["n_componentes"] = f[normcols].notna().sum(axis=1)
    f["iq_disponivel"] = f[normcols].mean(axis=1)
    f[["id_camara", "legislatura", "nome", "siglaPartido", "siglaUf", "co2_index"] + normcols + ["n_componentes", "iq_disponivel"]].to_csv(
        OUT / "indice_qualidade_por_legislatura.csv", index=False
    )

    # ---- correlações com o CO2 (56ª)
    completo = iq.dropna(subset=["iq"])
    resumo = {
        "componentes": list(COMPONENTES),
        "leg_iq": LEG_IQ, "leg_co2": LEG_CO2,
        "n_deputados": int(len(iq)),
        "n_iq_completo": int(len(completo)),
        "n_iq_completo_com_co2": int(completo.co2_index.notna().sum()),
        "n_componentes_dist": {str(k): int(v) for k, v in iq.n_componentes.value_counts().sort_index().items()},
        "iq_stats": {k: round(float(v), 4) for k, v in completo.iq.describe().items()},
        "corr_opcao1": corr_table(completo, "co2_index", ["iq"] + normcols),
        "corr_opcao2_leg56": corr_table(f[f.legislatura == LEG_CO2], "co2_index", ["iq_disponivel"] + normcols),
        "corr_iq_componentes": corr_table(completo, "iq", normcols),
        "co2_outlier": CO2_OUTLIER,
        "fatorial": {**fat, "corr_iq_igual": round(float(completo.iq.corr(completo.iq_fatorial)), 3),
                     "corr_iq_igual_spearman": round(float(completo.iq.corr(completo.iq_fatorial, method="spearman")), 3),
                     "reg_co2": regressao(completo, "iq_fatorial", "co2_index")},
        "reg": {
            "principal": regressao(completo, "iq", "co2_index"),
            "principal_sem_outlier": regressao(completo[completo.co2_index > CO2_OUTLIER], "iq", "co2_index"),
            "robustez": regressao(f[f.legislatura == LEG_CO2], "iq_disponivel", "co2_index"),
            "robustez_sem_outlier": regressao(f[(f.legislatura == LEG_CO2) & (f.co2_index > CO2_OUTLIER)], "iq_disponivel", "co2_index"),
        },
        "n_componentes_leg56": {str(k): int(v) for k, v in f[f.legislatura == LEG_CO2].n_componentes.value_counts().sort_index().items()},
    }
    (OUT / "indice_qualidade_resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"deputados na {LEG_IQ}ª: {len(iq)} | com os {len(COMPONENTES)} componentes: {len(completo)} | desses com CO2: {resumo['n_iq_completo_com_co2']}")
    print("componentes disponíveis por deputado:", resumo["n_componentes_dist"])
    print("\nopção 1 - correlação com CO2 (IQ completo):")
    print(pd.DataFrame(resumo["corr_opcao1"]).to_string(index=False))
    print("\nopção 2 - correlação com CO2 (56ª, componentes disponíveis):")
    print(pd.DataFrame(resumo["corr_opcao2_leg56"]).to_string(index=False))
    print("\nregressões CO2 ~ IQ:")
    print(pd.DataFrame(resumo["reg"]).T.to_string())
    print("\nanálise fatorial (sensibilidade):", {k: v for k, v in resumo["fatorial"].items() if k not in ("corr", "reg_co2")})
    print("\nIQ x componentes:")
    print(pd.DataFrame(resumo["corr_iq_componentes"]).to_string(index=False))


if __name__ == "__main__":
    main()
