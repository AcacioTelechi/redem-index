"""Índice de Qualidade (IQ) a partir dos scores consolidados.

Componentes: IPP, IRD (peso), IPIPE (exposição, invertido), IFS e IRR. CO2 não entra; é o alvo
da correlação. Novos componentes entram adicionando uma linha em COMPONENTES.

Cada componente vira uma posição percentual (0 = pior, 1 = melhor) dentro da legislatura em
que foi medido, e o IQ é a média geométrica dos componentes, reescalados para [0,1; 1] (o piso).
A média aritmética fica só como análise de sensibilidade (coluna iq_aritmetica).

Saídas:
  consolidado/indice_qualidade.csv                  - IQ dos deputados da 57ª legislatura (exige todos os
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
    "irr": ("irr", True),
}
LEG_IQ = 57  # legislatura com todos os componentes
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


PISO_GEOMETRICA = 0.1  # cada componente é reescalado para [piso, 1] antes da média geométrica


def media_geometrica(X: pd.DataFrame, piso: float = PISO_GEOMETRICA, pesos: list[float] | None = None) -> pd.Series:
    """Média geométrica (ponderada, se houver pesos) dos componentes reescalados para [piso, 1].

    É a forma de agregação do IQ: um desempenho muito baixo em um componente não é compensado
    pelos demais, e o piso impede que um último lugar (posição percentual perto de zero) anule
    o índice. Componentes ausentes são ignorados; quem chama decide se exige todos.
    """
    L = np.log(piso + (1 - piso) * X)
    if pesos is None:
        return np.exp(L.mean(axis=1))
    w = pd.Series(pesos, index=X.columns, dtype=float)
    return np.exp((L * w).sum(axis=1, min_count=1) / (L.notna() * w).sum(axis=1))


def corr_table(df: pd.DataFrame, x: str, cols: list[str]) -> list[dict]:
    rows = []
    for c in cols:
        d = df[[x, c]].dropna()
        if len(d) < 3:
            rows.append({"var": c, "n": len(d), "spearman": None, "p_spearman": None, "pearson": None, "p_pearson": None})
            continue
        sp = stats.spearmanr(d[x], d[c])
        pe = stats.pearsonr(d[x], d[c])
        rows.append({"var": c, "n": int(len(d)), "spearman": round(float(sp.statistic), 3), "p_spearman": round(float(sp.pvalue), 6),
                     "pearson": round(float(pe.statistic), 3), "p_pearson": round(float(pe.pvalue), 6)})
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
            "spearman": r(sp.statistic, 3), "p_spearman": r(sp.pvalue, 6),
            "x_min": r(d[x].min()), "x_max": r(d[x].max())}


def vif(X: pd.DataFrame) -> list[float]:
    """Fator de inflação da variância de cada coluna (diagonal da inversa da matriz de correlações)."""
    return [round(float(v), 3) for v in np.diag(np.linalg.inv(X.corr().values))]


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


def senioridade(ids: pd.Index) -> pd.DataFrame:
    """Por deputado: nº de eleições ordinárias disputadas até 2022 (TSE, via CPF) e ln do tempo de casa (IPP)."""
    det = pd.read_csv(ROOT / "data" / "df_parl_detalhes.csv")[["id", "cpf"]].drop_duplicates("id").set_index("id")
    tse = pd.read_csv(ROOT / "data" / "df_tse_candidatos.csv", low_memory=False)
    tse = tse[(tse.ano <= 2022) & (tse.tipo_eleicao == "eleicao ordinaria")]
    n_el = tse.groupby("cpf").ano.nunique()
    ipp = pd.read_excel(OUT / "scores" / "IPP" / "df_ipp_v4.xlsx")
    tempo = ipp[ipp.idLegislatura == LEG_IQ].set_index("id")["tempo_atuacao_dias_cum_ln"]
    return pd.DataFrame({"n_eleicoes": det.cpf.reindex(ids).map(n_el).values, "ln_tempo_casa": tempo.reindex(ids).values}, index=ids)


def agregacoes(c: pd.DataFrame, normcols: list[str]) -> dict:
    """Compara o IQ (geométrica com piso) com a média aritmética e a geométrica pura."""
    d = c.copy()
    d["menor_componente"] = d[normcols].min(axis=1)
    pos = lambda k: d[k].rank(ascending=False)
    top = lambda k: set(d.nlargest(10, k).id_camara)
    bot = lambda k: set(d.nsmallest(10, k).id_camara)
    versoes = {}
    for k in ["iq", "iq_aritmetica", "iq_geometrica"]:
        mov = (pos("iq") - pos(k)).abs()
        versoes[k] = {
            "min": round(float(d[k].min()), 3), "mediana": round(float(d[k].median()), 3), "max": round(float(d[k].max()), 3),
            "corr_iq": round(float(d.iq.corr(d[k], method="spearman")), 3),
            "mov_mediana": float(mov.median()), "mov_max": float(mov.max()),
            "top10_comum": len(top("iq") & top(k)), "bottom10_comum": len(bot("iq") & bot(k)),
            "corr_menor_componente": round(float(d[k].corr(d.menor_componente, method="spearman")), 3),
            "reg_co2": regressao(d, k, "co2_index"),
        }
    d["delta"] = pos("iq_aritmetica") - pos("iq")  # positivo = posição melhor no IQ do que na média aritmética
    linha = lambda r: {"nome": str(r.nome), "partido": str(r.siglaPartido), "uf": str(r.siglaUf), "arit": round(float(r.iq_aritmetica), 2),
                       "iq": round(float(r.iq), 2), "delta": int(round(r.delta)),
                       "comp": [round(float(getattr(r, n)), 2) for n in normcols]}
    g = d.groupby("siglaPartido").agg(n=("iq", "size"), iq=("iq", "mean"), arit=("iq_aritmetica", "mean"), geo=("iq_geometrica", "mean"))
    g = g[g.n >= 8].sort_values("iq", ascending=False)
    return {"piso": PISO_GEOMETRICA, "n": int(len(d)), "menor_valor_componente": round(float(d[normcols].min().min()), 4), "versoes": versoes,
            "sobem": [linha(r) for r in d.nlargest(5, "delta").itertuples()],
            "caem": [linha(r) for r in d.nsmallest(5, "delta").itertuples()],
            "partidos": [{"partido": str(i), "n": int(r.n), "iq": round(float(r.iq), 3), "arit": round(float(r.arit), 3), "geo": round(float(r.geo), 3)} for i, r in g.iterrows()]}


def mazziotta_pareto(M: pd.DataFrame) -> pd.Series:
    """Índice de Mazziotta-Pareto com penalidade negativa: média dos componentes padronizados
    (média 100, desvio 10) menos desvio x coeficiente de variação entre os componentes da linha."""
    Z = 100 + 10 * (M - M.mean()) / M.std(ddof=0)
    m, s = Z.mean(axis=1), Z.std(axis=1, ddof=0)
    return m - s * (s / m)


def robustez_agregacao(c: pd.DataFrame, normcols: list[str]) -> dict:
    """Sensibilidade do IQ ao valor do piso e comparação com a aritmética e o Mazziotta-Pareto."""
    d = c.copy()
    X = d[normcols]
    ref = d["iq"]
    pos = lambda s: s.rank(ascending=False)
    top = lambda s: set(d.loc[s.nlargest(10).index, "id_camara"])
    bot = lambda s: set(d.loc[s.nsmallest(10).index, "id_camara"])
    rho = lambda a, b: round(float(a.corr(b, method="spearman")), 3)
    pisos = []
    for p in [0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        s = media_geometrica(X, piso=p)
        d["_s"] = s
        r = regressao(d, "_s", "co2_index")
        pisos.append({"piso": p, "rho_ref": rho(s, ref), "rho_aritm": rho(s, d.iq_aritmetica), "mov_max": float((pos(s) - pos(ref)).abs().max()),
                      "top10": len(top(s) & top(ref)), "bottom10": len(bot(s) & bot(ref)), "co2_spearman": r["spearman"], "co2_p": r["p_spearman"]})
    d["dp"] = X.std(axis=1)
    metodos = {}
    for k in ["iq", "iq_aritmetica", "iq_mazziotta_pareto"]:
        r = regressao(d, k, "co2_index")
        metodos[k] = {"rho_iq": rho(d[k], ref), "rho_aritm": rho(d[k], d.iq_aritmetica), "mov_max": float((pos(d[k]) - pos(ref)).abs().max()),
                      "top10_iq": len(top(d[k]) & top(ref)), "bottom10_iq": len(bot(d[k]) & bot(ref)),
                      "rho_desequilibrio": rho(d[k], d.dp), "co2_spearman": r["spearman"], "co2_p": r["p_spearman"]}
    g = d.groupby("siglaPartido").agg(n=("iq", "size"), iq=("iq", "mean"), arit=("iq_aritmetica", "mean"), mp=("iq_mazziotta_pareto", "mean"))
    g = g[g.n >= 8]
    ordem = {k: list(g.sort_values(k, ascending=False).index) for k in ["iq", "arit", "mp"]}
    return {"pisos": pisos, "metodos": metodos, "partidos_ordem": ordem}


CO2_OUTLIER = -10  # sensibilidade: sem os deputados com CO2-Index abaixo disso


def sens_ipipe(c: pd.DataFrame) -> dict:
    """Resumo da sensibilidade do IQ ao IPIPE ajustado por eleição disputada."""
    d = c.dropna(subset=["iq_ipipe_ajustado"]).copy()
    sp = lambda a, b: round(float(d[a].corr(d[b], method="spearman")), 3)
    pos, pos_adj = d.iq.rank(ascending=False), d.iq_ipipe_ajustado.rank(ascending=False)
    d["delta"] = pos - pos_adj  # positivo = sobe com o ajuste
    linha = lambda r: {"nome": str(r.nome), "partido": str(r.siglaPartido), "uf": str(r.siglaUf), "n_eleicoes": int(r.n_eleicoes),
                       "iq": round(float(r.iq), 3), "iq_adj": round(float(r.iq_ipipe_ajustado), 3), "delta": int(round(r.delta))}
    top = lambda k: set(d.nlargest(10, k).id_camara)
    bot = lambda k: set(d.nsmallest(10, k).id_camara)
    return {
        "n": int(len(d)),
        "corr_ipipe": {"tempo": [sp("ipipe_norm", "ln_tempo_casa"), sp("ipipe_adj_norm", "ln_tempo_casa")],
                       "ipp": [sp("ipipe_norm", "ipp_norm"), sp("ipipe_adj_norm", "ipp_norm")],
                       "n_eleicoes": [sp("ipipe_norm", "n_eleicoes"), sp("ipipe_adj_norm", "n_eleicoes")]},
        "corr_ipipe_original_ajustado": sp("ipipe_norm", "ipipe_adj_norm"),
        "corr_iq": sp("iq", "iq_ipipe_ajustado"),
        "iq_tempo": [sp("iq", "ln_tempo_casa"), sp("iq_ipipe_ajustado", "ln_tempo_casa")],
        "mov_mediana": float((pos - pos_adj).abs().median()), "mov_p90": float((pos - pos_adj).abs().quantile(0.9)),
        "top10_comum": len(top("iq") & top("iq_ipipe_ajustado")), "bottom10_comum": len(bot("iq") & bot("iq_ipipe_ajustado")),
        "sobem": [linha(r) for r in d.nlargest(5, "delta").itertuples()],
        "caem": [linha(r) for r in d.nsmallest(5, "delta").itertuples()],
        "reg_co2": regressao(d, "iq_ipipe_ajustado", "co2_index"),
    }


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
    completo_mask = iq["n_componentes"] == len(COMPONENTES)
    # IQ = média geométrica com piso dos componentes; exige todos
    iq["iq_parcial"] = media_geometrica(iq[normcols])  # com os componentes que existem
    iq["iq"] = iq["iq_parcial"].where(completo_mask)
    # formas alternativas de agregação, só para a análise de sensibilidade
    iq["iq_aritmetica"] = iq[normcols].mean(axis=1).where(completo_mask)
    iq["iq_geometrica"] = media_geometrica(iq[normcols], piso=0).where(completo_mask)
    iq["iq_mazziotta_pareto"] = mazziotta_pareto(iq.loc[completo_mask, normcols])
    # sensibilidade: pesos derivados da análise fatorial (carga² normalizada)
    fat = analise_fatorial(iq.loc[completo_mask, normcols])
    ids_completos = iq.index[completo_mask]
    fat["vif"] = vif(iq.loc[ids_completos, normcols])
    fat["vif_original"] = vif(base.loc[ids_completos, [col for col, _ in COMPONENTES.values()]])
    iq["iq_fatorial"] = media_geometrica(iq[normcols], pesos=fat["pesos"]).where(completo_mask)
    # sensibilidade: IPIPE por eleição disputada (propensão a processos, não tempo de exposição)
    iq = iq.join(senioridade(iq.index))
    iq["ipipe_adj_norm"] = rank01(base["ipipe_score_exposicao"] / iq["n_eleicoes"], False)
    cols_adj = [c if c != "ipipe_norm" else "ipipe_adj_norm" for c in normcols]
    iq["iq_ipipe_ajustado"] = media_geometrica(iq[cols_adj]).where(completo_mask & iq["ipipe_adj_norm"].notna())
    iq = iq.reset_index().sort_values("iq", ascending=False)
    iq.to_csv(OUT / "indice_qualidade.csv", index=False)

    # ---- opção 2: IQ por parlamentar-legislatura com o que existe
    f["n_componentes"] = f[normcols].notna().sum(axis=1)
    f["iq_disponivel"] = media_geometrica(f[normcols])  # mesma agregação, com os componentes da linha
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
        "ipipe_ajustado": sens_ipipe(completo),
        "agregacao": {**agregacoes(completo, normcols), "robustez": robustez_agregacao(completo, normcols)},
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
