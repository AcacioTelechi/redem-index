"""Validação do Índice de Qualidade (IQ).

Teste discriminante: o IQ não deveria depender de características que não são qualidade
parlamentar (tempo de casa, idade, escolaridade, ideologia, região, tamanho do partido).
Gênero e raça entram no IQ por desenho, através do IRD, e são reportados à parte.

Validade preditiva: o IQ contra o resultado dos deputados nas eleições de 2026 (reeleição para a
Câmara e candidaturas ao Senado), a partir de consolidado/scores/TSE/eleicoes_deputados_57.csv,
gerado por preparar_tse.py.

Validade convergente: o IQ contra as listas "Cabeças do Congresso" do DIAP (2023 a 2025), a partir de
consolidado/scores/DIAP/diap_cabecas.csv, gerado por preparar_diap.py.

Saídas:
  consolidado/validacao_caracteristicas.csv  - deputados com IQ completo e suas características
  consolidado/validacao_resumo.json          - resultados usados no relatório

Uso: python3 consolidado/validacao.py  (depois de indice_qualidade.py)
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "consolidado"
LEG = 57
COMPS = ["ipp_norm", "ird_norm", "ipipe_norm", "ifs_norm", "irr_norm"]
INDICES = ["iq"] + COMPS
REGIOES = {"Norte": ["AC", "AP", "AM", "PA", "RO", "RR", "TO"], "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
           "Centro-Oeste": ["DF", "GO", "MT", "MS"], "Sudeste": ["ES", "MG", "RJ", "SP"], "Sul": ["PR", "RS", "SC"]}


def norm(s: object) -> str | None:
    if pd.isna(s):
        return None
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z ]", " ", s)).strip()


def caracteristicas() -> pd.DataFrame:
    """Deputados com IQ completo + ideologia, gênero, raça (arquivo do IPIPE), idade e escolaridade (planilha do IRR)."""
    iq = pd.read_csv(OUT / "indice_qualidade.csv")
    sc = pd.read_csv(OUT / "scores_consolidados.csv")
    sc = sc[sc.legislatura == LEG][["id_camara", "ipipe_id_parlamentar"]]
    ip = pd.read_csv(OUT / "scores" / "IPIPE" / f"IPIPE_{LEG}A_PARLAMENTARES.csv")
    irr = pd.read_excel(OUT / "scores" / "IRR" / "V6_agregado_ponderado_final_IRR_ponderado_57.xlsx", sheet_name="Agregado_Ponderado")
    ip["k"] = ip.nome.map(norm)
    irr["k"] = irr.PARLAMENTAR.map(norm)
    ip = ip.merge(irr[["k", "IDADE", "INSTRUCAO"]], on="k", how="left")
    a = sc.merge(ip[["id_parlamentar", "IDEOLOGIA", "IDEOLOGIA_MATIZADA", "RACA", "GENERO", "reeleito", "IDADE", "INSTRUCAO"]],
                 left_on="ipipe_id_parlamentar", right_on="id_parlamentar", how="left")
    c = iq.merge(a, on="id_camara", how="left").dropna(subset=["iq"]).copy()
    c["mulher"] = (c.GENERO == "F").astype(int)
    c["nao_branco"] = (c.RACA.str.upper() != "BRANCA").astype(int)
    c["esquerda"] = (c.IDEOLOGIA == "esquerda").astype(int)
    c["regiao"] = c.siglaUf.map({uf: r for r, ufs in REGIOES.items() for uf in ufs})
    c["tam_partido"] = c.siglaPartido.map(iq.groupby("siglaPartido").size())  # bancada na 57ª
    c["iq_sem_ird"] = c[[k for k in COMPS if k != "ird_norm"]].mean(axis=1)
    return c


def r2(c: pd.DataFrame, y: str, cols: list[str]) -> float:
    X = pd.get_dummies(c[cols], drop_first=True, dtype=float)
    X.insert(0, "const", 1.0)
    res = c[y].values - X.values @ np.linalg.lstsq(X.values, c[y].values, rcond=None)[0]
    return round(float(1 - res.var() / c[y].var()), 4)


def eta2(x: pd.Series, g: pd.Series) -> tuple[float, float]:
    grupos = [x[g == k].dropna() for k in pd.unique(g.dropna())]
    grupos = [q for q in grupos if len(q) >= 5]
    h, p = stats.kruskal(*grupos)
    n = sum(map(len, grupos))
    return round(float(max(0.0, (h - len(grupos) + 1) / (n - len(grupos)))), 4), round(float(p), 6)


def medias(c: pd.DataFrame, col: str, rotulos: dict | None = None) -> list[dict]:
    g = c.groupby(col).agg(n=("iq", "size"), iq=("iq", "mean"), iq_sem_ird=("iq_sem_ird", "mean")).reset_index()
    return [{"grupo": (rotulos or {}).get(r[col], str(r[col])), "n": int(r.n), "iq": round(float(r.iq), 3), "iq_sem_ird": round(float(r.iq_sem_ird), 3)}
            for _, r in g.sort_values("iq", ascending=False).iterrows()]


def discriminante(c: pd.DataFrame) -> dict:
    numericas = {"ln_tempo_casa": "Tempo de casa", "n_eleicoes": "Eleições disputadas", "IDADE": "Idade", "tam_partido": "Tamanho do partido"}
    categoricas = {"INSTRUCAO": "Escolaridade", "IDEOLOGIA_MATIZADA": "Ideologia (6 níveis)", "regiao": "Região", "GENERO": "Gênero", "RACA_BIN": "Raça (branca ou não)"}
    c = c.assign(RACA_BIN=c.nao_branco.map({0: "branca", 1: "não branca"}))
    num = []
    for v, nome in numericas.items():
        linha = {"var": nome}
        for k in INDICES:
            d = c[[v, k]].dropna()
            sp = stats.spearmanr(d[v], d[k])
            linha[k] = {"rho": round(float(sp.statistic), 3), "p": round(float(sp.pvalue), 6)}
        num.append(linha)
    cat = []
    for v, nome in categoricas.items():
        linha = {"var": nome}
        for k in INDICES:
            e, p = eta2(c[k], c[v])
            linha[k] = {"eta2": e, "p": p}
        cat.append(linha)
    blocos = {"Gênero e raça": ["mulher", "nao_branco"], "Região": ["regiao"], "Ideologia (6 níveis)": ["IDEOLOGIA_MATIZADA"],
              "Tamanho do partido": ["tam_partido"], "Tempo de casa e idade": ["ln_tempo_casa", "IDADE"], "Escolaridade": ["INSTRUCAO"]}
    todos = sum(blocos.values(), [])
    sem_gr = [v for v in todos if v not in ("mulher", "nao_branco")]
    r2s = [{"bloco": nome, "iq": r2(c, "iq", cols), "iq_sem_ird": r2(c, "iq_sem_ird", cols)} for nome, cols in blocos.items()]
    sem_pl = c[c.siglaPartido != "PL"]
    X = np.column_stack([np.ones(len(c)), c.mulher, c.nao_branco])
    res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    return {
        "n": int(len(c)), "numericas": num, "categoricas": cat, "r2_blocos": r2s,
        "r2_todos": {"iq": r2(c, "iq", todos), "iq_sem_ird": r2(c, "iq_sem_ird", todos)},
        "r2_sem_genero_raca": {"iq": r2(c, "iq", sem_gr), "iq_sem_ird": r2(c, "iq_sem_ird", sem_gr)},
        "medias": {"genero": medias(c, "GENERO", {"F": "Mulheres", "M": "Homens"}), "raca": medias(c, "RACA_BIN", {"branca": "Brancos", "não branca": "Não brancos"}),
                   "ideologia": medias(c, "IDEOLOGIA", {"esquerda": "Esquerda", "direita": "Direita"}), "regiao": medias(c, "regiao"),
                   "reeleito": medias(c, "reeleito", {"SIM": "Reeleitos em 2022", "NÃO": "Novatos em 2022"})},
        "ideologia": {"d_cohen": round(float((c[c.esquerda == 1].iq.mean() - c[c.esquerda == 0].iq.mean()) / c.iq.std()), 3),
                      "r": round(float(c.iq.corr(c.esquerda)), 3),
                      "r_parcial_genero_raca": round(float(np.corrcoef(res(c.iq.values), res(c.esquerda.values.astype(float)))[0, 1]), 3),
                      "pct_mulheres_esquerda": round(float(c[c.esquerda == 1].mulher.mean()), 3), "pct_mulheres_direita": round(float(c[c.esquerda == 0].mulher.mean()), 3)},
        "partido": {"rho": round(float(stats.spearmanr(c.tam_partido, c.iq).statistic), 3),
                    "rho_sem_pl": round(float(stats.spearmanr(sem_pl.tam_partido, sem_pl.iq).statistic), 3)},
    }


def auc(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Probabilidade de um caso com y=1 ter x maior que um com y=0 (0,5 = acaso), e p de Mann-Whitney."""
    a, b = x[y == 1], x[y == 0]
    u = stats.mannwhitneyu(a, b)
    return round(float(u.statistic / (len(a) * len(b))), 3), round(float(u.pvalue), 6)


def compara(d: pd.DataFrame, y: str, extra: str | None = None) -> list[dict]:
    """IQ e componentes: média com y=1 e y=0, AUC e, se pedido, correlação de postos com `extra`."""
    linhas = []
    for k in INDICES:
        a, p = auc(d[k], d[y])
        linha = {"var": k, "sim": round(float(d.loc[d[y] == 1, k].mean()), 3), "nao": round(float(d.loc[d[y] == 0, k].mean()), 3), "auc": a, "p": p}
        if extra:
            e = d[[k, extra]].dropna()
            sp = stats.spearmanr(e[k], e[extra])
            linha.update({"rho": round(float(sp.statistic), 3), "p_rho": round(float(sp.pvalue), 6), "n_rho": int(len(e))})
        linhas.append(linha)
    return linhas


def eleicao_2026(c: pd.DataFrame) -> dict:
    """Validade preditiva: o IQ (medido na 57ª) contra o resultado dos deputados nas eleições de 2026."""
    e = pd.read_csv(OUT / "scores" / "TSE" / "eleicoes_deputados_57.csv")
    d = c.merge(e, on="id_camara", how="left")
    eleito = lambda s: s.fillna("").str.startswith("ELEITO")
    d["titular_2022"] = eleito(d.resultado_2022)
    d["eleito_2026"] = eleito(d.resultado_2026).astype(int)
    fed = d.cargo_2026 == "DEPUTADO FEDERAL"
    disputou = fed & d.votos_2026.notna() & (d.resultado_2026 != "#NULO")
    d["destino"] = np.select(
        [fed & disputou & (d.eleito_2026 == 1), fed & disputou, fed, d.cargo_2026 == "SENADOR", d.cargo_2026 == "NÃO CONCORREU"],
        ["Deputado federal: eleito", "Deputado federal: não eleito", "Deputado federal: candidatura sem votos", "Senado", "Não concorreu"], "Outro cargo")
    destinos = [{"destino": k, "n": int(len(g)), "iq": round(float(g.iq.mean()), 3)} for k, g in d.groupby("destino")]
    ordem = ["Deputado federal: eleito", "Deputado federal: não eleito", "Senado", "Outro cargo", "Não concorreu", "Deputado federal: candidatura sem votos"]
    destinos.sort(key=lambda r: ordem.index(r["destino"]))

    # reeleição em sentido estrito: eleito em 2022, candidato a deputado federal em 2026 no mesmo estado
    amplo = d[disputou]
    r = amplo[amplo.titular_2022 & (amplo.uf_2026 == amplo.uf_2022)].copy()
    r["var_votos"] = np.log(r.participacao_2026 / r.participacao_2022)
    r["quartil"] = pd.qcut(r.iq, 4, labels=["1º (menor IQ)", "2º", "3º", "4º (maior IQ)"])
    a_amplo, p_amplo = auc(amplo.iq, amplo.eleito_2026)
    X = np.column_stack([np.ones(len(r)), r.iq, r.ln_tempo_casa, np.log(r.participacao_2022)])
    b = np.linalg.lstsq(X, r.var_votos, rcond=None)[0]
    res = r.var_votos - X @ b
    ep = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * res.var(ddof=X.shape[1]))
    reeleicao = {
        "n": int(len(r)), "eleitos": int(r.eleito_2026.sum()), "taxa": round(float(r.eleito_2026.mean()), 3),
        "n_amplo": int(len(amplo)), "taxa_amplo": round(float(amplo.eleito_2026.mean()), 3), "auc_amplo": a_amplo, "p_amplo": p_amplo,
        "n_suplentes": int((~amplo.titular_2022).sum()), "n_poucos_votos": int((amplo.votos_2026 < 1000).sum()),
        "comparacao": compara(r, "eleito_2026", "var_votos"),
        "quartis": [{"quartil": str(k), "n": int(len(g)), "taxa": round(float(g.eleito_2026.mean()), 3)} for k, g in r.groupby("quartil", observed=True)],
        "var_votos_mediana": round(float(np.exp(r.var_votos.median()) - 1), 4), "pct_cresceram": round(float((r.var_votos > 0).mean()), 3),
        "ols_iq_t": round(float(b[1] / ep[1]), 2),
    }

    # Senado
    s = d[d.cargo_2026 == "SENADOR"].copy()
    rng = np.random.default_rng(0)
    obs = s[s.eleito_2026 == 1].iq.mean() - s[s.eleito_2026 == 0].iq.mean()
    x, y = s.iq.values, s.eleito_2026.values
    sim = np.array([(lambda q: x[q == 1].mean() - x[q == 0].mean())(rng.permutation(y)) for _ in range(20000)])
    direita = s[s.IDEOLOGIA == "direita"]
    a_dir, p_dir = auc(direita.iq, direita.eleito_2026)
    senado = {
        "n": int(len(s)), "eleitos": int(s.eleito_2026.sum()), "comparacao": compara(s, "eleito_2026", "participacao_2026"),
        "p_permutacao": round(float((np.abs(sim) >= abs(obs)).mean()), 4),
        "esquerda": {"n": int((s.IDEOLOGIA == "esquerda").sum()), "eleitos": int(s[s.IDEOLOGIA == "esquerda"].eleito_2026.sum()), "iq": round(float(s[s.IDEOLOGIA == "esquerda"].iq.mean()), 3)},
        "direita": {"n": int(len(direita)), "eleitos": int(direita.eleito_2026.sum()), "iq": round(float(direita.iq.mean()), 3),
                    "iq_eleitos": round(float(direita[direita.eleito_2026 == 1].iq.mean()), 3), "iq_nao": round(float(direita[direita.eleito_2026 == 0].iq.mean()), 3), "auc": a_dir, "p": p_dir},
        "candidatos": [{"nome": str(q.nome), "partido": str(q.partido_2026), "uf": str(q.uf_2026), "iq": round(float(q.iq), 3), "eleito": int(q.eleito_2026),
                        "participacao": None if pd.isna(q.participacao_2026) else round(float(q.participacao_2026), 4)} for q in s.sort_values("iq", ascending=False).itertuples()],
        "vs_reeleicao": [{"var": k, "senado": round(float(s[k].mean()), 3), "reeleicao": round(float(amplo[k].mean()), 3),
                          "p": round(float(stats.mannwhitneyu(s[k], amplo[k]).pvalue), 6)} for k in INDICES],
    }
    return {"n": int(len(d)), "destinos": destinos, "reeleicao": reeleicao, "senado": senado}


def diap(c: pd.DataFrame) -> dict:
    """Validade convergente: o IQ contra o reconhecimento pelo DIAP (Cabeças do Congresso, 2023 a 2025)."""
    d = pd.read_csv(OUT / "scores" / "DIAP" / "diap_cabecas.csv")
    d = d[d.cargo == "deputado"].dropna(subset=["id_camara"])
    anos = sorted(int(a) for a in d.ano.unique())
    t = d.pivot_table(index="id_camara", columns="lista", values="ano", aggfunc="count", fill_value=0)
    c = c.merge(t, left_on="id_camara", right_index=True, how="left").fillna({"cabeca": 0, "ascensao": 0})
    c["e_cabeca"] = (c.cabeca > 0).astype(int)
    c["e_150"] = ((c.cabeca + c.ascensao) > 0).astype(int)
    c["iq_sem_ipp"] = c[[k for k in COMPS if k != "ipp_norm"]].mean(axis=1)
    variaveis = INDICES + ["ln_tempo_casa", "iq_sem_ipp", "iq_sem_ird", "iq_geometrica_piso"]

    def tabela(y: str) -> list[dict]:
        linhas = []
        for k in variaveis:
            a, p = auc(c[k], c[y])
            linhas.append({"var": k, "sim": round(float(c.loc[c[y] == 1, k].mean()), 3), "nao": round(float(c.loc[c[y] == 0, k].mean()), 3), "auc": a, "p": p})
        return linhas

    def quartis(k: str) -> list[dict]:
        q = pd.qcut(c[k], 4, labels=["1º (menor)", "2º", "3º", "4º (maior)"])
        return [{"quartil": str(i), "n": int(len(g)), "taxa": round(float(g.e_cabeca.mean()), 3)} for i, g in c.groupby(q, observed=True)]

    return {
        "anos": anos, "n": int(len(c)), "n_cabecas": int(c.e_cabeca.sum()), "n_150": int(c.e_150.sum()),
        "deputados_por_edicao": {str(a): int(((d.ano == a) & (d.lista == "cabeca")).sum()) for a in anos},
        "anos_como_cabeca": {str(int(k)): int(v) for k, v in c.cabeca.value_counts().sort_index().items()},
        "cabecas": tabela("e_cabeca"), "top150": tabela("e_150"),
        "quartis_iq": quartis("iq"), "quartis_ipp": quartis("ipp_norm"),
        "tempo_casa_iq_sem_ird": round(float(stats.spearmanr(c.iq_sem_ird, c.ln_tempo_casa).statistic), 3),
        "tempo_casa_iq": round(float(stats.spearmanr(c.iq, c.ln_tempo_casa).statistic), 3),
    }


def main() -> None:
    c = caracteristicas()
    c.to_csv(OUT / "validacao_caracteristicas.csv", index=False)
    resumo = {"discriminante": discriminante(c), "eleicao": eleicao_2026(c), "diap": diap(c)}
    (OUT / "validacao_resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    d = resumo["discriminante"]
    print(f"teste discriminante, n = {d['n']}")
    print(pd.DataFrame(d["r2_blocos"]).to_string(index=False))
    print("todas juntas:", d["r2_todos"], "| sem gênero e raça:", d["r2_sem_genero_raca"])
    print("ideologia:", d["ideologia"], "| partido:", d["partido"])
    e = resumo["eleicao"]
    print("\neleição de 2026:", e["destinos"])
    print("reeleição:", {k: v for k, v in e["reeleicao"].items() if k not in ("comparacao", "quartis")})
    print(pd.DataFrame(e["reeleicao"]["comparacao"]).to_string(index=False))
    print("\nDIAP:", {k: v for k, v in resumo["diap"].items() if k not in ("cabecas", "top150")})
    print(pd.DataFrame(resumo["diap"]["cabecas"]).to_string(index=False))
    print("Senado:", {k: v for k, v in e["senado"].items() if k not in ("comparacao", "candidatos", "vs_reeleicao")})
    print(pd.DataFrame(e["senado"]["comparacao"]).to_string(index=False))


if __name__ == "__main__":
    main()
