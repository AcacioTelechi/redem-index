"""Consolida os scores (IPP, IRD, CO2, IPIPE, IFS) por parlamentar-legislatura.

Base: data/df_parlamentares_por_legislatura.csv (uma linha por id da Câmara x legislatura).
Cada índice é juntado por left join. Índices sem id da Câmara (CO2, IPIPE, IFS) são
casados por nome normalizado (sem acentos, caixa alta); o que sobrar pode ser resolvido
à mão em consolidado/depara_nomes.csv (colunas: indice, nome_origem, id_camara).

Saídas:
  consolidado/scores_consolidados.csv  - tabela final
  consolidado/nao_casados.csv          - linhas de cada índice que não acharam parlamentar

Uso: python3 consolidado/consolidar_scores.py   (precisa de pandas + openpyxl)
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SCORES = ROOT / "consolidado" / "scores"
OUT_DIR = ROOT / "consolidado"
DEPARA_PATH = OUT_DIR / "depara_nomes.csv"

LEG_CO2 = 56  # CO2-INDEX cobre a 56a legislatura (2019-2023)
LEG_IPIPE = 56  # arquivos IPIPE_56A_*
LEG_IFS = 57  # postagens de 2023


# --------------------------------------------------------------------------- helpers
def norm(s: object) -> str | None:
    """Nome sem acentos, caixa alta, só letras e espaços simples."""
    if s is None or (isinstance(s, float) and pd.isna(s)) or pd.isna(s):
        return None
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    s = re.sub(r"[^A-Z ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def load_depara() -> dict[tuple[str, str], int]:
    if not DEPARA_PATH.exists():
        return {}
    d = pd.read_csv(DEPARA_PATH, dtype=str).dropna(subset=["id_camara"])
    return {(r.indice, norm(r.nome_origem)): int(r.id_camara) for r in d.itertuples()}


def unique_map(df: pd.DataFrame, key: str, value: str) -> dict:
    """Mapa key -> value apenas para chaves que não são ambíguas."""
    d = df.dropna(subset=[key]).drop_duplicates([key, value])
    counts = d[key].value_counts()
    d = d[d[key].map(counts) == 1]
    return dict(zip(d[key], d[value]))


def read_xlsx_columns(path: Path, sheet: str, columns: list[str]) -> pd.DataFrame:
    """Lê só algumas colunas de uma planilha grande (bem mais rápido que read_excel)."""
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb[sheet]
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    idx = [header.index(c) for c in columns]
    data = [[r[i] for i in idx] for r in rows]
    wb.close()
    return pd.DataFrame(data, columns=columns)


# --------------------------------------------------------------------------- base
def load_base() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """base: uma linha por id x legislatura; aliases: todas as variantes de nome/partido
    que a API da Câmara registrou para cada id x legislatura (usadas só para casar nomes)."""
    parls = pd.read_csv(DATA / "df_parlamentares_por_legislatura.csv").rename(
        columns={"id": "id_camara", "idLegislatura": "legislatura"}
    )
    aliases = parls[["id_camara", "legislatura", "nome", "siglaPartido", "siglaUf"]].copy()
    aliases["nome_norm"] = aliases["nome"].map(norm)
    base = (
        aliases.sort_values(["id_camara", "legislatura", "nome"], na_position="last")
        .drop_duplicates(["id_camara", "legislatura"])
        .drop(columns="nome_norm")
        .reset_index(drop=True)
    )

    det = pd.read_csv(DATA / "df_parl_detalhes.csv")[["id", "nomeCivil", "ultimoStatus.nome"]]
    det = det.rename(columns={"id": "id_camara"})
    det["civil_norm"] = det["nomeCivil"].map(norm)
    return base, aliases, det


def match_names(
    df: pd.DataFrame,
    name_col: str,
    indice: str,
    candidates: list[dict],
    depara: dict,
) -> pd.Series:
    """Tenta cada dicionário nome_norm -> id_camara em ordem; depois o de-para manual."""
    key = df[name_col].map(norm)
    ids = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
    for cand in candidates:
        ids = ids.fillna(key.map(cand).astype("Int64"))
    manual = key.map(lambda k: depara.get((indice, k))).astype("Int64")
    return ids.fillna(manual)


# --------------------------------------------------------------------------- índices
def load_ipp() -> pd.DataFrame:
    ipp = pd.read_excel(SCORES / "IPP" / "df_ipp_v4.xlsx")
    return ipp.rename(columns={"id": "id_camara", "idLegislatura": "legislatura"})[
        ["id_camara", "legislatura", "ipp", "ipp_final_v4"]
    ]


def load_ird() -> pd.DataFrame:
    ird = pd.read_csv(SCORES / "IRD" / "df_equality_index.csv").rename(
        columns={"parl_id": "id_camara", "weights": "ird_peso"}
    )
    agg = pd.read_csv(SCORES / "IRD" / "df_ird_agg_legislatura.csv").rename(
        columns={"adjustment_index": "ird_ajuste_legislatura"}
    )[["legislatura", "ird_ajuste_legislatura"]]
    return ird.merge(agg, on="legislatura", how="left")


def load_co2(aliases: pd.DataFrame, depara: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    co2 = pd.read_excel(SCORES / "CO2" / "CO2-INDEX.XLSX")
    leg = aliases[aliases.legislatura == LEG_CO2]
    by_name_uf = {
        (n, uf): i
        for (n, uf), i in unique_map(
            leg.assign(k=list(zip(leg.nome_norm, leg.siglaUf))), "k", "id_camara"
        ).items()
    }
    co2["id_camara"] = pd.Series(
        [by_name_uf.get((norm(n), uf)) for n, uf in zip(co2.nome_padronizado, co2.siglaUf)],
        index=co2.index,
        dtype="Int64",
    )
    co2["id_camara"] = co2["id_camara"].fillna(
        match_names(co2, "nome_padronizado", "CO2", [unique_map(leg, "nome_norm", "id_camara")], depara)
    )
    co2["legislatura"] = LEG_CO2
    unmatched = co2[co2.id_camara.isna()][["nome_padronizado", "siglaPartido", "siglaUf"]]
    ok = co2.dropna(subset=["id_camara"]).rename(
        columns={
            "INDICE_VOTOS": "co2_indice_votos",
            "INDICE_PROJETOS": "co2_indice_projetos",
            "INDICE_DISCURSOS": "co2_indice_discursos",
            "CO2_INDEX": "co2_index",
        }
    )[["id_camara", "legislatura", "co2_indice_votos", "co2_indice_projetos", "co2_indice_discursos", "co2_index"]]
    return ok, unmatched


def load_ipipe(aliases: pd.DataFrame, det: pd.DataFrame, depara: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ip = pd.read_csv(SCORES / "IPIPE" / "IPIPE_56A_PARLAMENTARES.csv")
    leg_ids = set(aliases.loc[aliases.legislatura == LEG_IPIPE, "id_camara"])
    det_leg = det[det.id_camara.isin(leg_ids)]
    civil = unique_map(det_leg, "civil_norm", "id_camara")
    civil_any = unique_map(det, "civil_norm", "id_camara")
    ip["id_camara"] = match_names(ip, "nome", "IPIPE", [civil, civil_any], depara)
    ip["legislatura"] = LEG_IPIPE
    unmatched = ip[ip.id_camara.isna()][["id_parlamentar", "nome", "uf", "partido"]]
    ok = ip.dropna(subset=["id_camara"]).rename(
        columns={
            "id_parlamentar": "ipipe_id_parlamentar",
            "IPIPE": "ipipe",
            "score_exposicao_ipipe": "ipipe_score_exposicao",
            "n_processos_classificados": "ipipe_n_processos",
            "status_pipeline": "ipipe_status",
        }
    )[["id_camara", "legislatura", "ipipe_id_parlamentar", "ipipe", "ipipe_score_exposicao", "ipipe_n_processos", "ipipe_status"]]
    return ok, unmatched


def load_ifs(aliases: pd.DataFrame, det: pd.DataFrame, depara: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    posts = read_xlsx_columns(
        SCORES / "IFS" / "df_predictions.xlsx",
        "Sheet1",
        ["screen_name", "user_id", "prediction", "fake_probability"],
    )
    posts["prediction"] = pd.to_numeric(posts.prediction)
    posts["fake_probability"] = pd.to_numeric(posts.fake_probability)
    posts["score"] = posts.prediction * posts.fake_probability
    prof = (
        posts.groupby(["screen_name", "user_id"])
        .agg(ifs_n_posts=("score", "size"), ifs_n_fake=("prediction", "sum"), soma_score=("score", "sum"))
        .reset_index()
    )
    # IFS conforme a documentação: 1 - (posts classificados como fake / total de posts)
    prof["ifs"] = 1 - prof.ifs_n_fake / prof.ifs_n_posts
    # variante usada na planilha df_predictions.xlsx: 1 - soma(f*p) / total de posts
    prof["ifs_soma_score"] = 1 - prof.soma_score / prof.ifs_n_posts

    perfis = pd.read_csv(SCORES / "IFS" / "perfis_2023.csv")
    perfis["civil_norm"] = perfis.nome_registro.map(norm)
    prof = prof.merge(
        perfis[["Profile-ID", "nome_registro", "cargo", "civil_norm"]].drop_duplicates("Profile-ID"),
        left_on="user_id",
        right_on="Profile-ID",
        how="left",
    )

    leg = aliases[aliases.legislatura == LEG_IFS]
    det_leg = det[det.id_camara.isin(set(leg.id_camara))]
    civil = unique_map(det_leg, "civil_norm", "id_camara")
    prof["id_camara"] = prof.civil_norm.map(civil).astype("Int64")
    prof["id_camara"] = prof["id_camara"].fillna(
        match_names(prof, "screen_name", "IFS", [unique_map(leg, "nome_norm", "id_camara")], depara)
    )
    prof["legislatura"] = LEG_IFS
    unmatched = prof[prof.id_camara.isna()][["screen_name", "user_id", "nome_registro", "cargo", "ifs_n_posts"]]
    ok = prof.dropna(subset=["id_camara"])
    # um mesmo deputado pode ter mais de um perfil: agrega os posts
    ok = (
        ok.groupby(["id_camara", "legislatura"])
        .agg(ifs_n_posts=("ifs_n_posts", "sum"), ifs_n_fake=("ifs_n_fake", "sum"), soma_score=("soma_score", "sum"))
        .reset_index()
    )
    ok["ifs"] = 1 - ok.ifs_n_fake / ok.ifs_n_posts
    ok["ifs_soma_score"] = 1 - ok.soma_score / ok.ifs_n_posts
    return ok[["id_camara", "legislatura", "ifs", "ifs_soma_score", "ifs_n_posts", "ifs_n_fake"]], unmatched


# --------------------------------------------------------------------------- main
def main() -> None:
    depara = load_depara()
    base, aliases, det = load_base()

    ipp = load_ipp()
    ird = load_ird()
    co2, co2_nc = load_co2(aliases, depara)
    ipipe, ipipe_nc = load_ipipe(aliases, det, depara)
    ifs, ifs_nc = load_ifs(aliases, det, depara)

    final = base
    for name, df in [("ipp", ipp), ("ird", ird), ("co2", co2), ("ipipe", ipipe), ("ifs", ifs)]:
        assert not df.duplicated(["id_camara", "legislatura"]).any(), f"{name}: chave duplicada"
        df = df.copy()
        df["id_camara"] = df["id_camara"].astype("int64")
        df["legislatura"] = df["legislatura"].astype("int64")
        orphans = df.merge(final[["id_camara", "legislatura"]], how="left", indicator=True)
        n_orphan = (orphans["_merge"] == "left_only").sum()
        if n_orphan:
            print(f"  aviso: {name}: {n_orphan} linhas com id/legislatura fora da base (descartadas)")
        final = final.merge(df, on=["id_camara", "legislatura"], how="left")

    final = final.sort_values(["legislatura", "nome"]).reset_index(drop=True)
    final.to_csv(OUT_DIR / "scores_consolidados.csv", index=False)

    nc = pd.concat(
        [
            co2_nc.assign(indice="CO2", nome_origem=co2_nc.nome_padronizado),
            ipipe_nc.assign(indice="IPIPE", nome_origem=ipipe_nc.nome),
            ifs_nc.assign(indice="IFS", nome_origem=ifs_nc.screen_name),
        ],
        ignore_index=True,
    )
    cols = ["indice", "nome_origem"] + [c for c in nc.columns if c not in ("indice", "nome_origem")]
    nc[cols].to_csv(OUT_DIR / "nao_casados.csv", index=False)

    print(f"base: {len(final)} parlamentar-legislaturas")
    for name, col, total in [
        ("IPP", "ipp_final_v4", len(ipp)),
        ("IRD", "ird_peso", len(ird)),
        ("CO2", "co2_index", len(co2) + len(co2_nc)),
        ("IPIPE", "ipipe_id_parlamentar", len(ipipe) + len(ipipe_nc)),
        ("IFS", "ifs", len(ifs) + len(ifs_nc)),
    ]:
        print(f"  {name:6s} {final[col].notna().sum():5d} linhas com valor  (origem: {total})")
    print(f"não casados: {len(nc)}  -> {OUT_DIR / 'nao_casados.csv'}")
    print(nc.groupby("indice").size().to_dict())


if __name__ == "__main__":
    main()
