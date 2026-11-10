"""Prepara os dados do TSE usados na validação do IQ (eleições de 2022 e 2026).

Lê os arquivos públicos do TSE (baixa se não estiverem em consolidado/scores/TSE/brutos/, que
fica fora do git por ter ~1 GB) e grava, para os deputados da 57ª legislatura, identificados pelo CPF:

  consolidado/scores/TSE/eleicoes_deputados_57.csv
      id_camara, cargo/UF/partido/resultado/votos em 2026 e resultado/UF/votos para deputado
      federal em 2022, com a participação nos votos nominais do cargo no estado.

O CPF não é gravado no arquivo de saída.

Uso: python3 consolidado/preparar_tse.py
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TSE = ROOT / "consolidado" / "scores" / "TSE"
BRUTOS = TSE / "brutos"
BASE_URL = "https://cdn.tse.jus.br/estatistica/sead/odsele"
ARQUIVOS = {
    "consulta_cand_2026.zip": f"{BASE_URL}/consulta_cand/consulta_cand_2026.zip",
    "votacao_candidato_munzona_2026.zip": f"{BASE_URL}/votacao_candidato_munzona/votacao_candidato_munzona_2026.zip",
    "votacao_candidato_munzona_2022.zip": f"{BASE_URL}/votacao_candidato_munzona/votacao_candidato_munzona_2022.zip",
}
LEG = 57


def baixar() -> None:
    BRUTOS.mkdir(parents=True, exist_ok=True)
    for nome, url in ARQUIVOS.items():
        destino = BRUTOS / nome
        if not destino.exists():
            print(f"baixando {nome} ...")
            urllib.request.urlretrieve(url, destino)


def csvs_por_uf(z: zipfile.ZipFile) -> list[str]:
    return [n for n in z.namelist() if n.endswith(".csv") and "BRASIL" not in n and not n.endswith("_BR.csv")]


def votos_por_candidato(zip_path: Path, cargos: list[str]) -> pd.DataFrame:
    """Soma os votos nominais do 1º turno por candidato (os arquivos vêm por município e zona)."""
    cols = ["SG_UF", "DS_CARGO", "SQ_CANDIDATO", "NR_TURNO", "QT_VOTOS_NOMINAIS", "SG_PARTIDO", "DS_SIT_TOT_TURNO"]
    partes = []
    with zipfile.ZipFile(zip_path) as z:
        for nome in csvs_por_uf(z):
            with z.open(nome) as f:
                d = pd.read_csv(f, sep=";", encoding="latin1", dtype=str, usecols=cols)
            d["DS_CARGO"] = d.DS_CARGO.str.upper()
            d = d[d.DS_CARGO.isin(cargos) & (d.NR_TURNO == "1")]
            d["votos"] = d.QT_VOTOS_NOMINAIS.astype(int)
            partes.append(d.groupby(["SG_UF", "DS_CARGO", "SQ_CANDIDATO", "SG_PARTIDO", "DS_SIT_TOT_TURNO"], dropna=False).votos.sum().reset_index())
    v = pd.concat(partes, ignore_index=True)
    v["participacao"] = v.votos / v.groupby(["SG_UF", "DS_CARGO"]).votos.transform("sum")
    return v


def cpf11(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\D", "", regex=True).str.zfill(11)


def main() -> None:
    baixar()
    det = pd.read_csv(ROOT / "data" / "df_parl_detalhes.csv", dtype={"cpf": str})[["id", "cpf"]].rename(columns={"id": "id_camara"})
    det["cpf"] = cpf11(det.cpf)
    base = pd.read_csv(ROOT / "consolidado" / "scores_consolidados.csv")
    dep = base[base.legislatura == LEG][["id_camara"]].merge(det, on="id_camara", how="left")

    # ---- 2026: candidaturas (todas, com CPF) + votos por candidato
    with zipfile.ZipFile(BRUTOS / "consulta_cand_2026.zip") as z:
        with z.open("consulta_cand_2026_BRASIL.csv") as f:
            cand = pd.read_csv(f, sep=";", encoding="latin1", dtype=str)
    cand = cand[cand.NR_TURNO == "1"].assign(cpf=lambda d: cpf11(d.NR_CPF_CANDIDATO), DS_CARGO=lambda d: d.DS_CARGO.str.upper())
    v26 = votos_por_candidato(BRUTOS / "votacao_candidato_munzona_2026.zip", ["DEPUTADO FEDERAL", "SENADOR"])
    c26 = cand.merge(v26[["SQ_CANDIDATO", "votos", "participacao"]], on="SQ_CANDIDATO", how="left")
    # quem tem mais de uma candidatura registrada fica com a que recebeu votos
    c26 = c26.sort_values("votos", ascending=False, na_position="last").drop_duplicates("cpf")
    c26 = c26[["cpf", "DS_CARGO", "SG_UF", "SG_PARTIDO", "DS_SIT_TOT_TURNO", "votos", "participacao"]]
    c26.columns = ["cpf", "cargo_2026", "uf_2026", "partido_2026", "resultado_2026", "votos_2026", "participacao_2026"]

    # ---- 2022: votos para deputado federal; o CPF vem de data/df_tse_candidatos.csv
    v22 = votos_por_candidato(BRUTOS / "votacao_candidato_munzona_2022.zip", ["DEPUTADO FEDERAL"])
    t = pd.read_csv(ROOT / "data" / "df_tse_candidatos.csv", low_memory=False, dtype={"cpf": str, "sequencial": str})
    t = t[(t.ano == 2022) & (t.cargo == "deputado federal")].assign(cpf=lambda d: cpf11(d.cpf)).drop_duplicates("sequencial")
    c22 = v22.merge(t[["sequencial", "cpf"]], left_on="SQ_CANDIDATO", right_on="sequencial").drop_duplicates("cpf")
    c22 = c22[["cpf", "SG_UF", "SG_PARTIDO", "DS_SIT_TOT_TURNO", "votos", "participacao"]]
    c22.columns = ["cpf", "uf_2022", "partido_2022", "resultado_2022", "votos_2022", "participacao_2022"]

    out = dep.merge(c26, on="cpf", how="left").merge(c22, on="cpf", how="left").drop(columns="cpf")
    out["cargo_2026"] = out.cargo_2026.fillna("NÃO CONCORREU")
    TSE.mkdir(parents=True, exist_ok=True)
    out.to_csv(TSE / "eleicoes_deputados_57.csv", index=False)
    print(f"{len(out)} deputados da {LEG}ª | destino em 2026:", out.cargo_2026.value_counts().to_dict())
    print("resultado em 2022:", out.resultado_2022.value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
