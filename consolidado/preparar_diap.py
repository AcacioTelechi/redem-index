"""Extrai as listas "Cabeças do Congresso Nacional" do DIAP (edições de 2023, 2024 e 2025).

Lê os resumos de imprensa em PDF guardados em consolidado/scores/DIAP/ (baixa do site do DIAP
se faltarem), extrai os 100 "Cabeças" e os 50 parlamentares "em ascensão" de cada edição e
identifica os deputados pelo nome parlamentar na 57ª legislatura.

Saída: consolidado/scores/DIAP/diap_cabecas.csv
       (ano, lista, cargo, nome, partido, uf, id_camara, nome_camara; senadores ficam sem id)

Requer o utilitário `pdftotext` (poppler-utils).
Uso: python3 consolidado/preparar_diap.py
"""

from __future__ import annotations

import difflib
import re
import subprocess
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIAP = ROOT / "consolidado" / "scores" / "DIAP"
URLS = {
    2023: "https://diap.org.br/images/stories/cabecas_2023_resumo_imprensa.pdf",
    2024: "https://diap.org.br/images/stories/cabecas_2024_resumo_imprensa_diap.pdf",
    2025: "https://diap.org.br/images/stories/cabecas_2025_resumo_imprensa.pdf",
}
LEG = 57
LINHA = re.compile(r"^\s*(?:\d+\.\s*)?(DEPUTAD[OA]|SENADORA?)\s+(.+?)\s*\(([^()]+?)\s*[-–]\s*([A-Za-z]{2})\)\s*$", re.I)


def norm(s: object) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z ]", " ", s)).strip()


def texto(ano: int) -> list[str]:
    pdf = DIAP / f"cabecas_{ano}_resumo_imprensa.pdf"
    if not pdf.exists():
        DIAP.mkdir(parents=True, exist_ok=True)
        print(f"baixando {pdf.name} ...")
        urllib.request.urlretrieve(URLS[ano], pdf)
    return subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout.split("\n")


def extrair(ano: int) -> list[dict]:
    """A lista dos 100 vem primeiro, em ordem alfabética, seguida da lista dos 50 em ascensão."""
    linhas = texto(ano)
    ini = next(i for i, l in enumerate(linhas) if re.search(rf"100 .CABE.AS. DO CONGRESSO NACIONAL {ano}", l))
    asc = next(i for i, l in enumerate(linhas) if i > ini and re.search(r"^\s*50 (PARLAMENTARES|DEPUTADOS)", l))
    limite = {"cabeca": 100, "ascensao": 50}
    n = {"cabeca": 0, "ascensao": 0}
    vistos: set = set()
    out = []
    for i, l in enumerate(linhas[ini:], ini):
        lista = "cabeca" if i < asc else "ascensao"
        m = LINHA.match(l)
        if not m:
            continue
        if n[lista] >= limite[lista]:
            if lista == "ascensao":
                break
            continue
        chave = (lista, norm(m.group(2)))
        if chave in vistos:
            continue
        vistos.add(chave)
        n[lista] += 1
        out.append({"ano": ano, "lista": lista, "cargo": "deputado" if m.group(1).upper().startswith("DEP") else "senador",
                    "nome": m.group(2).strip(), "partido": m.group(3).strip(), "uf": m.group(4).upper()})
    assert n == limite, f"{ano}: extraídos {n}, esperado {limite}"
    return out


def main() -> None:
    d = pd.DataFrame([r for ano in URLS for r in extrair(ano)])
    parl = pd.read_csv(ROOT / "data" / "df_parlamentares_por_legislatura.csv")
    parl = parl[parl.idLegislatura == LEG][["id", "nome", "siglaUf"]].dropna(subset=["nome"]).drop_duplicates(["id", "nome"])
    parl["k"] = parl.nome.map(norm)
    exato = {(r.k, r.siglaUf): (r.id, r.nome) for r in parl.itertuples()}
    por_nome = {r.k: (r.id, r.nome) for r in parl.itertuples()}
    ids, nomes, aproximados = [], [], set()
    for r in d.itertuples():
        achado = (None, None)
        if r.cargo == "deputado":
            k = norm(r.nome)
            achado = exato.get((k, r.uf))
            if achado is None:  # o DIAP tem erros de grafia e de UF: procura o nome mais parecido
                perto = difflib.get_close_matches(k, list(por_nome), n=1, cutoff=0.75)
                achado = por_nome[perto[0]] if perto else (None, None)
                aproximados.add((r.nome, achado[1]))
        ids.append(achado[0])
        nomes.append(achado[1])
    d["id_camara"] = pd.array(ids, dtype="Int64")
    d["nome_camara"] = nomes
    d.to_csv(DIAP / "diap_cabecas.csv", index=False)

    print(d.groupby(["ano", "lista", "cargo"]).size().unstack().to_string())
    dep = d[d.cargo == "deputado"]
    print(f"deputados: {len(dep)} registros, {dep.id_camara.notna().sum()} identificados")
    print("identificados por aproximação:", sorted(aproximados))


if __name__ == "__main__":
    main()
