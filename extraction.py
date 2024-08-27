from src.extractor import Extractor
import pandas as pd

LEGISLATURAS = range(48,58)

def main():
    extractor = Extractor()
    # legislaturas
    legisaturas = extractor.get_legislaturas()
    df_legislaturas = pd.DataFrame(legisaturas)
    df_legislaturas.to_csv('./data/df_legislaturas.csv', index=False)

    # parlamentares
    parlamentares = extractor.get_parlamentares_por_legislatura()
    df_parlamentares = pd.DataFrame(parlamentares)
    df_parlamentares.to_csv('./data/df_parlamentares_por_legislatura.csv', index=False)

    # paralamentares detalhes
    parl_detalhes = extractor.get_parlamentare_detalhes()
    df_parl_detalhes = pd.DataFrame(parl_detalhes)
    df_parl_detalhes.to_csv('./data/df_parl_detalhes.csv')

    # lideres
    legis_lideres = extractor.get_legislatura_lideres()
    df_legis_lideres = pd.DataFrame(legis_lideres)
    df_legis_lideres.to_csv('./data/df_legis_lideres.csv', index=False)

    # mesas
    legis_mesa = extractor.get_legislatura_mesa()
    df_legis_mesa = pd.DataFrame(legis_mesa)
    df_legis_mesa.to_csv('./data/df_legis_mesa.csv', index=False)

    # mandatos externos
    cargos_externos = extractor.get_parlamentares_mandatos_externos()
    df_cargos_externos = pd.DataFrame(cargos_externos)
    df_cargos_externos.to_csv('./data/df_cargos_externos.csv', index=False)

    # afastamentos
    historico = extractor.get_parlamentares_historico()
    df_historico = pd.DataFrame(historico)
    df_historico.to_csv('./data/df_historico.csv', index=False)

    # pareceres de relatoria
    pareceres = extractor.get_paralamentares_pareceres()
    df_pareceres = pd.DataFrame(pareceres)
    df_pareceres.to_csv('./data/df_pareceres.csv', index=False)

    # detalhes dos pareceres
    parecere_detalhes = extractor.get_pareceres_detalhes()
    df_pareceres_detalhes = pd.DataFrame(parecere_detalhes)
    df_pareceres_detalhes.to_csv('./data/df_pareceres_detalhes.csv', index=False)

    # membros orgaos
    orgaos_membros = extractor.get_orgaos_membros()
    df_orgaos_membros = pd.DataFrame(orgaos_membros)
    df_orgaos_membros.to_csv('./data/df_orgaos_membros.csv', index=False)


if __name__ == '__main__':
    main()