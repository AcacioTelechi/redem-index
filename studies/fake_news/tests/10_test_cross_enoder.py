import pandas as pd
from tqdm import tqdm
from sentence_transformers import CrossEncoder

df_res = pd.read_excel(
    r"E:/repos/pessoal/redem-index/studies/fake_news/df_res_hs.xlsx"
).sort_values("min_distance")[:2000]


cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


lies = [
    "Em nome da democracia, o Brasil acaba de ressuscitar os campos de concentração nazistas. Vídeo do ginásio da Polícia Federal comparado a campos de concentração nazistas",
    "Bolsonaro é o anticristo.",
    "Em nome da democracia o Brasil acaba de ressuscitar os campos de concentração nazista",
    "Ginásio da polícia federal é um gulag comunista",
    "Jair Bolsonaro foi condenado nos EUA e investigado pelo FBI por fraudar o cartão de vacinação contra Covid-19",
    "Boulos  está entre os dez deputados que mais aprovaram projetos na atual legislatura.",
    "Única responsabilidade do governo federal na concessão da Enel é a assinatura do contrato.",
    "Houve mais residências sem luz em São Paulo do que na Flórida após o furacão Milton.",
    " O ISS de João Pessoa seja menor que o de São Paulo.",
    "Eduardo Olivetto foi nomeado apenas na gestão de Ricardo Nunes.",
    "Guarda Civil Metropolitana deixou de realizar rondas para garantir a segurança das escolas",
    "São Paulo não está entre as cinco melhores cidades para negócios",
    "Policia Federal não apurou o destino dos presentes recebidos por Lula e Dilma",
    "Helicópteros das Forças Armadas não conseguiram alcançar a região devido ao mau tempo, enquanto aeronaves de civis como o empresário Luciano Hang estão atuando em missões de resgate e ajuda humanitária",
    "o ministro do Supremo Tribunal Federal Alexandre de Moraes interferiu no resultado das eleições presidenciais de 2022.",
    "Militares americanos no território brasileiro. Ações seriam uma forma de o governo Lula retribuir uma suposta interferência de Joe Biden que teria garantido sua vitória nas eleições de 2022",
    "Reportagem da Gazeta do Povo junto com a alegação de que a vacina da AstraZeneca fez “explodir” os casos de doenças cardíacas entre jovens.",
    "nota técnica do Ministério da Saúde ampliou o acesso ao aborto no Brasil",
    "mudança de sexo em crianças com verba da União",
    "PEC das Praias privatizara o litoral brasileiro",
    "Lei Rouanet (lei nº 8.313/1991) em 2023,  governo federal liberou  R$ 16 bilhões para artistas no fim do ano passado.",
    "empresas utilizam fetos abortados para fabricar cosméticos, especialmente após a divulgação do PL 1.904/2024, conhecido como “PL do Aborto”. Existe uma indústria mundial liderada por George Soros que depende de fetos humanos para produzir tais produtos",
    "uma idosa morreu nos ginásios da polícia federal",
]


def predict_cross_encoder(post: str, lies: list[str]) -> list[tuple[str, float]]:
    scores = cross_encoder.predict([(post, lie) for lie in lies])
    return sorted(zip(lies, scores), key=lambda x: x[1], reverse=True)


for post in tqdm(df_res["message"].tolist(), desc="Processing posts"):
    results = predict_cross_encoder(post, lies)
    df_res.loc[df_res["message"] == post, "cross_encoder_results"] = results

df_res.to_excel("10_cross_encoder_results.xlsx", index=False)
