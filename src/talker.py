from tqdm import tqdm
from utils import api, endpoints


def get_legislatura_lideres(id_):
    url = api.construct_url(api.HOST, endpoints.LEGISLATURA_LIDERES.format(id=id_))
    resp = api.request(url)
    return [flatten_json(p) for p in resp]


def get_legislatura_mesa(id_):
    url = api.construct_url(api.HOST, endpoints.LEGISLATURA_MESA.format(id=id_))
    resp = api.request(url)
    return resp


def get_parlamentares_por_legislatura(id_legis):
    url = api.construct_url(
        api.HOST, endpoints.PARLAMENTARES, {"idLegislatura": id_legis}
    )
    parls = api.request(url)
    return parls


def get_parlamentare_detalhes(id_parl):
    url = api.construct_url(
        api.HOST, endpoints.PARLAMENTARES_DETALHES.format(id_parl=id_parl)
    )
    resp = api.request(url)
    return resp


def get_parlamentare_mandatos_externos(id_parl):
    url = api.construct_url(
        api.HOST, endpoints.PARLAMENTARES_MANDATOS_EXTERNOS.format(id_parl=id_parl)
    )
    resp = api.request(url)
    final = [{"id_parl": id_parl, **r} for r in resp]
    return final


def get_parlamentare_historico(id_parl):
    url = api.construct_url(
        api.HOST, endpoints.PARLAMENTARES_HISTORICO.format(id_parl=id_parl)
    )
    resp = api.request(url)
    return resp


def get_legislaturas(id_legis):
    url = api.construct_url(api.HOST, endpoints.LEGISLATURA.format(id_legis=id_legis))
    resp = api.request(url)
    return resp


def get_paralamentares_pareceres(id_parl, **kwargs):
    url = api.construct_url(
        api.HOST,
        endpoints.PROPOSICOES,
        {"idDeputadoAutor": id_parl, "siglaTipo": "PRL", **kwargs},
    )
    resp = api.request(url)
    return resp


def get_proposicoes_detalhes(id_prop, **kwargs):
    url = api.construct_url(
        api.HOST, endpoints.PROPOSICAO_DETALHES.format(id_prop=id_prop), kwargs
    )
    resp = api.request(url)
    return resp

def get_orgaos():
    url = api.construct_url(api.HOST, endpoints.ORGAOS, params={"dataInicio": "1988-01-01"})
    resp = api.request(url)
    return resp

def get_orgaos_membros(id_org):
    url = api.construct_url(api.HOST, endpoints.ORGAOS_MEMBROS.format(id_org=id_org), params={"dataInicio": "1988-01-01"})
    resp = api.request(url)
    return resp


def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + '.')
        elif isinstance(x, list):
            i = 0
            for a in x:
                flatten(a, name + str(i) + '.')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

