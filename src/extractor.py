from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src import talker


class Extractor:
    def __init__(
        self,
        legislatura_inicial: int = 48,
        legilatura_final: int = 57,
        parlamentares_ids: list[str] = None,
        pareceres_ids: list[str] = None,
        orgaos_ids: list[str] = None,
    ) -> None:
        self.legislaturas = list(range(legislatura_inicial, legilatura_final + 1))
        self.parlamentares_ids: list[str] = parlamentares_ids
        self.pareceres_ids: list[str] = pareceres_ids
        self.orgaos_ids: list[str] = orgaos_ids

    def _execute(
        self, executable, iterator, tqdm_desc: str = "Baixando", max_workers=10
    ):
        final = []
        pbar = tqdm(range(0, len(iterator)), desc=tqdm_desc)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(executable, x) for x in iterator]
            for future in as_completed(futures):
                final += future.result()
                pbar.update(1)
        pbar.close()
        return final

    def get_legislaturas(self):
        return self._execute(
            talker.get_legislaturas,
            self.legislaturas,
            tqdm_desc="get_legislaturas",
        )

    def get_legislatura_lideres(self):
        return self._execute(
            talker.get_legislatura_lideres,
            self.legislaturas,
            tqdm_desc="get_legislatura_lideres",
        )

    def get_legislatura_mesa(self):
        return self._execute(
            talker.get_legislatura_mesa,
            self.legislaturas,
            tqdm_desc="get_legislatura_mesa",
        )

    def get_parlamentares_por_legislatura(self):
        resp = self._execute(
            talker.get_parlamentares_por_legislatura,
            self.legislaturas,
            tqdm_desc="get_parlamentares_por_legislatura",
        )
        self.parlamentares_ids = set([p["id"] for p in resp])
        return resp

    def get_parlamentare_detalhes(self):
        if self.parlamentares_ids is None:
            self.get_parlamentares_por_legislatura()
        return self._execute(
            talker.get_parlamentare_detalhes,
            self.parlamentares_ids,
            tqdm_desc="get_parlamentare_detalhes",
        )

    def get_parlamentares_mandatos_externos(self):
        if self.parlamentares_ids is None:
            self.get_parlamentares_por_legislatura()
        return self._execute(
            talker.get_parlamentare_mandatos_externos,
            self.parlamentares_ids,
            tqdm_desc="get_parlamentare_mandatos_externos",
        )

    def get_parlamentares_historico(self):
        if self.parlamentares_ids is None:
            self.get_parlamentares_por_legislatura()
        return self._execute(
            talker.get_parlamentare_historico,
            self.parlamentares_ids,
            tqdm_desc="get_parlamentare_historico",
        )

    def get_paralamentares_pareceres(self):
        if self.parlamentares_ids is None:
            self.get_parlamentares_por_legislatura()
        resp = self._execute(
            talker.get_paralamentares_pareceres,
            self.parlamentares_ids,
            tqdm_desc="get_paralamentares_pareceres",
        )
        self.pareceres_ids = set([p["id"] for p in resp])
        return resp

    def get_pareceres_detalhes(self):
        if self.pareceres_ids is None:
            self.get_paralamentares_pareceres()
        return self._execute(
            talker.get_proposicoes_detalhes,
            self.pareceres_ids,
            tqdm_desc="get_pareceres_detalhes",
        )

    def get_orgaos(self):
        orgaos =  talker.get_orgaos()
        self.orgaos_ids = set([o['id'] for o in orgaos])
        return orgaos

    def get_orgaos_membros(self):
        if self.orgaos_ids is None:
            self.get_orgaos()
        return self._execute(
            talker.get_orgaos_membros,
            self.orgaos_ids,
            tqdm_desc="get_orgaos_membros",
        )
