# -*- coding: utf-8 -*-
"""Teste de ponta a ponta com dados FICTÍCIOS (nomes inventados): 3ª Turma, 4ª Turma e 2ª Seção,
com metadados de assuntos CNJ simulados. NÃO são resultados da pesquisa."""
import json
import random
import shutil
from pathlib import Path

import pandas as pd

import pipeline_stj as p

random.seed(11)
T3 = ["Helena Duarte", "Rogério Albuquerque", "Marta Siqueira", "Otávio Brandão", "Lúcio Farias"]
T4 = ["Cecília Moura", "Fábio Rezende", "Irene Castelo", "Jorge Valadares", "Paulo de Andrade Neto"]
FEM = {"Helena", "Marta", "Cecília", "Irene"}
BLOCO = dict(zip(T3 + T4, [0, 0, 1, 1, 1, 0, 0, 0, 1, 1]))
COMPOSICAO = {"TERCEIRA TURMA": T3, "QUARTA TURMA": T4, "SEGUNDA SEÇÃO": T3 + T4}
TAXA_MAIORIA = {"TERCEIRA TURMA": 0.05, "QUARTA TURMA": 0.08, "SEGUNDA SEÇÃO": 0.15}
EMENTA_SS = "PLANO DE SAÚDE. NEGATIVA DE COBERTURA. ROL DA ANS. TRATAMENTO PRESCRITO."
EMENTA_FORA = "CONTRATO BANCÁRIO. JUROS REMUNERATÓRIOS. CAPITALIZAÇÃO."


def lista(n):
    return n[0] if len(n) == 1 else ", ".join(n[:-1]) + " e " + n[-1]


def trat(n):
    return "a Sra. Ministra" if n.split()[0] in FEM else "o Sr. Ministro"


def certidao(orgao, relator, votantes, vencidos):
    base = f"Vistos, relatados e discutidos estes autos, os Ministros da {orgao.title()} do STJ acordam, "
    outros = [m for m in votantes if m != relator]
    if not vencidos:
        return base + (f"por unanimidade, negar provimento ao agravo interno, nos termos do voto do(a) Sr(a). "
                       f"Ministro(a) Relator(a). Os Srs. Ministros {lista(outros)} votaram com o Sr. Ministro Relator."), relator
    maj = [m for m in votantes if m not in vencidos]
    if relator in vencidos:
        lav, demais = maj[0], [v for v in vencidos if v != relator]
        t = base + (f"por maioria, dar provimento ao recurso, nos termos do voto divergente {"da Sra. Ministra" if lav.split()[0] in FEM else "do Sr. Ministro"} {lav}, "
                    f"que lavrará o acórdão. Vencido o Sr. Ministro Relator, {relator},"
                    + (f" e {trat(demais[0])} {lista(demais)}" if demais else "") +
                    f". Votaram com {trat(lav)} {lav} os Srs. Ministros {lista(maj[1:])}.")
        return t, lav
    art = "Vencidos os Srs. Ministros " if len(vencidos) > 1 else ("Vencida a Sra. Ministra " if vencidos[0].split()[0] in FEM else "Vencido o Sr. Ministro ")
    return base + (f"por maioria, negar provimento ao recurso, nos termos do voto do Sr. Ministro Relator. "
                   f"{art}{lista(vencidos)}, que davam provimento. Os Srs. Ministros "
                   f"{lista([m for m in maj if m != relator])} votaram com o Sr. Ministro Relator."), relator


dados = Path("dados_sinteticos")
shutil.rmtree(dados, ignore_errors=True)
assuntos, verdade = [], []
for orgao, slug in p.DATASETS_ESPELHOS.items():
    regs = []
    for i in range(700):
        uid = f"{slug}-{i}"
        registro = f"20{random.randint(18, 25)}{random.randint(10**7, 10**8 - 1)}"
        ss = random.random() < 0.7
        mins = COMPOSICAO[orgao]
        votantes = [m for m in mins if random.random() > 0.05] if orgao == "SEGUNDA SEÇÃO" else mins
        votantes = votantes[:-1] if orgao == "SEGUNDA SEÇÃO" else votantes  # presidente da Seção não vota
        relator = random.choice(votantes)
        venc = []
        if random.random() < TAXA_MAIORIA[orgao]:
            lado = random.choice([0, 1])
            venc = [m for m in votantes if BLOCO[m] != lado]
            if len(venc) >= len(votantes) / 2 or random.random() < 0.35:
                venc = random.sample(votantes, 1)
        txt, rel_campo = certidao(orgao, relator, votantes, venc)
        regs.append({"id": uid, "numeroRegistro": registro, "numeroProcesso": str(i), "siglaClasse": "AgInt no AREsp",
                     "nomeOrgaoJulgador": orgao, "ministroRelator": rel_campo.upper(), "tipoDeDecisao": "ACÓRDÃO",
                     "dataDecisao": (pd.Timestamp("2022-06-01") + pd.Timedelta(days=random.randint(0, 1500))).strftime("%Y%m%d"),
                     "ementa": EMENTA_SS if ss else EMENTA_FORA, "decisao": txt,
                     "referenciasLegislativas": ["LEG:FED LEI:009656 ANO:1998"] if ss else []})
        # 90% dos registros aparecem nos metadados das Íntegras; às vezes o assunto CNJ é de consumidor
        if random.random() < 0.9:
            cod = random.choice(["12486, 12489", "12482", "12486, 12488", "12486, 14760", "12487, 12490"]) if ss else "10433, 7768"
            assuntos.append({"numeroRegistro": registro, "assuntos": cod, "tipoDocumento": "ACÓRDÃO", "dataPublicacao": ""})
        verdade.append({"id": uid, "ss_true": ss, "venc_true": sorted(p._norm(v) for v in venc)})
    d = dados / "espelhos" / slug
    d.mkdir(parents=True)
    json.dump(regs, open(d / "20260831.json", "w", encoding="utf-8"), ensure_ascii=False)
(dados / "assuntos").mkdir()
pd.DataFrame(assuntos).to_csv(dados / "assuntos" / "assuntos_por_registro.csv", index=False)

votos = p.executar(str(dados), "resultados_sinteticos", min_juntos=5)
v = pd.DataFrame(verdade)
fil = v.merge(votos[["id", "criterio"]], on="id", how="left")
print(f"\nFiltro — precisão: {fil.loc[fil.criterio.notna(), 'ss_true'].mean():.1%} | "
      f"recall: {fil.loc[fil.ss_true, 'criterio'].notna().mean():.1%} | "
      f"via CNJ: {(fil.criterio == 'assunto_cnj').sum()} | via texto: {(fil.criterio == 'texto_sem_assunto').sum()}")
ok = votos.merge(v, on="id")
ok = ok[ok["alerta"] == ""]
nu = ok[ok["venc_true"].str.len() > 0]
print(f"Parser — vencidos corretos nos não unânimes (sem alerta): "
      f"{(nu['vencidos'].apply(sorted) == nu['venc_true']).mean():.1%} (n={len(nu)})")
