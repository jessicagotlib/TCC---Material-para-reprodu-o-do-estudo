# -*- coding: utf-8 -*-
"""
Padrões decisórios em saúde suplementar no STJ (2ª Seção, 3ª Turma e 4ª Turma) — pipeline de jurimetria
=======================================================================================================

Fontes (Portal de Dados Abertos do STJ)
  * Espelhos de acórdãos - Terceira Turma  /  Quarta Turma  /  Segunda Seção   (conteúdo e votação)
  * Íntegras de Decisões Terminativas e Acórdãos do DJ  -> arquivos metadados*.json, que trazem o campo
    `assuntos` com os códigos da Tabela Processual Unificada (TPU/CNJ) e o `numeroRegistro` do processo.
    Os espelhos NÃO têm assunto CNJ; o cruzamento é feito pelo `numeroRegistro`.

Recorte: exclusivamente saúde suplementar, definido pelos assuntos da TPU/CNJ (SGT) listados em ASSUNTOS_CNJ.

Saída: três matrizes de discordância (3ª Turma, 4ª Turma, 2ª Seção) + gráficos no estilo FGV/Nexo.

Uso
  python pipeline_stj.py --baixar --baixar-assuntos --dados ./dados --saida ./resultados
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------------
# 0. Configuração
# ----------------------------------------------------------------------------------------------
PORTAL = "https://dadosabertos.web.stj.jus.br"
DATASETS_ESPELHOS = {
    "TERCEIRA TURMA": "espelhos-de-acordaos-terceira-turma",
    "QUARTA TURMA": "espelhos-de-acordaos-quarta-turma",
    "SEGUNDA SEÇÃO": "espelhos-de-acordaos-segunda-secao",
}
DATASET_INTEGRAS = "integras-de-decisoes-terminativas-e-acordaos-do-diario-da-justica"

# Marcadores de assunto da TPU/CNJ, conforme a consulta pública do SGT
# (https://www.cnj.jus.br/sgt/consulta_publica_assuntos.php):
#   12480 DIREITO DA SAÚDE
#     12482 Suplementar
#       12486 Planos de saúde
#         12490 Fornecimento de insumos | 12487 Fornecimento de medicamentos | 12488 Reajuste contratual
#         12489 Tratamento médico-hospitalar | 14760 Tratamento Domiciliar (Home Care)
# 12480 NÃO entra sozinho: ele também abrange a saúde pública. Entram o ramo Suplementar e todos os filhos.
ASSUNTOS_CNJ: dict[int, str] = {
    12482: "Suplementar",
    12486: "Planos de saúde",
    12490: "Fornecimento de insumos",
    12487: "Fornecimento de medicamentos",
    12488: "Reajuste contratual",
    12489: "Tratamento médico-hospitalar",
    14760: "Tratamento Domiciliar (Home Care)",
}
# Folhas da árvore: definem o subtema pelo próprio CNJ. Processos marcados só com 12482/12486 (nós
# genéricos) recebem o subtema pela heurística textual de SUBTEMAS, sinalizada em `origem_subtema`.
SUBTEMA_POR_CODIGO: dict[int, str] = {
    12490: "insumos",
    12487: "medicamentos",
    12488: "reajuste",
    12489: "tratamento_medico_hospitalar",
    14760: "home_care",
}

# Campos conforme o dicionário oficial (dicionario-espelhodoacordao.csv)
CAMPOS = {
    "id": ["id"],
    "processo": ["numeroProcesso"],
    "registro": ["numeroRegistro"],
    "classe": ["siglaClasse"],
    "orgao": ["nomeOrgaoJulgador"],
    # ATENÇÃO: segundo o dicionário, `ministroRelator` é o relator OU, se ele ficou vencido,
    # o relator para o acórdão. Isso é tratado em aplicar_votacao().
    "relator": ["ministroRelator"],
    "tipo": ["tipoDeDecisao"],
    "data": ["dataDecisao"],
    "data_pub": ["dataPublicacao"],
    "ementa": ["ementa"],
    "decisao": ["decisao"],
    "info_compl": ["informacoesComplementares"],
    "termos_aux": ["termosAuxiliares"],
    "ref_leg": ["referenciasLegislativas"],
    "tese": ["teseJuridica"],
    "tema": ["tema"],
    "notas": ["notas"],
    "jurisp_citada": ["jurisprudenciaCitada"],   # usado só na análise de precedentes (não no filtro)
}
# `jurisprudenciaCitada` e `acordaosSimilares` NÃO entram no filtro de propósito: um termo que aparece
# só em precedente citado não indica que o caso é de saúde suplementar (em Oliveira et al., 2025,
# 93 de 310 casos sorteados eram falsos positivos exatamente por isso).

MARCOS = [
    ("2022-06-08", "EREsp 1.886.929/1.889.704 (rol taxativo mitigado)"),
    ("2022-09-21", "Lei 14.454/2022"),
    ("2025-09-18", "STF, ADI 7265"),
    ("2026-03-11", "STJ, Tema 1295"),
]

# --- Filtro de saúde suplementar
REGEX_SAUDE_SUPL = re.compile(
    r"plano[s]? (?:privado[s]? )?de (?:assist[eê]ncia (?:[aà] )?)?sa[uú]de|sa[uú]de suplementar|"
    r"seguro[s]?[- ]sa[uú]de|operadora[s]? de (?:plano|sa[uú]de)|autogest[aã]o em sa[uú]de|"
    r"ag[eê]ncia nacional de sa[uú]de suplementar|\bANS\b|lei dos planos de sa[uú]de", re.I)
REGEX_LEI_9656 = re.compile(r"LEI:0*9656\b|LEI:0*14454\b|Lei (?:n[ºo.]*\s*)?9\.?656", re.I)
# Exclusões: casos em que "plano de saúde" aparece, mas o objeto não é a relação beneficiário-operadora
REGEX_EXCLUSAO = re.compile(
    r"previd[eê]ncia privada|plano de benef[ií]cios|complementa[çc][aã]o de aposentadoria|"
    r"\bSUS\b(?![^.]{0,80}ressarcimento)", re.I)

SUBTEMAS = {  # heurística; a ordem importa (o primeiro que casar vence). Validar na amostra manual.
    # grupos inspirados em Wang et al. (2023): cobertura assistencial, reajuste e manutenção do contrato
    "reajuste": r"reajuste|faixa et[aá]ria|sinistralidade",
    "manutencao_contrato": r"cancelamento|rescis[aã]o|resili|art(?:igo)?\.?\s*3[01]\b|aposentad|demitid|"
                           r"inadimpl|portabilidade|manuten[çc][aã]o d[oe] (?:plano|contrato|benefici)",
    "reembolso_rede": r"reembols|rede credenciada|fora da rede|descredenciamento",
    "processual_competencia": r"conflito de compet[eê]ncia|prescri[çc][aã]o|legitimidade",
    "cobertura_rol": r"\brol\b|cobertura|negativa|recusa|tratamento|medicamento|terapia|procedimento|"
                     r"home care|interna[çc][aã]o|cirurgia|[óo]rtese|pr[óo]tese|fertiliza|canabidiol|autis",
}


# ----------------------------------------------------------------------------------------------
# 0. Download (API CKAN do portal)
# ----------------------------------------------------------------------------------------------
def _recursos(dataset: str) -> list[dict]:
    import requests
    meta = requests.get(f"{PORTAL}/api/3/action/package_show", params={"id": dataset}, timeout=120).json()
    return meta["result"]["resources"]


def _baixar(url: str, tentativas: int = 4, timeout: int = 600) -> bytes:
    """GET com novas tentativas e espera crescente (o portal às vezes derruba conexões longas)."""
    import time
    import requests
    for t in range(tentativas):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:  # noqa: BLE001
            if t == tentativas - 1:
                raise
            print(f"   nova tentativa ({t + 1}) para {Path(url).name}: {e}")
            time.sleep(5 * (t + 1))
    return b""


def _data_do_arquivo(nome: str) -> str:
    """'20220531.json' -> '2022-05-31'; 'metadados202202.json' -> '2022-02-01'."""
    m = re.search(r"(\d{8}|\d{6})", nome)
    if not m:
        return "0000-00-00"
    d = m.group(1)
    return f"{d[:4]}-{d[4:6]}-{d[6:8] if len(d) == 8 else '01'}"


def baixar_espelhos(pasta: str | Path, paralelos: int = 4) -> None:
    """Baixa os três conjuntos de espelhos em paralelo, cada um em sua subpasta. Retomável."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    tarefas = []
    for orgao, slug in DATASETS_ESPELHOS.items():
        destino = Path(pasta) / "espelhos" / slug
        destino.mkdir(parents=True, exist_ok=True)
        for r in _recursos(slug):
            nome = Path(r["url"]).name
            if nome.lower().endswith((".json", ".zip")) and not (destino / nome).exists():
                tarefas.append((r["url"], destino / nome))
    print(f"[espelhos] {len(tarefas)} arquivos a baixar")

    def job(url, caminho):
        tmp = caminho.with_suffix(caminho.suffix + ".parcial")
        tmp.write_bytes(_baixar(url))
        tmp.rename(caminho)          # só vira arquivo "pronto" quando o download termina
        return caminho.name

    with ThreadPoolExecutor(paralelos) as ex:
        futuros = [ex.submit(job, u, c) for u, c in tarefas]
        for i, f in enumerate(as_completed(futuros), 1):
            try:
                print(f"[espelhos] {i}/{len(tarefas)} {f.result()}")
            except Exception as e:  # noqa: BLE001
                print(f"[espelhos] falhou: {e} (rode de novo para retomar)")


def baixar_assuntos(pasta: str | Path, inicio: str | None = None, paralelos: int = 6) -> Path:
    """Lê os metadados*.json das Íntegras em paralelo e guarda só numeroRegistro -> assuntos (TPU/CNJ).
    Não baixa os ZIPs de texto. Com `inicio`, pula arquivos publicados antes dessa data (o acórdão de um
    espelho aparece nas Íntegras na data da sua própria publicação). Retomável a qualquer momento."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    pasta = Path(pasta); pasta.mkdir(parents=True, exist_ok=True)
    saida, feitos_arq = pasta / "assuntos_por_registro.csv", pasta / "assuntos_processados.txt"
    feitos = set(feitos_arq.read_text().split()) if feitos_arq.exists() else set()
    corte = (pd.Timestamp(inicio) - pd.DateOffset(months=1)).strftime("%Y-%m-%d") if inicio else "0000-00-00"
    tarefas = [(r["url"], Path(r["url"]).name) for r in _recursos(DATASET_INTEGRAS)
               if Path(r["url"]).name.startswith("metadados") and Path(r["url"]).name.endswith(".json")
               and Path(r["url"]).name not in feitos and _data_do_arquivo(Path(r["url"]).name) >= corte]
    print(f"[assuntos] {len(tarefas)} arquivos de metadados a processar ({len(feitos)} já feitos)")

    def job(url, nome):
        dados = json.loads(_baixar(url).decode("utf-8-sig"))
        return nome, [{"numeroRegistro": d.get("numeroRegistro"), "assuntos": d.get("assuntos"),
                       "tipoDocumento": d.get("tipoDocumento"), "dataPublicacao": d.get("dataPublicacao")}
                      for d in dados if d.get("assuntos")]

    with ThreadPoolExecutor(paralelos) as ex:
        futuros = [ex.submit(job, u, n) for u, n in tarefas]
        for i, f in enumerate(as_completed(futuros), 1):
            try:
                nome, linhas = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"[assuntos] falhou: {e} (rode de novo para retomar)")
                continue
            # a gravação acontece só nesta thread principal: sem risco de corromper o CSV
            pd.DataFrame(linhas).to_csv(saida, mode="a", header=not saida.exists(), index=False)
            with open(feitos_arq, "a") as fh:
                fh.write(nome + "\n")
            if i % 25 == 0 or i == len(tarefas):
                print(f"[assuntos] {i}/{len(tarefas)} arquivos processados")
    return saida


def carregar_mapa_assuntos(arquivo: str | Path) -> dict[str, set[int]]:
    m: dict[str, set[int]] = {}
    for reg, ass in pd.read_csv(arquivo, dtype=str)[["numeroRegistro", "assuntos"]].dropna().itertuples(index=False):
        m.setdefault(re.sub(r"\D", "", reg), set()).update(int(x) for x in re.findall(r"\d+", ass))
    print(f"[assuntos] {len(m)} processos com assuntos TPU/CNJ.")
    return m


# ----------------------------------------------------------------------------------------------
# 1. Carga
# ----------------------------------------------------------------------------------------------
def _primeiro_campo(df: pd.DataFrame, opcoes: list[str]) -> pd.Series:
    for c in opcoes:
        if c in df.columns:
            return df[c]
    return pd.Series([None] * len(df), index=df.index)


ARQUIVOS_COM_ERRO: list[str] = []


def _reparar_json(txt: str) -> str:
    """Correções conservadoras para defeitos comuns em JSON exportado:
    vírgula ausente entre campos/objetos em linhas consecutivas. Strings JSON não podem conter quebra de
    linha literal, então inserir a vírgula no fim de linha nunca altera o conteúdo de um texto."""
    txt = re.sub(r'(["\d\]}]|true|false|null)[ \t]*\r?\n(\s*")', r"\1,\n\2", txt)
    txt = re.sub(r"}[ \t]*\r?\n(\s*){", r"},\n\1{", txt)
    txt = re.sub(r"}\s*}\s*\]\s*$", "}]", txt)          # chave de fechamento duplicada no fim
    return txt


def _ler_json_bytes(b: bytes, origem: str) -> list[dict]:
    txt = b.decode("utf-8-sig", errors="replace")
    dados, erro = None, None
    for tentativa in ("normal", "tolerante", "reparado"):
        try:
            if tentativa == "normal":
                dados = json.loads(txt)
            elif tentativa == "tolerante":        # aceita caracteres de controle dentro de textos
                dados = json.loads(txt, strict=False)
            else:
                dados = json.loads(_reparar_json(txt), strict=False)
                print(f"[carga] {origem}: JSON com defeito de formatação, reparado automaticamente.")
            break
        except json.JSONDecodeError as e:
            erro = e
    if dados is None:
        linhas = txt.splitlines()
        ini = max(0, erro.lineno - 3)
        trecho = "\n".join(f"   {i + 1:>6}: {linhas[i][:160]}" for i in range(ini, min(len(linhas), erro.lineno + 2)))
        msg = f"{origem}: {erro.msg} (linha {erro.lineno}, coluna {erro.colno}); tamanho {len(b)} bytes\n{trecho}"
        print(f"[carga] ARQUIVO IGNORADO -> {msg}")
        ARQUIVOS_COM_ERRO.append(msg)
        return []
    if isinstance(dados, dict):
        dados = next((v for v in dados.values() if isinstance(v, list)), [dados])
    for d in dados:
        d["_arquivo"] = origem
    return dados


def carregar_espelhos(pasta: str | Path) -> pd.DataFrame:
    registros = []
    for arq in sorted(Path(pasta).rglob("*")):
        if arq.suffix.lower() == ".json":
            registros += _ler_json_bytes(arq.read_bytes(), arq.name)
        elif arq.suffix.lower() == ".zip":
            with zipfile.ZipFile(arq) as z:
                for n in z.namelist():
                    if n.lower().endswith(".json"):
                        registros += _ler_json_bytes(z.read(n), f"{arq.name}/{n}")
    bruto = pd.DataFrame(registros)
    df = pd.DataFrame({k: _primeiro_campo(bruto, v) for k, v in CAMPOS.items()})
    for c in ["ementa", "decisao", "info_compl", "termos_aux", "ref_leg", "tese", "notas", "jurisp_citada"]:
        df[c] = df[c].apply(lambda x: " ".join(map(str, x)) if isinstance(x, list) else ("" if x is None else str(x)))
    df["arquivo"] = bruto["_arquivo"]
    df["data"] = _parse_data(df["data"])
    df["orgao"] = df["orgao"].astype(str).str.upper().str.strip()
    antes = len(df)
    # o arquivo mais recente prevalece em caso de duplicidade (pode trazer correções)
    df = df.sort_values("arquivo").drop_duplicates(subset=["id"], keep="last")
    if df["tipo"].notna().any():
        df = df[df["tipo"].astype(str).str.upper().str.contains("AC[OÓ]RD", regex=True)]
    print(f"[carga] {antes} registros lidos; {len(df)} acórdãos únicos.")
    return df.reset_index(drop=True)


def _parse_data(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d{8}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})")[0]
    out = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    out = out.fillna(pd.to_datetime(s, format="%d/%m/%Y", errors="coerce"))
    return out.fillna(pd.to_datetime(s, format="%Y-%m-%d", errors="coerce"))


# ----------------------------------------------------------------------------------------------
# 2. Filtro: exclusivamente saúde suplementar
# ----------------------------------------------------------------------------------------------
def filtrar_saude_suplementar(df: pd.DataFrame, mapa_assuntos: dict[str, set[int]] | None,
                              fallback_texto: bool = True) -> pd.DataFrame:
    """Critério principal: o processo tem ao menos um assunto TPU/CNJ de ASSUNTOS_CNJ.
    Critério de contingência (só quando o registro não aparece nos metadados das Íntegras): texto da
    ementa/indexação/referência legislativa. Os dois ficam identificados em `criterio` para permitir
    análise de sensibilidade (matriz só com CNJ x matriz com CNJ + texto)."""
    df = df.copy()
    alvo = set(ASSUNTOS_CNJ)
    reg = df["registro"].astype(str).str.replace(r"\D", "", regex=True)
    ass = reg.map(lambda r: (mapa_assuntos or {}).get(r))
    df["assuntos_cnj"] = ass.map(lambda a: ",".join(map(str, sorted(a))) if a else "")
    df["tem_assunto"] = ass.notna()
    df["cnj_saude_supl"] = ass.map(lambda a: bool(a and a & alvo))
    df["assunto_cnj_rotulo"] = ass.map(lambda a: " | ".join(ASSUNTOS_CNJ[c] for c in sorted(a & alvo)) if a else "")

    idx = df["info_compl"] + " " + df["termos_aux"] + " " + df["tese"]
    texto_ok = ((df["ementa"].str.contains(REGEX_SAUDE_SUPL) |
                 (df["ref_leg"].str.contains(REGEX_LEI_9656) & idx.str.contains(REGEX_SAUDE_SUPL)))
                & ~df["ementa"].str.contains(REGEX_EXCLUSAO))
    df["criterio"] = np.select(
        [df["cnj_saude_supl"], ~df["tem_assunto"] & texto_ok & fallback_texto],
        ["assunto_cnj", "texto_sem_assunto"], default="fora")
    # auditoria: casos que o texto aponta como saúde suplementar mas o assunto CNJ não (erro de autuação?)
    df["divergencia_texto_cnj"] = df["tem_assunto"] & texto_ok & ~df["cnj_saude_supl"]

    # apoio à escolha dos marcadores: códigos CNJ mais frequentes nos casos que o TEXTO indica como
    # saúde suplementar. Códigos frequentes aqui e ausentes em ASSUNTOS_CNJ merecem conferência no SGT.
    cand = Counter(c for a, ok in zip(ass, texto_ok) if a and ok for c in a)
    pd.DataFrame(cand.most_common(), columns=["codigo_tpu", "n_acordaos"]).assign(
        ja_em_ASSUNTOS_CNJ=lambda d: d["codigo_tpu"].isin(alvo)).to_csv("assuntos_candidatos.csv", index=False)

    sel = df[df["criterio"] != "fora"].copy()
    # subtema 1º pelas folhas do CNJ; se o processo tiver mais de uma folha, todas ficam em `subtemas_cnj`
    folhas = sel["assuntos_cnj"].map(lambda t: [SUBTEMA_POR_CODIGO[int(c)] for c in t.split(",")
                                                if c and int(c) in SUBTEMA_POR_CODIGO])
    sel["subtemas_cnj"] = folhas.map(lambda f: "|".join(sorted(set(f))))
    texto = sel["ementa"] + " " + sel["info_compl"]
    heur = pd.Series("outros", index=sel.index)
    for nome, rx in reversed(list(SUBTEMAS.items())):
        heur[texto.str.contains(rx, case=False, regex=True)] = nome
    tem_folha = folhas.str.len() > 0
    sel["subtema"] = np.where(tem_folha, folhas.map(lambda f: f[0] if f else ""), heur)
    sel["origem_subtema"] = np.where(tem_folha, "cnj", "heuristica_texto")
    cob = df["tem_assunto"].mean() if len(df) else 0
    print(f"[filtro] cobertura de assuntos CNJ nos espelhos: {cob:.1%}")
    print(f"[filtro] selecionados por órgão e critério:\n{pd.crosstab(sel['orgao'], sel['criterio'])}")
    print(f"[filtro] divergências texto x CNJ para auditoria: {int(df['divergencia_texto_cnj'].sum())}")
    return sel.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# 3. Extração da votação a partir da certidão (campo `decisao`)
# ----------------------------------------------------------------------------------------------
_HONORIFICOS = re.compile(r"\b(?:Exm[oa]s?\.?\s*)?S(?:ras|rs|ra|r)\b\.?\s*(?:\((?:a|as|s)\))?\.?\s*", re.I)
_PARADAS = re.compile(
    r",?\s+(?:que\b|quanto\b|o qual|a qual|os quais|no\s|na\s|nos\s|nas\s|em\s(?!parte)|apenas|"
    r"por\s|para\s|com\s|votaram|votou|lavrar|nos termos|conforme|e,\s|acompanhad)", re.I)


def _limpar(texto: str) -> str:
    t = re.sub(r"\s+", " ", str(texto or ""))
    t = _HONORIFICOS.sub(" ", t)
    t = re.sub(r"Ministr[oa]\((?:a|o)\)", "Ministro", t)
    t = re.sub(r"\bMin\.\s*", "Ministro ", t)
    t = re.sub(r"\((?:Presidente|Presidenta)\)", "", t, flags=re.I)
    # "Vencido o Ministro Relator, Fulano de Tal," -> "Vencido o Ministro Fulano de Tal,"
    t = re.sub(r"Relator[a]?,\s+((?:[A-ZÁÉÍÓÚÂÊÔÃÕÇ][^\s,.;]*)(?:\s+(?:d[aeo]s?\s+)?[A-ZÁÉÍÓÚÂÊÔÃÕÇ][^\s,.;]*)*)\s*,",
               r"\1,", t)
    return re.sub(r"\s+", " ", t)


def _separar_nomes(seg: str) -> list[str]:
    seg = _PARADAS.split(seg)[0]
    partes = re.split(r",\s*|\s+e\s+", seg)
    nomes = []
    for p in partes:
        p = re.sub(r"^(?:o|a|os|as)\s+", "", p.strip(), flags=re.I)
        p = re.sub(r"^(?:Ministr[oa]s?|Desembargador[a]?(?:\s+Convocad[oa])?)\s+", "", p, flags=re.I).strip(" ,;")
        if re.fullmatch(r"relator[a]?", p, re.I):
            nomes.append("__RELATOR__")
        elif len(p) >= 4 and re.search(r"[A-ZÁÉÍÓÚÂÊÔÃÕÇ]", p):
            nomes.append(p)
    return nomes


def extrair_votacao(decisao: str) -> dict:
    t = _limpar(decisao)
    tl = t.lower()
    maioria = "por maioria" in tl
    unanime = ("por unanimidade" in tl) and not maioria

    # "Vencido(s) [em parte | quanto à tese ... | na tese] o(s) Ministro(s) X e Y" — o segmento para
    # antes de ponto, ponto e vírgula ou de uma nova oração com "Votou/Votaram" (certidões sem ponto final)
    vencidos, parcial = [], False
    for m in re.finditer(r"vencid[oa]s?(?P<parte>,?\s*(?:em\s+parte|parcialmente),?)?\s+"
                         r"(?P<seg>(?:(?!\bvot(?:ou|aram)\b)[^.;])+)", t, re.I):
        seg = m.group("seg")
        qual = re.match(r"\s*(?:quanto|na|no|em)\b[^.;]*?(?=\b(?:o|a|os|as)\s+(?:Ministr|relator))", seg, re.I)
        if qual:                                  # "quanto à tese", "na tese": divergência parcial
            parcial = True
            seg = seg[qual.end():]
        parcial |= bool(m.group("parte"))
        seg = re.sub(r"^(?:o|a|os|as)\s+", "", seg.strip(), flags=re.I)
        vencidos += _separar_nomes(seg)

    participantes = []
    for pat in [r"Ministr[oa]s?\s+(?P<l>[^.;]+?)\s+vot(?:aram|ou)\s+com",
                r"vot(?:aram|ou)\s+com\s+(?:o|a)\s+Ministr[oa]\s+[^.;]*?\s+(?:os|o|a|as)\s+Ministr[oa]s?\s+(?P<l>[^.;]+)",
                r"Participaram\s+do\s+julgamento\s+(?:os|as|o|a)?\s*Ministr[oa]s?\s+(?P<l>[^.;]+)"]:
        for m in re.finditer(pat, t, re.I):
            participantes += _separar_nomes(m.group("l"))

    # quem lavra o acórdão (lado vencedor): "Lavrará o acórdão o Ministro X" ou "voto ... do Ministro X, que lavra(rá)"
    lavrador = None
    for pat in [r"lavrar[aá]\s+o\s+ac[oó]rd[aã]o\s+(?:o|a)\s+Ministr[oa]\s+(?P<l>[^.;,]+)",
                r"voto(?:[- ]vista)?(?:\s+divergente)?\s+d[oa]\s+Ministr[oa]\s+(?P<l>[^.;,]+?),?\s+que\s+lavra"]:
        m = re.search(pat, t, re.I)
        if m:
            nomes = _separar_nomes(m.group("l"))
            if nomes:
                lavrador = nomes[0]
                participantes.append(lavrador)
                break

    ausentes = []
    for m in re.finditer(r"(?:[Aa]usente[s]?,?\s*(?:justificadamente,?\s*)?|N[ãa]o\s+participa(?:ram|ou)\s+do\s+julgamento\s+)"
                         r"(?:(?:o|a|os|as)\s+)?(?P<l>[^.;(]+)", t):
        ausentes += _separar_nomes(m.group("l"))

    return {
        "unanime": unanime,
        "maioria": maioria,
        "vencido_parcial": parcial,
        "vencidos_txt": list(dict.fromkeys(vencidos)),
        "participantes_txt": list(dict.fromkeys(participantes)),
        "ausentes_txt": list(dict.fromkeys(ausentes)),
        "lavrador_txt": lavrador,
        "tese_repetitiva": bool(re.search(r"tese|fins repetitivos", tl)) and maioria,
        "afetacao": bool(re.search(r"afetar o processo ao rito", tl)),
        "voto_vista": bool(re.search(r"voto[- ]vista", tl)),
    }


# ----------------------------------------------------------------------------------------------
# 4. Padronização de nomes
# ----------------------------------------------------------------------------------------------
def _norm(nome: str) -> str:
    n = unicodedata.normalize("NFKD", str(nome)).encode("ascii", "ignore").decode()
    n = re.sub(r"[^A-Za-z ]", " ", n).upper()
    return re.sub(r"\s+", " ", n).strip()


class Canonizador:
    """Usa os nomes do campo `ministroRelator` como lista canônica e mapeia os nomes lidos no texto."""

    STOP = {"DE", "DA", "DO", "DOS", "DAS", "E"}

    def __init__(self, relatores: pd.Series, aliases: dict[str, str] | None = None):
        self.canon = sorted({_norm(r) for r in relatores.dropna() if str(r).strip()})
        self.aliases = {_norm(k): _norm(v) for k, v in (aliases or {}).items()}
        self.nao_mapeados = Counter()

    def __call__(self, nome: str) -> str | None:
        n = _norm(nome)
        if n in self.aliases:
            return self.aliases[n]
        if n in self.canon:
            return n
        tok = set(n.split()) - self.STOP
        melhor, score = None, 0.0
        for c in self.canon:
            ct = set(c.split()) - self.STOP
            j = len(tok & ct) / max(1, len(tok | ct))
            if j > score:
                melhor, score = c, j
        if score >= 0.5:
            return melhor
        import difflib
        prox = difflib.get_close_matches(n, self.canon, n=1, cutoff=0.85)
        if prox:
            return prox[0]
        self.nao_mapeados[n] += 1
        return None


def aplicar_votacao(df: pd.DataFrame, aliases: dict | None = None) -> tuple[pd.DataFrame, Canonizador]:
    can = Canonizador(df["relator"], aliases)
    linhas = []
    for _, r in df.iterrows():
        v = extrair_votacao(r["decisao"])
        rel_campo = can(r["relator"]) if pd.notna(r["relator"]) else None
        lavrador = can(v["lavrador_txt"]) if v["lavrador_txt"] else None
        # Nos dados reais, `ministroRelator` costuma manter o relator ORIGINAL mesmo quando ele fica vencido
        # (apesar do que diz o dicionário). Só quando o campo coincide com quem lavra o acórdão é que o
        # relator original fica desconhecido.
        relator_original = None if (lavrador and rel_campo == lavrador) else rel_campo

        def mapear(lst):
            out = [(relator_original if x == "__RELATOR__" else can(x)) for x in lst]
            return {x for x in out if x}

        venc = mapear(v["vencidos_txt"])
        aus = mapear(v["ausentes_txt"])
        part = (mapear(v["participantes_txt"]) | venc | ({rel_campo} if rel_campo else set())) - aus
        sem_texto = not str(r["decisao"] or "").strip()
        alerta = ("sem_texto_decisao" if sem_texto else
                  "relator_original_sem_nome" if ("__RELATOR__" in v["vencidos_txt"] and relator_original is None) else
                  "maioria_sem_vencido" if v["maioria"] and not venc else
                  "poucos_participantes" if len(part) < 3 else
                  "sem_indicacao_votacao" if not (v["maioria"] or v["unanime"]) else "")
        linhas.append({
            "unanime": v["unanime"], "maioria": v["maioria"], "vencido_parcial": v["vencido_parcial"],
            "relator_c": rel_campo, "lavrador": lavrador,
            "relator_substituido": bool(v["maioria"] and lavrador and lavrador != rel_campo and rel_campo in venc),
            "tese_repetitiva": v["tese_repetitiva"], "afetacao": v["afetacao"], "voto_vista": v["voto_vista"],
            "vencidos": sorted(venc), "participantes": sorted(part), "alerta": alerta,
        })
    out = pd.concat([df.reset_index(drop=True).drop(columns=[c for c in linhas[0] if c in df.columns],
                                                    errors="ignore"), pd.DataFrame(linhas)], axis=1)
    print(f"[votação] unânimes: {out['unanime'].sum()} | por maioria: {out['maioria'].sum()} | "
          f"relator vencido: {out['relator_substituido'].sum()} | com voto-vista: {out['voto_vista'].sum()} | "
          f"alertas: {(out['alerta'] != '').sum()}")
    if can.nao_mapeados:
        print("[votação] nomes não mapeados (adicione em ALIASES):", can.nao_mapeados.most_common(15))
    return out, can


# ----------------------------------------------------------------------------------------------
# 5. Índice de Discordância
# ----------------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / den
    meia = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (max(0, centro - meia), min(1, centro + meia))


def matriz_discordancia(df: pd.DataFrame, incluir_parcial: bool = True,
                        incluir_unanimes: bool = False, incluir_afetacao: bool = True) -> pd.DataFrame:
    """D_ij = nº de julgamentos em que i e j ficaram em lados opostos / nº de julgamentos com ambos.
    Padrão (como FGV/Nexo): apenas julgamentos NÃO unânimes com votação identificada."""
    base = df[df["alerta"] == ""]
    if not incluir_afetacao:
        base = base[~base["afetacao"]]
    if not incluir_parcial:
        base = base[~base["vencido_parcial"]]
    if not incluir_unanimes:
        base = base[base["maioria"]]
    co, dis = Counter(), Counter()
    for _, r in base.iterrows():
        venc = set(r["vencidos"])
        for a, b in itertools.combinations(sorted(r["participantes"]), 2):
            co[(a, b)] += 1
            dis[(a, b)] += (a in venc) != (b in venc)
    linhas = []
    for (a, b), n in co.items():
        lo, hi = wilson(dis[(a, b)], n)
        linhas.append({"min_a": a, "min_b": b, "n_juntos": n, "n_divergem": dis[(a, b)],
                       "discordancia": dis[(a, b)] / n, "ic95_inf": lo, "ic95_sup": hi})
    cols = ["min_a", "min_b", "n_juntos", "n_divergem", "discordancia", "ic95_inf", "ic95_sup"]
    return pd.DataFrame(linhas, columns=cols).sort_values("discordancia", ascending=False)


def perfil_ministros(df: pd.DataFrame) -> pd.DataFrame:
    """`vezes_vencido`: vencido no resultado (julgamentos por maioria).
    `vencido_fundamentacao`: resultado unânime, mas ficou vencido quanto à fundamentação/tese."""
    part, venc, fund, rel = Counter(), Counter(), Counter(), Counter()
    base = df[df["alerta"] == ""]
    for _, r in base.iterrows():
        for m in r["participantes"]:
            part[m] += 1
        for m in r["vencidos"]:
            (venc if r["maioria"] else fund)[m] += 1
        if r["relator_c"]:
            rel[r["relator_c"]] += 1
    linhas = []
    for m in part:
        lo, hi = wilson(venc[m], part[m])
        linhas.append({"ministro": m, "participacoes": part[m], "relatorias": rel[m], "vezes_vencido": venc[m],
                       "taxa_vencido": venc[m] / part[m], "ic95_inf": lo, "ic95_sup": hi,
                       "vencido_fundamentacao": fund[m]})
    return pd.DataFrame(linhas).sort_values("participacoes", ascending=False)


# ----------------------------------------------------------------------------------------------
# 6. Gráficos no estilo FGV/Nexo
# ----------------------------------------------------------------------------------------------
def _quadrada(pares: pd.DataFrame, min_juntos: int):
    nomes = sorted(set(pares["min_a"]) | set(pares["min_b"]))
    D = pd.DataFrame(np.nan, index=nomes, columns=nomes)
    N = pd.DataFrame(0, index=nomes, columns=nomes)
    for _, r in pares.iterrows():
        N.loc[r.min_a, r.min_b] = N.loc[r.min_b, r.min_a] = r.n_juntos
        if r.n_juntos >= min_juntos:
            D.loc[r.min_a, r.min_b] = D.loc[r.min_b, r.min_a] = r.discordancia
    for n in nomes:
        D.loc[n, n] = 0.0
    return D, N


def posicoes_mds(D: pd.DataFrame, seed: int = 42):
    """Pares sem dados suficientes recebem a distância geodésica (caminho mínimo), como no Isomap;
    depois aplica-se MDS métrico. Retorna posições e o stress normalizado (qualidade do mapa)."""
    import networkx as nx
    from sklearn.manifold import MDS
    G = nx.Graph()
    G.add_nodes_from(D.index)
    for a, b in itertools.combinations(D.index, 2):
        if pd.notna(D.loc[a, b]):
            G.add_edge(a, b, weight=max(D.loc[a, b], 1e-3))
    geo = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    M = np.array([[geo.get(a, {}).get(b, np.nan) for b in D.index] for a in D.index])
    M = np.where(np.isnan(M), np.nanmax(M) * 1.5 if np.isfinite(np.nanmax(M)) else 1, M)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed, n_init=8,
              normalized_stress="auto")
    xy = mds.fit_transform(M)
    return {n: xy[i] for i, n in enumerate(D.index)}, float(mds.stress_)


# grafia acentuada dos nomes, aplicada na saída (a base do STJ traz os nomes sem acento)
ACENTOS = {
    "raul araujo": "Raul Araújo", "luis felipe salomao": "Luis Felipe Salomão",
    "ricardo villas boas cueva": "Ricardo Villas Bôas Cueva", "joao otavio de noronha": "João Otávio de Noronha",
    "marco aurelio bellizze": "Marco Aurélio Bellizze", "antonio carlos ferreira": "Antonio Carlos Ferreira",
    "paulo de tarso sanseverino": "Paulo de Tarso Sanseverino", "maria isabel gallotti": "Maria Isabel Gallotti",
    "nancy andrighi": "Nancy Andrighi", "moura ribeiro": "Moura Ribeiro", "marco buzzi": "Marco Buzzi",
    "humberto martins": "Humberto Martins", "daniela teixeira": "Daniela Teixeira",
    "luis carlos gambogi": "Luis Carlos Gambogi", "carlos cini marchionatti": "Carlos Cini Marchionatti",
}


def _rotulo(nome: str) -> str:
    base = " ".join(w.capitalize() if w not in {"DE", "DA", "DO", "DOS"} else w.lower() for w in nome.split())
    curto = base.split(" Desembargador")[0]
    return ACENTOS.get(curto.lower(), base)


def grafico_nexo(pares: pd.DataFrame, perfil: pd.DataFrame, titulo: str, arquivo: str | Path,
                 min_juntos: int = 5, vmax: float = 0.8):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    D, N = _quadrada(pares, min_juntos)
    validos = int((N.values >= min_juntos).sum() // 2)
    if len(D) < 3 or validos < 3:
        print(f"[gráfico] {titulo}: só {validos} par(es) com >= {min_juntos} julgamentos em comum; "
              f"mapa de rede não é estimável (ver fig_vencidos_*).")
        return None
    pos, stress = posicoes_mds(D)
    cmap = plt.get_cmap("RdBu_r")
    norm = colors.Normalize(0, vmax)

    fig = plt.figure(figsize=(14, 6.8))

    # --- rede
    ax = fig.add_axes([0.02, 0.05, 0.50, 0.82])
    ax.set_axis_off()
    for a, b in itertools.combinations(D.index, 2):
        if pd.notna(D.loc[a, b]) and a != b:
            (x1, y1), (x2, y2) = pos[a], pos[b]
            ax.plot([x1, x2], [y1, y2], color=cmap(norm(D.loc[a, b])),
                    lw=0.6 + 2.2 * np.log1p(N.loc[a, b]) / np.log1p(N.values.max()), alpha=0.75, zorder=1)
    part = perfil.set_index("ministro")["participacoes"].reindex(D.index).fillna(1)
    # alterna o rótulo entre abaixo e acima do nó quando dois nós ficam próximos nos dois eixos,
    # para que os nomes não se sobreponham; a caixa branca protege o texto das arestas
    xs = np.array([pos[n][0] for n in pos]); ys = np.array([pos[n][1] for n in pos])
    ex, ey = (np.ptp(xs) or 1.0), (np.ptp(ys) or 1.0)
    colocados: list[tuple[float, float, float]] = []   # (x, y, dy)
    for n, (x, y) in sorted(pos.items(), key=lambda kv: -kv[1][1]):
        ax.scatter(x, y, s=180 + 900 * part[n] / part.max(), color="#e9e9e9", edgecolor="#333", lw=1.2, zorder=2)
        dy = -17.0
        for cand in (-17.0, 16.0, -31.0, 30.0):
            perto = [c for c in colocados
                     if abs(x - c[0]) < 0.30 * ex and abs(y - c[1]) < 0.16 * ey and abs(cand - c[2]) < 12]
            if not perto:
                dy = cand
                break
        colocados.append((x, y, dy))
        ax.annotate(_rotulo(n), (x, y), xytext=(0, dy), textcoords="offset points", ha="center",
                    va="top" if dy < 0 else "bottom", fontsize=8.5, fontweight="bold", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    ax.margins(0.18)
    ax.text(0, -0.02, f"Posições por MDS (stress normalizado = {stress:.2f}); espessura = nº de julgamentos juntos",
            transform=ax.transAxes, fontsize=7.5, color="#777")

    # --- heatmap (triângulo superior)
    ordem = sorted(D.index, key=lambda n: pos[n][0])
    ah = fig.add_axes([0.58, 0.20, 0.36, 0.62])
    k = len(ordem)
    for i, a in enumerate(ordem):
        for j, b in enumerate(ordem):
            if j <= i:
                continue
            val = D.loc[a, b]
            cor = "#d9d9d9" if pd.isna(val) else cmap(norm(val))
            ah.add_patch(plt.Rectangle((j, k - 1 - i), 1, 1, color=cor, ec="white", lw=1))
            if N.loc[a, b] > 0:
                txt = f"{val*100:.0f}" if pd.notna(val) else "–"
                ah.text(j + .5, k - .5 - i, txt, ha="center", va="center", fontsize=7,
                        color="white" if pd.notna(val) and abs(val - vmax / 2) > vmax * .3 else "#222")
    ah.set_xlim(0, k); ah.set_ylim(0, k)
    ah.set_xticks(np.arange(k) + .5); ah.set_xticklabels([_rotulo(n) for n in ordem], rotation=90, fontsize=7.5)
    ah.xaxis.tick_top()
    ah.set_yticks(np.arange(k) + .5); ah.set_yticklabels([_rotulo(n) for n in ordem][::-1], fontsize=7.5)
    for s in ah.spines.values():
        s.set_visible(False)
    ah.tick_params(length=0)

    ac = fig.add_axes([0.62, 0.10, 0.28, 0.025])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ac, orientation="horizontal")
    cb.set_ticks(np.linspace(0, vmax, 5)); cb.set_ticklabels([f"{t*100:.0f}" for t in np.linspace(0, vmax, 5)])
    cb.set_label("% de discordância (votações não unânimes)  ←concordam mais | concordam menos→", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    # a fonte da figura é informada no texto do trabalho, abaixo da legenda
    pass
    fig.savefig(arquivo, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[gráfico] salvo {arquivo}")
    return stress


def grafico_vencidos(perfil: pd.DataFrame, titulo: str, arquivo: str | Path):
    """Quantas vezes cada ministro ficou vencido, com o nº de participações — útil quando a divergência é
    rara demais para uma matriz par a par."""
    import matplotlib.pyplot as plt
    p = perfil[perfil["participacoes"] > 0].sort_values(["vezes_vencido", "participacoes"])
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(p) + 1.4))
    ax.barh([_rotulo(m) for m in p["ministro"]], p["vezes_vencido"], color="#c0504d")
    for i, (_, r) in enumerate(p.iterrows()):
        ax.text(r["vezes_vencido"] + 0.05, i, f"{int(r['vezes_vencido'])} de {int(r['participacoes'])} julgamentos",
                va="center", fontsize=8, color="#444")
    ax.set_xlabel("Vezes em que o ministro ficou vencido")
    fig.suptitle(titulo, x=0.02, ha="left", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(1, p["vezes_vencido"].max()) * 1.6)
    fig.tight_layout(); fig.savefig(arquivo, dpi=200); plt.close(fig)


def grafico_unanimidade(df: pd.DataFrame, arquivo: str | Path):
    import matplotlib.pyplot as plt
    d = df[df["alerta"] != "sem_indicacao_votacao"].copy()
    d["semestre"] = d["data"].dt.year.astype(str) + "-S" + ((d["data"].dt.month > 6) + 1).astype(str)
    t = d.groupby(["semestre", "orgao"])["maioria"].agg(["mean", "size"]).reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    for org, g in t.groupby("orgao"):
        ax.plot(g["semestre"], 100 * g["mean"], marker="o", label=org.title())
    for data, nome in MARCOS:
        s = f"{data[:4]}-S{1 if int(data[5:7]) <= 6 else 2}"
        if s in set(t["semestre"]):
            ax.axvline(s, color="#999", ls=":", lw=1)
            ax.text(s, ax.get_ylim()[1], nome.split("(")[0], rotation=90, fontsize=7, va="top", color="#666")
    ax.set_ylabel("% de julgamentos por maioria")
    ax.set_title("Julgamentos não unânimes por semestre — saúde suplementar", loc="left")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=45); fig.tight_layout(); fig.savefig(arquivo, dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------------------------
# 7. Amostra para validação manual (dupla codificação)
# ----------------------------------------------------------------------------------------------
def amostra_validacao(df: pd.DataFrame, n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Estratifica por órgão e força a inclusão de TODOS os julgamentos por maioria (são raros e são o
    insumo central). Colunas vazias são preenchidas por dois codificadores independentes."""
    maioria = df[df["maioria"]]
    resto = df[~df["maioria"]]
    k = max(0, n - len(maioria))
    amostra_resto = resto.groupby("orgao", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(1, round(k * len(g) / len(resto)))), random_state=seed))
    a = pd.concat([maioria, amostra_resto])
    cols = ["id", "processo", "classe", "orgao", "relator", "data", "criterio", "assuntos_cnj", "subtema", "origem_subtema", "ementa",
            "decisao", "maioria", "vencidos", "participantes", "alerta"]
    a = a[cols].copy()
    for cod in ("c1", "c2"):
        a[f"{cod}_pertinente(S/N)"] = ""
        a[f"{cod}_recorrente(beneficiario/operadora/ambos)"] = ""
        a[f"{cod}_resultado(pro_beneficiario/pro_operadora/misto/sem_merito)"] = ""
        a[f"{cod}_vencidos_corretos(S/N)"] = ""
    return a


# ----------------------------------------------------------------------------------------------
# Execução
# ----------------------------------------------------------------------------------------------
ALIASES: dict[str, str] = {
    # "NOME COMO APARECE NO TEXTO": "NOME COMO APARECE EM ministroRelator",
}


def executar(pasta_dados: str, pasta_saida: str, min_juntos: int = 5, inicio: str | None = None,
             so_cnj: bool = False, df_pronto: pd.DataFrame | None = None):
    out = Path(pasta_saida); out.mkdir(parents=True, exist_ok=True)
    df = df_pronto if df_pronto is not None else carregar_espelhos(Path(pasta_dados) / "espelhos")
    if ARQUIVOS_COM_ERRO:
        (out / "arquivos_com_erro.txt").write_text("\n\n".join(ARQUIVOS_COM_ERRO), encoding="utf-8")
        print(f"[aviso] {len(ARQUIVOS_COM_ERRO)} arquivo(s) ignorado(s); detalhes em arquivos_com_erro.txt")
    df = df[df["orgao"].isin(DATASETS_ESPELHOS)]
    if inicio:
        df = df[df["data"] >= inicio]
    arq_ass = Path(pasta_dados) / "assuntos" / "assuntos_por_registro.csv"
    mapa = carregar_mapa_assuntos(arq_ass) if arq_ass.exists() else None
    if mapa is None:
        print("[aviso] sem arquivo de assuntos CNJ: rode com --baixar-assuntos. Usando só o critério textual.")
    tema = filtrar_saude_suplementar(df, mapa, fallback_texto=not so_cnj)
    votos, can = aplicar_votacao(tema, ALIASES)
    votos.to_pickle(out / "base_votacao.pkl")
    votos.to_csv(out / "base_votacao.csv", index=False)

    desc = votos.groupby("orgao").agg(acordaos=("id", "size"), por_maioria=("maioria", "sum"),
                                      relator_vencido=("relator_substituido", "sum"),
                                      via_assunto_cnj=("criterio", lambda s: (s == "assunto_cnj").sum()),
                                      com_voto_vista=("voto_vista", "sum"),
                                      alertas=("alerta", lambda s: (s != "").sum()))
    desc["%_maioria"] = (100 * desc["por_maioria"] / desc["acordaos"]).round(2)
    desc.to_csv(out / "tabela_descritiva.csv")
    print(desc)
    pd.crosstab([votos["orgao"], votos["subtema"]], votos["maioria"].map({True: "maioria", False: "unanime"}),
                margins=True).to_csv(out / "tabela_subtemas.csv")

    grafico_unanimidade(votos, out / "fig_unanimidade_semestre.png")
    for org in DATASETS_ESPELHOS:           # as TRÊS matrizes
        g = votos[votos["orgao"] == org]
        slug = _norm(org).lower().replace(" ", "_")
        pares, perfil = matriz_discordancia(g), perfil_ministros(g)
        pares.to_csv(out / f"discordancia_{slug}.csv", index=False)
        perfil.to_csv(out / f"perfil_{slug}.csv", index=False)
        if len(pares):
            grafico_nexo(pares, perfil, f"{org.title()} — saúde suplementar", out / f"fig_rede_{slug}.png",
                         min_juntos=min_juntos)
        else:
            print(f"[{org}] nenhum julgamento não unânime válido: matriz vazia.")
        if len(perfil):
            grafico_vencidos(perfil, f"{org.title()} — votos vencidos em saúde suplementar",
                             out / f"fig_vencidos_{slug}.png")
    # lista completa dos julgamentos não unânimes: insumo da análise qualitativa
    cols = ["orgao", "classe", "processo", "data", "relator", "lavrador", "subtema", "assuntos_cnj",
            "vencidos", "vencido_parcial", "tese_repetitiva", "afetacao", "alerta", "ementa", "decisao"]
    votos[votos["maioria"]].sort_values(["orgao", "data"])[cols].to_csv(out / "julgamentos_nao_unanimes.csv",
                                                                         index=False)
    amostra_validacao(votos).to_csv(out / "amostra_validacao.csv", index=False)
    return votos


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", required=True)
    ap.add_argument("--saida", default="resultados")
    ap.add_argument("--min-juntos", type=int, default=5)
    ap.add_argument("--inicio", default=None, help="data mínima de julgamento, ex.: 2018-01-01")
    ap.add_argument("--baixar", action="store_true", help="baixa os espelhos das 3 turmas/seção")
    ap.add_argument("--baixar-assuntos", action="store_true", help="extrai assuntos CNJ dos metadados das Íntegras")
    ap.add_argument("--so-cnj", action="store_true", help="desliga o critério textual de contingência")
    ap.add_argument("--paralelos", type=int, default=6, help="downloads simultâneos (padrão 6)")
    a = ap.parse_args()
    if a.baixar:
        baixar_espelhos(a.dados, paralelos=min(a.paralelos, 4))
    if a.baixar_assuntos:
        baixar_assuntos(Path(a.dados) / "assuntos", inicio=a.inicio, paralelos=a.paralelos)
    executar(a.dados, a.saida, a.min_juntos, a.inicio, a.so_cnj)
