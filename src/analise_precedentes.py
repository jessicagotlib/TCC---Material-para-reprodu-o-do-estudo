# -*- coding: utf-8 -*-
"""
Etapa 3 — Formadores de jurisprudência e núcleo de teses em saúde suplementar
===============================================================================
Pergunta: quais precedentes os acórdãos de saúde suplementar citam como fundamento, quem os relatou,
quando e como foram julgados, e se esses precedentes formam um grupo central de teses.

Fonte das citações (dentro da base já baixada)
  * ementa e informações complementares (citações no texto e, nas ementas a partir de 2024, a seção
    "Jurisprudência relevante citada", que traz relator, órgão e data de cada precedente)
  * campo `jurisprudenciaCitada` do espelho, quando presente na base (versão atual do pipeline_stj.py)

Metadados do precedente citado (relator, órgão, data, ementa, tema), em ordem de preferência
  1. espelho completo do próprio precedente (--espelhos aponta para dados/espelhos; inclui o histórico)
  2. o que as citações informam ("relator Ministro X, Quarta Turma, julgado em 10/12/2019")
  3. a própria base, se o precedente também for de saúde suplementar no período

Saídas: ranking de precedentes, ranking de ministros formadores, linha do tempo, rede de cocitação com
comunidades (grupos de teses) e a tabela do núcleo de teses.

Uso
  python analise_precedentes.py --base ./resultados/base_votacao.pkl --espelhos ./dados/espelhos --saida ./resultados_precedentes
"""
from __future__ import annotations

import argparse
import itertools
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_stj import MARCOS, Canonizador, _norm, _rotulo

CLASSES = r"EREsp|EAREsp|REsp|AREsp|RMS|CC|MS|Rcl|PUIL|IAC"
# prefixos de incidentes: "AgInt no", "EDcl nos EDcl no AgInt no" ...
PREFIXO = r"(?:(?:AgInt|AgRg|EDcl|ED)\s+n[oa]s?\s+)*"
RX_CIT = re.compile(rf"\b(?P<rotulo>{PREFIXO}(?P<classe>{CLASSES}))\s*(?:n[º°.o]*\s*)?"
                    rf"(?P<num>\d{{1,2}}\.\d{{3}}\.\d{{3}}|\d{{6,7}})(?:\s*/\s*(?P<uf>[A-Z]{{2}}))?(?:-(?P<uf2>[A-Z]{{2}}))?")
RX_REL = re.compile(r"(?:relator[a]?|Rel\.)\s*(?:para\s+(?:o\s+)?ac[óo]rd[ãa]o\s*)?(?:p/\s*ac[óo]rd[ãa]o\s*)?"
                    r"(?:o\s+|a\s+)?Min(?:istr[oa])?\.?\s+(?P<nome>[A-ZÁ-Ú][\wÁ-úçãõ\.]+(?:\s+(?:de|da|do|dos)?\s*[A-ZÁ-Ú][\wÁ-úçãõ]+){0,4})",
                    re.I)
RX_ORG = re.compile(r"(Primeira|Segunda|Terceira|Quarta|Quinta|Sexta)\s+(Turma|Se[çc][ãa]o)|Corte\s+Especial", re.I)
RX_DATA = re.compile(r"julgad[oa]s?\s+em\s+(\d{1,2}/\d{1,2}/\d{4})|DJ[e]?\s*(?:de\s+)?(\d{1,2}/\d{1,2}/\d{4})", re.I)
RX_TEMA = re.compile(r"\bTema[s]?\s*(?:Repetitivo\s*)?(?:n[º°.]*\s*)?(\d\.?\d{2,3})\b", re.I)
RX_SUMULA = re.compile(r"S[úu]mula[s]?\s*(?:n[º°.]*\s*)?(\d{1,3})\s*/?\s*(?:do\s+)?STJ", re.I)


def _num(n: str) -> str:
    return re.sub(r"\D", "", n)


_FIM_FRASE = re.compile(r"(?<![nN]º)(?<!\bn)(?<!art)(?<!\d)\.\s+(?=[A-ZÁ-Ú\d\"“])")


def _frase_da_citacao(texto: str, ini: int, fim: int) -> str:
    """Frase da ementa que contém a citação, sem as referências entre parênteses. É ali que o acórdão
    citante enuncia a tese que atribui ao precedente ("No julgamento do EREsp X, a Segunda Seção firmou...")."""
    antes = texto[max(0, ini - 500):ini]
    cortes = list(_FIM_FRASE.finditer(antes))
    inicio = cortes[-1].end() if cortes else 0
    depois = texto[fim:fim + 700]
    m = _FIM_FRASE.search(depois)
    frase = antes[inicio:] + texto[ini:fim] + (depois[:m.start() + 1] if m else depois[:400])
    if re.search(r"Jurisprud[êe]ncia relevante citada|Dispositivos relevantes", frase, re.I):
        return ""
    frase = re.sub(r"\((?:[^()]|\([^()]*\))*\)", "", frase)                  # remove parênteses de referência
    frase = re.sub(r"^\s*\d+(?:\.\d+)*\.\s*", "", frase).strip(" ,;:")
    # descarta listas de referências ("Ministro X, Quarta Turma, julgado em ...; AgInt no ...")
    if re.search(r"julgad[oa]s? em|\bDJe?\b|\bRel\.|relator[a]? Ministr|Turma,|Se[çc][ãa]o,", frase, re.I):
        return ""
    if not re.match(r"[A-ZÁ-Ú\"“]", frase):
        return ""
    return frase if 60 <= len(frase) <= 600 else ""


def _medoide(frases: list[str]) -> str:
    """Frase mais representativa: maior semelhança média (Jaccard de palavras) com as demais."""
    frases = list(dict.fromkeys(f for f in frases if f))
    if len(frases) <= 2:
        return max(frases, key=len) if frases else ""
    toks = [set(re.findall(r"[a-záéíóúâêôãõç]{4,}", f.lower())) for f in frases[:150]]
    score = [np.mean([len(a & b) / max(1, len(a | b)) for b in toks]) for a in toks]
    return frases[int(np.argmax(score))]


# ----------------------------------------------------------------------------------------------
def extrair_citacoes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cits, outros = [], []
    for _, r in df.iterrows():
        proprio = _num(str(r["processo"]))
        fontes = [("ementa", str(r.get("ementa") or "")), ("info_compl", str(r.get("info_compl") or "")),
                  ("jurisp_citada", str(r.get("jurisp_citada") or ""))]
        vistos = set()
        for fonte, t in fontes:
            t = re.sub(r"\s+", " ", t)
            for m in RX_CIT.finditer(t):
                num = _num(m.group("num"))
                chave = f"{m.group('classe')} {num}"
                if num == proprio or (chave, fonte) in vistos:
                    continue
                vistos.add((chave, fonte))
                ctx = t[m.end(): m.end() + 260]
                ctx = re.split(r";|\)\.|\s(?=(?:AgInt|AgRg|EDcl|EREsp|REsp|AREsp)\s)", ctx)[0]
                rel = RX_REL.search(ctx); org = RX_ORG.search(ctx); dat = RX_DATA.search(ctx)
                cits.append({"id_citante": r["id"], "orgao_citante": r["orgao"], "relator_citante": r.get("relator_c"),
                             "data_citante": r["data"], "fonte": fonte, "precedente": chave,
                             "rotulo": re.sub(r"\s+", " ", m.group("rotulo")) + " " + f"{int(num):,}".replace(",", "."),
                             "uf": m.group("uf") or m.group("uf2"),
                             "rel_ctx": rel.group("nome") if rel else None,
                             "org_ctx": org.group(0).title() if org else None,
                             "data_ctx": (dat.group(1) or dat.group(2)) if dat else None,
                             "frase": _frase_da_citacao(t, m.start(), m.end()) if fonte == "ementa" else ""})
            for m in RX_TEMA.finditer(t):
                outros.append({"id_citante": r["id"], "tipo": "Tema repetitivo", "codigo": _num(m.group(1))})
            for m in RX_SUMULA.finditer(t):
                outros.append({"id_citante": r["id"], "tipo": "Súmula STJ", "codigo": m.group(1)})
    c = pd.DataFrame(cits)
    # uma citação por (acórdão citante, precedente), mesmo que apareça na ementa e na jurisprudência citada
    c = c.sort_values("fonte").drop_duplicates(["id_citante", "precedente"], keep="first")
    o = pd.DataFrame(outros).drop_duplicates() if outros else pd.DataFrame(columns=["id_citante", "tipo", "codigo"])
    return c.reset_index(drop=True), o


def indice_espelhos(pasta: str | Path | None) -> dict[str, dict]:
    """numero+classe -> metadados do espelho do próprio precedente (todo o histórico baixado)."""
    if not pasta or not Path(pasta).exists():
        return {}
    from pipeline_stj import carregar_espelhos
    e = carregar_espelhos(pasta)
    e["classe_base"] = e["classe"].astype(str).str.extract(rf"({CLASSES})\s*$")[0]
    e["chave"] = e["classe_base"] + " " + e["processo"].astype(str).str.replace(r"\D", "", regex=True)
    e["original"] = e["classe"].astype(str).str.strip() == e["classe_base"]   # REsp, não AgInt no REsp
    e = e.sort_values(["original", "data"], ascending=[False, True]).drop_duplicates("chave")
    return e.set_index("chave")[["relator", "orgao", "data", "ementa", "tema", "tese", "decisao"]].to_dict("index")


def resolver_metadados(c: pd.DataFrame, base: pd.DataFrame, espelhos: dict, can: Canonizador) -> pd.DataFrame:
    base = base.copy()
    base["classe_base"] = base["classe"].astype(str).str.extract(rf"({CLASSES})\s*$")[0]
    base["chave"] = base["classe_base"] + " " + base["processo"].astype(str).str.replace(r"\D", "", regex=True)
    base["original"] = base["classe"].astype(str).str.strip() == base["classe_base"]
    b = base.sort_values(["original", "data"], ascending=[False, True]).drop_duplicates("chave").set_index("chave")

    linhas = []
    for prec, g in c.groupby("precedente"):
        rel_ctx = Counter(filter(None, (can(x) for x in g["rel_ctx"].dropna())))
        org_ctx = Counter(g["org_ctx"].dropna()); dat_ctx = Counter(g["data_ctx"].dropna())
        fonte_meta, relator, orgao, data, ementa, tema = "citacao", None, None, None, "", ""
        if prec in espelhos:
            e = espelhos[prec]; fonte_meta = "espelho"
            relator, orgao, data, ementa, tema = can(e["relator"]), e["orgao"], e["data"], e["ementa"], e.get("tema") or ""
        elif prec in b.index:
            e = b.loc[prec]; fonte_meta = "base"
            relator, orgao, data, ementa = e["relator_c"], e["orgao"], e["data"], e["ementa"]
        relator = relator or (rel_ctx.most_common(1)[0][0] if rel_ctx else None)
        orgao = orgao or (org_ctx.most_common(1)[0][0].upper() if org_ctx else None)
        if data is None or pd.isna(data):
            data = pd.to_datetime(dat_ctx.most_common(1)[0][0], dayfirst=True, errors="coerce") if dat_ctx else pd.NaT
        frases = [f for f in g["frase"] if f]
        linhas.append({
            "precedente": prec, "rotulo_mais_comum": g["rotulo"].mode().iat[0],
            "citacoes": g["id_citante"].nunique(),
            "citantes_3a_turma": int((g["orgao_citante"] == "TERCEIRA TURMA").sum()),
            "citantes_4a_turma": int((g["orgao_citante"] == "QUARTA TURMA").sum()),
            "citantes_2a_secao": int((g["orgao_citante"] == "SEGUNDA SEÇÃO").sum()),
            "relator": relator, "orgao": str(orgao).upper() if orgao else None, "data_julgamento": data,
            "autocitacoes": int((g["relator_citante"] == relator).sum()) if relator else 0,
            "primeira_citacao": g["data_citante"].min(), "ultima_citacao": g["data_citante"].max(),
            "tema": tema, "fonte_metadados": fonte_meta,
            "tese_enunciada_pelos_citantes": _medoide(frases),
            "ementa_precedente": re.sub(r"\s+", " ", str(ementa or ""))[:600],
        })
    p = pd.DataFrame(linhas).sort_values("citacoes", ascending=False).reset_index(drop=True)
    p = _unir_julgamentos_conjuntos(p, c)
    p["qualificado"] = (p["precedente"].str.startswith(("EREsp", "EAREsp")) | p["orgao"].fillna("").str.contains("SEÇÃO|CORTE")
                        | p["tema"].astype(str).str.strip().ne(""))
    return p


def _unir_julgamentos_conjuntos(p: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    """EREsp 1.886.929 e 1.889.704, por exemplo, foram julgados juntos e são citados como um só precedente.
    Regra: mesma classe, relator, órgão de uniformização (Seção/Corte Especial) e data -> um precedente, com a união dos citantes.
    Os membros ficam listados em `julgados_em_conjunto` e o mapa é aplicado também à tabela de citações."""
    p = p.copy()
    p["classe_base"] = p["precedente"].str.split().str[0]
    # só em órgãos de uniformização: nas Turmas, agravos do mesmo relator na mesma sessão são casos distintos
    ok = (p["relator"].notna() & p["data_julgamento"].notna()
          & p["orgao"].fillna("").str.contains("SEÇÃO|SECAO|CORTE"))
    chave = p["classe_base"] + "|" + p["relator"].astype(str) + "|" + p["orgao"].astype(str) + "|" + p["data_julgamento"].astype(str)
    p["_g"] = np.where(ok, chave, p["precedente"])
    mapa = {}
    linhas = []
    for _, g in p.groupby("_g", sort=False):
        g = g.sort_values("citacoes", ascending=False)
        lider = g.iloc[0].copy()
        if len(g) > 1:
            membros = list(g["precedente"])
            for m in membros:
                mapa[m] = lider["precedente"]
            cit = c[c["precedente"].isin(membros)]
            lider["citacoes"] = cit["id_citante"].nunique()
            for col, org in [("citantes_3a_turma", "TERCEIRA TURMA"), ("citantes_4a_turma", "QUARTA TURMA"), ("citantes_2a_secao", "SEGUNDA SEÇÃO")]:
                lider[col] = cit.drop_duplicates("id_citante").eq(org)["orgao_citante"].sum()
            lider["autocitacoes"] = int(cit.drop_duplicates("id_citante")["relator_citante"].eq(lider["relator"]).sum())
            lider["rotulo_mais_comum"] = " + ".join(g["rotulo_mais_comum"])
            lider["primeira_citacao"], lider["ultima_citacao"] = cit["data_citante"].min(), cit["data_citante"].max()
        lider["julgados_em_conjunto"] = len(g)
        linhas.append(lider)
    c.loc[:, "precedente"] = c["precedente"].replace(mapa)
    return pd.DataFrame(linhas).drop(columns=["_g"]).sort_values("citacoes", ascending=False).reset_index(drop=True)


def ranking_ministros(p: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for rel, g in p[p["relator"].notna()].groupby("relator"):
        cit = sorted(g["citacoes"], reverse=True)
        h = sum(1 for i, x in enumerate(cit, 1) if x >= i)          # índice h jurisprudencial
        ids = c[c["precedente"].isin(g["precedente"])]
        linhas.append({"ministro": _rotulo(rel), "precedentes_citados": len(g), "citacoes_recebidas": int(g["citacoes"].sum()),
                       "indice_h": h, "precedentes_qualificados": int(g["qualificado"].sum()),
                       "%_citacoes_por_outros_relatores": round(100 * (ids["relator_citante"] != rel).mean(), 1),
                       "%_citacoes_de_outra_turma": round(100 * (ids["orgao_citante"] != g["orgao"].mode().iat[0]).mean(), 1)
                       if g["orgao"].notna().any() else np.nan,
                       "precedente_mais_citado": g.iloc[0]["rotulo_mais_comum"], "citacoes_do_mais_citado": int(g.iloc[0]["citacoes"])})
    return pd.DataFrame(linhas).sort_values(["citacoes_recebidas", "indice_h"], ascending=False)


def cocitacao(p: pd.DataFrame, c: pd.DataFrame, min_cit: int, seed: int = 42):
    """Rede de cocitação: dois precedentes ligados quando citados no mesmo acórdão (peso = nº de acórdãos).
    Comunidades por Louvain = grupos de teses que costumam ser mobilizadas juntas."""
    import networkx as nx
    nucleo = set(p.loc[p["citacoes"] >= min_cit, "precedente"])
    G = nx.Graph()
    G.add_nodes_from(sorted(nucleo))
    for _, g in c[c["precedente"].isin(nucleo)].groupby("id_citante"):
        for a, b in itertools.combinations(sorted(set(g["precedente"])), 2):
            G.add_edge(a, b, weight=G.get_edge_data(a, b, {"weight": 0})["weight"] + 1)
    comunidades = nx.community.louvain_communities(G, weight="weight", seed=seed) if G.number_of_edges() else [{n} for n in G]
    mapa = {n: i + 1 for i, com in enumerate(sorted(comunidades, key=lambda s: -sum(p.set_index("precedente").loc[list(s), "citacoes"])))
            for n in com}
    pr = nx.pagerank(G, weight="weight") if G.number_of_edges() else {n: 0 for n in G}
    return G, mapa, pr


def rotular_grupos(p: pd.DataFrame, c: pd.DataFrame, base: pd.DataFrame, n_termos: int = 6) -> dict[int, str]:
    """Rótulo de cada grupo = termos mais distintivos (TF-IDF) dos cabeçalhos das ementas que citam o grupo."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    cab = base.set_index("id")["ementa"].fillna("").str.replace(r"\s+", " ", regex=True).str.extract(
        r"^(.*?)(?:\s1\.\s|\s1\s-|\sI\.\s|$)")[0]
    docs, grupos = [], []
    for gid, g in p[p["grupo"].notna()].groupby("grupo"):
        ids = c.loc[c["precedente"].isin(g["precedente"]), "id_citante"].unique()
        docs.append(" ".join(cab.reindex(ids).fillna("")))
        grupos.append(int(gid))
    if not docs:
        return {}
    pare = ["agravo", "interno", "recurso", "especial", "processual", "civil", "direito", "no", "nos", "na", "de", "da", "do",
            "dos", "das", "e", "em", "a", "o", "ao", "ação", "plano", "saúde", "planos", "decisão", "mantida", "desprovido",
            "provido", "não", "súmula", "stj", "caso", "exame", "com", "por", "para", "art", "n", "conhecido", "embargos"]
    v = TfidfVectorizer(stop_words=pare, ngram_range=(1, 3), min_df=1, max_features=5000, token_pattern=r"(?u)\b[^\W\d_]{3,}\b")
    X = v.fit_transform(docs)
    termos = np.array(v.get_feature_names_out())
    return {gid: ", ".join(termos[np.argsort(-X[i].toarray()[0])[:n_termos]]) for i, gid in enumerate(grupos)}


# ----------------------------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------------------------
def fig_ministros(r: pd.DataFrame, arquivo: Path, top: int = 15):
    import matplotlib.pyplot as plt
    d = r.head(top).iloc[::-1]
    outros = d["citacoes_recebidas"] * d["%_citacoes_por_outros_relatores"] / 100
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(d) + 1.5))
    ax.barh(d["ministro"], outros, color="#2e6fad", label="citado por outros relatores")
    ax.barh(d["ministro"], d["citacoes_recebidas"] - outros, left=outros, color="#a9c6e4", label="autocitação")
    for i, (_, x) in enumerate(d.iterrows()):
        ax.text(x["citacoes_recebidas"] + 3, i, f"h={x['indice_h']} · {x['precedentes_citados']} precedentes", va="center", fontsize=7.5)
    ax.set_xlabel("Citações recebidas pelos precedentes relatados (acórdãos de saúde suplementar, 2022–2026)")
    ax.set_title("Ministros formadores de jurisprudência em saúde suplementar", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, d["citacoes_recebidas"].max() * 1.35)
    plt.tight_layout(); plt.savefig(arquivo, dpi=200); plt.close()


def fig_linha_tempo(p: pd.DataFrame, arquivo: Path, top: int = 12):
    import matplotlib.pyplot as plt
    d = p[p["data_julgamento"].notna()].head(60).copy()
    cores = {"SEGUNDA SEÇÃO": "#d62728", "TERCEIRA TURMA": "#1f77b4", "QUARTA TURMA": "#2ca02c", "CORTE ESPECIAL": "#9467bd"}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for org, g in d.groupby(d["orgao"].fillna("OUTRO")):
        ax.scatter(g["data_julgamento"], g["citacoes"], s=30 + 4 * g["citacoes"], alpha=0.65, color=cores.get(org, "#888"),
                   label=org.title(), edgecolor="white")
    for _, x in d.head(top).iterrows():
        rot = x["rotulo_mais_comum"] if len(x["rotulo_mais_comum"]) < 40 else x["rotulo_mais_comum"].split(" + ")[0] + " e outros"
        ax.annotate(f"{rot}\n{_rotulo(str(x['relator'] or '?'))}", (x["data_julgamento"], x["citacoes"]),
                    fontsize=6.5, xytext=(5, 3), textcoords="offset points")
    for data, nome in MARCOS:
        ax.axvline(pd.Timestamp(data), color="#aaa", ls=":", lw=1)
    ax.set_yscale("log"); ax.set_ylabel("Acórdãos que citam o precedente (escala log)")
    ax.set_xlabel("Data de julgamento do precedente")
    ax.set_title("Quando foram julgados os precedentes mais citados", loc="left", fontweight="bold")
    hs = [plt.Line2D([], [], marker="o", ls="", markersize=7, color=cores.get(o, "#888"), label=o.title())
          for o in sorted(d["orgao"].fillna("OUTRO").unique())]
    ax.legend(handles=hs, frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(arquivo, dpi=200); plt.close()


def fig_citacoes_no_tempo(p: pd.DataFrame, c: pd.DataFrame, arquivo: Path, top: int = 6):
    import matplotlib.pyplot as plt
    tops = p.head(top)["precedente"]
    d = c[c["precedente"].isin(tops)].copy()
    d["semestre"] = d["data_citante"].dt.year.astype(str) + "-S" + ((d["data_citante"].dt.month > 6) + 1).astype(str)
    t = d.groupby(["semestre", "precedente"])["id_citante"].nunique().unstack(fill_value=0)
    ax = t.plot(marker="o", figsize=(10, 4.2))
    ax.set_ylabel("Acórdãos que citam no semestre"); ax.set_xlabel("")
    ax.set_title("Uso dos precedentes mais citados ao longo do tempo", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(arquivo, dpi=200); plt.close()


def fig_rede(G, mapa, p: pd.DataFrame, rotulos: dict, arquivo: Path, seed: int = 42):
    import matplotlib.pyplot as plt
    import networkx as nx
    if G.number_of_nodes() == 0:
        return
    cit = p.set_index("precedente")["citacoes"]
    rel = p.set_index("precedente")["relator"]
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    principal = G.subgraph(comps[0])
    perifericos = [sorted(c, key=lambda n: -cit[n]) for c in comps[1:]]       # componentes pequenos e isolados
    pos = nx.kamada_kawai_layout(principal, weight=None)
    H = principal
    xs = [n for comp in perifericos for n in comp]
    for i, n in enumerate(xs):
        pos[n] = np.array([-1 + 2 * (i + 0.5) / max(1, len(xs)), -1.4])
    isolados = xs
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(13, 10))
    pesos = np.array([H[u][v]["weight"] for u, v in H.edges()]) if H.number_of_edges() else np.array([1])
    nx.draw_networkx_edges(H, pos, ax=ax, width=0.3 + 3 * pesos / pesos.max(), alpha=0.25, edge_color="#777")
    for n in G.nodes():
        ax.scatter(*pos[n], s=40 + 10 * cit[n], color=cmap((mapa[n] - 1) % 20), edgecolor="white", zorder=3)
    for n in sorted(G.nodes(), key=lambda n: -cit[n])[:15]:
        ax.annotate(f"{n}\n{_rotulo(str(rel[n] or '?'))}", pos[n], fontsize=7, ha="center", va="bottom",
                    xytext=(0, 7), textcoords="offset points", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    if isolados:
        ax.text(-1, -1.25, "Precedentes centrais citados fora do bloco principal (sem cocitação com ele):", fontsize=8, color="#555")
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap((g - 1) % 20), label=f"Grupo {g}: {rotulos.get(g, '')[:70]}")
               for g in sorted(set(mapa.values()))[:12]]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.01), ncol=1)
    ax.set_title("Rede de cocitação dos precedentes centrais (cor = grupo de teses; tamanho = citações)", loc="left",
                 fontweight="bold")
    ax.axis("off"); plt.tight_layout(); plt.savefig(arquivo, dpi=200, bbox_inches="tight"); plt.close()


# ----------------------------------------------------------------------------------------------
def executar(base_pkl: str, saida: str, pasta_espelhos: str | None = None, min_cit: int | None = None):
    out = Path(saida); out.mkdir(parents=True, exist_ok=True)
    base = pd.read_pickle(base_pkl)
    can = Canonizador(base["relator"])
    c, outros = extrair_citacoes(base)
    print(f"[citações] {len(c)} citações de {c['precedente'].nunique()} precedentes em "
          f"{c['id_citante'].nunique()} de {len(base)} acórdãos ({100 * c['id_citante'].nunique() / len(base):.1f}%)")
    print("[citações] por fonte:", c["fonte"].value_counts().to_dict())
    esp = indice_espelhos(pasta_espelhos)
    p = resolver_metadados(c, base, esp, can)
    # núcleo: precedentes com ao menos 10 citações (ajustável); ver a curva de concentração para justificar
    min_cit = min_cit or 10
    c = c.drop_duplicates(["id_citante", "precedente"])
    G, mapa, pr = cocitacao(p, c, min_cit)
    p["grupo"] = p["precedente"].map(mapa)
    p["pagerank_cocitacao"] = p["precedente"].map(pr)
    rot = rotular_grupos(p, c, base)
    p["rotulo_grupo"] = p["grupo"].map(rot)
    proc = (r"prequestionamento|impugna[çc][ãa]o|impugna[çc][ãa]o espec[íi]fica|diss[íi]dio|honor[áa]rios|astreintes|reexame|"
            r"s[úu]mula[s]? (?:n\.? ?)?(?:5|7|83|182|211|284)\b|art\. 1\.022|embargos de declara[çc][ãa]o|multa|prescri[çc][ãa]o")
    txt = p["tese_enunciada_pelos_citantes"].fillna("") + " " + p["ementa_precedente"].fillna("").str[:300]
    pelo_texto = (txt.str.contains(proc, case=False, regex=True) &
                  ~txt.str.contains(r"cobertura|\brol\b|reajuste|reembolso|tratamento|medicamento", case=False, regex=True))
    pelo_grupo = p["rotulo_grupo"].fillna("").str.contains(r"impugna|honor[áa]rios|astreintes|prequestionamento", case=False)
    p["natureza_tese"] = np.where(pelo_texto | (pelo_grupo & txt.str.strip().eq("")), "processual", "material")

    # concentração das citações
    s = p["citacoes"].sort_values(ascending=False).cumsum() / p["citacoes"].sum()
    n50 = int((s < 0.5).sum() + 1)
    x = np.sort(p["citacoes"].to_numpy(dtype=float)); n = len(x)
    gini = float(((2 * np.arange(1, n + 1) - n - 1) * x).sum() / (n * x.sum()))

    p.to_csv(out / "precedentes_ranking.csv", index=False)
    nucleo = p[p["grupo"].notna()].sort_values(["grupo", "citacoes"], ascending=[True, False])
    nucleo[["grupo", "rotulo_grupo", "natureza_tese", "rotulo_mais_comum", "relator", "orgao", "data_julgamento", "citacoes", "qualificado",
            "autocitacoes", "primeira_citacao", "ultima_citacao", "tema", "tese_enunciada_pelos_citantes", "ementa_precedente",
            "fonte_metadados"]].to_csv(out / "nucleo_de_teses.csv", index=False)
    rm = ranking_ministros(p, c); rm.to_csv(out / "ministros_formadores.csv", index=False)
    # relatores do grupo ponderados pelas citações que seus precedentes receberam
    peso = nucleo.groupby(["grupo", "relator"])["citacoes"].sum().reset_index().sort_values(["grupo", "citacoes"], ascending=[True, False])
    rel_grupo = peso.groupby("grupo").apply(lambda g: ", ".join(f"{_rotulo(r)} ({c})" for r, c in zip(g["relator"], g["citacoes"]))
                                            ).rename("relatores_ponderados")
    grupos = (nucleo.groupby(["grupo", "rotulo_grupo"]).agg(precedentes=("precedente", "size"), citacoes=("citacoes", "sum"),
                                                             relatores=("relator", "first"),
                                                             mais_citado=("rotulo_mais_comum", "first"))
              .reset_index().drop(columns="relatores").merge(rel_grupo, on="grupo", how="left"))
    grupos.to_csv(out / "grupos_de_teses.csv", index=False)
    outros.groupby(["tipo", "codigo"]).size().sort_values(ascending=False).rename("acordaos").to_csv(
        out / "temas_e_sumulas_citados.csv")
    c.to_csv(out / "citacoes.csv", index=False)

    fig_ministros(rm, out / "fig_ministros_formadores.png")
    fig_linha_tempo(p, out / "fig_linha_do_tempo_precedentes.png")
    fig_citacoes_no_tempo(p, c, out / "fig_uso_no_tempo.png")
    fig_rede(G, mapa, p, rot, out / "fig_rede_cocitacao.png")

    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
    print(f"\n[concentração] {len(p)} precedentes; {n50} respondem por 50% das citações; Gini = {gini:.2f}")
    print(f"[núcleo] precedentes com >= {min_cit} citações: {G.number_of_nodes()} em {len(set(mapa.values()))} grupos")
    print("\n=== 20 precedentes mais citados\n", p.head(20)[["rotulo_mais_comum", "relator", "orgao", "data_julgamento", "citacoes",
                                                         "qualificado", "grupo", "fonte_metadados"]].to_string(index=False))
    print("\n=== ministros formadores\n", rm.head(15).to_string(index=False))
    print("\n=== grupos de teses\n", grupos.to_string(index=False))
    return p, rm, grupos


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="resultados/base_votacao.pkl")
    ap.add_argument("--espelhos", default="dados/espelhos", help="pasta com TODOS os espelhos baixados (resolve relator/data)")
    ap.add_argument("--saida", default="resultados_precedentes")
    ap.add_argument("--min-citacoes", type=int, default=None, help="corte do núcleo (padrão: 10 citações)")
    a = ap.parse_args()
    executar(a.base, a.saida, a.espelhos, a.min_citacoes)
