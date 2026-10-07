# -*- coding: utf-8 -*-
"""
Etapa 7 — Gráficos exploratórios da jurimetria da saúde suplementar
====================================================================
Substitui os mapas de proximidade entre ministros (inviáveis com 19 julgamentos não unânimes) por
figuras que os dados sustentam. Todas saem da base já coletada.

  1. funil        — do acórdão ao provimento: quantos param em filtro processual e quantos são providos
  2. volume       — acórdãos por semestre e subtema (área empilhada), com os marcos normativos
  3. subtemas     — subtema x período: volume e taxa de provimento no mérito
  4. relatores    — % de óbice e % de provimento por relator, com IC 95% (gráfico de floresta)
  5. divergencia  — proporção de julgamentos não unânimes por órgão e classe recursal, com IC 95%
  6. razao        — o que distingue não unânimes de unânimes: razão de chances com IC (Fisher)
  7. argumentos   — quais argumentos aparecem juntos (lift) e como evoluem
  8. deliberacao  — voto-vista por semestre e por relator: a deliberação que precede a divergência
  9. precedentes  — idade do precedente no momento da citação e curva de concentração (Lorenz)

Uso
  python graficos_exploratorios.py --base ./resultados/base_votacao.pkl --saida ./resultados_graficos \
      --precedentes ./resultados_precedentes
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analise_fundamentos import ARGUMENTOS, OBICES, classificar
from pipeline_stj import MARCOS, _rotulo, wilson

ROTULOS = {
    "rol_taxativo_ou_mitigado": "rol taxativo ou mitigado", "rol_exemplificativo": "rol exemplificativo",
    "prescricao_medica_deferencia": "deferência à prescrição médica", "tea_autismo": "transtorno do espectro autista",
    "cdc_consumidor": "Código de Defesa do Consumidor", "dano_moral": "dano moral",
    "lei_14454_2022": "Lei 14.454/2022", "abusividade": "abusividade", "autogestao": "autogestão",
    "urgencia_emergencia": "urgência ou emergência", "registro_anvisa_offlabel": "registro na Anvisa ou off-label",
    "home_care": "atendimento domiciliar", "reembolso_rede": "reembolso e rede credenciada",
    "reajuste": "reajuste", "cancelamento_contrato_coletivo": "cancelamento de contrato coletivo",
    "aposentado_demitido_art30_31": "aposentados e demitidos",
    "tratamento_medico_hospitalar": "tratamento médico-hospitalar", "cobertura_rol": "cobertura e rol",
    "medicamentos": "medicamentos", "insumos": "insumos", "home care": "atendimento domiciliar",
    "manutencao_contrato": "manutenção do contrato", "reembolso_rede ": "reembolso e rede",
    "processual_competencia": "questões processuais", "outros": "outros",
    "voto_vista": "voto-vista", "recurso_especial": "recurso especial ou embargos de divergência",
    "obice_admissibilidade": "menção a filtro de admissibilidade", "sumula_7": "reexame de fatos e provas",
    "jurisp_consolidada": "jurisprudência consolidada", "tema_repetitivo": "tema repetitivo",
    "orgao_quarta": "Quarta Turma", "orgao_segunda": "Segunda Seção",
    "periodo_14.454": "entre a Lei 14.454 e a ADI 7265", "periodo_7265": "após a ADI 7265",
    "subtema_cobertura_rol": "subtema: cobertura e rol", "subtema_reajuste": "subtema: reajuste",
    "subtema_medicamentos": "subtema: medicamentos", "subtema_outros": "subtema: outros",
    "subtema_home_care": "subtema: atendimento domiciliar",
}
rot = lambda s: ROTULOS.get(s, s.replace("_", " "))

CORES = {"TERCEIRA TURMA": "#1f77b4", "QUARTA TURMA": "#2ca02c", "SEGUNDA SEÇÃO": "#d62728"}
plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold"})


def _sem(s: pd.Series) -> pd.Series:
    return s.dt.year.astype(str) + "-S" + ((s.dt.month > 6) + 1).astype(str)


def _marcos(ax, semestres):
    for data, nome in MARCOS:
        sem = f"{data[:4]}-S{1 if int(data[5:7]) <= 6 else 2}"
        if sem in list(semestres):
            ax.axvline(list(semestres).index(sem), color="#999", ls=":", lw=1)
            ax.text(list(semestres).index(sem), ax.get_ylim()[1], " " + nome.split("(")[0], rotation=90,
                    fontsize=6.5, va="top", color="#666")


def _limpa(ax):
    ax.spines[["top", "right"]].set_visible(False)


# 1 ---------------------------------------------------------------------------------------------
def fig_funil(df: pd.DataFrame, out: Path):
    etapas = [("Acórdãos de saúde suplementar", len(df)),
              ("Com certidão de julgamento", int((df["alerta"] != "sem_texto_decisao").sum())),
              ("Enfrentam o mérito", int((df["caminho"] == "merito").sum())),
              ("Recurso provido (total ou parcial)", int(df["resultado"].isin(["provido", "provimento_parcial"]).sum())),
              ("Julgados por maioria", int(df["maioria"].sum()))]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    y = np.arange(len(etapas))[::-1]
    ax.barh(y, [n for _, n in etapas], color=["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"], height=0.6)
    for yi, (rot, n) in zip(y, etapas):
        pct = f"{100 * n / len(df):.1f}".replace(".", ",")
        ax.text(n + 60, yi, f"{n:,}".replace(",", ".") + f"  ({pct}%)", va="center", fontsize=10)
    ax.set_yticks(y); ax.set_yticklabels([r for r, _ in etapas], fontsize=10)
    ax.set_xlim(0, len(df) * 1.24); ax.set_xlabel("Acórdãos", fontsize=10); _limpa(ax)
    ax.tick_params(axis="x", labelsize=9.5)
    fig.tight_layout(); fig.savefig(out / "fig_1_funil.png", dpi=300); plt.close(fig)


# 2 ---------------------------------------------------------------------------------------------
def fig_volume(df: pd.DataFrame, out: Path):
    d = df.copy(); d["semestre"] = _sem(d["data"])
    t = d.pivot_table(index="semestre", columns="subtema", values="id", aggfunc="count").fillna(0)
    t = t[t.sum().sort_values(ascending=False).index]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].stackplot(range(len(t)), t.T.values, labels=t.columns, colors=plt.get_cmap("tab20").colors[:len(t.columns)])
    axes[0].set_ylabel("Acórdãos no semestre"); axes[0].legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left")
    axes[0].set_title("Volume por subtema (classificação CNJ) e por órgão", loc="left"); _limpa(axes[0])
    _marcos(axes[0], t.index)
    o = d.pivot_table(index="semestre", columns="orgao", values="id", aggfunc="count").fillna(0)
    for col in o.columns:
        axes[1].plot(range(len(o)), o[col], marker="o", ms=4, color=CORES.get(col, "#888"), label=col.title())
    axes[1].set_ylabel("Acórdãos no semestre"); axes[1].legend(frameon=False, fontsize=8); _limpa(axes[1])
    axes[1].set_xticks(range(len(t))); axes[1].set_xticklabels(t.index, rotation=45, ha="right", fontsize=8)
    fig.text(0.01, 0.005, "Atenção: o volume reflete quantos acórdãos receberam espelho, não o total julgado. "
                          "A partir de 2025 a base passou a incluir recursos especiais e agravos em recurso "
                          "especial, antes quase ausentes; o último semestre é parcial.", fontsize=7.5, color="#777")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out / "fig_2_volume.png"); plt.close(fig)


# 3 ---------------------------------------------------------------------------------------------
def fig_subtemas(df: pd.DataFrame, out: Path):
    m = df[df["caminho"] == "merito"].copy()
    m["provido"] = m["resultado"].isin(["provido", "provimento_parcial"])
    linhas = []
    for sub, g in m.groupby("subtema"):
        if len(g) < 30:
            continue
        lo, hi = wilson(int(g["provido"].sum()), len(g))
        linhas.append({"subtema": sub, "n": len(g), "taxa": g["provido"].mean(), "lo": lo, "hi": hi})
    t = pd.DataFrame(linhas).sort_values("taxa")
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(t) + 2))
    ax.errorbar(t["taxa"] * 100, range(len(t)), xerr=[(t["taxa"] - t["lo"]) * 100, (t["hi"] - t["taxa"]) * 100],
                fmt="o", color="#08519c", capsize=3)
    ax.axvline(100 * m["provido"].mean(), color="#d62728", ls="--", lw=1, label="média do mérito")
    ax.set_yticks(range(len(t))); ax.set_yticklabels([f"{s}  (n={n})" for s, n in zip(t["subtema"], t["n"])], fontsize=8.5)
    ax.set_xlabel("% de recursos providos entre os acórdãos que enfrentam o mérito (IC 95%)")
    ax.set_title("Onde o STJ reforma: provimento por subtema", loc="left"); ax.legend(frameon=False, fontsize=8); _limpa(ax)
    fig.tight_layout(); fig.savefig(out / "fig_3_subtemas.png"); plt.close(fig)


# 4 ---------------------------------------------------------------------------------------------
def fig_relatores(df: pd.DataFrame, out: Path, min_n: int = 30):
    d = df[df["relator_c"].notna()].copy()
    d["provido"] = d["resultado"].isin(["provido", "provimento_parcial", "embargos_acolhidos"])
    orgaos = [o for o in ["TERCEIRA TURMA", "QUARTA TURMA"] if o in set(d["orgao"])]
    fig, axes = plt.subplots(len(orgaos), 2, figsize=(9.4, 3.1 * len(orgaos)), squeeze=False)
    for i, org in enumerate(orgaos):
        g0 = d[d["orgao"] == org]
        rel = [r for r, n in g0["relator_c"].value_counts().items() if n >= min_n]
        for j, (col, titulo) in enumerate([("algum_obice", "% com filtro de admissibilidade"), ("provido", "% de provimento")]):
            ax = axes[i][j]
            linhas = []
            for r in rel:
                g = g0[g0["relator_c"] == r]
                lo, hi = wilson(int(g[col].sum()), len(g))
                linhas.append({"rel": _rotulo(r), "n": len(g), "p": g[col].mean(), "lo": lo, "hi": hi})
            t = pd.DataFrame(linhas).sort_values("p")
            ax.errorbar(t["p"] * 100, range(len(t)), xerr=[(t["p"] - t["lo"]) * 100, (t["hi"] - t["p"]) * 100],
                        fmt="o", color=CORES.get(org, "#333"), capsize=3)
            ax.set_yticks(range(len(t)))
            ax.set_yticklabels([f"{r} ({n})" for r, n in zip(t["rel"], t["n"])], fontsize=8.6)
            ax.set_title(f"{org.title()} — {titulo}", loc="left", fontsize=9.6, fontweight="bold")
            ax.axvline(100 * g0[col].mean(), color="#999", ls="--", lw=1); _limpa(ax)
            ax.set_xlabel("% (IC 95%)", fontsize=9); ax.tick_params(axis="x", labelsize=8.6)
    fig.tight_layout(); fig.savefig(out / "fig_4_relatores.png", dpi=300); plt.close(fig)


# 5 ---------------------------------------------------------------------------------------------
CLASSES_ORDEM = ["Proposta de afetação", "Embargos de divergência", "Embargos de declaração",
                 "Recurso especial", "Agravo em recurso especial", "Agravo interno", "Demais classes"]


def _familia_recursal(classe: str) -> str:
    """Agrupa a classe processual do espelho na família recursal correspondente."""
    import re
    c = str(classe)
    if re.search(r"ProAfR", c, re.I):
        return "Proposta de afetação"
    if re.search(r"EREsp|EAREsp", c):
        return "Embargos de divergência"
    if c.startswith("EDcl"):
        return "Embargos de declaração"
    if c.startswith(("AgInt", "AgRg")):
        return "Agravo interno"
    if c.startswith("REsp"):
        return "Recurso especial"
    if c.startswith("AREsp"):
        return "Agravo em recurso especial"
    return "Demais classes"


def taxa_nao_unanimidade(df: pd.DataFrame, min_n: int = 10) -> pd.DataFrame:
    """Proporção de julgamentos não unânimes por órgão e família recursal, com IC de Wilson.

    Só entram acórdãos com certidão de julgamento. Famílias com menos de `min_n` acórdãos
    dentro do grupo são reunidas em "Demais classes", para não gerar linhas sem interpretação.
    """
    d = df[df["decisao"].notna() & (df["decisao"].astype(str).str.strip() != "")].copy()
    d["familia"] = d["classe"].map(_familia_recursal)
    d["grupo"] = np.where(d["orgao"].eq("SEGUNDA SEÇÃO"), "Segunda Seção", "Terceira e Quarta Turmas")
    tam = d.groupby(["grupo", "familia"]).size()
    pequenas = set(tam[tam < min_n].index)
    d["familia"] = [("Demais classes" if (g, f) in pequenas else f)
                    for g, f in zip(d["grupo"], d["familia"])]
    linhas = []
    for (grupo, familia), g in d.groupby(["grupo", "familia"]):
        n, k = len(g), int(g["maioria"].sum())
        lo, hi = wilson(k, n)
        linhas.append(dict(grupo=grupo, classe=familia, n=n, nao_unanimes=k,
                           taxa=100 * k / n, ic_inf=100 * lo, ic_sup=100 * hi))
    r = pd.DataFrame(linhas)
    r["_ordem"] = np.where(r["classe"].eq("Demais classes"), -1.0, r["taxa"])
    return r.sort_values(["grupo", "_ordem"], ascending=[True, False]).drop(columns="_ordem").reset_index(drop=True)


def fig_divergencia(df: pd.DataFrame, out: Path):
    """Gráfico de floresta: onde a não unanimidade se concentra.

    Substitui a linha do tempo dos 19 casos. O eixo temporal sugeria que a divergência variava ao
    longo do período, quando os dados mostram que ela depende do órgão e da classe recursal.
    """
    r = taxa_nao_unanimidade(df)
    r.to_csv(out / "tab_nao_unanimidade_por_classe.csv", index=False, encoding="utf-8-sig")

    acento, neutro, tinta, suave = "#1f6fb2", "#6e7275", "#1a1a1a", "#6a6a6a"
    grupos = ["Segunda Seção", "Terceira e Quarta Turmas"]
    itens, ys, y = [], [], 0.0
    for i, grupo in enumerate(grupos):
        if i:
            y -= 0.75
        itens.append(("cabecalho", grupo, y)); y -= 0.95
        for _, linha in r[r["grupo"] == grupo].iterrows():
            itens.append(("dado", linha, y)); ys.append(y); y -= 1.0

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    xmax, x0 = 72, -35.0
    for tipo, obj, y in itens:
        if tipo == "cabecalho":
            ax.text(x0, y, obj, fontsize=10.5, fontweight="bold", color=tinta, va="center", ha="left")
            ax.plot([x0, xmax], [y - 0.5, y - 0.5], color="#d4d4d4", lw=0.8, clip_on=False, zorder=1)
            continue
        cor = acento if obj["grupo"] == "Segunda Seção" else neutro
        ax.plot([obj["ic_inf"], obj["ic_sup"]], [y, y], color=cor, lw=1.4, alpha=0.45,
                solid_capstyle="butt", zorder=2)
        ax.plot([obj["taxa"]], [y], "o", ms=7.5, color=cor, mec="white", mew=1.4, zorder=3)
        ax.text(x0 + 0.7, y, obj["classe"], fontsize=9.6, color=tinta, va="center", ha="left")
        ax.text(-1.2, y, f"{obj['nao_unanimes']} de {obj['n']:,}".replace(",", "."),
                fontsize=8.8, color=suave, va="center", ha="right")
        ax.text(obj["ic_sup"] + 1.6, y, f"{obj['taxa']:.1f}%".replace(".", ","),
                fontsize=9.4, color=cor, va="center", ha="left", fontweight="bold")

    ax.set_xlim(0, xmax); ax.set_ylim(min(ys) - 0.9, 0.9)
    ax.set_xticks(range(0, 71, 10))
    ax.set_xticklabels([f"{v}%" for v in range(0, 71, 10)], fontsize=9.2, color=suave)
    ax.set_yticks([])
    ax.set_xlabel("Julgamentos não unânimes (%), com intervalo de confiança de 95%",
                  fontsize=9.8, color=tinta, labelpad=8)
    ax.xaxis.grid(True, color="#e6e6e6", lw=0.7); ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color("#cfcfcf"); ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", length=0, pad=5)
    fig.subplots_adjust(left=0.345, right=0.975, top=0.97, bottom=0.135)
    fig.savefig(out / "fig_5_divergencia.png", dpi=300, facecolor="white"); plt.close(fig)


# 6 ---------------------------------------------------------------------------------------------
def fig_razao(df: pd.DataFrame, out: Path):
    from scipy.stats import fisher_exact
    flags = {"Voto-vista no julgamento": "voto_vista",
             "Decidido no mérito": "decidido_no_merito",
             "REsp ou embargos de divergência": "recurso_de_merito_originario",
             "Cita tema repetitivo": "tema_repetitivo",
             "Tema TEA / terapias": "tea_autismo",
             "Cita rol taxativo ou mitigado": "rol_taxativo_ou_mitigado",
             "Algum óbice de admissibilidade": "algum_obice",
             "Cita Súmula 7 (reexame de provas)": "sumula_7_reexame_provas"}
    linhas = []
    for rot, f in flags.items():
        if f not in df:
            continue
        a = int((df["maioria"] & df[f]).sum()); b = int((df["maioria"] & ~df[f]).sum())
        c = int((~df["maioria"] & df[f]).sum()); e = int((~df["maioria"] & ~df[f]).sum())
        orr, p = fisher_exact([[a, b], [c, e]])
        se = np.sqrt(sum(1 / max(x, 0.5) for x in (a, b, c, e)))
        lo, hi = np.exp(np.log(max(orr, 1e-3)) - 1.96 * se), np.exp(np.log(max(orr, 1e-3)) + 1.96 * se)
        linhas.append({"rot": rot, "or": orr, "lo": lo, "hi": hi, "p": p, "n": a})
    t = pd.DataFrame(linhas).sort_values("or")
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(t) + 2))
    for i, r in enumerate(t.itertuples()):
        cor = "#d62728" if r.p < 0.05 and r._2 > 1 else "#08519c" if r.p < 0.05 else "#999"
        ax.plot([r.lo, r.hi], [i, i], color=cor, lw=1.6)
        ax.scatter(r._2, i, color=cor, zorder=3, s=45)
        ax.text(ax.get_xlim()[1], i, "", fontsize=7)
    ax.axvline(1, color="#333", lw=1)
    ax.set_xscale("log"); ax.set_yticks(range(len(t)))
    ax.set_yticklabels([f"{r.rot}  (em {r.n} dos {int(df['maioria'].sum())})" for r in t.itertuples()], fontsize=8.5)
    ax.set_xlabel("Razão de chances de o julgamento ser não unânime (escala log, IC 95%)")
    ax.set_title("O que acompanha a divergência", loc="left"); _limpa(ax)
    fig.tight_layout(); fig.savefig(out / "fig_6_razao_de_chances.png"); plt.close(fig)


# 7 ---------------------------------------------------------------------------------------------
def fig_argumentos(df: pd.DataFrame, out: Path):
    m = df[df["caminho"] == "merito"]
    args = [a for a in ARGUMENTOS if m[a].mean() >= 0.03]
    L = pd.DataFrame(index=args, columns=args, dtype=float)
    for a, b in itertools.product(args, args):
        pa, pb = m[a].mean(), m[b].mean()
        L.loc[a, b] = np.nan if a == b else (m[a] & m[b]).mean() / (pa * pb)
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 9.6), gridspec_kw={"height_ratios": [1.18, 1]})
    im = axes[0].imshow(np.log2(L.astype(float).values), cmap="RdBu_r", vmin=-2, vmax=2)
    axes[0].set_xticks(range(len(args)))
    axes[0].set_xticklabels([rot(a) for a in args], rotation=40, ha="right", fontsize=8.4)
    axes[0].set_yticks(range(len(args)))
    axes[0].set_yticklabels([rot(a) for a in args], fontsize=8.4)
    axes[0].set_title("Argumentos que aparecem juntos", loc="left", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=axes[0], fraction=0.045,
                 label="log₂ do lift: acima de zero, juntos; abaixo, separados")
    d = m.copy(); d["semestre"] = _sem(d["data"])
    principais = ["rol_taxativo_ou_mitigado", "rol_exemplificativo", "prescricao_medica_deferencia",
                  "tea_autismo", "cdc_consumidor", "dano_moral"]
    # a composição da base mudou em 2025; a série é traçada no estrato estável (agravos internos),
    # presente em todo o período, e a base inteira aparece tracejada apenas como referência
    est = d[d["recurso"] == "Agravo interno"]
    s = est.groupby("semestre")[principais].mean().mul(100)
    s_all = d.groupby("semestre")[principais].mean().mul(100).reindex(s.index)
    for c in principais:
        linha, = axes[1].plot(range(len(s)), s[c], marker="o", ms=4, label=rot(c))
        axes[1].plot(range(len(s_all)), s_all[c], ls=":", lw=1, alpha=.5, color=linha.get_color())
    axes[1].set_xticks(range(len(s))); axes[1].set_xticklabels(s.index, rotation=45, ha="right", fontsize=8.6)
    axes[1].set_ylabel("% das ementas de mérito no semestre", fontsize=9.2)
    axes[1].tick_params(axis="y", labelsize=8.6); _limpa(axes[1])
    axes[1].legend(frameon=False, fontsize=8.6, ncol=2)
    axes[1].set_title("Evolução semestral dos argumentos", loc="left", fontsize=10, fontweight="bold")
    _marcos(axes[1], s.index)
    fig.tight_layout(); fig.savefig(out / "fig_7_argumentos.png", dpi=300); plt.close(fig)


# 8 ---------------------------------------------------------------------------------------------
def fig_deliberacao(df: pd.DataFrame, out: Path, min_n: int = 30):
    d = df.copy(); d["semestre"] = _sem(d["data"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    for org, g in d.groupby("orgao"):
        s = g.groupby("semestre")["voto_vista"].mean().mul(100)
        axes[0].plot(range(len(s)), s.values, marker="o", ms=4, color=CORES.get(org, "#888"), label=org.title())
    s0 = d.groupby("semestre")["voto_vista"].mean()
    axes[0].set_xticks(range(len(s0))); axes[0].set_xticklabels(s0.index, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("% de acórdãos com voto-vista"); axes[0].legend(frameon=False, fontsize=8); _limpa(axes[0])
    axes[0].set_title("Deliberação: voto-vista por semestre", loc="left")
    rel = [r for r, n in d["relator_c"].value_counts().items() if n >= min_n]
    t = (d[d["relator_c"].isin(rel)].groupby("relator_c")
         .agg(n=("id", "size"), vv=("voto_vista", "sum")).sort_values("vv"))
    axes[1].barh([_rotulo(r).split(" Desembargador")[0] for r in t.index], t["vv"], color="#6a51a3")
    for i, (n, vv) in enumerate(zip(t["n"], t["vv"])):
        axes[1].text(vv + 0.1, i, f"{vv} de {n}", va="center", fontsize=7.5, color="#444")
    axes[1].set_xlabel("Acórdãos com voto-vista"); axes[1].tick_params(labelsize=7.5); _limpa(axes[1])
    axes[1].set_xlim(0, max(1, t["vv"].max()) * 1.45)
    axes[1].set_title("Voto-vista por relator do acórdão", loc="left")
    fig.tight_layout(); fig.savefig(out / "fig_8_deliberacao.png"); plt.close(fig)


# 9 ---------------------------------------------------------------------------------------------
def fig_precedentes(pasta: Path, out: Path):
    arq_c, arq_p = pasta / "citacoes.csv", pasta / "precedentes_ranking.csv"
    if not (arq_c.exists() and arq_p.exists()):
        print("[9] pasta de precedentes não encontrada; pulei a figura.")
        return
    c = pd.read_csv(arq_c, parse_dates=["data_citante"])
    p = pd.read_csv(arq_p, parse_dates=["data_julgamento"])
    c = c.merge(p[["precedente", "data_julgamento", "orgao"]], on="precedente", how="left")
    c["idade"] = (c["data_citante"] - c["data_julgamento"]).dt.days / 365.25
    v = c[c["idade"].between(0, 25)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    axes[0].hist(v["idade"], bins=np.arange(0, 16, 0.5), color="#2b8cbe")
    axes[0].axvline(v["idade"].median(), color="#d62728", ls="--", lw=1.2,
                    label="mediana = " + f"{v['idade'].median():.1f}".replace(".", ",") + " anos")
    axes[0].set_xlabel("Idade do precedente no momento da citação (anos)"); axes[0].set_ylabel("Citações")
    axes[0].legend(frameon=False, fontsize=8); _limpa(axes[0])
    axes[0].set_title("Que jurisprudência o tribunal usa", loc="left")
    x = np.sort(p["citacoes"].values)[::-1]
    y = np.cumsum(x) / x.sum()
    axes[1].plot(np.arange(1, len(x) + 1) / len(x) * 100, y * 100, color="#08519c")
    n50 = int((y < 0.5).sum() + 1)
    axes[1].axhline(50, color="#999", ls=":"); axes[1].axvline(100 * n50 / len(x), color="#d62728", ls="--",
                                                               label=f"{n50} precedentes = 50% das citações")
    axes[1].set_xlabel("% dos precedentes (do mais para o menos citado)")
    axes[1].set_ylabel("% acumulado das citações"); axes[1].legend(frameon=False, fontsize=8); _limpa(axes[1])
    axes[1].set_title("Concentração das citações", loc="left")
    fig.tight_layout(); fig.savefig(out / "fig_9_precedentes.png"); plt.close(fig)


# ------------------------------------------------------------------------------------------------
def executar(base_pkl: str, saida: str, pasta_prec: str | None = None):
    out = Path(saida); out.mkdir(parents=True, exist_ok=True)
    df = classificar(pd.read_pickle(base_pkl))
    df["decidido_no_merito"] = df["caminho"].eq("merito")
    df["recurso_de_merito_originario"] = df["recurso"].isin(["Recurso especial", "Embargos de divergência"])
    fig_funil(df, out); fig_volume(df, out); fig_subtemas(df, out); fig_relatores(df, out)
    fig_divergencia(df, out); fig_razao(df, out); fig_argumentos(df, out); fig_deliberacao(df, out)
    if pasta_prec:
        fig_precedentes(Path(pasta_prec), out)
    print(f"[ok] figuras em {out}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="resultados/base_votacao.pkl")
    ap.add_argument("--saida", default="resultados_graficos")
    ap.add_argument("--precedentes", default="resultados_precedentes")
    a = ap.parse_args()
    executar(a.base, a.saida, a.precedentes)
