# -*- coding: utf-8 -*-
"""
Etapa 8 — Quando e por que o STJ reforma a decisão de segunda instância
========================================================================
Duas perguntas distintas, com exigências de dados distintas:

A. REFORMA x MANUTENÇÃO — mensurável de forma automática e completa.
   O resultado processual diz se o acórdão de origem foi mantido (recurso desprovido ou não conhecido)
   ou reformado (provimento total ou parcial). Modela-se a chance de reforma por regressão logística.

B. DIREÇÃO (pró-beneficiário x pró-operadora) — exige saber QUEM recorreu, informação que não consta
   dos espelhos. O script gera uma amostra aleatória estratificada para codificação manual e, depois de
   preenchida, calcula as estimativas com intervalo de confiança. O dicionário textual (indicio_direcao)
   é usado apenas como comparação, nunca como resultado principal.

Uso
  python analise_reforma.py --base ./resultados/base_votacao.pkl --saida ./resultados_reforma
  python analise_reforma.py --base ./resultados/base_votacao.pkl --saida ./resultados_reforma \
      --amostra-codificada ./resultados_reforma/amostra_direcao_preenchida.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analise_fundamentos import classificar
from pipeline_stj import wilson, _rotulo

N_AMOSTRA = 200        # 200 casos -> margem de erro aprox. +-7 pontos percentuais


# ------------------------------------------------------------------------------------------------
# A. Reforma x manutenção
# ------------------------------------------------------------------------------------------------
def preparar(df: pd.DataFrame) -> pd.DataFrame:
    d = classificar(df)
    d = d[d["resultado"] != "sem_texto"].copy()
    # afetação e embargos de declaração não decidem sobre manter ou reformar o acórdão de origem
    d = d[~d["resultado"].isin(["afetacao"]) & ~d["recurso"].eq("Proposta de afetação")]
    d["reformou"] = d["resultado"].isin(["provido", "provimento_parcial"])
    d["mantido_por_obice"] = d["caminho"].eq("filtro_processual")
    return d


def taxas(d: pd.DataFrame, por: str, minimo: int = 30) -> pd.DataFrame:
    linhas = []
    for chave, g in d.groupby(por):
        if len(g) < minimo:
            continue
        k = int(g["reformou"].sum())
        lo, hi = wilson(k, len(g))
        linhas.append({por: chave, "acordaos": len(g), "reformas": k,
                       "taxa_reforma_%": round(100 * k / len(g), 1),
                       "ic95": f"{100 * lo:.1f}–{100 * hi:.1f}"})
    return pd.DataFrame(linhas).sort_values("taxa_reforma_%", ascending=False)


def padronizar(d: pd.DataFrame, ate: int = 2024) -> pd.DataFrame:
    """A composição da base por tipo de recurso mudou em 2025 (os espelhos passaram a incluir muitos
    recursos especiais e agravos em recurso especial). Comparar anos exige padronização direta: aplica-se
    a todos os anos a composição observada até `ate`."""
    d = d.copy(); d["ano"] = d["data"].dt.year
    peso = d[d["ano"] <= ate]["recurso"].value_counts(normalize=True)
    linhas = []
    for ano, g in d.groupby("ano"):
        tx = g.groupby("recurso")["reformou"].mean()
        num = sum(peso[r] * tx[r] for r in peso.index if r in tx.index and pd.notna(tx[r]))
        den = sum(peso[r] for r in peso.index if r in tx.index and pd.notna(tx[r]))
        linhas.append({"ano": ano, "acordaos": len(g),
                       "%_recurso_especial": round(100 * g["recurso"].isin(
                           ["Recurso especial", "Agravo em recurso especial"]).mean(), 1),
                       "reforma_observada_%": round(100 * g["reformou"].mean(), 1),
                       "reforma_padronizada_%": round(100 * num / den, 1) if den else np.nan})
    t = pd.DataFrame(linhas)
    print(f"\n=== Taxa de reforma observada x padronizada pela composição até {ate}\n", t.to_string(index=False))
    return t


def fig_composicao(d: pd.DataFrame, t: pd.DataFrame, arquivo: Path):
    import matplotlib.pyplot as plt
    d = d.copy(); d["ano"] = d["data"].dt.year
    comp = pd.crosstab(d["ano"], d["recurso"], normalize="index").mul(100)
    PALETA = {"Recurso especial": "#1f6fb2", "Agravo em recurso especial": "#8fc0e0",
              "Agravo interno": "#d9d9d9", "Embargos de declaração": "#f0a860",
              "Embargos de divergência": "#7a5195", "Outros": "#b4b4b4"}
    comp = comp[[c for c in PALETA if c in comp.columns] + [c for c in comp.columns if c not in PALETA]]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    comp.plot(kind="bar", stacked=True, ax=axes[0], width=0.72, legend=True,
              color=[PALETA.get(c, "#999") for c in comp.columns])
    axes[0].set_ylabel("% dos acórdãos do ano", fontsize=9); axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].set_title("Composição da base por tipo de recurso", loc="left", fontweight="bold", fontsize=9.6)
    axes[0].legend(frameon=False, fontsize=8, bbox_to_anchor=(0, -0.13), loc="upper left", ncol=2)
    axes[1].plot(t["ano"], t["reforma_observada_%"], marker="o", label="taxa observada", color="#d62728")
    axes[1].plot(t["ano"], t["reforma_padronizada_%"], marker="s", ls="--",
                 label="taxa padronizada pela composição até 2024", color="#08519c")
    axes[1].set_ylabel("% de acórdãos que reformaram", fontsize=9); axes[1].set_xticks(t["ano"])
    axes[1].set_title("Reforma: o efeito da mudança na composição", loc="left", fontweight="bold", fontsize=9.6)
    axes[1].legend(frameon=False, fontsize=8.4)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8.6)
    fig.tight_layout(); fig.savefig(arquivo, dpi=300, bbox_inches="tight"); plt.close(fig)


def modelo_logistico(d: pd.DataFrame, saida: Path) -> pd.DataFrame:
    """Chance de reforma explicada por características observáveis do julgamento."""
    import statsmodels.api as sm
    X = pd.DataFrame(index=d.index)
    X["obice_admissibilidade"] = d["algum_obice"].astype(int)
    X["sumula_7"] = d["sumula_7_reexame_provas"].astype(int)
    X["jurisp_consolidada"] = d["jurisprudencia_consolidada"].astype(int)
    X["tema_repetitivo"] = d["tema_repetitivo"].astype(int)
    X["voto_vista"] = d["voto_vista"].astype(int)
    X["recurso_especial"] = d["recurso"].isin(["Recurso especial", "Embargos de divergência"]).astype(int)
    for org in ["QUARTA TURMA", "SEGUNDA SEÇÃO"]:                       # referência: Terceira Turma
        X["orgao_" + org.split()[0].lower()] = d["orgao"].eq(org).astype(int)
    principais = d["subtema"].value_counts().index[:5]
    for s in principais[1:]:                                            # referência: subtema mais frequente
        X["subtema_" + s] = d["subtema"].eq(s).astype(int)
    for per in ["Lei 14.454 → ADI 7265", "após ADI 7265"]:              # referência: até a Lei 14.454
        X["periodo_" + per.split()[-1]] = d["periodo"].astype(str).eq(per).astype(int)

    y = d["reformou"].astype(int)
    mod = sm.Logit(y, sm.add_constant(X)).fit(disp=False)
    r = pd.DataFrame({"coef": mod.params, "erro_padrao": mod.bse, "p_valor": mod.pvalues,
                      "razao_de_chances": np.exp(mod.params),
                      "ic95_inf": np.exp(mod.conf_int()[0]), "ic95_sup": np.exp(mod.conf_int()[1])})
    r = r.drop(index="const").sort_values("razao_de_chances", ascending=False).round(3)
    r.to_csv(saida / "modelo_reforma.csv")
    with open(saida / "modelo_reforma_resumo.txt", "w", encoding="utf-8") as f:
        f.write(str(mod.summary()))
    print(f"\n=== Modelo logístico da chance de reforma (n = {len(d)}, pseudo-R2 = {mod.prsquared:.3f})\n",
          r.to_string())
    return r


def fig_reforma(d: pd.DataFrame, r: pd.DataFrame, saida: Path):
    import matplotlib.pyplot as plt
    from graficos_exploratorios import rot
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"width_ratios": [1, 1.2]})
    t = taxas(d, "subtema").sort_values("taxa_reforma_%")
    axes[0].barh([rot(x) for x in t["subtema"]], t["taxa_reforma_%"], color="#2b8cbe")
    for i, (_, x) in enumerate(t.iterrows()):
        axes[0].text(x["taxa_reforma_%"] + 0.3, i, f"{x['taxa_reforma_%']}% (n={x['acordaos']})".replace(".", ","), va="center", fontsize=8.2)
    axes[0].set_xlabel("% de acórdãos que reformaram a decisão de origem", fontsize=9)
    axes[0].tick_params(labelsize=8.4)
    axes[0].set_title("Reforma por subtema", loc="left", fontweight="bold", fontsize=9.6)
    axes[0].set_xlim(0, t["taxa_reforma_%"].max() * 1.45)
    axes[0].spines[["top", "right"]].set_visible(False)

    rr = r.sort_values("razao_de_chances")
    for i, (nome, x) in enumerate(rr.iterrows()):
        cor = "#d62728" if x["p_valor"] < 0.05 and x["razao_de_chances"] > 1 else "#08519c" if x["p_valor"] < 0.05 else "#999"
        axes[1].plot([x["ic95_inf"], x["ic95_sup"]], [i, i], color=cor, lw=1.6)
        axes[1].scatter(x["razao_de_chances"], i, color=cor, s=40, zorder=3)
    axes[1].axvline(1, color="#333", lw=1); axes[1].set_xscale("log")
    axes[1].set_yticks(range(len(rr))); axes[1].set_yticklabels([rot(n) for n in rr.index], fontsize=8.4)
    axes[1].set_xlabel("Razão de chances de reforma (escala log, IC 95%)", fontsize=9)
    axes[1].tick_params(axis="x", labelsize=8.4)
    axes[1].set_title("O que aumenta e o que reduz a chance de reforma", loc="left", fontweight="bold", fontsize=9.6)
    axes[1].spines[["top", "right"]].set_visible(False)
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([], [], color=c, marker="o", ls="-", ms=5, label=t) for c, t in
                            [("#d62728", "aumenta a chance (p < 0,05)"),
                             ("#08519c", "reduz a chance (p < 0,05)"),
                             ("#999999", "sem significância estatística")]],
                   frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(saida / "fig_10_reforma.png", dpi=300); plt.close(fig)


# ------------------------------------------------------------------------------------------------
# B. Direção do resultado (amostra manual)
# ------------------------------------------------------------------------------------------------
def gerar_amostra(d: pd.DataFrame, saida: Path, n: int = N_AMOSTRA, seed: int = 42):
    """Amostra aleatória estratificada por órgão e caminho decisório, para leitura da ementa e da
    certidão. Duas pessoas devem codificar de forma independente; a terceira desempata."""
    base = d[d["orgao"].notna()]
    frac = n / len(base)
    partes = [g.sample(max(5, int(round(len(g) * frac))), random_state=seed)
              for _, g in base.groupby(["orgao", "caminho"])]
    a = pd.concat(partes).sample(frac=1, random_state=seed)
    cols = ["id", "orgao", "classe", "processo", "data", "subtema", "resultado", "caminho", "ementa", "decisao"]
    a = a[cols].copy()
    for c in ["c1_recorrente(beneficiario/operadora/ambos/nao_identificado)",
              "c1_direcao(pro_beneficiario/pro_operadora/misto/sem_merito)",
              "c2_recorrente(beneficiario/operadora/ambos/nao_identificado)",
              "c2_direcao(pro_beneficiario/pro_operadora/misto/sem_merito)", "observacao"]:
        a[c] = ""
    a.to_csv(saida / "amostra_direcao.csv", index=False)
    print(f"[amostra] {len(a)} acórdãos em amostra_direcao.csv para dupla codificação")
    return a


def estimar_direcao(arquivo: str | Path, saida: Path):
    """Lê a amostra codificada, estima a direção com IC de Wilson e, quando houver estratos, calcula
    também a estimativa ponderada pelo tamanho de cada estrato no universo (pós-estratificação)."""
    from sklearn.metrics import cohen_kappa_score
    a = pd.read_csv(arquivo).fillna("")
    c1 = [c for c in a.columns if c.startswith("c1_direcao")][0]
    r1 = [c for c in a.columns if c.startswith("c1_recorrente")][0]
    c2 = [c for c in a.columns if c.startswith("c2_direcao")]
    val = a[a[c1].astype(str).str.strip().ne("")]
    print(f"[amostra] {len(val)} de {len(a)} acórdãos codificados")
    if c2 and val[c2[0]].astype(str).str.strip().ne("").sum() > 0:
        dupla = val[val[c2[0]].astype(str).str.strip().ne("")]
        print(f"[validação] kappa de Cohen: {cohen_kappa_score(dupla[c1], dupla[c2[0]]):.2f} "
              f"(n = {len(dupla)}); concordância simples: {(dupla[c1] == dupla[c2[0]]).mean():.1%}")

    linhas = []
    for nome, col in [("direção do resultado", c1), ("quem recorreu", r1)]:
        serie = val[col]
        for cat, k in serie.value_counts().items():
            lo, hi = wilson(int(k), len(serie))
            linhas.append({"recorte": "amostra (sem ponderação)", "variavel": nome, "categoria": cat,
                           "n": int(k), "%": round(100 * k / len(serie), 1),
                           "ic95": f"{100*lo:.1f}–{100*hi:.1f}"})
    # por estrato
    if "estrato" in val.columns:
        for est, g in val.groupby("estrato"):
            for cat, k in g[c1].value_counts().items():
                lo, hi = wilson(int(k), len(g))
                linhas.append({"recorte": f"estrato: {est}", "variavel": "direção do resultado", "categoria": cat,
                               "n": int(k), "%": round(100 * k / len(g), 1), "ic95": f"{100*lo:.1f}–{100*hi:.1f}"})
        arq_u = Path(saida) / "estratos_universo.csv"
        if arq_u.exists():
            u = pd.read_csv(arq_u).set_index("estrato")["acordaos_no_universo"]
            peso = u / u.sum()
            for cat in val[c1].unique():
                p, var = 0.0, 0.0
                for est, g in val.groupby("estrato"):
                    if est not in peso.index or len(g) == 0:
                        continue
                    ph = (g[c1] == cat).mean()
                    p += peso[est] * ph
                    var += (peso[est] ** 2) * ph * (1 - ph) / len(g)
                se = np.sqrt(var)
                linhas.append({"recorte": "universo de mérito (ponderado)", "variavel": "direção do resultado",
                               "categoria": cat, "n": "-", "%": round(100 * p, 1),
                               "ic95": f"{100*max(0, p - 1.96*se):.1f}–{100*min(1, p + 1.96*se):.1f}"})
    t = pd.DataFrame(linhas)
    t.to_csv(Path(saida) / "direcao_estimada.csv", index=False)
    print("\n=== Direção estimada\n", t.to_string(index=False))
    return t


# ------------------------------------------------------------------------------------------------
def executar(base_pkl: str, saida: str, amostra_codificada: str | None = None):
    out = Path(saida); out.mkdir(parents=True, exist_ok=True)
    d = preparar(pd.read_pickle(base_pkl))
    k = int(d["reformou"].sum()); lo, hi = wilson(k, len(d))
    print(f"[reforma] {k} de {len(d)} acórdãos reformaram a decisão de origem "
          f"({100*k/len(d):.1f}%; IC 95% {100*lo:.1f}–{100*hi:.1f})")
    for por in ["orgao", "subtema", "recurso"]:
        t = taxas(d, por); t.to_csv(out / f"reforma_por_{por}.csv", index=False)
        print(f"\n=== Taxa de reforma por {por}\n", t.to_string(index=False))
    rel = taxas(d[d["relator_c"].notna()].assign(relator=lambda x: x["relator_c"].map(_rotulo)), "relator")
    rel.to_csv(out / "reforma_por_relator.csv", index=False)
    print("\n=== Taxa de reforma por relator\n", rel.to_string(index=False))
    t_pad = padronizar(d); t_pad.to_csv(out / "reforma_observada_x_padronizada.csv", index=False)
    fig_composicao(d, t_pad, out / "fig_14_composicao.png")
    r = modelo_logistico(d, out)
    fig_reforma(d, r, out)
    if amostra_codificada:
        estimar_direcao(amostra_codificada, out)
    else:
        gerar_amostra(d, out)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="resultados/base_votacao.pkl")
    ap.add_argument("--saida", default="resultados_reforma")
    ap.add_argument("--amostra-codificada", default=None)
    a = ap.parse_args()
    executar(a.base, a.saida, a.amostra_codificada)
