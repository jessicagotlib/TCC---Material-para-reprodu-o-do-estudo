# -*- coding: utf-8 -*-
"""
Etapa 2 — Por que o STJ decide como decide em saúde suplementar
================================================================
Roda DEPOIS do pipeline_stj.py, sobre resultados/base_votacao.pkl (todos os acórdãos, não só os 19
não unânimes). Usa a ementa (100% preenchida), a certidão de julgamento e a referência legislativa.

Blocos de análise
  A. Perfil da base: tipo de recurso julgado (agravo interno, REsp, EDcl...) por órgão
  B. Resultado processual: negar provimento, dar provimento, não conhecer, embargos rejeitados...
  C. Filtros de admissibilidade (óbices): Súmula 7 (reexame de provas), 5 (cláusula contratual),
     83 e 568 (jurisprudência dominante), prequestionamento (282/356 STF, 211 STJ), 284/STF
  D. Fundamentos: uso de precedentes e dicionário de argumentos de mérito (livro de códigos exportado)
  E. Evolução temporal dos argumentos em torno dos marcos normativos
  F. Estilo decisório por relator (mesmo sem divergência: quem aplica mais óbices, quem mais provê)
  G. O que distingue os julgamentos não unânimes dos unânimes (teste exato de Fisher)

Uso
  python analise_fundamentos.py --base ./resultados/base_votacao.pkl --saida ./resultados_fundamentos

Toda classificação por dicionário é heurística: valide com a amostra gerada (amostra_validacao_fundamentos.csv).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_stj import MARCOS, wilson, _rotulo

# ----------------------------------------------------------------------------------------------
# Livro de códigos (dicionários). Cada categoria = expressão regular aplicada à ementa.
# Ele é exportado para CSV para constar como apêndice metodológico do trabalho.
# ----------------------------------------------------------------------------------------------
# \b antes dos números evita falsos positivos como "Súmula 597/STJ" contando como Súmula 7.
_SUM = r"s[úu]mulas?\s*(?:n[º°.]*\s*)?(?:\d+\s*(?:,|e)\s*(?:n[º°.]*\s*)?)*"   # "Súmulas 5 e 7", "Súmulas n. 5, 7"
OBICES = {
    "sumula_7_reexame_provas": _SUM + r"7\b(?!\s*\.\d)|\b7/STJ|\b7 do STJ|reexame (?:de|do conjunto|f[áa]tico)|revolvimento",
    "sumula_5_clausula_contratual": _SUM + r"5\b(?!\s*\.\d)|\b5/STJ|\b5 do STJ|interpreta[çc][ãa]o de cl[áa]usula",
    "sumula_83_568_jurisp_dominante": _SUM + r"(?:83|568)\b|\b83/STJ|\b568/STJ|\b83 do STJ",
    "prequestionamento": r"prequestionamento|" + _SUM + r"(?:282|356|211)\b|\b282/STF|\b356/STF|\b211/STJ",
    "sumula_284_fundamentacao_deficiente": _SUM + r"284\b|\b284/STF|fundamenta[çc][ãa]o deficiente",
    "dissidio_nao_demonstrado": r"diss[íi]dio (?:jurisprudencial )?n[ãa]o (?:demonstrado|comprovado)|aus[êe]ncia de (?:similitude|cotejo)",
}
PRECEDENTES = {
    "jurisprudencia_consolidada": r"jurisprud[êe]ncia (?:desta|do STJ|deste|pac[íi]fica|consolidada|dominante|firme)|"
                                  r"entendimento (?:desta|deste|do STJ|consolidado|pacificado|firmado)|orienta[çc][ãa]o (?:desta|jurisprudencial)",
    "eresp_rol_2022": r"1\.?886\.?929|1\.?889\.?704",
    "tema_repetitivo": r"tema[s]? (?:repetitivo[s]? )?(?:n[º°.]*\s*)?\d{3,4}|recurso[s]? (?:especial |especiais )?repetitivo|rito dos repetitivos",
    "sumula_608_cdc": r"\b608/STJ|s[úu]mula[s]?\s*(?:n[º°.]*\s*)?608\b",
    "adi_7265_stf": r"ADI\s*(?:n[º°.]*\s*)?7\.?265",
}
ARGUMENTOS = {
    "rol_taxativo_ou_mitigado": r"taxativ|mitigad",
    "rol_exemplificativo": r"exemplificativ",
    "lei_14454_2022": r"14\.?454",
    "prescricao_medica_deferencia": r"prescri[çc][ãa]o m[ée]dica|prescrito pelo m[ée]dico|indica[çc][ãa]o m[ée]dica|m[ée]dico assistente",
    "abusividade": r"abusiv",
    "cdc_consumidor": r"consumidor|\bCDC\b",
    "autogestao": r"autogest",
    "dano_moral": r"dano[s]? mora",
    "urgencia_emergencia": r"urg[êe]ncia|emerg[êe]ncia",
    "tea_autismo": r"autis|\bTEA\b|espectro autista|multidisciplinar",
    "registro_anvisa_offlabel": r"anvisa|off[- ]label|experimental",
    "home_care": r"home care|domiciliar",
    "reembolso_rede": r"reembols|rede credenciada|fora da rede",
    "reajuste": r"reajuste|faixa et[áa]ria",
    "cancelamento_contrato_coletivo": r"cancelamento|rescis|resili|coletivo",
    "aposentado_demitido_art30_31": r"art(?:igo)?s?\.?\s*3[01]\b|aposentad|demitid",
}
# Indício de direção (só para decisões de mérito). Heurística conservadora: sem marcador -> indeterminado.
DIRECAO = {
    "pro_operadora": r"n[ãa]o (?:h[áa]|havendo|se configura|configura(?:da|do)?|caracteriza\w*) (?:abusividade|obrigat\w*|dever|ilicitude)|"
                     r"aus[êe]ncia de (?:obrigatoriedade|dever|abusividade|ilicitude)|recusa (?:leg[íi]tima|devida|l[íi]cita|justificada)|"
                     r"leg[íi]tima a (?:recusa|negativa)|n[ãa]o (?:[ée]|est[áa]) obrigad|cobertura n[ãa]o obrigat|"
                     r"exclus[ãa]o (?:de cobertura )?(?:v[áa]lida|leg[íi]tima|l[íi]cita)|mero (?:inadimplemento|aborrecimento)",
    "pro_beneficiario": r"recusa (?:de cobertura )?(?:indevida|abusiva|il[íi]cita|ileg[íi]tima|injustificada)|"
                        r"negativa (?:de cobertura )?(?:indevida|abusiva|il[íi]cita|ileg[íi]tima|injustificada)|"
                        r"abusiv[ao] a (?:recusa|negativa|cl[áa]usula)|(?:dever|obriga[çc][ãa]o) de (?:custear|cobrir|fornecer|arcar)|"
                        r"cobertura (?:obrigat[óo]ria|devida)|dano moral (?:configurado|caracterizado)|"
                        r"direito (?:[àa] cobertura|ao custeio|ao tratamento)",
}


def _classe_recurso(c: str) -> str:
    c = str(c)
    if c.startswith(("AgInt", "AgRg")):
        return "Agravo interno"
    if c.startswith("EDcl"):
        return "Embargos de declaração"
    if c.startswith(("EREsp", "EAREsp")):
        return "Embargos de divergência"
    if c.startswith("REsp"):
        return "Recurso especial"
    if c.startswith("AREsp"):
        return "Agravo em recurso especial"
    if c.startswith("ProAfR"):
        return "Proposta de afetação"
    return "Outros"


def _resultado(dec: str) -> str:
    t = re.sub(r"\s+", " ", str(dec or "")).lower()
    if not t.strip():
        return "sem_texto"
    regras = [
        ("nao_conhecido", r"n[ãa]o conhece[ru]?|n[ãa]o conhecido|n[ãa]o conheceu"),
        ("provimento_parcial", r"(?:dar|deu|dando)[- ]lhe parcial provimento|(?:dar|deu|dando) parcial provimento|provido em parte|parcialmente provido"),
        ("provido", r"(?:dar|deu|dando)[- ]lhe provimento|(?:dar|deu|dando) provimento|conhecer do agravo para (?:conhecer|dar)"),
        ("desprovido", r"negar(?:am)?[- ](?:lhe )?provimento|negou[- ](?:lhe )?provimento|negando provimento"),
        ("embargos_rejeitados", r"rejeitar os embargos|rejeitou os embargos|embargos de declara[çc][ãa]o rejeitados"),
        ("embargos_acolhidos", r"acolher(?:am)? (?:em parte )?os embargos|acolheu (?:em parte )?os embargos"),
        ("afetacao", r"afetar o processo"),
    ]
    achados = [nome for nome, rx in regras if re.search(rx, t)]
    return achados[0] if achados else "outro"


def classificar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    em = df["ementa"].fillna("").str.replace(r"\s+", " ", regex=True)
    txt = em + " " + df["info_compl"].fillna("")
    df["recurso"] = df["classe"].map(_classe_recurso)
    df["resultado"] = df["decisao"].map(_resultado)
    for grupo in (OBICES, PRECEDENTES, ARGUMENTOS):
        for k, rx in grupo.items():
            df[k] = txt.str.contains(rx, case=False, regex=True)
    df["algum_obice"] = df[list(OBICES)].any(axis=1)
    df["algum_precedente"] = df[list(PRECEDENTES)].any(axis=1)
    op = txt.str.contains(DIRECAO["pro_operadora"], case=False, regex=True)
    be = txt.str.contains(DIRECAO["pro_beneficiario"], case=False, regex=True)
    df["indicio_direcao"] = np.select([op & ~be, be & ~op, op & be], ["pro_operadora", "pro_beneficiario", "ambiguo"],
                                      "indeterminado")
    # Caminho decisório: óbice (não reexamina o mérito) x mérito
    df["caminho"] = np.where(df["resultado"].eq("nao_conhecido") | (df["algum_obice"] & ~df["resultado"].isin(
        ["provido", "provimento_parcial"])), "filtro_processual", "merito")
    df.loc[df["resultado"].eq("afetacao"), "caminho"] = "afetacao"
    df["periodo"] = pd.cut(df["data"], bins=[pd.Timestamp("1900-01-01")] + [pd.Timestamp(d) for d, _ in MARCOS[1:3]] +
                           [pd.Timestamp("2100-01-01")], labels=["até Lei 14.454", "Lei 14.454 → ADI 7265", "após ADI 7265"],
                           right=False)
    return df


def _pct_tab(df, linhas, colunas):
    t = pd.crosstab(df[linhas], df[colunas], normalize="index").mul(100).round(1)
    t["n"] = df[linhas].value_counts()
    return t


def _pct_flags(df, flags, por="orgao"):
    t = df.groupby(por)[flags].mean().mul(100).round(1).T
    t.loc["n acórdãos"] = df[por].value_counts()
    return t


def fisher(df, flag):
    from scipy.stats import fisher_exact
    a = int((df["maioria"] & df[flag]).sum()); b = int((df["maioria"] & ~df[flag]).sum())
    c = int((~df["maioria"] & df[flag]).sum()); d = int((~df["maioria"] & ~df[flag]).sum())
    orr, p = fisher_exact([[a, b], [c, d]])
    return {"caracteristica": flag, "nao_unanimes_com": a, "nao_unanimes_total": a + b,
            "%_nos_nao_unanimes": round(100 * a / max(1, a + b), 1), "%_nos_unanimes": round(100 * c / max(1, c + d), 1),
            "razao_de_chances": round(orr, 2) if np.isfinite(orr) else np.inf, "p_valor_fisher": round(p, 4)}


# ----------------------------------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------------------------------
def fig_barras_empilhadas(t: pd.DataFrame, titulo: str, arquivo: Path):
    import matplotlib.pyplot as plt
    d = t.drop(columns="n")
    ax = d.plot(kind="barh", stacked=True, figsize=(9, 0.6 * len(d) + 1.6), colormap="tab20c", width=0.7)
    ax.set_xlabel("% dos acórdãos"); ax.set_ylabel("")
    ax.set_title(titulo, loc="left", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(arquivo, dpi=200); plt.close()


def fig_heatmap(t: pd.DataFrame, titulo: str, arquivo: Path, fmt="{:.0f}"):
    import matplotlib.pyplot as plt
    d = t.astype(float)
    fig, ax = plt.subplots(figsize=(1.1 * d.shape[1] + 3, 0.42 * d.shape[0] + 1.8))
    im = ax.imshow(d.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(d.shape[1])); ax.set_xticklabels(d.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(d.shape[0])); ax.set_yticklabels(d.index, fontsize=8)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                        color="white" if v > np.nanmax(d.values) * 0.6 else "#222")
    ax.set_title(titulo, loc="left", fontweight="bold", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, label="%")
    plt.tight_layout(); plt.savefig(arquivo, dpi=200); plt.close()


def fig_evolucao(df: pd.DataFrame, flags: list[str], titulo: str, arquivo: Path):
    import matplotlib.pyplot as plt
    d = df.copy()
    d["semestre"] = d["data"].dt.year.astype(str) + "-S" + ((d["data"].dt.month > 6) + 1).astype(str)
    s = d.groupby("semestre")[flags].mean().mul(100)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for f in flags:
        ax.plot(s.index, s[f], marker="o", label=f.replace("_", " "))
    for data, nome in MARCOS:
        sem = f"{data[:4]}-S{1 if int(data[5:7]) <= 6 else 2}"
        if sem in s.index:
            ax.axvline(sem, color="#999", ls=":", lw=1)
            ax.text(sem, ax.get_ylim()[1] * 0.98, nome.split("(")[0], rotation=90, fontsize=7, va="top", color="#666")
    ax.set_ylabel("% das ementas do semestre"); ax.set_title(titulo, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.spines[["top", "right"]].set_visible(False); plt.xticks(rotation=45)
    plt.tight_layout(); plt.savefig(arquivo, dpi=200); plt.close()


# ----------------------------------------------------------------------------------------------
def executar(base: str, saida: str, min_relatorias: int = 30):
    out = Path(saida); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(base)
    df = df[df["alerta"] != "sem_texto_decisao"] if "alerta" in df else df
    df = classificar(df)
    df.drop(columns=["decisao", "info_compl", "termos_aux"], errors="ignore").to_csv(out / "base_classificada.csv", index=False)

    # Livro de códigos
    cod = [{"bloco": b, "categoria": k, "expressao_regular": rx}
           for b, g in [("óbice", OBICES), ("precedente", PRECEDENTES), ("argumento", ARGUMENTOS), ("direção", DIRECAO)]
           for k, rx in g.items()]
    pd.DataFrame(cod).to_csv(out / "livro_de_codigos.csv", index=False)

    # A. Perfil
    tA = _pct_tab(df, "orgao", "recurso"); tA.to_csv(out / "A_tipo_recurso_por_orgao.csv")
    fig_barras_empilhadas(tA, "Tipo de recurso julgado", out / "fig_A_tipo_recurso.png")
    # B. Resultado
    tB = _pct_tab(df, "orgao", "resultado"); tB.to_csv(out / "B_resultado_por_orgao.csv")
    fig_barras_empilhadas(tB, "Resultado processual", out / "fig_B_resultado.png")
    tB2 = _pct_tab(df, "orgao", "caminho"); tB2.to_csv(out / "B2_caminho_decisorio_por_orgao.csv")
    # C. Óbices
    tC = _pct_flags(df, list(OBICES) + ["algum_obice"]); tC.to_csv(out / "C_obices_por_orgao.csv")
    fig_heatmap(tC.drop(index="n acórdãos"), "Filtros de admissibilidade citados na ementa (% dos acórdãos)",
                out / "fig_C_obices.png")
    # D. Precedentes e argumentos
    tD1 = _pct_flags(df, list(PRECEDENTES) + ["algum_precedente"]); tD1.to_csv(out / "D1_precedentes_por_orgao.csv")
    merito = df[df["caminho"] == "merito"]
    tD2 = _pct_flags(merito, list(ARGUMENTOS)); tD2.to_csv(out / "D2_argumentos_merito_por_orgao.csv")
    fig_heatmap(tD2.drop(index="n acórdãos"), "Argumentos nas decisões de mérito (% das ementas)", out / "fig_D2_argumentos.png")
    tD3 = _pct_tab(merito, "orgao", "indicio_direcao"); tD3.to_csv(out / "D3_indicio_direcao_merito.csv")
    tD4 = _pct_flags(merito, list(ARGUMENTOS), por="subtema"); tD4.to_csv(out / "D4_argumentos_por_subtema.csv")
    # E. Evolução
    fig_evolucao(df, ["rol_taxativo_ou_mitigado", "rol_exemplificativo", "lei_14454_2022", "eresp_rol_2022",
                      "sumula_7_reexame_provas"], "Evolução dos fundamentos nas ementas", out / "fig_E_evolucao.png")
    tE = _pct_flags(df, list(PRECEDENTES) + list(ARGUMENTOS)[:6] + ["algum_obice"], por="periodo")
    tE.to_csv(out / "E_fundamentos_por_periodo.csv")
    # F. Estilo por relator (dentro de cada órgão; só relatores com volume mínimo)
    linhas = []
    for (org, rel), g in df.groupby(["orgao", "relator_c"]):
        if len(g) < min_relatorias:
            continue
        prov = g["resultado"].isin(["provido", "provimento_parcial", "embargos_acolhidos"])
        m = g[g["caminho"] == "merito"]
        lo, hi = wilson(int(g["algum_obice"].sum()), len(g))
        linhas.append({"orgao": org, "relator": _rotulo(rel), "relatorias": len(g),
                       "%_com_obice": round(100 * g["algum_obice"].mean(), 1), "ic95_obice": f"{100*lo:.0f}–{100*hi:.0f}",
                       "%_provimento": round(100 * prov.mean(), 1),
                       "%_cita_jurisp_consolidada": round(100 * g["jurisprudencia_consolidada"].mean(), 1),
                       "merito_n": len(m),
                       "%_merito_pro_beneficiario": round(100 * (m["indicio_direcao"] == "pro_beneficiario").mean(), 1) if len(m) else np.nan,
                       "%_merito_pro_operadora": round(100 * (m["indicio_direcao"] == "pro_operadora").mean(), 1) if len(m) else np.nan})
    tF = pd.DataFrame(linhas).sort_values(["orgao", "relatorias"], ascending=[True, False])
    tF.to_csv(out / "F_estilo_por_relator.csv", index=False)
    from scipy.stats import chi2_contingency
    testes = []
    for org, g in df.groupby("orgao"):
        g = g[g["relator_c"].map(g["relator_c"].value_counts()) >= min_relatorias]
        if g["relator_c"].nunique() > 1:
            chi2, p, dof, _ = chi2_contingency(pd.crosstab(g["relator_c"], g["algum_obice"]))
            testes.append({"orgao": org, "teste": "relator x uso de óbice", "qui2": round(chi2, 2), "gl": dof, "p_valor": round(p, 4)})
    pd.DataFrame(testes).to_csv(out / "F2_testes_relator.csv", index=False)
    for org, g in tF.groupby("orgao"):
        fig_heatmap(g.set_index("relator")[["%_com_obice", "%_provimento", "%_cita_jurisp_consolidada",
                                             "%_merito_pro_beneficiario", "%_merito_pro_operadora"]],
                    f"Estilo decisório por relator — {org.title()}", out / f"fig_F_relatores_{org.lower().replace(' ', '_')}.png")
    # G. Não unânimes x unânimes
    df["recurso_de_merito_originario"] = df["recurso"].isin(["Recurso especial", "Embargos de divergência"])
    df["decidido_no_merito"] = df["caminho"].eq("merito")
    flags = ["recurso_de_merito_originario", "decidido_no_merito", "voto_vista", "algum_obice", "sumula_7_reexame_provas",
             "tema_repetitivo", "rol_taxativo_ou_mitigado", "tea_autismo", "lei_14454_2022"]
    tG = pd.DataFrame([fisher(df, f) for f in flags if f in df])
    tG.to_csv(out / "G_nao_unanimes_vs_unanimes.csv", index=False)

    # amostra de validação estratificada por caminho decisório
    amostra = pd.concat([g.sample(min(len(g), 50), random_state=42) for _, g in df.groupby("caminho")])
    amostra = amostra[["id", "orgao", "classe", "processo", "data", "resultado", "caminho", "algum_obice",
                       "indicio_direcao", "ementa"]].assign(**{"validacao_caminho(OK/errado)": "",
                                                              "validacao_direcao(OK/errado)": "", "obs": ""})
    amostra.to_csv(out / "amostra_validacao_fundamentos.csv", index=False)

    print("\n=== A. tipo de recurso (%)\n", tA, "\n\n=== B. resultado (%)\n", tB, "\n\n=== B2. caminho (%)\n", tB2,
          "\n\n=== C. óbices (%)\n", tC, "\n\n=== D1. precedentes (%)\n", tD1, "\n\n=== D2. argumentos de mérito (%)\n", tD2,
          "\n\n=== D3. indício de direção no mérito (%)\n", tD3, "\n\n=== E. por período (%)\n", tE,
          "\n\n=== F. relatores\n", tF.to_string(index=False), "\n\n=== G. não unânimes x unânimes\n", tG.to_string(index=False))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="resultados/base_votacao.pkl")
    ap.add_argument("--saida", default="resultados_fundamentos")
    ap.add_argument("--min-relatorias", type=int, default=30)
    a = ap.parse_args()
    executar(a.base, a.saida, a.min_relatorias)
