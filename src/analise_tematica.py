# -*- coding: utf-8 -*-
"""
Etapa 9 — O que se pede e por que se nega: análise temática fina
=================================================================
Vai além dos assuntos da tabela do CNJ, que param em "tratamento médico-hospitalar" ou "medicamentos".
A partir do texto das ementas, identifica:

  * OBJETO DA DEMANDA — qual tratamento, procedimento, terapia ou medicamento está em disputa
  * MOTIVO DA RECUSA — o fundamento que a operadora opôs à cobertura (fora do rol, caráter experimental,
    ausência de registro na Anvisa, fora da rede credenciada, carência, limite de sessões etc.)

e cruza cada um deles com o desfecho processual (reforma ou manutenção da decisão de origem) e, quando a
amostra de direção já estiver codificada, com o lado favorecido.

Limites: a ementa nem sempre descreve o objeto, de modo que parte dos acórdãos fica sem classificação; a
cobertura é informada em todas as tabelas. Os dicionários são heurísticos e devem ser validados em amostra.

Uso
  python analise_tematica.py --base ./resultados/base_votacao.pkl --saida ./resultados_tematica
  python analise_tematica.py --base ./resultados/base_votacao.pkl --saida ./resultados_tematica \
      --direcao ./resultados_reforma/amostra_direcao_merito_c1.csv
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from analise_reforma import preparar
from pipeline_stj import wilson

# ------------------------------------------------------------------------------------------------
# Dicionários (livro de códigos da etapa 9)
# ------------------------------------------------------------------------------------------------
OBJETOS = {
    "terapias para TEA (ABA, multidisciplinar)": r"\bABA\b|applied behavior|an[áa]lise do comportamento|"
        r"terapia multidisciplinar|integra[çc][ãa]o sensorial|autis|espectro autista|denver|pediasuit|"
        r"fonoaudi[óo]log|terapia ocupacional|psicopedagog|acompanhante terap[êe]utico|equoterapia|musicoterapia",
    "home care e internação domiciliar": r"home care|internação domiciliar|assist[êe]ncia domiciliar|"
        r"atendimento domiciliar|cuidador|enfermagem domiciliar",
    "medicamento oncológico": r"(?:medicamento|quimioter|tratamento)[^.]{0,60}(?:c[âa]ncer|oncol[óo]gic|neoplas|tumor)|"
        r"oncol[óo]gic[^.]{0,40}medicament|pembrolizu|trastuzu|nivolumab|olaparib|bevacizu",
    "medicamento de alto custo (fora oncologia)": r"zolgensma|spinraza|nusinerse|risdiplam|trikafta|elaprase|"
        r"soliris|eculizumab|imunobiol[óo]gic|imunoterapia|hemofilia|atrofia muscular espinhal|"
        r"doen[çc]a rara|medicamento de alto custo",
    "canabidiol": r"canabidiol|cannabis|canab[íi]dic",
    "medicamento off-label ou sem registro": r"off[- ]label|sem registro (?:na |perante )?(?:a )?anvisa|"
        r"n[ãa]o registrado (?:na )?anvisa|importa[çc][ãa]o de medicamento|uso experimental",
    "cirurgia bariátrica e obesidade": r"bari[áa]tric|obesidade m[óo]rbida|gastroplastia|dermolipectomia|"
        r"cirurgia pl[áa]stica reparadora",
    "órtese, prótese e materiais": r"\b[óo]rtese|pr[óo]tese|OPME|\bstent\b|marca[- ]passo|implante coclear|"
        r"[óo]rtese craniana|capacete",
    "saúde mental e internação psiquiátrica": r"psiqui[áa]tric|sa[úu]de mental|depend[êe]ncia qu[íi]mica|"
        r"comunidade terap[êe]utica|transtorno mental|internação compuls[óo]ria|psicoterapia",
    "fertilização e reprodução assistida": r"fertiliza[çc][ãa]o|reprodu[çc][ãa]o assistida|criopreserva|"
        r"insemina[çc][ãa]o|congelamento de [óo]vulos",
    "exames e diagnóstico": r"pet[- ]?scan|pet[- ]ct|resson[âa]ncia|tomografia|exame gen[ée]tico|"
        r"sequenciamento|cari[óo]tipo|painel gen[ée]tico|bi[óo]psia l[íi]quida",
    "cirurgia e internação hospitalar": r"cirurgia|procedimento cir[úu]rgic|interna[çc][ãa]o hospitalar|\bUTI\b|"
        r"unidade de terapia intensiva|transplante",
    "fisioterapia e reabilitação": r"fisioterap|reabilita[çc][ãa]o|hidroterapia|RPG\b|pilates",
    "tratamento fora da rede ou no exterior": r"fora da rede|rede n[ãa]o credenciada|cl[íi]nica particular|"
        r"reembolso integral|tratamento no exterior",
    "reajuste de mensalidade": r"reajuste|faixa et[áa]ria|sinistralidade|aumento da mensalidade",
    "manutenção do contrato (demitido, aposentado, morte)": r"art(?:igo)?s?\.?\s*3[01]\b|aposentad|demitid|"
        r"ex[- ]empregad|falecimento do titular|depend[êe]nte ap[óo]s",
    "cancelamento ou rescisão unilateral": r"cancelamento unilateral|rescis[ãa]o unilateral|resili[çãa]|"
        r"den[úu]ncia do contrato|inadimpl[êe]ncia",
}

MOTIVOS = {
    "tratamento fora do rol da ANS": r"fora do rol|n[ãa]o (?:consta|previsto|inclu[íi]d[oa]) no rol|rol da ANS|"
        r"taxativ|exemplificativ|rol de procedimentos",
    "ausência de previsão contratual": r"(?:aus[êe]ncia|falta|inexist[êe]ncia) de (?:previs[ãa]o|cobertura) contratual|"
        r"exclus[ãa]o contratual|cl[áa]usula de exclus[ãa]o|n[ãa]o previsto no contrato",
    "caráter experimental": r"experimental|n[ãa]o comprovada (?:a )?efic[áa]cia|sem evid[êe]ncia cient[íi]fica|"
        r"tratamento n[ãa]o consagrado",
    "ausência de registro na Anvisa ou uso off-label": r"sem registro (?:na |perante )?(?:a )?anvisa|off[- ]label|"
        r"n[ãa]o registrado (?:na )?anvisa|importa[çc][ãa]o",
    "prestador fora da rede credenciada": r"fora da rede|rede n[ãa]o credenciada|descredencia|"
        r"livre escolha|reembolso (?:limitado|conforme o contrato)",
    "limitação do número de sessões": r"limita[çc][ãa]o (?:do )?n[úu]mero de (?:sess[õo]es|consultas)|"
        r"n[úu]mero de sess[õo]es|sess[õo]es ilimitadas",
    "diretriz de utilização não cumprida": r"diretriz de utiliza[çc][ãa]o|\bDUT\b|crit[ée]rios de elegibilidade",
    "carência contratual": r"car[êe]ncia",
    "doença ou lesão preexistente": r"preexistente|declara[çc][ãa]o de sa[úu]de|omiss[ãa]o de informa[çc][ãa]o|\bCPT\b|\bDLP\b",
    "natureza educacional e não de saúde": r"car[áa]ter educacional|[âa]mbito escolar|acompanhante escolar|"
        r"n[ãa]o (?:[ée]|possui) natureza (?:m[ée]dica|de sa[úu]de)",
    "atendimento domiciliar não coberto": r"(?:aus[êe]ncia|n[ãa]o h[áa]|inexist[êe]ncia de) cobertura[^.]{0,40}domiciliar|"
        r"home care n[ãa]o (?:coberto|previsto)",
    "inadimplência do beneficiário": r"inadimpl[êe]ncia|falta de pagamento|mora do benefici[áa]rio",
}


def marcar(df: pd.DataFrame, dicionario: dict) -> pd.DataFrame:
    txt = (df["ementa"].fillna("") + " " + df["info_compl"].fillna("")).str.replace(r"\s+", " ", regex=True)
    return pd.DataFrame({k: txt.str.contains(rx, case=False, regex=True) for k, rx in dicionario.items()},
                        index=df.index)


def tabela_tema(df: pd.DataFrame, marcas: pd.DataFrame, rotulo: str) -> pd.DataFrame:
    linhas = []
    for tema in marcas.columns:
        g = df[marcas[tema]]
        if len(g) < 10:
            continue
        k = int(g["reformou"].sum()); lo, hi = wilson(k, len(g))
        merito = g["caminho"].eq("merito")
        linhas.append({rotulo: tema, "acordaos": len(g), "% da base": round(100 * len(g) / len(df), 1),
                       "% de mérito": round(100 * merito.mean(), 1),
                       "reformas": k, "taxa_reforma_%": round(100 * k / len(g), 1),
                       "ic95_reforma": f"{100*lo:.1f}–{100*hi:.1f}",
                       "% com filtro processual": round(100 * g["algum_obice"].mean(), 1)})
    return pd.DataFrame(linhas).sort_values("acordaos", ascending=False)


def fig_temas(t: pd.DataFrame, rotulo: str, titulo: str, arquivo: Path):
    import matplotlib.pyplot as plt
    d = t.sort_values("acordaos")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 0.33 * len(d) + 1.7), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1]})
    axes[0].barh(d[rotulo], d["acordaos"], color="#2b8cbe")
    for i, v in enumerate(d["acordaos"]):
        axes[0].text(v + max(d["acordaos"]) * 0.015, i, f"{v}", va="center", fontsize=8.4)
    axes[0].set_xlabel("Acórdãos em que o tema aparece", fontsize=9)
    axes[0].set_xlim(0, max(d["acordaos"]) * 1.16)
    axes[0].set_title("Frequência", loc="left", fontweight="bold", fontsize=9.6)
    for i, (_, r) in enumerate(d.iterrows()):
        lo, hi = [float(x) for x in r["ic95_reforma"].split("–")]
        axes[1].plot([lo, hi], [i, i], color="#666", lw=1.2)
        axes[1].scatter(r["taxa_reforma_%"], i, color="#d62728", zorder=3, s=35)
    axes[1].axvline(13.4, color="#999", ls="--", lw=1, label="média da base (13,4%)")
    axes[1].set_xlabel("% que reformaram a decisão de origem (IC 95%)", fontsize=9)
    axes[1].set_title("Reforma", loc="left", fontweight="bold", fontsize=9.6)
    axes[1].legend(frameon=False, fontsize=8.6)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8.6)
    fig.tight_layout(); fig.savefig(arquivo, dpi=300); plt.close(fig)


def fig_objeto_motivo(df, obj, mot, arquivo: Path, minimo: int = 15):
    """Quais motivos de recusa acompanham cada objeto (lift: acima de 1 significa associação)."""
    import matplotlib.pyplot as plt
    objs = [c for c in obj.columns if obj[c].sum() >= minimo]
    mots = [c for c in mot.columns if mot[c].sum() >= minimo]
    M = np.full((len(objs), len(mots)), np.nan)
    for i, a in enumerate(objs):
        for j, b in enumerate(mots):
            pa, pb = obj[a].mean(), mot[b].mean()
            M[i, j] = ((obj[a] & mot[b]).mean() / (pa * pb)) if pa and pb else np.nan
    fig, ax = plt.subplots(figsize=(1.0 * len(mots) + 6, 0.45 * len(objs) + 3))
    im = ax.imshow(np.log2(M), cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(mots))); ax.set_xticklabels(mots, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(objs))); ax.set_yticklabels(objs, fontsize=7.5)
    for i in range(len(objs)):
        for j in range(len(mots)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(np.log2(M[i, j])) > 1 else "#333")
    fig.colorbar(im, ax=ax, fraction=0.03, label="log2 do lift")
    ax.set_title("Motivo de recusa associado a cada objeto da demanda", loc="left", fontweight="bold", fontsize=11)
    fig.tight_layout(); fig.savefig(arquivo, dpi=200); plt.close(fig)


def cruzar_direcao(arquivo: str | Path, df: pd.DataFrame, obj: pd.DataFrame, saida: Path):
    """Quando a amostra de direção estiver codificada, mede o lado favorecido por objeto da demanda."""
    a = pd.read_csv(arquivo).fillna("")
    col = [c for c in a.columns if c.startswith("c1_direcao")][0]
    a = a[a[col].astype(str).str.strip().ne("")]
    m = df.reset_index(drop=True)
    idx = m["id"].astype(str).isin(a["id"].astype(str))
    sub_obj, sub_dir = obj[idx.values], m.loc[idx, "id"].astype(str).map(
        a.set_index(a["id"].astype(str))[col])
    linhas = []
    for tema in obj.columns:
        g = sub_dir[sub_obj[tema].values]
        if len(g) < 5:
            continue
        k = int((g == "pro_beneficiario").sum()); lo, hi = wilson(k, len(g))
        linhas.append({"objeto": tema, "casos_codificados": len(g), "pro_beneficiario": k,
                       "%_pro_beneficiario": round(100 * k / len(g), 1), "ic95": f"{100*lo:.1f}–{100*hi:.1f}"})
    t = pd.DataFrame(linhas).sort_values("%_pro_beneficiario", ascending=False)
    t.to_csv(saida / "direcao_por_objeto.csv", index=False)
    print("\n=== Direção por objeto da demanda (amostra codificada)\n", t.to_string(index=False))
    return t


def executar(base_pkl: str, saida: str, direcao: str | None = None):
    out = Path(saida); out.mkdir(parents=True, exist_ok=True)
    d = preparar(pd.read_pickle(base_pkl)).reset_index(drop=True)
    obj, mot = marcar(d, OBJETOS), marcar(d, MOTIVOS)
    print(f"[cobertura] objeto identificado em {100 * obj.any(axis=1).mean():.1f}% dos acórdãos; "
          f"motivo de recusa em {100 * mot.any(axis=1).mean():.1f}%")
    d[["id", "orgao", "classe", "data", "subtema", "resultado", "caminho"]].join(obj).join(
        mot, rsuffix="_motivo").to_csv(out / "base_tematica.csv", index=False)

    t_obj = tabela_tema(d, obj, "objeto"); t_obj.to_csv(out / "objetos.csv", index=False)
    t_mot = tabela_tema(d, mot, "motivo"); t_mot.to_csv(out / "motivos.csv", index=False)
    pd.DataFrame([{"bloco": b, "categoria": k, "expressao_regular": rx}
                  for b, g in [("objeto", OBJETOS), ("motivo", MOTIVOS)] for k, rx in g.items()]
                 ).to_csv(out / "livro_de_codigos_tematico.csv", index=False)

    fig_temas(t_obj, "objeto", "O que se discute nos acórdãos de saúde suplementar", out / "fig_11_objetos.png")
    fig_temas(t_mot, "motivo", "O que a operadora alegou para negar a cobertura", out / "fig_12_motivos.png")
    fig_objeto_motivo(d, obj, mot, out / "fig_13_objeto_x_motivo.png")

    # evolução ao longo do tempo: a composição da base mudou em 2025, então as séries por ano são
    # calculadas também no estrato estável (agravos internos), presente em todo o período
    tempo = obj.copy(); tempo["ano"] = d["data"].dt.year.values
    tempo.groupby("ano").mean().mul(100).round(1).to_csv(out / "objetos_por_ano.csv")
    est = d["recurso"].eq("Agravo interno").values
    tempo[est].groupby("ano").mean().mul(100).round(1).to_csv(out / "objetos_por_ano_agravo_interno.csv")
    linhas = []
    for tema in obj.columns:
        for ano, g2 in d[obj[tema] & est].groupby(d["data"].dt.year):
            if len(g2) >= 15:
                linhas.append({"objeto": tema, "ano": int(ano), "acordaos": len(g2),
                               "taxa_reforma_%": round(100 * g2["reformou"].mean(), 1)})
    pd.DataFrame(linhas).to_csv(out / "reforma_por_objeto_ano_agravo_interno.csv", index=False)

    print("\n=== Objeto da demanda\n", t_obj.to_string(index=False))
    print("\n=== Motivo da recusa\n", t_mot.to_string(index=False))
    if direcao:
        cruzar_direcao(direcao, d, obj, out)
    return d, obj, mot


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="resultados/base_votacao.pkl")
    ap.add_argument("--saida", default="resultados_tematica")
    ap.add_argument("--direcao", default=None)
    a = ap.parse_args()
    executar(a.base, a.saida, a.direcao)
