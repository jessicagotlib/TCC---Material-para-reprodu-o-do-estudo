# -*- coding: utf-8 -*-
"""
Etapa 8b — Codificação manual da direção do resultado (somente acórdãos de mérito)
==================================================================================
Gera:
  * amostra_direcao_merito.csv  — amostra aleatória estratificada por órgão e subtema
  * codificador.html            — página local para ler cada acórdão e codificar por teclado,
                                  com salvamento automático no navegador e exportação em CSV

A página é um arquivo local: basta abrir com duplo clique. Ela não envia nada para a internet.
Depois de codificar, exporte o CSV e rode:

  python analise_reforma.py --base ./resultados/base_votacao.pkl --saida ./resultados_reforma \
      --amostra-codificada ./resultados_reforma/amostra_direcao_merito_c1.csv

Uso
  python codificacao_direcao.py --base ./resultados/base_votacao.pkl --saida ./resultados_reforma --n 150
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from analise_reforma import preparar

COLS = ["id", "estrato", "orgao", "classe", "processo", "registro", "data", "subtema", "resultado",
        "ementa", "decisao"]

# Embargos de declaração e propostas de afetação ficam de fora: quase nunca decidem a controvérsia.
# Os demais são amostrados em dois estratos, porque a chance de reforma difere muito entre eles.
ESTRATOS = {
    "recurso especial": ["Recurso especial", "Agravo em recurso especial", "Embargos de divergência"],
    "agravo interno": ["Agravo interno"],
}


def gerar_amostra(base_pkl: str, n: int, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Amostra estratificada em dois grupos, com alocação proporcional dentro de cada um por órgão e
    subtema. Retorna a amostra e o tamanho de cada estrato no universo, usado depois na ponderação."""
    d = preparar(pd.read_pickle(base_pkl))
    m = d[d["caminho"] == "merito"].copy()
    m["estrato"] = ""
    for nome, recursos in ESTRATOS.items():
        m.loc[m["recurso"].isin(recursos), "estrato"] = nome
    m = m[m["estrato"] != ""]
    universo = m["estrato"].value_counts().rename("acordaos_no_universo").to_frame().reset_index(names="estrato")
    print("[universo] acórdãos de mérito por estrato:")
    print(universo.to_string(index=False))

    # metade da amostra em cada estrato: o de agravos é maior no universo, mas o de recursos especiais
    # concentra as decisões em que a corte efetivamente reexamina a controvérsia
    partes = []
    for nome in ESTRATOS:
        g0 = m[m["estrato"] == nome]
        frac = (n / 2) / len(g0)
        sub = [g.sample(max(1, int(round(len(g) * frac))), random_state=seed)
               for _, g in g0.groupby(["orgao", "subtema"]) if len(g) >= 1]
        partes.append(pd.concat(sub).sample(frac=1, random_state=seed).head(n // 2))
    a = pd.concat(partes).sample(frac=1, random_state=seed).copy()
    a["data"] = a["data"].dt.strftime("%d/%m/%Y")
    # número de registro no formato da consulta processual do STJ: 202302694079 -> 2023/0269407-9
    a["registro"] = a["registro"].astype(str).str.replace(r"\D", "", regex=True).map(
        lambda r: f"{r[:4]}/{r[4:11]}-{r[11:]}" if len(r) == 12 else r)
    for c in ["ementa", "decisao"]:
        a[c] = a[c].fillna("").map(lambda t: re.sub(r"\s+", " ", str(t)).strip())
    return a[COLS], universo


HTML = """<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Codificação da direção do resultado</title>
<style>
 :root { --fg:#1a1a1a; --mut:#666; --bd:#dcdcdc; --bg:#fff; --azul:#08519c; --verm:#b02418; }
 * { box-sizing:border-box; }
 body { font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif; color:var(--fg); background:#f5f5f5;
        margin:0; padding:16px; line-height:1.5; }
 .wrap { max-width:1000px; margin:0 auto; }
 .card { background:var(--bg); border:1px solid var(--bd); border-radius:8px; padding:18px 22px; margin-bottom:14px; }
 h1 { font-size:18px; margin:0 0 4px; } .sub { color:var(--mut); font-size:13px; margin-bottom:10px; }
 .barra { height:8px; background:#e6e6e6; border-radius:4px; overflow:hidden; margin:10px 0 4px; }
 .barra div { height:100%; background:var(--azul); width:0; transition:width .2s; }
 .meta { font-size:12.5px; color:var(--mut); margin-bottom:10px; }
 .ementa { font-size:14px; max-height:340px; overflow:auto; border-left:3px solid var(--bd); padding-left:12px; }
 .cert { font-size:13px; color:#333; margin-top:12px; border-left:3px solid #f0c000; padding-left:12px;
         max-height:150px; overflow:auto; }
 fieldset { border:1px solid var(--bd); border-radius:6px; margin:12px 0 0; padding:10px 14px; }
 legend { font-size:12.5px; color:var(--mut); padding:0 6px; }
 button { font:inherit; padding:7px 12px; margin:3px 5px 3px 0; border:1px solid var(--bd); background:#fafafa;
          border-radius:6px; cursor:pointer; }
 button:hover { background:#eef3fa; } button.on { background:var(--azul); color:#fff; border-color:var(--azul); }
 .acoes { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-top:14px; }
 .exp { background:var(--azul); color:#fff; border-color:var(--azul); }
 .aviso { color:var(--verm); font-size:12.5px; }
 kbd { background:#eee; border:1px solid #ccc; border-radius:4px; padding:0 4px; font-size:11.5px; }
 .regras { font-size:13px; color:#333; }
 .regras li { margin-bottom:5px; }
</style></head><body><div class="wrap">

<div class="card">
  <h1>Codificação da direção do resultado — acórdãos de mérito</h1>
  <div class="sub">Tudo fica salvo no seu navegador. Ao terminar, clique em exportar e rode o script de estimativa.</div>
  <div class="barra"><div id="prog"></div></div>
  <div class="meta" id="contagem"></div>
</div>

<div class="card">
  <div class="meta" id="cabecalho"></div>
  <div class="meta"><a id="link" href="#" target="_blank" rel="noopener">abrir a consulta processual do STJ (busca pelo número de registro)</a></div>
  <div class="ementa" id="ementa"></div>
  <div class="cert" id="certidao"></div>

  <fieldset><legend>Quem recorreu ao STJ (<kbd>1</kbd>–<kbd>4</kbd>)</legend><div id="rec"></div></fieldset>
  <fieldset><legend>A quem o resultado favoreceu (<kbd>q</kbd> <kbd>w</kbd> <kbd>e</kbd> <kbd>r</kbd>)</legend><div id="dir"></div></fieldset>
  <fieldset><legend>Observação (opcional)</legend>
    <input id="obs" style="width:100%;padding:7px;border:1px solid var(--bd);border-radius:6px" placeholder="dúvidas, casos limítrofes"></fieldset>

  <div class="acoes">
    <button onclick="ir(-1)">◀ Anterior (<kbd>←</kbd>)</button>
    <button onclick="ir(1)">Próximo ▶ (<kbd>→</kbd>)</button>
    <button class="exp" onclick="exportar()">Exportar CSV</button>
    <span class="aviso" id="pendente"></span>
  </div>
</div>

<div class="card regras">
  <strong>Regras de decisão</strong>
  <ul>
    <li><strong>Quem recorreu:</strong> identifique pela ementa quem interpôs o recurso julgado. Se o acórdão
        decidiu recursos das duas partes, marque ambos. Se não der para saber, marque não identificado —
        esse caso não entra na estimativa da direção.</li>
    <li><strong>Direção:</strong> combine quem recorreu com o resultado. Recurso da operadora provido, ou
        recurso do beneficiário desprovido, favorece a operadora. Recurso do beneficiário provido, ou recurso
        da operadora desprovido, favorece o beneficiário.</li>
    <li><strong>Misto:</strong> provimento parcial que acolhe parte do pedido de cada lado, ou acórdão que
        mantém a cobertura e afasta o dano moral.</li>
    <li><strong>Sem mérito:</strong> o acórdão não decidiu a controvérsia de saúde suplementar (competência,
        honorários, nulidade processual). Marque e siga adiante.</li>
    <li>Codifique pelo que está escrito no acórdão, sem buscar o processo fora da base.</li>
  </ul>
</div>
</div>

<script>
const DADOS = __DADOS__;
const CHAVE = "codificacao_direcao_v1";
const RECS = [["beneficiario","Beneficiário"],["operadora","Operadora"],["ambos","Ambos"],["nao_identificado","Não identificado"]];
const DIRS = [["pro_beneficiario","Pró-beneficiário"],["pro_operadora","Pró-operadora"],["misto","Misto"],["sem_merito","Sem mérito"]];
let i = 0, estado = JSON.parse(localStorage.getItem(CHAVE) || "{}");

function salvar() { localStorage.setItem(CHAVE, JSON.stringify(estado)); }
function atual() { const id = DADOS[i].id; return estado[id] || (estado[id] = {rec:"", dir:"", obs:""}); }

function botoes(div, lista, campo) {
  const e = atual();
  document.getElementById(div).innerHTML = lista.map(([v,t],k) =>
    `<button class="${e[campo]===v?'on':''}" onclick="marcar('${campo}','${v}')">${k+1}. ${t}</button>`).join("");
}

function marcar(campo, valor) {
  const e = atual(); e[campo] = (e[campo] === valor ? "" : valor); salvar(); render();
  if (campo === "dir" && e.dir) setTimeout(() => ir(1), 180);
}

function render() {
  const d = DADOS[i], e = atual();
  document.getElementById("cabecalho").textContent =
    `${i+1} de ${DADOS.length} · ${d.classe} ${d.processo} · registro ${d.registro} · ${d.orgao} · ${d.data} · estrato: ${d.estrato} · subtema: ${d.subtema} · resultado: ${d.resultado}`;
  document.getElementById("ementa").textContent = d.ementa;
  document.getElementById("link").href =
    "https://processo.stj.jus.br/processo/pesquisa/?termo=" + encodeURIComponent(d.registro);
  document.getElementById("certidao").textContent = d.decisao;
  botoes("rec", RECS, "rec"); botoes("dir", DIRS, "dir");
  document.getElementById("obs").value = e.obs || "";
  const feitos = DADOS.filter(x => (estado[x.id]||{}).dir).length;
  document.getElementById("prog").style.width = (100*feitos/DADOS.length) + "%";
  document.getElementById("contagem").textContent = `${feitos} de ${DADOS.length} codificados`;
  document.getElementById("pendente").textContent = feitos < DADOS.length ? "" : "Tudo codificado. Pode exportar.";
  window.scrollTo({top:0});
}

function ir(p) { i = Math.min(DADOS.length-1, Math.max(0, i+p)); render(); }

document.getElementById("obs").addEventListener("input", ev => { atual().obs = ev.target.value; salvar(); });

document.addEventListener("keydown", ev => {
  if (ev.target.tagName === "INPUT") return;
  const k = ev.key.toLowerCase();
  if (k === "arrowright") ir(1); else if (k === "arrowleft") ir(-1);
  else if ("1234".includes(k)) marcar("rec", RECS[+k-1][0]);
  else if ("qwer".includes(k)) marcar("dir", DIRS["qwer".indexOf(k)][0]);
});

function exportar() {
  const cab = ["id","estrato","orgao","classe","processo","registro","data","subtema","resultado",
               "c1_recorrente(beneficiario/operadora/ambos/nao_identificado)",
               "c1_direcao(pro_beneficiario/pro_operadora/misto/sem_merito)","observacao"];
  const esc = t => '"' + String(t==null?"":t).replace(/"/g,'""') + '"';
  const linhas = DADOS.map(d => { const e = estado[d.id] || {};
    return [d.id,d.estrato,d.orgao,d.classe,d.processo,d.registro,d.data,d.subtema,d.resultado,e.rec||"",e.dir||"",e.obs||""].map(esc).join(","); });
  const blob = new Blob(["\\ufeff" + [cab.map(esc).join(","), ...linhas].join("\\n")], {type:"text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "amostra_direcao_merito_c1.csv"; a.click();
}

render();
</script></body></html>
"""


def executar(base_pkl: str, saida: str, n: int):
    out = Path(saida); out.mkdir(parents=True, exist_ok=True)
    a, universo = gerar_amostra(base_pkl, n)
    a.to_csv(out / "amostra_direcao_merito.csv", index=False)
    universo.to_csv(out / "estratos_universo.csv", index=False)
    (out / "codificador.html").write_text(
        HTML.replace("__DADOS__", json.dumps(a.to_dict("records"), ensure_ascii=False)), encoding="utf-8")
    print(f"[amostra] {len(a)} acórdãos de mérito em amostra_direcao_merito.csv")
    print(f"[codificador] abra {out / 'codificador.html'} no navegador")
    print(a.groupby(["estrato", "orgao"]).size().to_string())
    return a


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="resultados/base_votacao.pkl")
    ap.add_argument("--saida", default="resultados_reforma")
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    executar(a.base, a.saida, a.n)
