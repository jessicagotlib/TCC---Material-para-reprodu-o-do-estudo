# Como o STJ decide em saúde suplementar: filtros processuais, consenso e precedentes

Material de reprodução do trabalho de conclusão de curso do MBA em Data Science e Analytics (USP/Esalq, 2026).
O estudo analisa como mecanismos processuais, características das controvérsias e o uso de precedentes
condicionam o processo decisório da Terceira Turma, da Quarta Turma e da Segunda Seção do Superior Tribunal
de Justiça em saúde suplementar.

Base: 5.170 espelhos de acórdãos julgados entre 8/2/2022 e 24/8/2026, extraídos em setembro de 2026.

Todos os dados usados são públicos. Nenhum dado pessoal de parte ou de beneficiário é coletado ou publicado:
os acórdãos são identificados por classe, número e registro do processo, e os únicos nomes tratados são os de
ministros, no exercício de função pública.

## Fontes

| Fonte | Uso |
| --- | --- |
| [Espelhos de acórdãos — Terceira Turma](https://dadosabertos.web.stj.jus.br/dataset/espelhos-de-acordaos-terceira-turma) | Ementa, certidão de julgamento, relator, classe, data |
| [Espelhos de acórdãos — Quarta Turma](https://dadosabertos.web.stj.jus.br/dataset/espelhos-de-acordaos-quarta-turma) | idem |
| [Espelhos de acórdãos — Segunda Seção](https://dadosabertos.web.stj.jus.br/dataset/espelhos-de-acordaos-segunda-secao) | idem |
| [Íntegras de decisões terminativas e acórdãos](https://dadosabertos.web.stj.jus.br/dataset/integras-de-decisoes-terminativas-e-acordaos-do-diario-da-justica) (apenas `metadados*.json`) | Códigos de assunto da TPU/CNJ, cruzados pelo `numeroRegistro` |
| [TPU/CNJ — consulta pública de assuntos](https://www.cnj.jus.br/sgt/consulta_publica_assuntos.php) | Definição do recorte de saúde suplementar |
| [Painel de estatísticas processuais de direito à saúde (CNJ)](https://justica-em-numeros.cnj.jus.br/painel-saude/) | Contexto citado na introdução |

Recorte temático: assuntos 12482, 12486, 12487, 12488, 12489, 12490 e 14760 (ramo Suplementar da árvore
Direito da Saúde), com critério textual de contingência para processos ausentes dos metadados.

## Como reproduzir

```bash
pip install -r requirements.txt

# 1. Coleta, recorte temático, extração da votação e métricas de votação
python src/pipeline_stj.py --baixar --baixar-assuntos --dados ./dados --saida ./resultados --inicio 2022-02-01

# 2. Fundamentos: filtros de admissibilidade, precedentes citados, argumentos, estilo por relator
python src/analise_fundamentos.py --base ./resultados/base_votacao.pkl --saida ./resultados_fundamentos

# 3. Precedentes: citações, índice h e grupos de teses (rede de cocitação)
python src/analise_precedentes.py --base ./resultados/base_votacao.pkl --espelhos ./dados/espelhos --saida ./resultados_precedentes

# 4. Reforma x manutenção, modelo logístico e padronização por composição
python src/analise_reforma.py --base ./resultados/base_votacao.pkl --saida ./resultados_reforma

# 5. Amostra estratificada de mérito e ferramenta local de codificação da direção
python src/codificacao_direcao.py --base ./resultados/base_votacao.pkl --saida ./resultados_reforma --n 150

# 6. Estimativas da direção a partir da amostra codificada
python src/analise_reforma.py --base ./resultados/base_votacao.pkl --saida ./resultados_reforma \
    --amostra-codificada ./resultados/tabelas/17_amostra_direcao_codificada.csv

# 7. Análise temática (objeto da demanda e motivo da recusa)
python src/analise_tematica.py --base ./resultados/base_votacao.pkl --saida ./resultados_tematica

# 8. Figuras do trabalho
python src/graficos_exploratorios.py --base ./resultados/base_votacao.pkl --saida ./resultados_graficos --precedentes ./resultados_precedentes

# Teste do código com dados fictícios e gabarito conhecido (não usa dados reais)
python src/teste_sintetico.py
```

Análise de sensibilidade sem o critério textual de contingência:

```bash
python src/pipeline_stj.py --dados ./dados --saida ./resultados_so_cnj --inicio 2022-02-01 --so-cnj
```

A pasta `dados/` não é versionada: ela é recriada pelos scripts a partir das fontes públicas.

## Estrutura

```
src/                     scripts de coleta e análise
docs/                    livros de códigos (expressões regulares) e dicionário de dados
resultados/tabelas/      tabelas do trabalho e saídas completas (.csv e .txt)
resultados/figuras/      figuras 1 a 8 do trabalho; complementares/ traz as demais
```

As figuras estão nomeadas conforme a numeração do texto. A tabela 17 é a amostra codificada manualmente,
que sustenta a seção sobre direção dos resultados; a coluna `fora_do_escopo` marca os cinco acórdãos
excluídos da Tabela 7 do trabalho. A tabela 19 traz as proporções e os intervalos de confiança
representados na Figura 3.

As tabelas 07, 08, 09, 20, 21 e 22 exigem a pasta `dados/espelhos` passada em `--espelhos`: é ela que
resolve relator, órgão e data dos precedentes citados que não estão na própria base. Sem essa pasta o
total de citações não muda, mas a atribuição por ministro fica incompleta e a Tabela 8 do trabalho não
se reproduz.

A tabela 22 traz as citações no nível da ocorrência. O script imprime na tela a contagem bruta das
referências encontradas; o arquivo exportado e todas as métricas do trabalho contam uma vez cada
referência de um acórdão a um mesmo precedente, de modo que a repetição do mesmo julgado dentro de
uma ementa não infla a concentração.

## Decisões metodológicas registradas no código

- **Votação extraída por regras**, não por modelo de linguagem, para permitir auditoria integral (`extrair_votacao`).
- **Relator vencido**: `ministroRelator` é comparado a quem lavrou o acórdão antes de atribuir o voto vencido.
- **Reforma** = provimento total ou parcial; **manutenção** = desprovimento ou não conhecimento; embargos de
  declaração acolhidos e propostas de afetação não entram como reforma.
- **Intervalos de confiança** de proporções pelo método de Wilson; Fisher para tabelas 2x2 com poucos eventos.
- **Mudança de composição em 2025**: padronização direta pela composição até 2024 e repetição das séries no
  estrato estável dos agravos internos.
- **Julgamentos conjuntos** (mesma classe, relator, órgão de uniformização e data) contam como um precedente.
- **Casos ambíguos** recebem alerta e ficam fora das métricas até revisão manual.

## Limitações

O STJ gera espelho apenas para acórdãos com novidade de tese ou representatividade, de modo que a base é uma
seleção institucional e não o universo de acórdãos; sua composição por classe recursal mudou em 2025. O código
de assunto é atribuído na autuação e pode estar incompleto. A série começa em fevereiro de 2022, quando começam
os metadados das íntegras. Os dicionários de classificação são heurísticos: a auditoria manual de 150 acórdãos
indicou 3,3% fora do escopo e 8,7% classificados como de mérito sem o serem. A codificação da direção foi feita
por uma única pesquisadora, sem medida de confiabilidade entre codificadores.

## Licença e citação

Código sob licença MIT (ver `LICENSE`). Para citar, ver `CITATION.cff`.

## Uso de inteligência artificial

Assistente de IA foi utilizado como apoio na escrita e na depuração dos scripts em Python e na revisão de
linguagem. A concepção da pesquisa, as decisões metodológicas, a codificação manual, a validação dos
resultados e a interpretação são de responsabilidade da autora.
