# Dicionário das bases derivadas

## `resultados/base_votacao.csv` (uma linha por acórdão)

| Campo | Origem | Descrição |
| --- | --- | --- |
| id, processo, registro, classe, orgao, relator, data | espelho do acórdão | identificação do julgado |
| ementa, decisao, info_compl, termos_aux, ref_leg, tese, jurisp_citada | espelho do acórdão | textos usados nas classificações |
| criterio | derivado | `assunto_cnj` ou `texto_sem_assunto` (critério de contingência) |
| assuntos_cnj | metadados das íntegras | códigos da TPU/CNJ do processo |
| subtema, origem_subtema | derivado | subtema pela folha do CNJ ou por dicionário textual |
| unanime, maioria, vencido_parcial | certidão | tipo de votação |
| vencidos, participantes | certidão | listas de ministros, nomes padronizados |
| relator_c, lavrador, relator_substituido | certidão + campo do relator | relator padronizado e quem lavrou o acórdão |
| voto_vista, tese_repetitiva, afetacao | certidão | marcadores do julgamento |
| alerta | derivado | motivo de exclusão das métricas (vazio = caso válido) |

## `resultados_fundamentos/base_classificada.csv`

Acrescenta à base acima: `recurso`, `resultado`, `caminho` (filtro processual, mérito ou afetação),
`periodo` e uma coluna booleana por categoria dos livros de códigos (`docs/livro-de-codigos-fundamentos.csv`),
nos blocos de óbices, precedentes e argumentos.

## `resultados_tematica/base_tematica.csv`

Uma coluna booleana por objeto da demanda e por motivo de recusa, conforme
`docs/livro-de-codigos-tematico.csv`.

## `resultados_precedentes/citacoes.csv` (uma linha por par acórdão citante / precedente citado)

| Campo | Descrição |
| --- | --- |
| id_citante, orgao_citante, relator_citante, data_citante | acórdão que cita |
| precedente, rotulo | classe e número do precedente citado |
| fonte | ementa, informações complementares ou jurisprudência citada |
| frase | oração da ementa em que a citação aparece |

## `resultados/tabelas/17_amostra_direcao_codificada.csv`

Amostra manual de 150 acórdãos classificados como de mérito, com estrato, identificação do processo,
quem recorreu, direção do resultado e observações da codificação.
