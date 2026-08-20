# ARCHITECTURE.md

## 1. Contexto e escopo

Assistente de suporte interno que responde dúvidas de colaboradores sobre políticas
e procedimentos de uma empresa fictícia (férias, reembolso, home office, segurança
da informação, onboarding).

- **Cenário:** processo interno (suporte a colaboradores).
- **Fonte de dados:** **A) RAG** sobre base documental local em Markdown.
- **Por quê:** o corpus é estável, permite avaliação objetiva das respostas e atende
  naturalmente ao requisito de citar fonte e trecho. Uma API externa (opção B)
  acrescentaria variabilidade de rede sem agregar nada à discussão de arquitetura.

Requisitos não funcionais assumidos: resposta sempre em português e nunca sem
evidência na base. No provedor cloud, o alvo é p95 ponta a ponta abaixo de 5 s; no
modelo local, latência e concorrência dependem da CPU e da memória disponíveis.

## 2. Decisão arquitetural central

**Monólito modular em camadas, exposto por HTTP, com fluxo RAG determinístico em
LangGraph.**

As camadas são módulos do mesmo processo, sem chamadas HTTP internas. A separação
existe no nível de interfaces (portas e adaptadores), não de rede: o custo de
serialização e o overhead operacional de microserviços não se justificam nesta
escala.

O grafo tem autonomia **restrita e limitada** (seção 4.2). Não há planner aberto,
loop livre nem rodadas arbitrárias de tool calling. Isso reduz custo, variância de
latência e superfície de alucinação, e torna cada caminho testável.

## 3. Visão geral

````mermaid
flowchart TD
    CLI["Adaptador CLI"] --> GRAFO
    WEB["HTML/CSS/JS"] --> API["FastAPI"]
    HTTP["Cliente HTTP"] --> API
    API --> GRAFO["Grafo LangGraph<br/>(camada de aplicação)"]
    GRAFO --> DOM["Domínio<br/>regras de suficiência,<br/>políticas de fallback"]
    GRAFO --> REPO["Porta: RetrievalRepository"]
    GRAFO --> LLMP["Porta: LLMClient"]
    INDEX --> EMBP["Porta: EmbeddingClient"]
    REPO --> INDEX["Adaptador vetorial híbrido<br/>(índice versionado em memória)"]
    LLMP --> OLLAMA["Ollama local<br/>Qwen3 1.7B"]
    EMBP --> ELOCAL["Ollama local<br/>EmbeddingGemma 300M"]
    EMBP --> ECLOUD["OpenAI embeddings"]
    LLMP --> OPENAI["OpenAI cloud"]
    LLMP --> MOCK["Mock testes/CI"]
    ING["Ingestão offline<br/>(comando separado)"] --> INDEX
    GRAFO --> TEL["Telemetria<br/>logs, métricas, trace_id"]
````

### Camadas e responsabilidades

| Camada | Responsabilidade | Depende de |
| --- | --- | --- |
| Apresentação | Endpoint FastAPI, adaptador CLI, validação de schema, mapeamento de erro para status HTTP | Aplicação |
| Aplicação | Grafo LangGraph, orquestração dos nós, timeouts, retry, tratamento de falha | Domínio + portas |
| Domínio | Estado da execução, regra de suficiência de evidência, modelos de resposta, políticas de fallback | nada |
| Portas | Interfaces `RetrievalRepository`, `LLMClient` e `EmbeddingClient` | Domínio |
| Infraestrutura | Adaptadores de índice vetorial e LLM, embeddings, configuração, logs, métricas | Portas |

A regra de dependência aponta sempre para dentro. Trocar o índice local por pgvector ou o
provedor de LLM não altera o grafo nem o domínio, apenas o adaptador e a configuração.
O módulo `src/bootstrap.py` é o composition root: ele é o único ponto que conecta o
grafo às implementações concretas das portas usadas em produção.

### 3.1 Porta de LLM e troca de provedor

O grafo depende exclusivamente da porta assíncrona `LLMClient`, que define geração,
reescrita, health check e encerramento. A factory de infraestrutura escolhe o
adaptador no startup por `LLM_PROVIDER`; nenhuma condição de fornecedor existe na
camada de aplicação.

| Adaptador de saída | Uso | Dependência externa |
| --- | --- | --- |
| `OllamaLLMAdapter` | padrão local, CPU, dados não saem da máquina | serviço Ollama no Compose |
| `OpenAICompatibleLLMAdapter` | cloud, maior qualidade e elasticidade | API e chave OpenAI |
| `MockLLMAdapter` | testes, CI e diagnóstico determinístico | nenhuma |

A troca é feita por configuração e restart, sem alterar rotas, grafo, domínio ou
formato das respostas. O `/health` identifica explicitamente o provedor e o modelo
ativos.

No adaptador local, o LLM leve executa reescrita e reranking. A composição final é
extrativa e reutiliza apenas frases dos chunks selecionados; essa decisão evita que
um modelo pequeno introduza regras, números ou prazos ausentes. No adaptador cloud, a
síntese pode ser gerativa, mas continua obrigada a citar IDs recuperados e passa
pela mesma validação do grafo.

### 3.2 Porta de embeddings e recuperação semântica

A criação do índice e a consulta dependem da mesma porta `EmbeddingClient`. No modo
local, `OllamaEmbeddingAdapter` usa EmbeddingGemma; no modo cloud,
`OpenAIEmbeddingAdapter` usa a API OpenAI; o adaptador `dense` por hashing fica
restrito a testes e CI. O índice grava fingerprints do provedor/modelo e do corpus,
além de sua dimensão. No startup, qualquer incompatibilidade ou alteração nos
Markdowns dispara uma nova ingestão, evitando vetores obsoletos ou produzidos por
modelos diferentes.

O ranking combina similaridade semântica (80%) e cobertura lexical (20%). A primeira
recupera intenções e paráfrases sem palavras idênticas; a segunda preserva precisão
para siglas, números e nomes. Um verificador local aceita somente os chunks que
respondem diretamente, e há fallback lexical apenas para correspondências literais
de alta confiança.

## 4. Fluxo do agente

### 4.1 Grafo

````mermaid
stateDiagram-v2
    [*] --> validate_input
    validate_input --> retrieve_context: entrada válida
    validate_input --> fallback_invalid: vazia ou longa demais

    retrieve_context --> check_evidence
    retrieve_context --> fallback_source: repositório indisponível

    check_evidence --> generate_answer: evidência suficiente
    check_evidence --> rewrite_query: evidência fraca e ainda não reescrito
    check_evidence --> fallback_no_answer: evidência insuficiente após reescrita

    rewrite_query --> retrieve_context

    generate_answer --> validate_citations
    generate_answer --> fallback_llm: timeout ou erro do LLM

    validate_citations --> [*]: citações válidas
    validate_citations --> fallback_no_answer: citou ID fora do contexto

    fallback_invalid --> [*]
    fallback_source --> [*]
    fallback_no_answer --> [*]
    fallback_llm --> [*]
````

### 4.2 Autonomia limitada

Quando a recuperação vem fraca, `rewrite_query` permite que o LLM reformule a
pergunta (expanda siglas, resolva linguagem coloquial e extraia termos-chave), e a
busca é refeita **uma única vez**. Depois da recuperação, o LLM local julga cada
trecho isoladamente com uma decisão binária (`SIM`/`NAO`). Ele nunca redige fatos.
Para consultas puramente semânticas, a recuperação também aplica ablação de termos:
remove cada conceito relevante em lote e mede se a similaridade aumenta. Um aumento
material indica que a busca só encontrou o documento ao ignorar uma restrição da
pergunta; nesse caso, a evidência é recusada sem reescrita. Assim, paráfrases como
"trabalhar em meu lar" permanecem válidas, enquanto objetos ausentes da base não
são apagados para forçar uma resposta. Esses limites mantêm a execução previsível e
impedem loop infinito.

Um agente com planner livre resolveria o mesmo problema com custo e variância muito
maiores, sem ganho mensurável neste domínio.

### 4.3 Estado

````python
class AgentState(TypedDict):
    trace_id: str
    question: str
    effective_query: str        # muda apenas se rewrite_query rodar
    rewrite_count: int          # teto: 1
    retrieved_chunks: list[Chunk]
    evidence_score: float
    answer: str | None
    cited_chunk_ids: list[str]
    timings: dict[str, float]   # por nó, em ms
    failure: FailureKind | None
````

### 4.4 Comportamento quando não sabe responder

Os modos de falha são distintos e nunca se confundem com uma resposta normal:

| Situação | `failure` | HTTP | Mensagem ao usuário |
| --- | --- | --- | --- |
| Entrada inválida | `INVALID_INPUT` | 400 | pedido de reformulação |
| Nada relevante na base | `NO_EVIDENCE` | 200 | "não encontrei essa informação na base" |
| Índice/repositório fora | `SOURCE_UNAVAILABLE` | 503 | "a fonte está temporariamente indisponível" |
| LLM em timeout ou erro | `LLM_UNAVAILABLE` | 503 | "não foi possível gerar a resposta agora" |
| Citação inválida | `NO_EVIDENCE` | 200 | mesma de nada relevante |

O agente nunca improvisa a partir de conhecimento paramétrico: o prompt restringe a
geração aos trechos recuperados e `validate_citations` descarta a resposta se
qualquer `chunk_id` citado não estiver no contexto daquela execução.

## 5. Recuperação e ingestão

````mermaid
flowchart LR
    DOC["docs/*.md"] --> SPLIT["Chunking<br/>~700 tokens, overlap 100<br/>respeitando cabeçalhos"]
    SPLIT --> META["Metadados<br/>source, section, chunk_id"]
    META --> EMB["Embeddings<br/>(multilingual)"]
    EMB --> IDX["Índice vetorial NumPy<br/>artefato versionado"]
    IDX --> LOAD["Carga em memória<br/>no startup da API"]
````

- **Modelo de embedding local:** `embeddinggemma:300m-qat-q4_0`, multilíngue e
  executado pelo Ollama; no modo cloud, `text-embedding-3-small`.
- **Ingestão é feita antes de servir consultas** e pode ser executada por comando
  (`make ingest`). Se o índice estiver ausente ou incompatível, o startup o recria.
  Nenhum embedding de documento é calculado no caminho de uma requisição normal.
- **Índice de runtime versionado** (`index/v{N}/`), carregado uma vez no startup e
  não armazenado no Git. No Compose ele persiste em volume próprio; fora do Docker,
  fica no diretório local ignorado `index/`. O fingerprint de corpus, provedor,
  modelo e dimensão determina se pode ser reutilizado. Em escala, a publicação de
  nova versão é troca de ponteiro + restart das réplicas.

### Suficiência de evidência

Regra composta, calibrada com o próprio conjunto de avaliação (que inclui perguntas
propositalmente fora da base):

````
evidência suficiente  ⟺  score_max ≥ τ_forte  OU
                          (score_max ≥ τ  E  ao menos 2 chunks da mesma fonte
                           com score ≥ τ - δ)
````

Os valores de `τ`, `τ_forte` e `δ` estão em `config.py`. Uma correspondência forte pode sustentar
a resposta sozinha porque diversas regras aparecem em apenas uma seção do corpus;
resultados limítrofes continuam exigindo corroboração. Threshold único de similaridade
é frágil, pois o valor absoluto varia por modelo e por domínio.

## 6. Latência

### Gargalos identificados

````mermaid
flowchart LR
    A["Embedding da pergunta<br/>~10%"] --> B["Busca vetorial NumPy<br/>~2%"]
    B --> C["Geração no LLM<br/>~85%"]
    C --> D["Validação + serialização<br/>~3%"]
````

A geração no LLM domina. Otimizar recuperação tem retorno marginal; otimizar
**tamanho do contexto e do output** tem retorno direto.

### Otimizações aplicadas

| # | Mudança | Efeito esperado |
| --- | --- | --- |
| 1 | Índice carregado no startup, não sob demanda | elimina cold start do primeiro request |
| 2 | `top_k` de 8 → 4 e truncagem de chunk no prompt | menos tokens de entrada, menos tempo de prefill |
| 3 | `max_tokens` limitado + instrução de resposta objetiva | corta a cauda de respostas longas no p95 |
| 4 | Cliente HTTP assíncrono com pool reaproveitado | remove handshake TLS por request |
| 5 | Cache LRU do embedding da pergunta | perguntas repetidas na demo não pagam o encode |
| 6 | Streaming opcional no endpoint | melhora latência percebida (TTFT), não a total |

Medição de 2026-08-20, com 26 perguntas × 3 rodadas:

| Bateria | Provedores | Acerto funcional | p50 | p95 | Evidência |
| --- | --- | ---: | ---: | ---: | --- |
| A | mock + dense hashing | 48/78 (61,5%) | 2,00 ms | 2,59 ms | `evidence/latency_mock.csv` |
| B | Qwen3 1.7B + EmbeddingGemma | 78/78 (100%) | 1.740,24 ms | 2.312,25 ms | `evidence/latency_ollama.csv` |

A bateria A isola o teto da implementação, mas não mede inferência real e **não
substitui** a bateria B ponta a ponta. A medição final foi feita com o modelo já
carregado; as percentis incluem recuperação, reescrita, verificação, composição e
validação de citações.

O acerto menor no mock/dense é esperado: hashing de n-gramas não oferece a semântica
necessária para as paráfrases do dataset. Sem reescritas específicas de perguntas,
falham q02, q05, q06, q07, q09, q18, q20, q21, q22 e q24; o Ollama semântico mantém
78/78.

O conjunto de avaliação tem 26 perguntas em cinco grupos: resposta direta,
paráfrase, pergunta sem resposta na base, casos de borda e descoberta do escopo.
Cada execução registra provedor, latência ponta a ponta, falha, score de recuperação,
quantidade de reescritas e conformidade com a resposta/fonte esperada.

## 7. Escalabilidade

### Evolução

````mermaid
flowchart TD
    C["Clientes"] --> LB["Balanceador"]
    LB --> A1["Réplica API<br/>(stateless)"]
    LB --> A2["Réplica API<br/>(stateless)"]
    LB --> A3["Réplica API<br/>(stateless)"]
    A1 --> IDX["Índice versionado<br/>(volume compartilhado ou pgvector)"]
    A2 --> IDX
    A3 --> IDX
    A1 --> SEM["Limite de concorrência<br/>por réplica"]
    A2 --> SEM
    A3 --> SEM
    SEM --> LLM["LLM externo"]
    JOB["Job de ingestão"] --> IDX
    A1 --> OBS["Métricas e tracing"]
    A2 --> OBS
    A3 --> OBS
````

A aplicação é stateless por construção: todo o estado vive no `AgentState` de uma
execução. Escala horizontal é replicação simples atrás de um balanceador.

### Duas baterias de teste, propositalmente separadas

Medir escala com LLM real e concorrência alta mede o **rate limit do provedor**, não
a aplicação. Por isso:

| Bateria | LLM | Concorrência | O que mede |
| --- | --- | --- | --- |
| A (teto da aplicação) | mock determinístico + dense | 1, 5, 10, 20, 40 | throughput do código, tempo em fila, ponto de saturação, erros |
| B (ponta a ponta honesto) | Qwen3 + EmbeddingGemma no Ollama | 1, 5, 10 | p50/p95 verdadeiros, saturação e erros do modelo local |

Reportar as duas separadamente. A bateria A **não** substitui a B: ela isola a
infraestrutura, e isso precisa estar escrito no relatório.

Resultados medidos em 2026-08-20, com 100 requisições por nível:

| Bateria | Concorrência | Throughput | p95 | Erros |
| --- | ---: | ---: | ---: | ---: |
| A | 1 | 507,9 req/s | 2,52 ms | 0% |
| A | 5 | 711,8 req/s | 8,32 ms | 0% |
| A | 10 | 759,8 req/s | 15,42 ms | 0% |
| A | 20 | 784,0 req/s | 28,61 ms | 0% |
| A | 40 | 758,9 req/s | 61,83 ms | 0% |
| B | 1 | 0,6 req/s | 2.962,65 ms | 0% |
| B | 5 | 0,3 req/s | 30.118,52 ms | 0% |
| B | 10 | 0,3 req/s | 57.612,79 ms | 0% |

Na bateria A, o throughput atinge o platô entre 10 e 20 concorrentes e a fila passa
a dominar o p95. Na bateria B, o único processo Ollama já está saturado em
concorrência 1; níveis 5 e 10 reduzem o throughput e elevam o p95 para dezenas de
segundos. Os dados completos estão em `evidence/scale_mock.csv` e
`evidence/scale_ollama.csv`.

### Controle de admissão

Somente o `POST /api/v1/query` passa pelo controle de admissão da borda HTTP. Cada
processo admite uma execução por vez (`MAX_CONCURRENT_REQUESTS=1`) e mantém até
quatro requisições aguardando (`MAX_QUEUE_DEPTH=4`). Uma fila já cheia é rejeitada
imediatamente; uma espera superior a `ADMISSION_TIMEOUT_SECONDS=10.0` também recebe
HTTP 429 com `Retry-After: 8`. O valor do header usa quatro posições × p50 aproximado
de 2 s. Saúde, métricas, arquivos estáticos e preflight não disputam esse semáforo.

O limite inicialmente proposto de duas execuções foi reduzido para uma após a
medição real: com duas, C=5 entregou somente 0,30 req/s e p95 de 23,92 s; com uma,
entregou 0,65 req/s e p95 de 10,76 s.

| Concorrência | Sem admissão: throughput | Sem admissão: p95 | Com admissão: throughput admitido | Com admissão: p95 admitido | HTTP 429 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0,6 req/s | 2.962,65 ms | 0,67 req/s | 2.348,25 ms | 0% |
| 5 | 0,3 req/s | 30.118,52 ms | 0,65 req/s | 10.755,80 ms | 0% |
| 10 | 0,3 req/s | 57.612,79 ms | 0,36 req/s | 10.608,35 ms | 92% |

A medição sem admissão usa 100 chamadas por nível diretamente no grafo; a medição
HTTP usa 50 chamadas válidas por nível e calcula percentis somente sobre as
admitidas. Em C=10, rejeitar 92% rapidamente preservou o p95 admitido próximo de
10,6 s, em vez de degradar toda a carga até 57,6 s. A rejeição é, portanto, o
comportamento de proteção esperado. Os dados HTTP estão em
`evidence/scale_ollama_api.csv`.

## 8. Riscos e mitigações

| Risco | Mitigação |
| --- | --- |
| Acesso não autorizado | API key interna obrigatória em consultas e métricas, comparação em tempo constante e CORS restrito por allowlist |
| Rate limit ou lentidão do LLM | timeout por chamada, no máximo 1 retry com backoff, semáforo de concorrência por réplica |
| Sobrecarga da API | controle implementado com `MAX_CONCURRENT_REQUESTS`, `MAX_QUEUE_DEPTH` e `ADMISSION_TIMEOUT_SECONDS`; fila limitada e HTTP 429 com `Retry-After` |
| Base indisponível | health check no startup e no `/health`, `SOURCE_UNAVAILABLE` sem improvisar resposta |
| Alucinação | prompt restrito ao contexto, `validate_citations` obrigatória, descarte da resposta em citação inválida |
| Prompt injection nos documentos | documentos tratados como dados; o agente não tem ferramenta de escrita nem ação externa; instruções embutidas em chunk não têm efeito |
| Índice desatualizado | fingerprint de corpus/provedor/modelo/dimensão, reconstrução automática e versão registrada em cada resposta |
| Custo de LLM | contexto curto, `top_k` limitado, `max_tokens` teto, log de tokens por request, alerta de custo diário |
| Threshold mal calibrado | conjunto de avaliação com perguntas fora da base; regressão roda em CI |

## 9. Trade-offs assumidos

| Escolha | Ganho | Custo aceito |
| --- | --- | --- |
| Monólito modular | simplicidade, deploy único, sem latência de rede interna | escala por réplica inteira, não por componente |
| Grafo determinístico | previsibilidade, custo e latência controlados, testável | menos flexível diante de perguntas multi-hop complexas |
| Índice NumPy em memória | latência de busca desprezível, zero infraestrutura extra | índice cabe na RAM; atualização exige republicação |
| Ingestão offline | nenhum embedding de documento no caminho quente | base não reflete mudanças em tempo real |
| Uma única reescrita de query | teto de latência garantido | perde alguns casos que 2 a 3 reescritas resolveriam |

## 10. ADR-001: Separação de responsabilidades para dezenas de usuários concorrentes

**Status:** aceita
**Data:** 2026-08-20

### Contexto

O agente precisa atender dezenas de usuários concorrentes em produção. A
implementação atual é um processo único que faz ingestão, recuperação, geração e
exposição HTTP. O gargalo dominante é externo (LLM), e o índice é um artefato
imutável carregado em memória.

### Decisão

Manter o monólito modular como unidade de deploy e separar responsabilidades por
**ciclo de vida**, não por função:

1. **Plano online:** API + grafo, stateless, replicado horizontalmente atrás de um
   balanceador. Escala com o número de requisições.
2. **Plano offline:** ingestão e indexação como job independente, produzindo
   artefato versionado. Escala com o volume de documentos, que muda em ordem de
   grandeza diferente.
3. **Armazenamento:** índice como artefato compartilhado; migra para pgvector ou
   serviço vetorial gerenciado quando a base deixar de caber em memória ou exigir
   atualização incremental.
4. **Acesso ao LLM:** encapsulado em adaptador com semáforo de concorrência,
   timeout e retry único. É o recurso escasso e precisa de controle explícito de
   admissão, não de mais réplicas.
5. **Telemetria:** coleta desacoplada (métricas + tracing com `trace_id`
   propagado), fora do caminho crítico.

### Alternativas consideradas

- **Microserviços por nó do grafo:** rejeitada, pois acrescenta latência de rede e
  complexidade operacional para resolver um gargalo que é externo à aplicação.
- **Agente com planner e tool calling livre:** rejeitada, pois multiplica chamadas ao
  LLM, que é justamente o recurso limitante; piora p95 e custo.
- **Serviço vetorial gerenciado desde o início:** adiada, devido a dependência externa e
  latência de rede sem ganho na escala atual. A porta `RetrievalRepository` mantém
  essa migração barata.

### Consequências

**Positivas:** deploy simples, escala horizontal trivial, gargalo isolado atrás de
uma porta única, migração de armazenamento sem tocar no grafo.

**Negativas:** réplicas escalam como bloco inteiro; atualização da base exige
republicação de artefato e restart; sem cache compartilhado entre réplicas.

**Próximos passos se a carga crescer:** cache de respostas por pergunta normalizada
(Redis), pgvector para atualização incremental, autoscaling por profundidade de
fila em vez de CPU.

## 11. O que mudaria

**Sob mais carga:** cache compartilhado de respostas frequentes, autoscaling por
profundidade de fila, pgvector no lugar do índice NumPy em memória e admissão de
requisições com prioridade.

**Sob falhas de rede:** circuit breaker no adaptador do LLM (após N falhas, responder
`LLM_UNAVAILABLE` direto sem tentar), degradação para modo "apenas trechos" (devolver
os trechos recuperados sem síntese, o que ainda é útil ao usuário).

**Sob pressão de custo:** roteamento por complexidade (modelo menor para perguntas
diretas), cache semântico de perguntas parafraseadas, redução de `top_k` guiada pelo
impacto medido na qualidade, orçamento diário com corte gracioso.
