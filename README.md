# Agente de Suporte Interno (RAG Determinístico)

Assistente corporativo para responder dúvidas de colaboradores sobre políticas internas (férias, reembolso, home office, segurança da informação e onboarding), construído com arquitetura monolítica modular em camadas e fluxo determinístico em LangGraph.

---

## 🏛️ Arquitetura e Decisões de Projeto

O projeto segue à risca as diretrizes estabelecidas em [`ARQUITETURA.md`](ARQUITETURA.md):

- **Monólito Modular em Camadas:**
  - **Apresentação (`src/presentation`):** API FastAPI com mapeamento estrito de erros para status HTTP (400, 200, 503) e CLI para terminal.
  - **Aplicação (`src/application`):** Grafo de estados determinístico (LangGraph) e pipeline de ingestão offline.
  - **Domínio (`src/domain`):** Estado imutável `AgentState`, regra de suficiência de evidência (`score_max >= τ` e 2 chunks `>= τ - δ`), modelos de dados e `FailureKind`.
  - **Portas (`src/ports`):** Interfaces abstratas `RetrievalRepository`, `LLMClient` e `EmbeddingClient`.
  - **Infraestrutura (`src/infrastructure`):** Adaptadores intercambiáveis de LLM e embeddings (Ollama local, OpenAI cloud e alternativas determinísticas de testes), repositório vetorial híbrido em memória, cache LRU e telemetria (métricas p50/p95, `trace_id` e logger estruturado).

---

## 🚀 Como Executar

### 1. Configurar o ambiente
```bash
# macOS e Linux
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```
Troque `INTERNAL_API_KEY` por uma chave aleatória longa. Essa chave é obrigatória nos
endpoints de consulta e métricas pelo header `X-Internal-API-Key`.

### 2. Instalar dependências
```bash
make install
```
*(ou crie seu ambiente com `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`)*

Antes dos passos 3 e 4, escolha um modo para executar sem Docker:

- **Ollama nativo:** instale e inicie o Ollama, baixe `qwen3:1.7b` e
  `embeddinggemma:300m-qat-q4_0` e altere `OLLAMA_BASE_URL` no `.env` para
  `http://localhost:11434/api`. O hostname `ollama` do `.env.example` só é
  resolvido pela rede interna do Docker Compose.
- **Avaliação rápida sem downloads:** altere o `.env` para
  `LLM_PROVIDER=mock`, `EMBEDDING_PROVIDER=dense` e `COMPOSE_PROFILES=`. Esse
  modo determinístico mede o teto da infraestrutura, mas não substitui a
  avaliação semântica ponta a ponta com Ollama.

### 3. Executar ingestão offline da base de conhecimento
O pipeline lê os documentos em `data/docs/*.md`, divide em chunks respeitando
cabeçalhos e cria o índice local em `index/v1/`:
```bash
make ingest
```

### 4. Rodar a API HTTP (FastAPI)
```bash
make run
```
A API estará disponível em: `http://localhost:8000` (documentação Swagger interativa em `http://localhost:8000/docs`).

### 5. Rodar com Docker Compose e modelo local 🐳

Pré-requisito: Docker Desktop ou Docker Engine com Compose v2.20 ou superior e
containers Linux habilitados.

O modo padrão sobe a API e o Ollama em containers Linux CPU-only. Na primeira
execução, o Compose baixa o LLM `qwen3:1.7b` (aproximadamente 1,4 GB) e o modelo
semântico multilíngue `embeddinggemma:300m-qat-q4_0` (aproximadamente 239 MB),
mantendo ambos no volume persistente `ollama-models`:

```bash
# macOS, Linux ou Windows PowerShell
docker compose up --build -d

# Acompanhar apenas o primeiro download do modelo
docker compose logs -f ollama-model
```

Espere o endpoint `/health` informar `healthy` e abra:

- Interface: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- Estado e modelo ativo: `http://localhost:8000/health`

O Qwen padrão tem 1,7 bilhão de parâmetros, suporta português e roda sem GPU.
Para uma experiência previsível, reserve pelo menos 4 GB de RAM ao Docker e espaço
em disco para a imagem do Ollama e o modelo. A imagem oficial possui variantes
Linux `amd64` e `arm64`, cobrindo PCs Windows/Linux e Macs Intel/Apple Silicon com
Docker Desktop em modo de containers Linux.

O EmbeddingGemma transforma pergunta e documentos em vetores semânticos. Assim,
uma pergunta como "Como se previnem de hackers?" pode encontrar regras de MFA,
senhas, antivírus e atualizações mesmo sem repetir as palavras do Markdown. O Qwen
verifica quais trechos realmente respondem à pergunta, enquanto a resposta final é
composta de forma extrativa com frases da própria base. Isso melhora paráfrases sem
permitir que números, prazos ou regras inventados pelo modelo pequeno cheguem ao
usuário. O adaptador cloud continua capaz de fazer síntese gerativa, submetida à
mesma validação de citações.

#### Trocar o provedor sem alterar código

A aplicação depende das portas `LLMClient` e `EmbeddingClient`. A escolha dos
adaptadores ocorre na inicialização por `LLM_PROVIDER` e `EMBEDDING_PROVIDER`:

| Modo | `LLM_PROVIDER` | `EMBEDDING_PROVIDER` | `COMPOSE_PROFILES` | Credencial |
|---|---|---|---|---|
| Local padrão | `ollama` | `ollama` | `local` | nenhuma |
| Cloud OpenAI | `openai` | `openai` | vazio | `OPENAI_API_KEY` |
| Testes/CI | `mock` | `dense` | vazio | nenhuma |

Para trocar, altere o `.env` e recrie os containers:

```bash
docker compose down --remove-orphans
docker compose up --build -d
```

No modo cloud, configure por exemplo:

```dotenv
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
COMPOSE_PROFILES=
OPENAI_API_KEY=sua-chave
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Ao adicionar ou editar arquivos em `data/docs/*.md`, recrie/reinicie a aplicação.
O startup detecta mudança nos documentos, no provedor ou no modelo, invalida índices
incompatíveis e refaz a indexação com o embedding ativo.

`index/` é um artefato regenerável de runtime e não é versionado no Git. No Compose,
ele fica no volume `agente-index`; fora do Docker, fica em `index/v1/`. A imagem não
embute um índice `dense`: o startup gera o índice com o provedor de embeddings
realmente configurado e o reutiliza enquanto corpus, provedor, modelo e dimensão
permanecerem compatíveis.

No primeiro acesso à interface, clique em **Configurar acesso** e informe o mesmo
valor de `INTERNAL_API_KEY` definido no arquivo `.env`. A chave fica somente na
sessão da aba do navegador.

O mesmo container entrega a interface HTML/CSS/JS, a API e a documentação em
`http://localhost:8000/docs`; não é necessário instalar Node.js ou executar outro
serviço.

As respostas exibidas no navegador e no CLI seguem o mesmo formato: afirmações
com referências numéricas (`[1]`, `[2]`, ...), seguidas dos trechos exatos que as
fundamentam. Identificadores internos de chunk e métricas de recuperação ficam
separados como metadados técnicos; o score de recuperação não representa uma
probabilidade calibrada de a resposta estar correta.

### 6. Consultar via Terminal (CLI)
Você pode interagir interativamente ou passar a pergunta diretamente como argumento:
```bash
# Pergunta única
.venv/bin/python3 -m src.presentation.cli "Como funciona o fracionamento de férias?"

# Modo chat interativo
make cli
```

---

## 🧪 Testes Automatizados

Para rodar a suíte completa de testes (unitários de suficiência de evidência, testes de todos os fluxos e fallbacks do grafo LangGraph, testes de integração da API e pipeline de ingestão):
```bash
make test
```

---

## 📊 Avaliação e Benchmarks de Latência e Escala

O projeto conta com um script de avaliação automatizado cobrindo 26 perguntas
(resposta direta, paráfrases semânticas, descoberta dinâmica do escopo, perguntas
fora da base e casos de borda). As medições são separadas por provedor:

```bash
# Bateria A: teto da infraestrutura, sem downloads de modelos
make eval-mock

# Bateria B: ponta a ponta real; requer o stack local em execução
make compose-local
make eval-ollama
```

Resultados medidos em 2026-08-20:

| Bateria | Provedores | Acerto funcional | p50 | p95 |
| --- | --- | ---: | ---: | ---: |
| A | mock + dense hashing | 48/78 (61,5%) | 2,00 ms | 2,59 ms |
| B | Qwen3 1.7B + EmbeddingGemma | 78/78 (100%) | 1.740,24 ms | 2.312,25 ms |

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

A bateria A mede o teto do código e da recuperação determinística, mas **não
substitui** a bateria B. A medição real mostra que o único processo Ollama local
satura acima de concorrência 1: aumentar o paralelismo reduz o throughput e aumenta
fortemente o tempo em fila.

A taxa funcional menor da bateria A é esperada: o embedding `dense` é hashing de
n-gramas para testes rápidos, sem capacidade semântica para resolver paráfrases. O
mock também não executa julgamento semântico e, após a remoção das reescritas
específicas da avaliação, falha em q02, q05, q06, q07, q09, q18, q20, q21, q22 e
q24. Esses casos passam na bateria B, exceto q18, que passa justamente por ser
recusada corretamente como `NO_EVIDENCE`.

Os CSVs preservam o provedor em cada linha:

- `evidence/latency_mock.csv` e `evidence/scale_mock.csv`;
- `evidence/latency_ollama.csv` e `evidence/scale_ollama.csv`.

---

## 📡 Endpoints da API

- `POST /api/v1/query`: Processa a dúvida do colaborador e retorna resposta
  fundamentada. O campo `answer` usa referências numéricas e cada item de
  `sources` relaciona a referência ao arquivo, seção, chunk e trecho utilizado.
  `evidence_score` é um score de recuperação para diagnóstico, não uma medida de
  confiança. Requer `X-Internal-API-Key`.
  - Retorna `200 OK`: Sucesso com resposta ou mensagem de que não encontrou evidências na base (`NO_EVIDENCE`).
  - Retorna `400 Bad Request`: Pergunta vazia ou muito longa (`INVALID_INPUT`).
  - Retorna `503 Service Unavailable`: Repositório ou LLM indisponível (`SOURCE_UNAVAILABLE` / `LLM_UNAVAILABLE`).
- `GET /health`: Health check do repositório vetorial, LLM e embeddings. Também
  informa provedores e modelos ativos, sem expor credenciais.
- `GET /metrics`: Métricas de requisições totais, erros e percentis de latência (p50, p95, p99). Requer `X-Internal-API-Key`.
