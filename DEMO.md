# Roteiro de demonstração

## Preparação

Suba a API, o modelo local e os embeddings pelo Docker Compose:

```bash
make compose-local
```

Consulte pelo CLI executado dentro do container:

```bash
docker exec -it agente-suporte-interno python -m src.presentation.cli
```

Ou consulte a API diretamente, substituindo `SUA_CHAVE_INTERNA` pelo valor de
`INTERNAL_API_KEY` configurado no `.env`:

```bash
curl -sS http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -H 'X-Internal-API-Key: SUA_CHAVE_INTERNA' \
  --data '{"question":"Qual o valor da ajuda de custo mensal de home office?"}'
```

Em respostas fundamentadas, a interface converte o `chunk_id` interno em uma
referência numérica como `[1]` e apresenta a fonte logo abaixo.

## Perguntas e resultados esperados

### 1. Resposta direta — reembolso em viagem

**Pergunta:** Qual o limite de valor para reembolso de almoço e jantar em viagens?

**Esperado:** resposta fundamentada contendo `R$ 120,00`, referência `[1]` e fonte
`reembolso.md`. Chunk esperado: `reembolso_c2`.

### 2. Resposta direta — senha corporativa

**Pergunta:** Quantos caracteres no mínimo deve ter a senha corporativa?

**Esperado:** resposta fundamentada contendo `12 caracteres`, referência `[1]` e
fonte `seguranca_informacao.md`. Chunk esperado: `seguranca_informacao_c1`.

### 3. Paráfrase — uso de veículo próprio

**Pergunta:** Quanto a firma devolve se eu rodar com meu próprio carro a trabalho?

**Esperado:** resposta fundamentada contendo `R$ 1,20`, referência `[1]` e fonte
`reembolso.md`. Chunk esperado: `reembolso_c2`.

### 4. Paráfrase — notebook furtado

**Pergunta:** O que acontece se meu notebook for furtado?

**Esperado:** resposta fundamentada contendo o prazo de `24 horas`, referência `[1]`
e fonte `seguranca_informacao.md`. Chunk esperado: `seguranca_informacao_c4`.

### 5. Fora da base — receita

**Pergunta:** Qual a receita do bolo de cenoura com cobertura de chocolate?

**Esperado:** `NO_EVIDENCE`, sem fontes, com a mensagem padrão: “Não encontrei essa
informação na base de políticas e procedimentos.”

### 6. Fora da base — cachorro no escritório

**Pergunta:** Posso levar meu cachorro para trabalhar no escritório todo dia?

**Esperado:** `NO_EVIDENCE`, sem fontes, com a mensagem padrão: “Não encontrei essa
informação na base de políticas e procedimentos.”

### 7. Caso de borda — reembolso de VR

**Pergunta:** Como pedir reembolso de VR?

**Esperado:** `NO_EVIDENCE`, sem fontes, com a mensagem padrão: “Não encontrei essa
informação na base de políticas e procedimentos.” A expansão de `VR` não deve
transformar a política de benefício em evidência de reembolso.

### 8. Descoberta do escopo

**Pergunta:** Sobre quais assuntos posso perguntar?

**Esperado:** resposta fundamentada listando os assuntos derivados dos documentos,
incluindo “Política de Segurança da Informação”, referência `[1]` e fonte
`base_interna`. Chunk esperado: `knowledge_base_catalog_c1`.
