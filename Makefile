.PHONY: help install ingest run cli test eval clean compose-local compose-cloud compose-mock

PYTHON := .venv/bin/python3
PIP := .venv/bin/pip
UVICORN := .venv/bin/uvicorn
PYTEST := .venv/bin/pytest

help:
	@echo "Comandos disponíveis:"
	@echo "  make install         - Instala dependências no ambiente virtual"
	@echo "  make ingest          - Executa ingestão e indexação offline dos documentos"
	@echo "  make run             - Inicia o servidor da API FastAPI em http://localhost:8000"
	@echo "  make cli             - Inicia o terminal interativo do agente de suporte"
	@echo "  make test            - Executa todos os testes unitários e de integração"
	@echo "  make eval            - Executa as baterias de teste de latência e concorrência"
	@echo "  make docker-build    - Constrói a imagem Docker da aplicação"
	@echo "  make docker-run      - Executa o container Docker na porta 8000"
	@echo "  make compose-up      - Sobe o serviço via docker-compose"
	@echo "  make compose-local   - Sobe API e modelo local Ollama (padrão)"
	@echo "  make compose-cloud   - Sobe somente a API usando OpenAI"
	@echo "  make compose-mock    - Sobe somente a API usando o mock de testes"
	@echo "  make compose-down    - Para o serviço via docker-compose"
	@echo "  make clean           - Remove arquivos temporários e caches"

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

ingest:
	PYTHONPATH=. $(PYTHON) -m src.application.ingestion

run:
	PYTHONPATH=. $(UVICORN) src.presentation.api:app --host 0.0.0.0 --port 8000 --reload

cli:
	PYTHONPATH=. $(PYTHON) -m src.presentation.cli

test:
	PYTHONPATH=. $(PYTEST) -v

eval:
	PYTHONPATH=. $(PYTHON) scripts/evaluate.py

docker-build:
	docker build -t agente-suporte-interno .

docker-run:
	docker run --env-file .env -p 8000:8000 --name agente-suporte-interno --rm agente-suporte-interno

compose-up:
	docker compose up --build -d

compose-local:
	COMPOSE_PROFILES=local LLM_PROVIDER=ollama docker compose up --build -d

compose-cloud:
	COMPOSE_PROFILES= LLM_PROVIDER=openai docker compose up --build -d

compose-mock:
	COMPOSE_PROFILES= LLM_PROVIDER=mock docker compose up --build -d

compose-down:
	COMPOSE_PROFILES=local docker compose down --remove-orphans

clean:
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
