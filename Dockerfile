# Imagem base Python slim para menor consumo de recursos e segurança
FROM python:3.11-slim

# Variáveis de ambiente para Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Instalação de utilitários do sistema (curl para healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalação de dependências
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Cópia do código-fonte, dados e scripts
COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY Makefile .

# Porta da API
EXPOSE 8000

# Healthcheck do container
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicialização
CMD ["uvicorn", "src.presentation.api:app", "--host", "0.0.0.0", "--port", "8000"]
