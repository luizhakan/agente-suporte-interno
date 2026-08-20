"""Configuração determinística da suíte, independente do provedor escolhido no .env."""
import os


os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "dense"
