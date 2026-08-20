const queryForm = document.querySelector("#query-form");
const questionInput = document.querySelector("#question");
const submitButton = document.querySelector("#submit-button");
const resultCard = document.querySelector("#result");
const answerElement = document.querySelector("#answer");
const resultContext = document.querySelector("#result-context");
const sourcesSection = document.querySelector("#sources-section");
const sourcesElement = document.querySelector("#sources");
const traceElement = document.querySelector("#trace-id");
const latencyElement = document.querySelector("#latency");
const errorCard = document.querySelector("#error");
const errorTitle = document.querySelector("#error-title");
const errorMessage = document.querySelector("#error-message");
const credentialsDialog = document.querySelector("#credentials-dialog");
const credentialsForm = document.querySelector("#credentials-form");
const apiKeyInput = document.querySelector("#api-key");
const credentialsError = document.querySelector("#credentials-error");
const API_KEY_STORAGE = "internal-support-api-key";

function getApiKey() {
  return sessionStorage.getItem(API_KEY_STORAGE) || "";
}

function openCredentials() {
  apiKeyInput.value = getApiKey();
  credentialsError.hidden = true;
  credentialsDialog.showModal();
  window.setTimeout(() => apiKeyInput.focus(), 0);
}

function showError(title, message) {
  resultCard.hidden = true;
  errorTitle.textContent = title;
  errorMessage.textContent = message;
  errorCard.hidden = false;
}

function clearFeedback() {
  resultCard.hidden = true;
  errorCard.hidden = true;
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  questionInput.disabled = isLoading;
  submitButton.querySelector(".button-label").textContent = isLoading ? "Consultando…" : "Perguntar";
}

function renderSources(sources) {
  sourcesElement.replaceChildren();

  for (const source of sources) {
    const item = document.createElement("article");
    const heading = document.createElement("div");
    const badge = document.createElement("span");
    const metadata = document.createElement("div");
    const title = document.createElement("strong");
    const detail = document.createElement("span");
    const excerpt = document.createElement("blockquote");

    item.className = "source-item";
    heading.className = "source-heading";
    badge.className = "citation-badge";
    metadata.className = "source-metadata";
    excerpt.className = "source-excerpt";
    badge.textContent = `[${source.citation_number}]`;
    title.textContent = source.section || source.source;
    detail.textContent = `${source.source} · trecho ${source.chunk_id}`;
    excerpt.textContent = source.excerpt || "Trecho validado na base interna.";
    metadata.append(title, detail);
    heading.append(badge, metadata);
    item.append(heading, excerpt);
    sourcesElement.append(item);
  }

  sourcesSection.hidden = sources.length === 0;
}

function renderResult(payload) {
  errorCard.hidden = true;
  answerElement.textContent = payload.answer || "Não encontrei uma resposta na base interna.";
  resultContext.textContent = payload.failure === "NO_EVIDENCE"
    ? "A base não contém evidências suficientes para responder."
    : `${payload.sources?.length || 0} ${payload.sources?.length === 1 ? "evidência verificada" : "evidências verificadas"}`;
  traceElement.textContent = payload.trace_id ? `trace ${payload.trace_id}` : "";
  latencyElement.textContent = payload.timings?.total !== undefined
    ? `${payload.timings.total} ms`
    : "";
  renderSources(payload.sources || []);
  resultCard.hidden = false;
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function submitQuestion(question) {
  const apiKey = getApiKey();
  if (!apiKey) {
    openCredentials();
    return;
  }

  clearFeedback();
  setLoading(true);
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 30000);

  try {
    const response = await fetch("/api/v1/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Internal-API-Key": apiKey,
      },
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });

    let payload;
    try {
      payload = await response.json();
    } catch {
      throw new Error("O servidor retornou uma resposta inválida.");
    }

    if (response.status === 401) {
      sessionStorage.removeItem(API_KEY_STORAGE);
      showError("Chave de acesso inválida", "Configure novamente a chave interna para continuar.");
      openCredentials();
      return;
    }

    if (!response.ok) {
      const message = payload.detail || payload.answer || "O serviço não conseguiu processar a pergunta.";
      throw new Error(message);
    }

    renderResult(payload);
  } catch (error) {
    if (error.name === "AbortError") {
      showError("A consulta demorou demais", "Tente novamente em alguns instantes.");
    } else {
      showError("Não foi possível concluir a consulta", error.message);
    }
  } finally {
    window.clearTimeout(timeout);
    setLoading(false);
  }
}

queryForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (question) {
    submitQuestion(question);
  }
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    queryForm.requestSubmit();
  }
});

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question;
    questionInput.focus();
  });
});

document.querySelector("#credentials-button").addEventListener("click", openCredentials);
document.querySelector("#close-dialog").addEventListener("click", () => credentialsDialog.close());

document.querySelector("#clear-key").addEventListener("click", () => {
  sessionStorage.removeItem(API_KEY_STORAGE);
  apiKeyInput.value = "";
  apiKeyInput.focus();
});

credentialsForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const key = apiKeyInput.value.trim();
  if (key.length < 32) {
    credentialsError.hidden = false;
    return;
  }

  sessionStorage.setItem(API_KEY_STORAGE, key);
  credentialsDialog.close();
  questionInput.focus();
});

credentialsDialog.addEventListener("click", (event) => {
  if (event.target === credentialsDialog) {
    credentialsDialog.close();
  }
});

async function checkHealth() {
  const dot = document.querySelector("#status-dot");
  const label = document.querySelector("#status-text");

  try {
    const response = await fetch("/health", { cache: "no-store" });
    if (!response.ok) {
      throw new Error("unhealthy");
    }
    dot.className = "status-dot is-online";
    label.textContent = "Serviço disponível";
  } catch {
    dot.className = "status-dot is-offline";
    label.textContent = "Serviço indisponível";
  }
}

checkHealth();
