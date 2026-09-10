const state = {
  system: "llm",
};

const SYSTEM_HINTS = {
  trivial: "Always predicts the majority intent, sends one canned reply, never escalates. The floor, not the goal.",
  simple: "TF-IDF + Logistic Regression intent classifier, template replies, keyword/risk-tier escalation rules.",
  llm: "Few-shot LLM classification + retrieval-grounded reply + LLM-reasoned escalation.",
};

const el = (id) => document.getElementById(id);

function setStatusPill(text, kind) {
  const pill = el("status-pill");
  pill.textContent = text;
  pill.className = `status-pill status-pill--${kind}`;
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.llm_configured) {
      setStatusPill(`backend ready · ${data.llm_provider} · ${data.num_historical_threads} threads indexed`, "ok");
    } else {
      setStatusPill("backend up, no LLM key set in .env — trivial/simple systems still work", "warn");
    }
  } catch (e) {
    setStatusPill("cannot reach backend — is it running?", "warn");
  }
}

function initSystemSelector() {
  document.querySelectorAll(".system-option").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".system-option").forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      state.system = btn.dataset.system;
      el("system-hint").textContent = SYSTEM_HINTS[state.system];
    });
  });
}

function renderResult(data) {
  el("result-empty").hidden = true;
  el("result-error").hidden = true;
  el("result-body").hidden = false;

  const verdict = el("verdict");
  verdict.classList.remove("is-auto", "is-escalate");
  verdict.classList.add(data.escalate ? "is-escalate" : "is-auto");
  el("verdict-badge").textContent = data.escalate ? "Escalate to human" : "Auto-handle";
  el("verdict-reason").textContent = data.escalate_reason;

  el("intent-tag").textContent = data.intent;
  const pct = Math.round((data.confidence || 0) * 100);
  el("confidence-fill").style.width = `${pct}%`;
  el("confidence-label").textContent = `confidence ${pct}%`;

  el("reply-text").textContent = data.reply;

  const list = el("grounding-list");
  list.innerHTML = "";
  if (!data.retrieved || data.retrieved.length === 0) {
    const li = document.createElement("li");
    li.className = "grounding-empty";
    li.textContent = "No similar historical resolutions found in the indexed thread data.";
    list.appendChild(li);
  } else {
    data.retrieved.forEach((r) => {
      const li = document.createElement("li");
      li.innerHTML = `
        <p class="g-cust">"${escapeHtml(r.customer_text)}"</p>
        <p class="g-reply">→ ${escapeHtml(r.brand_reply_text)}</p>
        <p class="g-sim">similarity ${r.similarity}</p>
      `;
      list.appendChild(li);
    });
  }
}

function renderError(message) {
  el("result-empty").hidden = true;
  el("result-body").hidden = true;
  const errBox = el("result-error");
  errBox.hidden = false;
  errBox.textContent = message;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

async function submitMessage() {
  const text = el("message-input").value.trim();
  if (!text) return;

  const btn = el("submit-btn");
  btn.disabled = true;
  btn.textContent = "Running…";

  try {
    const res = await fetch("/api/respond", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, system: state.system }),
    });
    const data = await res.json();
    if (!res.ok) {
      renderError(data.error || "Something went wrong.");
    } else {
      renderResult(data);
    }
  } catch (e) {
    renderError("Could not reach the backend. Is it running?");
  } finally {
    btn.disabled = false;
    btn.textContent = "Run agent →";
  }
}

async function loadMetrics() {
  try {
    const res = await fetch("/api/metrics");
    if (!res.ok) return;
    const data = await res.json();
    renderMetricsTable(data);
    el("metrics-hint").hidden = true;
  } catch (e) {
    // leave the hint showing
  }
}

function renderMetricsTable(results) {
  const systems = Object.keys(results);
  if (systems.length === 0) return;
  const allKeys = new Set();
  systems.forEach((s) => Object.keys(results[s]).forEach((k) => allKeys.add(k)));

  let html = '<table class="metrics-table"><thead><tr><th>metric</th>';
  systems.forEach((s) => (html += `<th>${s}</th>`));
  html += "</tr></thead><tbody>";
  allKeys.forEach((k) => {
    html += `<tr><td>${k}</td>`;
    systems.forEach((s) => {
      const v = results[s][k];
      const display = typeof v === "number" ? v.toFixed(3) : v ?? "—";
      html += `<td>${display}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table>";
  el("metrics-table-wrap").innerHTML = html;
}

el("submit-btn").addEventListener("click", submitMessage);
el("message-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submitMessage();
});

initSystemSelector();
checkHealth();
loadMetrics();
