(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const DEFAULT_CONFIG = {
    hardware: {
      source: "simulation",
      sampling_interval_seconds: 2,
      auto_start: false,
      allow_real_actuators: false,
      simulation_seed: 42,
      serial: { port: "", baud_rate: 115200, timeout_seconds: 2, line_ending: "\n" },
      sensors: [
        { name: "temperature_c", feature_name: "temperature_c", label: "Temperatura", source_key: "temperature", unit: "°C", value_type: "float", simulation_min: 18, simulation_max: 38, precision: 2 },
        { name: "humidity_pct", feature_name: "humidity_pct", label: "Umidità", source_key: "humidity", unit: "%", value_type: "float", simulation_min: 30, simulation_max: 85, precision: 2 },
        { name: "vibration_g", feature_name: "vibration_g", label: "Vibrazione", source_key: "vibration", unit: "g", value_type: "float", simulation_min: 0, simulation_max: 2.5, precision: 3 },
        { name: "light_lux", feature_name: "light_lux", label: "Luce", source_key: "light", unit: "lux", value_type: "float", simulation_min: 80, simulation_max: 900, precision: 0 },
      ],
    },
    active_model_id: null,
  };

  const state = {
    config: structuredClone(DEFAULT_CONFIG),
    savedConfig: structuredClone(DEFAULT_CONFIG),
    models: [],
    rules: [],
    events: [],
    history: [],
    latest: null,
    dataset: null,
    trainedModel: null,
    streaming: false,
    pollingTimer: null,
    chartSensor: "",
    sensorConnected: false,
  };

  const pageTitles = {
    monitor: "Monitor in tempo reale",
    training: "Laboratorio di machine learning",
    hardware: "Configurazione Arduino",
    actuators: "Regole per gli attuatori",
  };

  const operatorLabels = { gt: ">", gte: "≥", lt: "<", lte: "≤", eq: "=", ne: "≠" };
  const formToApiOperator = { ">": "gt", ">=": "gte", "<": "lt", "<=": "lte", "==": "eq", "!=": "ne" };
  const apiToFormOperator = { gt: ">", gte: ">=", lt: "<", lte: "<=", eq: "==", ne: "!=" };
  const sensorPalette = [
    { color: "#3f7dcc", tint: "#edf5ff", icon: "T" },
    { color: "#0b8f86", tint: "#e7f7f4", icon: "%" },
    { color: "#9b6a25", tint: "#fff5df", icon: "∿" },
    { color: "#7657bb", tint: "#f1edfb", icon: "S" },
    { color: "#c25062", tint: "#fff0f2", icon: "●" },
  ];

  async function api(path, options = {}) {
    const init = { ...options, headers: { ...(options.headers || {}) } };
    if (init.body && !(init.body instanceof FormData) && typeof init.body !== "string") {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(init.body);
    }
    init.headers.Accept = "application/json";

    let response;
    try {
      response = await fetch(`/api${path}`, init);
    } catch (error) {
      throw new Error("Backend non raggiungibile. Verifica che il server sia avviato.");
    }

    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json") ? await response.json() : await response.text();
    if (!response.ok) {
      const detail = payload?.detail ?? payload?.message ?? payload;
      const message = Array.isArray(detail)
        ? detail.map((item) => item.msg || JSON.stringify(item)).join(" · ")
        : typeof detail === "object"
          ? JSON.stringify(detail)
          : String(detail || `Errore HTTP ${response.status}`);
      throw new Error(message);
    }
    return payload;
  }

  function toast(title, message = "", type = "success") {
    const node = document.createElement("div");
    node.className = `toast${type === "error" ? " is-error" : ""}`;
    node.innerHTML = `
      <span class="toast-icon" aria-hidden="true">${type === "error" ? "!" : "✓"}</span>
      <div><strong></strong><span></span></div>
      <button type="button" aria-label="Chiudi notifica">×</button>`;
    $("strong", node).textContent = title;
    $("div span", node).textContent = message;
    $("button", node).addEventListener("click", () => removeToast(node));
    $("#toast-region").append(node);
    window.setTimeout(() => removeToast(node), 5200);
  }

  function removeToast(node) {
    if (!node?.isConnected || node.classList.contains("is-leaving")) return;
    node.classList.add("is-leaving");
    window.setTimeout(() => node.remove(), 220);
  }

  function formatNumber(value, maximumFractionDigits = 2) {
    const number = Number(value);
    return Number.isFinite(number)
      ? new Intl.NumberFormat("it-IT", { maximumFractionDigits }).format(number)
      : String(value ?? "—");
  }

  function formatDate(value, timeOnly = false) {
    if (!value) return "—";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return new Intl.DateTimeFormat("it-IT", timeOnly
      ? { hour: "2-digit", minute: "2-digit", second: "2-digit" }
      : { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }).format(date);
  }

  function humanize(value) {
    return String(value || "Sensore")
      .replace(/_c$/i, "")
      .replace(/_pct$/i, "")
      .replace(/[_-]+/g, " ")
      .replace(/\b\w/g, (char) => char.toUpperCase());
  }

  function extractArray(payload, keys = []) {
    if (Array.isArray(payload)) return payload;
    for (const key of keys) if (Array.isArray(payload?.[key])) return payload[key];
    return [];
  }

  function normalizeReading(payload) {
    const raw = payload?.reading || payload?.sample || payload?.data || payload;
    if (!raw || typeof raw !== "object") return null;
    const reserved = new Set(["id", "timestamp", "source", "prediction", "model", "events", "reading"]);
    let values = raw.values || raw.sensors || raw.features;
    if (!values) {
      values = Object.fromEntries(Object.entries(raw).filter(([key, value]) => !reserved.has(key) && (typeof value === "number" || typeof value === "string")));
    }
    if (!values || !Object.keys(values).length) return null;
    return {
      id: raw.id || crypto.randomUUID(),
      timestamp: raw.timestamp || new Date().toISOString(),
      source: raw.source || state.config.hardware.source,
      values,
      prediction: raw.prediction || payload?.prediction || null,
    };
  }

  function normalizeConfig(payload) {
    const config = payload?.config || payload || {};
    const hardware = config.hardware || config;
    const serial = hardware.serial || {};
    const sensors = extractArray(hardware.sensors || hardware.sensor_mapping || hardware.mappings).map((item) => ({
      ...item,
      source_key: item.source_key || item.key || item.arduino_key || item.name || "",
      feature_name: item.feature_name || item.feature || item.name || "",
      label: item.label || humanize(item.feature_name || item.feature || item.name),
      unit: String(item.unit || "").replace("Â", ""),
    }));
    return {
      ...DEFAULT_CONFIG,
      ...config,
      hardware: {
        ...DEFAULT_CONFIG.hardware,
        ...hardware,
        source: hardware.source || hardware.mode || DEFAULT_CONFIG.hardware.source,
        sampling_interval_seconds: Number(hardware.sampling_interval_seconds || hardware.interval_seconds || DEFAULT_CONFIG.hardware.sampling_interval_seconds),
        simulation_seed: Number(hardware.simulation_seed ?? DEFAULT_CONFIG.hardware.simulation_seed),
        allow_real_actuators: Boolean(hardware.allow_real_actuators ?? hardware.actuators_enabled),
        serial: {
          ...DEFAULT_CONFIG.hardware.serial,
          ...serial,
          baud_rate: Number(serial.baud_rate || serial.baud || DEFAULT_CONFIG.hardware.serial.baud_rate),
          timeout_seconds: Number(serial.timeout_seconds || serial.timeout || DEFAULT_CONFIG.hardware.serial.timeout_seconds),
        },
        sensors: sensors.length ? sensors : structuredClone(DEFAULT_CONFIG.hardware.sensors),
      },
    };
  }

  function navigate() {
    const requested = location.hash.replace("#", "") || "monitor";
    const page = pageTitles[requested] ? requested : "monitor";
    $$(".page").forEach((node) => {
      const active = node.dataset.page === page;
      node.hidden = !active;
      node.classList.toggle("is-active", active);
    });
    $$(".nav-link").forEach((node) => {
      const active = node.dataset.nav === page;
      node.classList.toggle("is-active", active);
      active ? node.setAttribute("aria-current", "page") : node.removeAttribute("aria-current");
    });
    $("#page-title").textContent = pageTitles[page];
    $(".sidebar").classList.remove("is-open");
    $("#mobile-menu").setAttribute("aria-expanded", "false");
    if (page === "monitor") window.setTimeout(drawChart, 50);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function setConnectionUI(status, label) {
    const pill = $("#connection-pill");
    pill.classList.toggle("is-pending", status === "pending");
    pill.classList.toggle("is-error", status === "error");
    $("#connection-label").textContent = label;
    $("#sidebar-status-dot").style.background = status === "error" ? "var(--red)" : status === "pending" ? "var(--amber)" : "var(--green)";
  }

  function sensorUnit(name) {
    const mapping = state.config.hardware.sensors.find((item) => item.feature_name === name || item.source_key === name);
    if (mapping?.unit) return mapping.unit;
    if (/_c$/i.test(name) || /temp/i.test(name)) return "°C";
    if (/_pct$/i.test(name) || /humid/i.test(name)) return "%";
    return "";
  }

  function sensorLabel(name) {
    const mapping = state.config.hardware.sensors.find((item) => item.feature_name === name || item.name === name || item.source_key === name);
    return mapping?.label || humanize(name);
  }

  function renderSensorCards() {
    const grid = $("#sensor-grid");
    const values = state.latest?.values || {};
    const entries = Object.entries(values);
    if (!entries.length) {
      grid.innerHTML = `<article class="metric-card"><div class="chart-empty"><strong>Nessun sensore disponibile</strong><span>Configura il mapping o avvia una lettura.</span></div></article>`;
      return;
    }

    const previous = state.history.length > 1 ? state.history[state.history.length - 2]?.values || {} : {};
    grid.innerHTML = entries.map(([key, value], index) => {
      const palette = sensorPalette[index % sensorPalette.length];
      const current = Number(value);
      const prior = Number(previous[key]);
      const delta = Number.isFinite(current) && Number.isFinite(prior) ? current - prior : null;
      const trend = delta == null || Math.abs(delta) < 0.0001
        ? "stabile rispetto alla lettura precedente"
        : `${delta > 0 ? "↑" : "↓"} ${formatNumber(Math.abs(delta))} dalla lettura precedente`;
      return `<article class="metric-card" style="--metric-color:${palette.color};--metric-tint:${palette.tint}">
        <div class="metric-top"><span class="metric-name">${escapeHtml(sensorLabel(key))}</span><span class="metric-icon" aria-hidden="true">${palette.icon}</span></div>
        <div class="metric-value"><strong>${escapeHtml(formatNumber(value))}</strong><span>${escapeHtml(sensorUnit(key))}</span></div>
        <span class="metric-trend">${escapeHtml(trend)}</span>
      </article>`;
    }).join("");
    updateChartSensorOptions(Object.keys(values));
  }

  function updateChartSensorOptions(keys) {
    const select = $("#chart-sensor");
    const current = state.chartSensor && keys.includes(state.chartSensor) ? state.chartSensor : keys[0] || "";
    const optionsSignature = keys.join("|");
    if (select.dataset.signature !== optionsSignature) {
      select.innerHTML = keys.map((key) => `<option value="${escapeHtml(key)}">${escapeHtml(sensorLabel(key))}</option>`).join("");
      select.dataset.signature = optionsSignature;
    }
    state.chartSensor = current;
    select.value = current;
    $("#chart-legend").textContent = current ? `${sensorLabel(current)} ${sensorUnit(current)}`.trim() : "Valore sensore";
  }

  function drawChart() {
    const canvas = $("#live-chart");
    const wrap = $("#chart-wrap");
    if (!canvas || !wrap || wrap.offsetWidth === 0) return;
    const points = state.history
      .map((reading) => ({ value: Number(reading.values?.[state.chartSensor]), time: reading.timestamp }))
      .filter((point) => Number.isFinite(point.value))
      .slice(-60);
    $("#chart-empty").hidden = points.length > 0;
    $("#chart-window").textContent = `Ultime ${points.length} letture`;

    const cssWidth = wrap.clientWidth;
    const cssHeight = wrap.clientHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(cssWidth * dpr);
    canvas.height = Math.floor(cssHeight * dpr);
    canvas.style.width = `${cssWidth}px`;
    canvas.style.height = `${cssHeight}px`;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cssWidth, cssHeight);
    if (!points.length) return;

    const pad = { top: 16, right: 18, bottom: 30, left: 46 };
    const width = cssWidth - pad.left - pad.right;
    const height = cssHeight - pad.top - pad.bottom;
    let min = Math.min(...points.map((point) => point.value));
    let max = Math.max(...points.map((point) => point.value));
    const extra = (max - min || Math.max(Math.abs(max) * .1, 1)) * .18;
    min -= extra;
    max += extra;

    ctx.font = "9px Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let i = 0; i <= 4; i += 1) {
      const y = pad.top + (height / 4) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(cssWidth - pad.right, y);
      ctx.strokeStyle = "#e7edea";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = "#8b9993";
      ctx.fillText(formatNumber(max - ((max - min) / 4) * i, 1), pad.left - 8, y);
    }

    const xFor = (index) => pad.left + (points.length === 1 ? width / 2 : (index / (points.length - 1)) * width);
    const yFor = (value) => pad.top + (1 - (value - min) / (max - min)) * height;
    const gradient = ctx.createLinearGradient(0, pad.top, 0, cssHeight - pad.bottom);
    gradient.addColorStop(0, "rgba(20,163,115,.25)");
    gradient.addColorStop(1, "rgba(20,163,115,0)");
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = xFor(index);
      const y = yFor(point.value);
      index ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.lineTo(xFor(points.length - 1), cssHeight - pad.bottom);
    ctx.lineTo(xFor(0), cssHeight - pad.bottom);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    ctx.beginPath();
    points.forEach((point, index) => {
      const x = xFor(index);
      const y = yFor(point.value);
      index ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.strokeStyle = "#11976b";
    ctx.lineWidth = 2;
    ctx.stroke();
    const last = points[points.length - 1];
    ctx.beginPath();
    ctx.arc(xFor(points.length - 1), yFor(last.value), 3.5, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.strokeStyle = "#087b58";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  function renderPrediction(prediction = state.latest?.prediction) {
    const activeModel = state.models.find((model) => model.active) || state.models.find((model) => model.id === state.config.active_model_id);
    if (!prediction) {
      $("#prediction-empty").hidden = false;
      $("#prediction-result").hidden = true;
      $("#prediction-empty strong").textContent = activeModel ? "In attesa della prima previsione" : "Nessun modello attivo";
      $("#prediction-empty p").textContent = activeModel
        ? "Avvia il monitor per elaborare una lettura con il modello attivo."
        : "Allena e attiva un modello per ottenere previsioni dai sensori.";
      return;
    }
    const value = prediction.value ?? prediction.prediction ?? prediction.label ?? prediction.class ?? prediction;
    const confidence = prediction.confidence ?? prediction.probability;
    $("#prediction-empty").hidden = true;
    $("#prediction-result").hidden = false;
    $("#prediction-label").textContent = activeModel?.problem_type === "classification" ? "Classe prevista" : "Valore previsto";
    $("#prediction-value").textContent = typeof value === "number" ? formatNumber(value, 3) : String(value);
    $("#prediction-confidence").textContent = confidence == null ? "" : `Confidenza ${formatNumber(Number(confidence) * (Number(confidence) <= 1 ? 100 : 1), 1)}%`;
    $("#prediction-model").textContent = activeModel?.name || prediction.model_name || "Modello attivo";
    $("#prediction-time").textContent = formatDate(state.latest?.timestamp || new Date(), true);
  }

  function renderEvents() {
    const events = state.events.slice(0, 30);
    const activity = $("#activity-body");
    const actuatorBody = $("#actuator-events-body");
    if (!events.length) {
      activity.innerHTML = `<tr class="empty-row"><td colspan="4">Nessuna attività registrata.</td></tr>`;
      actuatorBody.innerHTML = `<tr class="empty-row"><td colspan="5">Nessun evento attuatore.</td></tr>`;
      return;
    }
    activity.innerHTML = events.slice(0, 6).map((event) => `<tr>
      <td>${escapeHtml(formatDate(event.timestamp, true))}</td>
      <td>${escapeHtml(event.rule_name || "Regola attuatore")}</td>
      <td>${escapeHtml(event.message || `${event.actuator || "Attuatore"}: ${event.command || "—"}`)}</td>
      <td><span class="event-badge ${event.success === false ? "error" : event.execution_mode === "serial" ? "success" : "dry"}">${event.success === false ? "Errore" : event.execution_mode === "serial" ? "Eseguito" : "Simulato"}</span></td>
    </tr>`).join("");
    actuatorBody.innerHTML = events.map((event) => `<tr>
      <td>${escapeHtml(formatDate(event.timestamp))}</td>
      <td>${escapeHtml(event.rule_name || event.rule_id || "—")}</td>
      <td>${escapeHtml(formatNumber(event.prediction))}</td>
      <td><code>${escapeHtml(event.command || "—")}</code></td>
      <td><span class="event-badge ${event.execution_mode === "serial" ? "success" : "dry"}">${event.execution_mode === "serial" ? "Seriale" : "Simulata"}</span></td>
    </tr>`).join("");
    $("#rules-triggered").textContent = events.filter((event) => {
      const eventDate = new Date(event.timestamp);
      const now = new Date();
      return eventDate.toDateString() === now.toDateString();
    }).length;
  }

  function setStreaming(active) {
    state.streaming = active;
    const button = $("#toggle-stream");
    $(".button-icon", button).textContent = active ? "■" : "▶";
    $("span:last-child", button).textContent = active ? "Ferma monitor" : "Avvia monitor";
    button.classList.toggle("button-primary", !active);
    button.classList.toggle("button-secondary", active);
    $("#sidebar-source").textContent = active
      ? state.config.hardware.source === "serial" ? "Arduino connesso" : "Simulatore attivo"
      : "Monitor fermo";
    setConnectionUI(active ? "ok" : "pending", active ? "Acquisizione attiva" : "Sistema pronto");
    clearTimeout(state.pollingTimer);
    state.pollingTimer = null;
    if (active) pollSensor();
  }

  async function toggleStreaming() {
    const button = $("#toggle-stream");
    button.disabled = true;
    try {
      if (state.streaming) {
        await api("/sensors/disconnect", { method: "POST" });
        state.sensorConnected = false;
        setStreaming(false);
      } else {
        await api("/sensors/connect", { method: "POST" });
        state.sensorConnected = true;
        setStreaming(true);
      }
    } catch (error) {
      toast("Connessione non riuscita", error.message, "error");
      setConnectionUI("error", "Errore connessione");
    } finally {
      button.disabled = false;
    }
  }

  async function pollSensor() {
    if (!state.streaming) return;
    try {
      let payload = await api("/sensors/latest").catch(() => null);
      if (!payload) payload = await api("/sensors/sample", { method: "POST" });
      const reading = normalizeReading(payload);
      if (reading) {
        if (!reading.prediction && payload?.prediction) reading.prediction = payload.prediction;
        if (!reading.prediction && state.models.some((model) => model.active)) {
          const inference = await api("/predictions/latest").catch(() => null);
          reading.prediction = inference?.prediction || null;
          if (inference?.actuator_events?.length) {
            state.events = [...inference.actuator_events, ...state.events.filter((event) => !inference.actuator_events.some((next) => next.id === event.id))];
          }
        }
        state.latest = reading;
        if (!state.history.some((item) => item.id === reading.id)) state.history.push(reading);
        if (state.history.length > 120) state.history = state.history.slice(-120);
        $("#sidebar-sample").textContent = `Ultimo dato ${formatDate(reading.timestamp, true)}`;
        renderSensorCards();
        drawChart();
        renderPrediction(reading.prediction);
      }
      if (payload?.events?.length) {
        state.events = [...payload.events, ...state.events];
        renderEvents();
      }
    } catch (error) {
      toast("Lettura non riuscita", error.message, "error");
      setStreaming(false);
      setConnectionUI("error", "Acquisizione interrotta");
      return;
    }
    const delay = Math.max(250, Number(state.config.hardware.sampling_interval_seconds || 2) * 1000);
    state.pollingTimer = window.setTimeout(pollSensor, delay);
  }

  function normalizeDataset(payload, fallbackName) {
    const raw = payload?.dataset || payload || {};
    const columns = (raw.columns || raw.column_names || []).map((item) => typeof item === "string" ? item : item.name);
    return {
      id: raw.dataset_id || raw.id,
      name: raw.name || raw.filename || fallbackName || "dataset.csv",
      columns,
      rows: Number(raw.row_count || raw.rows || raw.n_rows || 0),
      preview: raw.preview || raw.sample || [],
      suggestedTarget: raw.suggested_target || raw.target || "",
      problemType: raw.suggested_problem_type || raw.problem_type || "regression",
    };
  }

  async function uploadDataset(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      toast("Formato non supportato", "Seleziona un file con estensione .csv.", "error");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      toast("File troppo grande", "Il limite è 10 MB.", "error");
      return;
    }
    const form = new FormData();
    form.append("file", file);
    setTrainingLoading(true, "Caricamento CSV…");
    try {
      const payload = await api("/datasets/upload", { method: "POST", body: form });
      setDataset(normalizeDataset(payload, file.name));
      toast("Dataset caricato", "Controlla l'anteprima e configura il modello.");
    } catch (error) {
      toast("Impossibile caricare il dataset", error.message, "error");
    } finally {
      setTrainingLoading(false);
    }
  }

  async function loadDemo(problemType, button) {
    button.disabled = true;
    try {
      const payload = await api("/datasets/demo", { method: "POST", body: { problem_type: problemType, rows: 240, seed: 42 } });
      const dataset = normalizeDataset(payload, `demo_${problemType}.csv`);
      dataset.problemType = problemType;
      setDataset(dataset);
      const task = $(`input[name="task"][value="${problemType}"]`);
      if (task) task.checked = true;
      syncTaskAlgorithm();
      toast("Dataset dimostrativo pronto", problemType === "classification" ? "Scenario di classificazione generato." : "Scenario di regressione generato.");
    } catch (error) {
      toast("Generazione non riuscita", error.message, "error");
    } finally {
      button.disabled = false;
    }
  }

  function setTrainingLoading(active, label = "") {
    const upload = $("#upload-zone");
    upload.style.opacity = active ? ".55" : "1";
    upload.style.pointerEvents = active ? "none" : "auto";
    if (active) toast("Operazione in corso", label);
  }

  function setDataset(dataset) {
    state.dataset = dataset;
    $("#dataset-summary").hidden = false;
    $("#dataset-name").textContent = dataset.name;
    $("#dataset-shape").textContent = `${dataset.rows || "?"} righe · ${dataset.columns.length} colonne`;
    $("#to-model-step").disabled = !dataset.id || dataset.columns.length < 2;
    renderPreview(dataset);
    populateTrainingColumns(dataset.columns, dataset.suggestedTarget);
    const suggestedTask = $(`input[name="task"][value="${dataset.problemType}"]`);
    if (suggestedTask) suggestedTask.checked = true;
    syncTaskAlgorithm();
  }

  function renderPreview(dataset) {
    const table = $("#dataset-preview");
    if (!dataset.preview?.length) {
      table.innerHTML = `<tbody><tr><td>Nessuna anteprima restituita dal server.</td></tr></tbody>`;
      return;
    }
    const columns = dataset.columns.length ? dataset.columns : Object.keys(dataset.preview[0]);
    table.innerHTML = `<thead><tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr></thead>
      <tbody>${dataset.preview.slice(0, 5).map((row) => `<tr>${columns.map((column) => `<td>${escapeHtml(formatNumber(row[column]))}</td>`).join("")}</tr>`).join("")}</tbody>`;
  }

  function populateTrainingColumns(columns, suggestedTarget = "") {
    const target = $("#target-column");
    target.innerHTML = `<option value="">Seleziona la colonna da prevedere…</option>${columns.map((column) => `<option value="${escapeHtml(column)}">${escapeHtml(column)}</option>`).join("")}`;
    const preferred = columns.includes(suggestedTarget) ? suggestedTarget : columns.find((column) => /target|label|classe|stato|consumo/i.test(column)) || columns.at(-1);
    target.value = preferred || "";
    $("#feature-options").innerHTML = columns.map((column) => `<label class="feature-chip"><input type="checkbox" value="${escapeHtml(column)}" ${column === preferred ? "disabled" : "checked"} /><span>${escapeHtml(column)}</span></label>`).join("");
  }

  function showTrainingStep(step) {
    $$("[data-training-step]").forEach((node) => { node.hidden = Number(node.dataset.trainingStep) !== step; });
    $$("[data-step-marker]").forEach((node) => {
      const marker = Number(node.dataset.stepMarker);
      node.classList.toggle("is-current", marker === step);
      node.classList.toggle("is-complete", marker < step);
      if (marker < step) $(":scope > span", node).textContent = "✓";
      else $(":scope > span", node).textContent = marker;
    });
    $("#training-page").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function syncTargetFeatures() {
    const target = $("#target-column").value;
    $$("#feature-options input").forEach((input) => {
      input.disabled = input.value === target;
      if (input.disabled) input.checked = false;
    });
  }

  function syncTaskAlgorithm() {
    const task = $("input[name=" + CSS.escape("task") + "]:checked")?.value || "regression";
    const linear = $("#linear-choice input");
    linear.disabled = task === "classification";
    if (linear.disabled && linear.checked) $("input[name=" + CSS.escape("algorithm") + "][value=" + CSS.escape("xgboost") + "]").checked = true;
    $("#algorithm-hint").textContent = task === "classification"
      ? "Per classificare è necessario XGBoost; Linear Regression prevede solo valori numerici."
      : "Linear Regression privilegia semplicità e interpretabilità.";
    syncAlgorithmOptions();
  }

  function syncAlgorithmOptions() {
    const algorithm = $("input[name=" + CSS.escape("algorithm") + "]:checked")?.value;
    $("#xgb-options").hidden = algorithm !== "xgboost";
  }

  async function trainModel(event) {
    event.preventDefault();
    if (!state.dataset?.id) {
      toast("Dataset mancante", "Carica o genera un dataset prima di continuare.", "error");
      return;
    }
    const task = $("input[name=" + CSS.escape("task") + "]:checked").value;
    const algorithm = $("input[name=" + CSS.escape("algorithm") + "]:checked").value;
    const target = $("#target-column").value;
    const features = $$("#feature-options input:checked").map((input) => input.value);
    if (!target || !features.length) {
      toast("Configurazione incompleta", "Scegli una variabile obiettivo e almeno una variabile di ingresso.", "error");
      return;
    }
    const body = {
      dataset_id: state.dataset.id,
      name: $("#model-name").value.trim() || `${algorithm === "xgboost" ? "XGBoost" : "Linear Regression"} · ${target}`,
      target,
      features,
      problem_type: task,
      algorithm,
      test_size: Number($("#test-size").value) / 100,
      random_state: Number($("#random-state").value),
      hyperparameters: algorithm === "xgboost" ? {
        n_estimators: Number($("#n-estimators").value),
        max_depth: Number($("#max-depth").value),
        learning_rate: Number($("#learning-rate").value),
      } : {},
    };
    const button = $("#train-model");
    button.disabled = true;
    button.classList.add("is-loading");
    $(".button-label", button).textContent = "Addestramento…";
    try {
      const payload = await api("/models/train", { method: "POST", body });
      state.trainedModel = payload.model || payload;
      renderTrainingResult(state.trainedModel);
      await loadModels();
      showTrainingStep(3);
      toast("Modello addestrato", "Le metriche sono pronte per la valutazione.");
    } catch (error) {
      toast("Addestramento non riuscito", error.message, "error");
    } finally {
      button.disabled = false;
      button.classList.remove("is-loading");
      $(".button-label", button).textContent = "Avvia addestramento";
    }
  }

  function renderTrainingResult(model) {
    $("#result-model-name").textContent = model.name || "Modello pronto";
    $("#result-summary").textContent = `${model.algorithm === "xgboost" ? "XGBoost" : "Linear Regression"} · ${model.train_rows || "?"} righe training · ${model.test_rows || "?"} righe test`;
    const metricHints = {
      r2: "Più vicino a 1 è meglio", r2_score: "Più vicino a 1 è meglio", rmse: "Più basso è meglio", mae: "Più basso è meglio",
      accuracy: "Quota di risposte corrette", f1: "Equilibrio tra precisione e richiamo", f1_score: "Equilibrio tra precisione e richiamo",
      f1_weighted: "F1 medio pesato per numerosità delle classi",
      precision: "Correttezza delle classi positive", precision_weighted: "Precisione media pesata per classe",
      recall: "Copertura delle classi positive", recall_weighted: "Recall medio pesato per classe",
    };
    const metrics = Object.entries(model.metrics || {}).filter(([, value]) => typeof value === "number");
    $("#metrics-grid").innerHTML = metrics.map(([key, value]) => `<article class="metric-result"><small>${escapeHtml(key.replaceAll("_", " "))}</small><strong>${escapeHtml(formatNumber(value, 4))}</strong><span>${escapeHtml(metricHints[key.toLowerCase()] || "Metrica sul set di test")}</span></article>`).join("") || `<p>Nessuna metrica numerica restituita.</p>`;
    $("#result-notice").textContent = model.problem_type === "classification"
      ? "Controlla accuracy e F1 insieme: una sola metrica può essere fuorviante se le classi non sono bilanciate."
      : "Confronta R² con gli errori MAE/RMSE: il modello migliore spiega bene i dati e mantiene l'errore basso.";
    $("#activate-trained-model").disabled = Boolean(model.active);
    $("#activate-trained-model").textContent = model.active ? "Già attivo" : "Attiva sul monitor";
  }

  async function loadModels() {
    const payload = await api("/models");
    state.models = extractArray(payload, ["models", "items"]);
    renderModels();
    renderPrediction();
  }

  function modelScore(model) {
    const metrics = model.metrics || {};
    const key = model.problem_type === "classification"
      ? Object.keys(metrics).find((name) => /accuracy|f1/i.test(name))
      : Object.keys(metrics).find((name) => /^r2|r2_score/i.test(name));
    return key ? `${key.replaceAll("_", " ")}: ${formatNumber(metrics[key], 3)}` : "Metriche disponibili";
  }

  function renderModels() {
    $("#model-count").textContent = state.models.length;
    const list = $("#model-list");
    if (!state.models.length) {
      list.innerHTML = `<div class="inline-empty">Nessun modello disponibile.</div>`;
      return;
    }
    list.innerHTML = state.models.map((model) => `<article class="model-item">
      <div class="model-item-top"><div><strong>${escapeHtml(model.name || model.id)}</strong><small>${escapeHtml(model.algorithm)} · ${escapeHtml(model.problem_type)}</small></div>${model.active ? `<span class="model-active">Attivo</span>` : ""}</div>
      <div class="model-item-actions"><span class="model-score">${escapeHtml(modelScore(model))}</span>${model.active ? "" : `<button class="mini-action" type="button" data-activate-model="${escapeHtml(model.id)}">Attiva</button>`}</div>
    </article>`).join("");
  }

  async function activateModel(modelId) {
    if (!modelId) return;
    try {
      await api(`/models/${encodeURIComponent(modelId)}/activate`, { method: "POST" });
      state.config.active_model_id = modelId;
      await loadModels();
      toast("Modello attivato", "Le prossime letture verranno elaborate con questo modello.");
    } catch (error) {
      toast("Attivazione non riuscita", error.message, "error");
    }
  }

  function applyConfigToForm() {
    const hardware = state.config.hardware;
    const sourceRadio = $(`input[name="source-mode"][value="${hardware.source}"]`) || $("input[name=" + CSS.escape("source-mode") + "][value=" + CSS.escape("simulation") + "]");
    sourceRadio.checked = true;
    $("#simulation-interval").value = Math.round(hardware.sampling_interval_seconds * 1000);
    $("#simulation-seed").value = hardware.simulation_seed;
    $("#serial-port").value = hardware.serial.port || "";
    $("#baud-rate").value = String(hardware.serial.baud_rate || 115200);
    $("#serial-timeout").value = hardware.serial.timeout_seconds || 2;
    $("#line-ending").value = hardware.serial.line_ending === "\r\n" ? "CRLF" : "LF";
    $("#hardware-output-enabled").checked = Boolean(hardware.allow_real_actuators);
    renderMappings(hardware.sensors);
    updateSourceSettings();
    updateOutputUI();
    $("#config-dirty").hidden = true;
  }

  function renderMappings(mappings) {
    const list = $("#sensor-mappings");
    list.innerHTML = "";
    (mappings.length ? mappings : [{ source_key: "", feature_name: "", unit: "" }]).forEach(addMappingRow);
    updateProtocolExample();
  }

  function addMappingRow(mapping = { source_key: "", feature_name: "", unit: "" }) {
    const row = document.createElement("div");
    row.className = "mapping-row";
    row.dataset.originalName = mapping.name || mapping.feature_name || "";
    row.innerHTML = `<input type="text" data-map="source" aria-label="Chiave Arduino" placeholder="es. temp" />
      <input type="text" data-map="feature" aria-label="Nome nel modello" placeholder="es. temperature_c" />
      <input type="text" data-map="unit" aria-label="Unità" placeholder="°C" />
      <button type="button" aria-label="Rimuovi mapping">×</button>`;
    $("[data-map=source]", row).value = mapping.source_key || "";
    $("[data-map=feature]", row).value = mapping.feature_name || "";
    $("[data-map=unit]", row).value = mapping.unit || "";
    $("button", row).addEventListener("click", () => {
      row.remove();
      markConfigDirty();
      updateProtocolExample();
    });
    row.addEventListener("input", () => { markConfigDirty(); updateProtocolExample(); });
    $("#sensor-mappings").append(row);
  }

  function collectMappings() {
    return $$(".mapping-row").map((row) => {
      const sourceKey = $("[data-map=source]", row).value.trim();
      const featureName = $("[data-map=feature]", row).value.trim();
      const existing = state.config.hardware.sensors.find((item) => (item.name || item.feature_name) === row.dataset.originalName) || {};
      return {
        ...existing,
        name: featureName,
        feature_name: featureName,
        label: existing.label || humanize(featureName),
        unit: $("[data-map=unit]", row).value.trim(),
        value_type: existing.value_type || "float",
        source_key: sourceKey,
        simulation_min: Number(existing.simulation_min ?? 0),
        simulation_max: Number(existing.simulation_max ?? 100),
        precision: Number(existing.precision ?? 2),
      };
    }).filter((item) => item.source_key && item.name);
  }

  function updateProtocolExample() {
    const entries = collectMappings().slice(0, 5);
    const object = Object.fromEntries(entries.map((item, index) => [item.source_key, [22.6, 48.2, .14, 1013, 320][index] ?? 0]));
    $("#protocol-example code").textContent = JSON.stringify(Object.keys(object).length ? object : { temp: 22.6, humidity: 48.2 }, null, 2);
  }

  function updateSourceSettings() {
    const source = $("input[name=" + CSS.escape("source-mode") + "]:checked")?.value || "simulation";
    $("#simulation-settings").hidden = source !== "simulation";
    $("#serial-settings").hidden = source !== "serial";
  }

  function markConfigDirty() { $("#config-dirty").hidden = false; }

  function collectConfig() {
    const source = $("input[name=" + CSS.escape("source-mode") + "]:checked").value;
    const hardware = {
      ...state.config.hardware,
      source,
      sampling_interval_seconds: Number($("#simulation-interval").value) / 1000,
      simulation_seed: Number($("#simulation-seed").value),
      allow_real_actuators: $("#hardware-output-enabled").checked,
      serial: {
        ...state.config.hardware.serial,
        port: $("#serial-port").value.trim(),
        baud_rate: Number($("#baud-rate").value),
        timeout_seconds: Number($("#serial-timeout").value),
        line_ending: $("#line-ending").value === "CRLF" ? "\r\n" : "\n",
      },
      sensors: collectMappings(),
    };
    return { ...state.config, hardware };
  }

  async function saveConfig(event) {
    event?.preventDefault();
    const config = collectConfig();
    if (config.hardware.source === "serial" && !config.hardware.serial.port) {
      toast("Porta seriale mancante", "Indica la porta Arduino, ad esempio COM3.", "error");
      return false;
    }
    if (!config.hardware.sensors.length) {
      toast("Mapping mancante", "Aggiungi almeno un sensore con chiave e nome modello.", "error");
      return false;
    }
    try {
      const payload = await api("/config", { method: "PUT", body: config });
      state.config = normalizeConfig(payload?.config || payload || config);
      state.savedConfig = structuredClone(state.config);
      applyConfigToForm();
      updateModeUI();
      toast("Configurazione salvata", "I nuovi parametri verranno usati dalla prossima connessione.");
      return true;
    } catch (error) {
      toast("Salvataggio non riuscito", error.message, "error");
      return false;
    }
  }

  function updateOutputUI() {
    const enabled = $("#hardware-output-enabled").checked;
    $(".switch em").textContent = enabled ? "Abilitato" : "Disattivato";
    $("#output-mode").textContent = enabled ? "SERIALE" : "DRY RUN";
    const banner = $("#actuator-safety-banner");
    $("strong", banner).textContent = enabled ? "Uscite fisiche abilitate." : "Modalità sicura: solo simulazione.";
    $("p", banner).textContent = enabled
      ? "Le regole attive possono inviare comandi sulla porta seriale configurata."
      : "Le regole vengono valutate e registrate, ma nessun comando fisico viene inviato.";
  }

  function updateModeUI() {
    const simulated = state.config.hardware.source !== "serial";
    $("#mode-badge").textContent = simulated ? "Modalità simulata" : "Arduino seriale";
    $("#sidebar-source").textContent = simulated ? "Simulatore pronto" : "Arduino configurato";
    updateOutputUI();
  }

  async function testHardware() {
    const result = $("#test-result");
    const button = $("#test-hardware");
    button.disabled = true;
    result.hidden = false;
    result.className = "test-result";
    result.textContent = "Connessione e lettura di prova in corso…";
    try {
      const saved = await saveConfig();
      if (!saved) throw new Error("Correggi la configurazione prima del test.");
      const payload = await api("/hardware/test", { method: "POST" });
      const reading = normalizeReading(payload);
      result.className = "test-result success";
      result.textContent = reading ? `Test riuscito: ${Object.keys(reading.values).length} sensori ricevuti.` : "Connessione riuscita.";
      toast("Test completato", "La sorgente dati risponde correttamente.");
    } catch (error) {
      result.className = "test-result error";
      result.textContent = error.message;
      toast("Test non riuscito", error.message, "error");
    } finally {
      button.disabled = false;
    }
  }

  async function loadRules() {
    const payload = await api("/actuators/rules");
    state.rules = extractArray(payload, ["rules", "items"]);
    renderRules();
  }

  function renderRules() {
    $("#rules-total").textContent = state.rules.length;
    $("#rules-enabled").textContent = state.rules.filter((rule) => rule.enabled).length;
    const list = $("#rules-list");
    if (!state.rules.length) {
      list.innerHTML = `<div class="large-empty"><span aria-hidden="true">⌘</span><strong>Nessuna regola configurata</strong><p>Crea la prima regola per collegare una previsione a un attuatore.</p><button class="button button-secondary" type="button" data-open-rule>+ Crea una regola</button></div>`;
      return;
    }
    list.innerHTML = state.rules.map((rule) => `<article class="rule-card${rule.enabled ? " is-enabled" : ""}">
      <span class="rule-state" title="${rule.enabled ? "Attiva" : "Disattivata"}"></span>
      <div><strong>${escapeHtml(rule.name)}</strong><p class="rule-expression">Se previsione <code>${escapeHtml(operatorLabels[rule.operator] || rule.operator)} ${escapeHtml(formatNumber(rule.threshold))}</code> → ${escapeHtml(rule.actuator)} · <code>${escapeHtml(rule.command_on || rule.command || "—")}</code></p></div>
      <div class="rule-actions"><button class="icon-button" type="button" data-toggle-rule="${escapeHtml(rule.id)}" aria-label="${rule.enabled ? "Disattiva" : "Attiva"} regola">${rule.enabled ? "Ⅱ" : "▶"}</button><button class="icon-button" type="button" data-edit-rule="${escapeHtml(rule.id)}" aria-label="Modifica regola">✎</button><button class="icon-button" type="button" data-delete-rule="${escapeHtml(rule.id)}" aria-label="Elimina regola">×</button></div>
    </article>`).join("");
  }

  function openRuleDialog(rule = null) {
    $("#rule-dialog-title").textContent = rule ? "Modifica regola" : "Nuova regola";
    $("#rule-id").value = rule?.id || "";
    $("#rule-name").value = rule?.name || "";
    $("#rule-operator").value = apiToFormOperator[rule?.operator] || ">";
    $("#rule-threshold").value = rule?.threshold ?? "";
    $("#actuator-name").value = rule?.actuator || "";
    $("#actuator-pin").value = rule?.pin || "";
    $("#actuator-command").value = rule?.command_on || rule?.command || "";
    $("#actuator-command-off").value = rule?.command_off || "";
    $("#rule-enabled").checked = rule?.enabled ?? true;
    $("#rule-dialog").showModal();
    $("#rule-name").focus();
  }

  async function persistRules(nextRules, successMessage) {
    try {
      const payload = await api("/actuators/rules", { method: "PUT", body: { rules: nextRules } });
      state.rules = extractArray(payload, ["rules", "items"]);
      if (!state.rules.length && nextRules.length) state.rules = nextRules;
      renderRules();
      toast(successMessage, "La configurazione è stata aggiornata.");
      return true;
    } catch (error) {
      toast("Aggiornamento non riuscito", error.message, "error");
      return false;
    }
  }

  async function saveRule(event) {
    event.preventDefault();
    if (!$("#rule-form").reportValidity()) return;
    const existingId = $("#rule-id").value;
    const thresholdRaw = $("#rule-threshold").value.trim();
    const thresholdNumber = Number(thresholdRaw);
    const rule = {
      id: existingId || crypto.randomUUID(),
      name: $("#rule-name").value.trim(),
      enabled: $("#rule-enabled").checked,
      actuator: $("#actuator-name").value.trim(),
      operator: formToApiOperator[$("#rule-operator").value],
      threshold: Number.isFinite(thresholdNumber) && thresholdRaw !== "" ? thresholdNumber : thresholdRaw,
      command_on: $("#actuator-command").value.trim(),
      command_off: $("#actuator-command-off").value.trim() || null,
      pin: $("#actuator-pin").value.trim() || null,
      cooldown_seconds: state.rules.find((item) => item.id === existingId)?.cooldown_seconds || 0,
    };
    const next = existingId ? state.rules.map((item) => item.id === existingId ? { ...item, ...rule } : item) : [...state.rules, rule];
    const success = await persistRules(next, existingId ? "Regola modificata" : "Regola creata");
    if (success) $("#rule-dialog").close();
  }

  async function handleRuleAction(button) {
    const toggleId = button.dataset.toggleRule;
    const editId = button.dataset.editRule;
    const deleteId = button.dataset.deleteRule;
    if (editId) return openRuleDialog(state.rules.find((rule) => rule.id === editId));
    if (toggleId) {
      const next = state.rules.map((rule) => rule.id === toggleId ? { ...rule, enabled: !rule.enabled } : rule);
      return persistRules(next, "Stato regola aggiornato");
    }
    if (deleteId && window.confirm("Eliminare definitivamente questa regola?")) {
      return persistRules(state.rules.filter((rule) => rule.id !== deleteId), "Regola eliminata");
    }
  }

  async function refreshCurrentPage() {
    const page = location.hash.replace("#", "") || "monitor";
    const button = $("#refresh-page");
    button.disabled = true;
    try {
      if (page === "monitor") await loadMonitorData();
      if (page === "training") await loadModels();
      if (page === "actuators") await Promise.all([loadRules(), loadEvents()]);
      if (page === "hardware") {
        const payload = await api("/config");
        state.config = normalizeConfig(payload);
        state.savedConfig = structuredClone(state.config);
        applyConfigToForm();
      }
      toast("Dati aggiornati");
    } catch (error) {
      toast("Aggiornamento non riuscito", error.message, "error");
    } finally { button.disabled = false; }
  }

  async function loadMonitorData() {
    const [historyPayload, latestPayload] = await Promise.all([
      api("/sensors/history?limit=60"),
      api("/sensors/latest").catch(() => null),
    ]);
    state.history = extractArray(historyPayload, ["readings", "history", "items"]).map(normalizeReading).filter(Boolean);
    state.latest = normalizeReading(latestPayload) || state.history.at(-1) || null;
    renderSensorCards();
    drawChart();
    renderPrediction();
  }

  async function loadEvents() {
    const payload = await api("/actuators/events");
    state.events = extractArray(payload, ["events", "items"]);
    renderEvents();
  }

  async function initialize() {
    $("#loading-overlay").hidden = false;
    navigate();
    bindEvents();
    const results = await Promise.allSettled([
      api("/health"),
      api("/config"),
      api("/models"),
      api("/actuators/rules"),
      api("/actuators/events"),
      api("/sensors/history?limit=60"),
      api("/sensors/status"),
      api("/predictions/latest"),
    ]);

    const [healthResult, configResult, modelResult, ruleResult, eventResult, historyResult, statusResult, predictionResult] = results;
    if (configResult.status === "fulfilled") state.config = normalizeConfig(configResult.value);
    state.savedConfig = structuredClone(state.config);
    state.models = modelResult.status === "fulfilled" ? extractArray(modelResult.value, ["models", "items"]) : [];
    state.rules = ruleResult.status === "fulfilled" ? extractArray(ruleResult.value, ["rules", "items"]) : [];
    state.events = eventResult.status === "fulfilled" ? extractArray(eventResult.value, ["events", "items"]) : [];
    state.history = historyResult.status === "fulfilled" ? extractArray(historyResult.value, ["readings", "history", "items"]).map(normalizeReading).filter(Boolean) : [];
    state.latest = state.history.at(-1) || null;
    if (state.latest && predictionResult.status === "fulfilled") state.latest.prediction = predictionResult.value?.prediction || null;
    const sensorStatus = statusResult.status === "fulfilled" ? statusResult.value : {};
    state.sensorConnected = Boolean(sensorStatus.connected || sensorStatus.running);

    applyConfigToForm();
    updateModeUI();
    renderModels();
    renderRules();
    renderEvents();
    renderSensorCards();
    renderPrediction();
    window.setTimeout(drawChart, 50);
    $("#loading-overlay").hidden = true;

    if (healthResult.status === "fulfilled") {
      setConnectionUI(state.sensorConnected ? "ok" : "pending", state.sensorConnected ? "Sensori connessi" : "Sistema pronto");
      $("#sidebar-sample").textContent = state.latest ? `Ultimo dato ${formatDate(state.latest.timestamp, true)}` : "In attesa di dati";
      if (state.sensorConnected) setStreaming(true);
    } else {
      setConnectionUI("error", "Backend non disponibile");
      toast("Avvio parziale", healthResult.reason?.message || "Alcuni servizi non rispondono.", "error");
    }
  }

  function bindEvents() {
    window.addEventListener("hashchange", navigate);
    window.addEventListener("resize", debounce(drawChart, 120));
    $("#mobile-menu").addEventListener("click", () => {
      const sidebar = $(".sidebar");
      const open = sidebar.classList.toggle("is-open");
      $("#mobile-menu").setAttribute("aria-expanded", String(open));
    });
    $("#refresh-page").addEventListener("click", refreshCurrentPage);
    $("#toggle-stream").addEventListener("click", toggleStreaming);
    $("#chart-sensor").addEventListener("change", (event) => { state.chartSensor = event.target.value; $("#chart-legend").textContent = `${sensorLabel(state.chartSensor)} ${sensorUnit(state.chartSensor)}`.trim(); drawChart(); });

    const uploadZone = $("#upload-zone");
    $("#dataset-file").addEventListener("change", (event) => uploadDataset(event.target.files[0]));
    ["dragenter", "dragover"].forEach((name) => uploadZone.addEventListener(name, (event) => { event.preventDefault(); uploadZone.classList.add("is-dragging"); }));
    ["dragleave", "drop"].forEach((name) => uploadZone.addEventListener(name, (event) => { event.preventDefault(); uploadZone.classList.remove("is-dragging"); }));
    uploadZone.addEventListener("drop", (event) => uploadDataset(event.dataTransfer.files[0]));
    $$('[data-load-demo]').forEach((button) => button.addEventListener("click", () => loadDemo(button.dataset.loadDemo, button)));
    $("#to-model-step").addEventListener("click", () => showTrainingStep(2));
    $$('[data-back-step]').forEach((button) => button.addEventListener("click", () => showTrainingStep(Number(button.dataset.backStep))));
    $$('input[name="task"]').forEach((input) => input.addEventListener("change", syncTaskAlgorithm));
    $$('input[name="algorithm"]').forEach((input) => input.addEventListener("change", syncAlgorithmOptions));
    $("#target-column").addEventListener("change", syncTargetFeatures);
    $("#test-size").addEventListener("input", (event) => { $("#test-size-output").textContent = `${event.target.value}%`; });
    $("#training-form").addEventListener("submit", trainModel);
    $("#new-experiment").addEventListener("click", () => { state.trainedModel = null; showTrainingStep(1); });
    $("#activate-trained-model").addEventListener("click", () => activateModel(state.trainedModel?.id));
    $("#model-list").addEventListener("click", (event) => { const button = event.target.closest("[data-activate-model]"); if (button) activateModel(button.dataset.activateModel); });

    $("#hardware-form").addEventListener("submit", saveConfig);
    $("#hardware-form").addEventListener("input", markConfigDirty);
    $$('input[name="source-mode"]').forEach((input) => input.addEventListener("change", updateSourceSettings));
    $("#add-mapping").addEventListener("click", () => { addMappingRow(); markConfigDirty(); });
    $("#reset-config").addEventListener("click", () => { state.config = structuredClone(state.savedConfig); applyConfigToForm(); });
    $("#hardware-output-enabled").addEventListener("change", (event) => {
      if (event.target.checked && !window.confirm("Abilitare l'invio reale dei comandi? Usa questa opzione solo dopo aver verificato cablaggio e attuatori.")) event.target.checked = false;
      updateOutputUI();
      markConfigDirty();
    });
    $("#test-hardware").addEventListener("click", testHardware);
    $("#copy-protocol").addEventListener("click", async () => {
      try { await navigator.clipboard.writeText($("#protocol-example code").textContent); toast("Esempio copiato"); }
      catch { toast("Copia non disponibile", "Seleziona manualmente il codice.", "error"); }
    });

    $("#new-rule").addEventListener("click", () => openRuleDialog());
    $("#rules-list").addEventListener("click", (event) => {
      if (event.target.closest("[data-open-rule]")) return openRuleDialog();
      const button = event.target.closest("[data-toggle-rule],[data-edit-rule],[data-delete-rule]");
      if (button) handleRuleAction(button);
    });
    $("#rule-form").addEventListener("submit", saveRule);
    $("#reload-rules").addEventListener("click", async () => { try { await loadRules(); toast("Regole aggiornate"); } catch (error) { toast("Aggiornamento non riuscito", error.message, "error"); } });
  }

  function debounce(callback, delay) {
    let timer;
    return (...args) => { clearTimeout(timer); timer = setTimeout(() => callback(...args), delay); };
  }

  function escapeHtml(value) {
    const node = document.createElement("div");
    node.textContent = value == null ? "" : String(value);
    return node.innerHTML;
  }

  initialize();
})();
