const $ = (id) => document.getElementById(id);
const LABELS = {
  cpu_usage: ["CPU usage", "%", 1], gpu_usage: ["GPU usage", "%", 1], ram_usage: ["RAM usage", "%", 1],
  cpu_freq_ghz: ["CPU frequency", " GHz", 0.05], cpu_temp_c: ["CPU temperature", " °C", 1],
  gpu_temp_c: ["GPU temperature", " °C", 1], disk_activity: ["Disk activity", "", 1],
  network_activity: ["Network activity", "", 1],
};
const PRESETS = {
  idle:   { cpu_usage: 5,  gpu_usage: 2,  ram_usage: 35, cpu_freq_ghz: 1.4, cpu_temp_c: 38, gpu_temp_c: 34, disk_activity: 3,  network_activity: 2 },
  office: { cpu_usage: 25, gpu_usage: 10, ram_usage: 50, cpu_freq_ghz: 2.6, cpu_temp_c: 48, gpu_temp_c: 38, disk_activity: 10, network_activity: 8 },
  gaming: { cpu_usage: 55, gpu_usage: 85, ram_usage: 65, cpu_freq_ghz: 3.9, cpu_temp_c: 68, gpu_temp_c: 72, disk_activity: 15, network_activity: 10 },
  render: { cpu_usage: 95, gpu_usage: 40, ram_usage: 75, cpu_freq_ghz: 4.5, cpu_temp_c: 84, gpu_temp_c: 55, disk_activity: 25, network_activity: 4 },
};
const PLOTS = {
  "Actual vs predicted": "actual_vs_predicted.png", "Model comparison": "model_comparison.png",
  "Feature importance": "feature_importance.png", "Feature vs power": "feature_vs_power.png",
  "Correlation": "correlation.png", "Error distribution": "residuals.png",
};
let META, timer;

async function init() {
  META = await (await fetch("/api/meta")).json();
  const metrics = await (await fetch("/api/metrics")).json();
  $("dataBadge").textContent = `Data: ${META.data_source} · ${META.n_rows} rows`;

  const form = $("form");
  META.features.forEach((f) => {
    const [label, unit, step] = LABELS[f] || [f, "", 1];
    const [lo, hi] = META.feature_ranges[f];
    const mn = Math.floor(lo), mx = Math.ceil(hi);
    const d = document.createElement("div"); d.className = "field";
    d.innerHTML = `<div class="top"><span>${label}</span><output id="o_${f}"></output></div>
      <input type="range" id="${f}" min="${mn}" max="${mx}" step="${step}">`;
    form.appendChild(d);
    const el = d.querySelector("input");
    el.value = META.feature_means[f];
    el.addEventListener("input", () => { show(f, unit); schedule(); });
    show(f, unit);
  });

  metrics.forEach((m) => $("model").add(new Option(m.model + (m.model === META.best_model ? " (best)" : ""), m.model)));
  $("model").value = META.best_model;
  $("model").onchange = $("hours").oninput = schedule;
  document.querySelectorAll("[data-p]").forEach((b) => (b.onclick = () => preset(b.dataset.p)));

  renderMetrics(metrics);
  const tabs = $("tabs");
  Object.keys(PLOTS).forEach((k, i) => {
    const b = document.createElement("button"); b.textContent = k;
    b.onclick = () => { $("plot").src = "/reports/" + PLOTS[k]; tabs.querySelectorAll("button").forEach((x) => x.classList.remove("on")); b.classList.add("on"); };
    tabs.appendChild(b); if (i === 0) b.click();
  });
  predict();
}

function show(f, unit) {
  const v = +$(f).value;
  $("o_" + f).textContent = (Number.isInteger(v) ? v : v.toFixed(2)) + unit;
}
function preset(name) {
  for (const [f, v] of Object.entries(PRESETS[name])) { $(f).value = v; show(f, LABELS[f][1]); }
  predict();
}
function schedule() { clearTimeout(timer); timer = setTimeout(predict, 120); }

async function predict() {
  const body = { model: $("model").value, hours_per_day: +$("hours").value || 0 };
  META.features.forEach((f) => (body[f] = +$(f).value));
  const r = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) { $("watts").textContent = "!"; return; }
  render(await r.json());
}

function render(d) {
  $("watts").textContent = d.predicted_watts.toFixed(1);
  const p = $("level"); p.textContent = d.level.label + " consumption"; p.className = "pill " + d.level.tone;
  $("kwhd").textContent = d.energy.kwh_per_day;
  $("kwhm").textContent = d.energy.kwh_per_month;
  $("co2").textContent = d.energy.co2_kg_per_month + " kg";
  $("cost").textContent = "₹" + d.energy.cost_inr_per_month;
  $("tips").innerHTML = d.tips.map((t) => `<li>${t}</li>`).join("");
  $("note").textContent = d.note;

  const mx = Math.max(...Object.values(d.all_models), 1);
  $("cmp").innerHTML = Object.entries(d.all_models).map(([n, v]) =>
    `<div><span class="name">${n}</span><span class="bar"><i style="width:${(v / mx) * 100}%"></i></span><span class="v">${v} W</span></div>`).join("");

  // gauge: green -> amber -> red across the dataset's observed range
  const c = $("gauge"), g = c.getContext("2d"), W = c.width, H = c.height;
  g.clearRect(0, 0, W, H);
  const grad = g.createLinearGradient(0, 0, W, 0);
  grad.addColorStop(0, "#1f8a70"); grad.addColorStop(.5, "#d99a1c"); grad.addColorStop(1, "#d1493f");
  g.fillStyle = grad; g.globalAlpha = .25; g.fillRect(0, 0, W, H); g.globalAlpha = 1;
  const t = Math.min(1, Math.max(0, (d.predicted_watts - META.target_min_w) / (META.target_max_w - META.target_min_w)));
  g.fillStyle = grad; g.fillRect(0, 0, W * t, H);
}

function renderMetrics(ms) {
  const best = META.best_model;
  const f = (x, n = 2) => (+x).toFixed(n);
  $("metrics").innerHTML = `<tr><th>Model</th><th>MAE (W)</th><th>MSE</th><th>RMSE (W)</th><th>R²</th><th>Train time (s)</th><th>Predict (ms / 1000 rows)</th></tr>` +
    ms.map((m) => `<tr class="${m.model === best ? "best" : ""}"><td>${m.model}</td><td>${f(m.MAE)}</td><td>${f(m.MSE)}</td><td>${f(m.RMSE)}</td><td>${f(m.R2, 4)}</td><td>${f(m.train_time_s)}</td><td>${f(m.predict_time_ms_per_1000, 1)}</td></tr>`).join("");
}
init();
