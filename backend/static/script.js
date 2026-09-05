(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  const areaSelect = $("#area");
  const itemSelect = $("#item");
  const yearInput = $("#year");
  const rainfallInput = $("#rainfall");
  const tempInput = $("#temp");
  const pesticidesInput = $("#pesticides");
  const autofillBtn = $("#autofill-btn");
  const formStatus = $("#form-status");

  const resultSubtitle = $("#result-subtitle");
  const resultValue = $("#result-value");
  const resultHgHa = $("#result-hgha");
  const resultBand = $("#result-band");
  const resultDelta = $("#result-delta");
  const stalkWrap = $("#stalk-wrap");

  const trendSubtitle = $("#trend-subtitle");
  const trendEmpty = $("#trend-empty");
  const importanceList = $("#importance-list");

  let currentLookup = null; // defaults for the selected area/item
  let trendChart = null;
  let predictTimer = null;

  const fmtNum = (n, digits = 2) =>
    Number(n).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });

  // ---------------------------------------------------------------------
  // Slider fill + output text
  // ---------------------------------------------------------------------
  function syncRangeFill(input) {
    const min = Number(input.min), max = Number(input.max), val = Number(input.value);
    const pct = ((val - min) / (max - min)) * 100;
    input.style.setProperty("--fill", `${pct}%`);
  }

  function updateOutputs() {
    $("#year-out").textContent = yearInput.value;
    $("#rainfall-out").textContent = `${Number(rainfallInput.value).toLocaleString()} mm/yr`;
    $("#temp-out").textContent = `${Number(tempInput.value).toFixed(1)} °C`;
    $("#pesticides-out").textContent = `${Number(pesticidesInput.value).toLocaleString()} tonnes`;
    [yearInput, rainfallInput, tempInput, pesticidesInput].forEach(syncRangeFill);
  }

  // ---------------------------------------------------------------------
  // Signature element: the grain stalk meter
  // ---------------------------------------------------------------------
  const KERNEL_ROWS = 11;

  function buildStalk() {
    const width = 108, height = 190;
    let kernels = "";
    const topY = 14, bottomY = height - 10;
    for (let i = 0; i < KERNEL_ROWS; i++) {
      const t = i / (KERNEL_ROWS - 1);
      const y = topY + t * (bottomY - topY);
      const spread = 8 + t * 20; // wider near the base
      kernels += `
        <ellipse class="kernel" data-i="${i}" cx="${width / 2 - spread}" cy="${y}" rx="7" ry="4.2" transform="rotate(-28 ${width / 2 - spread} ${y})"></ellipse>
        <ellipse class="kernel" data-i="${i}" cx="${width / 2 + spread}" cy="${y}" rx="7" ry="4.2" transform="rotate(28 ${width / 2 + spread} ${y})"></ellipse>`;
    }
    const svg = `
      <svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">
        <line class="stalk-stem" x1="${width / 2}" y1="${bottomY + 6}" x2="${width / 2}" y2="${height}"></line>
        ${kernels}
      </svg>`;
    stalkWrap.innerHTML = svg;
  }

  function setStalkFill(percent) {
    const kernels = stalkWrap.querySelectorAll(".kernel");
    const stem = stalkWrap.querySelector(".stalk-stem");
    const litCount = Math.round((percent / 100) * KERNEL_ROWS);
    kernels.forEach((k) => {
      const row = Number(k.dataset.i);
      // fill from the base (high row index) upward
      const litRow = row >= KERNEL_ROWS - litCount;
      k.style.transitionDelay = `${(KERNEL_ROWS - row) * 22}ms`;
      k.classList.toggle("lit", litRow);
    });
    if (stem) stem.classList.toggle("lit", percent > 4);
  }

  // ---------------------------------------------------------------------
  // Load static options (areas, items, feature importance)
  // ---------------------------------------------------------------------
  async function loadOptions() {
    const res = await fetch("/api/options");
    const data = await res.json();

    data.areas.forEach((a) => {
      const opt = document.createElement("option");
      opt.value = a; opt.textContent = a;
      areaSelect.appendChild(opt);
    });
    data.items.forEach((i) => {
      const opt = document.createElement("option");
      opt.value = i; opt.textContent = i;
      itemSelect.appendChild(opt);
    });

    renderImportance(data.feature_importance, data.feature_labels);

    // Sensible first impression
    if (data.areas.includes("India")) areaSelect.value = "India";
    if (data.items.includes("Wheat")) itemSelect.value = "Wheat";
  }

  function renderImportance(items, labels) {
    importanceList.innerHTML = "";
    const max = Math.max(...items.map((i) => i.value));
    items.forEach((it) => {
      const row = document.createElement("div");
      row.className = "imp-row";
      const label = labels[it.label] || it.label;
      row.innerHTML = `
        <span class="imp-label">${label}</span>
        <span class="imp-track"><span class="imp-fill" data-target="${(it.value / max) * 100}"></span></span>
        <span class="imp-value">${(it.value * 100).toFixed(1)}%</span>`;
      importanceList.appendChild(row);
    });

    // animate on scroll into view
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.querySelectorAll(".imp-fill").forEach((f) => {
            f.style.width = `${f.dataset.target}%`;
          });
          observer.disconnect();
        }
      });
    }, { threshold: 0.2 });
    observer.observe(importanceList);
  }

  // ---------------------------------------------------------------------
  // Defaults (auto-fill) + history chart on area/item change
  // ---------------------------------------------------------------------
  async function onAreaItemChange() {
    const area = areaSelect.value, item = itemSelect.value;
    if (!area || !item) return;

    formStatus.textContent = "Loading regional records…";
    formStatus.classList.remove("active");

    const [defaultsRes, historyRes] = await Promise.all([
      fetch(`/api/defaults?area=${encodeURIComponent(area)}&item=${encodeURIComponent(item)}`),
      fetch(`/api/history?area=${encodeURIComponent(area)}&item=${encodeURIComponent(item)}`),
    ]);
    const defaultsData = await defaultsRes.json();
    const historyData = await historyRes.json();

    if (defaultsData.found) {
      currentLookup = defaultsData;
      autofillBtn.disabled = false;
      formStatus.textContent = `${historyData.years.length} recorded seasons found for ${item} in ${area}. Adjust the sliders or auto-fill below.`;
      formStatus.classList.add("active");
      applyDefaults(defaultsData, { silent: true });
    } else {
      currentLookup = null;
      autofillBtn.disabled = true;
      formStatus.textContent = `No historical record for ${item} in ${area} — set conditions manually.`;
    }

    trendSubtitle.textContent = `Recorded yields for ${item} in ${area}, 1990–2013.`;
    schedulePredict(true);
    try {
      renderTrendChart(historyData);
    } catch (err) {
      trendEmpty.textContent = "Chart couldn't load, but forecasting still works.";
      trendEmpty.style.display = "flex";
    }
  }

  function applyDefaults(d, { silent = false } = {}) {
    rainfallInput.value = Math.min(d.avg_rainfall, Number(rainfallInput.max));
    tempInput.value = Math.max(Math.min(d.avg_temp, Number(tempInput.max)), Number(tempInput.min));
    pesticidesInput.value = Math.min(d.avg_pesticides, Number(pesticidesInput.max));
    updateOutputs();
    if (!silent) schedulePredict(true);
  }

  // ---------------------------------------------------------------------
  // Trend chart
  // ---------------------------------------------------------------------
  function renderTrendChart(historyData) {
    const canvas = $("#trend-chart");
    if (!historyData.years.length) {
      trendEmpty.style.display = "flex";
      canvas.style.display = "none";
      if (trendChart) { trendChart.destroy(); trendChart = null; }
      return;
    }
    trendEmpty.style.display = "none";
    canvas.style.display = "block";

    const ctx = canvas.getContext("2d");
    const config = {
      type: "line",
      data: {
        labels: historyData.years,
        datasets: [
          {
            label: "Recorded yield (hg/ha)",
            data: historyData.values,
            borderColor: "#4B7A50",
            backgroundColor: "rgba(75, 122, 80, 0.12)",
            borderWidth: 2,
            pointRadius: 2.5,
            pointBackgroundColor: "#4B7A50",
            fill: true,
            tension: 0.32,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 700, easing: "easeOutQuart" },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#1D2317",
            titleFont: { family: "IBM Plex Mono" },
            bodyFont: { family: "IBM Plex Mono" },
            padding: 10,
            cornerRadius: 8,
          },
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: "#57604C", font: { family: "IBM Plex Mono", size: 10 } } },
          y: {
            grid: { color: "#D9DCC5" },
            ticks: { color: "#57604C", font: { family: "IBM Plex Mono", size: 10 } },
          },
        },
      },
    };

    if (trendChart) {
      trendChart.data = config.data;
      trendChart.update();
    } else {
      trendChart = new Chart(ctx, config);
    }
  }

  function markPredictionOnChart(year, valueHgHa) {
    if (!trendChart) return;
    const datasets = trendChart.data.datasets.filter((d) => d.label !== "Your forecast");
    datasets.push({
      label: "Your forecast",
      data: trendChart.data.labels.map((y) => (Number(y) === Number(year) ? valueHgHa : null)),
      borderColor: "#C6982F",
      backgroundColor: "#C6982F",
      pointRadius: 7,
      pointHoverRadius: 8,
      pointStyle: "star",
      showLine: false,
    });
    // If the chosen year isn't in the labels, add it
    if (!trendChart.data.labels.includes(year)) {
      trendChart.data.labels.push(year);
      trendChart.data.labels.sort((a, b) => a - b);
      datasets.forEach((ds) => {
        while (ds.data.length < trendChart.data.labels.length) ds.data.push(null);
      });
      const idx = trendChart.data.labels.indexOf(year);
      const forecastDs = datasets.find((d) => d.label === "Your forecast");
      forecastDs.data = trendChart.data.labels.map((y) => (Number(y) === Number(year) ? valueHgHa : null));
    }
    trendChart.data.datasets = datasets;
    trendChart.update();
  }

  // ---------------------------------------------------------------------
  // Explainable AI (XAI) Attribution Renderer
  // ---------------------------------------------------------------------
  const xaiList = $("#xai-list");

  function renderXAI(xaiBreakdown) {
    if (!xaiList || !xaiBreakdown || !xaiBreakdown.length) return;
    xaiList.innerHTML = "";

    const maxAbs = Math.max(...xaiBreakdown.map((item) => Math.abs(item.impact_t_ha) || 0.1));

    xaiBreakdown.forEach((item) => {
      const row = document.createElement("div");
      row.className = "xai-item";

      const isBase = item.type === "base";
      const isPos = item.impact_t_ha >= 0;
      const pct = Math.min(100, Math.max(8, (Math.abs(item.impact_t_ha) / maxAbs) * 100));

      const barClass = isBase ? "base" : isPos ? "pos" : "neg";
      const valClass = isBase ? "" : isPos ? "pos" : "neg";
      const sign = isBase ? "" : isPos ? "+" : "";

      row.innerHTML = `
        <span class="xai-label" title="${item.feature}">${item.feature}</span>
        <div class="xai-bar-wrap">
          <div class="xai-bar ${barClass}" style="width: ${pct}%"></div>
        </div>
        <span class="xai-val ${valClass}">${sign}${fmtNum(item.impact_t_ha, 2)} t/ha</span>
      `;
      xaiList.appendChild(row);
    });
  }

  // ---------------------------------------------------------------------
  // Climate Scenario Presets
  // ---------------------------------------------------------------------
  function initClimateScenarios() {
    const scenarioBtns = document.querySelectorAll(".scenario-btn");
    scenarioBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const scenario = btn.dataset.scenario;
        if (!currentLookup) return;

        const baseTemp = currentLookup.avg_temp || 20.0;
        const baseRain = currentLookup.avg_rainfall || 1000.0;

        if (scenario === "ipcc15") {
          tempInput.value = Math.min(Number(tempInput.max), baseTemp + 1.5);
          rainfallInput.value = Math.max(0, baseRain * 0.95);
        } else if (scenario === "drought") {
          tempInput.value = Math.min(Number(tempInput.max), baseTemp + 3.5);
          rainfallInput.value = Math.max(0, baseRain * 0.70);
        } else if (scenario === "monsoon") {
          tempInput.value = Math.max(Number(tempInput.min), baseTemp - 1.0);
          rainfallInput.value = Math.min(Number(rainfallInput.max), baseRain * 1.35);
        } else if (scenario === "optimal") {
          tempInput.value = baseTemp;
          rainfallInput.value = Math.min(Number(rainfallInput.max), baseRain * 1.10);
        }

        updateOutputs();
        schedulePredict(true);
      });
    });
  }

  // ---------------------------------------------------------------------
  // Global Leaflet Map Visualizer
  // ---------------------------------------------------------------------
  let mapInstance = null;
  let mapMarkers = [];

  const COUNTRY_COORDS = {
    India: [20.5937, 78.9629],
    Brazil: [-14.235, -51.9253],
    "United States of America": [37.0902, -95.7129],
    France: [46.2276, 2.2137],
    Australia: [-25.2744, 133.7751],
    Egypt: [26.8206, 30.8025],
    Germany: [51.1657, 10.4515],
    China: [35.8617, 104.1954],
    Argentina: [-38.4161, -63.6167],
    Canada: [56.1304, -106.3468],
    Chile: [-35.6751, -71.543],
    Spain: [40.4637, -3.7492],
    Italy: [41.8719, 12.5674],
    Japan: [36.2048, 138.2529],
    Mexico: [23.6345, -102.5528],
    "South Africa": [-30.5595, 22.9375],
    Turkey: [38.9637, 35.2433],
    "United Kingdom": [55.3781, -3.436],
    Vietnam: [14.0583, 108.2772],
    Indonesia: [-0.7893, 113.9213],
    Pakistan: [30.3753, 69.3451],
    Thailand: [15.87, 100.9925],
    Nigeria: [9.082, 8.6753],
    Ethiopia: [9.145, 40.4897],
    Kenya: [-0.0236, 37.9062],
    Morocco: [31.7917, -7.0926],
    Algeria: [28.0339, 1.6596],
    Poland: [51.9194, 19.1451],
    Romania: [45.9432, 24.9668],
    Ukraine: [48.3794, 31.1656],
    Hungary: [47.1625, 19.5033],
    Colombia: [4.5709, -74.2973],
    Peru: [-9.19, -75.0152],
    Ecuador: [-1.8312, -78.1834],
    Uruguay: [-32.5228, -55.7658],
    Greece: [39.0742, 21.8243],
    Portugal: [39.3999, -8.2245],
    Sweden: [60.1282, 18.6435],
    Denmark: [56.2639, 9.5018],
    Belgium: [50.5039, 4.4699],
  };

  function initMap() {
    const mapEl = $("#map");
    if (!mapEl || typeof L === "undefined") return;

    mapInstance = L.map("map", {
      center: [20, 10],
      zoom: 2,
      zoomControl: true,
      scrollWheelZoom: false,
    });

    L.tileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png", {
      attribution: "&copy; OpenStreetMap &copy; CARTO",
      subdomains: "abcd",
      maxZoom: 8,
    }).addTo(mapInstance);
  }

  async function updateMap(item) {
    if (!mapInstance) return;

    mapMarkers.forEach((m) => mapInstance.removeLayer(m));
    mapMarkers = [];

    try {
      const res = await fetch(`/api/map_data?item=${encodeURIComponent(item)}`);
      const data = await res.json();

      Object.entries(data.country_yields).forEach(([country, yieldVal]) => {
        const coords = COUNTRY_COORDS[country];
        if (coords) {
          const radius = Math.min(22, Math.max(6, yieldVal * 2.2));
          const marker = L.circleMarker(coords, {
            radius: radius,
            fillColor: "#C6982F",
            color: "#345439",
            weight: 1.5,
            opacity: 0.9,
            fillOpacity: 0.65,
          });

          marker.bindPopup(`
            <div style="font-family: var(--font-body); padding: 4px;">
              <strong style="font-size: 0.95rem; color: #232A1E;">${country}</strong><br>
              <span style="font-size: 0.84rem; color: #57604C;">Crop: ${item}</span><br>
              <strong style="font-size: 1.05rem; color: #9C7621;">${yieldVal} t/ha</strong>
            </div>
          `);

          marker.addTo(mapInstance);
          mapMarkers.push(marker);
        }
      });
    } catch (err) {
      console.warn("Map data update failed:", err);
    }
  }

  // ---------------------------------------------------------------------
  // Model Evaluation Dashboard (Scatter Chart)
  // ---------------------------------------------------------------------
  let evalScatterChart = null;

  async function initEvalDashboard() {
    const canvas = $("#eval-scatter-chart");
    if (!canvas) return;

    try {
      const res = await fetch("/api/evaluation");
      const data = await res.json();

      const scatterData = data.scatter.map((pt) => ({ x: pt.actual, y: pt.predicted, area: pt.area }));

      const ctx = canvas.getContext("2d");
      evalScatterChart = new Chart(ctx, {
        type: "scatter",
        data: {
          datasets: [
            {
              label: "Actual vs. Predicted Yield (t/ha)",
              data: scatterData,
              backgroundColor: "rgba(198, 152, 47, 0.75)",
              borderColor: "#9C7621",
              borderWidth: 1,
              pointRadius: 5,
              pointHoverRadius: 7,
            },
            {
              label: "Ideal 1:1 Reference",
              data: [
                { x: 0, y: 0 },
                { x: 12, y: 12 },
              ],
              type: "line",
              borderColor: "rgba(75, 122, 80, 0.5)",
              borderDash: [5, 5],
              borderWidth: 1.5,
              pointRadius: 0,
              fill: false,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const pt = ctx.raw;
                  if (!pt.area) return "";
                  return `${pt.area}: Actual ${pt.x} t/ha | Predicted ${pt.y} t/ha`;
                },
              },
            },
          },
          scales: {
            x: {
              title: { display: true, text: "Actual Yield (t/ha)", color: "#57604C", font: { size: 11 } },
              grid: { color: "#D9DCC5" },
              ticks: { color: "#57604C", font: { size: 10 } },
            },
            y: {
              title: { display: true, text: "Predicted Yield (t/ha)", color: "#57604C", font: { size: 11 } },
              grid: { color: "#D9DCC5" },
              ticks: { color: "#57604C", font: { size: 10 } },
            },
          },
        },
      });
    } catch (err) {
      console.warn("Evaluation scatter chart error:", err);
    }
  }

  // ---------------------------------------------------------------------
  // Prediction Execution
  // ---------------------------------------------------------------------
  function schedulePredict(immediate = false) {
    clearTimeout(predictTimer);
    if (immediate) return runPredict();
    predictTimer = setTimeout(runPredict, 260);
  }

  async function runPredict() {
    const area = areaSelect.value, item = itemSelect.value;
    if (!area || !item) return;

    const payload = {
      area, item,
      year: Number(yearInput.value),
      rainfall: Number(rainfallInput.value),
      pesticides: Number(pesticidesInput.value),
      temp: Number(tempInput.value),
    };

    resultValue.style.opacity = "0.4";

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("prediction failed");
      const data = await res.json();
      renderResult(area, item, payload.year, data);
      if (data.xai_breakdown) renderXAI(data.xai_breakdown);
      updateMap(item);
    } catch (err) {
      resultSubtitle.textContent = "Couldn't reach the model — try again.";
    } finally {
      resultValue.style.opacity = "1";
    }
  }

  function renderResult(area, item, year, data) {
    resultSubtitle.textContent = `${item} in ${area}, season ${year}`;
    resultValue.textContent = fmtNum(data.prediction_t_ha, 2);
    resultHgHa.textContent = `${fmtNum(data.prediction_hg_ha, 0)} hg/ha`;
    resultBand.textContent = `${fmtNum(data.band_t_ha[0], 2)} – ${fmtNum(data.band_t_ha[1], 2)} t/ha`;

    if (data.delta_pct === null || data.delta_pct === undefined) {
      resultDelta.textContent = "no regional record";
      resultDelta.className = "meta-value";
    } else {
      const sign = data.delta_pct >= 0 ? "+" : "";
      resultDelta.textContent = `${sign}${fmtNum(data.delta_pct, 1)}%`;
      resultDelta.className = "meta-value " + (data.delta_pct >= 0 ? "up" : "down");
    }

    // Stalk percent: relative to this crop/region's own historical range
    let percent = 55;
    if (currentLookup && currentLookup.max_yield > currentLookup.min_yield) {
      percent = ((data.prediction_hg_ha - currentLookup.min_yield) /
        (currentLookup.max_yield - currentLookup.min_yield)) * 100;
      percent = Math.max(4, Math.min(100, percent));
    }
    setStalkFill(percent);

    markPredictionOnChart(year, data.prediction_hg_ha);
  }

  // ---------------------------------------------------------------------
  // Wire up events
  // ---------------------------------------------------------------------
  function init() {
    buildStalk();
    updateOutputs();
    initClimateScenarios();

    [yearInput, rainfallInput, tempInput, pesticidesInput].forEach((input) => {
      input.addEventListener("input", () => {
        updateOutputs();
        schedulePredict(false);
      });
    });

    areaSelect.addEventListener("change", onAreaItemChange);
    itemSelect.addEventListener("change", onAreaItemChange);
    autofillBtn.addEventListener("click", () => {
      if (currentLookup) applyDefaults(currentLookup);
    });

    initMap();
    initEvalDashboard();

    loadOptions().then(() => {
      onAreaItemChange();
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();

