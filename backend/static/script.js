(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  // Form Inputs
  const areaSelect = $("#area");
  const itemSelect = $("#item");
  const yearInput = $("#year");
  const rainfallInput = $("#rainfall");
  const tempInput = $("#temp");
  const pesticidesInput = $("#pesticides");
  const autofillBtn = $("#autofill-btn");
  const formStatus = $("#form-status");

  // Forecast Outputs
  const resultSubtitle = $("#result-subtitle");
  const resultValue = $("#result-value");
  const resultUnitLabel = $("#result-unit-label");
  const resultHgHa = $("#result-hgha");
  const resultBand = $("#result-band");
  const resultDelta = $("#result-delta");
  const stalkWrap = $("#stalk-wrap");

  // Equivalencies outputs
  const eqQHa = $("#eq-q-ha");
  const eqQAcre = $("#eq-q-acre");
  const eqTHa = $("#eq-t-ha");
  const eqKgAcre = $("#eq-kg-acre");

  // Trend & Importance
  const trendSubtitle = $("#trend-subtitle");
  const trendEmpty = $("#trend-empty");
  const importanceList = $("#importance-list");

  // State
  let currentLookup = null;
  let lastPredictionData = null;
  let currentUnit = "q_ha"; // 'q_ha', 't_ha', 'q_acre', 'kg_acre', 'hg_ha'
  let trendChart = null;
  let predictTimer = null;
  let indiaGlobalChart = null;
  let scenarioCompareChart = null;
  let modelComparisonChart = null;
  let featureImpChart = null;
  let evalScatterChart = null;
  let mapInstance = null;
  let mapMarkers = [];
  let indiaDataCache = null;
  let evalDataCache = null;

  const fmtNum = (n, digits = 2) =>
    Number(n).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });

  // ---------------------------------------------------------------------
  // Unit Conversion Helpers
  // ---------------------------------------------------------------------
  const UNIT_NAMES = {
    q_ha: "quintals / hectare",
    t_ha: "tonnes / hectare",
    q_acre: "quintals / acre",
    kg_acre: "kg / acre",
    hg_ha: "hg / ha",
  };

  function convertYield(t_ha, targetUnit) {
    if (t_ha === null || t_ha === undefined || isNaN(t_ha)) return 0;
    switch (targetUnit) {
      case "q_ha":
        return t_ha * 10.0;
      case "t_ha":
        return t_ha;
      case "q_acre":
        return (t_ha * 10.0) / 2.47105;
      case "kg_acre":
        return (t_ha * 1000.0) / 2.47105;
      case "hg_ha":
        return t_ha * 10000.0;
      default:
        return t_ha;
    }
  }

  function updateDisplayedYield() {
    if (!lastPredictionData) return;
    const t_ha = lastPredictionData.prediction_t_ha;
    const bandLow_tha = lastPredictionData.band_t_ha[0];
    const bandHigh_tha = lastPredictionData.band_t_ha[1];

    const val = convertYield(t_ha, currentUnit);
    const bandLow = convertYield(bandLow_tha, currentUnit);
    const bandHigh = convertYield(bandHigh_tha, currentUnit);

    const digits = currentUnit === "kg_acre" || currentUnit === "hg_ha" ? 0 : 2;
    resultValue.textContent = fmtNum(val, digits);
    resultUnitLabel.textContent = UNIT_NAMES[currentUnit] || currentUnit;
    resultBand.textContent = `${fmtNum(bandLow, digits)} – ${fmtNum(bandHigh, digits)} ${currentUnit.replace("_", "/")}`;

    // Update Equivalencies
    if (eqQHa) eqQHa.textContent = `${fmtNum(t_ha * 10.0, 1)} q/ha`;
    if (eqQAcre) eqQAcre.textContent = `${fmtNum((t_ha * 10.0) / 2.47105, 1)} q/acre`;
    if (eqTHa) eqTHa.textContent = `${fmtNum(t_ha, 2)} t/ha`;
    if (eqKgAcre) eqKgAcre.textContent = `${fmtNum((t_ha * 1000.0) / 2.47105, 0)} kg/acre`;
  }

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
      const spread = 8 + t * 20;
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
      const litRow = row >= KERNEL_ROWS - litCount;
      k.style.transitionDelay = `${(KERNEL_ROWS - row) * 22}ms`;
      k.classList.toggle("lit", litRow);
    });
    if (stem) stem.classList.toggle("lit", percent > 4);
  }

  // ---------------------------------------------------------------------
  // Load static options
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

    // Sensible first impression: India & Wheat
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
  // Defaults & History Chart
  // ---------------------------------------------------------------------
  async function onAreaItemChange() {
    const area = areaSelect.value, item = itemSelect.value;
    if (!area || !item) return;

    formStatus.textContent = `Loading historical records for ${item} in ${area}…`;
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
      formStatus.textContent = `Baseline loaded for ${area}: ${fmtNum(defaultsData.avg_yield / 1000, 1)} q/ha (${fmtNum(defaultsData.avg_rainfall, 0)} mm rain, ${fmtNum(defaultsData.avg_temp, 1)} °C).`;
      formStatus.classList.add("active");
    } else {
      currentLookup = null;
      autofillBtn.disabled = true;
      formStatus.textContent = `No historical baseline found for ${item} in ${area} — enter custom conditions.`;
    }

    renderHistoryChart(historyData, area, item);
    schedulePredict(true);
  }

  function applyDefaults(lookup) {
    if (!lookup) return;
    yearInput.value = lookup.max_year || 2013;
    rainfallInput.value = Math.round(lookup.avg_rainfall);
    tempInput.value = Number(lookup.avg_temp).toFixed(1);
    pesticidesInput.value = Math.round(lookup.avg_pesticides);
    updateOutputs();
    schedulePredict(true);
  }

  function renderHistoryChart(historyData, area, item) {
    const canvas = $("#trend-chart");
    if (!canvas || typeof Chart === "undefined") return;

    if (!historyData.years || !historyData.years.length) {
      trendSubtitle.textContent = `No recorded harvests in the FAO ledger for ${item} in ${area}.`;
      trendEmpty.style.display = "block";
      canvas.style.display = "none";
      if (trendChart) { trendChart.destroy(); trendChart = null; }
      return;
    }

    trendSubtitle.textContent = `Recorded harvests for ${item} in ${area} across ${historyData.years.length} recorded seasons (1990–2013).`;
    trendEmpty.style.display = "none";
    canvas.style.display = "block";

    const ctx = canvas.getContext("2d");
    const convertedValues = historyData.values.map((v) => convertYield(v / 10000.0, currentUnit));

    const config = {
      type: "line",
      data: {
        labels: historyData.years,
        datasets: [
          {
            label: `Recorded yield (${currentUnit.replace("_", "/")})`,
            data: convertedValues,
            borderColor: "#4B7A50",
            backgroundColor: "rgba(75, 122, 80, 0.12)",
            borderWidth: 2,
            pointRadius: 3,
            pointBackgroundColor: "#4B7A50",
            fill: true,
            tension: 0.32,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#1D2317",
            titleFont: { family: "IBM Plex Mono" },
            bodyFont: { family: "IBM Plex Mono" },
            padding: 10,
            cornerRadius: 8,
            callbacks: {
              label: (ctx) => `${fmtNum(ctx.raw, 2)} ${currentUnit.replace("_", "/")}`,
            },
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

  function markPredictionOnChart(year, t_ha) {
    if (!trendChart) return;
    const valueInUnit = convertYield(t_ha, currentUnit);
    const datasets = trendChart.data.datasets.filter((d) => d.label !== "Your forecast");
    datasets.push({
      label: "Your forecast",
      data: trendChart.data.labels.map((y) => (Number(y) === Number(year) ? valueInUnit : null)),
      borderColor: "#C6982F",
      backgroundColor: "#C6982F",
      pointRadius: 8,
      pointHoverRadius: 9,
      pointStyle: "star",
      showLine: false,
    });
    if (!trendChart.data.labels.includes(year)) {
      trendChart.data.labels.push(year);
      trendChart.data.labels.sort((a, b) => a - b);
      datasets.forEach((ds) => {
        while (ds.data.length < trendChart.data.labels.length) ds.data.push(null);
      });
      const forecastDs = datasets.find((d) => d.label === "Your forecast");
      forecastDs.data = trendChart.data.labels.map((y) => (Number(y) === Number(year) ? valueInUnit : null));
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

      const impactInUnit = convertYield(item.impact_t_ha, currentUnit);
      const digits = currentUnit === "kg_acre" || currentUnit === "hg_ha" ? 0 : 2;

      row.innerHTML = `
        <span class="xai-label" title="${item.feature}">${item.feature}</span>
        <div class="xai-bar-wrap">
          <div class="xai-bar ${barClass}" style="width: ${pct}%"></div>
        </div>
        <span class="xai-val ${valClass}">${sign}${fmtNum(impactInUnit, digits)} ${currentUnit.replace("_", "/")}</span>
      `;
      xaiList.appendChild(row);
    });
  }

  // ---------------------------------------------------------------------
  // Presets (Indian Seasons & Agro-Climatic Zones)
  // ---------------------------------------------------------------------
  function initPresets() {
    const presetBtns = $$(".preset-btn");
    presetBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const p = btn.dataset.preset;
        areaSelect.value = "India";

        if (p === "kharif") {
          if (!["Rice, paddy", "Maize", "Soybeans", "Sorghum"].includes(itemSelect.value)) {
            itemSelect.value = "Rice, paddy";
          }
          tempInput.value = 28.0;
          rainfallInput.value = 1250;
          pesticidesInput.value = 48000;
        } else if (p === "rabi") {
          itemSelect.value = "Wheat";
          tempInput.value = 19.5;
          rainfallInput.value = 250;
          pesticidesInput.value = 45000;
        } else if (p === "zaid") {
          itemSelect.value = "Maize";
          tempInput.value = 33.0;
          rainfallInput.value = 100;
          pesticidesInput.value = 40000;
        } else if (p === "gangetic") {
          itemSelect.value = "Wheat";
          tempInput.value = 21.5;
          rainfallInput.value = 650;
          pesticidesInput.value = 55000;
        } else if (p === "eastern") {
          itemSelect.value = "Rice, paddy";
          tempInput.value = 27.0;
          rainfallInput.value = 1450;
          pesticidesInput.value = 42000;
        } else if (p === "deccan") {
          itemSelect.value = "Soybeans";
          tempInput.value = 26.5;
          rainfallInput.value = 850;
          pesticidesInput.value = 40000;
        } else if (p === "south") {
          itemSelect.value = "Cassava";
          tempInput.value = 28.5;
          rainfallInput.value = 1150;
          pesticidesInput.value = 50000;
        }

        updateOutputs();
        schedulePredict(true);
      });
    });

    // Climate Scenario Simulator Presets
    const scenarioBtns = $$(".scenario-btn");
    scenarioBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const scenario = btn.dataset.scenario;
        const baseTemp = currentLookup ? currentLookup.avg_temp : 26.0;
        const baseRain = currentLookup ? currentLookup.avg_rainfall : 1083.0;

        if (scenario === "ipcc15") {
          tempInput.value = Math.min(Number(tempInput.max), Number(baseTemp) + 1.5);
          rainfallInput.value = Math.max(0, Math.round(baseRain * 0.95));
        } else if (scenario === "drought") {
          tempInput.value = Math.min(Number(tempInput.max), Number(baseTemp) + 3.5);
          rainfallInput.value = Math.max(0, Math.round(baseRain * 0.70));
        } else if (scenario === "monsoon") {
          tempInput.value = Math.max(Number(tempInput.min), Number(baseTemp) - 1.0);
          rainfallInput.value = Math.min(Number(rainfallInput.max), Math.round(baseRain * 1.35));
        } else if (scenario === "optimal") {
          tempInput.value = baseTemp;
          rainfallInput.value = Math.min(Number(rainfallInput.max), Math.round(baseRain * 1.10));
        }

        updateOutputs();
        schedulePredict(true);
      });
    });

    // Unit switcher buttons
    const unitBtns = $$(".unit-btn");
    unitBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        unitBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        currentUnit = btn.dataset.unit;
        updateDisplayedYield();
        if (lastPredictionData) {
          markPredictionOnChart(yearInput.value, lastPredictionData.prediction_t_ha);
          if (lastPredictionData.xai_breakdown) renderXAI(lastPredictionData.xai_breakdown);
        }
      });
    });
  }

  // ---------------------------------------------------------------------
  // Prediction Execution & Scenario Simulator Comparison
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
      const [predRes, scenRes] = await Promise.all([
        fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }),
        fetch("/api/scenarios_compare", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }),
      ]);

      if (!predRes.ok) throw new Error("Prediction API call failed");
      const predData = await predRes.json();
      lastPredictionData = predData;

      renderResult(area, item, payload.year, predData);
      if (predData.xai_breakdown) renderXAI(predData.xai_breakdown);
      updateMap(item);

      if (scenRes.ok) {
        const scenData = await scenRes.json();
        renderScenarioComparison(scenData);
      }
    } catch (err) {
      resultSubtitle.textContent = "Couldn't reach the model pipeline — try again.";
      console.warn("Prediction error:", err);
    } finally {
      resultValue.style.opacity = "1";
    }
  }

  function renderResult(area, item, year, data) {
    resultSubtitle.textContent = `${item} in ${area}, Season ${year}`;
    resultHgHa.textContent = `${fmtNum(data.prediction_hg_ha, 0)} hg/ha`;

    if (data.delta_pct === null || data.delta_pct === undefined) {
      resultDelta.textContent = "no regional baseline record";
      resultDelta.className = "meta-value";
    } else {
      const sign = data.delta_pct >= 0 ? "+" : "";
      resultDelta.textContent = `${sign}${fmtNum(data.delta_pct, 1)}%`;
      resultDelta.className = "meta-value " + (data.delta_pct >= 0 ? "up" : "down");
    }

    // Stalk percent: relative to region's historical range
    let percent = 55;
    if (currentLookup && currentLookup.max_yield > currentLookup.min_yield) {
      percent = ((data.prediction_hg_ha - currentLookup.min_yield) /
        (currentLookup.max_yield - currentLookup.min_yield)) * 100;
      percent = Math.max(4, Math.min(100, percent));
    }
    setStalkFill(percent);

    updateDisplayedYield();
    markPredictionOnChart(year, data.prediction_t_ha);
  }

  // ---------------------------------------------------------------------
  // Scenario Stress Comparison Chart & Cards
  // ---------------------------------------------------------------------
  function renderScenarioComparison(scenData) {
    const canvas = $("#scenario-compare-chart");
    const container = $("#scenario-cards-container");
    const titleEl = $("#scenario-chart-title");
    if (!canvas || !scenData || !scenData.scenarios) return;

    if (titleEl) {
      titleEl.textContent = `Climate Stress Simulation: ${scenData.item} in ${scenData.area} (${scenData.year})`;
    }

    const labels = scenData.scenarios.map((s) => s.name);
    const yields = scenData.scenarios.map((s) => convertYield(s.yield_t_ha, currentUnit));
    const colors = [
      "#C6982F", // baseline
      "#E27D60", // ipcc +1.5
      "#C0392B", // drought
      "#2980B9", // monsoon
      "#27AE60", // optimal
    ];

    const ctx = canvas.getContext("2d");
    if (scenarioCompareChart) {
      scenarioCompareChart.data.labels = labels;
      scenarioCompareChart.data.datasets[0].data = yields;
      scenarioCompareChart.data.datasets[0].label = `Projected Yield (${currentUnit.replace("_", "/")})`;
      scenarioCompareChart.update();
    } else {
      scenarioCompareChart = new Chart(ctx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            {
              label: `Projected Yield (${currentUnit.replace("_", "/")})`,
              data: yields,
              backgroundColor: colors,
              borderRadius: 6,
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
                  const sc = scenData.scenarios[ctx.dataIndex];
                  const sign = sc.delta_pct >= 0 ? "+" : "";
                  return `${fmtNum(ctx.raw, 2)} ${currentUnit.replace("_", "/")} (${sign}${sc.delta_pct}% vs baseline)`;
                },
              },
            },
          },
          scales: {
            x: { ticks: { color: "#EFEAD7", font: { size: 10 } }, grid: { display: false } },
            y: {
              ticks: { color: "#B9BFA4", font: { family: "IBM Plex Mono", size: 10 } },
              grid: { color: "rgba(239, 234, 215, 0.1)" },
            },
          },
        },
      });
    }

    // Render cards
    if (container) {
      container.innerHTML = "";
      scenData.scenarios.forEach((sc) => {
        const card = document.createElement("div");
        card.className = "scen-card";
        const sign = sc.delta_pct >= 0 ? "+" : "";
        const deltaClass = sc.delta_pct >= 0 ? "pos" : "neg";
        const valInUnit = convertYield(sc.yield_t_ha, currentUnit);

        card.innerHTML = `
          <div class="scen-head">
            <span class="scen-icon">${sc.icon}</span>
            <span class="scen-name">${sc.name}</span>
          </div>
          <div class="scen-val">${fmtNum(valInUnit, 2)} <small style="font-size: 0.72rem;">${currentUnit.replace("_", "/")}</small></div>
          <div class="scen-delta ${deltaClass}">${sign}${sc.delta_pct}% vs baseline</div>
          <div style="font-size: 0.72rem; color: #B9BFA4; margin-top: 6px;">${sc.temp}°C · ${sc.rain}mm rain</div>
        `;
        container.appendChild(card);
      });
    }
  }

  // ---------------------------------------------------------------------
  // India Intelligence (India vs Global Chart & Table)
  // ---------------------------------------------------------------------
  let indiaChartUnit = "q_ha";

  async function initIndiaInsights() {
    try {
      const res = await fetch("/api/india_insights");
      const data = await res.json();
      indiaDataCache = data;

      renderIndiaGlobalChart();
      renderIndiaCropTable(data.indian_crops);

      const btnQHa = $("#india-unit-qha");
      const btnTHa = $("#india-unit-tha");
      if (btnQHa && btnTHa) {
        btnQHa.addEventListener("click", () => {
          btnQHa.classList.add("active");
          btnTHa.classList.remove("active");
          indiaChartUnit = "q_ha";
          renderIndiaGlobalChart();
        });
        btnTHa.addEventListener("click", () => {
          btnTHa.classList.add("active");
          btnQHa.classList.remove("active");
          indiaChartUnit = "t_ha";
          renderIndiaGlobalChart();
        });
      }
    } catch (err) {
      console.warn("India insights fetch error:", err);
    }
  }

  function renderIndiaGlobalChart() {
    const canvas = $("#india-global-chart");
    if (!canvas || !indiaDataCache) return;

    const crops = indiaDataCache.indian_crops;
    const labels = crops.map((c) => c.item);
    const indiaYields = crops.map((c) => (indiaChartUnit === "q_ha" ? c.india_q_ha : c.india_t_ha));
    const globalYields = crops.map((c) => (indiaChartUnit === "q_ha" ? c.global_q_ha : c.global_t_ha));

    const ctx = canvas.getContext("2d");
    const unitLabel = indiaChartUnit === "q_ha" ? "Quintals / Hectare (q/ha)" : "Tonnes / Hectare (t/ha)";

    if (indiaGlobalChart) {
      indiaGlobalChart.data.labels = labels;
      indiaGlobalChart.data.datasets[0].data = indiaYields;
      indiaGlobalChart.data.datasets[1].data = globalYields;
      indiaGlobalChart.options.scales.y.title.text = unitLabel;
      indiaGlobalChart.update();
    } else {
      indiaGlobalChart = new Chart(ctx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            {
              label: "🇮🇳 India Average Yield",
              data: indiaYields,
              backgroundColor: "#C6982F",
              borderRadius: 6,
            },
            {
              label: "🌐 Global Average Yield (101 Countries)",
              data: globalYields,
              backgroundColor: "#4B7A50",
              borderRadius: 6,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "top",
              labels: { font: { family: "IBM Plex Sans", size: 11 }, color: "#232A1E" },
            },
            tooltip: {
              callbacks: {
                afterLabel: (ctx) => {
                  const crop = crops[ctx.dataIndex];
                  return `India is ${crop.yield_ratio_pct}% of world benchmark\nStates: ${crop.major_states}`;
                },
              },
            },
          },
          scales: {
            x: { ticks: { color: "#232A1E", font: { size: 11, weight: "bold" } }, grid: { display: false } },
            y: {
              title: { display: true, text: unitLabel, color: "#57604C" },
              ticks: { color: "#57604C", font: { family: "IBM Plex Mono", size: 10 } },
              grid: { color: "#D9DCC5" },
            },
          },
        },
      });
    }
  }

  function renderIndiaCropTable(crops) {
    const tbody = $("#india-crops-table tbody");
    if (!tbody || !crops) return;
    tbody.innerHTML = "";

    crops.forEach((c) => {
      const tr = document.createElement("tr");
      if (c.yield_ratio_pct > 100) tr.className = "highlight-row";
      tr.innerHTML = `
        <td><strong>${c.item}</strong></td>
        <td>${c.hindi_name}</td>
        <td><span class="pill-badge">${c.season}</span></td>
        <td><strong>${c.india_q_ha}</strong> <small>(${c.india_q_acre} q/ac)</small></td>
        <td>${c.global_q_ha}</td>
        <td><strong>${c.yield_ratio_pct}%</strong></td>
        <td><small>${c.major_states}</small></td>
      `;
      tbody.appendChild(tr);
    });
  }

  // ---------------------------------------------------------------------
  // Model Evaluation Dashboard (Comparison & Scatter Charts)
  // ---------------------------------------------------------------------
  let modelMetricMode = "r2"; // 'r2' or 'rmse'

  async function initEvalDashboard() {
    try {
      const res = await fetch("/api/evaluation");
      const data = await res.json();
      evalDataCache = data;

      renderModelComparisonChart();
      renderFeatureImpChart(data.feature_imp_comparison);
      renderScatterChart(data.scatter);

      const btnR2 = $("#btn-chart-r2");
      const btnRMSE = $("#btn-chart-rmse");
      if (btnR2 && btnRMSE) {
        btnR2.addEventListener("click", () => {
          btnR2.classList.add("active");
          btnRMSE.classList.remove("active");
          modelMetricMode = "r2";
          renderModelComparisonChart();
        });
        btnRMSE.addEventListener("click", () => {
          btnRMSE.classList.add("active");
          btnR2.classList.remove("active");
          modelMetricMode = "rmse";
          renderModelComparisonChart();
        });
      }
    } catch (err) {
      console.warn("Evaluation dashboard init error:", err);
    }
  }

  function renderModelComparisonChart() {
    const canvas = $("#model-comparison-chart");
    if (!canvas || !evalDataCache) return;

    const metrics = evalDataCache.metrics;
    const modelKeys = ["RandomForest", "XGBoost", "DecisionTree", "LinearRegression", "SVR", "MeanBaseline"];
    const labels = modelKeys.map((k) => metrics[k].name);

    const values = modelKeys.map((k) =>
      modelMetricMode === "r2" ? metrics[k].r2 : metrics[k].rmse_t_ha
    );

    const colors = modelKeys.map((k) => {
      if (k === "RandomForest") return "#C6982F"; // champion gold
      if (k === "XGBoost") return "#4B7A50"; // leaf green
      if (k === "SVR") return "#AE5236"; // rust failed
      return "#8A9179";
    });

    const ctx = canvas.getContext("2d");
    const metricTitle = modelMetricMode === "r2" ? "R² Score (Goodness of Fit)" : "RMSE (t/ha - Lower is Better)";

    if (modelComparisonChart) {
      modelComparisonChart.data.labels = labels;
      modelComparisonChart.data.datasets[0].data = values;
      modelComparisonChart.data.datasets[0].backgroundColor = colors;
      modelComparisonChart.data.datasets[0].label = metricTitle;
      modelComparisonChart.options.scales.y.title.text = metricTitle;
      modelComparisonChart.update();
    } else {
      modelComparisonChart = new Chart(ctx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            {
              label: metricTitle,
              data: values,
              backgroundColor: colors,
              borderRadius: 6,
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
                  const m = metrics[modelKeys[ctx.dataIndex]];
                  return `${modelMetricMode.toUpperCase()}: ${ctx.raw} | 5-Fold CV: ${m.cv_rmse} ± ${m.cv_std}`;
                },
              },
            },
          },
          scales: {
            x: { ticks: { color: "#57604C", font: { size: 10 } }, grid: { display: false } },
            y: {
              title: { display: true, text: metricTitle, color: "#57604C" },
              ticks: { color: "#57604C", font: { family: "IBM Plex Mono", size: 10 } },
              grid: { color: "#D9DCC5" },
            },
          },
        },
      });
    }
  }

  function renderFeatureImpChart(featureImps) {
    const canvas = $("#feature-imp-chart");
    if (!canvas || !featureImps) return;

    const labels = featureImps.map((f) => f.feature);
    const impurityValues = featureImps.map((f) => f.impurity * 100);
    const permValues = featureImps.map((f) => f.perm_delta_t_ha);

    const ctx = canvas.getContext("2d");
    if (featureImpChart) {
      featureImpChart.update();
      return;
    }

    featureImpChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "MDI Gini Impurity Share (%)",
            data: impurityValues,
            backgroundColor: "#C6982F",
            borderRadius: 4,
          },
          {
            label: "Permutation RMSE Delta (t/ha)",
            data: permValues,
            backgroundColor: "#4B7A50",
            borderRadius: 4,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top", labels: { font: { size: 10 }, color: "#57604C" } },
        },
        scales: {
          x: { ticks: { color: "#57604C", font: { size: 10 } }, grid: { color: "#D9DCC5" } },
          y: { ticks: { color: "#232A1E", font: { size: 11, weight: "bold" } }, grid: { display: false } },
        },
      },
    });
  }

  function renderScatterChart(scatter) {
    const canvas = $("#eval-scatter-chart");
    if (!canvas || !scatter) return;

    const scatterData = scatter.map((pt) => ({ x: pt.actual, y: pt.predicted, area: pt.area }));
    const ctx = canvas.getContext("2d");

    evalScatterChart = new Chart(ctx, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Test Set Actual vs Predicted (t/ha)",
            data: scatterData,
            backgroundColor: "rgba(198, 152, 47, 0.8)",
            borderColor: "#9C7621",
            borderWidth: 1,
            pointRadius: 5,
            pointHoverRadius: 7,
          },
          {
            label: "Ideal 1:1 Reference Line",
            data: [{ x: 0, y: 0 }, { x: 12, y: 12 }],
            type: "line",
            borderColor: "rgba(75, 122, 80, 0.6)",
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
                return `${pt.area}: Actual ${pt.x} t/ha | Predicted ${pt.y} t/ha`;
              },
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: "Actual Yield (t/ha)", color: "#57604C", font: { size: 10 } },
            ticks: { color: "#57604C", font: { size: 10 } },
            grid: { color: "#D9DCC5" },
          },
          y: {
            title: { display: true, text: "Predicted Yield (t/ha)", color: "#57604C", font: { size: 10 } },
            ticks: { color: "#57604C", font: { size: 10 } },
            grid: { color: "#D9DCC5" },
          },
        },
      },
    });
  }

  // ---------------------------------------------------------------------
  // Leaflet Map Visualizer
  // ---------------------------------------------------------------------
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
      center: [20.5937, 78.9629], // Center over India
      zoom: 3,
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
          const radius = Math.min(24, Math.max(7, yieldVal * 2.2));
          const isIndia = country === "India";
          const marker = L.circleMarker(coords, {
            radius: isIndia ? radius + 3 : radius,
            fillColor: isIndia ? "#2E7D32" : "#C6982F",
            color: isIndia ? "#1B5E20" : "#345439",
            weight: isIndia ? 2.5 : 1.5,
            opacity: 0.9,
            fillOpacity: isIndia ? 0.85 : 0.65,
          });

          const qha = (yieldVal * 10).toFixed(1);
          marker.bindPopup(`
            <div style="font-family: var(--font-body); padding: 4px;">
              <strong style="font-size: 0.95rem; color: #232A1E;">${country}</strong><br>
              <span style="font-size: 0.84rem; color: #57604C;">Crop: ${item}</span><br>
              <strong style="font-size: 1.05rem; color: #9C7621;">${yieldVal} t/ha</strong>
              <div style="font-size: 0.8rem; color: #4B7A50; font-weight: 600;">(${qha} q/ha)</div>
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
  // Initialization
  // ---------------------------------------------------------------------
  function init() {
    buildStalk();
    updateOutputs();
    initPresets();

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
    initIndiaInsights();
    initEvalDashboard();

    loadOptions().then(() => {
      onAreaItemChange();
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
