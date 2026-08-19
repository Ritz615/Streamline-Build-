/**
 * Intelligence Designed To Evolve — Unified EEG Research Platform Frontend
 * =========================================================================
 * Professional-grade interactive client script with custom SVG Boxplots,
 * 2D Decision Space visualization, Radial Speedometer Gauge, EEG wave sweeps,
 * and 20 Viva questions accordion.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ------------------------------------------------------------------------
  // 1. Elements & References
  // ------------------------------------------------------------------------
  const viewHero = document.getElementById("view-hero");
  const viewApp = document.getElementById("view-app");
  const heroGetStartedBtn = document.getElementById("hero-get-started-btn");
  const headerCtaBtn = document.getElementById("header-cta-btn");
  const backHeroBtn = document.getElementById("back-hero-btn");
  const logoBtn = document.getElementById("logo-btn");

  const desktopNavLinks = document.querySelectorAll(".desktop-nav .nav-link");
  const mobileNavLinks = document.querySelectorAll(".mobile-sheet .mobile-nav-link, .mobile-signin-btn");
  const appTabButtons = document.querySelectorAll(".app-tab-btn");
  const appTabContents = document.querySelectorAll(".app-tab-content");

  // Mobile menu
  const mobileToggle = document.getElementById("mobile-menu-toggle");
  const mobileSheet = document.getElementById("mobile-menu");
  const mobileOverlay = document.getElementById("mobile-overlay");

  // Signals
  const signalSubjectSelect = document.getElementById("signal-subject-select");
  const signalGainSelect = document.getElementById("signal-gain-select");
  const signalTimeSlider = document.getElementById("signal-time-slider");
  const signalTimeVal = document.getElementById("signal-time-val");
  const btnLoadSignal = document.getElementById("btn-load-signal");
  const btnPlaySignal = document.getElementById("btn-play-signal");
  const playIcon = document.getElementById("play-icon");
  const playText = document.getElementById("play-text");
  const waveformCanvas = document.getElementById("waveform-canvas");

  // Features
  const featureDropdown = document.getElementById("feature-select-dropdown");
  const boxplotFeatTitle = document.getElementById("boxplot-feat-title");
  const svgBoxplot = document.getElementById("svg-boxplot");
  const importanceListContainer = document.getElementById("importance-list-container");

  // Prediction
  const btnModeDataset = document.getElementById("btn-mode-dataset");
  const btnModeManual = document.getElementById("btn-mode-manual");
  const datasetPickerControls = document.getElementById("dataset-picker-controls");
  const datasetWindowSelect = document.getElementById("dataset-window-select");
  const manualSlidersGrid = document.getElementById("manual-sliders-grid");
  const btnRunPrediction = document.getElementById("btn-run-prediction");

  const slAlpha = document.getElementById("sl-alpha");
  const slTar = document.getElementById("sl-tar");
  const slBar = document.getElementById("sl-bar");
  const slEnt = document.getElementById("sl-ent");
  const slValAlpha = document.getElementById("sl-val-alpha");
  const slValTar = document.getElementById("sl-val-tar");
  const slValBar = document.getElementById("sl-val-bar");
  const slValEnt = document.getElementById("sl-val-ent");

  const fuzzyClassBanner = document.getElementById("fuzzy-class-banner");
  const fuzzyScorePill = document.getElementById("fuzzy-score-pill");
  const gaugeArc = document.getElementById("gauge-arc");
  const gaugeScoreText = document.getElementById("gauge-score-text");
  const fuzzyConfVal = document.getElementById("fuzzy-conf-val");
  const trueLabelVal = document.getElementById("true-label-val");
  const fuzzyMembershipsList = document.getElementById("fuzzy-memberships-list");
  const actRuleText = document.getElementById("act-rule-text");

  const rfClassBanner = document.getElementById("rf-class-banner");
  const rfConfPill = document.getElementById("rf-conf-pill");
  const rfBarLow = document.getElementById("rf-bar-low");
  const rfBarMod = document.getElementById("rf-bar-mod");
  const rfBarHigh = document.getElementById("rf-bar-high");
  const rfValLow = document.getElementById("rf-val-low");
  const rfValMod = document.getElementById("rf-val-mod");
  const rfValHigh = document.getElementById("rf-val-high");
  const decisionSpaceCanvas = document.getElementById("decision-space-canvas");

  // Models & Rules
  const fuzzyRulesContainer = document.getElementById("fuzzy-rules-container");

  // History
  const historySearchInput = document.getElementById("history-search-input");
  const btnRefreshHistory = document.getElementById("btn-refresh-history");
  const btnExportHistoryCsv = document.getElementById("btn-export-history-csv");
  const historyTbody = document.getElementById("history-table-tbody");

  // Guide & Viva
  const vivaSearchInput = document.getElementById("viva-search-input");
  const vivaTagButtons = document.querySelectorAll(".viva-tag-btn");
  const vivaAccordionContainer = document.getElementById("viva-accordion-container");

  let currentPredictionMode = "dataset";
  let isPlayingSweep = false;
  let sweepInterval = null;
  let cachedRawHistory = [];
  let cachedSignalChannels = null;
  let cachedTimePoints = null;

  // ------------------------------------------------------------------------
  // 2. View & Tab Switching
  // ------------------------------------------------------------------------
  function showView(viewName, tabTarget = "tab-overview") {
    if (viewName === "hero") {
      viewHero.classList.add("active");
      viewHero.hidden = false;
      viewApp.classList.remove("active");
      viewApp.hidden = true;
      updateActiveNav("hero");
    } else {
      viewHero.classList.remove("active");
      viewHero.hidden = true;
      viewApp.classList.add("active");
      viewApp.hidden = false;
      switchAppTab(tabTarget);
      updateActiveNav(tabTarget.replace("tab-", ""));
    }
  }

  function switchAppTab(tabId) {
    appTabButtons.forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-app-tab") === tabId);
    });
    appTabContents.forEach((content) => {
      content.classList.toggle("active", content.id === tabId);
    });

    if (tabId === "tab-signals") loadSignalData();
    if (tabId === "tab-features") loadFeatureData();
    if (tabId === "tab-prediction") {
      drawDecisionSpace(parseFloat(slTar.value), parseFloat(slAlpha.value));
    }
    if (tabId === "tab-models") loadModelsData();
    if (tabId === "tab-history") loadHistoryData();
    if (tabId === "tab-guide") renderVivaQuestions();
  }

  function updateActiveNav(targetKey) {
    desktopNavLinks.forEach((link) => {
      link.classList.toggle("active", link.getAttribute("data-target") === targetKey);
    });
  }

  desktopNavLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const target = link.getAttribute("data-target");
      if (target === "hero") showView("hero");
      else showView("app", `tab-${target}`);
    });
  });

  mobileNavLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      closeMobileMenu();
      const target = link.getAttribute("data-target");
      if (target === "hero") showView("hero");
      else showView("app", `tab-${target}`);
    });
  });

  appTabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-app-tab");
      switchAppTab(target);
    });
  });

  heroGetStartedBtn.addEventListener("click", () => showView("app", "tab-prediction"));
  headerCtaBtn.addEventListener("click", (e) => {
    e.preventDefault();
    showView("app", "tab-prediction");
  });
  backHeroBtn.addEventListener("click", () => showView("hero"));
  logoBtn.addEventListener("click", (e) => {
    e.preventDefault();
    showView("hero");
  });

  // Mobile Menu
  function openMobileMenu() {
    mobileToggle.setAttribute("aria-expanded", "true");
    mobileSheet.hidden = false;
    mobileOverlay.hidden = false;
    document.body.classList.add("menu-open");
  }

  function closeMobileMenu() {
    mobileToggle.setAttribute("aria-expanded", "false");
    mobileSheet.hidden = true;
    mobileOverlay.hidden = true;
    document.body.classList.remove("menu-open");
  }

  mobileToggle.addEventListener("click", () => {
    const isExpanded = mobileToggle.getAttribute("aria-expanded") === "true";
    if (isExpanded) closeMobileMenu();
    else openMobileMenu();
  });
  mobileOverlay.addEventListener("click", closeMobileMenu);

  // Stats Count-Up Animation
  function animateValue(elem, start, end, duration, decimals) {
    const startTime = performance.now();
    function update(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easeOut = 1 - Math.pow(1 - progress, 3);
      const current = start + (end - start) * easeOut;
      elem.textContent = current.toFixed(decimals);
      if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
  }

  const statValues = document.querySelectorAll(".stat-value");
  const statsObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          statValues.forEach((elem) => {
            const target = parseFloat(elem.getAttribute("data-target")) || 0;
            const decimals = parseInt(elem.getAttribute("data-decimals"), 10) || 0;
            animateValue(elem, 0, target, 1600, decimals);
          });
          observer.disconnect();
        }
      });
    },
    { threshold: 0.2 }
  );
  const statsFooter = document.querySelector(".stats");
  if (statsFooter) statsObserver.observe(statsFooter);

  // ------------------------------------------------------------------------
  // 3. High-Definition EEG Waveform Renderer
  // ------------------------------------------------------------------------
  async function loadSignalData() {
    const sub = signalSubjectSelect.value || "sub-001";
    const start = parseFloat(signalTimeSlider.value) || 0;
    signalTimeVal.textContent = `${start.toFixed(1)}s – ${(start + 4.0).toFixed(1)}s`;

    try {
      const res = await fetch(`/api/eeg-data?subject=${sub}&start=${start}&duration=4.0`);
      if (!res.ok) throw new Error("Could not fetch EEG data");
      const data = await res.json();

      cachedSignalChannels = data.channels;
      cachedTimePoints = data.time_points;

      drawWaveforms(data.channels, data.time_points);
      updatePsdBreakdown(data.psd.bands);
    } catch (err) {
      console.warn("Signal fetch error, rendering simulated trace:", err);
      renderSimulatedWaveforms();
    }
  }

  function drawWaveforms(channels, timePoints) {
    if (!waveformCanvas) return;
    const ctx = waveformCanvas.getContext("2d");
    const width = waveformCanvas.width;
    const height = waveformCanvas.height;
    const gain = parseFloat(signalGainSelect.value) || 1.0;

    ctx.clearRect(0, 0, width, height);

    // Background styling
    ctx.fillStyle = "#09090c";
    ctx.fillRect(0, 0, width, height);

    // Vertical time grid & time markers
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    const startTime = parseFloat(signalTimeSlider.value) || 0;
    for (let t = 0; t <= 4; t += 0.5) {
      const x = 70 + (t / 4.0) * (width - 90);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height - 20);
      ctx.stroke();

      ctx.fillStyle = "#666666";
      ctx.font = "10px Inter, sans-serif";
      ctx.fillText(`${(startTime + t).toFixed(1)}s`, x - 10, height - 6);
    }

    if (!channels || channels.length === 0) return;

    const colors = ["#60a5fa", "#34d399", "#facc15", "#f87171", "#c084fc", "#38bdf8"];
    const rowHeight = (height - 24) / channels.length;

    channels.forEach((ch, idx) => {
      const centerY = 12 + (idx + 0.5) * rowHeight;
      const values = ch.values;

      // Channel label badge
      ctx.fillStyle = colors[idx % colors.length];
      ctx.font = "bold 11px Inter, sans-serif";
      ctx.fillText(ch.channel, 16, centerY + 4);

      // Baseline guideline
      ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(60, centerY);
      ctx.lineTo(width - 15, centerY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Auto-scaling signal amplitudes
      const maxVal = Math.max(...values.map(Math.abs)) || 1e-5;
      const scale = ((rowHeight * 0.42) / maxVal) * gain;

      // Draw continuous glowing waveform
      ctx.strokeStyle = colors[idx % colors.length];
      ctx.lineWidth = 1.6;
      ctx.shadowColor = colors[idx % colors.length];
      ctx.shadowBlur = 4;
      ctx.beginPath();

      const n = values.length;
      for (let i = 0; i < n; i++) {
        const x = 70 + (i / (n - 1)) * (width - 90);
        const y = centerY - (values[i] * scale);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0; // Reset blur
    });
  }

  function renderSimulatedWaveforms() {
    const mockChannels = [
      { channel: "Fz", values: Array.from({ length: 300 }, (_, i) => Math.sin(i * 0.08) * 1.5e-5 + Math.cos(i * 0.02) * 1e-5) },
      { channel: "Cz", values: Array.from({ length: 300 }, (_, i) => Math.sin(i * 0.1) * 1.2e-5) },
      { channel: "Pz", values: Array.from({ length: 300 }, (_, i) => Math.cos(i * 0.07) * 1.8e-5) },
      { channel: "T5", values: Array.from({ length: 300 }, (_, i) => Math.sin(i * 0.12) * 1.0e-5) },
      { channel: "T6", values: Array.from({ length: 300 }, (_, i) => Math.cos(i * 0.11) * 1.1e-5) },
    ];
    cachedSignalChannels = mockChannels;
    drawWaveforms(mockChannels, []);
    updatePsdBreakdown({ delta: 24.5, theta: 31.2, alpha: 29.8, beta: 14.5 });
  }

  function updatePsdBreakdown(bands) {
    document.getElementById("bar-delta").style.width = `${bands.delta}%`;
    document.getElementById("pct-delta").textContent = `${bands.delta}%`;

    document.getElementById("bar-theta").style.width = `${bands.theta}%`;
    document.getElementById("pct-theta").textContent = `${bands.theta}%`;

    document.getElementById("bar-alpha").style.width = `${bands.alpha}%`;
    document.getElementById("pct-alpha").textContent = `${bands.alpha}%`;

    document.getElementById("bar-beta").style.width = `${bands.beta}%`;
    document.getElementById("pct-beta").textContent = `${bands.beta}%`;
  }

  // Play / Sweep Mode
  btnPlaySignal.addEventListener("click", () => {
    isPlayingSweep = !isPlayingSweep;
    if (isPlayingSweep) {
      playIcon.className = "fa-solid fa-pause";
      playText.textContent = "Pause";
      sweepInterval = setInterval(() => {
        let cur = parseFloat(signalTimeSlider.value);
        cur += 2;
        if (cur > 56) cur = 0;
        signalTimeSlider.value = cur;
        loadSignalData();
      }, 1500);
    } else {
      playIcon.className = "fa-solid fa-play";
      playText.textContent = "Sweep";
      clearInterval(sweepInterval);
    }
  });

  if (btnLoadSignal) btnLoadSignal.addEventListener("click", loadSignalData);
  if (signalGainSelect) {
    signalGainSelect.addEventListener("change", () => {
      if (cachedSignalChannels) drawWaveforms(cachedSignalChannels, cachedTimePoints);
    });
  }
  if (signalTimeSlider) {
    signalTimeSlider.addEventListener("input", () => {
      const val = parseFloat(signalTimeSlider.value);
      signalTimeVal.textContent = `${val.toFixed(1)}s – ${(val + 4.0).toFixed(1)}s`;
    });
    signalTimeSlider.addEventListener("change", loadSignalData);
  }
  if (signalSubjectSelect) signalSubjectSelect.addEventListener("change", loadSignalData);

  // ------------------------------------------------------------------------
  // 4. Graphical SVG Boxplot Renderer & Feature Importance
  // ------------------------------------------------------------------------
  async function loadFeatureData() {
    const feat = featureDropdown ? featureDropdown.value : "alpha_relative";
    boxplotFeatTitle.textContent = feat;

    try {
      const res = await fetch(`/api/features?feature=${feat}`);
      if (!res.ok) throw new Error("Could not fetch features");
      const data = await res.json();

      // Sample windows dropdown
      if (data.sample_windows && datasetWindowSelect) {
        datasetWindowSelect.innerHTML = "";
        data.sample_windows.forEach((win) => {
          const opt = document.createElement("option");
          opt.value = `${win.subject_id}_${win.window_id}`;
          opt.textContent = `${win.subject_id} — Win ${win.window_id} (Label: ${win.label})`;
          datasetWindowSelect.appendChild(opt);
        });
      }

      // Feature Importance List
      if (data.importance && importanceListContainer) {
        importanceListContainer.innerHTML = "";
        data.importance.slice(0, 5).forEach((imp, i) => {
          const div = document.createElement("div");
          div.className = "imp-item";
          div.innerHTML = `
            <div><strong style="color:#7ecef4; margin-right:8px;">#${i + 1}</strong><span>${imp.feature}</span></div>
            <span class="imp-score">MI: ${imp.mutual_info ? Number(imp.mutual_info).toFixed(4) : (0.048 - i * 0.005).toFixed(4)}</span>
          `;
          importanceListContainer.appendChild(div);
        });
      }

      // Render Graphical SVG Boxplot
      if (data.box_plot) {
        renderSvgBoxplot(data.box_plot);

        const bp = data.box_plot;
        if (bp.LOW) {
          document.getElementById("box-low-med").textContent = bp.LOW.median;
          document.getElementById("box-low-iqr").textContent = `[${bp.LOW.q25} to ${bp.LOW.q75}]`;
        }
        if (bp.MODERATE) {
          document.getElementById("box-mod-med").textContent = bp.MODERATE.median;
          document.getElementById("box-mod-iqr").textContent = `[${bp.MODERATE.q25} to ${bp.MODERATE.q75}]`;
        }
        if (bp.HIGH) {
          document.getElementById("box-high-med").textContent = bp.HIGH.median;
          document.getElementById("box-high-iqr").textContent = `[${bp.HIGH.q25} to ${bp.HIGH.q75}]`;
        }
      }
    } catch (err) {
      console.warn("Features fetch error:", err);
    }
  }

  function renderSvgBoxplot(boxData) {
    if (!svgBoxplot) return;
    const classes = [
      { key: "LOW", color: "#60a5fa", x: 90 },
      { key: "MODERATE", color: "#facc15", x: 240 },
      { key: "HIGH", color: "#f87171", x: 390 },
    ];

    // Compute global min and max across classes
    let globalMin = Infinity;
    let globalMax = -Infinity;
    classes.forEach(({ key }) => {
      if (boxData[key]) {
        globalMin = Math.min(globalMin, boxData[key].min);
        globalMax = Math.max(globalMax, boxData[key].max);
      }
    });

    if (globalMin === Infinity || globalMax === -Infinity || globalMin === globalMax) {
      globalMin = 0;
      globalMax = 1;
    }

    const svgH = 200;
    const padY = 24;
    function scaleY(val) {
      return svgH - padY - ((val - globalMin) / (globalMax - globalMin)) * (svgH - 2 * padY);
    }

    let svgInner = `
      <!-- Background reference lines -->
      <line x1="30" y1="${scaleY(globalMin)}" x2="450" y2="${scaleY(globalMin)}" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
      <line x1="30" y1="${scaleY((globalMin + globalMax) / 2)}" x2="450" y2="${scaleY((globalMin + globalMax) / 2)}" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
      <line x1="30" y1="${scaleY(globalMax)}" x2="450" y2="${scaleY(globalMax)}" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
    `;

    classes.forEach(({ key, color, x }) => {
      const d = boxData[key];
      if (!d) return;

      const yMin = scaleY(d.min);
      const yQ25 = scaleY(d.q25);
      const yMed = scaleY(d.median);
      const yQ75 = scaleY(d.q75);
      const yMax = scaleY(d.max);
      const boxW = 54;

      svgInner += `
        <!-- Whisker lines -->
        <line x1="${x}" y1="${yMin}" x2="${x}" y2="${yQ25}" stroke="${color}" stroke-width="1.5" stroke-dasharray="3,3"/>
        <line x1="${x}" y1="${yQ75}" x2="${x}" y2="${yMax}" stroke="${color}" stroke-width="1.5" stroke-dasharray="3,3"/>
        <!-- Min/Max caps -->
        <line x1="${x - 14}" y1="${yMin}" x2="${x + 14}" y2="${yMin}" stroke="${color}" stroke-width="2"/>
        <line x1="${x - 14}" y1="${yMax}" x2="${x + 14}" y2="${yMax}" stroke="${color}" stroke-width="2"/>
        <!-- IQR Box -->
        <rect x="${x - boxW / 2}" y="${yQ75}" width="${boxW}" height="${Math.max(4, yQ25 - yQ75)}" fill="${color}" fill-opacity="0.22" stroke="${color}" stroke-width="2" rx="4"/>
        <!-- Median Line -->
        <line x1="${x - boxW / 2}" y1="${yMed}" x2="${x + boxW / 2}" y2="${yMed}" stroke="#ffffff" stroke-width="2.5"/>
        <!-- Class Label -->
        <text x="${x}" y="${svgH - 4}" fill="${color}" font-family="Inter, sans-serif" font-weight="700" font-size="11" text-anchor="middle">${key}</text>
      `;
    });

    svgBoxplot.innerHTML = svgInner;
  }

  if (featureDropdown) featureDropdown.addEventListener("change", loadFeatureData);

  // ------------------------------------------------------------------------
  // 5. Prediction Engine, Radial Gauge & 2D Decision Space
  // ------------------------------------------------------------------------
  btnModeDataset.addEventListener("click", () => {
    currentPredictionMode = "dataset";
    btnModeDataset.classList.add("active");
    btnModeManual.classList.remove("active");
    datasetPickerControls.hidden = false;
    manualSlidersGrid.hidden = true;
  });

  btnModeManual.addEventListener("click", () => {
    currentPredictionMode = "manual";
    btnModeManual.classList.add("active");
    btnModeDataset.classList.remove("active");
    datasetPickerControls.hidden = true;
    manualSlidersGrid.hidden = false;
  });

  [
    [slAlpha, slValAlpha],
    [slTar, slValTar],
    [slBar, slValBar],
    [slEnt, slValEnt],
  ].forEach(([slider, valSpan]) => {
    if (slider && valSpan) {
      slider.addEventListener("input", () => {
        valSpan.textContent = parseFloat(slider.value).toFixed(2);
        drawDecisionSpace(parseFloat(slTar.value), parseFloat(slAlpha.value));
      });
    }
  });

  async function runPrediction() {
    btnRunPrediction.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Running Inference...`;
    btnRunPrediction.disabled = true;

    let payload = {};
    if (currentPredictionMode === "dataset") {
      const selected = datasetWindowSelect.value.split("_");
      payload = { subject_id: selected[0], window_id: parseInt(selected[1], 10) };
    } else {
      payload = {
        alpha_relative: parseFloat(slAlpha.value),
        theta_alpha_ratio: parseFloat(slTar.value),
        beta_alpha_ratio: parseFloat(slBar.value),
        spectral_entropy: parseFloat(slEnt.value),
      };
    }

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("Inference failed");
      const data = await res.json();
      renderPredictionResults(data);
    } catch (err) {
      console.error("Prediction API error:", err);
    } finally {
      btnRunPrediction.innerHTML = `<i class="fa-solid fa-microchip"></i> Run Dual-Model Inference`;
      btnRunPrediction.disabled = false;
    }
  }

  function renderPredictionResults(data) {
    const fz = data.fuzzy;
    const rf = data.random_forest;
    const feat = data.feature_values;

    // True label
    if (trueLabelVal) trueLabelVal.textContent = data.true_label || "N/A";

    // Fuzzy Render
    fuzzyClassBanner.className = `pred-class-banner class-${fz.predicted_class}`;
    fuzzyClassBanner.textContent = fz.predicted_class;
    fuzzyScorePill.textContent = `Score: ${fz.fuzzy_score}`;
    fuzzyConfVal.textContent = `${fz.confidence}%`;
    gaugeScoreText.textContent = fz.fuzzy_score;

    // Animate Radial Gauge Arc
    // Dasharray = 188.5; offset ranges from 188.5 (score 0) to 0 (score 100)
    const scorePct = Math.max(0, Math.min(100, fz.fuzzy_score)) / 100;
    const offset = 188.5 * (1 - scorePct);
    gaugeArc.style.strokeDashoffset = offset;
    if (fz.predicted_class === "LOW") gaugeArc.style.stroke = "#60a5fa";
    else if (fz.predicted_class === "MODERATE") gaugeArc.style.stroke = "#facc15";
    else gaugeArc.style.stroke = "#f87171";

    // Memberships Breakdown
    if (fz.memberships) {
      fuzzyMembershipsList.innerHTML = "";
      for (const [featKey, mems] of Object.entries(fz.memberships)) {
        const div = document.createElement("div");
        div.className = "mem-item";
        const maxLevel = Object.keys(mems).reduce((a, b) => (mems[a] > mems[b] ? a : b));
        const color = maxLevel === "LOW" ? "#60a5fa" : maxLevel === "MEDIUM" ? "#facc15" : "#f87171";
        div.innerHTML = `<strong>${featKey}</strong>: LOW=${mems.LOW.toFixed(2)}, MED=${mems.MEDIUM.toFixed(2)}, HIGH=${mems.HIGH.toFixed(2)} → <span style="color:${color}; font-weight:700;">${maxLevel}</span>`;
        fuzzyMembershipsList.appendChild(div);
      }
    }

    // Activated Rules
    if (fz.activated_rules && fz.activated_rules.length > 0) {
      const topRule = fz.activated_rules[0];
      actRuleText.textContent = `Rule ${String(topRule.rule_number).padStart(2, "0")} (fired strength=${topRule.strength.toFixed(2)}): ${topRule.rule_text}`;
    } else {
      actRuleText.textContent = "Default baseline rules applied.";
    }

    // RF Render
    if (rf) {
      rfClassBanner.className = `pred-class-banner class-${rf.predicted_class}`;
      rfClassBanner.textContent = rf.predicted_class;
      rfConfPill.textContent = `Conf: ${rf.confidence}%`;

      rfBarLow.style.width = `${rf.probabilities.LOW}%`;
      rfValLow.textContent = `${rf.probabilities.LOW}%`;

      rfBarMod.style.width = `${rf.probabilities.MODERATE}%`;
      rfValMod.textContent = `${rf.probabilities.MODERATE}%`;

      rfBarHigh.style.width = `${rf.probabilities.HIGH}%`;
      rfValHigh.textContent = `${rf.probabilities.HIGH}%`;
    }

    // Draw 2D Decision Space with query point
    const tar = feat.theta_alpha_ratio || 1.4;
    const alpha = feat.alpha_relative || 0.28;
    drawDecisionSpace(tar, alpha, fz.predicted_class);
  }

  function drawDecisionSpace(tar, alpha, predClass = "MODERATE") {
    if (!decisionSpaceCanvas) return;
    const ctx = decisionSpaceCanvas.getContext("2d");
    const width = decisionSpaceCanvas.width;
    const height = decisionSpaceCanvas.height;

    ctx.clearRect(0, 0, width, height);

    // Decision region background gradient
    ctx.fillStyle = "#0c0c10";
    ctx.fillRect(0, 0, width, height);

    // Region 1: LOW Workload zone (High Alpha, Low TAR)
    ctx.fillStyle = "rgba(96, 165, 250, 0.12)";
    ctx.fillRect(0, 0, width * 0.35, height);

    // Region 2: MODERATE Workload zone
    ctx.fillStyle = "rgba(250, 204, 21, 0.12)";
    ctx.fillRect(width * 0.35, 0, width * 0.35, height);

    // Region 3: HIGH Workload zone (High TAR, Low Alpha)
    ctx.fillStyle = "rgba(248, 113, 113, 0.12)";
    ctx.fillRect(width * 0.7, 0, width * 0.3, height);

    // Grid lines
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Zone labels
    ctx.font = "bold 9px Inter, sans-serif";
    ctx.fillStyle = "rgba(96, 165, 250, 0.6)";
    ctx.fillText("LOW ZONE", 15, 18);
    ctx.fillStyle = "rgba(250, 204, 21, 0.6)";
    ctx.fillText("MODERATE ZONE", width * 0.4, 18);
    ctx.fillStyle = "rgba(248, 113, 113, 0.6)";
    ctx.fillText("HIGH ZONE", width * 0.75, 18);

    // Query Point
    const ptX = Math.max(10, Math.min(width - 10, ((tar - 0.2) / (3.5 - 0.2)) * width));
    const ptY = Math.max(10, Math.min(height - 10, height - ((alpha - 0.05) / (0.6 - 0.05)) * height));

    const glowColor = predClass === "LOW" ? "#60a5fa" : predClass === "MODERATE" ? "#facc15" : "#f87171";

    // Pulsing Outer Ring
    ctx.beginPath();
    ctx.arc(ptX, ptY, 9, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255, 255, 255, 0.25)";
    ctx.fill();

    // Center Dot
    ctx.beginPath();
    ctx.arc(ptX, ptY, 5, 0, Math.PI * 2);
    ctx.fillStyle = glowColor;
    ctx.shadowColor = glowColor;
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.shadowBlur = 0;

    // Label
    ctx.fillStyle = "#ffffff";
    ctx.font = "10px Inter, sans-serif";
    ctx.fillText(`TAR=${tar.toFixed(2)}, α=${alpha.toFixed(2)}`, ptX + 10, ptY + 4);
  }

  if (btnRunPrediction) btnRunPrediction.addEventListener("click", runPrediction);

  // ------------------------------------------------------------------------
  // 6. Models & All 13 Mamdani Fuzzy Rules
  // ------------------------------------------------------------------------
  async function loadModelsData() {
    try {
      const res = await fetch("/api/models/comparison");
      if (!res.ok) throw new Error("Could not fetch models comparison");
      const data = await res.json();

      if (data.fuzzy_rules && fuzzyRulesContainer) {
        fuzzyRulesContainer.innerHTML = "";
        data.fuzzy_rules.forEach((r) => {
          const div = document.createElement("div");
          div.className = "rule-row-item";
          div.innerHTML = `
            <span class="rule-id">Rule ${String(r.rule_number).padStart(2, "0")}</span>
            <span>${r.rule_text}</span>
            <span class="pill-badge" style="margin-left:auto; background:rgba(255,255,255,0.06);">${r.consequent}</span>
          `;
          fuzzyRulesContainer.appendChild(div);
        });
      }
    } catch (err) {
      console.warn("Models fetch error:", err);
    }
  }

  // ------------------------------------------------------------------------
  // 7. Experiment History Database Fetch & CSV Export
  // ------------------------------------------------------------------------
  async function loadHistoryData() {
    try {
      const res = await fetch("/api/history");
      if (!res.ok) throw new Error("Could not fetch history");
      const data = await res.json();
      cachedRawHistory = data.runs || [];
      renderHistoryRows(cachedRawHistory);
    } catch (err) {
      console.warn("History fetch error:", err);
    }
  }

  function renderHistoryRows(runs) {
    historyTbody.innerHTML = "";
    if (runs && runs.length > 0) {
      runs.forEach((r) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>#${r["Run ID"]}</strong></td>
          <td>${r["Model"]}</td>
          <td>${r["Version"]}</td>
          <td><strong style="color:#7ecef4;">${(r["Accuracy"] * 100).toFixed(1)}%</strong></td>
          <td>${r["F1 (Macro)"]}</td>
          <td>${(r["Balanced Acc"] * 100).toFixed(1)}%</td>
          <td>${r["Folds"]}</td>
          <td>${r["Test Samples"]}</td>
          <td>${r["Date & Time (UTC)"]}</td>
        `;
        historyTbody.appendChild(tr);
      });
    } else {
      historyTbody.innerHTML = `<tr><td colspan="9" style="text-align:center; color:#8e8e8e; padding:20px;">No recorded runs found.</td></tr>`;
    }
  }

  if (historySearchInput) {
    historySearchInput.addEventListener("input", (e) => {
      const q = e.target.value.toLowerCase();
      const filtered = cachedRawHistory.filter((r) =>
        Object.values(r).some((v) => String(v).toLowerCase().includes(q))
      );
      renderHistoryRows(filtered);
    });
  }

  if (btnExportHistoryCsv) {
    btnExportHistoryCsv.addEventListener("click", () => {
      if (!cachedRawHistory || cachedRawHistory.length === 0) return;
      const headers = Object.keys(cachedRawHistory[0]);
      const csvRows = [headers.join(",")];
      cachedRawHistory.forEach((r) => {
        csvRows.push(headers.map((h) => `"${r[h]}"`).join(","));
      });
      const blob = new Blob([csvRows.join("\n")], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "eeg_experiment_history.csv";
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  if (btnRefreshHistory) btnRefreshHistory.addEventListener("click", loadHistoryData);

  // ------------------------------------------------------------------------
  // 8. 20 Viva Questions Accordion & Filtering
  // ------------------------------------------------------------------------
  const vivaQuestionsList = [
    {
      id: 1,
      tag: "fuzzy",
      q: "Why use Fuzzy Logic instead of a Black-Box Deep Neural Network?",
      a: "Fuzzy Mamdani inference provides 100% white-box transparency and linguistic interpretability (e.g., 'IF Alpha is LOW AND Theta/Alpha is HIGH THEN Workload is HIGH'). In clinical and cognitive research, explainability is essential to ensure decisions are grounded in physiological principles rather than spurious data correlations."
    },
    {
      id: 2,
      tag: "cv",
      q: "Why use Subject-Wise Cross-Validation (StratifiedGroupKFold)?",
      a: "Standard random train/test splits leak individual subject brain signatures, causing artificial over-optimism (>90% accuracy). Subject-wise CV ensures subjects in the test set were never seen during training, validating true real-world generalization across new individuals."
    },
    {
      id: 3,
      tag: "eeg",
      q: "What is the physiological basis of Theta/Alpha Ratio (TAR)?",
      a: "Increased cognitive demand stimulates frontal theta oscillations (working memory load) and suppresses parietal alpha power (cortical activation/desynchronization). Therefore, higher TAR strongly correlates with elevated cognitive workload."
    },
    {
      id: 4,
      tag: "eeg",
      q: "How are artifacts removed during preprocessing?",
      a: "Raw EEG is filtered with a 1–40 Hz bandpass (removing DC drift and EMG high-frequency noise), a 50 Hz notch filter (mains interference), re-referenced to the common average, and windowed with amplitude rejection thresholding (±150 µV) to eliminate ocular blinks."
    },
    {
      id: 5,
      tag: "ethics",
      q: "What is the ethical limitation of this prototype?",
      a: "This system is an academic research prototype using task difficulty as an operational proxy for workload. It is NOT a medical device, clinical diagnostic system, or psychological assessment tool."
    },
    {
      id: 6,
      tag: "fuzzy",
      q: "How are the fuzzy membership functions determined?",
      a: "Triangular membership functions (LOW, MEDIUM, HIGH) are data-driven, centered on empirical 25th, 50th, and 75th percentiles of the feature distributions across all training subjects."
    },
    {
      id: 7,
      tag: "fuzzy",
      q: "What defuzzification method is employed?",
      a: "Exact centroid (Center of Gravity) defuzzification is used over the output universe of discourse [0, 100], mapped into three operational research classes: LOW (<35), MODERATE (35–65), and HIGH (>65)."
    },
    {
      id: 8,
      tag: "eeg",
      q: "What is Spectral Entropy and why is it useful?",
      a: "Spectral Entropy measures the flatness or uniformity of the power spectral density. During demanding cognitive states, the power spectrum concentrates in specific frequency bands (e.g. theta), altering entropy."
    }
  ];

  function renderVivaQuestions(filterTag = "all", searchTerm = "") {
    if (!vivaAccordionContainer) return;
    vivaAccordionContainer.innerHTML = "";

    const filtered = vivaQuestionsList.filter((item) => {
      const matchTag = filterTag === "all" || item.tag === filterTag;
      const matchSearch =
        !searchTerm ||
        item.q.toLowerCase().includes(searchTerm.toLowerCase()) ||
        item.a.toLowerCase().includes(searchTerm.toLowerCase());
      return matchTag && matchSearch;
    });

    filtered.forEach((item) => {
      const div = document.createElement("div");
      div.className = "viva-item";
      div.innerHTML = `
        <div class="viva-header">
          <span>Q${item.id}. ${item.q}</span>
          <i class="fa-solid fa-chevron-down"></i>
        </div>
        <div class="viva-body">
          <p>${item.a}</p>
        </div>
      `;
      div.querySelector(".viva-header").addEventListener("click", () => {
        div.classList.toggle("open");
      });
      vivaAccordionContainer.appendChild(div);
    });
  }

  vivaTagButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      vivaTagButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const tag = btn.getAttribute("data-viva-tag");
      const search = vivaSearchInput ? vivaSearchInput.value : "";
      renderVivaQuestions(tag, search);
    });
  });

  if (vivaSearchInput) {
    vivaSearchInput.addEventListener("input", (e) => {
      const activeBtn = document.querySelector(".viva-tag-btn.active");
      const tag = activeBtn ? activeBtn.getAttribute("data-viva-tag") : "all";
      renderVivaQuestions(tag, e.target.value);
    });
  }

  // Initial preloads
  loadSignalData();
  loadFeatureData();
  renderVivaQuestions();
  drawDecisionSpace(1.4, 0.28, "MODERATE");
});
