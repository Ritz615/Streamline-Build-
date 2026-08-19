/**
 * Intelligence Designed To Evolve — Unified EEG Research Platform Frontend
 * =========================================================================
 * Full-featured interactive client script connecting the approved UI/UX
 * to real Python backend endpoints.
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

  // Mobile menu elements
  const mobileToggle = document.getElementById("mobile-menu-toggle");
  const mobileSheet = document.getElementById("mobile-menu");
  const mobileOverlay = document.getElementById("mobile-overlay");

  // Signals Elements
  const signalSubjectSelect = document.getElementById("signal-subject-select");
  const signalTimeSlider = document.getElementById("signal-time-slider");
  const signalTimeVal = document.getElementById("signal-time-val");
  const btnLoadSignal = document.getElementById("btn-load-signal");
  const waveformCanvas = document.getElementById("waveform-canvas");

  // Feature Elements
  const featureDropdown = document.getElementById("feature-select-dropdown");

  // Prediction Elements
  const btnModeDataset = document.getElementById("btn-mode-dataset");
  const btnModeManual = document.getElementById("btn-mode-manual");
  const datasetPickerControls = document.getElementById("dataset-picker-controls");
  const datasetWindowSelect = document.getElementById("dataset-window-select");
  const manualSlidersGrid = document.getElementById("manual-sliders-grid");
  const btnRunPrediction = document.getElementById("btn-run-prediction");

  // Manual sliders
  const slAlpha = document.getElementById("sl-alpha");
  const slTar = document.getElementById("sl-tar");
  const slBar = document.getElementById("sl-bar");
  const slEnt = document.getElementById("sl-ent");
  const slValAlpha = document.getElementById("sl-val-alpha");
  const slValTar = document.getElementById("sl-val-tar");
  const slValBar = document.getElementById("sl-val-bar");
  const slValEnt = document.getElementById("sl-val-ent");

  // Prediction outputs
  const fuzzyClassBanner = document.getElementById("fuzzy-class-banner");
  const fuzzyScoreVal = document.getElementById("fuzzy-score-val");
  const fuzzyConfVal = document.getElementById("fuzzy-conf-val");
  const fuzzyMembershipsList = document.getElementById("fuzzy-memberships-list");
  const actRuleText = document.getElementById("act-rule-text");

  const rfClassBanner = document.getElementById("rf-class-banner");
  const rfConfVal = document.getElementById("rf-conf-val");
  const rfBarLow = document.getElementById("rf-bar-low");
  const rfBarMod = document.getElementById("rf-bar-mod");
  const rfBarHigh = document.getElementById("rf-bar-high");
  const rfValLow = document.getElementById("rf-val-low");
  const rfValMod = document.getElementById("rf-val-mod");
  const rfValHigh = document.getElementById("rf-val-high");

  // History & Refresh
  const btnRefreshHistory = document.getElementById("btn-refresh-history");
  const historyTbody = document.getElementById("history-table-tbody");

  let currentPredictionMode = "dataset"; // "dataset" or "manual"
  let cachedFeaturesData = null;

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

    // Auto-fetch data if switching to specific tabs
    if (tabId === "tab-signals") loadSignalData();
    if (tabId === "tab-features") loadFeatureData();
    if (tabId === "tab-history") loadHistoryData();
  }

  function updateActiveNav(targetKey) {
    desktopNavLinks.forEach((link) => {
      link.classList.toggle("active", link.getAttribute("data-target") === targetKey);
    });
  }

  // Navigation click bindings
  desktopNavLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const target = link.getAttribute("data-target");
      if (target === "hero") {
        showView("hero");
      } else {
        showView("app", `tab-${target}`);
      }
    });
  });

  mobileNavLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      closeMobileMenu();
      const target = link.getAttribute("data-target");
      if (target === "hero") {
        showView("hero");
      } else {
        showView("app", `tab-${target}`);
      }
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

  // ------------------------------------------------------------------------
  // 3. Mobile Sheet Menu Controller
  // ------------------------------------------------------------------------
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

  // ------------------------------------------------------------------------
  // 4. Hero Stats Count-Up Animation
  // ------------------------------------------------------------------------
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
  // 5. Signal Viewer Canvas Drawing & API Fetch
  // ------------------------------------------------------------------------
  async function loadSignalData() {
    const sub = signalSubjectSelect.value || "sub-001";
    const start = parseFloat(signalTimeSlider.value) || 0;
    signalTimeVal.textContent = `${start.toFixed(1)}s – ${(start + 4.0).toFixed(1)}s`;

    try {
      const res = await fetch(`/api/eeg-data?subject=${sub}&start=${start}&duration=4.0`);
      if (!res.ok) throw new Error("Could not fetch EEG data");
      const data = await res.json();

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

    ctx.clearRect(0, 0, width, height);

    // Background Grid
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    if (!channels || channels.length === 0) return;

    const colors = ["#7ecef4", "#f6d86d", "#a3e635", "#f47a80", "#c084fc", "#38bdf8"];
    const rowHeight = height / channels.length;

    channels.forEach((ch, idx) => {
      const centerY = (idx + 0.5) * rowHeight;
      const values = ch.values;

      // Channel label
      ctx.fillStyle = "#8e8e8e";
      ctx.font = "11px Inter, sans-serif";
      ctx.fillText(ch.channel, 12, centerY - 8);

      // Channel center guideline
      ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();

      // Waveform trace
      ctx.strokeStyle = colors[idx % colors.length];
      ctx.lineWidth = 1.4;
      ctx.beginPath();

      const n = values.length;
      for (let i = 0; i < n; i++) {
        const x = (i / (n - 1)) * (width - 60) + 50;
        const scale = 2.5e5; // Microvolt scale
        const y = centerY - (values[i] * scale);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });
  }

  function renderSimulatedWaveforms() {
    const mockChannels = [
      { channel: "Fz", values: Array.from({ length: 400 }, (_, i) => Math.sin(i * 0.1) * 2e-5 + Math.cos(i * 0.03) * 1e-5) },
      { channel: "Cz", values: Array.from({ length: 400 }, (_, i) => Math.sin(i * 0.12) * 1.8e-5) },
      { channel: "Pz", values: Array.from({ length: 400 }, (_, i) => Math.cos(i * 0.08) * 2.2e-5) },
      { channel: "T5", values: Array.from({ length: 400 }, (_, i) => Math.sin(i * 0.15) * 1.5e-5) },
      { channel: "T6", values: Array.from({ length: 400 }, (_, i) => Math.cos(i * 0.14) * 1.6e-5) },
    ];
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

  if (btnLoadSignal) btnLoadSignal.addEventListener("click", loadSignalData);
  if (signalTimeSlider) {
    signalTimeSlider.addEventListener("input", () => {
      const val = parseFloat(signalTimeSlider.value);
      signalTimeVal.textContent = `${val.toFixed(1)}s – ${(val + 4.0).toFixed(1)}s`;
    });
    signalTimeSlider.addEventListener("change", loadSignalData);
  }
  if (signalSubjectSelect) signalSubjectSelect.addEventListener("change", loadSignalData);

  // ------------------------------------------------------------------------
  // 6. Feature Space Analysis & Boxplots
  // ------------------------------------------------------------------------
  async function loadFeatureData() {
    const feat = featureDropdown ? featureDropdown.value : "alpha_relative";
    try {
      const res = await fetch(`/api/features?feature=${feat}`);
      if (!res.ok) throw new Error("Could not fetch features");
      const data = await res.json();
      cachedFeaturesData = data;

      // Update sample windows dropdown if available
      if (data.sample_windows && datasetWindowSelect) {
        datasetWindowSelect.innerHTML = "";
        data.sample_windows.forEach((win) => {
          const opt = document.createElement("option");
          opt.value = `${win.subject_id}_${win.window_id}`;
          opt.textContent = `${win.subject_id} — Win ${win.window_id} (True: ${win.label})`;
          datasetWindowSelect.appendChild(opt);
        });
      }

      // Update box stats
      if (data.box_plot) {
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

  if (featureDropdown) featureDropdown.addEventListener("change", loadFeatureData);

  // ------------------------------------------------------------------------
  // 7. Prediction & Fuzzy Explainability Engine
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

  // Slider feedback
  [
    [slAlpha, slValAlpha],
    [slTar, slValTar],
    [slBar, slValBar],
    [slEnt, slValEnt],
  ].forEach(([slider, valSpan]) => {
    if (slider && valSpan) {
      slider.addEventListener("input", () => {
        valSpan.textContent = parseFloat(slider.value).toFixed(2);
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

    // Fuzzy Render
    fuzzyClassBanner.className = `pred-class-banner class-${fz.predicted_class}`;
    fuzzyClassBanner.textContent = fz.predicted_class;
    fuzzyScoreVal.textContent = `${fz.fuzzy_score} / 100`;
    fuzzyConfVal.textContent = `${fz.confidence}%`;

    // Memberships
    if (fz.memberships) {
      fuzzyMembershipsList.innerHTML = "";
      for (const [feat, mems] of Object.entries(fz.memberships)) {
        const div = document.createElement("div");
        div.className = "mem-item";
        const maxLevel = Object.keys(mems).reduce((a, b) => (mems[a] > mems[b] ? a : b));
        div.innerHTML = `<strong>${feat}</strong>: LOW=${mems.LOW.toFixed(2)}, MED=${mems.MEDIUM.toFixed(2)}, HIGH=${mems.HIGH.toFixed(2)} → <span style="color:#7ecef4; font-weight:600;">${maxLevel}</span>`;
        fuzzyMembershipsList.appendChild(div);
      }
    }

    // Activated Rules
    if (fz.activated_rules && fz.activated_rules.length > 0) {
      const topRule = fz.activated_rules[0];
      actRuleText.textContent = `Rule ${String(topRule.rule_number).padStart(2, "0")} (strength=${topRule.strength.toFixed(2)}): ${topRule.rule_text}`;
    } else {
      actRuleText.textContent = "Default baseline inference applied.";
    }

    // RF Render
    if (rf) {
      rfClassBanner.className = `pred-class-banner class-${rf.predicted_class}`;
      rfClassBanner.textContent = rf.predicted_class;
      rfConfVal.textContent = `${rf.confidence}%`;

      rfBarLow.style.width = `${rf.probabilities.LOW}%`;
      rfValLow.textContent = `${rf.probabilities.LOW}%`;

      rfBarMod.style.width = `${rf.probabilities.MODERATE}%`;
      rfValMod.textContent = `${rf.probabilities.MODERATE}%`;

      rfBarHigh.style.width = `${rf.probabilities.HIGH}%`;
      rfValHigh.textContent = `${rf.probabilities.HIGH}%`;
    }
  }

  if (btnRunPrediction) btnRunPrediction.addEventListener("click", runPrediction);

  // ------------------------------------------------------------------------
  // 8. Experiment History Database Fetch
  // ------------------------------------------------------------------------
  async function loadHistoryData() {
    try {
      const res = await fetch("/api/history");
      if (!res.ok) throw new Error("Could not fetch history");
      const data = await res.json();

      historyTbody.innerHTML = "";
      if (data.runs && data.runs.length > 0) {
        data.runs.forEach((r) => {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td><strong>#${r["Run ID"]}</strong></td>
            <td>${r["Model"]}</td>
            <td>${r["Version"]}</td>
            <td><strong>${(r["Accuracy"] * 100).toFixed(1)}%</strong></td>
            <td>${r["F1 (Macro)"]}</td>
            <td>${(r["Balanced Acc"] * 100).toFixed(1)}%</td>
            <td>${r["Folds"]}</td>
            <td>${r["Test Samples"]}</td>
            <td>${r["Date & Time (UTC)"]}</td>
          `;
          historyTbody.appendChild(tr);
        });
      } else {
        historyTbody.innerHTML = `<tr><td colspan="9" style="text-align:center; color:#8e8e8e;">No recorded runs found.</td></tr>`;
      }
    } catch (err) {
      console.warn("History fetch error:", err);
    }
  }

  if (btnRefreshHistory) btnRefreshHistory.addEventListener("click", loadHistoryData);

  // Initial data preloads
  loadSignalData();
  loadFeatureData();
});
