/**
 * Intelligence Designed To Evolve — Main JavaScript
 * Handles Stats Counter Animation and Mobile Navigation Menu
 */

document.addEventListener("DOMContentLoaded", () => {
  initStatsCounter();
  initMobileMenu();
  initNavLinks();
});

/* --------------------------------------------------------------------------
   Stats Count-Up Animation
   -------------------------------------------------------------------------- */
function initStatsCounter() {
  const statValues = document.querySelectorAll(".stat-value");
  if (!statValues.length) return;

  const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);

  function animateCounter(el, index) {
    const target = parseFloat(el.getAttribute("data-target")) || 0;
    const decimals = parseInt(el.getAttribute("data-decimals"), 10) || 0;
    const duration = 1500 + index * 80;
    const startDelay = 480 + index * 90;

    let startTime = null;

    function step(timestamp) {
      if (!startTime) startTime = timestamp;
      const elapsed = timestamp - startTime;

      if (elapsed < startDelay) {
        requestAnimationFrame(step);
        return;
      }

      const progress = Math.min((elapsed - startDelay) / duration, 1);
      const easedProgress = easeOutCubic(progress);
      const currentValue = easedProgress * target;

      el.textContent = currentValue.toFixed(decimals);

      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        el.textContent = target.toFixed(decimals);
      }
    }

    requestAnimationFrame(step);
  }

  // Use IntersectionObserver with threshold 0.25 to trigger count-up
  const statsContainer = document.querySelector(".stats");
  if (statsContainer && "IntersectionObserver" in window) {
    let triggered = false;
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !triggered) {
            triggered = true;
            statValues.forEach((el, idx) => animateCounter(el, idx));
            observer.disconnect();
          }
        });
      },
      { threshold: 0.25 }
    );
    observer.observe(statsContainer);
  } else {
    // Fallback if IntersectionObserver is not supported
    statValues.forEach((el, idx) => animateCounter(el, idx));
  }
}

/* --------------------------------------------------------------------------
   Mobile Navigation Menu
   -------------------------------------------------------------------------- */
function initMobileMenu() {
  const menuBtn = document.querySelector(".menu-btn");
  const mobileMenu = document.getElementById("mobile-menu");
  const mobileOverlay = document.getElementById("mobile-overlay");

  if (!menuBtn || !mobileMenu || !mobileOverlay) return;

  function openMenu() {
    menuBtn.setAttribute("aria-expanded", "true");
    mobileMenu.removeAttribute("hidden");
    mobileOverlay.removeAttribute("hidden");
    document.body.classList.add("menu-open");
  }

  function closeMenu() {
    menuBtn.setAttribute("aria-expanded", "false");
    mobileMenu.setAttribute("hidden", "");
    mobileOverlay.setAttribute("hidden", "");
    document.body.classList.remove("menu-open");
  }

  function toggleMenu() {
    const isExpanded = menuBtn.getAttribute("aria-expanded") === "true";
    if (isExpanded) {
      closeMenu();
    } else {
      openMenu();
    }
  }

  menuBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleMenu();
  });

  mobileOverlay.addEventListener("click", () => {
    closeMenu();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && menuBtn.getAttribute("aria-expanded") === "true") {
      closeMenu();
    }
  });

  window.addEventListener("resize", () => {
    if (window.innerWidth > 720 && menuBtn.getAttribute("aria-expanded") === "true") {
      closeMenu();
    }
  });

  // Close when clicking any link inside mobile sheet
  const mobileLinks = mobileMenu.querySelectorAll("a");
  mobileLinks.forEach((link) => {
    link.addEventListener("click", () => {
      closeMenu();
    });
  });
}

/* --------------------------------------------------------------------------
   Active Navigation Link Handling
   -------------------------------------------------------------------------- */
function initNavLinks() {
  const allNavLinks = document.querySelectorAll(".nav-link, .mobile-nav-link");

  allNavLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      const href = link.getAttribute("href");
      if (href && href.startsWith("#")) {
        // Update active class on matching links across desktop and mobile
        allNavLinks.forEach((l) => {
          if (l.getAttribute("href") === href) {
            l.classList.add("active");
          } else {
            l.classList.remove("active");
          }
        });
      }
    });
  });
}

/* --------------------------------------------------------------------------
   Live Fuzzy Cognitive Load Inference Simulator
   -------------------------------------------------------------------------- */
function initInferenceModal() {
  const modal = document.getElementById("inference-modal");
  const backdrop = document.getElementById("modal-backdrop");
  const closeBtn = document.getElementById("modal-close-btn");
  const ctaBtn = document.querySelector(".cta-btn");

  if (!modal) return;

  function openModal() {
    modal.removeAttribute("hidden");
    document.body.classList.add("menu-open");
    runInference();
  }

  function closeModal() {
    modal.setAttribute("hidden", "");
    document.body.classList.remove("menu-open");
  }

  // Open modal on CTA click or product click
  if (ctaBtn) {
    ctaBtn.addEventListener("click", (e) => {
      e.preventDefault();
      openModal();
    });
  }

  if (closeBtn) closeBtn.addEventListener("click", closeModal);
  if (backdrop) backdrop.addEventListener("click", closeModal);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.hasAttribute("hidden")) {
      closeModal();
    }
  });

  // Inputs
  const sAlpha = document.getElementById("slider-alpha");
  const sTar = document.getElementById("slider-tar");
  const sBar = document.getElementById("slider-bar");
  const sEnt = document.getElementById("slider-ent");

  const vAlpha = document.getElementById("val-alpha");
  const vTar = document.getElementById("val-tar");
  const vBar = document.getElementById("val-bar");
  const vEnt = document.getElementById("val-ent");

  const predCard = document.getElementById("pred-card");
  const predClass = document.getElementById("pred-class-text");
  const predScore = document.getElementById("pred-score-text");
  const predConf = document.getElementById("pred-conf-text");
  const predRule = document.getElementById("pred-rule-text");

  function runInference() {
    const alpha = parseFloat(sAlpha.value);
    const tar = parseFloat(sTar.value);
    const bar = parseFloat(sBar.value);
    const ent = parseFloat(sEnt.value);

    vAlpha.textContent = alpha.toFixed(2);
    vTar.textContent = tar.toFixed(2);
    vBar.textContent = bar.toFixed(2);
    vEnt.textContent = ent.toFixed(2);

    // Triangular membership function helper
    const trimf = (x, a, b, c) => Math.max(0, Math.min((x - a) / (b - a || 0.001), (c - x) / (c - b || 0.001)));

    // Membership values
    const alphaLow = trimf(alpha, 0.05, 0.12, 0.22);
    const alphaMed = trimf(alpha, 0.18, 0.28, 0.38);
    const alphaHigh = trimf(alpha, 0.32, 0.45, 0.60);

    const tarLow = trimf(tar, 0.2, 0.7, 1.2);
    const tarMed = trimf(tar, 1.0, 1.5, 2.0);
    const tarHigh = trimf(tar, 1.8, 2.6, 3.5);

    // Workload score aggregation
    const lowWeight = Math.max(alphaHigh, tarLow);
    const medWeight = Math.max(alphaMed, tarMed);
    const highWeight = Math.max(alphaLow, tarHigh);

    const sumWeights = lowWeight + medWeight + highWeight || 1;
    const score = ((lowWeight * 20) + (medWeight * 50) + (highWeight * 85)) / sumWeights;

    let cls = "MODERATE";
    let ruleText = "IF Alpha is Medium AND Theta/Alpha is Medium THEN Workload is MODERATE";
    let conf = Math.max(0.65, Math.min(0.95, (Math.max(lowWeight, medWeight, highWeight) / sumWeights) * 0.95));

    if (score < 36) {
      cls = "LOW";
      ruleText = "IF Alpha is High (Synchronized) AND Theta/Alpha is Low THEN Workload is LOW";
    } else if (score > 64) {
      cls = "HIGH";
      ruleText = "IF Alpha is Low (Suppressed) AND Theta/Alpha is High THEN Workload is HIGH";
    }

    predClass.textContent = cls;
    predScore.textContent = `Score: ${score.toFixed(1)} / 100`;
    predConf.textContent = `Confidence: ${(conf * 100).toFixed(1)}%`;
    predRule.textContent = ruleText;

    predCard.className = `prediction-output-card class-${cls}`;
  }

  [sAlpha, sTar, sBar, sEnt].forEach((slider) => {
    if (slider) slider.addEventListener("input", runInference);
  });

  // Presets
  const presetBtns = document.querySelectorAll(".preset-btn");
  presetBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      presetBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      sAlpha.value = btn.dataset.alpha;
      sTar.value = btn.dataset.tar;
      sBar.value = btn.dataset.bar;
      sEnt.value = btn.dataset.ent;

      runInference();
    });
  });
}

// Ensure initInferenceModal is called on DOM load
document.addEventListener("DOMContentLoaded", () => {
  initInferenceModal();
});

