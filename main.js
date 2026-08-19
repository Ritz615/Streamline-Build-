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
