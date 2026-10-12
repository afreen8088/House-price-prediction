/**
 * HOMESTEAD — Editorial Property Valuation Frontend Application
 * Interacts with FastAPI /predict, /model-info, and /health endpoints.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Determine API Base URL dynamically
  const API_BASE = window.location.origin && window.location.origin !== "null" && !window.location.origin.startsWith("file")
    ? window.location.origin
    : "http://localhost:8000";

  // State
  let currentView = "home";

  // Elements
  const navLinks = document.querySelectorAll(".nav-link, .footer-nav-btn");
  const pageViews = document.querySelectorAll(".page-view");
  const navBrand = document.getElementById("nav-brand");
  const navEstimateBtn = document.getElementById("nav-estimate-btn");
  const heroCtaBtn = document.getElementById("hero-cta-btn");
  const heroInsightsBtn = document.getElementById("hero-insights-btn");
  const insightsPredictBtn = document.getElementById("insights-predict-btn");
  const insightsRefreshBtn = document.getElementById("insights-refresh-btn");
  const btnTryAnother = document.getElementById("btn-try-another");
  const btnResultToInsights = document.getElementById("btn-result-to-insights");

  // Form Elements
  const form = document.getElementById("valuation-form");
  const inputSqft = document.getElementById("input-sqft");
  const inputBedrooms = document.getElementById("input-bedrooms");
  const inputBathrooms = document.getElementById("input-bathrooms");
  const inputNeighborhood = document.getElementById("input-neighborhood");
  const submitBtn = document.getElementById("btn-submit-prediction");
  const btnText = submitBtn.querySelector(".btn-text");
  const btnSpinner = submitBtn.querySelector(".btn-spinner");
  const formAlert = document.getElementById("form-api-alert");

  // Error Containers
  const errorSqft = document.getElementById("error-sqft");
  const errorBedrooms = document.getElementById("error-bedrooms");
  const errorBathrooms = document.getElementById("error-bathrooms");
  const errorNeighborhood = document.getElementById("error-neighborhood");

  // Backend Status Indicator
  const backendStatusText = document.getElementById("backend-status-text");
  const statusDot = document.querySelector(".status-indicator-dot");

  // --- View Router ---
  function switchView(viewName) {
    currentView = viewName;

    // Update active view section
    pageViews.forEach(view => {
      view.classList.toggle("active", view.id === `view-${viewName}`);
    });

    // Update nav links active class
    document.querySelectorAll(".nav-link").forEach(link => {
      link.classList.toggle("active", link.dataset.view === viewName);
    });

    window.scrollTo({ top: 0, behavior: "smooth" });

    // Fetch view-specific dynamic data
    if (viewName === "insights") {
      fetchModelInsights();
    }
  }

  // Bind Navigation Clicks
  navLinks.forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const view = link.dataset.view;
      if (view) switchView(view);
    });
  });

  if (navBrand) {
    navBrand.addEventListener("click", (e) => {
      e.preventDefault();
      switchView("home");
    });
  }

  if (navEstimateBtn) {
    navEstimateBtn.addEventListener("click", () => switchView("predict"));
  }

  if (heroCtaBtn) {
    heroCtaBtn.addEventListener("click", () => switchView("predict"));
  }

  if (heroInsightsBtn) {
    heroInsightsBtn.addEventListener("click", () => switchView("insights"));
  }

  if (insightsPredictBtn) {
    insightsPredictBtn.addEventListener("click", () => switchView("predict"));
  }

  if (insightsRefreshBtn) {
    insightsRefreshBtn.addEventListener("click", () => fetchModelInsights());
  }

  if (btnTryAnother) {
    btnTryAnother.addEventListener("click", () => {
      form.reset();
      clearErrors();
      switchView("predict");
    });
  }

  if (btnResultToInsights) {
    btnResultToInsights.addEventListener("click", () => switchView("insights"));
  }

  // --- Health Probe ---
  async function checkApiHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        if (backendStatusText) backendStatusText.textContent = "Backend & Model Ready";
        if (statusDot) {
          statusDot.className = "status-indicator-dot online";
        }
      } else {
        const data = await res.json().catch(() => ({}));
        if (backendStatusText) backendStatusText.textContent = "Model Offline (Need Training)";
        if (statusDot) {
          statusDot.className = "status-indicator-dot offline";
        }
      }
    } catch (err) {
      if (backendStatusText) backendStatusText.textContent = "API Unreachable";
      if (statusDot) {
        statusDot.className = "status-indicator-dot offline";
      }
    }
  }

  // --- Input Validation ---
  function clearErrors() {
    [errorSqft, errorBedrooms, errorBathrooms, errorNeighborhood].forEach(el => {
      if (el) el.textContent = "";
    });
    [inputSqft, inputBedrooms, inputBathrooms, inputNeighborhood].forEach(el => {
      if (el) el.classList.remove("is-invalid");
    });
    if (formAlert) {
      formAlert.style.display = "none";
      formAlert.textContent = "";
    }
  }

  function validateInputs() {
    clearErrors();
    let isValid = true;

    const sqft = parseFloat(inputSqft.value);
    if (isNaN(sqft) || sqft < 150 || sqft > 20000) {
      errorSqft.textContent = "Enter a valid interior area between 150 and 20,000 sq ft.";
      inputSqft.classList.add("is-invalid");
      isValid = false;
    }

    const bedrooms = parseInt(inputBedrooms.value, 10);
    if (isNaN(bedrooms) || bedrooms < 1 || bedrooms > 15) {
      errorBedrooms.textContent = "Enter a realistic bedroom count between 1 and 15.";
      inputBedrooms.classList.add("is-invalid");
      isValid = false;
    }

    const bathrooms = parseFloat(inputBathrooms.value);
    if (isNaN(bathrooms) || bathrooms < 0.5 || bathrooms > 12) {
      errorBathrooms.textContent = "Enter a valid bathroom count (minimum 0.5, e.g. 1.5, 2.0).";
      inputBathrooms.classList.add("is-invalid");
      isValid = false;
    }

    const neighborhood = inputNeighborhood.value;
    if (!neighborhood || neighborhood.trim() === "") {
      errorNeighborhood.textContent = "Please select a neighborhood district.";
      inputNeighborhood.classList.add("is-invalid");
      isValid = false;
    }

    return isValid;
  }

  // --- Form Submission to /predict ---
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!validateInputs()) {
      return;
    }

    const payload = {
      sqft: parseFloat(inputSqft.value),
      bedrooms: parseInt(inputBedrooms.value, 10),
      bathrooms: parseFloat(inputBathrooms.value),
      neighborhood: inputNeighborhood.value.trim()
    };

    // UI Loading State
    submitBtn.disabled = true;
    btnText.textContent = "Computing Valuation...";
    btnSpinner.style.display = "inline-block";
    formAlert.style.display = "none";

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Unknown server error" }));
        const errorMessage = errorData.detail || `Server responded with status ${response.status}`;
        throw new Error(errorMessage);
      }

      const result = await response.json();
      displayPredictionResult(result);
      switchView("result");

    } catch (error) {
      formAlert.className = "api-alert error";
      formAlert.style.display = "block";
      formAlert.textContent = `Valuation Request Failed: ${error.message}`;
    } finally {
      submitBtn.disabled = false;
      btnText.textContent = "Generate Valuation";
      btnSpinner.style.display = "none";
    }
  });

  // --- Render Result View (Page C) ---
  function displayPredictionResult(data) {
    document.getElementById("res-price").textContent = data.formatted_price;
    document.getElementById("res-summary").textContent = data.summary;

    const details = data.property_details;
    document.getElementById("res-sqft").textContent = `${Number(details.sqft).toLocaleString()} sq ft`;
    document.getElementById("res-bedrooms").textContent = details.bedrooms;
    document.getElementById("res-bathrooms").textContent = details.bathrooms;
    document.getElementById("res-neighborhood").textContent = details.neighborhood;

    const dateStr = new Date(data.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    document.getElementById("res-timestamp").textContent = `Derived ${dateStr}`;

    if (data.disclaimer) {
      document.getElementById("res-disclaimer").textContent = data.disclaimer;
    }
  }

  // --- Render Model Insights (Page D) ---
  async function fetchModelInsights() {
    try {
      const res = await fetch(`${API_BASE}/model-info`);
      if (!res.ok) throw new Error("Unable to retrieve model metadata");

      const info = await res.json();

      document.getElementById("insight-model-name").textContent = info.model_name || "Not available";
      document.getElementById("insight-model-version").textContent = `Version ${info.model_version || "1.0.0"}`;

      document.getElementById("insight-r2").textContent = 
        typeof info.r2_score === "number" ? info.r2_score.toFixed(4) : (info.r2_score || "Not available");

      document.getElementById("insight-rmse").textContent = 
        typeof info.rmse === "number" ? `$${info.rmse.toLocaleString()}` : (info.rmse || "Not available");

      document.getElementById("insight-mae").textContent = 
        typeof info.mae === "number" ? `$${info.mae.toLocaleString()}` : (info.mae || "Not available");

      document.getElementById("insight-timestamp").textContent = 
        info.training_timestamp !== "Not available" 
          ? new Date(info.training_timestamp).toLocaleString() 
          : "Not available";

      document.getElementById("insight-dataset").textContent = info.dataset_version || "Not available";
      document.getElementById("insight-train-samples").textContent = 
        info.train_samples !== "Not available" ? Number(info.train_samples).toLocaleString() : "Not available";
      document.getElementById("insight-test-samples").textContent = 
        info.test_samples !== "Not available" ? Number(info.test_samples).toLocaleString() : "Not available";

      const statusEl = document.getElementById("insight-status");
      if (info.model_available) {
        statusEl.innerHTML = `<span class="badge badge-success">Active & Verified</span>`;
      } else {
        statusEl.innerHTML = `<span class="badge" style="background: #fde8e8; color: #9c1c1c;">Not Loaded</span>`;
      }

      const paramsBox = document.getElementById("insight-hyperparams");
      if (info.experiment_parameters && typeof info.experiment_parameters === "object") {
        paramsBox.textContent = JSON.stringify(info.experiment_parameters, null, 2);
      } else {
        paramsBox.textContent = String(info.experiment_parameters || "Not available");
      }

    } catch (err) {
      console.warn("Could not load model info:", err);
      document.getElementById("insight-model-name").textContent = "Not available";
      document.getElementById("insight-r2").textContent = "Not available";
      document.getElementById("insight-rmse").textContent = "Not available";
      document.getElementById("insight-mae").textContent = "Not available";
    }
  }

  // Initial Probe
  checkApiHealth();
});
