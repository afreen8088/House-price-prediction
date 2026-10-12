/**
 * HOMESTEAD — Minimal House Price Prediction Application
 * Handles input validation, API calls to FastAPI /predict, and INR result display.
 */

document.addEventListener("DOMContentLoaded", () => {
  const API_BASE = window.location.origin && !window.location.origin.startsWith("file")
    ? window.location.origin
    : "http://localhost:8000";

  // Form Elements
  const form = document.getElementById("prediction-form");
  const selectLocation = document.getElementById("location");
  const inputSqft = document.getElementById("sqft");
  const inputBedrooms = document.getElementById("bedrooms");
  const inputBathrooms = document.getElementById("bathrooms");
  const btnPredict = document.getElementById("btn-predict");
  const btnText = btnPredict.querySelector(".btn-text");
  const btnSpinner = btnPredict.querySelector(".btn-spinner");
  const formAlert = document.getElementById("form-alert");

  // Error Containers
  const errorLocation = document.getElementById("error-location");
  const errorSqft = document.getElementById("error-sqft");
  const errorBedrooms = document.getElementById("error-bedrooms");
  const errorBathrooms = document.getElementById("error-bathrooms");

  // Result Elements
  const resultSection = document.getElementById("result-section");
  const resultPrice = document.getElementById("result-price");
  const resultLocation = document.getElementById("result-location");
  const resultArea = document.getElementById("result-area");
  const resultRooms = document.getElementById("result-rooms");
  const resultRate = document.getElementById("result-rate");

  // Indian Rupee Formatter helper (Indian numbering system: Lakhs, Crores)
  function formatINR(amount) {
    if (isNaN(amount) || amount === null || amount === undefined) return "—";
    const num = Math.round(Number(amount));
    if (num < 0) return "-" + formatINR(-num);
    const s = String(num);
    if (s.length <= 3) return "₹" + s;
    const last3 = s.slice(-3);
    let remaining = s.slice(0, -3);
    const groups = [];
    while (remaining.length > 2) {
      groups.unshift(remaining.slice(-2));
      remaining = remaining.slice(0, -2);
    }
    if (remaining.length > 0) {
      groups.unshift(remaining);
    }
    return "₹" + groups.join(",") + "," + last3;
  }

  function clearErrors() {
    [errorLocation, errorSqft, errorBedrooms, errorBathrooms].forEach(el => {
      if (el) el.textContent = "";
    });
    [selectLocation, inputSqft, inputBedrooms, inputBathrooms].forEach(el => {
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

    // Location
    const loc = selectLocation.value;
    if (!loc || loc.trim() === "") {
      errorLocation.textContent = "Please select a location in Karnataka.";
      selectLocation.classList.add("is-invalid");
      isValid = false;
    }

    // Area (sqft)
    const sqft = parseFloat(inputSqft.value);
    if (isNaN(sqft) || sqft <= 100 || sqft >= 25000) {
      errorSqft.textContent = "Please enter a valid area between 101 and 25,000 sq ft.";
      inputSqft.classList.add("is-invalid");
      isValid = false;
    }

    // Bedrooms
    const bedrooms = parseInt(inputBedrooms.value, 10);
    if (isNaN(bedrooms) || bedrooms < 1 || bedrooms > 20) {
      errorBedrooms.textContent = "Please enter 1 to 20 bedrooms.";
      inputBedrooms.classList.add("is-invalid");
      isValid = false;
    }

    // Bathrooms
    const bathrooms = parseFloat(inputBathrooms.value);
    if (isNaN(bathrooms) || bathrooms < 0.5 || bathrooms > 15) {
      errorBathrooms.textContent = "Please enter 0.5 to 15 bathrooms.";
      inputBathrooms.classList.add("is-invalid");
      isValid = false;
    }

    return isValid;
  }

  // Handle Form Submission
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!validateInputs()) {
      return;
    }

    const payload = {
      location: selectLocation.value.trim(),
      sqft: parseFloat(inputSqft.value),
      bedrooms: parseInt(inputBedrooms.value, 10),
      bathrooms: parseFloat(inputBathrooms.value)
    };

    // UI Loading State
    btnPredict.disabled = true;
    btnText.textContent = "Estimating Price...";
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
        const errorData = await response.json().catch(() => ({}));
        const message = errorData.detail || `Server responded with status ${response.status}`;
        throw new Error(message);
      }

      const data = await response.json();
      displayResult(data, payload);

    } catch (err) {
      formAlert.style.display = "block";
      formAlert.textContent = `Unable to generate estimate: ${err.message}`;
    } finally {
      btnPredict.disabled = false;
      btnText.textContent = "Predict Price";
      btnSpinner.style.display = "none";
    }
  });

  function displayResult(data, req) {
    const details = data.property_details || req;

    // Display formatted price in ₹
    resultPrice.textContent = data.formatted_price || formatINR(data.predicted_price);
    resultLocation.textContent = details.location || req.location;
    resultArea.textContent = `${Number(details.sqft || req.sqft).toLocaleString("en-IN")} sq ft`;
    resultRooms.textContent = `${details.bedrooms || req.bedrooms} Beds • ${details.bathrooms || req.bathrooms} Baths`;

    // Display price per sq ft
    if (data.formatted_price_per_sqft) {
      resultRate.textContent = data.formatted_price_per_sqft;
    } else if (data.predicted_price && req.sqft) {
      const rate = Math.round(data.predicted_price / req.sqft);
      resultRate.textContent = `${formatINR(rate)}/sq ft`;
    } else {
      resultRate.textContent = "—";
    }

    resultSection.style.display = "block";
    resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
});
