"use strict";

(function () {
  const API = (window.APP_CONFIG && window.APP_CONFIG.apiBaseUrl || "").replace(/\/+$/, "");
  const RULES = { minSqft: 300, maxSqft: 30000, minSqftPerBhk: 300, minBhk: 1, maxBhk: 10, maxExtraBaths: 1 };
  const FIELDS = ["location", "total_sqft", "bhk", "bath"];

  const $ = (id) => document.getElementById(id);
  const form = $("estimate-form");
  const button = $("submit-btn");

  async function api(path, options) {
    let res;
    try {
      res = await fetch(API + path, options);
    } catch (err) {
      throw { kind: "network" };
    }
    let body = null;
    try { body = await res.json(); } catch (err) { /* non-JSON body */ }
    if (!res.ok) throw { kind: "http", status: res.status, body: body };
    return body;
  }

  function showApiError(message) {
    const el = $("api-status");
    el.textContent = message;
    el.hidden = !message;
  }

  function setFieldError(name, message) {
    const input = $(name);
    $(name + "-error").textContent = message || "";
    if (message) input.setAttribute("aria-invalid", "true");
    else input.removeAttribute("aria-invalid");
  }

  function clearErrors() {
    FIELDS.forEach((f) => setFieldError(f, ""));
    showApiError("");
  }

  function validate(values) {
    const errors = {};
    if (!values.location) errors.location = "Location is required.";
    const sqft = Number(values.total_sqft);
    const bhk = Number(values.bhk);
    const bath = Number(values.bath);
    if (values.total_sqft === "" || !Number.isFinite(sqft)) errors.total_sqft = "Area must be a number.";
    else if (sqft < RULES.minSqft || sqft > RULES.maxSqft)
      errors.total_sqft = `Area must be between ${RULES.minSqft} and ${RULES.maxSqft} sq ft.`;
    if (!Number.isInteger(bhk) || bhk < RULES.minBhk || bhk > RULES.maxBhk)
      errors.bhk = `BHK must be a whole number from ${RULES.minBhk} to ${RULES.maxBhk}.`;
    if (!Number.isInteger(bath) || bath < 1) errors.bath = "Bathrooms must be a whole number of at least 1.";
    if (!errors.bhk && !errors.bath && bath > bhk + RULES.maxExtraBaths)
      errors.bath = `Bathrooms can be at most BHK + ${RULES.maxExtraBaths} (${bhk + RULES.maxExtraBaths}).`;
    if (!errors.total_sqft && !errors.bhk && sqft / bhk < RULES.minSqftPerBhk)
      errors.total_sqft = `At least ${RULES.minSqftPerBhk} sq ft per bedroom (${RULES.minSqftPerBhk * bhk} for ${bhk} BHK).`;
    return errors;
  }

  const inr = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });

  function formatLakhs(lakhs) {
    if (lakhs >= 100) return `₹${(lakhs / 100).toFixed(2)} crore`;
    return `₹${lakhs.toFixed(1)} lakh`;
  }

  function renderResult(r) {
    $("result-empty").hidden = true;
    $("result").hidden = false;
    $("estimate-main").textContent = formatLakhs(r.estimate_lakhs);
    $("estimate-sub").textContent = `${r.estimate_lakhs.toFixed(2)} lakhs (₹${inr.format(r.estimate_lakhs * 100000)})`;
    $("estimate-range").textContent = `${formatLakhs(r.range_lakhs.low)} to ${formatLakhs(r.range_lakhs.high)}`;
    $("estimate-pps").textContent = `₹${inr.format(r.price_per_sqft_inr)}`;
    $("range-note").textContent =
      `About ${Math.round(r.range_lakhs.coverage * 100)}% of held-out listings fell inside ranges like this one.`;
    $("unknown-location").hidden = r.location_known;
  }

  async function onSubmit(event) {
    event.preventDefault();
    clearErrors();
    const values = {
      location: $("location").value.trim(),
      total_sqft: $("total_sqft").value.trim(),
      bhk: $("bhk").value.trim(),
      bath: $("bath").value.trim(),
    };
    const errors = validate(values);
    if (Object.keys(errors).length) {
      Object.entries(errors).forEach(([k, v]) => setFieldError(k, v));
      $(Object.keys(errors)[0]).focus();
      return;
    }

    button.disabled = true;
    button.textContent = "Estimating...";
    try {
      const result = await api("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          location: values.location,
          total_sqft: Number(values.total_sqft),
          bhk: Number(values.bhk),
          bath: Number(values.bath),
        }),
      });
      renderResult(result);
    } catch (err) {
      if (err.kind === "http" && err.status === 400 && err.body && err.body.fields) {
        Object.entries(err.body.fields).forEach(([k, v]) => setFieldError(k, v));
      } else if (err.kind === "network") {
        showApiError("Cannot reach the prediction service. Check your connection or try again shortly.");
      } else {
        showApiError((err.body && err.body.message) || "The prediction service returned an error.");
      }
    } finally {
      button.disabled = false;
      button.textContent = "Estimate price";
    }
  }

  async function loadLocations() {
    try {
      const data = await api("/api/locations");
      const list = $("location-options");
      const frag = document.createDocumentFragment();
      data.locations.forEach((name) => {
        const opt = document.createElement("option");
        opt.value = name;
        frag.appendChild(opt);
      });
      list.replaceChildren(frag);
    } catch (err) {
      showApiError("Cannot reach the prediction service. Location suggestions and estimates are unavailable.");
    }
  }

  function pct(x) { return `${(x * 100).toFixed(1)}%`; }

  async function loadModelInfo() {
    const box = $("model-info");
    try {
      const m = await api("/api/model");
      const h = m.test_metrics.honest;
      const rows = m.cv_results.map((c) =>
        `<tr${c.model === m.model_name ? ' class="selected"' : ""}><td>${c.model}</td>` +
        `<td>${c.cv_mae_mean.toFixed(2)}</td><td>${c.cv_r2_mean.toFixed(3)}</td></tr>`).join("");
      box.innerHTML =
        `<p>Selected model: <strong>${m.model_description}</strong>. On ${h.n} held-out listings it has ` +
        `a mean absolute error of <strong>${h.mae_lakhs.toFixed(1)} lakhs</strong>, a median percentage error of ` +
        `<strong>${pct(h.median_ape)}</strong> and R² of <strong>${h.r2.toFixed(3)}</strong>.</p>` +
        `<div class="table-wrap"><table><caption>5-fold cross-validation on the training split</caption>` +
        `<thead><tr><th scope="col">Model</th><th scope="col">MAE (lakhs)</th><th scope="col">R²</th></tr></thead>` +
        `<tbody>${rows}</tbody></table></div>`;
    } catch (err) {
      box.innerHTML = '<p class="hint">Model details are unavailable while the service is unreachable.</p>';
    }
  }

  form.addEventListener("submit", onSubmit);
  FIELDS.forEach((f) => $(f).addEventListener("input", () => setFieldError(f, "")));
  loadLocations();
  loadModelInfo();
})();
