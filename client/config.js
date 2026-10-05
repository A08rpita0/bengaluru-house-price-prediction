// Runtime configuration. Edit apiBaseUrl at deploy time (e.g. your Render backend URL).
window.APP_CONFIG = Object.assign(
  { apiBaseUrl: "http://127.0.0.1:5000" },
  window.APP_CONFIG || {}
);
