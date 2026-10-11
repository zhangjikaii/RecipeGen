(() => {
  "use strict";
  const button = document.getElementById("theme-toggle");
  if (!button) return;
  const colorScheme = window.matchMedia("(prefers-color-scheme: dark)");
  const isDark = () => document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : colorScheme.matches;
  const updateTheme = () => {
    const dark = isDark();
    button.textContent = dark ? "浅色" : "深色";
    button.setAttribute("aria-label", dark ? "切换浅色模式" : "切换深色模式");
    button.setAttribute("aria-pressed", String(dark));
    const themeColor = document.querySelector('meta[name="theme-color"]');
    if (themeColor) themeColor.content = dark ? "#20231e" : "#f7f4ed";
  };
  button.addEventListener("click", () => {
    document.documentElement.dataset.theme = isDark() ? "light" : "dark";
    try { localStorage.setItem("recipegen-theme", document.documentElement.dataset.theme); } catch (_) {}
    updateTheme();
  });
  colorScheme.addEventListener("change", updateTheme);
  window.addEventListener("storage", (event) => {
    if (event.key !== "recipegen-theme") return;
    if (event.newValue === "light" || event.newValue === "dark") document.documentElement.dataset.theme = event.newValue;
    else delete document.documentElement.dataset.theme;
    updateTheme();
  });
  updateTheme();
})();
