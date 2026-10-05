/* 真实 HTTP 浏览器验收。只调用 grounded 模式，不启动模型或媒体处理。
 * 运行：node scripts/verify_system_browser.cjs [--base-url http://127.0.0.1:8765]
 * 依赖已存在的 Playwright 与 Chrome；不会安装或下载依赖。
 */
"use strict";
const fs = require("fs"), path = require("path"), os = require("os"), assert = require("assert");
const root = path.resolve(__dirname, "..");
const args = process.argv.slice(2);
let baseURL = "http://127.0.0.1:8765";
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--base-url" && args[i + 1]) baseURL = args[++i];
  else throw new Error(`未知参数：${args[i]}`);
}
const base = new URL(baseURL);
assert(base.protocol === "http:" && ["127.0.0.1", "localhost", "[::1]"].includes(base.hostname) && !base.username && !base.password, "只允许无凭据的本机 HTTP 地址");
baseURL = base.origin;
function playwright() {
  const candidates = ["playwright", path.join(os.homedir(), ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright")];
  for (const candidate of candidates) { try { return require(candidate); } catch (error) { if (error.code !== "MODULE_NOT_FOUND") throw error; } }
  throw new Error("找不到已安装的 Playwright；请先配置运行环境。本脚本不会下载依赖。");
}
function chromePath() {
  const candidates = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser"];
  return candidates.find((candidate) => fs.existsSync(candidate));
}
const reportPath = path.join(root, "reports/system-browser-smoke.json"), screenshotDir = path.join(root, "reports/screenshots");
fs.mkdirSync(screenshotDir, { recursive: true });
const report = { schema_version: "recipegen-system-browser-v1", started_at: new Date().toISOString(), status: "running", base_url: baseURL,
  scope: "real_http_grounded_browser_smoke", local_model_requested: null, generation_requests: [], checks: {}, page_errors: [], console_errors: [], requests: [], request_failures: [], screenshots: {}, retries: [] };
let browser;
function check(name, condition) { report.checks[name] = Boolean(condition); assert(condition, name); }
function safeURL(value) { try { const url = new URL(value); return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password; } catch (_) { return false; } }
function detailBody(value) { return value.recipe || value; }
(async () => {
  const executablePath = chromePath();
  browser = await playwright().chromium.launch({ ...(executablePath ? { executablePath } : {}), headless: true });
  report.browser = { version: browser.version(), executable: executablePath || "installed_playwright_browser", isolated_context: true };
  const context = await browser.newContext({ viewport: { width: 1280, height: 1000 }, reducedMotion: "reduce" });
  const page = await context.newPage(); page.setDefaultTimeout(60000);
  page.on("pageerror", (error) => report.page_errors.push(error.message));
  page.on("console", (message) => { if (message.type() === "error") report.console_errors.push(message.text()); });
  page.on("response", (response) => { const url = new URL(response.url()); if (url.origin === baseURL) report.requests.push({ method: response.request().method(), path: url.pathname + url.search, status: response.status() }); });
  page.on("requestfailed", (request) => report.request_failures.push({ url: request.url(), error: request.failure()?.errorText || "unknown" }));
  // 防止验收流程意外访问外部网站；原来源链接只检查，不跳转。
  await context.route("**/*", async (route) => {
    const url = new URL(route.request().url());
    if (url.origin !== baseURL) { report.request_failures.push({ url: url.origin + url.pathname, error: "non_local_request_blocked" }); await route.abort(); }
    else {
      const request = route.request();
      if (url.pathname.startsWith("/api/")) {
        const readGET = request.method() === "GET" && (["/api/system/status", "/api/system/graph"].includes(url.pathname) || /^\/api\/recipes\/[^/]+$/.test(url.pathname));
        let readPOST = request.method() === "POST" && url.pathname === "/api/search";
        if (request.method() === "POST" && url.pathname === "/api/generate") {
          const payload = request.postDataJSON(); report.generation_requests.push({ mode: payload?.mode, recipe_ids: payload?.recipe_ids });
          report.local_model_requested = report.generation_requests.some((item) => item.mode !== "grounded");
          readPOST = payload?.mode === "grounded";
        }
        if (!readGET && !readPOST) { report.request_failures.push({ url: url.pathname, error: "non_readonly_or_model_request_blocked" }); await route.abort(); return; }
      }
      await route.continue();
    }
  });
  let status;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const pending = page.waitForResponse((response) => new URL(response.url()).pathname === "/api/system/status", { timeout: 20000 });
      await page.goto(baseURL, { waitUntil: "domcontentloaded", timeout: 20000 });
      const response = await pending; if (!response.ok()) throw new Error(`状态接口 HTTP ${response.status()}`);
      status = await response.json(); await page.locator("#stat-recipes").filter({ hasText: /\d/ }).waitFor(); break;
    } catch (error) { report.retries.push({ stage: "initial_connection", attempt, error: error.message }); if (attempt === 3) throw error; await new Promise((resolve) => setTimeout(resolve, 1000)); }
  }
  report.system = { backend: status.backend, dataset: status.dataset, graph: status.graph, generation: status.generation };
  check("real_neo4j_backend", status.backend === "neo4j" && status.dataset?.is_demo === false);
  check("actual_recipe_count_5898", status.graph?.recipes === 5898);
  check("displayed_recipe_count_matches_api", (await page.locator("#stat-recipes").innerText()).replace(/,/g, "") === String(status.graph.recipes));
  check("grounded_toggle_unchecked", !(await page.locator("#local-model").isChecked()));
  async function search(query, ingredients = "") {
    await page.fill("#search-question", query); await page.fill("#ingredients", ingredients); await page.fill("#excluded-ingredients", "");
    for (let attempt = 1; attempt <= 2; attempt++) {
      const pending = page.waitForResponse((response) => new URL(response.url()).pathname === "/api/search" && response.request().method() === "POST");
      await page.click("#search-button"); const response = await pending; const result = await response.json();
      await page.locator("#search-button:not([disabled])").waitFor();
      if (response.ok()) return result;
      report.retries.push({ stage: "search", attempt, status: response.status() });
      if (attempt === 2) throw new Error(`搜索 HTTP ${response.status()}`); await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }
  const found = await search("番茄 鸡蛋", "番茄、鸡蛋");
  report.search = { query: found.query, count: found.count, recipe_ids: found.recipes.map((recipe) => recipe.id), filters: found.applied_filters || found.filters, read_only: found.read_only };
  check("tomato_egg_search_has_real_results", found.count > 0 && found.recipes.length === found.count);
  check("rendered_cards_match_search", await page.locator(".recipe-card").count() === found.recipes.length);
  const choices = found.recipes.map((recipe, index) => ({ recipe, index })).filter(({ recipe }) => recipe.steps?.length && recipe.sources?.length).sort((a, b) => a.recipe.steps.length - b.recipe.steps.length);
  check("recipe_with_original_steps_and_sources", choices.length > 0);
  const chosen = choices[0];
  await page.locator(".select-button").nth(chosen.index).click();
  check("selection_enabled_after_search", await page.locator(".selected-chip").count() === 1);
  const pendingDetail = page.waitForResponse((response) => new URL(response.url()).pathname === `/api/recipes/${encodeURIComponent(chosen.recipe.id)}`);
  const pendingGraph = page.waitForResponse((response) => new URL(response.url()).pathname === "/api/system/graph");
  await page.locator(".card-view-button").nth(chosen.index).click();
  const detailResponse = await pendingDetail; check("recipe_detail_http_ok", detailResponse.ok());
  const detail = detailBody(await detailResponse.json()); await page.locator(".original-step").last().waitFor();
  const renderedSteps = await page.locator(".original-step p").allTextContents();
  check("original_step_text_unchanged", renderedSteps.length === detail.steps.length && renderedSteps.every((value, index) => value === detail.steps[index].text));
  check("original_step_order_preserved", detail.steps.every((step, index) => index === 0 || step.order >= detail.steps[index - 1].order));
  const sourceText = await page.locator(".source-list").innerText();
  check("source_ids_displayed", detail.sources.every((source) => typeof source.source_id === "string" && sourceText.includes(source.source_id)));
  const urls = await page.locator("#detail-body a[href]").evaluateAll((nodes) => nodes.map((node) => node.href));
  const expectedURLs = detail.sources.map((source) => source.url).filter(safeURL).map((url) => new URL(url).href);
  check("provided_source_links_rendered_safely", urls.every(safeURL) && expectedURLs.every((url) => urls.includes(url)));
  report.recipe_detail = { id: detail.id, title: detail.title, original_steps: detail.steps.length, source_records: detail.sources.length, source_ids: detail.sources.map((source) => source.source_id), source_urls: urls, ingredient_semantics: detail.semantics };
  const graphResponse = await pendingGraph; check("graph_http_ok", graphResponse.ok()); const graph = await graphResponse.json();
  await page.locator(".graph-node").last().waitFor();
  check("graph_recipe_visible", await page.locator('.graph-node[aria-label^="Recipe："]').count() > 0);
  await page.locator(".graph-node").last().click(); const selectedNode = await page.locator("#graph-node-detail").innerText();
  check("svg_node_click_displays_real_identity", graph.nodes.some((node) => selectedNode.includes(`ID：${node.id}`)));
  report.graph = { returned_nodes: graph.nodes.length, returned_edges: graph.edges.length, truncation_exercised: graph.nodes.length > 36, displayed_nodes: await page.locator(".graph-node").count(), clicked_detail: selectedNode };
  // 原文/来源先保存到报告；截图保留用户实际点击后的图谱状态。
  report.screenshots.detail = "reports/screenshots/system-detail.png"; await page.screenshot({ path: path.join(root, report.screenshots.detail) });
  await page.click("#close-detail");
  await page.fill("#generation-question", "按顺序整理这道食谱的原始做法，标出每一步的证据来源。");
  const pendingGeneration = page.waitForResponse((response) => new URL(response.url()).pathname === "/api/generate" && response.request().method() === "POST");
  await page.click("#generate-button"); const generationResponse = await pendingGeneration; check("generation_http_ok", generationResponse.ok()); const generated = await generationResponse.json();
  await page.locator("#answer-wrap").waitFor();
  check("grounded_effective_mode", generated.generation?.mode === "grounded" && generated.generation?.effective_mode === "grounded");
  check("no_model_called", generated.generation?.llm_called === false && generated.generation?.inference_started_unknown !== true);
  check("generated_recipe_is_selected_recipe", generated.recipes.length === 1 && generated.recipes[0].id === chosen.recipe.id);
  check("selected_original_steps_retained", detail.steps.every((step) => generated.answer.includes(step.text)));
  check("answer_display_matches_api", await page.locator("#answer-text").textContent() === generated.answer);
  const evidence = generated.evidence || [], evidenceIDs = new Set(evidence.map((item) => item.id));
  const citations = [...generated.answer.matchAll(/\[(E\d+)\]/g)].map((match) => match[1]);
  check("answer_citations_resolve", citations.length > 0 && citations.every((id) => evidenceIDs.has(id)));
  check("original_steps_retained_in_answer", generated.recipes.every((recipe) => recipe.steps.every((step) => generated.answer.includes(step.text))));
  check("step_evidence_sources_resolve", evidence.some((item) => item.kind === "step") && evidence.filter((item) => item.kind === "step").every((item) => typeof item.source_id === "string" && generated.recipes.some((recipe) => recipe.id === item.recipe_id && recipe.sources.some((source) => source.source_id === item.source_id))));
  check("step_evidence_matches_original_detail", detail.steps.every((step) => evidence.some((item) => item.kind === "step" && item.recipe_id === detail.id && item.graph_id === step.id && item.text === step.text && item.source_id === step.source_id)));
  check("ui_evidence_count_matches_api", await page.locator(".evidence-item").count() === evidence.length);
  report.generation = { status: generated.status, selected_recipe_ids: generated.recipes.map((recipe) => recipe.id), generation: generated.generation, validation: generated.validation, evidence_count: evidence.length, citation_count: citations.length, non_step_evidence_without_source_id: evidence.filter((item) => item.kind !== "step" && !item.source_id && !item.source_ids?.length).map((item) => ({id:item.id,kind:item.kind,graph_id:item.graph_id})), answer: generated.answer, evidence, trace: generated.trace };
  report.screenshots.desktop = "reports/screenshots/system-desktop.png"; await page.evaluate(() => window.scrollTo({top:0,behavior:"instant"})); await page.screenshot({ path: path.join(root, report.screenshots.desktop), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  check("mobile_no_horizontal_overflow", await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
  report.mobile = { viewport: { width: 390, height: 844 }, document_width: await page.evaluate(() => document.documentElement.scrollWidth) };
  report.screenshots.mobile = "reports/screenshots/system-mobile.png"; await page.evaluate(() => window.scrollTo({top:0,behavior:"instant"})); await page.screenshot({ path: path.join(root, report.screenshots.mobile), fullPage: true });
  // 读取已经存在的图片识别候选，未下载媒体、未重新调用视觉模型。
  const visualID = "recipegen:test:test-video:004de93faeb0fba12069";
  await page.setViewportSize({ width: 1280, height: 1000 });
  const visualResponse = await page.evaluate(async (id) => { const response = await fetch(`/api/recipes/${encodeURIComponent(id)}`); return { status: response.status, body: await response.json() }; }, visualID);
  check("existing_visual_recipe_http_ok", visualResponse.status === 200); const visualDetail = detailBody(visualResponse.body);
  let visualSearch = await search(visualDetail.title), visualIndex = visualSearch.recipes.findIndex((recipe) => recipe.id === visualID);
  const titleSearchFoundTarget = visualIndex >= 0;
  if (visualIndex < 0) { visualSearch = await search(visualID); visualIndex = visualSearch.recipes.findIndex((recipe) => recipe.id === visualID); }
  check("existing_visual_recipe_search", visualIndex >= 0);
  const pendingVisual = page.waitForResponse((response) => new URL(response.url()).pathname === `/api/recipes/${encodeURIComponent(visualID)}`);
  await page.locator(".card-view-button").nth(visualIndex).click(); const finalVisualResponse = await pendingVisual; const finalVisual = detailBody(await finalVisualResponse.json()); await page.locator(".observation-card").last().waitFor();
  const visualText = await page.locator(".visual-observations").innerText(), visualEvidence = finalVisual.visual_evidence || [];
  check("existing_visual_evidence_displayed", visualEvidence.length > 0 && await page.locator(".observation-card").count() === Math.min(8, visualEvidence.length) && visualEvidence.slice(0, 8).every((item) => visualText.includes(item.observation_id) && visualText.includes(item.caption)));
  report.visual_evidence = { recipe_id: visualID, title: finalVisual.title, title_search_found_target: titleSearchFoundTarget, search_query: visualSearch.query, available_observations: visualEvidence.length, displayed_observations: await page.locator(".observation-card").count(), observation_ids: visualEvidence.map((item) => item.observation_id), model_inference_requested: false };
  await page.click("#close-detail");
  check("no_model_request_sent", report.local_model_requested === false && report.generation_requests.length > 0);
  check("no_page_javascript_errors", report.page_errors.length === 0);
  check("api_requests_successful", report.requests.filter((item) => item.path.startsWith("/api/")).every((item) => item.status >= 200 && item.status < 300));
  report.status = "verified";
})().catch((error) => { report.status = "failed"; report.error = error.message; process.exitCode = 1; }).finally(async () => {
  if (browser) await browser.close(); report.finished_at = new Date().toISOString();
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify({ status: report.status, recipe_count: report.system?.graph?.recipes, search_count: report.search?.count, effective_mode: report.generation?.generation?.effective_mode, llm_called: report.generation?.generation?.llm_called, visual_observations: report.visual_evidence?.displayed_observations, checks: report.checks, page_errors: report.page_errors, error: report.error, report: reportPath }));
});
