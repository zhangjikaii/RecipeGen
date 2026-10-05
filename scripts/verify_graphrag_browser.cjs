/* 文本 GraphRAG 的真实浏览器验收。
 * 仅访问本机现有服务，读取图谱和 grounded 回答，不启动模型生成或媒体任务。
 * 运行：node scripts/verify_graphrag_browser.cjs [--fallback-only] [--base-url http://127.0.0.1:8765]
 * 使用已安装的 Playwright 与 Chrome；不会安装或下载依赖。
 */
"use strict";
const fs = require("fs"), path = require("path"), os = require("os"), assert = require("assert");
const root = path.resolve(__dirname, "..");
const args = process.argv.slice(2);
let baseURL = "http://127.0.0.1:8765", fallbackOnly = false;
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--fallback-only") fallbackOnly = true;
  else if (args[i] === "--base-url" && args[i + 1]) baseURL = args[++i];
  else throw new Error(`未知参数：${args[i]}`);
}
const base = new URL(baseURL);
assert(base.protocol === "http:" && ["127.0.0.1", "localhost", "[::1]"].includes(base.hostname) && !base.username && !base.password, "只允许无凭据的本机 HTTP 地址");
baseURL = base.origin;
function playwright() {
  const candidates = [process.env.PLAYWRIGHT_MODULE_PATH, "playwright", path.join(os.homedir(), ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright")].filter(Boolean);
  for (const candidate of candidates) { try { return require(candidate); } catch (error) { if (error.code !== "MODULE_NOT_FOUND") throw error; } }
  throw new Error("找不到已安装的 Playwright；请先配置运行环境。本脚本不会下载依赖。");
}
function chromePath() {
  const candidates = [process.env.CHROME_EXECUTABLE_PATH, "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser", process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, "Google/Chrome/Application/chrome.exe"), process.env.PROGRAMFILES && path.join(process.env.PROGRAMFILES, "Google/Chrome/Application/chrome.exe")].filter(Boolean);
  return candidates.find((candidate) => fs.existsSync(candidate));
}
const reportPath = path.join(root, fallbackOnly ? "reports/graphrag-browser-fallback.json" : "reports/graphrag-browser-smoke.json");
const screenshots = path.join(root, "reports/graphrag-browser-screenshots");
fs.mkdirSync(screenshots, { recursive: true });
const report = { schema_version: "recipegen-graphrag-browser-v1", started_at: new Date().toISOString(), status: "running",
  scope: fallbackOnly ? "real_http_hybrid_keyword_fallback_browser_smoke" : "real_http_text_graphrag_and_grounded_browser_smoke", base_url: baseURL, checks: {}, searches: [],
  api_generation_requested: false, local_generation_requested: false, media_processing_requested: false,
  semantic_accuracy_verified: false, page_errors: [], console_errors: [], blocked_requests: [], request_failures: [], requests: [], screenshots: {} };
let browser;
function check(name, condition) { report.checks[name] = Boolean(condition); assert(condition, name); }
function sameIDs(a, b) { return a.length === b.length && new Set(a).size === a.length && a.every((id) => b.includes(id)); }
function detailBody(value) { return value.recipe || value; }
function safeURL(value) { try { const url = new URL(value); return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password; } catch (_) { return false; } }
(async () => {
  const executablePath = chromePath();
  browser = await playwright().chromium.launch({ ...(executablePath ? { executablePath } : {}), headless: true });
  report.browser = { version: browser.version(), executable: executablePath || "installed_playwright_browser", isolated_context: true };
  const context = await browser.newContext({ viewport: { width: 1280, height: 1000 }, reducedMotion: "reduce" });
  const page = await context.newPage(); page.setDefaultTimeout(130000);
  page.on("pageerror", (error) => report.page_errors.push(error.message));
  page.on("console", (message) => { if (message.type() === "error") report.console_errors.push(message.text()); });
  page.on("response", (response) => { const url = new URL(response.url()); if (url.origin === baseURL) report.requests.push({ method: response.request().method(), path: url.pathname, status: response.status() }); });
  page.on("requestfailed", (request) => report.request_failures.push({ path: new URL(request.url()).pathname, error: request.failure()?.errorText || "unknown" }));
  // 不打开原来源链接，也不允许浏览器意外发送模型请求。
  await context.route("**/*", async (route) => {
    const request = route.request(), url = new URL(request.url());
    let allowed = url.origin === baseURL;
    if (allowed && url.pathname.startsWith("/api/")) {
      allowed = request.method() === "GET" && (["/api/system/status", "/api/system/graph"].includes(url.pathname) || /^\/api\/recipes\/[^/]+$/.test(url.pathname));
      if (request.method() === "POST" && url.pathname === "/api/search") allowed = true;
      if (request.method() === "POST" && url.pathname === "/api/generate") {
        const payload = request.postDataJSON();
        report.api_generation_requested ||= payload?.mode === "api";
        report.local_generation_requested ||= payload?.mode === "local";
        allowed = payload?.mode === "grounded";
      }
    }
    if (!allowed) { report.blocked_requests.push({ origin: url.origin, path: url.pathname, method: request.method() }); await route.abort(); }
    else await route.continue();
  });
  const statusPending = page.waitForResponse((response) => new URL(response.url()).pathname === "/api/system/status");
  await page.goto(baseURL, { waitUntil: "domcontentloaded" });
  const statusResponse = await statusPending; check("system_status_http_ok", statusResponse.ok());
  const status = await statusResponse.json();
  await page.waitForFunction(() => document.getElementById("stat-recipes").textContent.includes("5"));
  report.system = { backend: status.backend, dataset: status.dataset, graph: status.graph, retrieval: status.retrieval, generation: status.generation };
  check("actual_neo4j_test_dataset", status.backend === "neo4j" && status.dataset?.split === "test" && status.dataset?.is_demo === false);
  check("original_recipe_count_5898", status.graph?.recipes === 5898);
  check("original_step_count_44802", status.graph?.steps === 44802);
  if (fallbackOnly) {
    check("semantic_unavailable", status.retrieval?.semantic_available === false);
    check("semantic_unavailable_displayed", (await page.locator("#retrieval-capability").innerText()).includes("文本语义检索当前不可用"));
  } else {
    check("semantic_index_ready_5898", status.retrieval?.semantic_available === true && status.retrieval?.index?.count === 5898 && status.retrieval?.index?.status === "ONLINE");
    check("semantic_capability_displayed", (await page.locator("#retrieval-capability").innerText()).includes("文本语义检索可用"));
    check("index_count_displayed", (await page.locator("#retrieval-capability").innerText()).includes("5,898"));
  }
  check("default_hybrid_matches_server", await page.locator("#retrieval-mode").inputValue() === status.retrieval.default_mode);
  check("api_configuration_pending", status.generation?.api_configured === false);
  check("api_toggle_disabled", await page.locator("#api-model").isDisabled());
  check("api_toggle_unchecked", !(await page.locator("#api-model").isChecked()));
  check("api_pending_label_displayed", await page.locator("#api-model-label").innerText() === "API 待配置");
  check("no_key_input", await page.locator('input[type="password"],input[name*="key"],input[id*="key"]').count() === 0);
  check("local_toggle_unchecked", !(await page.locator("#local-model").isChecked()));

  async function search(mode, query, ingredients = "") {
    await page.selectOption("#retrieval-mode", mode);
    await page.fill("#search-question", query); await page.fill("#ingredients", ingredients); await page.fill("#excluded-ingredients", "");
    const pending = page.waitForResponse((response) => new URL(response.url()).pathname === "/api/search" && response.request().method() === "POST");
    await page.click("#search-button");
    const response = await pending; const result = await response.json();
    check(`${mode}_search_http_ok`, response.ok());
    await page.locator("#search-button:not([disabled])").waitFor();
    const payload = response.request().postDataJSON();
    check(`${mode}_forwarded_to_api`, payload.retrieval_mode === mode && result.retrieval?.mode === mode);
    const actualMode = fallbackOnly ? "keyword" : mode;
    check(`${mode}_actual_mode_disclosed`, result.retrieval?.status === (fallbackOnly ? "fallback" : "ok") && result.retrieval?.effective_mode === actualMode);
    check(`${mode}_actual_mode_rendered`, (await page.locator("#retrieval-badges").innerText()).includes(`实际：${{semantic:"语义检索",hybrid:"混合检索",keyword:"关键词检索"}[actualMode]}`));
    check(`${mode}_has_real_results`, result.recipes?.length > 0 && result.count === result.recipes.length);
    check(`${mode}_cards_match_api`, await page.locator(".recipe-card").count() === result.recipes.length);
    check(`${mode}_candidate_scope_disclosed`, result.retrieval.coverage === "bounded_candidates" && result.retrieval.exhaustive === false && (await page.locator("#retrieval-description").innerText()).includes("候选数量不代表全库匹配数"));
    report.searches.push({ payload, count: result.count, recipe_ids: result.recipes.map((r) => r.id), titles: result.recipes.map((r) => r.title), retrieval: result.retrieval, graph_evidence: result.graph_evidence, trace: result.trace });
    return result;
  }
  if (fallbackOnly) {
    const fallback = await search("hybrid", "番茄 鸡蛋", "番茄、鸡蛋");
    check("fallback_actual_keyword_method", fallback.retrieval.method === "keyword" && fallback.retrieval.semantic?.available === false);
    check("fallback_reason_present", typeof fallback.retrieval.fallback_reason === "string" && fallback.retrieval.fallback_reason.length > 0);
    check("fallback_chinese_message_displayed", typeof fallback.retrieval.fallback_message === "string" && (await page.locator("#retrieval-description").innerText()).includes(fallback.retrieval.fallback_message));
    check("fallback_badge_displayed", (await page.locator("#retrieval-badges").innerText()).includes("发生检索回退"));
    check("fallback_no_page_errors", report.page_errors.length === 0);
    check("fallback_no_blocked_requests", report.blocked_requests.length === 0);
    check("fallback_zero_generation_requests", report.api_generation_requested === false && report.local_generation_requested === false && report.requests.every((item) => item.path !== "/api/generate"));
    report.screenshots.fallback = "reports/graphrag-browser-screenshots/fallback.png";
    await page.screenshot({ path: path.join(root, report.screenshots.fallback), fullPage: true });
    report.status = "verified"; return;
  }
  const semantic = await search("semantic", "想做一道清爽的蔬菜沙拉，请给我相关食谱");
  check("native_vector_cypher_retriever_used", semantic.retrieval.retriever === "neo4j_graphrag.VectorCypherRetriever" && semantic.retrieval.semantic?.available === true);
  check("semantic_graph_expansion_has_evidence", semantic.graph_evidence?.length > 0);
  check("semantic_graph_ids_match_original_records", semantic.graph_evidence.every((hit) => {
    const recipe = semantic.recipes.find((r) => r.id === hit.recipe_id);
    return recipe && sameIDs(hit.step_ids, recipe.steps.map((s) => s.id)) && sameIDs(hit.source_ids, recipe.sources.map((s) => s.source_id)) && sameIDs(hit.ingredient_ids, recipe.ingredient_mentions.map((i) => i.id));
  }));
  await search("keyword", "番茄 鸡蛋", "番茄、鸡蛋");
  const hybrid = await search("hybrid", "番茄 鸡蛋", "番茄、鸡蛋");
  check("hybrid_rrf_used", hybrid.retrieval.method === "rrf" && hybrid.retrieval.rrf_k === 60 && hybrid.retrieval.keyword?.available === true && hybrid.retrieval.semantic?.available === true);
  check("candidate_count_consistent", hybrid.retrieval.returned_candidates >= hybrid.retrieval.eligible_candidates && hybrid.retrieval.returned_candidates === hybrid.retrieval.eligible_candidates + hybrid.retrieval.filtered_candidates);
  const choices = hybrid.recipes.map((recipe, index) => ({ recipe, index })).filter(({ recipe }) => recipe.steps?.length && recipe.sources?.length).sort((a, b) => a.recipe.steps.length - b.recipe.steps.length);
  check("source_backed_recipe_available", choices.length > 0); const chosen = choices[0];
  await page.locator(".select-button").nth(chosen.index).click();
  check("selection_enabled", await page.locator(".selected-chip").count() === 1);
  const detailPending = page.waitForResponse((response) => new URL(response.url()).pathname === `/api/recipes/${encodeURIComponent(chosen.recipe.id)}`);
  const graphPending = page.waitForResponse((response) => new URL(response.url()).pathname === "/api/system/graph");
  await page.locator(".card-view-button").nth(chosen.index).click();
  const detailResponse = await detailPending; check("detail_http_ok", detailResponse.ok());
  const detail = detailBody(await detailResponse.json()); await page.locator(".original-step").last().waitFor();
  const stepTexts = await page.locator(".original-step p").allTextContents();
  check("detail_original_steps_unchanged", detail.steps.length === stepTexts.length && detail.steps.every((step, i) => step.text === stepTexts[i]));
  const sourceText = await page.locator(".source-list").innerText();
  check("detail_source_ids_rendered", detail.sources.every((s) => sourceText.includes(s.source_id)));
  const urls = await page.locator("#detail-body a[href]").evaluateAll((nodes) => nodes.map((n) => n.href));
  check("detail_source_links_safe", urls.every(safeURL) && detail.sources.map((s) => s.url).filter(safeURL).every((url) => urls.includes(new URL(url).href)));
  const graphResponse = await graphPending; check("graph_http_ok", graphResponse.ok()); const graph = await graphResponse.json();
  await page.locator(".graph-node").last().waitFor();
  check("recipe_graph_visible", await page.locator('.graph-node[aria-label^="Recipe："]').count() > 0);
  await page.locator(".graph-node").last().click();
  const nodeIdentity = await page.locator("#graph-node-detail").innerText();
  check("graph_node_identity_visible", graph.nodes.some((node) => nodeIdentity.includes(`ID：${node.id}`)));
  report.detail = { recipe_id: detail.id, title: detail.title, step_ids: detail.steps.map((s) => s.id), source_ids: detail.sources.map((s) => s.source_id), source_urls: urls, graph_nodes: graph.nodes.length, graph_edges: graph.edges.length };
  report.screenshots.detail = "reports/graphrag-browser-screenshots/detail.png";
  await page.screenshot({ path: path.join(root, report.screenshots.detail) });
  await page.click("#close-detail");
  await page.fill("#generation-question", "按原始顺序整理这道食谱的做法，并标出每一步的来源。");
  const generationPending = page.waitForResponse((response) => new URL(response.url()).pathname === "/api/generate" && response.request().method() === "POST");
  await page.click("#generate-button");
  const generationResponse = await generationPending; check("generation_http_ok", generationResponse.ok());
  const generated = await generationResponse.json(), generationPayload = generationResponse.request().postDataJSON();
  await page.locator("#answer-wrap").waitFor();
  check("generation_hybrid_mode_forwarded", generationPayload.retrieval_mode === "hybrid" && generated.retrieval?.mode === "hybrid");
  check("generation_selected_ids_no_recall_claim", generated.retrieval?.effective_mode === "selected_recipes" && (await page.locator("#generation-retrieval-description").innerText()).includes("没有再次执行语义或关键词召回"));
  check("generation_selected_recipe_retained", generated.recipes?.length === 1 && generated.recipes[0].id === chosen.recipe.id);
  check("generation_grounded_actual_mode", generationPayload.mode === "grounded" && generated.generation?.effective_mode === "grounded");
  check("generation_no_llm_called", generated.generation?.llm_called === false && generated.generation?.inference_started_unknown !== true);
  check("generation_original_steps_retained", detail.steps.every((step) => generated.answer.includes(step.text)));
  check("generation_answer_rendered_exactly", await page.locator("#answer-text").textContent() === generated.answer);
  const evidence = generated.evidence || [], evidenceIDs = new Set(evidence.map((item) => item.id));
  const citationIDs = [...generated.answer.matchAll(/\[(E\d+)\]/g)].map((match) => match[1]);
  check("generation_citation_ids_resolve", citationIDs.length > 0 && citationIDs.every((id) => evidenceIDs.has(id)));
  check("generation_step_evidence_matches_graph", detail.steps.every((step) => evidence.some((item) => item.kind === "step" && item.recipe_id === detail.id && item.graph_id === step.id && item.text === step.text && item.source_id === step.source_id)));
  check("generation_evidence_sources_resolve", evidence.filter((item) => item.kind === "step").every((item) => detail.sources.some((source) => source.source_id === item.source_id)));
  check("generation_evidence_count_matches_ui", await page.locator(".evidence-item").count() === evidence.length);
  check("generation_no_model_badge_displayed", (await page.locator("#generation-badges").innerText()).includes("未调用模型"));
  report.generation = { payload: generationPayload, status: generated.status, generation: generated.generation, retrieval: generated.retrieval,
    recipe_ids: generated.recipes.map((r) => r.id), evidence_count: evidence.length, citation_count: citationIDs.length, evidence, answer: generated.answer, validation: generated.validation, trace: generated.trace };
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
  report.screenshots.desktop = "reports/graphrag-browser-screenshots/desktop.png";
  await page.screenshot({ path: path.join(root, report.screenshots.desktop), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  check("mobile_no_horizontal_overflow", await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
  report.mobile = { viewport_width: 390, document_width: await page.evaluate(() => document.documentElement.scrollWidth) };
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
  report.screenshots.mobile = "reports/graphrag-browser-screenshots/mobile.png";
  await page.screenshot({ path: path.join(root, report.screenshots.mobile), fullPage: true });
  check("api_toggle_remains_disabled", await page.locator("#api-model").isDisabled());
  check("no_api_generation_sent", report.api_generation_requested === false);
  check("no_local_generation_sent", report.local_generation_requested === false);
  check("no_external_or_mutation_requests", report.blocked_requests.length === 0);
  check("no_page_javascript_errors", report.page_errors.length === 0);
  check("api_requests_successful", report.requests.filter((item) => item.path.startsWith("/api/")).every((item) => item.status >= 200 && item.status < 300));
  report.status = "verified";
})().catch((error) => { report.status = "failed"; report.error = error.message; process.exitCode = 1; }).finally(async () => {
  if (browser) await browser.close(); report.finished_at = new Date().toISOString();
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify({ status: report.status, checks: report.checks, error: report.error, report: reportPath }));
});
