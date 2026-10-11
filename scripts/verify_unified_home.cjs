/* Unified RecipeGen home verification. HTML/CSS/JS are served by localhost;
 * all API requests are explicit mock fixtures. No .env or real LLM calls.
 * node scripts/verify_unified_home.cjs [--base-url http://127.0.0.1:8765]
 */
"use strict";
const fs = require("fs"), path = require("path"), os = require("os"), assert = require("assert");
const project = path.resolve(__dirname, "..");
let baseURL = "http://127.0.0.1:8765";
let reportLabel = "unified-home-v1";
const args = process.argv.slice(2);
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--base-url" && args[i + 1]) baseURL = args[++i];
  else if (args[i] === "--report-label" && args[i + 1]) { reportLabel = args[++i]; assert(/^[a-z0-9-]+$/.test(reportLabel)); }
  else throw new Error(`Unknown argument: ${args[i]}`);
}
const base = new URL(baseURL);
assert(base.protocol === "http:" && ["127.0.0.1", "localhost", "[::1]"].includes(base.hostname) && base.port === "8765" && !base.username && !base.password, "Only local HTTP 8765 is allowed");
baseURL = base.origin;
const playwright = require(path.join(os.homedir(), ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright"));
const executablePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const output = path.join(project, "reports", reportLabel);
fs.mkdirSync(output, { recursive: true });
const reportPath = path.join(output, "browser.json");
const report = {
  schema_version: "recipegen-unified-home-v1", report_label: reportLabel, started_at: new Date().toISOString(), status: "running", base_url: baseURL,
  scope: "Real localhost HTML/CSS/JS and navigation; mocked API JSON only", new_real_llm_api_calls: 0,
  backend_search_quality_verified: false, semantic_qa_accuracy_verified: false,
  checks: {}, screenshots: {}, document_requests: [], static_requests: [], mock_search_requests: [], mock_chat_requests: [],
  page_errors: [], console_errors: [], failed_requests: [], blocked_requests: [], viewport_results: []
};
const status = { backend: "neo4j", dataset: { name: "MockRecipeGen", is_demo: false }, graph: { recipes: 5898, steps: 44802 }, retrieval: { default_mode: "hybrid", semantic_available: true }, generation: { api_configured: true, api_model: "mock-model", provider: "chat_completions" }, chat: { available: true, api_configured: true } };
const recipes = [
  { id: "mock_recipe_1", title: "西兰花炒胡萝卜", minutes: 15, ingredient_mentions: [{ zh_name: "西兰花" }, { zh_name: "胡萝卜" }], steps: [{ id: "mock_step_1", order: 1, text: "将西兰花与胡萝卜切块。" }, { id: "mock_step_2", order: 2, text: "把蔬菜炒熟后盛出。" }], source_id: "mock_source_1", source_url: "https://example.invalid/mock-recipe" },
  { id: "mock_recipe_2", title: "清炒双蔬", minutes: 10, ingredient_mentions: [{ zh_name: "西兰花" }, { zh_name: "胡萝卜" }], steps: [{ id: "mock_step_3", order: 1, text: "把蔬菜炒熟。" }], source_id: "mock_source_2" }
];
const retrieval = { status: "ok", method: "graph_first_bm25_e5_rrf", requested_mode: "hybrid", effective_mode: "hybrid", semantic: { graph_candidate_count: 40, scored_recipe_count: 40, coverage: "all_graph_eligible_recipes_scored", score_semantics: "neo4j_cosine_similarity_0_1", approximate_search: false, index_used_for_ann: false }, trace: [], limitations: [] };
const paragraph = { text: "可以做西兰花炒胡萝卜：先切块，再把蔬菜炒熟。", evidence_ids: ["E1", "E2"] };
const chatAnswer = {
  status: "answered", answer: paragraph.text, route: { action: "retrieve", llm_decided: true }, retrieval,
  generation: { mode: "api", effective_mode: "graphrag_api_evidence_qa", llm_called: true, api_calls_attempted: 3, api_status: "ok", fallback: false, model: "mock-model" },
  rerank: { status: "ok", ranking: ["mock_recipe_1"] }, qa: { status: "answered", paragraphs: [paragraph] }, paragraphs: [paragraph], recipes: recipes.slice(0, 1), recipe_ids: ["mock_recipe_1"],
  evidence: recipes[0].steps.map((step, index) => ({ id: `E${index + 1}`, kind: "step", recipe_id: "mock_recipe_1", graph_id: step.id, step_id: step.id, order: step.order, source_id: "mock_source_1", text: step.text, original_text: step.text, source_url: "https://example.invalid/mock-recipe" })), validation: { passed: true }
};
function check(name, value) { report.checks[name] = Boolean(value); assert(value, name); }
function save() { fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n"); }
async function shot(page, name) { const file = path.join(output, `${name}.png`); await page.screenshot({ path: file, fullPage: true }); report.screenshots[name] = path.relative(project, file); }
async function json(route, value, code = 200) { return route.fulfill({ status: code, contentType: "application/json; charset=utf-8", body: JSON.stringify(value) }); }
async function noOverflow(page, name) { check(name, await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)); }
async function uniqueIDs(page, name) { check(name, await page.evaluate(() => { const ids = [...document.querySelectorAll("[id]")].map(e => e.id); return new Set(ids).size === ids.length; })); }
async function noSmallCopy(page, name) {
  const visible = await page.locator("body").innerText();
  check(name, !/你的厨房搭子|Enter 发送|Shift \+ Enter|多个食材用|从冰箱里的食材，找到|用现有食材，发现|按食材探索真实食谱|数据与使用说明|厨神的一般建议|已查找食谱 · 根据原文回答/.test(visible));
}
async function home(page, name) {
  check(`${name}_home_page`, await page.locator("body[data-page='home']").count() === 1);
  check(`${name}_two_main_choices_only`, await page.locator("main a").count() === 2 && await page.locator(".mode-option[href='/?view=explore']").count() === 1 && await page.locator(".mode-option[href='/?view=chat']").count() === 1);
  check(`${name}_no_hero_or_footer`, await page.locator(".hero-art,.hero-description,.site-footer,main img").count() === 0);
  check(`${name}_minimal_header`, await page.locator(".site-header a").count() === 1 && await page.locator("#theme-toggle").count() === 1);
  const controls = page.locator(".mode-option"), boxes = await Promise.all([controls.nth(0).boundingBox(), controls.nth(1).boundingBox()]);
  check(`${name}_equal_choice_dimensions`, boxes.every(Boolean) && Math.abs(boxes[0].width - boxes[1].width) < 1 && Math.abs(boxes[0].height - boxes[1].height) < 1);
  check(`${name}_keyboard_focusable_choices`, await controls.evaluateAll(nodes => nodes.every(n => n.href && n.tabIndex === 0 && n.innerText.trim().length > 0)));
  await uniqueIDs(page, `${name}_unique_ids`); await noOverflow(page, `${name}_no_overflow`); await noSmallCopy(page, `${name}_no_unused_copy`);
  report.viewport_results.push({ view: "home", name, viewport: page.viewportSize(), boxes });
}
async function explore(page, name) {
  check(`${name}_explore_page`, await page.locator("body[data-page='explore']").count() === 1);
  check(`${name}_two_fields_only`, await page.locator("main input,main textarea,main select").count() === 2);
  check(`${name}_field_labels`, await page.getByLabel("想用的食材", { exact: true }).count() === 1 && await page.getByLabel("不想用的食材", { exact: true }).count() === 1);
  check(`${name}_no_field_helper`, await page.locator("#ingredients-help,.explore-form-note,.page-heading .field-help").count() === 0);
  await uniqueIDs(page, `${name}_unique_ids`); await noOverflow(page, `${name}_no_overflow`); await noSmallCopy(page, `${name}_no_unused_copy`);
}
async function chat(page, name) {
  check(`${name}_chat_page`, await page.locator("body[data-page='chef-chat']").count() === 1);
  check(`${name}_single_prompt_only`, await page.locator("main input,main textarea,main select").count() === 1 && await page.locator("#chat-input").count() === 1);
  check(`${name}_chef_icon`, await page.locator("#chat-welcome .chef-avatar[role='img']").count() === 1);
  check(`${name}_accessible_prompt`, await page.getByLabel("给厨神发送消息", { exact: true }).count() === 1);
  check(`${name}_new_chat_available`, await page.locator("#chat-new").isVisible());
  check(`${name}_no_keyboard_or_generic_meta`, await page.locator("#chat-input-help,.chat-topbar-title,.chat-answer-meta").count() === 0);
  await uniqueIDs(page, `${name}_unique_ids`); await noOverflow(page, `${name}_no_overflow`); await noSmallCopy(page, `${name}_no_unused_copy`);
}
let browser;
(async () => {
  browser = await playwright.chromium.launch({ executablePath, headless: true });
  report.browser = { version: browser.version(), executable: executablePath, isolated_context: true };
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, colorScheme: "light", reducedMotion: "reduce" });
  const page = await context.newPage(); page.setDefaultTimeout(12000);
  page.on("pageerror", error => report.page_errors.push(error.message));
  page.on("console", entry => { if (entry.type() === "error") report.console_errors.push(entry.text()); });
  page.on("requestfailed", request => report.failed_requests.push({ path: new URL(request.url()).pathname, error: request.failure()?.errorText }));
  page.on("response", response => { const request = response.request(), url = new URL(response.url()); if (url.origin !== baseURL) return; const entry = { path: url.pathname, query: url.search, status: response.status() }; if (request.resourceType() === "document") report.document_requests.push(entry); else if (url.pathname.startsWith("/static/")) report.static_requests.push(entry); });
  let waitingCancel;
  await context.route("**/*", async route => {
    const request = route.request(), url = new URL(request.url());
    if (url.origin !== baseURL) { report.blocked_requests.push(url.href); return route.abort(); }
    if (url.pathname === "/api/system/status") return json(route, status);
    if (url.pathname === "/api/search" && request.method() === "POST") { const payload = request.postDataJSON(); report.mock_search_requests.push(payload); return json(route, { query: payload.query, recipes, retrieval, limitations: [], trace: [] }); }
    if (url.pathname === "/api/chat" && request.method() === "POST") {
      const payload = request.postDataJSON(); report.mock_chat_requests.push(payload);
      if (payload.message === "取消测试") {
        await new Promise(resolve => { waitingCancel = resolve; });
        try { return await json(route, { ...chatAnswer, answer: "取消后不应出现" }); } catch (_) { return; }
      }
      return json(route, chatAnswer);
    }
    if (url.pathname.startsWith("/api/")) { report.blocked_requests.push(url.href); return route.abort(); }
    return route.continue();
  });
  await page.goto(`${baseURL}/`, { waitUntil: "networkidle" });
  await home(page, "desktop"); await shot(page, "home-desktop-light");
  check("desktop_choices_side_by_side", await page.locator(".mode-option").evaluateAll(nodes => Math.abs(nodes[0].getBoundingClientRect().top - nodes[1].getBoundingClientRect().top) < 1));
  await page.locator(".mode-option[href='/?view=explore']").click(); await page.waitForLoadState("networkidle");
  check("explore_root_query_navigation", page.url() === `${baseURL}/?view=explore`);
  await explore(page, "desktop");
  check("explore_blank_results_initially_hidden", !(await page.locator("#results-column").isVisible()) && !(await page.locator("#empty-state").isVisible()));
  await shot(page, "explore-desktop-empty");
  await page.getByLabel("想用的食材", { exact: true }).fill("西兰花、胡萝卜");
  await page.getByLabel("不想用的食材", { exact: true }).fill("花生");
  await page.locator("#search-button").click(); await page.waitForSelector(".recipe-card");
  check("explore_search_conditions_and_mode", report.mock_search_requests.length === 1 && JSON.stringify(report.mock_search_requests[0].ingredients) === '["西兰花","胡萝卜"]' && JSON.stringify(report.mock_search_requests[0].excluded_ingredients) === '["花生"]' && report.mock_search_requests[0].retrieval_mode === "hybrid" && report.mock_search_requests[0].limit === 12);
  check("explore_direct_recipe_results", await page.locator(".recipe-card").count() === 2 && await page.locator(".card-view-button").count() === 2 && await page.locator("#results-column").isVisible());
  await shot(page, "explore-desktop-results");
  await page.reload({ waitUntil: "networkidle" });
  check("explore_refresh_keeps_view_and_results", page.url() === `${baseURL}/?view=explore` && await page.locator(".recipe-card").count() === 2 && await page.locator("#ingredients").inputValue() === "西兰花、胡萝卜");
  await page.locator("a.brand").click(); await page.waitForLoadState("networkidle");
  check("explore_back_to_clean_home", page.url() === `${baseURL}/` && await page.locator(".mode-option").count() === 2);
  await page.locator(".mode-option[href='/?view=chat']").click(); await page.waitForLoadState("networkidle");
  check("chat_root_query_navigation", page.url() === `${baseURL}/?view=chat`);
  await chat(page, "desktop"); await shot(page, "chat-desktop-empty");
  await page.locator("#chat-input").fill("西兰花和胡萝卜可以做什么？");
  await page.locator("#chat-send").click();
  await page.waitForFunction(() => document.querySelector("#chat-form")?.getAttribute("aria-busy") === "false");
  check("chat_single_prompt_api_payload", report.mock_chat_requests.length === 1 && Object.keys(report.mock_chat_requests[0]).sort().join(",") === "history,message" && report.mock_chat_requests[0].history.length === 0);
  check("chat_question_answer_rendered", await page.locator(".chat-message[data-role='user']").count() === 1 && await page.locator(".chat-message[data-role='assistant']").count() === 1 && (await page.locator(".chat-message[data-role='assistant']").textContent()).includes(paragraph.text));
  check("chat_no_generic_meta_after_answer", await page.locator(".chat-answer-meta").count() === 0);
  await page.locator(".chat-citation").first().click();
  check("chat_citation_expands_original", await page.locator("details.chat-sources[open]").isVisible() && (await page.locator("details.chat-sources[open]").textContent()).includes("将西兰花与胡萝卜切块。") && (await page.locator("details.chat-sources[open]").textContent()).includes("mock_source_1"));
  await shot(page, "chat-desktop-answer-citations");
  await page.reload({ waitUntil: "networkidle" });
  check("chat_refresh_keeps_view_and_history", page.url() === `${baseURL}/?view=chat` && await page.locator(".chat-message").count() === 2);
  await page.locator("#chat-new").click();
  check("chat_new_clears_conversation", await page.locator(".chat-message").count() === 0 && await page.locator("#chat-welcome").isVisible());
  await page.locator("#chat-input").fill("取消测试");
  const started = page.waitForRequest(r => new URL(r.url()).pathname === "/api/chat");
  await page.locator("#chat-send").click(); await started;
  check("chat_cancel_available", await page.locator("#chat-cancel").isVisible());
  await page.locator("#chat-cancel").click(); waitingCancel();
  await page.waitForTimeout(150);
  check("chat_cancel_stops_waiting", await page.locator("#chat-form").getAttribute("aria-busy") === "false" && (await page.locator("#chat-input").inputValue()) === "取消测试" && !(await page.locator("#chat-messages").textContent()).includes("取消后不应出现"));
  await page.locator("#chat-new").click();
  await page.locator("a.brand").click(); await page.waitForLoadState("networkidle");
  check("chat_back_to_home", page.url() === `${baseURL}/`);
  await page.setViewportSize({ width: 360, height: 800 });
  await home(page, "mobile360"); await shot(page, "home-mobile360-light");
  check("mobile_choices_both_visible", await page.locator(".mode-option").nth(0).isVisible() && await page.locator(".mode-option").nth(1).isVisible());
  await page.locator("#theme-toggle").click();
  check("theme_switches_dark", await page.evaluate(() => document.documentElement.dataset.theme === "dark"));
  await home(page, "mobile360dark"); await shot(page, "home-mobile360-dark");
  await page.locator(".mode-option[href='/?view=explore']").click(); await page.waitForLoadState("networkidle");
  await explore(page, "mobile360dark"); await shot(page, "explore-mobile360-dark");
  check("theme_persists_view_navigation", await page.evaluate(() => document.documentElement.dataset.theme === "dark"));
  await page.goto(`${baseURL}/?view=chat`, { waitUntil: "networkidle" });
  await page.setViewportSize({ width: 390, height: 844 });
  await chat(page, "mobile390dark"); await shot(page, "chat-mobile390-dark");
  await page.locator("#theme-toggle").click();
  check("theme_switches_light", await page.evaluate(() => document.documentElement.dataset.theme === "light"));
  await chat(page, "mobile390light"); await shot(page, "chat-mobile390-light");
  await page.goto(`${baseURL}/explore`, { waitUntil: "networkidle" });
  check("legacy_explore_compatible", await page.locator("#search-form").isVisible());
  await page.goto(`${baseURL}/compose`, { waitUntil: "networkidle" });
  check("legacy_compose_compatible", await page.locator("#chat-input").isVisible());
  check("no_page_errors", report.page_errors.length === 0);
  check("no_console_errors", report.console_errors.length === 0);
  check("no_external_or_unhandled_api", report.blocked_requests.length === 0);
  check("only_expected_cancel_failure", report.failed_requests.every(r => r.path === "/api/chat" && /ABORT|CANCEL/.test(r.error || "")));
  check("served_documents_ok", report.document_requests.length >= 10 && report.document_requests.every(r => r.status === 200 || [301, 302, 303, 307, 308].includes(r.status)));
  check("served_static_files_ok", report.static_requests.length > 0 && report.static_requests.every(r => r.status === 200));
  check("no_hero_image_requested", !report.static_requests.some(r => /kitchen-hero/.test(r.path)));
  check("zero_new_real_llm_calls", report.new_real_llm_api_calls === 0);
  report.status = "passed";
  await context.close();
})().catch(error => { report.status = "failed"; report.error = error.message; process.exitCode = 1; }).finally(async () => {
  if (browser) await browser.close(); report.finished_at = new Date().toISOString(); save();
  console.log(JSON.stringify({ status: report.status, checks: Object.keys(report.checks).length, report: reportPath, new_real_llm_api_calls: 0, error: report.error || null }, null, 2));
});
