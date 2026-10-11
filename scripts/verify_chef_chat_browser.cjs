/* RecipeGen 两入口与厨神聊天浏览器验收。
 * 默认仅以本地静态文件和明确的 mock JSON 验证 UI，不访问后端或大模型。
 * --run-api 额外验证已经启动的本机服务：一条普通技巧问答、一条真实图谱检索问答。
 * 不读取 .env，不点击来源外链；mock 与真实证据分别记录。
 * --recheck-v4 仅复用已保存 v4 第2轮历史/问题与一次追问，不重跑普通问题/食谱探索。
 * node scripts/verify_chef_chat_browser.cjs [--run-api] [--followup|--recheck-v4] [--report-label v1]
 */
"use strict";
const fs = require("fs"), path = require("path"), os = require("os"), assert = require("assert");
const root = path.resolve(__dirname, "..");
let baseURL = "http://127.0.0.1:8765", runAPI = false, followup = false, recheckV4 = false, label = "v1";
const args = process.argv.slice(2);
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--run-api") runAPI = true;
  else if (args[i] === "--followup") followup = true;
  else if (args[i] === "--recheck-v4") recheckV4 = true;
  else if (args[i] === "--base-url" && args[i + 1]) baseURL = args[++i];
  else if (args[i] === "--report-label" && args[i + 1]) label = args[++i];
  else throw new Error(`未知参数：${args[i]}`);
}
assert(/^v[1-9]\d*$/.test(label), "报告标签须为 v1、v2 等版本号");
assert(!followup || runAPI, "--followup 需要 --run-api，防止误认为真实模型验证");
assert(!recheckV4 || runAPI && !followup, "--recheck-v4 需要 --run-api 且不能与 --followup 同时使用");
const base = new URL(baseURL);
assert(base.protocol === "http:" && ["127.0.0.1", "localhost", "[::1]"].includes(base.hostname) && base.port === "8765" && !base.username && !base.password, "仅允许无凭据的本机 HTTP 8765 服务");
baseURL = base.origin;
const playwright = require(path.join(os.homedir(), ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright"));
const executablePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const reportPath = path.join(root, `reports/chef-chat-browser-${label}.json`);
const screenshotDir = path.join(root, `reports/chef-chat-browser-screenshots-${label}`);
fs.mkdirSync(screenshotDir, { recursive: true });
const report = { schema_version: "recipegen-chef-chat-browser-v1", report_label: label, started_at: new Date().toISOString(), status: "running", base_url: baseURL, run_api_authorized: runAPI, real_followup_authorized: followup, recheck_v4_authorized: recheckV4, semantic_accuracy_verified: false, mock: null, real: null };
const xssText = '<img src="x" onerror="window.__chefXSS=1">';
const mockAnswer = `【mock UI 数据】先把米洗净，再根据米种调整水量。字面示例：${xssText}`;
const mockEvidence = [
  { id: "E1", kind: "step", recipe_id: "mock_recipe_1", graph_id: "mock_step_1", step_id: "mock_step_1", order: 1, source_id: "mock_source_1", text: `将西兰花与胡萝卜切块。${xssText}`, original_text: `将西兰花与胡萝卜切块。${xssText}`, source_url: "javascript:window.__chefXSS=2" },
  { id: "E2", kind: "step", recipe_id: "mock_recipe_1", graph_id: "mock_step_2", step_id: "mock_step_2", order: 2, source_id: "mock_source_1", text: "将蔬菜炒熟后盛出。", original_text: "将蔬菜炒熟后盛出。", source_url: "https://example.invalid/mock-recipe" }
];
const mockRecipes = [
  { id: "mock_recipe_1", title: `【mock】双蔬菜 ${xssText}`, minutes: 15, ingredient_mentions: [{ zh_name: "西兰花" }, { zh_name: "胡萝卜" }], steps: [{ id: "mock_step_1", order: 1, text: mockEvidence[0].text }, { id: "mock_step_2", order: 2, text: mockEvidence[1].text }], source_id: "mock_source_1", source_url: "https://example.invalid/mock-recipe" },
  { id: "mock_recipe_2", title: "【mock】清炒时蔬", minutes: 10, ingredient_mentions: [{ zh_name: "西兰花" }, { zh_name: "胡萝卜" }], steps: [{ id: "mock_step_3", order: 1, text: "炒熟蔬菜。" }], source_id: "mock_source_2" }
];
const mockRetrieval = { method: "graph_first_e5_exact_cosine", requested_mode: "semantic", effective_mode: "semantic", semantic: { graph_candidate_count: 40, scored_recipe_count: 40, coverage: "all_graph_eligible_recipes_scored", score_semantics: "neo4j_cosine_similarity_0_1", approximate_search: false, index_used_for_ann: false }, trace: ["【mock】图谱筛选后对全部候选 E5 排序。"], limitations: ["【mock】用于界面验证，不是实际食谱检索。"] };
const mockStatus = { backend: "neo4j", dataset: { name: "MockRecipeGen", is_demo: false }, graph: { recipes: 5898, steps: 44802 }, retrieval: { default_mode: "semantic", semantic_available: true }, generation: { api_configured: true, api_model: "mock-model", provider: "chat_completions" }, chat: { available: true, api_configured: true } };
function directResponse(answer = mockAnswer) {
  return { status: "answered", answer, route: { action: "respond", llm_decided: true }, generation: { mode: "api", effective_mode: "chef_chat_direct", llm_called: true, api_call_attempted: true, api_calls_attempted: 1, api_status: "ok", fallback: false, model: "mock-model" }, evidence: [], paragraphs: [], recipes: [], trace: ["【mock】普通技巧直接回答，不检索。"] };
}
function retrievalResponse() {
  const paragraphs = [{ text: "【mock】在本次候选中，可制作双蔬菜，先切块再炒熟。", evidence_ids: ["E1", "E2"] }];
  return { status: "answered", answer: paragraphs[0].text, route: { action: "retrieve", llm_decided: true, query: "使用西兰花和胡萝卜的食谱，不使用花生", ingredients: ["西兰花", "胡萝卜"], excluded_ingredients: ["花生"] }, retrieval: mockRetrieval, generation: { mode: "api", effective_mode: "graphrag_api_evidence_qa", llm_called: true, api_call_attempted: true, api_calls_attempted: 3, api_status: "ok", fallback: false, model: "mock-model" }, rerank: { status: "ok", api_call_attempted: true, ranking: ["mock_recipe_1", "mock_recipe_2"] }, qa: { status: "answered", paragraphs }, paragraphs, recipes: mockRecipes.slice(0, 1), recipe_ids: ["mock_recipe_1"], evidence: mockEvidence, validation: { passed: true, checks: [{ name: "evidence_ids_exist", passed: true }] }, trace: ["【mock】图谱筛选 → E5 排序 → API 重排 → 引用问答。"] };
}
function scope(kind) { return { kind, status: "running", checks: {}, requests: [], chat_requests: [], responses: [], page_errors: [], console_errors: [], blocked_requests: [], request_failures: [], screenshots: {}, new_real_api_calls: kind === "mock" ? 0 : null }; }
function check(target, name, value) { target.checks[name] = Boolean(value); assert(value, `${target.kind}: ${name}`); }
function saveReport() { fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n"); }
function contentType(file) { return ({ ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "application/javascript; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png", ".webp": "image/webp", ".json": "application/json" })[path.extname(file)] || "application/octet-stream"; }
async function fulfillJSON(route, value, status = 200) { await route.fulfill({ status, contentType: "application/json; charset=utf-8", body: JSON.stringify(value) }); }
async function screenshot(page, target, name) { const file = path.join(screenshotDir, `${target.kind}-${name}.png`); await page.screenshot({ path: file, fullPage: true }); target.screenshots[name] = path.relative(root, file); }
function capture(page, target) {
  page.on("pageerror", error => target.page_errors.push(error.message));
  page.on("console", message => { if (message.type() === "error") target.console_errors.push(message.text()); });
  page.on("response", response => { const url = new URL(response.url()); if (url.origin === baseURL) target.requests.push({ method: response.request().method(), path: url.pathname, status: response.status() }); });
  page.on("requestfailed", request => target.request_failures.push({ path: new URL(request.url()).pathname, error: request.failure()?.errorText || "unknown" }));
}
async function noOverflow(page, target, name) { check(target, name, await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)); }
async function submit(page, message, sendWithEnter = false) {
  await page.locator("#chat-input").fill(message);
  const pending = page.waitForResponse(r => new URL(r.url()).pathname === "/api/chat" && r.request().method() === "POST", { timeout: 240000 });
  if (sendWithEnter) await page.locator("#chat-input").press("Enter");
  else await page.locator("#chat-send").click();
  const response = await pending; const data = await response.json();
  await page.waitForFunction(() => document.querySelector("#chat-form")?.getAttribute("aria-busy") === "false", null, { timeout: 240000 });
  return { response, data };
}
async function chatDOM(page, target, prefix = "") {
  check(target, `${prefix}chat_page`, await page.locator("body[data-page='chef-chat']").count() === 1);
  check(target, `${prefix}single_prompt_control`, await page.locator("#main-content textarea, #main-content input, #main-content select").count() === 1);
  check(target, `${prefix}no_legacy_controls`, await page.locator("#generation-question,#generation-ingredients,#generation-excluded-ingredients,#api-model,#choose-recipes-link,#clear-selection,#selected-recipes,.generation-modes,.filter-details").count() === 0);
  check(target, `${prefix}chef_icon`, await page.locator("#chat-welcome .chef-avatar[role='img']").count() === 1);
  check(target, `${prefix}new_chat_control`, await page.locator("#chat-new").isVisible());
}
async function checkCitation(page, target, prefix = "") {
  const citation = page.locator(".chat-message[data-role='assistant'] .chat-citation").first();
  check(target, `${prefix}citation_exists`, await citation.count() === 1);
  await citation.click();
  check(target, `${prefix}citation_reveals_original_evidence`, await page.locator("details.chat-sources[open] .chat-evidence").first().isVisible());
  const evidenceText = await page.locator("details.chat-sources[open]").first().textContent();
  check(target, `${prefix}citation_has_source_and_step`, evidenceText.includes("mock_source_1") || /来源|source:/i.test(evidenceText) && /图谱|步骤|step:/i.test(evidenceText));
}
let browser;
async function runMock() {
  const target = report.mock = scope("mock");
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, colorScheme: "light", reducedMotion: "reduce" });
  const page = await context.newPage(); page.setDefaultTimeout(10000); capture(page, target);
  let chatCount = 0;
  await context.route("**/*", async route => {
    const request = route.request(), url = new URL(request.url());
    if (url.origin !== baseURL) { target.blocked_requests.push({ origin: url.origin, path: url.pathname, method: request.method() }); return route.abort(); }
    if (url.pathname === "/api/system/status" && request.method() === "GET") return fulfillJSON(route, mockStatus);
    if (url.pathname === "/api/search" && request.method() === "POST") {
      const payload = request.postDataJSON(); target.search_request = payload;
      return fulfillJSON(route, { query: payload.query, recipes: mockRecipes, retrieval: mockRetrieval, limitations: ["【mock】界面验证数据"], trace: mockRetrieval.trace });
    }
    if (url.pathname === "/api/chat" && request.method() === "POST") {
      const payload = request.postDataJSON(); target.chat_requests.push(payload); chatCount++;
      if (chatCount === 3) return fulfillJSON(route, { status: "error", answer: null, error: { code: "upstream_error", message: "【mock】API 暂时不可用，请重试。" }, detail: "【mock】API 暂时不可用，请重试。", route: null }, 503);
      if (payload.message === "【mock】取消中的请求") {
        await new Promise(resolve => setTimeout(resolve, 1000));
        try { return await fulfillJSON(route, directResponse("【mock】取消请求不应被显示。")); } catch (_) { return; }
      }
      const value = chatCount === 1 ? directResponse() : retrievalResponse(); target.responses.push(value);
      return fulfillJSON(route, value);
    }
    let file = null;
    const pages = { "/": "index.html", "/compose": "compose.html", "/explore": "explore.html", "/about": "about.html" };
    if (request.method() === "GET" && pages[url.pathname]) file = path.join(root, "static", pages[url.pathname]);
    if (request.method() === "GET" && url.pathname.startsWith("/static/")) {
      const candidate = path.resolve(root, `.${url.pathname}`);
      if (candidate.startsWith(path.join(root, "static") + path.sep) && fs.existsSync(candidate) && fs.statSync(candidate).isFile()) file = candidate;
    }
    if (file) return route.fulfill({ status: 200, contentType: contentType(file), body: fs.readFileSync(file) });
    target.blocked_requests.push({ origin: url.origin, path: url.pathname, method: request.method() }); return route.abort();
  });
  try {
    await page.goto(`${baseURL}/compose`, { waitUntil: "networkidle" });
    await chatDOM(page, target);
    check(target, "welcome_visible", await page.locator("#chat-welcome").isVisible());
    check(target, "no_initial_messages", await page.locator(".chat-message").count() === 0);
    await screenshot(page, target, "desktop-empty");
    await noOverflow(page, target, "desktop_empty_no_overflow");
    await page.locator("#chat-input").fill("【mock】键盘换行检查");
    await page.locator("#chat-input").press("Shift+Enter");
    check(target, "shift_enter_adds_newline_without_send", (await page.locator("#chat-input").inputValue()).includes("\n") && target.chat_requests.length === 0);
    await page.locator("#chat-input").dispatchEvent("keydown", { key: "Enter", code: "Enter", keyCode: 229, isComposing: true });
    check(target, "ime_enter_does_not_send", target.chat_requests.length === 0);
    const firstMessage = `【mock】如何煮饭？${xssText}`;
    const first = await submit(page, firstMessage);
    check(target, "first_request_only_message_history", Object.keys(target.chat_requests[0]).sort().join(",") === "history,message");
    check(target, "first_history_empty", target.chat_requests[0].history.length === 0);
    check(target, "direct_response_displayed", (await page.locator(".chat-message[data-role='assistant']").first().textContent()).includes("先把米洗净"));
    check(target, "user_and_assistant_pair", await page.locator(".chat-message[data-role='user']").count() === 1 && await page.locator(".chat-message[data-role='assistant']").count() === 1);
    check(target, "user_input_xss_literal", (await page.locator(".chat-message[data-role='user']").first().textContent()).includes(xssText));
    check(target, "assistant_xss_literal", (await page.locator(".chat-message[data-role='assistant']").first().textContent()).includes(xssText));
    check(target, "chat_no_html_execution", await page.evaluate(() => !window.__chefXSS) && await page.locator(".chat-message img").count() === 0);
    const secondMessage = "【mock】西兰花和胡萝卜，不放花生，有哪些菜？";
    await submit(page, secondMessage, true);
    const second = target.chat_requests[1];
    check(target, "enter_sends_one_message", target.chat_requests.length === 2 && second.message === secondMessage);
    check(target, "history_pair_forwarded", JSON.stringify(second.history) === JSON.stringify([{ role: "user", content: firstMessage }, { role: "assistant", content: first.data.answer }]));
    check(target, "retrieve_response_displayed", (await page.locator(".chat-message[data-role='assistant']").last().textContent()).includes("在本次候选中"));
    await checkCitation(page, target);
    check(target, "evidence_xss_literal", (await page.locator("details.chat-sources[open]").first().textContent()).includes(xssText));
    check(target, "unsafe_source_link_rejected", await page.locator(".chat-message a[href^='javascript:']").count() === 0);
    const failedMessage = "【mock】再说明关键做法";
    const failed = await submit(page, failedMessage);
    check(target, "error_http_status", failed.response.status() === 503);
    check(target, "error_visible_and_retry_offered", await page.locator("#chat-error").isVisible() && await page.locator("#chat-retry").isVisible());
    const beforeRetryUsers = await page.locator(".chat-message[data-role='user']").count();
    const retryResponse = page.waitForResponse(r => new URL(r.url()).pathname === "/api/chat" && r.request().method() === "POST");
    await page.locator("#chat-retry").click(); await retryResponse;
    await page.waitForFunction(() => document.querySelector("#chat-form")?.getAttribute("aria-busy") === "false");
    check(target, "retry_same_payload_no_failed_history", JSON.stringify(target.chat_requests[2]) === JSON.stringify(target.chat_requests[3]) && target.chat_requests[3].history.length === 4);
    check(target, "retry_does_not_duplicate_user", await page.locator(".chat-message[data-role='user']").count() === beforeRetryUsers);
    check(target, "retry_success_hides_error", !(await page.locator("#chat-error").isVisible()));
    check(target, "message_scoped_unique_evidence_ids", await page.evaluate(() => { const ids = [...document.querySelectorAll(".chat-evidence[id]")].map(n => n.id); return ids.length >= 4 && new Set(ids).size === ids.length; }));
    await page.reload({ waitUntil: "networkidle" });
    check(target, "reload_restores_three_turns", await page.locator(".chat-message[data-role='user']").count() === 3 && await page.locator(".chat-message[data-role='assistant']").count() === 3);
    await checkCitation(page, target, "reload_");
    await screenshot(page, target, "desktop-answers");
    await page.setViewportSize({ width: 390, height: 844 });
    await noOverflow(page, target, "mobile_chat_no_overflow");
    check(target, "mobile_prompt_available", await page.locator("#chat-input").isVisible());
    await screenshot(page, target, "mobile-answers");
    await page.locator("#theme-toggle").click();
    check(target, "dark_theme_applied", await page.evaluate(() => document.documentElement.dataset.theme === "dark"));
    await noOverflow(page, target, "mobile_dark_no_overflow");
    await screenshot(page, target, "mobile-dark");
    await page.locator("#chat-new").click();
    check(target, "new_chat_clears_messages_and_input", await page.locator(".chat-message").count() === 0 && await page.locator("#chat-input").inputValue() === "");
    check(target, "new_chat_welcome_restored", await page.locator("#chat-welcome").isVisible());
    await page.locator("#chat-input").fill("【mock】取消中的请求");
    const requestStarted = page.waitForRequest(r => new URL(r.url()).pathname === "/api/chat" && r.method() === "POST");
    await page.locator("#chat-send").click(); await requestStarted;
    check(target, "busy_prevents_duplicate_send", await page.locator("#chat-send").isDisabled());
    check(target, "cancel_available_while_pending", await page.locator("#chat-cancel").isVisible());
    await page.locator("#chat-cancel").click();
    await page.waitForFunction(() => document.querySelector("#chat-form")?.getAttribute("aria-busy") === "false");
    await page.waitForTimeout(1200);
    check(target, "canceled_response_not_rendered", !(await page.locator("#chat-messages").textContent()).includes("取消请求不应被显示"));
    await page.locator("#chat-new").click();
    await page.goto(`${baseURL}/explore`, { waitUntil: "networkidle" });
    check(target, "explore_two_ingredient_controls_only", await page.locator("#search-form input,#search-form textarea,#search-form select").count() === 2);
    check(target, "explore_no_query_mode_selection_controls", await page.locator("#search-question,#retrieval-mode,#selection-bar,#selected-recipes,.select-button").count() === 0);
    await page.locator("#ingredients").fill("西兰花、胡萝卜");
    await page.locator("#excluded-ingredients").fill("花生");
    const searchResponse = page.waitForResponse(r => new URL(r.url()).pathname === "/api/search" && r.request().method() === "POST");
    await page.locator("#search-button").click(); await searchResponse;
    await page.waitForSelector(".recipe-card");
    check(target, "explore_fixed_graph_first_semantic_limit_12", target.search_request.retrieval_mode === "semantic" && target.search_request.limit === 12);
    check(target, "explore_conditions_passed", JSON.stringify(target.search_request.ingredients) === JSON.stringify(["西兰花", "胡萝卜"]) && JSON.stringify(target.search_request.excluded_ingredients) === JSON.stringify(["花生"]));
    check(target, "explore_direct_recipe_results", await page.locator(".recipe-card").count() === 2 && await page.locator(".card-view-button").count() === 2);
    check(target, "recipe_title_xss_literal", (await page.locator(".recipe-card").first().textContent()).includes(xssText) && await page.locator(".recipe-card img").count() === 0);
    await noOverflow(page, target, "mobile_explore_no_overflow");
    await screenshot(page, target, "mobile-explore");
    await page.goto(`${baseURL}/compose`, { waitUntil: "networkidle" });
    check(target, "new_chat_remains_empty_across_navigation", await page.locator(".chat-message").count() === 0);
    await submit(page, "【mock】新对话的第一问");
    check(target, "new_conversation_history_empty", target.chat_requests.at(-1).history.length === 0);
    check(target, "dark_theme_persists_navigation", await page.evaluate(() => document.documentElement.dataset.theme === "dark"));
    const longMessages = Array.from({ length: 20 }, (_, index) => {
      const prefix = `【mock 完整长历史 ${index}】`;
      return { id: `mock-long-${index}`, role: index % 2 === 0 ? "user" : "assistant", content: prefix + "长".repeat(4000 - prefix.length), confirmed: true };
    });
    // Leave chat before seeding: its pagehide handler correctly saves its current state.
    await page.goto(`${baseURL}/about`, { waitUntil: "networkidle" });
    await page.evaluate(messages => sessionStorage.setItem("recipegen-chef-chat-v1", JSON.stringify({ version: 1, messages, draft: "" })), longMessages);
    await page.goto(`${baseURL}/compose`, { waitUntil: "networkidle" });
    check(target, "long_history_restores_twenty_full_messages", await page.locator(".chat-message").count() === 20 && (await page.locator("#chat-message-mock-long-19 .chat-message-body").textContent()).includes(longMessages[19].content));
    const longMessage = "【mock 1000字本轮消息】" + "问".repeat(1000 - "【mock 1000字本轮消息】".length);
    await submit(page, longMessage);
    const bounded = target.chat_requests.at(-1);
    check(target, "message_plus_history_bounded_20000", bounded.message.length + bounded.history.reduce((total, item) => total + item.content.length, 0) <= 20000);
    check(target, "long_history_each_message_bounded_4000", bounded.history.every(item => item.content.length <= 4000));
    check(target, "history_message_count_bounded_20", bounded.history.length <= 20 && await page.locator(".chat-message").count() <= 20);
    check(target, "long_current_message_not_truncated", bounded.message === longMessage && bounded.message.length === 1000);
    check(target, "display_retains_full_history_text", (await page.locator("#chat-message-mock-long-19 .chat-message-body").textContent()).includes(longMessages[19].content));
    await noOverflow(page, target, "long_history_mobile_no_overflow");
    check(target, "zero_real_api_calls", target.new_real_api_calls === 0);
    check(target, "no_page_errors", target.page_errors.length === 0);
    check(target, "no_unexpected_outbound_requests", target.blocked_requests.length === 0);
    // A cancelled mock request may emit ERR_ABORTED; deliberate HTTP 503 is also expected.
    check(target, "only_expected_request_failures", target.request_failures.every(r => r.path === "/api/chat" && /ABORT|CANCEL/i.test(r.error)));
    target.expected_error_paths = ["one HTTP 503 mock retry", "one cancelled /api/chat mock request"];
    target.status = "passed";
  } catch (error) { target.status = "failed"; target.error = error.message; throw error; }
  finally { await context.close(); }
}
async function runReal() {
  const target = report.real = scope("real");
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, colorScheme: "light", reducedMotion: "reduce" });
  const page = await context.newPage(); page.setDefaultTimeout(240000); capture(page, target);
  await context.route("**/*", async route => {
    const request = route.request(), url = new URL(request.url());
    let allowed = url.origin === baseURL;
    if (allowed && url.pathname.startsWith("/api/")) {
      allowed = request.method() === "GET" && url.pathname === "/api/system/status";
      if (request.method() === "POST" && url.pathname === "/api/chat") {
        const payload = request.postDataJSON();
        allowed = runAPI && Object.keys(payload || {}).sort().join(",") === "history,message" && Array.isArray(payload.history) && typeof payload.message === "string";
        target.chat_requests.push({ payload, allowed });
      }
      if (request.method() === "POST" && url.pathname === "/api/search") { allowed = request.postDataJSON()?.retrieval_mode === "semantic"; target.search_request = request.postDataJSON(); }
    }
    if (!allowed) { target.blocked_requests.push({ origin: url.origin, path: url.pathname, method: request.method() }); return route.abort(); }
    return route.continue();
  });
  try {
    const statusPending = page.waitForResponse(r => new URL(r.url()).pathname === "/api/system/status");
    await page.goto(`${baseURL}/compose`, { waitUntil: "domcontentloaded" });
    await page.evaluate(() => fetch("/api/system/status").then(response => response.json()));
    const statusResponse = await statusPending; const status = await statusResponse.json();
    target.system = { backend: status.backend, graph: status.graph, retrieval: status.retrieval, generation: status.generation };
    check(target, "system_status_http_200", statusResponse.status() === 200);
    check(target, "neo4j_backend", status.backend === "neo4j");
    check(target, "semantic_ready", status.retrieval?.default_mode === "semantic" && status.retrieval?.semantic_available === true);
    check(target, "api_configured", status.generation?.api_configured === true);
    await chatDOM(page, target, "real_");
    await screenshot(page, target, "desktop-empty");
    const direct = await submit(page, "切菜时怎样握刀更稳？请解释通用技巧，不用查食谱。");
    target.responses.push({ label: "general_cooking_direct", http_status: direct.response.status(), ...direct.data });
    check(target, "general_http_200", direct.response.status() === 200);
    check(target, "general_llm_respond_route", direct.data.route?.action === "respond" && direct.data.route?.llm_decided === true);
    check(target, "general_answer_present", typeof direct.data.answer === "string" && direct.data.answer.length > 0);
    check(target, "general_no_retrieval", !direct.data.retrieval && (direct.data.evidence || []).length === 0);
    check(target, "general_one_real_api_call", direct.data.generation?.api_calls_attempted === 1 && direct.data.generation?.llm_called === true);
    check(target, "general_answer_rendered", (await page.locator(".chat-message[data-role='assistant']").first().textContent()).includes(direct.data.answer.slice(0, 20)));
    const retrieved = await submit(page, "请在 RecipeGen 中推荐同时用西兰花和胡萝卜、不放花生的食谱，说明关键做法和出处。");
    target.responses.push({ label: "broccoli_carrot_exclude_peanut_graph_qa", http_status: retrieved.response.status(), ...retrieved.data });
    check(target, "retrieval_http_200", retrieved.response.status() === 200);
    check(target, "retrieval_llm_decides_retrieve", retrieved.data.route?.action === "retrieve" && retrieved.data.route?.llm_decided === true);
    check(target, "retrieval_positive_ingredients_parsed", ["西兰花", "胡萝卜"].every(v => retrieved.data.route.ingredients.includes(v)));
    check(target, "retrieval_exclusion_parsed", retrieved.data.route.excluded_ingredients.includes("花生"));
    check(target, "graph_first_method", retrieved.data.retrieval?.method === "graph_first_e5_exact_cosine");
    check(target, "all_graph_candidates_e5_scored", retrieved.data.retrieval?.semantic?.graph_candidate_count > 0 && retrieved.data.retrieval.semantic.scored_recipe_count === retrieved.data.retrieval.semantic.graph_candidate_count);
    check(target, "real_deepseek_rerank", retrieved.data.rerank?.status === "ok" && retrieved.data.rerank?.api_call_attempted === true);
    check(target, "real_evidence_qa", retrieved.data.qa?.status === "answered" && retrieved.data.qa?.paragraphs?.length > 0);
    check(target, "three_real_api_calls_for_retrieval", retrieved.data.generation?.api_calls_attempted === 3 && retrieved.data.generation?.llm_called === true);
    check(target, "retrieval_no_generation_fallback", retrieved.data.generation?.fallback === false && retrieved.data.generation?.api_status === "ok");
    check(target, "retrieval_programmatic_validation", retrieved.data.validation?.passed === true);
    const evidence = new Map((retrieved.data.evidence || []).map(v => [v.id, v]));
    check(target, "all_paragraph_citations_resolve", retrieved.data.qa.paragraphs.every(p => p.evidence_ids?.length && p.evidence_ids.every(id => evidence.has(id))));
    check(target, "citations_have_original_steps_and_sources", retrieved.data.qa.paragraphs.every(p => p.evidence_ids.every(id => { const e = evidence.get(id); return e.kind === "step" && typeof e.text === "string" && e.text.length && typeof e.graph_id === "string" && typeof e.source_id === "string"; })));
    check(target, "multi_turn_request_history", target.chat_requests[1].payload.history.length === 2 && target.chat_requests[1].payload.history[0].role === "user" && target.chat_requests[1].payload.history[1].content === direct.data.answer);
    await checkCitation(page, target, "real_");
    await screenshot(page, target, "desktop-answers");
    await page.setViewportSize({ width: 390, height: 844 });
    await noOverflow(page, target, "real_mobile_no_overflow");
    await screenshot(page, target, "mobile-answers");
    await page.locator("#theme-toggle").click();
    check(target, "real_dark_theme", await page.evaluate(() => document.documentElement.dataset.theme === "dark"));
    await noOverflow(page, target, "real_mobile_dark_no_overflow");
    await screenshot(page, target, "mobile-dark");
    if (followup) {
      const last = await submit(page, "只看你刚才推荐的第一道，它的蔬菜怎么处理？");
      target.responses.push({ label: "contextual_followup", http_status: last.response.status(), ...last.data });
      check(target, "followup_http_200", last.response.status() === 200);
      check(target, "followup_four_history_messages", target.chat_requests[2].payload.history.length === 4);
      check(target, "followup_answer_present", typeof last.data.answer === "string" && last.data.answer.length > 0);
      check(target, "followup_llm_decision", ["respond", "retrieve"].includes(last.data.route?.action) && last.data.route?.llm_decided === true);
      if (last.data.route.action === "retrieve") {
        check(target, "followup_retrieval_has_source_citations", last.data.qa?.paragraphs?.length > 0 && last.data.evidence?.length > 0 && last.data.validation?.passed === true);
      } else {
        check(target, "followup_direct_no_claimed_new_graph_evidence", !last.data.retrieval && (last.data.evidence || []).length === 0 && last.data.generation?.api_calls_attempted === 1);
      }
    }
    await page.goto(`${baseURL}/explore`, { waitUntil: "domcontentloaded" });
    check(target, "real_explore_two_controls", await page.locator("#search-form input,#search-form textarea,#search-form select").count() === 2);
    await page.locator("#ingredients").fill("西兰花、胡萝卜");
    await page.locator("#excluded-ingredients").fill("花生");
    const searchPending = page.waitForResponse(r => new URL(r.url()).pathname === "/api/search" && r.request().method() === "POST");
    await page.locator("#search-button").click();
    const searchResponse = await searchPending, searched = await searchResponse.json();
    target.search_response = { http_status: searchResponse.status(), ...searched };
    check(target, "real_explore_http_200", searchResponse.status() === 200);
    check(target, "real_explore_limit_12_semantic", target.search_request.limit === 12 && target.search_request.retrieval_mode === "semantic");
    check(target, "real_explore_graph_first_e5", searched.retrieval?.method === "graph_first_e5_exact_cosine");
    check(target, "real_explore_recipe_count", searched.recipes?.length > 0 && searched.recipes.length <= 12);
    check(target, "real_explore_conditions_passed", JSON.stringify(target.search_request.ingredients) === JSON.stringify(["西兰花", "胡萝卜"]) && JSON.stringify(target.search_request.excluded_ingredients) === JSON.stringify(["花生"]));
    const mentions = recipe => (recipe.ingredient_mentions || []).map(v => v.zh_name || v.name || "");
    check(target, "real_explore_results_meet_graph_conditions", searched.recipes.every(r => ["西兰花", "胡萝卜"].every(v => mentions(r).some(m => m.includes(v))) && !mentions(r).some(m => m.includes("花生"))));
    await page.waitForSelector(".recipe-card");
    check(target, "real_explore_cards_rendered", await page.locator(".recipe-card").count() === searched.recipes.length);
    await noOverflow(page, target, "real_explore_mobile_no_overflow");
    await screenshot(page, target, "mobile-explore");
    target.new_real_api_calls = target.responses.reduce((sum, value) => sum + (value.generation?.api_calls_attempted || 0), 0);
    check(target, "authorized_real_request_count", target.chat_requests.length === (followup ? 3 : 2) && target.chat_requests.every(r => r.allowed));
    check(target, "no_page_errors", target.page_errors.length === 0);
    check(target, "no_external_or_legacy_api_requests", target.blocked_requests.length === 0);
    check(target, "no_request_failures", target.request_failures.length === 0);
    target.status = "passed";
  } catch (error) { target.status = "failed"; target.error = error.message; throw error; }
  finally { await context.close(); }
}
async function runRealRecheck() {
  const target = report.real = scope("real");
  target.scope = "two_real_retrieve_turns_recheck_using_v4_recorded_history";
  target.other_previous_turns_replayed_from_saved_record = true;
  const sourcePath = path.join(root, "reports/chef-chat-browser-v4.json");
  const source = JSON.parse(fs.readFileSync(sourcePath, "utf8"));
  assert(source.status === "passed" && source.real?.status === "passed", "有限复测需要已通过的真实 v4 报告");
  const previous = source.real.chat_requests[1].payload;
  assert(previous.history.length === 2 && previous.history[0].role === "user" && previous.history[1].role === "assistant", "v4 第二轮须包含一轮已确认历史");
  const crypto = require("crypto");
  target.input_report = { path: "reports/chef-chat-browser-v4.json", sha256: crypto.createHash("sha256").update(fs.readFileSync(sourcePath)).digest("hex") };
  const seed = previous.history.map((message, index) => ({ id: `v4-saved-${index}`, role: message.role, content: message.content, confirmed: true, ...(message.role === "assistant" ? { result: source.real.responses[0] } : {}) }));
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, colorScheme: "light", reducedMotion: "reduce" });
  await context.addInitScript(({ origin, messages }) => { if (location.origin === origin) sessionStorage.setItem("recipegen-chef-chat-v1", JSON.stringify({ version: 1, messages, draft: "" })); }, { origin: baseURL, messages: seed });
  const page = await context.newPage(); page.setDefaultTimeout(240000); capture(page, target);
  let expectedFollowup = null;
  await context.route("**/*", async route => {
    const request = route.request(), url = new URL(request.url());
    let allowed = url.origin === baseURL;
    if (allowed && url.pathname.startsWith("/api/")) {
      allowed = request.method() === "GET" && url.pathname === "/api/system/status";
      if (request.method() === "POST" && url.pathname === "/api/chat") {
        const payload = request.postDataJSON(), expected = target.chat_requests.length === 0 ? previous : expectedFollowup;
        allowed = runAPI && target.chat_requests.length < 2 && expected && JSON.stringify(payload) === JSON.stringify(expected);
        target.chat_requests.push({ payload, allowed });
      }
    }
    if (!allowed) { target.blocked_requests.push({ origin: url.origin, path: url.pathname, method: request.method() }); return route.abort(); }
    return route.continue();
  });
  try {
    await page.goto(`${baseURL}/compose`, { waitUntil: "domcontentloaded" });
    const status = await page.evaluate(() => fetch("/api/system/status").then(response => response.json()));
    target.system = { backend: status.backend, retrieval: status.retrieval, generation: status.generation };
    check(target, "real_neo4j_api_semantic_ready", status.backend === "neo4j" && status.retrieval?.semantic_available === true && status.generation?.api_configured === true);
    check(target, "v4_first_turn_replayed_without_new_api", await page.locator(".chat-message[data-role='user']").count() === 1 && await page.locator(".chat-message[data-role='assistant']").count() === 1 && target.chat_requests.length === 0);
    check(target, "single_prompt_no_legacy_controls", await page.locator("#main-content textarea,#main-content input,#main-content select").count() === 1 && await page.locator("#api-model,#generation-ingredients,#generation-excluded-ingredients").count() === 0);
    const result = await submit(page, previous.message);
    target.responses.push({ label: "v4_retrieval_question_after_source_support_fix", http_status: result.response.status(), ...result.data });
    target.new_real_api_calls = result.data.generation?.api_calls_attempted ?? null; saveReport();
    check(target, "same_message_history_as_v4", target.chat_requests.length === 1 && target.chat_requests[0].allowed);
    check(target, "http_200", result.response.status() === 200);
    check(target, "llm_retrieve_route", result.data.route?.action === "retrieve" && result.data.route?.llm_decided === true);
    check(target, "parsed_ingredient_conditions", ["西兰花", "胡萝卜"].every(v => result.data.route.ingredients.includes(v)) && result.data.route.excluded_ingredients.includes("花生"));
    check(target, "graph_first_e5", result.data.retrieval?.method === "graph_first_e5_exact_cosine");
    check(target, "router_structured_conditions_no_second_regex_inference", result.data.retrieval?.query_filter_inference === false);
    check(target, "all_candidates_scored", result.data.retrieval?.semantic?.graph_candidate_count > 0 && result.data.retrieval.semantic.scored_recipe_count === result.data.retrieval.semantic.graph_candidate_count);
    check(target, "real_rerank_and_qa", result.data.rerank?.status === "ok" && result.data.qa?.status === "answered");
    check(target, "no_fallback", result.data.generation?.api_status === "ok" && result.data.generation?.fallback === false);
    check(target, "three_real_api_calls", result.data.generation?.api_calls_attempted === 3 && result.data.generation?.llm_called === true);
    check(target, "programmatic_validation_passed", result.data.validation?.passed === true);
    const evidence = new Map((result.data.evidence || []).map(item => [item.id, item]));
    const paragraphs = result.data.paragraphs || result.data.qa?.paragraphs || [];
    check(target, "paragraph_citations_resolve", paragraphs.length > 0 && paragraphs.every(p => p.evidence_ids?.length && p.evidence_ids.every(id => evidence.has(id))));
    check(target, "citations_original_step_source", paragraphs.every(p => p.evidence_ids.every(id => { const e = evidence.get(id); return e.kind === "step" && typeof e.text === "string" && e.text.length > 0 && typeof e.source_id === "string" && typeof e.graph_id === "string"; })));
    const additionalPan = /另起.{0,4}锅|另外.{0,6}锅|另一(?:口|个)?锅|another\s+(?:pot|pan|wok)|separate\s+(?:pot|pan|wok)|second\s+(?:pot|pan|wok)|new\s+(?:pot|pan|wok)/i;
    check(target, "no_unbacked_additional_pan_wording", paragraphs.every(p => !additionalPan.test(p.text) || additionalPan.test(p.evidence_ids.map(id => evidence.get(id).original_text || evidence.get(id).text).join("\n"))));
    await checkCitation(page, target, "recheck_");
    expectedFollowup = { message: source.real.chat_requests[2].payload.message, history: [...previous.history, { role: "user", content: previous.message }, { role: "assistant", content: result.data.answer }] };
    const follow = await submit(page, expectedFollowup.message);
    target.responses.push({ label: "v4_contextual_followup_after_source_support_fix", http_status: follow.response.status(), ...follow.data });
    target.new_real_api_calls = target.responses.reduce((total, item) => total + (item.generation?.api_calls_attempted || 0), 0); saveReport();
    check(target, "followup_http_200", follow.response.status() === 200);
    check(target, "followup_history_uses_new_answer", target.chat_requests.length === 2 && target.chat_requests[1].allowed && target.chat_requests[1].payload.history[3].content === result.data.answer);
    check(target, "followup_retrieve_route", follow.data.route?.action === "retrieve" && follow.data.route?.llm_decided === true);
    check(target, "followup_inherits_peanut_exclusion", follow.data.route?.excluded_ingredients?.includes("花生"));
    check(target, "followup_no_second_query_filter_inference", follow.data.retrieval?.query_filter_inference === false);
    check(target, "followup_graph_first_e5", follow.data.retrieval?.method === "graph_first_e5_exact_cosine");
    check(target, "followup_same_first_recommended_recipe", result.data.recipes?.length > 0 && follow.data.recipes?.[0]?.id === result.data.recipes[0].id);
    check(target, "followup_no_fallback_api_qa", follow.data.generation?.api_status === "ok" && follow.data.generation?.fallback === false && follow.data.rerank?.status === "ok" && follow.data.qa?.status === "answered");
    check(target, "followup_three_new_api_calls", follow.data.generation?.api_calls_attempted === 3);
    check(target, "followup_programmatic_validation", follow.data.validation?.passed === true);
    const followEvidence = new Map((follow.data.evidence || []).map(item => [item.id, item]));
    const followParagraphs = follow.data.paragraphs || follow.data.qa?.paragraphs || [];
    check(target, "followup_citations_resolve", followParagraphs.length > 0 && followParagraphs.every(p => p.evidence_ids?.length && p.evidence_ids.every(id => followEvidence.has(id))));
    check(target, "followup_no_unbacked_additional_pan", followParagraphs.every(p => !additionalPan.test(p.text) || additionalPan.test(p.evidence_ids.map(id => followEvidence.get(id).original_text || followEvidence.get(id).text).join("\n"))));
    await screenshot(page, target, "desktop-source-support-recheck");
    await page.setViewportSize({ width: 390, height: 844 });
    await noOverflow(page, target, "mobile_no_overflow");
    await screenshot(page, target, "mobile-source-support-recheck");
    check(target, "no_page_errors", target.page_errors.length === 0);
    check(target, "no_external_or_unbudgeted_requests", target.blocked_requests.length === 0);
    check(target, "no_request_failures", target.request_failures.length === 0);
    target.status = "passed";
  } catch (error) { target.status = "failed"; target.error = error.message; throw error; }
  finally { await context.close(); }
}
(async () => {
  browser = await playwright.chromium.launch({ executablePath, headless: true });
  report.browser = { version: browser.version(), executable: executablePath, isolated_contexts: true };
  await runMock(); saveReport();
  if (runAPI) { if (recheckV4) await runRealRecheck(); else await runReal(); }
  report.status = "passed";
})().catch(error => { report.status = "failed"; report.error = error.message; process.exitCode = 1; }).finally(async () => {
  if (browser) await browser.close(); report.finished_at = new Date().toISOString(); saveReport();
  console.log(JSON.stringify({ status: report.status, report: reportPath, mock_checks: report.mock ? Object.keys(report.mock.checks).length : 0, real_checks: report.real ? Object.keys(report.real.checks).length : 0, new_real_api_calls: report.real?.new_real_api_calls || 0, error: report.error || null }, null, 2));
});
