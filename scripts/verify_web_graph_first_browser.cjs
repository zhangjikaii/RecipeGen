/* RecipeGen 图谱优先检索与真实 API 问答的网页验收。
 * 运行：node scripts/verify_web_graph_first_browser.cjs --run-api [--report-label v2]
 * --run-api 明确允许网页发起 DeepSeek 重排及问答；不读取密钥，不访问来源外链。
 */
"use strict";
const fs = require("fs"), path = require("path"), os = require("os"), assert = require("assert");
const root = path.resolve(__dirname, "..");
const args = process.argv.slice(2);
let baseURL = "http://127.0.0.1:8765", runAPI = false, reportLabel = "v1";
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--run-api") runAPI = true;
  else if (args[i] === "--base-url" && args[i + 1]) baseURL = args[++i];
  else if (args[i] === "--report-label" && args[i + 1]) reportLabel = args[++i];
  else throw new Error(`未知参数：${args[i]}`);
}
assert(runAPI, "真实完整流程验收需要显式 --run-api；当前未发送任何请求。");
assert(/^v[1-9]\d*$/.test(reportLabel), "报告标签须为 v1、v2 等版本号");
const base = new URL(baseURL);
assert(base.protocol === "http:" && ["127.0.0.1", "localhost", "[::1]"].includes(base.hostname) && base.port === "8765" && !base.username && !base.password, "仅允许无凭据的本机 HTTP 8765 服务");
baseURL = base.origin;
const playwright = require(path.join(os.homedir(), ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright"));
const executablePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const reportPath = path.join(root, `reports/web-graph-first-browser-${reportLabel}.json`);
const screenshotDir = path.join(root, reportLabel === "v1" ? "reports/web-graph-first-browser-screenshots" : `reports/web-graph-first-browser-screenshots-${reportLabel}`);
fs.mkdirSync(screenshotDir, { recursive: true });
const report = { schema_version: "recipegen-web-graph-first-browser-v1", report_label: reportLabel, started_at: new Date().toISOString(), scope: "real_local_http_graph_first_e5_deepseek_rerank_evidence_qa", status: "running", base_url: baseURL, run_api_authorized: runAPI, checks: {}, page_errors: [], console_errors: [], blocked_requests: [], request_failures: [], requests: [], screenshots: {}, semantic_accuracy_verified: false, real_api_requests: [] };
let browser;
function check(name, value) { report.checks[name] = Boolean(value); assert(value, name); }
function saveReport() { fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n"); }
(async () => {
  browser = await playwright.chromium.launch({ executablePath, headless: true });
  report.browser = { version: browser.version(), executable: executablePath, isolated_context: true };
  const context = await browser.newContext({ viewport: { width: 1280, height: 1000 }, reducedMotion: "reduce" });
  const page = await context.newPage(); page.setDefaultTimeout(240000);
  page.on("pageerror", error => report.page_errors.push(error.message));
  page.on("console", message => { if (message.type() === "error") report.console_errors.push(message.text()); });
  page.on("response", response => { const url = new URL(response.url()); if (url.origin === baseURL) report.requests.push({ method: response.request().method(), path: url.pathname, status: response.status() }); });
  page.on("requestfailed", request => report.request_failures.push({ path: new URL(request.url()).pathname, error: request.failure()?.errorText || "unknown" }));
  await context.route("**/*", async route => {
    const request = route.request(), url = new URL(request.url());
    let allowed = url.origin === baseURL;
    if (allowed && url.pathname.startsWith("/api/")) {
      allowed = request.method() === "GET" && url.pathname === "/api/system/status";
      if (request.method() === "POST" && url.pathname === "/api/generate") {
        const payload = request.postDataJSON();
        allowed = runAPI && payload?.mode === "api" && payload?.retrieval_mode === "semantic" && (!Object.hasOwn(payload, "recipe_ids") || payload.recipe_ids.length === 0);
        report.real_api_requests.push({ payload, allowed });
      }
    }
    if (!allowed) { report.blocked_requests.push({ origin: url.origin, path: url.pathname, method: request.method() }); await route.abort(); }
    else await route.continue();
  });
  const statusPromise = page.waitForResponse(response => new URL(response.url()).pathname === "/api/system/status");
  await page.goto(`${baseURL}/compose`, { waitUntil: "domcontentloaded" });
  const statusResponse = await statusPromise; check("system_status_http_200", statusResponse.status() === 200);
  const status = await statusResponse.json();
  report.system = { backend: status.backend, dataset: status.dataset, graph: status.graph, retrieval: status.retrieval, generation: status.generation };
  check("neo4j_backend", status.backend === "neo4j");
  check("default_semantic", status.retrieval?.default_mode === "semantic");
  check("semantic_ready", status.retrieval?.semantic_available === true);
  check("api_configured", status.generation?.api_configured === true);
  check("api_toggle_checked_by_default", await page.locator("#api-model").isChecked());
  check("no_selected_recipe", !(await page.locator("#selection-label").innerText()).includes("已选择"));
  await page.locator("#generation-question").fill("同时使用西兰花和胡萝卜的食谱有哪些？请结合原始步骤说明一道菜的关键做法。");
  await page.locator(".filter-details summary").click();
  await page.locator("#generation-ingredients").fill("西兰花、胡萝卜");
  check("direct_question_enabled", !(await page.locator("#generate-button").isDisabled()));
  const generatedPromise = page.waitForResponse(response => new URL(response.url()).pathname === "/api/generate" && response.request().method() === "POST");
  await page.locator("#generate-button").click();
  const response = await generatedPromise; check("generate_http_200", response.status() === 200);
  const data = await response.json();
  report.response = { status: data.status, answer: data.answer, recipe_ids: (data.recipes || []).map(r => r.id), retrieval: data.retrieval, rerank: data.rerank, qa: data.qa, generation: data.generation, validation: data.validation, evidence: data.evidence, trace: data.trace };
  check("single_api_generate_request", report.real_api_requests.length === 1 && report.real_api_requests[0].allowed);
  check("graph_first_method", data.retrieval?.method === "graph_first_e5_exact_cosine");
  check("effective_semantic", data.retrieval?.effective_mode === "semantic");
  check("graph_candidates_40", data.retrieval?.semantic?.graph_candidate_count === 40);
  check("all_graph_candidates_scored", data.retrieval?.semantic?.scored_recipe_count === 40);
  check("api_rerank_completed", data.rerank?.status === "ok" && data.rerank?.api_call_attempted === true);
  check("qa_answered", data.qa?.status === "answered" && Array.isArray(data.qa?.paragraphs) && data.qa.paragraphs.length > 0);
  check("qa_model_called", data.generation?.llm_called === true && data.generation?.api_status === "ok");
  check("qa_effective_mode", data.generation?.effective_mode === "graphrag_api_evidence_qa");
  check("no_generation_fallback", data.generation?.fallback === false);
  check("programmatic_validation_passed", data.validation?.passed === true);
  const evidence = new Map((data.evidence || []).map(item => [item.id, item]));
  check("paragraph_citations_exist", data.qa.paragraphs.every(item => item.evidence_ids.length > 0 && item.evidence_ids.every(id => evidence.has(id))));
  check("cited_evidence_has_original_record", data.qa.paragraphs.every(item => item.evidence_ids.every(id => { const record = evidence.get(id); return record.kind === "step" && typeof record.text === "string" && record.text.length > 0 && typeof record.graph_id === "string"; })));
  await page.waitForSelector("#answer-text a[href^='#answer-evidence-']");
  check("qa_answer_visible", await page.locator("#answer-text").isVisible());
  check("rerank_status_visible_in_disclosure", (await page.locator("#generation-badges").textContent()).includes("API 重排完成"));
  const firstCitation = page.locator("#answer-text a[href^='#answer-evidence-']").first();
  const targetID = (await firstCitation.getAttribute("href")).slice(1);
  await firstCitation.click();
  const evidenceNode = page.locator(`#${targetID}`);
  check("citation_reveals_original_evidence", await evidenceNode.isVisible());
  check("citation_shows_source_id", (await evidenceNode.textContent()).includes("来源 ID："));
  check("citation_shows_graph_id", (await evidenceNode.textContent()).includes("图谱记录："));
  await page.locator("#generation-retrieval-summary > summary").click();
  await page.locator(".answer-detail > summary").last().click();
  await page.screenshot({ path: path.join(screenshotDir, "compose-desktop.png"), fullPage: true });
  report.screenshots.desktop = path.relative(root, path.join(screenshotDir, "compose-desktop.png"));
  await page.setViewportSize({ width: 390, height: 844 });
  check("mobile_no_horizontal_overflow", await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
  await page.screenshot({ path: path.join(screenshotDir, "compose-mobile.png"), fullPage: true });
  report.screenshots.mobile = path.relative(root, path.join(screenshotDir, "compose-mobile.png"));
  check("no_page_errors", report.page_errors.length === 0);
  check("no_external_requests", report.blocked_requests.length === 0);
  check("no_request_failures", report.request_failures.length === 0);
  report.status = "passed";
})().catch(error => { report.status = "failed"; report.error = error.message; process.exitCode = 1; }).finally(async () => {
  if (browser) await browser.close(); report.finished_at = new Date().toISOString(); saveReport();
  console.log(JSON.stringify({ status: report.status, report: reportPath, checks: report.checks, error: report.error || null }, null, 2));
});
