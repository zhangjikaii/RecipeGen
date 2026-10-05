"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const state = { status: null, recipes: [], selected: new Map(), searching: false, generating: false, lastSearch: null, detail: null, detailController: null, graphController: null, detailVersion: 0, toastTimer: null, retrievalModeTouched: false };
  const SVG_NS = "http://www.w3.org/2000/svg";
  const array = (value) => Array.isArray(value) ? value : [];
  const text = (value, fallback = "") => typeof value === "string" ? value : typeof value === "number" ? String(value) : fallback;
  const count = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
  const number = (value) => count(value) === null ? "—" : value.toLocaleString("zh-CN");
  function el(tag, className, value) { const node = document.createElement(tag); if (className) node.className = className; if (value !== undefined) node.textContent = String(value); return node; }
  function svgEl(tag, attributes = {}, value) { const node = document.createElementNS(SVG_NS, tag); Object.entries(attributes).forEach(([key, val]) => node.setAttribute(key, String(val))); if (value !== undefined) node.textContent = String(value); return node; }
  function split(value) { return [...new Set(value.split(/[,，、;；\n]+/).map((v) => v.trim()).filter(Boolean))]; }
  function safeURL(value) { if (typeof value !== "string") return null; try { const url = new URL(value); return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password ? url.href : null; } catch (_) { return null; } }
  function link(value, label = "查看原始来源 ↗") { const url = safeURL(value); if (!url) return el("span", "field-help", "未提供可访问的网页来源"); const node = el("a", "source-link", label); node.href = url; node.target = "_blank"; node.rel = "noopener noreferrer"; return node; }
  function errorText(data, status) { const detail = data?.detail || data?.message || data?.error; if (typeof detail === "string") return detail; if (Array.isArray(detail)) return detail.map((v) => text(v.msg, "请求参数不符合接口要求")).join("；"); return `请求未完成（HTTP ${status}），请稍后重试。`; }
  async function request(path, options = {}, timeout = 60000) {
    const controller = new AbortController(); const abort = () => controller.abort(); let timedOut = false;
    if (options.signal?.aborted) controller.abort(); options.signal?.addEventListener("abort", abort, { once: true });
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeout);
    try { const response = await fetch(path, { ...options, signal: controller.signal }); let data; try { data = await response.json(); } catch (_) { throw new Error(`无法读取服务响应（HTTP ${response.status}）。`); } if (!response.ok) throw new Error(errorText(data, response.status)); if (!data || typeof data !== "object") throw new Error("服务返回了不完整的数据，请重试。"); return data; }
    catch (error) { if (timedOut) throw new Error("服务响应超时，当前操作尚未确认完成。请稍后重试。"); if (error.name === "AbortError") throw error; if (error instanceof TypeError) throw new Error("无法连接服务，请确认 RecipeGen 已启动后重试。"); throw error; }
    finally { clearTimeout(timer); options.signal?.removeEventListener("abort", abort); }
  }
  function post(path, payload, timeout) { return request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }, timeout); }
  function showError(id, message) { $(id).textContent = message; $(id).hidden = false; }
  function toast(message) { clearTimeout(state.toastTimer); $("toast").textContent = message; $("toast").hidden = false; state.toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3600); }
  function modelName(model) { return typeof model === "string" ? model : text(model?.name || model?.model || model?.repo, "名称未提供"); }
  function modelAvailable() { return state.status?.generation?.local_model_available === true; }
  function apiAvailable() { return state.status?.generation?.api_configured === true; }
  function modelCallStatus(generation) { if (generation.mode === "api" && generation.api_call_attempted === true && generation.api_status !== "ok") return "API 请求已尝试 · 生成结果未确认"; if (generation.inference_started_unknown === true) return "模型调用状态未确认"; return generation.llm_called === true ? "模型已实际调用" : generation.llm_called === false ? "未调用模型" : "模型调用状态未提供"; }
  function retrievalMode() { return ["keyword", "semantic", "hybrid"].includes($("retrieval-mode").value) ? $("retrieval-mode").value : "hybrid"; }
  function retrievalName(mode) { return { keyword: "关键词检索", semantic: "语义检索", hybrid: "混合检索", selected_recipes: "已选食谱证据" }[mode] || text(mode, "模式未提供"); }
  function updateCapabilities() {
    const retrieval = state.status?.retrieval || {}, index = retrieval.index || {};
    if (!state.retrievalModeTouched && ["keyword", "semantic", "hybrid"].includes(retrieval.default_mode)) $("retrieval-mode").value = retrieval.default_mode;
    const availability = retrieval.semantic_available === true ? "文本语义检索可用" : retrieval.semantic_available === false ? "文本语义检索当前不可用" : "服务尚未提供语义检索状态";
    $("retrieval-capability").textContent = `${availability}${typeof index.count === "number" ? ` · ${number(index.count)} 条索引记录` : ""}${typeof index.status === "string" ? ` · ${index.status}` : ""}。实际召回与回退会在结果中披露。`;
    $("retrieval-mode").disabled = state.searching || state.generating;
    $("api-model").disabled = !apiAvailable() || state.generating;
    if (!apiAvailable()) $("api-model").checked = false;
    $("api-model-label").textContent = apiAvailable() ? "API 依据证据改写" : "API 待配置";
    $("api-model-description").textContent = apiAvailable() ? "可选启用；与本地模型互斥" : "密钥仅在服务端配置";
  }
  function renderRetrieval(result, context = "search") {
    const prefix = context === "generation" ? "generation-" : "";
    $(prefix + "retrieval-summary").hidden = false;
    const retrieval = result.retrieval || {}, badges = [], index = retrieval.index || {};
    badges.push(el("span", "badge neutral", `请求：${retrievalName(retrieval.mode || state.lastSearch?.retrieval_mode)}`), el("span", "badge neutral", `实际：${retrievalName(retrieval.effective_mode)}`));
    const statusLabel = retrieval.effective_mode === "selected_recipes" && retrieval.status === "ok" ? "选中证据读取完成" : { ok: "召回完成", fallback: "发生检索回退", unavailable: "检索不可用" }[retrieval.status] || text(retrieval.status, "检索状态未提供");
    badges.push(el("span", retrieval.status === "ok" ? "badge" : "badge warning", statusLabel));
    if (retrieval.insufficient_candidates === true) badges.push(el("span", "badge warning", "本轮候选不足"));
    $(prefix + "retrieval-badges").replaceChildren(...badges);
    const counts = []; if (count(retrieval.returned_candidates) !== null) counts.push(`本轮返回候选 ${number(retrieval.returned_candidates)} 条`); if (count(retrieval.eligible_candidates) !== null) counts.push(`符合已记录提及条件 ${number(retrieval.eligible_candidates)} 条`); if (count(retrieval.filtered_candidates) !== null) counts.push(`过滤 ${number(retrieval.filtered_candidates)} 条`);
    const fallbackMessage = text(retrieval.fallback_message);
    $(prefix + "retrieval-description").textContent = retrieval.effective_mode === "selected_recipes" ? "按选中的原始 Recipe ID 读取图谱证据，没有再次执行语义或关键词召回。" : `${counts.join("，") || "服务未提供本轮候选数量"}。候选数量不代表全库匹配数。${fallbackMessage ? ` ${fallbackMessage}` : retrieval.status === "fallback" ? " 当前使用可用的检索方式，详细原因见下方。" : ""}`;
    const lines = [`检索状态：${text(retrieval.status, "未提供")}`, `检索方法：${text(retrieval.method, "未提供")}`, `候选上限：${number(retrieval.candidate_limit)}`, `检索覆盖：${text(retrieval.coverage, "未提供")}`, `无结果范围：${text(retrieval.no_match_scope, "未提供")}`, `索引：${typeof retrieval.index === "string" ? retrieval.index : text(index.name, "未提供")} · ${text(index.status, "状态未提供")}`];
    if (retrieval.fallback_reason) lines.push(`回退原因：${text(retrieval.fallback_reason, "未提供")}`);
    $(prefix + "retrieval-disclosure").textContent = lines.join("\n");
    const trace = array(result.trace).length ? array(result.trace) : array(retrieval.trace);
    $(prefix + "retrieval-trace").replaceChildren(...trace.map((item) => { const row = el("li", "", `${text(item.stage, "阶段未提供")} · ${text(item.status, "状态未提供")}`); if (count(item.elapsed_ms) !== null) row.append(el("span", "", `${item.elapsed_ms.toFixed(0)} ms`)); return row; }));
    if (!trace.length) $(prefix + "retrieval-trace").append(el("li", "", "服务未提供逐阶段检索记录。"));
  }
  async function loadStatus() {
    $("refresh-status").disabled = true;
    try {
      const info = await request("/api/system/status"); state.status = info; const dataset = info.dataset || {}, graph = info.graph || {};
      $("service-dot").className = "ready"; $("service-label").textContent = info.backend === "neo4j" ? "真实图谱已连接" : "知识库已连接";
      $("dataset-label").textContent = `${text(dataset.name, "食谱知识库")} · ${text(dataset.split, "范围未提供")}`;
      [["stat-recipes", graph.recipes], ["stat-steps", graph.steps], ["stat-ingredients", graph.ingredient_entities], ["stat-images", graph.images], ["stat-recognized", graph.distinct_recognized_images]].forEach(([id, v]) => { $(id).textContent = number(v); });
      $("source-note").textContent = `${dataset.is_demo === true ? "示例数据" : "真实来源记录"} · ${text(dataset.name, "名称未提供")}\n${text(dataset.split, "范围未提供")} · revision ${text(dataset.revision, "未提供")}`;
      $("footer-status").textContent = `${number(graph.recipes)} 条食谱 · ${number(graph.image_observations)} 条图片观察候选 · ${number(graph.videos)} 个原始视频`;
      $("local-model").disabled = !modelAvailable() || state.generating; if (!modelAvailable()) $("local-model").checked = false;
      $("model-description").textContent = modelAvailable() ? "可选启用；实际调用与回退会披露" : "当前不可用，默认按原文组织"; updateCapabilities();
      const warnings = []; if (dataset.is_demo === true) warnings.push("当前连接的是示例数据，不能当作真实 RecipeGen 数据集成果。"); if (info.backend !== "neo4j") warnings.push("当前服务后端与真实 Neo4j 模式不同，请核对服务配置。");
      $("system-notice").textContent = warnings.join(" "); $("system-notice").hidden = warnings.length === 0;
    } catch (error) { state.status = null; $("service-dot").className = "error"; $("service-label").textContent = "连接未确认"; $("local-model").disabled = true; $("local-model").checked = false; $("model-description").textContent = "未确认模型可用，保持原文组织"; $("system-notice").textContent = error.message; $("system-notice").hidden = false; updateCapabilities(); }
    finally { $("refresh-status").disabled = false; }
  }
  function searchPayload() { return { query: $("search-question").value.trim(), ingredients: split($("ingredients").value), excluded_ingredients: split($("excluded-ingredients").value), limit: 6, retrieval_mode: retrievalMode() }; }
  function setSearchBusy(busy) { state.searching = busy; $("search-button").disabled = busy || state.generating; $("search-button-label").textContent = busy ? "正在检索真实食谱" : "检索真实食谱"; $("search-button").querySelector(".spinner").hidden = !busy; $("search-button").querySelector(".button-arrow").hidden = busy; $("search-loading").hidden = !busy; $("search-results").setAttribute("aria-busy", String(busy)); updateSelection(); renderCards(); updateCapabilities(); }
  async function search(event) {
    event?.preventDefault(); if (state.searching || state.generating) return; const payload = searchPayload();
    if (!payload.query && !payload.ingredients.length) { showError("search-error", "请输入菜名、问题或希望用到的食材。"); $("search-question").focus(); return; }
    $("search-error").hidden = true; $("empty-state").hidden = true; setSearchBusy(true);
    try {
      const result = await post("/api/search", payload, 120000); state.recipes = array(result.recipes).filter((r) => typeof r.id === "string" && r.id); state.lastSearch = payload; state.selected.clear(); invalidateAnswer(); updateSelection(); renderCards(); renderRetrieval(result);
      $("result-count").textContent = `找到 ${state.recipes.length} 道 · 可选 1–3 道`; $("results-description").textContent = `检索「${text(result.query, payload.query) || payload.ingredients.join("、")}」的真实记录。食材仅为原文提及，配方可能不完整。`;
      $("search-limitations").textContent = array(result.limitations).filter((v) => typeof v === "string").join(" "); $("search-limitations").hidden = !$("search-limitations").textContent; $("empty-state").hidden = state.recipes.length > 0;
      if (!state.recipes.length) { $("empty-state").querySelector("h3").textContent = "本轮没有返回匹配的食谱。"; $("empty-state").querySelector("p:not(.eyebrow)").textContent = "试试具体菜名或单个食材。本轮候选有限，食材记录也可能不完整，不代表全库没有相关做法。"; }
      if (window.matchMedia("(max-width: 760px)").matches) $("search-results").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) { showError("search-error", `${error.message}${state.recipes.length ? " 下方保留上次检索结果。" : ""}`); $("empty-state").hidden = state.recipes.length > 0; }
    finally { setSearchBusy(false); }
  }
  function mentionName(item) { return text(item.zh_name || item.name, "未命名提及"); }
  function mediaText(recipe) { const media = recipe.media || {}; return [`图片 ${number(media.images)}`, `视频 ${number(media.videos)}`, `已识别图片 ${number(media.recognized_images)}`]; }
  function recipeMeta(recipe) { return `${count(recipe.minutes) !== null ? `记录用时 ${recipe.minutes} 分钟` : "用时未记录"} · ${text(recipe.servings) ? `记录分量 ${text(recipe.servings)}` : "分量未记录"}`; }
  function invalidateAnswer() { $("answer-wrap").hidden = true; $("generation-error").hidden = true; }
  function renderCards() {
    const focusedRecipe = document.activeElement?.classList.contains("select-button") ? document.activeElement.dataset.recipeId : null;
    $("recipe-grid").replaceChildren(...state.recipes.map((recipe, index) => {
      const card = el("article", `recipe-card panel${state.selected.has(recipe.id) ? " selected" : ""}`), top = el("div", "card-topline"); top.append(el("span", "card-index", `RECIPE ${String(index + 1).padStart(2, "0")}`), el("span", "card-source", "图谱真实记录"));
      card.append(top, el("h3", "", text(recipe.title, "标题未提供")), el("p", "recipe-meta", recipeMeta(recipe)), el("p", "mention-label", "食材提及 · 配方可能不完整"));
      const chips = el("div", "ingredient-chips"), mentions = array(recipe.ingredient_mentions); chips.append(...mentions.slice(0, 6).map((v) => el("span", "ingredient-chip", mentionName(v)))); if (!mentions.length) chips.append(el("span", "field-help", "未单独抽取到食材提及")); if (mentions.length > 6) chips.append(el("span", "ingredient-chip", `+${mentions.length - 6}`));
      const media = el("div", "card-media"); media.append(...mediaText(recipe).map((v) => el("span", "", v)));
      const actions = el("div", "card-actions"), view = el("button", "card-view-button", "查看步骤与来源 ↗"), select = el("button", "select-button", state.selected.has(recipe.id) ? "✓ 已选择" : "+ 选择食谱"); view.type = "button"; view.dataset.recipeId = recipe.id; view.addEventListener("click", () => openDetail(recipe, view)); select.type = "button"; select.dataset.recipeId = recipe.id;
      select.setAttribute("aria-pressed", String(state.selected.has(recipe.id))); select.setAttribute("aria-label", `${state.selected.has(recipe.id) ? "取消选择" : "选择"} ${text(recipe.title, "食谱")}`); select.disabled = state.generating || state.searching; select.addEventListener("click", () => toggleSelection(recipe)); actions.append(view, select); card.append(chips, media, actions); return card;
    }));
    if (focusedRecipe) [...document.querySelectorAll(".select-button")].find((node) => node.dataset.recipeId === focusedRecipe)?.focus();
  }
  function toggleSelection(recipe) { if (state.generating || state.searching) return; if (state.selected.has(recipe.id)) state.selected.delete(recipe.id); else { if (state.selected.size >= 3) { toast("最多选择 3 道食谱，请先移除一道。"); return; } state.selected.set(recipe.id, recipe); } invalidateAnswer(); updateSelection(); renderCards(); }
  function updateSelection() {
    const values = [...state.selected.values()]; $("selection-label").textContent = values.length ? `已选择 ${values.length} / 3 道食谱` : "还没有选择食谱"; $("clear-selection").hidden = values.length === 0; $("clear-selection").disabled = state.generating || state.searching;
    $("selected-recipes").replaceChildren(...values.map((recipe) => { const chip = el("div", "selected-chip"), remove = el("button", "", "×"); remove.type = "button"; remove.disabled = state.generating || state.searching; remove.setAttribute("aria-label", `移除 ${text(recipe.title, "食谱")}`); remove.addEventListener("click", () => toggleSelection(recipe)); chip.append(el("span", "", text(recipe.title, "标题未提供")), remove); return chip; }));
    if (!values.length) $("selected-recipes").append(el("span", "selection-empty", "在上方食谱卡片中，选择你想做的 1–3 道菜。")); $("generate-button").disabled = state.generating || state.searching || values.length === 0;
    if (state.detail) { $("detail-select").disabled = state.generating || state.searching; $("detail-select").textContent = state.selected.has(state.detail.id) ? "取消选择这道食谱" : "选择这道食谱"; }
  }
  function setGenerationBusy(busy) { state.generating = busy; $("generate-button-label").textContent = busy ? "正在整理做法与证据" : "生成做法回答"; $("generate-button").querySelector(".spinner").hidden = !busy; $("generate-button").querySelector(".button-arrow").hidden = busy; $("local-model").disabled = busy || !modelAvailable(); $("search-button").disabled = busy || state.searching; $("generation-question").disabled = busy; $("answer-wrap").setAttribute("aria-busy", String(busy)); updateSelection(); renderCards(); updateCapabilities(); }
  async function generate(event) {
    event.preventDefault(); if (state.generating || state.searching || !state.selected.size) return;
    const payload = { question: $("generation-question").value.trim() || state.lastSearch?.query || "按顺序整理已选食谱的原始做法，并列出每一步的证据来源。", recipe_ids: [...state.selected.keys()], ingredients: split($("ingredients").value), excluded_ingredients: split($("excluded-ingredients").value), mode: apiAvailable() && $("api-model").checked ? "api" : modelAvailable() && $("local-model").checked ? "local" : "grounded", retrieval_mode: retrievalMode(), limit: 3 };
    $("generation-error").hidden = true; setGenerationBusy(true);
    try { renderAnswer(await post("/api/generate", payload, 180000)); } catch (error) { showError("generation-error", `${error.message}${!$("answer-wrap").hidden ? " 仍保留上次生成的回答。" : ""}`); } finally { setGenerationBusy(false); }
  }
  function renderAnswer(result) {
    if (result.retrieval && typeof result.retrieval === "object") renderRetrieval(result, "generation"); else $("generation-retrieval-summary").hidden = true;
    $("answer-wrap").hidden = false; $("answer-text").textContent = text(result.answer, result.status === "no_match" ? "没有找到可用于当前问题的食谱证据，请调整问题或重新选择。" : "服务未提供做法回答。"); const generation = result.generation || {}, badges = [];
    badges.push(el("span", "badge neutral", ["api_evidence_rewrite", "graphrag_api_grounded_paraphrase"].includes(generation.effective_mode) ? "API 依据证据改写" : generation.effective_mode === "local_selection_grounded_answer" ? "本地模型筛选 · 原文组织" : generation.effective_mode === "grounded" ? "依据原文组织" : "实际模式未提供"), el("span", "badge neutral", modelCallStatus(generation))); if (generation.pending_api === true || generation.api_status === "not_configured") badges.push(el("span", "badge warning", "API 待配置 · 按原文组织")); if (generation.fallback) badges.push(el("span", "badge warning", "发生回退")); if (result.validation?.passed === true) badges.push(el("span", "badge", "程序校验通过")); else if (result.validation?.passed === false) badges.push(el("span", "badge warning", "程序校验未通过")); if (result.status === "no_match") badges.push(el("span", "badge warning", "没有匹配证据")); $("generation-badges").replaceChildren(...badges);
    const evidence = array(result.evidence); $("evidence-count").textContent = `${evidence.length} 条`; $("evidence-list").replaceChildren(...evidence.map((item, index) => { const node = el("article", "evidence-item"); node.append(el("h4", "", `${text(item.id, `证据 ${index + 1}`)} · ${text(item.kind, "原始记录")}`), el("p", "", text(item.text, "原文未提供"))); const sourceIDs = [...new Set([item.source_id, ...array(item.source_ids)].filter((value) => typeof value === "string" && value))]; if (sourceIDs.length) node.append(el("p", "step-source mono", `来源 ID：${sourceIDs.join("\n")}`)); else node.append(el("p", "step-source mono", `图谱记录：${text(item.graph_id || item.recipe_id, "未提供")} · 未提供唯一原始来源`)); const sourceURLs = [...new Set([item.source_url, ...array(item.source_urls)].filter(safeURL))]; if (sourceURLs.length) sourceURLs.forEach((url) => node.append(link(url))); else node.append(link(null)); return node; })); if (!evidence.length) $("evidence-list").append(el("p", "field-help", "当前回答未提供可展示的证据条目。"));
    const checks = array(result.validation?.checks); $("validation-list").replaceChildren(...checks.map((item) => el("span", item.passed === true ? "badge" : "badge warning", `${item.passed === true ? "✓" : item.passed === false ? "!" : "?"} ${text(item.name, "未命名校验")}`))); if (!checks.length) $("validation-list").append(el("span", "field-help", "未提供逐项校验结论。"));
    $("trace-list").replaceChildren(...array(result.trace).map((item) => { const li = el("li", "", `${text(item.stage, "步骤")} · ${text(item.status, "状态未提供")}`); li.append(el("span", "", count(item.elapsed_ms) === null ? "" : `${item.elapsed_ms.toFixed(0)} ms`)); return li; }));
    const details = [`请求模式：${text(generation.mode, "未提供")}`, `实际模式：${text(generation.effective_mode, "未提供")}`, `模型：${generation.model ? modelName(generation.model) : "未提供 / 未使用"}`, `API 状态：${text(generation.api_status, "未提供")}`, `模型调用：${modelCallStatus(generation)}`, `回退：${generation.fallback ? text(generation.fallback, "是") : "否"}`], errors = array(generation.errors).map((v) => typeof v === "string" ? v : text(v?.message || v?.error || v?.type, "模型阶段发生错误")); if (errors.length) details.push(`阶段错误：${errors.join("；")}`); $("generation-disclosure").textContent = details.join("\n");
  }
  async function openDetail(recipe, trigger) {
    state.detailController?.abort(); state.graphController?.abort(); const controller = new AbortController(), version = ++state.detailVersion; state.detailController = controller; state.detail = null; state.detailTrigger = trigger;
    $("detail-title").textContent = text(recipe.title, "食谱详情"); $("detail-loading").hidden = false; $("detail-error").hidden = true; $("detail-body").hidden = true; $("graph-section").hidden = true; $("detail-select").disabled = true; if (!$("recipe-dialog").open) $("recipe-dialog").showModal();
    try { const data = await request(`/api/recipes/${encodeURIComponent(recipe.id)}`, { signal: controller.signal }); if (version !== state.detailVersion) return; const detail = data.recipe || data; if (typeof detail.id !== "string" || detail.id !== recipe.id) throw new Error("食谱详情身份与所选记录不一致，请重试。"); state.detail = detail; $("detail-title").textContent = text(detail.title, "标题未提供"); renderDetail(detail); updateSelection(); loadGraph(detail.id, version); }
    catch (error) { if (error.name !== "AbortError" && version === state.detailVersion) showError("detail-error", error.message); } finally { if (version === state.detailVersion) $("detail-loading").hidden = true; }
  }
  function section(title) { const node = el("section", "detail-section"); node.append(el("h3", "", title)); return node; }
  function renderDetail(recipe) {
    const body = $("detail-body"); body.replaceChildren(); body.hidden = false;
    const mentions = section("食材提及"), chips = el("div", "ingredient-chips"); chips.append(...array(recipe.ingredient_mentions).map((v) => el("span", "ingredient-chip", mentionName(v)))); if (!chips.childNodes.length) chips.append(el("span", "field-help", "原文中未抽取到可展示的食材提及。")); mentions.append(chips, el("p", "detail-warning", "这些是文本提及候选，不能视为完整配方或已验证的库存清单。"), el("p", "recipe-meta", recipeMeta(recipe)));
    const steps = section("有序原始步骤"), ordered = el("ol", "original-steps"); array(recipe.steps).forEach((step, index) => { const row = el("li", "original-step"), content = el("div"); content.append(el("p", "", text(step.text, "原文缺失"))); if (step.source_id) content.append(el("div", "step-source mono", `来源：${text(step.source_id)}`)); row.append(el("span", "", count(step.order) === null ? index + 1 : step.order), content); ordered.append(row); }); if (!ordered.childNodes.length) steps.append(el("p", "detail-warning", "这条记录缺少原始步骤，系统不会补写。")); else steps.append(ordered);
    const sources = section("原始来源"), sourceList = el("div", "source-list"); array(recipe.sources).forEach((source) => { const row = el("div", "source-record"); row.append(el("strong", "", text(source.role, "原始记录")), el("span", "mono", `来源 ID：${text(source.source_id, "未提供")}`)); if (source.artifact_path) row.append(el("span", "mono", `归档成员：${text(source.artifact_path)}`)); row.append(link(source.url)); sourceList.append(row); }); if (!sourceList.childNodes.length) sourceList.append(el("p", "detail-warning", "当前接口未提供来源条目，请勿将缺失的来源视为已核验。")); sources.append(sourceList);
    const visual = section("图像描述候选"), observationList = el("div", "visual-observations"), observations = array(recipe.visual_evidence || recipe.image_observations || recipe.visual_observations || recipe.media?.observations);
    observations.slice(0, 8).forEach((item) => { const node = el("article", "observation-card"), mid = text(item.source_media_id), isVideo = mid.startsWith("video:"); node.append(el("span", "badge neutral", isVideo ? "视频观察候选 · 待核对" : "图像观察候选 · 待核对"), el("p", "", text(item.caption, "未提供可展示的描述")), el("p", "observation-meta", `观察：${text(item.observation_id || item.id, "未提供")}\n媒体：${mid || "未提供"}\n模型：${modelName(item.model)}`)); if (item.source_url) node.append(link(item.source_url)); observationList.append(node); }); if (!observationList.childNodes.length) visual.append(el("p", "detail-warning", "当前食谱没有可展示的图片识别候选。媒体节点存在，不等于已经识别。")); else visual.append(observationList, el("p", "detail-warning", `展示 ${Math.min(8, observations.length)} / ${observations.length} 条观察。描述来自模型，未经逐条语义确认，不证明食材、动作或精确步骤对应。`)); body.append(mentions, steps, sources, visual);
  }
  async function loadGraph(recipeId, version) {
    state.graphController?.abort(); const controller = new AbortController(); state.graphController = controller; $("graph-section").hidden = false; $("graph-status").textContent = "正在读取图谱关联…"; $("graph-canvas").replaceChildren(); $("graph-node-detail").textContent = "选择一个节点，查看它的名称与关联。";
    try { const graph = await request(`/api/system/graph?recipe_id=${encodeURIComponent(recipeId)}`, { signal: controller.signal }); if (version !== state.detailVersion) return; renderGraph(graph); } catch (error) { if (error.name !== "AbortError" && version === state.detailVersion) $("graph-status").textContent = `关联图读取未完成：${error.message}`; }
  }
  function renderGraph(graph) {
    const allNodes = array(graph.nodes).filter((v) => typeof v.id === "string"), nodes = [...allNodes].sort((a, b) => Number(b.label === "Recipe") - Number(a.label === "Recipe")).slice(0, 36), lookup = new Map(), edges = array(graph.edges).filter((v) => typeof v.source === "string" && typeof v.target === "string"); if (!nodes.length) { $("graph-status").textContent = "当前食谱没有返回可展示的知识关联。"; return; }
    const center = nodes.findIndex((n) => n.label === "Recipe"); if (center > 0) nodes.unshift(nodes.splice(center, 1)[0]); const colors = { Recipe: "#bd5a34", Ingredient: "#839667", Step: "#d2af73", Source: "#92a6af", Image: "#92a6af", Video: "#92a6af", VisualObservation: "#bc9cbd", VisualObject: "#bc9cbd" };
    nodes.forEach((node, index) => { let x = 360, y = 160; if (index > 0) { const outer = index > 12, within = outer ? index - 13 : index - 1, amount = outer ? Math.max(1, nodes.length - 13) : Math.min(12, nodes.length - 1), angle = within / amount * Math.PI * 2 - Math.PI / 2; x += Math.cos(angle) * (outer ? 274 : 161); y += Math.sin(angle) * (outer ? 123 : 88); } lookup.set(node.id, { ...node, x, y }); });
    const svg = svgEl("svg", { viewBox: "0 0 720 335", role: "group", "aria-label": "食谱知识关联图，使用 Tab 选择节点并按 Enter 查看详情" }), lineGroup = svgEl("g"), nodeGroup = svgEl("g"), edgeElements = [], nodeElements = new Map();
    function choose(node) { const connected = edges.filter((e) => e.source === node.id || e.target === node.id); nodeElements.forEach((element, id) => element.classList.toggle("active", id === node.id)); edgeElements.forEach(({ element, edge }) => element.classList.toggle("active", edge.source === node.id || edge.target === node.id)); const relations = connected.slice(0, 8).map((e) => { const other = lookup.get(e.source === node.id ? e.target : e.source); return `${text(e.type, "关联")} → ${text(other?.name, "未展示节点")}`; }); $("graph-node-detail").textContent = `${text(node.name, "名称未提供")}\n类型：${text(node.label, "未提供")} · ID：${node.id}${relations.length ? `\n${relations.join("\n")}` : "\n当前视图未展示关联边。"}`; }
    edges.forEach((edge) => { const from = lookup.get(edge.source), to = lookup.get(edge.target); if (!from || !to) return; const line = svgEl("line", { x1: from.x, y1: from.y, x2: to.x, y2: to.y, class: "graph-edge" }); line.append(svgEl("title", {}, text(edge.type, "关联"))); edgeElements.push({ element: line, edge }); lineGroup.append(line); });
    lookup.forEach((node, id) => { const group = svgEl("g", { class: "graph-node", transform: `translate(${node.x},${node.y})`, role: "button", tabindex: "0", "aria-label": `${text(node.label, "节点")}：${text(node.name, id)}` }), name = text(node.name, node.id); group.append(svgEl("circle", { r: node.label === "Recipe" ? 15 : 9, fill: colors[node.label] || "#a3aca0" }), svgEl("text", { x: 0, y: node.label === "Recipe" ? 31 : 23, "text-anchor": "middle" }, name.length > 17 ? `${name.slice(0, 16)}…` : name), svgEl("title", {}, `${text(node.label)} · ${name}`)); group.addEventListener("click", () => choose(node)); group.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(node); } }); nodeElements.set(id, group); nodeGroup.append(group); }); svg.append(lineGroup, nodeGroup); $("graph-canvas").replaceChildren(svg); $("graph-status").textContent = `显示 ${nodes.length} / ${allNodes.length} 个节点。关系表示图谱关联；模型关系仍是候选。`;
  }
  $("search-form").addEventListener("submit", search); $("generate-form").addEventListener("submit", generate); document.querySelectorAll(".preset").forEach((button) => button.addEventListener("click", () => { if (state.searching || state.generating) return; $("search-question").value = button.dataset.query; $("ingredients").value = button.dataset.ingredients || ""; $("search-question").focus(); }));
  $("clear-selection").addEventListener("click", () => { if (!state.generating && !state.searching) { state.selected.clear(); invalidateAnswer(); updateSelection(); renderCards(); } }); $("detail-select").addEventListener("click", () => { if (state.detail) { const selecting = !state.selected.has(state.detail.id); toggleSelection(state.detail); if (selecting && state.selected.has(state.detail.id)) toast("已加入食谱清单，可以继续选择或整理做法。"); } });
  $("close-detail").addEventListener("click", () => $("recipe-dialog").close()); $("recipe-dialog").addEventListener("close", () => { state.detailController?.abort(); state.graphController?.abort(); state.detailVersion++; state.detail = null; if (state.detailTrigger?.isConnected) state.detailTrigger.focus(); else $("search-question").focus(); }); $("recipe-dialog").addEventListener("click", (event) => { if (event.target === $("recipe-dialog")) { const rect = $("recipe-dialog").getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) $("recipe-dialog").close(); } });
  $("retrieval-mode").addEventListener("change", () => { state.retrievalModeTouched = true; invalidateAnswer(); });
  $("local-model").addEventListener("change", () => { if ($("local-model").checked) $("api-model").checked = false; invalidateAnswer(); });
  $("api-model").addEventListener("change", () => { if ($("api-model").checked) $("local-model").checked = false; invalidateAnswer(); });
  $("refresh-status").addEventListener("click", loadStatus); updateSelection(); loadStatus();
})();
