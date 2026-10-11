"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const page = document.body.dataset.page;
  const WORKSPACE_KEY = "recipegen-explore-v1";
  const state = { status: null, recipes: [], searching: false, lastSearch: null, lastResult: null, draft: null, detail: null, searchController: null, detailController: null, graphController: null, searchVersion: 0, detailVersion: 0, graphLoaded: false };
  const SVG_NS = "http://www.w3.org/2000/svg";
  const array = (value) => Array.isArray(value) ? value : [];
  const text = (value, fallback = "") => typeof value === "string" ? value : typeof value === "number" ? String(value) : fallback;
  const count = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
  const number = (value) => count(value) === null ? "未提供" : value.toLocaleString("zh-CN");
  function el(tag, className, value) { const node = document.createElement(tag); if (className) node.className = className; if (value !== undefined) node.textContent = String(value); return node; }
  function svgEl(tag, attributes = {}, value) { const node = document.createElementNS(SVG_NS, tag); Object.entries(attributes).forEach(([key, val]) => node.setAttribute(key, String(val))); if (value !== undefined) node.textContent = String(value); return node; }
  function split(value) { return [...new Set(value.split(/[,，、;；\n]+/).map((v) => v.trim()).filter(Boolean))]; }
  function safeURL(value) { if (typeof value !== "string") return null; try { const url = new URL(value); return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password ? url.href : null; } catch (_) { return null; } }
  function link(value, label = "查看原始来源 ↗") { const url = safeURL(value); if (!url) return el("span", "field-help", "未提供可访问的网页来源"); const node = el("a", "source-link", label); node.href = url; node.target = "_blank"; node.rel = "noopener noreferrer"; return node; }
  function errorText(data, status) { const detail = data?.detail || data?.message || data?.error?.message || data?.error; if (typeof detail === "string") return detail; if (Array.isArray(detail)) return detail.map((v) => text(v.msg, "请求参数不符合接口要求")).join("；"); return `请求未完成（HTTP ${status}），请稍后重试。`; }
  async function request(path, options = {}, timeout = 60000) {
    const controller = new AbortController(); const abort = () => controller.abort(); let timedOut = false;
    if (options.signal?.aborted) controller.abort(); options.signal?.addEventListener("abort", abort, { once: true });
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeout);
    try { const response = await fetch(path, { ...options, signal: controller.signal }); let data; try { data = await response.json(); } catch (_) { throw new Error(`无法读取服务响应（HTTP ${response.status}）。`); } if (!response.ok) throw new Error(errorText(data, response.status)); if (!data || typeof data !== "object") throw new Error("服务返回了不完整的数据，请重试。"); return data; }
    catch (error) { if (timedOut) throw new Error("服务响应超时，当前操作尚未确认完成。请稍后重试。"); if (error.name === "AbortError") throw error; if (error instanceof TypeError) throw new Error("无法连接服务，请确认 RecipeGen 已启动后重试。"); throw error; }
    finally { clearTimeout(timer); options.signal?.removeEventListener("abort", abort); }
  }
  function post(path, payload, timeout, signal) { return request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal }, timeout); }
  function showError(id, message) { const node = $(id); if (node) { node.textContent = message; node.hidden = false; } }
  const setText = (id, value) => { if ($(id)) $(id).textContent = value; };
  function modelName(model) { return typeof model === "string" ? model : text(model?.name || model?.model || model?.repo, "名称未提供"); }
  function retrievalName(mode) { return { keyword: "关键词检索", semantic: "食材筛选 → E5 排序", hybrid: "三路混合检索" }[mode] || text(mode, "模式未提供"); }
  function updateCapabilities() {
    const retrieval = state.status?.retrieval || {}, index = retrieval.index || {};
    setText("retrieval-capability", `${retrieval.semantic_available === true ? "食材筛选与 E5 排序可用" : "语义排序当前不可用"}${typeof index.count === "number" ? `，向量记录 ${number(index.count)} 条` : ""}。`);
  }
  function saveWorkspace() {
    try { sessionStorage.setItem(WORKSPACE_KEY, JSON.stringify({ version: 1, recipes: state.recipes, lastSearch: state.lastSearch, lastResult: state.lastResult, draft: state.draft })); } catch (_) {}
  }
  function restoreWorkspace() {
    try {
      const saved = JSON.parse(sessionStorage.getItem(WORKSPACE_KEY) || "null");
      if (saved?.version !== 1) return;
      state.recipes = array(saved.recipes).filter(recipe => typeof recipe?.id === "string");
      state.lastSearch = saved.lastSearch; state.lastResult = saved.lastResult; state.draft = saved.draft;
    } catch (_) {}
  }
  function searchPayload() {
    const ingredients = split($("ingredients").value), excluded = split($("excluded-ingredients").value);
    const query = [ingredients.length ? `使用${ingredients.join("、")}的食谱` : "推荐食谱", excluded.length ? `排除${excluded.join("、")}` : ""].filter(Boolean).join("，");
    return { query, ingredients, excluded_ingredients: excluded, limit: 12, retrieval_mode: "hybrid" };
  }
  function saveSearchDraft() { state.draft = { ingredients: $("ingredients").value, excluded_ingredients: $("excluded-ingredients").value }; saveWorkspace(); }
  function restoreSearchView() {
    const draft = state.draft || state.lastSearch || {};
    $("ingredients").value = Array.isArray(draft.ingredients) ? draft.ingredients.join("、") : text(draft.ingredients);
    $("excluded-ingredients").value = Array.isArray(draft.excluded_ingredients) ? draft.excluded_ingredients.join("、") : text(draft.excluded_ingredients);
    $("empty-state").hidden = !state.lastResult || state.recipes.length > 0; $("search-error").hidden = true;
    if ($("results-column")) $("results-column").hidden = !state.lastResult;
    if ($("results-heading")) $("results-heading").hidden = !state.lastResult;
    renderCards(); if (state.lastResult) renderSearchResult(state.lastResult);
  }
  function renderSearchResult(result) {
    setText("result-count", `${state.recipes.length} 道食谱`);
    if ($("results-column")) $("results-column").hidden = false;
    if ($("results-heading")) $("results-heading").hidden = false;
    renderRetrieval(result);
    $("search-limitations").textContent = array(result.limitations).filter(value => typeof value === "string").join(" ");
    $("search-limitations").hidden = !$("search-limitations").textContent;
    $("empty-state").hidden = state.recipes.length > 0;
    if (!state.recipes.length) {
      $("empty-state").querySelector("h3").textContent = "没有找到符合条件的食谱";
      $("empty-state").querySelector("p").textContent = "试着减少食材条件。";
    }
  }
  function setSearchBusy(busy) {
    state.searching = busy; $("search-button").disabled = busy;
    setText("search-button-label", busy ? "正在查找" : "查找食谱");
    $("search-button").querySelector(".spinner").hidden = !busy;
    $("search-button").querySelector(".button-arrow").hidden = busy;
    $("search-loading").hidden = !busy; $("search-results").setAttribute("aria-busy", String(busy));
    if ($("results-column")) $("results-column").hidden = !busy && !state.lastResult;
    if ($("results-heading")) $("results-heading").hidden = busy || !state.lastResult;
    ["ingredients", "excluded-ingredients"].forEach(id => $(id).disabled = busy);
  }
  async function search(event) {
    event.preventDefault(); if (state.searching) return;
    const payload = searchPayload();
    if (!payload.ingredients.length && !payload.excluded_ingredients.length) {
      showError("search-error", "请输入希望用到或不想使用的食材。"); $("ingredients").setAttribute("aria-invalid", "true"); $("ingredients").focus(); return;
    }
    if ([payload.ingredients, payload.excluded_ingredients].some(items => items.length > 20 || items.some(item => item.length > 80 || /[\x00-\x1f]/.test(item)))) {
      showError("search-error", "每项条件最多 20 种食材，单个食材名称最多 80 字。请精简后再查找。"); return;
    }
    if (payload.query.length > 500) { showError("search-error", "食材条件过长，请精简后再查找。"); return; }
    $("ingredients").removeAttribute("aria-invalid"); $("search-error").hidden = true; $("empty-state").hidden = true;
    setSearchBusy(true); saveSearchDraft();
    const controller = new AbortController(), version = ++state.searchVersion; state.searchController = controller;
    try {
      const result = await post("/api/search", payload, 120000, controller.signal);
      if (controller.signal.aborted || version !== state.searchVersion) return;
      state.recipes = array(result.recipes).filter(recipe => typeof recipe?.id === "string" && recipe.id);
      state.lastSearch = payload; state.lastResult = result;
      renderCards(); renderSearchResult(result); saveWorkspace();
      if (window.matchMedia("(max-width: 767px)").matches) $("search-results").scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" });
    } catch (error) {
      if (error.name === "AbortError" || version !== state.searchVersion) return;
      showError("search-error", `${error.message}${state.recipes.length ? " 已保留上次结果。" : ""}`); $("empty-state").hidden = !state.lastResult || state.recipes.length > 0;
    } finally { if (version === state.searchVersion) setSearchBusy(false); }
  }
  function renderCards() {
    if (!$("recipe-grid")) return;
    $("recipe-grid").replaceChildren(...state.recipes.map(recipe => {
      const card = el("article", "recipe-card panel"); card.append(el("h3", "", text(recipe.title, "标题未提供")));
      const meta = recipeMeta(recipe); if (meta) card.append(el("p", "recipe-meta", meta));
      const chips = el("div", "ingredient-chips"), mentions = array(recipe.ingredient_mentions);
      chips.append(...mentions.slice(0, 6).map(item => el("span", "ingredient-chip", mentionName(item))));
      if (mentions.length > 6) chips.append(el("span", "ingredient-chip", `+${mentions.length - 6}`));
      const actions = el("div", "card-actions"), view = el("a", "card-view-button", "查看做法 ↗");
      view.href = `/recipes/${encodeURIComponent(recipe.id)}`; actions.append(view); card.append(chips, actions); return card;
    }));
  }
  async function loadStatus() {
    try {
      const info = await request("/api/system/status"); state.status = info;
      const dataset = info.dataset || {}, graph = info.graph || {};
      setText("service-label", info.backend === "neo4j" ? "知识库已连接" : "示例服务已连接");
      setText("dataset-label", `${text(dataset.name, "食谱知识库")} · ${text(dataset.split, "范围未提供")}`);
      [["stat-recipes", graph.recipes], ["stat-steps", graph.steps], ["stat-ingredients", graph.ingredient_entities], ["stat-images", graph.images], ["stat-recognized", graph.distinct_recognized_images]].forEach(([id, value]) => setText(id, number(value)));
      setText("source-note", `${dataset.is_demo === true ? "示例数据" : "来源记录"}：${text(dataset.name)}\n范围：${text(dataset.split, "未提供")}\n版本：${text(dataset.revision, "未提供")}`);
      updateCapabilities();
      if (dataset.is_demo === true || info.backend !== "neo4j") showError("system-notice", "当前连接示例服务，请以数据说明页标注的范围为准。");
      else if ($("system-notice")) $("system-notice").hidden = true;
    } catch (error) {
      state.status = null; updateCapabilities(); setText("service-label", "知识库连接未确认");
      showError("system-notice", error.message);
    }
  }
  function mentionName(item) { return text(item.zh_name || item.name, "未命名提及"); }
  function recipeMeta(recipe) {
    const values = [];
    if (count(recipe.minutes) !== null) values.push(`${recipe.minutes} 分钟`);
    if (text(recipe.servings)) values.push(`分量：${text(recipe.servings)}`);
    return values.join(" · ");
  }
  function renderRetrieval(result, context = "search") {
    const prefix = context === "generation" ? "generation-" : "";
    $(prefix + "retrieval-summary").hidden = false;
    const retrieval = result.retrieval || {}, badges = [], index = retrieval.index || {}, semantic = retrieval.semantic || {};
    badges.push(el("span", "badge neutral", `请求：${retrievalName(retrieval.mode || state.lastSearch?.retrieval_mode)}`), el("span", "badge neutral", `实际：${retrievalName(retrieval.effective_mode)}`));
    const statusLabel = retrieval.effective_mode === "selected_recipes" && retrieval.status === "ok" ? "选中证据读取完成" : { ok: "召回完成", fallback: "发生检索回退", unavailable: "检索不可用" }[retrieval.status] || text(retrieval.status, "检索状态未提供");
    badges.push(el("span", retrieval.status === "ok" ? "badge" : "badge warning", statusLabel));
    if (retrieval.insufficient_candidates === true) badges.push(el("span", "badge warning", "本轮候选不足"));
    $(prefix + "retrieval-badges").replaceChildren(...badges);
    const graphFirst = retrieval.method === "graph_first_e5_exact_cosine";
    const counts = [];
    if (count(semantic.graph_candidate_count) !== null) counts.push(`图谱筛选候选 ${number(semantic.graph_candidate_count)} 条`);
    if (count(semantic.scored_recipe_count) !== null) counts.push(`E5 排序 ${number(semantic.scored_recipe_count)} 条`);
    if (count(retrieval.returned_candidates) !== null) counts.push(`返回 ${number(retrieval.returned_candidates)} 条`);
    if (!graphFirst && count(retrieval.eligible_candidates) !== null) counts.push(`符合已记录提及条件 ${number(retrieval.eligible_candidates)} 条`);
    const fallbackMessage = text(retrieval.fallback_message);
    $(prefix + "retrieval-description").textContent = retrieval.effective_mode === "selected_recipes" ? "按选中的食谱读取原始图谱证据，本次未重新检索其他食谱。" : `${graphFirst ? "先筛选符合食材条件的全部图谱候选，再进行 E5 语义排序。" : ""}${counts.join("，") || "服务未提供本轮候选数量"}。${fallbackMessage ? ` ${fallbackMessage}` : retrieval.status === "fallback" ? " 当前使用可用的检索方式，详细原因见下方。" : ""}`;
    const lines = [`检索状态：${text(retrieval.status, "未提供")}`, `检索方法：${text(retrieval.method, "未提供")}`, `候选上限：${number(retrieval.candidate_limit)}`, `检索覆盖：${text(retrieval.coverage, "未提供")}`, `无结果范围：${text(retrieval.no_match_scope, "未提供")}`, `索引：${typeof retrieval.index === "string" ? retrieval.index : text(index.name, "未提供")} · ${text(index.status, "状态未提供")}`];
    if (retrieval.fallback_reason) lines.push(`回退原因：${text(retrieval.fallback_reason, "未提供")}`);
    $(prefix + "retrieval-disclosure").textContent = lines.join("\n");
    const trace = array(result.trace).length ? array(result.trace) : array(retrieval.trace);
    $(prefix + "retrieval-trace").replaceChildren(...trace.map((item) => { const row = el("li", "", `${text(item.stage, "阶段未提供")} · ${text(item.status, "状态未提供")}`); if (count(item.elapsed_ms) !== null) row.append(el("span", "", `${item.elapsed_ms.toFixed(0)} ms`)); return row; }));
    if (!trace.length) $(prefix + "retrieval-trace").append(el("li", "", "服务未提供逐阶段检索记录。"));
  }
  async function loadRecipe() {
    let id;
    try { id = decodeURIComponent(location.pathname.slice("/recipes/".length)); } catch (_) { showError("detail-error", "食谱地址不完整，请返回探索页。"); $("detail-loading").hidden = true; return; }
    if (!id || !location.pathname.startsWith("/recipes/")) { showError("detail-error", "请从探索页打开一道食谱。"); $("detail-loading").hidden = true; return; }
    const controller = new AbortController(), version = ++state.detailVersion; state.detailController = controller;
    try {
      const data = await request(`/api/recipes/${encodeURIComponent(id)}`, { signal: controller.signal });
      if (version !== state.detailVersion) return;
      const recipe = data.recipe || data;
      if (recipe.id !== id) throw new Error("食谱记录不一致，请重试。");
      state.detail = recipe; setText("detail-title", text(recipe.title, "食谱详情"));
      document.title = `${text(recipe.title, "食谱详情")} | RecipeGen`;
      renderDetail(recipe);
      $("graph-section").hidden = false;
      if ($("graph-section").open) loadGraph(id, version);
    } catch (error) { if (error.name !== "AbortError") showError("detail-error", error.message); }
    finally { $("detail-loading").hidden = true; }
  }
  function section(title) { const node = el("section", "detail-section"); node.append(el("h2", "", title)); return node; }
  function disclosure(title, className = "detail-section") { const node = el("details", className); node.append(el("summary", "", title)); return node; }
  function renderDetail(recipe) {
    const body = $("detail-body"); body.replaceChildren(); body.hidden = false;
    const mentions = section("食材"), chips = el("div", "ingredient-chips");
    chips.append(...array(recipe.ingredient_mentions).map(v => el("span", "ingredient-chip", mentionName(v))));
    if (!chips.childNodes.length) chips.append(el("p", "", "原文未单独列出食材。"));
    mentions.append(chips, el("p", "detail-warning", "原文提及可能不完整，请核对配方。"));
    const meta = recipeMeta(recipe); if (meta) mentions.append(el("p", "recipe-meta", meta));
    const steps = section("原始步骤"), ordered = el("ol", "original-steps");
    array(recipe.steps).forEach((step, index) => {
      const row = el("li", "original-step"), content = el("div");
      content.append(el("p", "", text(step.text, "原文缺失")));
      if (step.source_id) { const source = disclosure("步骤来源", "step-provenance"); source.append(el("p", "step-source mono", text(step.source_id))); content.append(source); }
      row.append(el("span", "", count(step.order) === null ? index + 1 : step.order), content); ordered.append(row);
    });
    if (ordered.childNodes.length) steps.append(ordered); else steps.append(el("p", "", "这条记录没有原始步骤。"));
    const sources = disclosure("原始来源"), sourceList = el("div", "source-list");
    array(recipe.sources).forEach(source => {
      const row = el("div", "source-record"); row.append(link(source.url), el("span", "mono", `来源 ID：${text(source.source_id, "未提供")}`));
      if (source.artifact_path) row.append(el("span", "mono", text(source.artifact_path))); sourceList.append(row);
    });
    if (!sourceList.childNodes.length) sourceList.append(el("p", "", "这条记录未提供来源条目。")); sources.append(sourceList); body.append(mentions, steps, sources);
    const observations = array(recipe.visual_evidence || recipe.image_observations || recipe.visual_observations || recipe.media?.observations);
    if (observations.length) {
      const visual = disclosure("图像描述候选"), list = el("div", "visual-observations");
      observations.slice(0, 8).forEach(item => {
        const node = el("article", "observation-card"), meta = disclosure("观察来源", "observation-provenance");
        node.append(el("p", "", text(item.caption, "未提供描述")));
        meta.append(el("p", "observation-meta", `观察：${text(item.observation_id || item.id, "未提供")}\n媒体：${text(item.source_media_id, "未提供")}\n模型：${modelName(item.model)}`));
        if (item.source_url) meta.append(link(item.source_url)); node.append(meta); list.append(node);
      });
      visual.append(el("p", "detail-warning", "模型描述仅作候选，需与原图核对。"), list); body.append(visual);
    }
  }
  async function loadGraph(recipeId, version) {
    if (state.graphLoaded) return;
    state.graphLoaded = true; state.graphController?.abort();
    const controller = new AbortController(); state.graphController = controller;
    setText("graph-status", "正在读取关联…"); $("graph-canvas").replaceChildren();
    try { const graph = await request(`/api/system/graph?recipe_id=${encodeURIComponent(recipeId)}`, { signal: controller.signal }); if (version === state.detailVersion) renderGraph(graph); }
    catch (error) { state.graphLoaded = false; if (error.name !== "AbortError") setText("graph-status", `关联读取失败：${error.message}`); }
  }
  function renderGraph(graph) {
    const allNodes = array(graph.nodes).filter((v) => typeof v.id === "string"), nodes = [...allNodes].sort((a, b) => Number(b.label === "Recipe") - Number(a.label === "Recipe")).slice(0, 36), lookup = new Map(), edges = array(graph.edges).filter((v) => typeof v.source === "string" && typeof v.target === "string"); if (!nodes.length) { $("graph-status").textContent = "当前食谱没有返回可展示的知识关联。"; return; }
    const center = nodes.findIndex((n) => n.label === "Recipe"); if (center > 0) nodes.unshift(nodes.splice(center, 1)[0]); const colors = { Recipe: "var(--accent)", Ingredient: "#839667", Step: "#d2af73", Source: "#92a6af", Image: "#92a6af", Video: "#92a6af", VisualObservation: "#bc9cbd", VisualObject: "#bc9cbd" };
    nodes.forEach((node, index) => { let x = 360, y = 160; if (index > 0) { const outer = index > 12, within = outer ? index - 13 : index - 1, amount = outer ? Math.max(1, nodes.length - 13) : Math.min(12, nodes.length - 1), angle = within / amount * Math.PI * 2 - Math.PI / 2; x += Math.cos(angle) * (outer ? 274 : 161); y += Math.sin(angle) * (outer ? 123 : 88); } lookup.set(node.id, { ...node, x, y }); });
    const svg = svgEl("svg", { viewBox: "0 0 720 335", role: "group", "aria-label": "食谱知识关联图，使用 Tab 选择节点并按 Enter 查看详情" }), lineGroup = svgEl("g"), nodeGroup = svgEl("g"), edgeElements = [], nodeElements = new Map();
    function choose(node) { const connected = edges.filter((e) => e.source === node.id || e.target === node.id); nodeElements.forEach((element, id) => element.classList.toggle("active", id === node.id)); edgeElements.forEach(({ element, edge }) => element.classList.toggle("active", edge.source === node.id || edge.target === node.id)); const relations = connected.slice(0, 8).map((e) => { const other = lookup.get(e.source === node.id ? e.target : e.source); return `${text(e.type, "关联")} → ${text(other?.name, "未展示节点")}`; }); $("graph-node-detail").textContent = `${text(node.name, "名称未提供")}\n类型：${text(node.label, "未提供")} · ID：${node.id}${relations.length ? `\n${relations.join("\n")}` : "\n当前视图未展示关联边。"}`; }
    edges.forEach((edge) => { const from = lookup.get(edge.source), to = lookup.get(edge.target); if (!from || !to) return; const line = svgEl("line", { x1: from.x, y1: from.y, x2: to.x, y2: to.y, class: "graph-edge" }); line.append(svgEl("title", {}, text(edge.type, "关联"))); edgeElements.push({ element: line, edge }); lineGroup.append(line); });
    lookup.forEach((node, id) => { const group = svgEl("g", { class: "graph-node", transform: `translate(${node.x},${node.y})`, role: "button", tabindex: "0", "aria-label": `${text(node.label, "节点")}：${text(node.name, id)}` }), name = text(node.name, node.id); group.append(svgEl("circle", { r: node.label === "Recipe" ? 15 : 9, fill: colors[node.label] || "#a3aca0" }), svgEl("text", { x: 0, y: node.label === "Recipe" ? 31 : 23, "text-anchor": "middle" }, name.length > 17 ? `${name.slice(0, 16)}…` : name), svgEl("title", {}, `${text(node.label)} · ${name}`)); group.addEventListener("click", () => choose(node)); group.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(node); } }); nodeElements.set(id, group); nodeGroup.append(group); }); svg.append(lineGroup, nodeGroup); $("graph-canvas").replaceChildren(svg); $("graph-status").textContent = `显示 ${nodes.length} / ${allNodes.length} 个节点。关系表示图谱关联；模型关系仍是候选。`;
  }
  const on = (id, event, listener) => $(id)?.addEventListener(event, listener);
  restoreWorkspace();
  on("search-form", "submit", search);
  ["ingredients", "excluded-ingredients"].forEach(id => on(id, "input", () => { $("ingredients").removeAttribute("aria-invalid"); saveSearchDraft(); }));
  on("graph-section", "toggle", () => { if ($("graph-section").open && state.detail) loadGraph(state.detail.id, state.detailVersion); });
  window.addEventListener("pagehide", () => { ++state.searchVersion; state.searchController?.abort(); state.detailController?.abort(); state.graphController?.abort(); });
  window.addEventListener("pageshow", event => { if (event.persisted && page === "explore") { restoreWorkspace(); restoreSearchView(); setSearchBusy(false); } });
  if (page === "explore") { restoreSearchView(); loadStatus(); }
  if (page === "recipe") loadRecipe();
  if (page === "about") loadStatus();
})();
