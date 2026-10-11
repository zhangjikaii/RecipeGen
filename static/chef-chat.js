/* Native RecipeGen chat. Model output is always rendered as text, never HTML. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const STORAGE_KEY = "recipegen-chef-chat-v1";
  const state = { messages: [], busy: false, controller: null, version: 0, failed: null, draft: "" };
  const list = value => Array.isArray(value) ? value : [];
  const text = (value, fallback = "") => typeof value === "string" ? value : fallback;
  const node = (tag, className, value) => { const element = document.createElement(tag); if (className) element.className = className; if (value !== undefined) element.textContent = value; return element; };
  const uid = () => typeof crypto.randomUUID === "function" ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  function safeURL(value) { try { const url = new URL(value); return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password ? url.href : null; } catch (_) { return null; } }
  function save() {
    try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 1, messages: state.messages.filter(message => message.confirmed).slice(-20), draft: $("chat-input").value })); } catch (_) { /* Chat still works when browser storage is disabled. */ }
  }
  function restore() {
    try {
      const saved = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "null");
      if (saved?.version !== 1) return;
      state.messages = list(saved.messages).filter(message => ["user", "assistant"].includes(message?.role) && typeof message.content === "string" && message.confirmed === true).slice(-20).map(message => ({ ...message, id: typeof message.id === "string" && /^[a-zA-Z0-9-]+$/.test(message.id) ? message.id : uid() }));
      $("chat-input").value = text(saved.draft).slice(0, 1000);
    } catch (_) { state.messages = []; }
  }
  function historyForRequest(message) {
    const recent = state.messages.filter(message => message.confirmed).slice(-20);
    const result = []; let budget = Math.max(0, 20000 - message.length);
    for (let index = recent.length - 1; index >= 0 && budget > 0; index--) {
      const content = recent[index].content.slice(0, Math.min(4000, budget));
      if (content) { result.unshift({ role: recent[index].role, content }); budget -= content.length; }
    }
    return result;
  }
  function resizeInput() {
    const input = $("chat-input"); input.style.height = "auto"; input.style.height = `${Math.min(180, Math.max(50, input.scrollHeight))}px`;
    input.style.overflowY = input.scrollHeight > 180 ? "auto" : "hidden";
  }
  function updateControls() {
    $("chat-send").disabled = state.busy || !$("chat-input").value.trim();
    $("chat-cancel").hidden = !state.busy; $("chat-send").hidden = state.busy;
    $("chat-input").readOnly = state.busy; $("chat-form").setAttribute("aria-busy", String(state.busy));
    $("chat-status").textContent = state.busy ? "厨神正在思考…" : "";
    $("chat-retry").disabled = state.busy; resizeInput();
  }
  function clearError() { $("chat-error").hidden = true; $("chat-error-text").textContent = ""; }
  function showError(message) { $("chat-error-text").textContent = message; $("chat-error").hidden = false; }
  function sourceLink(url) {
    const safe = safeURL(url); if (!safe) return null;
    const anchor = node("a", "source-link", "查看原始来源 ↗"); anchor.href = safe; anchor.target = "_blank"; anchor.rel = "noopener noreferrer"; return anchor;
  }
  function renderAssistant(message, article) {
    const result = message.result || {}, evidence = list(result.evidence).filter(item => typeof item?.id === "string");
    const evidenceTargets = new Map(evidence.map((item, index) => [item.id, `chat-${message.id}-evidence-${index + 1}`]));
    const body = node("div", "chat-message-body"), paragraphs = list(result.paragraphs || result.qa?.paragraphs).filter(item => typeof item?.text === "string");
    if (paragraphs.length) {
      paragraphs.forEach(item => {
        const paragraph = node("p", "", item.text);
        const ids = list(item.evidence_ids || item.citations).filter(id => typeof id === "string");
        [...new Set(ids)].forEach(id => {
          if (!evidenceTargets.has(id)) return;
          const citation = node("a", "chat-citation", id); citation.href = `#${evidenceTargets.get(id)}`;
          citation.setAttribute("aria-label", `查看出处 ${id}`);
          citation.addEventListener("click", event => {
            event.preventDefault(); const target = document.getElementById(evidenceTargets.get(id)); if (!target) return;
            const disclosure = target.closest("details"); if (disclosure) disclosure.open = true;
            target.focus(); target.scrollIntoView({ behavior: "auto", block: "nearest" });
          }); paragraph.append(document.createTextNode(" "), citation);
        }); body.append(paragraph);
      });
    } else body.append(node("p", "", message.content));
    article.append(body);
    if (result.generation?.fallback) {
      article.append(node("p", "chat-answer-warning chat-fallback", "回答服务暂不可用，已显示回退结果。"));
    } else if (result.route?.action === "retrieve" && !evidence.length) {
      article.append(node("p", "chat-answer-warning", "没有找到足够的食谱依据。"));
    }
    const recipes = list(result.recipes).filter(recipe => typeof recipe?.id === "string");
    if (recipes.length) {
      const links = node("div", "chat-recipe-links");
      recipes.forEach(recipe => { const anchor = node("a", "", text(recipe.title, "查看食谱")); anchor.href = `/recipes/${encodeURIComponent(recipe.id)}`; links.append(anchor); }); article.append(links);
    }
    if (evidence.length) {
      const sources = node("details", "chat-sources"), summary = node("summary", "", `出处与原文 · ${evidence.length} 条`), container = node("div", "chat-evidence-list");
      evidence.forEach((item, index) => {
        const entry = node("article", "chat-evidence"); entry.id = evidenceTargets.get(item.id); entry.tabIndex = -1;
        entry.append(node("h3", "", `${item.id} · ${text(item.recipe_title, "原始步骤")}`), node("p", "", text(item.text || item.original_text, "未提供原文")));
        const detail = [item.step_id || item.graph_id, item.source_id].filter(value => typeof value === "string" && value);
        if (detail.length) entry.append(node("p", "chat-evidence-id mono", detail.join("\n")));
        const urls = [...new Set([item.source_url, ...list(item.source_urls)].filter(value => typeof value === "string"))];
        urls.forEach(url => { const anchor = sourceLink(url); if (anchor) entry.append(anchor); }); container.append(entry);
      }); sources.append(summary, container); article.append(sources);
    }
  }
  function render() {
    $("chat-welcome").hidden = state.messages.length > 0; $("chat-messages").hidden = !state.messages.length;
    $("chat-messages").replaceChildren(...state.messages.map(message => {
      const article = node("article", `chat-message ${message.role === "user" ? "chat-user" : "chat-assistant"}`); article.dataset.role = message.role; article.id = `chat-message-${message.id}`;
      if (message.role === "assistant") {
        const heading = node("div", "chat-message-heading"), avatar = node("span", "chef-avatar", "👨‍🍳"); avatar.setAttribute("aria-hidden", "true"); heading.append(avatar, node("span", "", "厨神")); article.append(heading); renderAssistant(message, article);
      } else { article.setAttribute("aria-label", "你的消息"); article.append(node("p", "chat-user-text", message.content)); if (!message.confirmed) article.append(node("span", "chat-pending-note", state.busy ? "发送中" : "尚未完成")); }
      return article;
    }));
    if (state.busy) {
      const loading = node("div", "chat-thinking"); loading.setAttribute("role", "status");
      loading.append(node("span", "chef-avatar", "👨‍🍳"), node("span", "", "厨神正在思考…")); $("chat-messages").append(loading);
    }
  }
  function scrollToLatest() { $("chat-composer-dock")?.scrollIntoView({ behavior: "auto", block: "end" }); window.scrollTo({ top: document.body.scrollHeight, behavior: "auto" }); }
  async function send(content) {
    const message = content.trim(); if (!message || state.busy) return;
    if (message.length > 1000) { showError("消息最多 1000 字，请稍微精简后再发送。"); return; }
    clearError(); state.failed = null;
    // An unconfirmed message never becomes model history, including after retry or cancellation.
    state.messages = state.messages.filter(entry => entry.confirmed);
    const history = historyForRequest(message), user = { id: uid(), role: "user", content: message, confirmed: false };
    state.messages.push(user); state.busy = true; $("chat-input").value = "";
    const controller = new AbortController(), version = ++state.version; state.controller = controller;
    updateControls(); render(); scrollToLatest();
    let timedOut = false; const timer = setTimeout(() => { timedOut = true; controller.abort(); }, 240000);
    try {
      const response = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message, history }), signal: controller.signal });
      let result; try { result = await response.json(); } catch (_) { throw new Error(`无法读取服务响应（HTTP ${response.status}）。`); }
      if (version !== state.version || controller.signal.aborted) return;
      if (!response.ok || result?.status === "error") throw new Error(text(result?.error?.message || result?.detail, `回答未完成（HTTP ${response.status}），请重试。`));
      if (!result || typeof result.answer !== "string" || !result.answer.trim()) throw new Error("服务没有返回完整回答，请重试。");
      user.confirmed = true; state.messages.push({ id: uid(), role: "assistant", content: result.answer, confirmed: true, result });
      state.messages = state.messages.slice(-20); save();
    } catch (error) {
      if (version !== state.version) return;
      state.failed = { message }; $("chat-input").value = message;
      showError(timedOut ? "回答超时，当前请求尚未确认完成。可以稍后重试。" : error.name === "AbortError" ? "已停止等待此回答。" : error instanceof TypeError ? "无法连接厨神，请确认服务已启动后重试。" : error.message);
    } finally {
      clearTimeout(timer);
      if (version === state.version) { state.busy = false; state.controller = null; updateControls(); render(); save(); scrollToLatest(); $("chat-input").focus({ preventScroll: true }); }
    }
  }
  function cancel() {
    if (!state.busy) return;
    const pending = state.messages.findLast?.(message => !message.confirmed) || [...state.messages].reverse().find(message => !message.confirmed);
    ++state.version; state.controller?.abort(); state.controller = null; state.busy = false;
    if (pending) { state.failed = { message: pending.content }; $("chat-input").value = pending.content; }
    showError("已停止等待此回答。可修改消息或重新发送。"); updateControls(); render(); save(); $("chat-input").focus({ preventScroll: true });
  }
  $("chat-form").addEventListener("submit", event => { event.preventDefault(); send($("chat-input").value); });
  $("chat-input").addEventListener("input", () => { updateControls(); save(); });
  $("chat-input").addEventListener("keydown", event => { if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) { event.preventDefault(); if (!state.busy) send($("chat-input").value); } });
  $("chat-cancel").addEventListener("click", cancel);
  $("chat-retry").addEventListener("click", () => send(state.failed?.message || $("chat-input").value));
  $("chat-new").addEventListener("click", () => {
    ++state.version; state.controller?.abort(); state.controller = null; state.busy = false; state.failed = null; state.messages = []; $("chat-input").value = "";
    clearError(); try { sessionStorage.removeItem(STORAGE_KEY); } catch (_) {} updateControls(); render(); window.scrollTo({ top: 0, behavior: "auto" }); $("chat-input").focus({ preventScroll: true });
  });
  document.querySelectorAll("[data-prompt]").forEach(button => button.addEventListener("click", () => { if (state.busy) return; $("chat-input").value = button.dataset.prompt || ""; updateControls(); save(); $("chat-input").focus(); }));
  window.addEventListener("pagehide", () => { ++state.version; state.controller?.abort(); save(); });
  window.addEventListener("pageshow", event => {
    if (!event.persisted) return;
    state.busy = false; state.controller = null; state.messages = state.messages.filter(message => message.confirmed);
    restore(); render(); updateControls();
  });
  try { sessionStorage.removeItem("recipegen-workspace-v2"); } catch (_) {}
  restore(); render(); updateControls();
})();
