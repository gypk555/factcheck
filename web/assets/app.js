// XFND-CLIP web UI. Plain JavaScript, no libraries.
// Everything from the server or the user is inserted with textContent, never innerHTML.
"use strict";

const $ = (id) => document.getElementById(id);
const setText = (el, t) => { el.textContent = t; };
const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); };
function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("data-") || k.startsWith("aria-") || k === "role") n.setAttribute(k, v);
    else n[k] = v;
  }
  for (const kid of kids) if (kid != null) n.append(kid);
  return n;
}
const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;

// localStorage can be missing or full (private windows): never let it break the page.
const store = {
  get(k, fallback) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : fallback; } catch { return fallback; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); return true; } catch { return false; } },
  del(k) { try { localStorage.removeItem(k); } catch { /* ignore */ } },
};

// ============================================================ connection
const conn = { url: null, key: null, online: false, info: null };
const statusBox = $("model-status");

function parseCode(code) {
  code = (code || "").trim();
  let u;
  try { u = new URL(code); } catch { throw new Error("That doesn't look like a connection code. Copy it again from the notebook's green box."); }
  const key = new URLSearchParams(u.hash.slice(1)).get("key");
  const local = ["127.0.0.1", "localhost"].includes(u.hostname);
  // Only a Cloudflare quick tunnel (or the local preview) is accepted. This is a
  // format check, not proof of ownership: that's why codes are pasted by hand.
  const tunnel = u.protocol === "https:" && /^[a-z0-9-]+\.trycloudflare\.com$/.test(u.hostname);
  if (!tunnel && !(local && u.protocol === "http:")) {
    throw new Error("The connection code must be the https://….trycloudflare.com link printed by the notebook.");
  }
  if (!key || key.length < 16) throw new Error("The connection code is missing its key (the part after #key=).");
  return { url: u.origin, key };
}

function showStatus(state, title, detail) {
  statusBox.dataset.state = state;
  setText($("status-title"), title);
  setText($("status-detail"), detail);
  setText($("connect-open"), state === "online" ? "Change" : "Connect");
  updateAnalyseButton();
}

async function api(path, { method = "GET", body, timeout = 20000 } = {}) {
  if (!conn.url) throw new Error("Connect the model first.");
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeout);
  let res;
  try {
    res = await fetch(conn.url + path, { method, body, headers: { "X-Access-Key": conn.key }, signal: ctl.signal, mode: "cors" });
  } catch (err) {
    throw new Error(err.name === "AbortError"
      ? "The model took too long to answer. Check the Colab tab is still running."
      : "Can't reach the model. Is the Colab notebook still running? Run Box 4 again and reconnect.");
  } finally {
    clearTimeout(timer);
  }
  let data = {};
  try { data = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `The model returned an error (${res.status}).`);
  return data;
}

const hostOf = (url) => { try { return new URL(url).hostname; } catch { return ""; } };

async function checkHealth({ quiet = false } = {}) {
  if (!conn.url) { showStatus("offline", "Model offline", "Run the Colab notebook to connect"); return false; }
  if (!quiet) showStatus("checking", "Connecting…", "Loading can take a minute");
  try {
    const h = await api("/api/health", { timeout: 120000 });
    conn.online = true;
    conn.info = h;
    showStatus("online", "Model online", `${h.gpu || "Colab"} via ${hostOf(conn.url).split(".")[0]}`);
    return true;
  } catch (err) {
    conn.online = false;
    showStatus("offline", "Model offline", "Colab isn't answering. Run Box 4 and reconnect.");
    if (!quiet) throw err;
    return false;
  }
}

async function connectWith(code) {
  const parsed = parseCode(code);
  const previous = { ...conn };
  Object.assign(conn, parsed, { online: false });
  try {
    await checkHealth();
  } catch (err) {
    Object.assign(conn, previous); // keep the old, working connection
    if (previous.url) checkHealth({ quiet: true }); else showStatus("offline", "Model offline", "Run the Colab notebook to connect");
    throw err;
  }
  store.set("xfnd.conn", parsed);
}

function disconnect() {
  Object.assign(conn, { url: null, key: null, online: false, info: null });
  store.del("xfnd.conn");
  showStatus("offline", "Model offline", "Run the Colab notebook to connect");
}

// keep the status honest while the page is open
setInterval(() => { if (conn.url && document.visibilityState === "visible") checkHealth({ quiet: true }); }, 45000);

// ---- connect dialog
const dialog = $("connect-dialog");
const connectError = $("connect-error");
function openConnect(message, code) {
  setText(connectError, message || "");
  connectError.hidden = !message;
  $("connect-code").value = code ?? (conn.url ? `${conn.url}#key=${conn.key}` : "");
  $("disconnect-btn").hidden = !conn.url;
  dialog.showModal();
  $("connect-code").focus();
}
$("connect-open").addEventListener("click", () => openConnect());
$("connect-close").addEventListener("click", () => dialog.close());
$("disconnect-btn").addEventListener("click", () => { disconnect(); dialog.close(); });
$("connect-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("connect-btn");
  btn.disabled = true;
  setText(btn, "Connecting…");
  connectError.hidden = true;
  try {
    await connectWith($("connect-code").value);
    dialog.close();
  } catch (err) {
    setText(connectError, err.message);
    connectError.hidden = false;
  } finally {
    btn.disabled = false;
    setText(btn, "Connect");
  }
});

// ============================================================ router
const VIEWS = ["check", "how", "results", "history"];
function route() {
  const name = (location.hash.match(/^#\/(\w+)/) || [])[1];
  const view = VIEWS.includes(name) ? name : "check";
  for (const v of VIEWS) $(`view-${v}`).hidden = v !== view;
  document.querySelectorAll("[data-route]").forEach((a) => {
    if (a.dataset.route === view) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  const titles = { check: "Check a post", how: "How it works", results: "Results", history: "History" };
  document.title = `${titles[view]} | XFND-CLIP`;
  if (view === "results") loadResults();
  if (view === "history") renderHistory();
}
window.addEventListener("hashchange", () => {
  if (handleConnectLink()) return; // a notebook link opened in an already-open tab
  route();
  $("main").focus({ preventScroll: true });
  window.scrollTo(0, 0);
});

// Connection codes are only ever pasted by hand from your own Colab output.
// Links carrying a code (#connect=...) are ignored on purpose: anyone can create
// a trycloudflare.com address, so a link from someone else could point this site
// at their server. Such a link is just removed from the address bar.
function handleConnectLink() {
  if (!/^#connect=/.test(location.hash)) return false;
  history.replaceState(null, "", `${location.pathname}#/check`);
  route();
  return true;
}

// ============================================================ check a post
const form = $("check-form");
const claim = $("claim");
const analyseBtn = $("analyse-btn");
const errorBox = $("form-error");
const TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"];
let chosen = null;
let busy = false;

function showError(msg) { setText(errorBox, msg || ""); errorBox.hidden = !msg; }
const MAX_WORDS = 60; // the model reads about 64 BERT tokens: a headline, not an article
const wordCount = () => claim.value.trim().split(/\s+/).filter(Boolean).length;
function updateCount() {
  const n = wordCount();
  const c = $("claim-count");
  setText(c, `${n} / ${MAX_WORDS} words`);
  c.classList.toggle("over", n > MAX_WORDS);
}
function updateAnalyseButton() {
  setText(analyseBtn, busy ? "Analysing…" : conn.online ? "Analyse post" : "Connect model to analyse");
  analyseBtn.disabled = busy;
}
claim.addEventListener("input", updateCount);

function chooseImage(file) {
  showError("");
  if (!file) return;
  if (!TYPES.includes(file.type)) { showError("Use a JPG, PNG, WEBP, BMP or GIF image."); return; }
  if (file.size > 10 * 1024 * 1024) { showError("That image is over 10 MB. Use a smaller one."); return; }
  chosen = file;
  const prev = $("drop-preview");
  if (prev.src) URL.revokeObjectURL(prev.src);
  prev.src = URL.createObjectURL(file);
  prev.hidden = false;
  $("drop-empty").hidden = true;
  $("image-remove").hidden = false;
}
function removeImage() {
  chosen = null;
  $("image-input").value = "";
  const prev = $("drop-preview");
  if (prev.src) URL.revokeObjectURL(prev.src);
  prev.removeAttribute("src");
  prev.hidden = true;
  $("drop-empty").hidden = false;
  $("image-remove").hidden = true;
}
$("image-input").addEventListener("change", (e) => chooseImage(e.target.files[0]));
$("image-remove").addEventListener("click", removeImage);
const drop = $("drop");
drop.addEventListener("dragover", (e) => { if ([...e.dataTransfer.types].includes("Files")) { e.preventDefault(); drop.classList.add("dropping"); } });
drop.addEventListener("dragleave", () => drop.classList.remove("dropping"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("dropping");
  chooseImage([...e.dataTransfer.files].find((f) => f.type.startsWith("image/")));
});
document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (!item || $("view-check").hidden) return;
  e.preventDefault();
  chooseImage(item.getAsFile());
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = claim.value.trim();
  if (wordCount() < 3) { showError("Type the full claim (at least 3 words)."); claim.focus(); return; }
  if (wordCount() > MAX_WORDS) { showError(`Keep the claim to ${MAX_WORDS} words or fewer: the model reads a headline, not a whole article.`); claim.focus(); return; }
  if (!conn.url) { openConnect(); return; }
  showError("");
  busy = true;
  updateAnalyseButton();
  $("result").hidden = true;
  $("busy").hidden = false;
  try {
    const body = new FormData();
    body.append("text", text);
    if (chosen) body.append("image", chosen);
    const r = await api("/api/analyse", { method: "POST", body, timeout: 180000 });
    conn.online = true;
    showStatus("online", "Model online", `${conn.info?.gpu || "Colab"} via ${hostOf(conn.url).split(".")[0]}`);
    renderResult(r);
    saveToHistory(r);
  } catch (err) {
    showError(err.message);
    checkHealth({ quiet: true });
  } finally {
    busy = false;
    $("busy").hidden = true;
    updateAnalyseButton();
  }
});

$("again-btn").addEventListener("click", () => {
  $("result").hidden = true;
  claim.value = "";
  updateCount();
  removeImage();
  window.scrollTo(0, 0);
  claim.focus();
});

// ---- result rendering
const safeImg = (src) => (typeof src === "string" && src.startsWith("data:image/") ? src : "");

function renderResult(r) {
  const res = $("result");
  const v = $("verdict");
  v.dataset.label = r.label;
  $("verdict-icon").setAttribute("href", r.label === "FAKE" ? "#i-fake" : "#i-real");
  setText($("verdict-chip-text"), r.label === "FAKE" ? "Likely fake news" : "Likely real news");
  setText($("verdict-title"), r.label);
  setText($("verdict-claim"), r.claim);
  setText($("score-value"), pct(r.prob_fake));
  requestAnimationFrame(() => { $("score-fill").style.width = pct(r.prob_fake, 1); });

  // words: highlighted claim + bars
  const words = Array.isArray(r.words) ? r.words : [];
  const maxAbs = Math.max(1e-6, ...words.map((w) => Math.abs(w.weight)));
  const marked = $("claim-marked");
  clear(marked);
  words.forEach((w, i) => {
    const s = Math.abs(w.weight) / maxAbs;
    const span = el("span", { class: s < 0.2 ? "w" : `w ${w.weight > 0 ? "f" : "r"}`, text: w.word,
      title: `${w.weight > 0 ? "Towards fake" : "Towards real"}: ${w.weight > 0 ? "+" : ""}${(w.weight * 100).toFixed(1)} points` });
    span.style.setProperty("--s", (0.25 + 0.75 * s).toFixed(2));
    marked.append(span, i < words.length - 1 ? " " : "");
  });
  const bars = $("word-bars");
  clear(bars);
  [...words].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight)).slice(0, 8).forEach((w) => {
    const bar = el("span", { class: `bar ${w.weight > 0 ? "f" : "r"}` });
    bar.style.width = `${(Math.abs(w.weight) / maxAbs) * 50}%`;
    bars.append(el("li", {},
      el("span", { class: "word", text: w.word }),
      el("span", { class: "track" }, bar),
      el("span", { class: "val", text: `${w.weight > 0 ? "+" : "−"}${Math.abs(w.weight * 100).toFixed(1)}` })));
  });

  // modality weights
  const mods = $("mod-bars");
  clear(mods);
  const M = [["Text", "text", "--m-text"], ["Image", "image", "--m-image"], ["Text-image", "text_image", "--m-mm"]];
  for (const [name, key, color] of M) {
    const val = r.modality?.[key] ?? 0;
    const fill = el("span", { class: "mod-fill" });
    const row = el("div", { class: "mod" }, el("span", { class: "mod-name", text: name }), el("span", { class: "mod-track" }, fill), el("span", { class: "mod-val", text: pct(val) }));
    row.style.setProperty("--c", `var(${color})`);
    mods.append(row);
    requestAnimationFrame(() => { fill.style.width = pct(val, 1); });
  }

  // CLIP similarity (only with an image)
  const hasImage = Boolean(r.image?.preview && r.clip_sim != null);
  $("clip-box").hidden = !hasImage;
  $("panel-image").hidden = !hasImage;
  if (hasImage) {
    const lo = 0.1, hi = 0.45;
    const pos = (x) => `${Math.min(100, Math.max(0, ((x - lo) / (hi - lo)) * 100))}%`;
    $("clip-avg").style.left = pos(r.clip_sim_mean);
    $("clip-avg").title = `Training average ${r.clip_sim_mean.toFixed(2)}`;
    requestAnimationFrame(() => { $("clip-dot").style.left = pos(r.clip_sim); });
    setText($("clip-text"), r.clip_match === "match"
      ? `CLIP similarity ${r.clip_sim.toFixed(2)}, above the training average (${r.clip_sim_mean.toFixed(2)}): the picture fits the claim.`
      : `CLIP similarity ${r.clip_sim.toFixed(2)}, below the training average (${r.clip_sim_mean.toFixed(2)}): the picture fits the claim poorly. Real photos reused with a false caption often look like this.`);
    $("view-base").src = safeImg(r.image.preview);
    $("view-heat").src = safeImg(r.image.heatmap);
    setImageView("heat");
  }

  res.hidden = false;
  res.focus({ preventScroll: true });
  res.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
}

const segButtons = [...document.querySelectorAll("#panel-image [data-view]")];
function setImageView(view) {
  for (const b of segButtons) b.setAttribute("aria-checked", String(b.dataset.view === view));
  $("view-heat").hidden = view !== "heat";
  $("view-base").parentElement.classList.toggle("dim", view === "heat");
}
segButtons.forEach((b) => b.addEventListener("click", () => setImageView(b.dataset.view)));

// ============================================================ history (this browser only)
const HKEY = "xfnd.history.v1";
const HMAX = 20;

function shrinkDataUrl(src, side, type) {
  return new Promise((resolve) => {
    if (!src) return resolve(null);
    const img = new Image();
    img.onload = () => {
      const s = Math.min(1, side / Math.max(img.width, img.height));
      const c = document.createElement("canvas");
      c.width = Math.round(img.width * s);
      c.height = Math.round(img.height * s);
      c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
      resolve(c.toDataURL(type, 0.8));
    };
    img.onerror = () => resolve(null);
    img.src = src;
  });
}

async function saveToHistory(r) {
  const copy = JSON.parse(JSON.stringify(r));
  if (copy.image) {
    copy.image.preview = await shrinkDataUrl(copy.image.preview, 480, "image/jpeg");
    copy.image.heatmap = await shrinkDataUrl(copy.image.heatmap, 480, "image/png");
    if (!copy.image.preview) copy.image = null; // couldn't shrink: keep the text result only
  }
  const list = store.get(HKEY, []);
  list.unshift({ id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`, at: new Date().toISOString(), result: copy });
  list.length = Math.min(list.length, HMAX);
  while (list.length && !store.set(HKEY, list)) list.pop(); // storage full: drop the oldest
  updateHistoryCount();
}

function updateHistoryCount() {
  const n = store.get(HKEY, []).length;
  const badge = $("history-count");
  setText(badge, String(n));
  badge.hidden = n === 0;
}

function renderHistory() {
  const list = store.get(HKEY, []);
  const ul = $("history-list");
  clear(ul);
  $("history-empty").hidden = list.length > 0;
  $("history-clear").hidden = list.length === 0;
  for (const item of list) {
    const r = item.result;
    const thumb = r.image?.preview ? el("img", { src: safeImg(r.image.preview), alt: "" }) : el("span", { class: "h-noimg", text: "Text only" });
    const when = new Date(item.at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
    const open = el("button", { class: "h-open", type: "button" },
      el("span", { class: "h-claim", text: r.claim }),
      el("span", { class: "h-meta" }, el("span", { class: `pill ${r.label}`, text: r.label }), el("span", { text: `${pct(r.prob_fake)} fake` }), el("span", { text: when })));
    open.addEventListener("click", () => {
      location.hash = "#/check";
      setTimeout(() => renderResult(r), 0);
    });
    const del = el("button", { class: "icon-btn", type: "button", "aria-label": "Delete this check" });
    del.append(svgUse("#i-trash"));
    del.addEventListener("click", () => {
      store.set(HKEY, store.get(HKEY, []).filter((x) => x.id !== item.id));
      renderHistory();
      updateHistoryCount();
    });
    ul.append(el("li", { class: "card h-item" }, thumb, open, del));
  }
}
function svgUse(href) {
  const ns = "http://www.w3.org/2000/svg";
  const s = document.createElementNS(ns, "svg");
  s.setAttribute("aria-hidden", "true");
  const u = document.createElementNS(ns, "use");
  u.setAttribute("href", href);
  s.append(u);
  return s;
}
$("history-clear").addEventListener("click", () => {
  if (!confirm("Delete all checks from this browser's history?")) return;
  store.del(HKEY);
  renderHistory();
  updateHistoryCount();
});

// ============================================================ results (static files)
function parseCSV(text) {
  const rows = [];
  let row = [], field = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') q = false;
      else field += c;
    } else if (c === '"') q = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.some((x) => x !== "")) rows.push(row);
      row = [];
    } else field += c;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  const [head, ...body] = rows;
  return body.map((r) => Object.fromEntries(head.map((h, i) => [h, r[i] ?? ""])));
}

const MODEL_NAMES = {
  tfidf_lr: ["TF-IDF + Logistic Regression", "classic text baseline"],
  text_only: ["Text only", "BERT"],
  image_only: ["Image only", "ResNet-50"],
  concat: ["Text + image, joined", "simple fusion"],
  fnd_clip: ["FND-CLIP", "base paper"],
  xfnd_clip: ["XFND-CLIP", "proposed model"],
  xfnd_no_consistency: ["XFND-CLIP without consistency features", "ablation"],
  xfnd_no_availability: ["XFND-CLIP without availability-aware attention", "ablation"],
  xfnd_no_attention: ["XFND-CLIP without attention", "ablation"],
  xfnd_ce_loss: ["XFND-CLIP with plain cross-entropy loss", "ablation"],
};

let resultsLoaded = false;
async function loadResults() {
  if (resultsLoaded) return;
  resultsLoaded = true;
  try {
    const [metrics, live, wrong] = await Promise.all([
      fetch("data/metrics.json").then((r) => r.json()),
      fetch("data/live_test.csv").then((r) => r.text()).then(parseCSV),
      fetch("data/wrong_examples.csv").then((r) => r.text()).then(parseCSV),
    ]);
    renderKpis(metrics);
    renderModelBars(metrics);
    renderLive(live);
    renderWrong(wrong);
    fillSamples(live);
  } catch {
    resultsLoaded = false;
    $("kpis").append(el("p", { class: "form-error", text: "Couldn't load the results files." }));
  }
}

function renderKpis(M) {
  const m = M.xfnd_clip.mean, meta = M._meta || {};
  const kpis = $("kpis");
  clear(kpis);
  for (const [label, val] of [["Accuracy", pct(m.accuracy, 1)], ["Macro F1", m.f1_macro.toFixed(3)], ["ROC-AUC", m.roc_auc.toFixed(3)],
    ["Precision (fake)", pct(m.precision_fake, 1)], ["Recall (fake)", pct(m.recall_fake, 1)]]) {
    kpis.append(el("div", { class: "card kpi" }, el("strong", { text: val }), el("span", { text: label })));
  }
  setText($("results-sub"), `XFND-CLIP on the IFND test set, averaged over ${(meta.seeds || []).length || 3} training runs `
    + `(seeds ${(meta.seeds || []).join(", ")}). Training all models took about ${Math.round(meta.minutes || 0)} minutes on a Colab GPU, after the features were extracted.`);
}

function renderModelBars(M) {
  const rows = Object.entries(M).filter(([k, v]) => k !== "_meta" && v.mean)
    .map(([k, v]) => ({ k, acc: v.mean.accuracy, sd: v.std?.accuracy ?? 0 }))
    .sort((a, b) => b.acc - a.acc);
  const lo = 0.7;
  const box = $("model-bars");
  clear(box);
  for (const r of rows) {
    const [name, note] = MODEL_NAMES[r.k] || [r.k, ""];
    const fill = el("span", { class: "hbar-fill" });
    fill.style.width = `${Math.max(0, (r.acc - lo) / (1 - lo)) * 100}%`;
    box.append(el("div", { class: `hbar${r.k === "xfnd_clip" ? " ours" : r.k === "fnd_clip" ? " base" : ""}` },
      el("span", { class: "hbar-name" }, name, el("small", { text: note })),
      el("span", { class: "hbar-track" }, fill),
      el("span", { class: "hbar-val" }, pct(r.acc, 1), el("small", { text: r.sd ? `± ${(r.sd * 100).toFixed(1)}` : "1 run" }))));
  }
  const ours = M.xfnd_clip.mean.accuracy, base = M.fnd_clip.mean.accuracy;
  setText($("cmp-note"), `Bars start at 70%. XFND-CLIP beats the base paper by ${((ours - base) * 100).toFixed(1)} points, a small gain close to the run-to-run variation.`);
}

function renderLive(rows) {
  const tb = $("live-table").querySelector("tbody");
  clear(tb);
  let agree = 0;
  rows.forEach((r, i) => {
    const ok = r.fact_check === r.model;
    if (ok) agree++;
    const btn = el("button", { class: "link-btn", type: "button", text: "Explanation" });
    btn.addEventListener("click", () => openLightbox(`data/figures/live_${r["#"] || i + 1}.png`, r.claim));
    tb.append(el("tr", {},
      el("td", { class: "num idx", text: r["#"] }),
      el("td", { class: "wide", text: r.claim }),
      el("td", { "data-label": "Fact-check" }, el("span", { class: `pill ${r.fact_check}`, text: r.fact_check })),
      el("td", { "data-label": "Model" }, el("span", { class: `pill ${r.model}`, text: r.model }), " ", el("span", { class: `agree ${ok ? "yes" : "no"}`, text: ok ? "✓" : "✗", title: ok ? "Agrees" : "Disagrees" })),
      el("td", { class: "num", "data-label": "P(fake)", text: Number(r["P(fake)"]).toFixed(2) }),
      el("td", { class: "num", "data-label": "CLIP sim", text: Number(r.clip_sim).toFixed(2) }),
      el("td", {}, btn)));
  });
  const fakes = rows.filter((r) => r.fact_check === "FAKE");
  const missed = fakes.filter((r) => r.model !== "FAKE").length;
  setText($("live-summary"), `The model agreed with the fact-checkers on ${agree} of ${rows.length} posts published after its training data. `
    + `It called ${missed} of ${fakes.length} fake posts real: new events and well-written false claims are its main weakness.`);
}

function renderWrong(rows) {
  const g = $("wrong-grid");
  clear(g);
  rows.forEach((r, i) => {
    const src = `data/figures/wrong_${i + 1}.png`;
    const btn = el("button", { type: "button", "aria-label": "Enlarge explanation" }, el("img", { src, alt: `Explanation figure for: ${r.text}`, loading: "lazy" }));
    btn.addEventListener("click", () => openLightbox(src, r.text));
    g.append(el("article", { class: "wrong" }, btn,
      el("div", {}, el("span", { class: `pill ${r.true}`, text: `True: ${r.true}` }), " ", el("span", { class: `pill ${r.model}`, text: `Model: ${r.model} (${Number(r["P(fake)"]).toFixed(3)})` }),
        el("p", { text: r.text }))));
  });
}

function openLightbox(src, alt) {
  $("lightbox-img").src = src;
  $("lightbox-img").alt = `Explanation figure: ${alt}`;
  $("lightbox").showModal();
}
$("lightbox-close").addEventListener("click", () => $("lightbox").close());
$("lightbox").addEventListener("click", (e) => { if (e.target === $("lightbox")) $("lightbox").close(); });

// sample claims from the live test (fills the claim box only; the images aren't public)
function fillSamples(rows) {
  const menu = $("samples-menu");
  if (menu.childElementCount) return;
  for (const r of rows) {
    const b = el("button", { type: "button" }, el("span", { text: r.claim }), el("small", { text: `Fact-check verdict: ${r.fact_check}` }));
    b.addEventListener("click", () => {
      claim.value = r.claim;
      updateCount();
      b.closest("details").open = false;
      claim.focus();
    });
    menu.append(b);
  }
}

// ============================================================ start
(async function start() {
  updateCount();
  updateHistoryCount();
  if (!handleConnectLink()) {
    route();
    const saved = store.get("xfnd.conn", null);
    if (saved?.url && saved?.key) { Object.assign(conn, saved); checkHealth({ quiet: true }); }
    else showStatus("offline", "Model offline", "Run the Colab notebook to connect");
  }
  // the sample claims come from the results data
  fetch("data/live_test.csv").then((r) => r.text()).then(parseCSV).then(fillSamples).catch(() => {});
})();
