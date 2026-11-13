// TruthLens front end. Server text is always inserted with textContent (never
// innerHTML), so nothing typed, uploaded or returned can run as code.
"use strict";

const $ = (id) => document.getElementById(id);
const setText = (el, t) => { el.textContent = t; };
const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); };
function pct(x) {
  if (x > 0.99) return ">99%";
  if (x < 0.01) return "<1%";
  return `${Math.round(x * 100)}%`;
}

const errorBox = $("form-error");
function showError(msg) { setText(errorBox, msg || ""); errorBox.hidden = !msg; }

async function postJSON(url, body) {
  return handle(await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }));
}
async function postForm(url, form) {
  return handle(await fetch(url, { method: "POST", body: form }));
}
async function handle(res) {
  let data = {};
  try { data = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `Request failed (${res.status}).`);
  return data;
}

let requestId = 0; // every new check gets a number; answers to older ones are ignored

// =============================================================== tabs
const tabs = [$("tab-text"), $("tab-image")];
function selectTab(tab) {
  for (const t of tabs) {
    const on = t === tab;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
    $(t.getAttribute("aria-controls")).hidden = !on;
  }
  resetResult();
  showError("");
}
tabs.forEach((t, i) => {
  t.addEventListener("click", () => selectTab(t));
  t.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    next.focus();
    selectTab(next);
  });
});

// =============================================================== result card
const result = $("result");
const card = $("verdict-card");
const ICONS = { fake: "#i-fake", real: "#i-real", unsure: "#i-unsure", ai: "#i-ai", edit: "#i-edit" };

function resetResult() {
  requestId++;
  result.hidden = true;
  for (const id of ["why-text", "why-image", "next-text", "next-image", "meters"]) $(id).hidden = true;
  clear($("notes"));
  textarea.hidden = false;
  annotated.hidden = true;
  editBtn.hidden = true;
}

function showVerdict({ tone, kind, title, summary, score, caption, notes }) {
  card.dataset.tone = tone;
  $("verdict-icon").setAttribute("href", ICONS[tone] || ICONS.unsure);
  setText($("verdict-kind"), kind);
  setText($("verdict-title"), title);
  setText($("verdict-summary"), summary);
  setText($("gauge-value"), score == null ? "–" : pct(score));
  setText($("gauge-caption"), caption);
  const arc = $("gauge-arc");
  arc.classList.toggle("empty", score == null);
  arc.style.strokeDasharray = `${score == null ? 0 : Math.max(1, score * 100)} 100`;
  const list = $("notes");
  clear(list);
  for (const n of notes || []) {
    const li = document.createElement("li");
    li.textContent = n;
    list.appendChild(li);
  }
  result.hidden = false;
  result.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
}

$("again-btn").addEventListener("click", () => {
  resetResult();
  const imageMode = $("tab-image").getAttribute("aria-selected") === "true";
  if (imageMode) { clearImage(); imageInput.focus(); } else { textarea.value = ""; updateCount(); textarea.focus(); }
  window.scrollTo({ top: 0 });
});

// =============================================================== text mode
const textForm = $("text-form");
const textarea = $("news-text");
const annotated = $("annotated");
const textBtn = $("text-btn");
const editBtn = $("edit-btn");
const maxChars = Number(textarea.getAttribute("maxlength"));
const MIN_HIGHLIGHT = 0.2;

const EXAMPLES = {
  sensational:
    "BREAKING: Doctors STUNNED as simple kitchen trick cures diabetes in 3 days!\n\n" +
    "You won't believe what Big Pharma has been hiding from you. This shocking video is going viral " +
    "and they are desperate to take it down. Watch before it gets deleted! Share this with everyone " +
    "you know, because the mainstream media will never tell you the truth about this miracle cure.",
  wire:
    "City council approves new budget for public transport\n\n" +
    "The city council voted 9 to 2 on Thursday to approve a budget that increases spending on buses " +
    "and commuter rail by 4 percent next year, officials said. The plan, which still needs the " +
    "mayor's signature, includes funding for 30 new electric buses. \"This is a step toward cleaner " +
    "air and shorter waits,\" the council's transport committee chair said in a statement. Opponents " +
    "said the increase would require cuts elsewhere.",
  claim: "Drinking hot water every hour kills the flu virus in your throat.",
};

function updateCount() {
  setText($("char-count"), `${textarea.value.length.toLocaleString()} / ${maxChars.toLocaleString()}`);
}
textarea.addEventListener("input", updateCount);

document.querySelectorAll("[data-example]").forEach((btn) => {
  btn.addEventListener("click", () => {
    btn.closest("details").open = false;
    resetResult();
    textarea.value = EXAMPLES[btn.dataset.example] || "";
    updateCount();
    textForm.requestSubmit();
  });
});

editBtn.addEventListener("click", () => { resetResult(); textarea.focus(); });

function renderHighlights(segments) {
  clear(annotated);
  const maxAbs = Math.max(1e-9, ...segments.map((s) => Math.abs(s.weight)));
  const frag = document.createDocumentFragment();
  for (const seg of segments) {
    const strength = Math.abs(seg.weight) / maxAbs;
    if (strength < MIN_HIGHLIGHT) { frag.appendChild(document.createTextNode(seg.text)); continue; }
    const span = document.createElement("span");
    span.className = `tok ${seg.weight > 0 ? "f" : "r"}`;
    span.style.setProperty("--w", (0.35 + 0.65 * strength).toFixed(3));
    span.title = `${seg.weight > 0 ? "Towards fake" : "Towards real"} (${seg.weight > 0 ? "+" : ""}${seg.weight.toFixed(3)})`;
    span.textContent = seg.text;
    frag.appendChild(span);
  }
  annotated.appendChild(frag);
}

function renderTerms(list, terms) {
  clear(list);
  if (!terms.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "None";
    list.appendChild(li);
    return;
  }
  const maxAbs = Math.max(...terms.map((t) => Math.abs(t.weight)));
  for (const t of terms) {
    const li = document.createElement("li");
    const word = document.createElement("span");
    word.textContent = t.term;
    const bar = document.createElement("span");
    bar.className = "bar";
    bar.style.width = `${Math.max(8, (Math.abs(t.weight) / maxAbs) * 80)}px`;
    li.append(word, bar);
    list.appendChild(li);
  }
}

const TEXT_VERDICT = {
  FAKE: { tone: "fake", title: "Likely fake" },
  REAL: { tone: "real", title: "Likely real" },
  UNCERTAIN: { tone: "unsure", title: "Uncertain" },
};

textForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = textarea.value.trim();
  if (text.split(/\s+/).filter(Boolean).length < 3) {
    showError("Enter at least 3 words, ideally a headline and its article.");
    textarea.focus();
    return;
  }
  showError("");
  const myId = ++requestId;
  textBtn.disabled = true;
  setText(textBtn, "Checking…");
  try {
    const r = await postJSON("/api/analyze", { text });
    if (myId !== requestId) return;
    const v = TEXT_VERDICT[r.label] || TEXT_VERDICT.UNCERTAIN;
    renderHighlights(r.highlights);
    renderTerms($("fake-terms"), r.top_fake_terms);
    renderTerms($("real-terms"), r.top_real_terms);
    textarea.hidden = true;
    annotated.hidden = false;
    editBtn.hidden = false;
    $("why-text").hidden = false;
    $("next-text").hidden = false;
    showVerdict({ tone: v.tone, kind: "News text", title: v.title, summary: r.verdict,
      score: r.prob_fake, caption: "fake score", notes: r.warnings });
    const fc = $("factcheck-results");
    if (fc) clear(fc);
  } catch (err) {
    if (myId === requestId) showError(err.message || "Something went wrong. Try again.");
  } finally {
    if (myId === requestId) { textBtn.disabled = false; setText(textBtn, "Check text"); }
  }
});

// fact-check search (only when the server has an API key)
const fcBtn = $("factcheck-btn");
if (fcBtn) {
  fcBtn.addEventListener("click", async () => {
    const out = $("factcheck-results");
    clear(out);
    const all = textarea.value.trim().replace(/\s+/g, " ");
    const parts = textarea.value.trim().split(/\n|(?<=[.!?])\s/).map((s) => s.trim());
    const query = (parts.find((s) => s.split(/\s+/).length >= 5) || all).slice(0, 200);
    fcBtn.disabled = true;
    setText(fcBtn, "Searching…");
    try {
      const { results } = await postJSON("/api/factcheck", { query });
      if (!results.length) {
        const p = document.createElement("p");
        p.className = "muted";
        p.textContent = "No published fact-checks match. That doesn't make it true: many claims are never checked.";
        out.appendChild(p);
      }
      for (const r of results) {
        const item = document.createElement("div");
        item.className = "fc-item";
        const rating = document.createElement("div");
        rating.className = "fc-rating";
        rating.textContent = `${r.publisher || "Fact-checker"}: ${r.rating || "rated"}`;
        const claim = document.createElement("p");
        claim.textContent = r.claim;
        const link = document.createElement("a");
        link.href = r.url; // the server only passes http(s) links
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = "Read the fact-check";
        item.append(rating, claim, link);
        out.appendChild(item);
      }
    } catch (err) {
      const p = document.createElement("p");
      p.className = "form-error";
      p.textContent = err.message;
      out.appendChild(p);
    } finally {
      fcBtn.disabled = false;
      setText(fcBtn, "Search fact-checks");
    }
  });
}

// =============================================================== image mode
const imageForm = $("image-form");
const imageInput = $("image-input");
const dropzone = $("dropzone");
const preview = $("drop-preview");
const imageBtn = $("image-btn");
const TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"];
const MAX_BYTES = 10 * 1024 * 1024;
let chosen = null;

function clearImage() {
  chosen = null;
  imageInput.value = "";
  if (preview.src) URL.revokeObjectURL(preview.src);
  preview.removeAttribute("src");
  preview.hidden = true;
  $("drop-empty").hidden = false;
  setText($("file-name"), "No image chosen");
  imageBtn.disabled = true;
}

function chooseImage(file) {
  showError("");
  resetResult();
  if (!file) return;
  if (!TYPES.includes(file.type)) { showError("Use a JPG, PNG, WEBP, BMP or GIF image."); return; }
  if (file.size > MAX_BYTES) { showError("That image is over 10 MB. Use a smaller one."); return; }
  chosen = file;
  if (preview.src) URL.revokeObjectURL(preview.src);
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  $("drop-empty").hidden = true;
  setText($("file-name"), `${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`);
  imageBtn.disabled = false;
}

imageInput.addEventListener("change", () => chooseImage(imageInput.files[0]));
dropzone.addEventListener("dragover", (e) => {
  if ([...e.dataTransfer.types].includes("Files")) { e.preventDefault(); dropzone.classList.add("dropping"); }
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dropping"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dropping");
  chooseImage([...e.dataTransfer.files].find((f) => f.type.startsWith("image/")));
});
document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (!item) return;
  e.preventDefault();
  selectTab($("tab-image"));
  chooseImage(item.getAsFile());
});

// image viewer: original / AI heatmap / edit map
const viewBase = $("view-base");
const viewOverlay = $("view-overlay");
const viewButtons = [...document.querySelectorAll("#view-switch [data-view]")];
let maps = { ai: null, edit: null };
const VIEW_CAPTIONS = {
  original: "The image as analysed.",
  ai: "Brighter, warmer areas raised the AI-generated score most.",
  edit: "Error level map: bright areas compress differently from the rest of the image. Edges and fine texture are naturally bright; look for patches that stand out from similar surroundings.",
};
function setView(view) {
  for (const b of viewButtons) b.setAttribute("aria-checked", String(b.dataset.view === view));
  const src = view === "original" ? null : maps[view];
  if (src) { viewOverlay.src = src; viewOverlay.hidden = false; } else { viewOverlay.hidden = true; }
  viewBase.parentElement.classList.toggle("mapped", Boolean(src));
  setText($("view-caption"), VIEW_CAPTIONS[view]);
}
viewButtons.forEach((b) => b.addEventListener("click", () => { if (!b.disabled) setView(b.dataset.view); }));

const IMAGE_VERDICT = {
  AI_GENERATED: { tone: "ai" },
  EDITED: { tone: "edit" },
  REAL: { tone: "real" },
  UNCERTAIN: { tone: "unsure" },
  UNKNOWN: { tone: "unsure" },
};

function setMeter(id, signal) {
  const m = $(id);
  const fill = m.querySelector(".meter-fill");
  const value = m.querySelector(".meter-value");
  if (signal.available) {
    m.classList.remove("off");
    fill.style.width = `${Math.round(signal.prob * 100)}%`;
    setText(value, pct(signal.prob));
  } else {
    m.classList.add("off");
    fill.style.width = "0";
    setText(value, "n/a");
  }
}

imageForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!chosen) { showError("Choose an image first."); return; }
  showError("");
  const myId = ++requestId;
  imageBtn.disabled = true;
  setText(imageBtn, "Checking…");
  try {
    const form = new FormData();
    form.append("file", chosen);
    const r = await postForm("/api/image", form);
    if (myId !== requestId) return;
    maps = { ai: r.ai.map_png, edit: r.edit.map_png };
    viewBase.src = r.preview_png;
    for (const b of viewButtons) b.disabled = b.dataset.view !== "original" && !maps[b.dataset.view];
    setView("original");
    setMeter("meter-ai", r.ai);
    setMeter("meter-edit", r.edit);
    $("meters").hidden = false;
    $("why-image").hidden = false;
    $("next-image").hidden = false;
    const scores = [r.ai, r.edit].filter((s) => s.available).map((s) => s.prob);
    const notes = [...r.warnings];
    for (const s of [r.ai, r.edit]) if (!s.available && s.note) notes.push(s.note);
    showVerdict({
      tone: (IMAGE_VERDICT[r.label] || IMAGE_VERDICT.UNKNOWN).tone,
      kind: `Image, ${r.width} x ${r.height} ${r.format}`,
      title: r.headline, summary: r.summary,
      score: scores.length ? Math.max(...scores) : null,
      caption: "fake score", notes,
    });
  } catch (err) {
    if (myId === requestId) showError(err.message || "Something went wrong. Try again.");
  } finally {
    if (myId === requestId) { imageBtn.disabled = !chosen; setText(imageBtn, "Check image"); }
  }
});

updateCount();
