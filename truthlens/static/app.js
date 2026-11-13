// TruthLens front end. All server text is inserted with textContent (never
// innerHTML), so nothing typed or returned can run as HTML/JS.
"use strict";

const $ = (id) => document.getElementById(id);
const form = $("analyze-form");
const textarea = $("news-text");
const annotated = $("annotated");
const analyzeBtn = $("analyze-btn");
const editBtn = $("edit-btn");
const errorBox = $("form-error");
const verdict = $("verdict");
const maxChars = Number(textarea.getAttribute("maxlength"));
let requestId = 0; // increases with every analysis, so stale answers can be ignored

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

// ---------- helpers
function setText(el, text) { el.textContent = text; }
function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }
function pct(x) {
  if (x > 0.99) return ">99%";
  if (x < 0.01) return "<1%";
  return `${Math.round(x * 100)}%`;
}
const MIN_HIGHLIGHT = 0.2; // only mark words with at least 20% of the strongest push

function updateCount() {
  setText($("char-count"), `${textarea.value.length.toLocaleString()} / ${maxChars.toLocaleString()} characters`);
}

function showError(msg) {
  setText(errorBox, msg);
  errorBox.hidden = !msg;
}

// Position the shaded "uncertain" zone from data attributes (CSP forbids inline styles).
(function placeZone() {
  const zone = $("zone");
  const lo = Number(zone.dataset.lo), hi = Number(zone.dataset.hi);
  zone.style.left = `${lo * 100}%`;
  zone.style.width = `${(hi - lo) * 100}%`;
})();

// ---------- rendering
function renderHighlights(segments) {
  clear(annotated);
  const maxAbs = Math.max(1e-9, ...segments.map((s) => Math.abs(s.weight)));
  const frag = document.createDocumentFragment();
  for (const seg of segments) {
    const strength = Math.abs(seg.weight) / maxAbs;
    if (strength < MIN_HIGHLIGHT) {
      frag.appendChild(document.createTextNode(seg.text));
      continue;
    }
    const span = document.createElement("span");
    span.className = `tok ${seg.weight > 0 ? "f" : "r"}`;
    span.style.setProperty("--w", (0.35 + 0.65 * strength).toFixed(3));
    span.title = `${seg.weight > 0 ? "Towards fake" : "Towards real"} (${seg.weight > 0 ? "+" : ""}${seg.weight.toFixed(3)})`;
    span.textContent = seg.text;
    frag.appendChild(span);
  }
  annotated.appendChild(frag);
}

function renderTerms(listEl, terms) {
  clear(listEl);
  if (!terms.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "None";
    listEl.appendChild(li);
    return;
  }
  const maxAbs = Math.max(...terms.map((t) => Math.abs(t.weight)));
  for (const t of terms) {
    const li = document.createElement("li");
    const word = document.createElement("span");
    word.textContent = t.term;
    const bar = document.createElement("span");
    bar.className = "bar";
    bar.style.width = `${Math.max(6, (Math.abs(t.weight) / maxAbs) * 64)}px`;
    bar.title = t.weight.toFixed(3);
    li.append(word, bar);
    listEl.appendChild(li);
  }
}

const TITLES = { FAKE: "Likely fake", REAL: "Likely real", UNCERTAIN: "Uncertain" };

function renderResult(r) {
  verdict.dataset.state = r.label;
  setText($("verdict-label"), TITLES[r.label] || r.label);
  setText($("verdict-text"), r.verdict);
  $("needle").style.left = `${r.prob_fake * 100}%`;

  const score = $("score");
  clear(score);
  const strong = document.createElement("strong");
  strong.textContent = pct(r.prob_fake);
  score.append("Fake-news score: ", strong, ` (based on ${pct(r.known_word_ratio)} of words the model knows)`);
  score.hidden = false;

  const warnings = $("warnings");
  clear(warnings);
  for (const w of r.warnings) {
    const li = document.createElement("li");
    li.textContent = w;
    warnings.appendChild(li);
  }

  renderTerms($("fake-terms"), r.top_fake_terms);
  renderTerms($("real-terms"), r.top_real_terms);
  renderHighlights(r.highlights);

  $("evidence").hidden = false;
  $("factcheck").hidden = false;
  $("legend").hidden = false;
  textarea.hidden = true;
  annotated.hidden = false;
  editBtn.hidden = false;
  const fcResults = $("factcheck-results");
  if (fcResults) clear(fcResults);
}

function backToEditing() {
  requestId++; // ignore any answer still on its way for the old text
  textarea.hidden = false;
  annotated.hidden = true;
  editBtn.hidden = true;
  $("legend").hidden = true;
  // Old results no longer describe the text being edited, so reset them.
  verdict.dataset.state = "idle";
  setText($("verdict-label"), "No text analyzed yet");
  setText($("verdict-text"), "Paste some news text and choose Analyze text.");
  $("score").hidden = true;
  clear($("warnings"));
  $("evidence").hidden = true;
  $("factcheck").hidden = true;
  textarea.focus();
}

// ---------- API calls
async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let data = {};
  try { data = await res.json(); } catch { /* non-JSON error page */ }
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `Request failed (${res.status}).`);
  return data;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = textarea.value.trim();
  if (text.split(/\s+/).filter(Boolean).length < 3) {
    showError("Enter at least 3 words, ideally a headline and its article.");
    textarea.focus();
    return;
  }
  showError("");
  const myId = ++requestId;
  analyzeBtn.disabled = true;
  setText(analyzeBtn, "Analyzing…");
  try {
    const result = await postJSON("/api/analyze", { text });
    if (myId === requestId) renderResult(result); // drop answers to older requests
  } catch (err) {
    if (myId === requestId) showError(err.message || "Something went wrong. Try again.");
  } finally {
    if (myId === requestId) {
      analyzeBtn.disabled = false;
      setText(analyzeBtn, "Analyze text");
    }
  }
});

editBtn.addEventListener("click", backToEditing);
textarea.addEventListener("input", updateCount);

document.querySelectorAll("[data-example]").forEach((btn) => {
  btn.addEventListener("click", () => {
    backToEditing();
    textarea.value = EXAMPLES[btn.dataset.example] || "";
    updateCount();
    form.requestSubmit();
  });
});

// ---------- fact-check lookup (only present when the server has an API key)
const fcBtn = $("factcheck-btn");
if (fcBtn) {
  fcBtn.addEventListener("click", async () => {
    const out = $("factcheck-results");
    clear(out);
    // Fact-check search works best with one short claim: use the first line or
    // sentence, or the opening of the text if that is too short to search for.
    const all = textarea.value.trim().replace(/\s+/g, " ");
    let firstLine = textarea.value.trim().split(/\n|(?<=[.!?])\s/)[0].trim();
    if (firstLine.length < 20) firstLine = all;
    firstLine = firstLine.slice(0, 200);
    fcBtn.disabled = true;
    setText(fcBtn, "Searching…");
    try {
      const { results } = await postJSON("/api/factcheck", { query: firstLine });
      if (!results.length) {
        const p = document.createElement("p");
        p.className = "muted";
        p.textContent = "No published fact-checks match this text. That doesn't mean it's true: many claims are never checked.";
        out.appendChild(p);
      }
      for (const r of results) {
        const item = document.createElement("div");
        item.className = "fc-item";
        const rating = document.createElement("div");
        rating.className = "fc-rating";
        rating.textContent = `${r.publisher || "Fact-checker"}: ${r.rating || "rated"}`;
        const claim = document.createElement("p");
        claim.className = "fc-claim";
        claim.textContent = r.claim;
        const link = document.createElement("a");
        link.href = r.url; // server only passes http(s) links
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

updateCount();
