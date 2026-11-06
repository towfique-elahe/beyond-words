/* Beyond Words web front end. Plain JS, no build step.
   Set API_BASE to your Modal deployment (no trailing slash). */
const API_BASE = "https://towfiqueelahe6--beyond-words.modal.run";
const MAX_SECONDS = 20;
const REQUEST_TIMEOUT_MS = 150000;   // transcription on CPU can take 40 s; cold start adds 30 s
const RETRIES = 2;

const $ = (id) => document.getElementById(id);
const statusEl = $("status"), recBtn = $("recBtn"), recTime = $("recTime"), level = $("level");
const fileIn = $("fileIn"), player = $("player"), textIn = $("textIn"), goBtn = $("goBtn");
const clearBtn = $("clearBtn"), results = $("results"), verdict = $("verdict");
const transcriptEl = $("transcript"), evidenceEl = $("evidence"), metaEl = $("meta");

let blob = null, blobName = "recording.webm", serverOk = false;
let rec = null, chunks = [], timer = null, t0 = 0, ctx = null, analyser = null, raf = null, stream = null;

function setStatus(text, cls) { statusEl.textContent = text; statusEl.className = "status " + (cls || ""); }
function enableGo() { goBtn.disabled = !(serverOk && (blob || textIn.value.trim())); }

/* ---------- server health (also wakes a sleeping container) ---------- */
async function health(showWaking) {
  if (showWaking) setStatus("Waking the model server (about 30 seconds on first use)…");
  try {
    const r = await fetchWithTimeout(API_BASE + "/api/health", {}, 90000);
    const j = await r.json();
    serverOk = !!j.ok;
    setStatus("Model server ready" + (j.asr ? " (sound, words and transcription)" : " (sound and words)"), "ok");
    $("transcribe").disabled = !j.asr;
  } catch (e) {
    serverOk = false;
    setStatus("The model server did not answer. Check your connection and reload; it may take 30 seconds to wake.", "err");
  }
  enableGo();
}

function fetchWithTimeout(url, opts, ms) {
  const ctl = new AbortController();
  const id = setTimeout(() => ctl.abort(), ms);
  return fetch(url, { ...opts, signal: ctl.signal }).finally(() => clearTimeout(id));
}

async function postWithRetry(url, opts) {
  let lastErr;
  for (let i = 0; i <= RETRIES; i++) {
    try {
      const r = await fetchWithTimeout(url, opts, REQUEST_TIMEOUT_MS);
      if (r.status >= 500) throw new Error("server error " + r.status);
      return r;
    } catch (e) {
      lastErr = e;
      if (i < RETRIES) { setStatus("Server busy or waking, retrying (" + (i + 1) + "/" + RETRIES + ")…", "warn"); await new Promise(res => setTimeout(res, 3000 * (i + 1))); }
    }
  }
  throw lastErr;
}

/* ---------- recording ---------- */
function pickMime() {
  const c = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus", "audio/ogg"];
  for (const m of c) if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
  return "";
}

async function startRec() {
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    setStatus("This browser cannot record; upload a file instead.", "warn"); return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: true } });
  } catch (e) {
    setStatus("Microphone access was refused. Allow it in the browser's site settings, or upload a file.", "err"); return;
  }
  const mime = pickMime();
  rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  chunks = [];
  rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
  rec.onstop = () => {
    const type = rec.mimeType || mime || "audio/webm";
    blob = new Blob(chunks, { type });
    blobName = "recording." + (type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm");
    player.src = URL.createObjectURL(blob); player.hidden = false;
    stopMeter(); stream.getTracks().forEach(t => t.stop());
    recBtn.textContent = "Record again"; recBtn.classList.remove("rec-on");
    fileIn.value = ""; enableGo();
    const secs = (Date.now() - t0) / 1000;
    if (secs < 3) setStatus("That was " + secs.toFixed(1) + " s. Five seconds or more works much better.", "warn");
  };
  rec.start(250);
  t0 = Date.now(); recBtn.textContent = "Stop"; recBtn.classList.add("rec-on");
  timer = setInterval(() => {
    const s = (Date.now() - t0) / 1000; recTime.textContent = s.toFixed(1) + " s";
    if (s >= MAX_SECONDS) stopRec();
  }, 100);
  startMeter(stream);
}

function stopRec() {
  if (rec && rec.state !== "inactive") rec.stop();
  clearInterval(timer);
}

function startMeter(s) {
  try {
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = ctx.createMediaStreamSource(s); analyser = ctx.createAnalyser(); analyser.fftSize = 1024; src.connect(analyser);
    const buf = new Float32Array(analyser.fftSize);
    const tick = () => { analyser.getFloatTimeDomainData(buf); let sum = 0; for (const v of buf) sum += v * v; level.value = Math.min(1, Math.sqrt(sum / buf.length) * 6); raf = requestAnimationFrame(tick); };
    tick();
  } catch (e) { /* meter is optional */ }
}
function stopMeter() { if (raf) cancelAnimationFrame(raf); if (ctx) ctx.close(); level.value = 0; }

recBtn.addEventListener("click", () => (rec && rec.state === "recording") ? stopRec() : startRec());

fileIn.addEventListener("change", () => {
  const f = fileIn.files && fileIn.files[0];
  if (!f) return;
  if (f.size > 25 * 1024 * 1024) { setStatus("That file is larger than 25 MB.", "err"); fileIn.value = ""; return; }
  blob = f; blobName = f.name; player.src = URL.createObjectURL(f); player.hidden = false; enableGo();
});
textIn.addEventListener("input", enableGo);

/* ---------- request ---------- */
goBtn.addEventListener("click", async () => {
  const text = textIn.value.trim();
  goBtn.disabled = true; goBtn.textContent = "Listening…";
  try {
    let data;
    if (blob) {
      const fd = new FormData();
      fd.append("file", blob, blobName);
      if (text) fd.append("text", text);
      if ($("transcribe").checked) fd.append("transcribe", "1");
      const r = await postWithRetry(API_BASE + "/api/audio", { method: "POST", body: fd });
      data = await r.json();
      if (!r.ok) throw new Error(data.detail || "the server could not score this clip");
      render(data);
    } else {
      const r = await postWithRetry(API_BASE + "/api/text", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text }) });
      data = await r.json();
      if (!r.ok) throw new Error(data.detail || "the server could not read this text");
      render({ acoustic: null, words: data, fused: null, decided_by: "words" });
    }
    setStatus("Done", "ok");
  } catch (e) {
    const msg = (e && e.name === "AbortError") ? "The request timed out. The server may be waking up; try again in a moment." : (e.message || String(e));
    setStatus(msg, "err");
  } finally {
    goBtn.textContent = "Identify the dialect"; enableGo();
  }
});

clearBtn.addEventListener("click", () => {
  blob = null; fileIn.value = ""; textIn.value = ""; player.hidden = true; player.removeAttribute("src");
  results.hidden = true; recBtn.textContent = "Start recording"; recTime.textContent = "0.0 s"; enableGo();
});

/* ---------- rendering ---------- */
function bars(container, probs) {
  container.innerHTML = "";
  if (!probs) { container.innerHTML = '<p class="hint">Nothing to show for this route.</p>'; return; }
  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  const top = entries[0][0];
  for (const [label, p] of entries) {
    const row = document.createElement("div"); row.className = "bar" + (label === top ? " top" : "");
    row.innerHTML = `<span>${label}</span><div class="track"><div class="fill"></div></div><span class="v">${(p * 100).toFixed(1)} %</span>`;
    container.appendChild(row);
    requestAnimationFrame(() => { row.querySelector(".fill").style.width = (p * 100).toFixed(1) + "%"; });
  }
}

function topOf(probs) { const e = Object.entries(probs).sort((a, b) => b[1] - a[1])[0]; return { label: e[0], p: e[1] }; }

function render(d) {
  results.hidden = false;
  const a = d.acoustic, w = d.words, f = d.fused;
  const main = f ? topOf(f) : a ? topOf(a.probs) : topOf(w.tfidf);
  verdict.innerHTML = `${main.label} <span class="pct">${(main.p * 100).toFixed(1)} % · decided by ${d.decided_by}</span>`;
  bars($("bars-sound"), a ? a.probs : null);
  bars($("bars-words"), w && w.tfidf ? w.tfidf : null);
  bars($("bars-fused"), f);
  document.querySelectorAll(".tab").forEach(t => { t.disabled = (t.dataset.tab === "sound" && !a) || (t.dataset.tab === "words" && !(w && w.tfidf)) || (t.dataset.tab === "fused" && !f); });
  showTab(f ? "fused" : a ? "sound" : "words");

  transcriptEl.hidden = !(d.transcript !== null && d.transcript !== undefined);
  if (!transcriptEl.hidden) transcriptEl.innerHTML = `<span class="lbl">Transcript (Tugstugi Ben-10)</span>${escapeHtml(d.transcript) || "<em>nothing recognised</em>"}`;

  const ev = w && w.evidence;
  evidenceEl.hidden = !ev;
  if (ev) {
    const chips = ev.map(e => e.regions.length
      ? `<span class="chip">${escapeHtml(e.word)}<small>${e.regions[0].region} ${e.regions[0].weight.toFixed(2)}</small></span>`
      : `<span class="chip none">${escapeHtml(e.word)}</span>`).join("");
    evidenceEl.innerHTML = `<span class="lbl">Words outside standard Bangla: ${Math.round(w.nonstandard_share * 100)} % of ${w.n_tokens}. Word matching says ${w.match.top}; the text classifier says ${w.tfidf_top}; words verdict ${w.words_top}</span><div class="chips">${chips || '<span class="hint">none</span>'}</div>`;
  }
  const bits = [];
  if (a) bits.push(`${a.duration.toFixed(1)} s of audio in ${a.windows} window${a.windows > 1 ? "s" : ""}`);
  if (d.seconds) bits.push(`${d.seconds.toFixed(1)} s on the server`);
  if (f) bits.push(`fusion weight α = ${d.alpha}`);
  metaEl.textContent = bits.join(" · ");
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

function showTab(name) {
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("on", t.dataset.tab === name));
  document.querySelectorAll(".tabpane").forEach(p => p.classList.toggle("on", p.id === "bars-" + name));
}
document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => { if (!t.disabled) showTab(t.dataset.tab); }));

function escapeHtml(s) { return String(s || "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

window.addEventListener("offline", () => setStatus("You are offline.", "err"));
window.addEventListener("online", () => health(false));
if (!window.MediaRecorder) { recBtn.hidden = true; $("recHint").textContent = "Recording is not supported in this browser; upload a file instead."; } else { recBtn.disabled = false; }
health(true);
