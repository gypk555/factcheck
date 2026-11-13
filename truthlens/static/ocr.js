// Reads the text in an uploaded screenshot, entirely in the browser.
// Uses a self-hosted copy of Tesseract.js (static/vendor/tesseract), loaded only
// the first time someone uploads an image. The image never leaves the device.
"use strict";

window.TruthLensOCR = (() => {
  const BASE = "/static/vendor/tesseract/";
  const MAX_FILE_BYTES = 10 * 1024 * 1024;
  const TYPES = ["image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp"];
  let workerPromise = null;
  let onProgress = null;

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.onload = resolve;
      s.onerror = () => reject(new Error("Couldn't load the text reader. Check your connection and try again."));
      document.head.appendChild(s);
    });
  }

  function getWorker() {
    if (!workerPromise) {
      workerPromise = (async () => {
        await loadScript(BASE + "tesseract.min.js");
        return window.Tesseract.createWorker("eng", 1, {
          workerPath: BASE + "worker.min.js",
          corePath: BASE + "core",
          langPath: BASE + "lang",
          workerBlobURL: false, // a blob: worker would need a looser security policy
          gzip: true,
          logger: (m) => onProgress && onProgress(m),
        });
      })().catch((err) => {
        workerPromise = null; // allow a retry
        throw err;
      });
    }
    return workerPromise;
  }

  // Enlarge narrow screenshots (small text reads badly), but cap the total size
  // at about 6 megapixels so very large or very tall images stay fast.
  const MAX_PIXELS = 6e6;
  async function toCanvas(file) {
    let bitmap;
    try {
      bitmap = await createImageBitmap(file);
    } catch {
      throw new Error("This file couldn't be opened as an image. Try a PNG or JPG screenshot.");
    }
    const { width: w, height: h } = bitmap;
    let scale = w < 900 ? Math.min(2, 1400 / w) : 1;
    if (w * h * scale * scale > MAX_PIXELS) scale = Math.sqrt(MAX_PIXELS / (w * h));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close();
    return canvas;
  }

  // OCR returns one line per visual line. Join lines into paragraphs, keep
  // blank lines as paragraph breaks, and drop lines that are only symbols.
  function tidy(text) {
    return text
      .split(/\n{2,}/)
      .map((para) => para.split("\n").map((l) => l.trim()).filter((l) => /[A-Za-z]{2}/.test(l)).join(" "))
      .filter(Boolean)
      .join("\n\n")
      .replace(/[ \t]{2,}/g, " ")
      .trim();
  }

  function check(file) {
    if (!file) throw new Error("No image selected.");
    if (!TYPES.includes(file.type)) throw new Error("Use a PNG, JPG, WEBP, GIF or BMP image.");
    if (file.size > MAX_FILE_BYTES) throw new Error("That image is over 10 MB. Use a smaller screenshot.");
  }

  async function read(file, progress) {
    check(file);
    onProgress = progress;
    try {
      const canvas = await toCanvas(file);
      const worker = await getWorker();
      const { data } = await worker.recognize(canvas);
      return { text: tidy(data.text || ""), confidence: data.confidence || 0 };
    } finally {
      onProgress = null;
    }
  }

  return { read, check, tidy };
})();
