Self-hosted copy of Tesseract.js (OCR in the browser), so the page needs no third-party CDN.

- tesseract.min.js, worker.min.js: tesseract.js 5.1.1 (Apache-2.0)
- core/*.wasm.js: tesseract.js-core 5.1.1 (Apache-2.0), LSTM builds with and without SIMD
- lang/eng.traineddata.gz: @tesseract.js-data/eng 1.0.0, 4.0.0_best_int (MIT; Tesseract data Apache-2.0)

Loaded only when a visitor uploads an image.
