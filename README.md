# TruthLens

An explainable fake news detector. Paste a headline or article and TruthLens says whether it *reads like* fake news, highlights the exact words that pushed it each way, and (optionally) searches published fact-checks for the same claim.

College mini project. Classic machine learning (TF-IDF + Logistic Regression), trained on the WELFake dataset, with an honest cross-dataset test on LIAR2 (PolitiFact claims).

## Results at a glance

| Test set | Accuracy | Macro F1 | Always-guess-the-commonest-label baseline |
|---|---|---|---|
| WELFake, unseen articles (9,617) | **94.1%** | 0.940 | 53.7% |
| WELFake, unseen headlines only (9,497) | 84.7% | 0.845 | |
| LIAR2, PolitiFact claims (1,565) | 53.2% | 0.527 | 61.5% |

With the app's "Uncertain" zone (scores between 35% and 65%), it gives a verdict on 95.4% of unseen articles and is right 95.9% of the time when it does.

The model is strong on news written like its training data, and **worse than a trivial guess** on short political claims from another source. It learns writing style, not facts. Full write-up: [`docs/report.md`](docs/report.md).

## Run it locally

Needs [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you).

```bash
uv sync                                           # install the app's dependencies
uv run uvicorn truthlens.app:app --reload         # open http://127.0.0.1:8000
```

The trained model is already in `models/`, so no training is needed to run the app.

Optional fact-check panel: copy `.env.example` to `.env`, add a free [Google Fact Check Tools API](https://developers.google.com/fact-check/tools/api) key, then start with `--env-file .env`.

## Retrain and regenerate the report figures

```bash
uv run --group train python scripts/train.py      # about 2 minutes on a laptop, no GPU
```

This downloads the datasets into `data/raw/` (not committed), trains and compares Naive Bayes, Logistic Regression and Linear SVM, then writes:

- `models/model.json`, `models/weights.npz`: the demo model
- `reports/metrics.json`: every number in the report
- `reports/figures/*.png`: confusion matrices, model comparison, top words

## Tests

```bash
uv run pytest -q
```

## Deploy on Render (free)

1. Push this repository to GitHub.
2. In Render: **New → Blueprint**, pick the repository. `render.yaml` sets everything up (build: `pip install -r requirements.txt`, start: `uvicorn`, health check `/healthz`).
3. Optional: in the service's **Environment** tab, set `GOOGLE_FACTCHECK_API_KEY`.

Rate limits are per visitor IP. Proxy headers are trusted only on Render (which sets `RENDER=true`) or when `TRUST_PROXY=1`; otherwise the connection's own address is used. On Render the IP is the last `X-Forwarded-For` entry. If every visitor seems to share one limit there, set `CLIENT_IP_HEADER` (for example `cf-connecting-ip`) in the Environment tab. Fact-check look-ups also have a site-wide cap (`FACTCHECK_GLOBAL_PER_MIN`, default 120) to protect the API key's quota.

Free instances sleep after inactivity; the first visit after a pause takes about a minute. Open the site a few minutes before a demo.

If you change dependencies, regenerate `requirements.txt`:
`uv export --no-dev --no-hashes --no-emit-project -o requirements.txt`

## How it works

```
text ─► clean_text() ─► TF-IDF (words + word pairs) ─► Logistic Regression ─► P(fake)
          │                                                  │
          └ removes source give-aways                        └ per-word contribution = tfidf × weight
            ("(Reuters)", "Featured image via",                (exact explanation, shown as highlights)
             outlet names, URLs, @handles)
```

- `truthlens/text.py`: cleaning shared by training and the app
- `truthlens/model.py`: loading, prediction and explanation (pure NumPy, no pickle)
- `truthlens/factcheck.py`: optional Google Fact Check Tools lookup
- `truthlens/app.py`: FastAPI server and security settings
- `truthlens/templates`, `truthlens/static`: the web page
- `scripts/train.py`: data preparation, training, evaluation and figures

## Security

- The model is stored as JSON + NumPy arrays and loaded with `allow_pickle=False`, so loading it can't run code.
- Strict Content-Security-Policy (no inline scripts or styles, no third-party hosts). The page inserts all text with `textContent`.
- Input limits (20,000 characters, 64 KB body, `Content-Length` required so chunked uploads can't bypass the cap), and short validation errors that don't echo input.
- Linear-time cleaning regexes (tested against ReDoS inputs). Prediction runs in a worker thread, so one long request can't block other visitors.
- Per-visitor rate limits, applied before the body is read. The API key stays on the server and is never returned in errors.
- Only `http(s)` links from the fact-check API reach the page.
- API docs are off in production (set `TRUTHLENS_API_DOCS=1` to enable locally).

## Data and credits

- **WELFake**: Verma et al., *WELFake: Word Embedding Over Linguistic Features for Fake News Detection*, IEEE Transactions on Computational Social Systems, 2021. Copy used: [davanstrien/WELFake](https://huggingface.co/datasets/davanstrien/WELFake). Its dataset card says label 1 = real, but the data shows 1 = fake (21,266 of 21,282 Reuters articles are label 0), so this project treats 1 as fake.
- **LIAR2**: [chengxuphd/liar2](https://huggingface.co/datasets/chengxuphd/liar2) (Apache-2.0), an extension of LIAR (Wang, 2017).
- Fonts: Atkinson Hyperlegible and Source Serif 4, both SIL Open Font License, self-hosted.
