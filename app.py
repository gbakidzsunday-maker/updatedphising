"""
app.py — Phishing URL Detection API

A small FastAPI service that wraps the trained pipeline
(phishing_url_rf_pipeline.pkl) behind an HTTP endpoint.

Run locally / on a server:
    pip install fastapi uvicorn joblib pandas requests scikit-learn
    uvicorn app:app --host 0.0.0.0 --port 8000

Then:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"url": "https://example.com/login"}'

Running from Google Colab:
    Colab doesn't expose ports to the public internet directly. Easiest
    options: (1) use Colab's own port-forwarding preview if you're on a
    paid tier with that feature enabled, or (2) tunnel it, e.g.:

        pip install pyngrok
        from pyngrok import ngrok
        public_url = ngrok.connect(8000)
        print(public_url)

    then run the uvicorn server (e.g. in a background thread or `!uvicorn
    app:app --port 8000 &`) and hit the ngrok URL instead of localhost.
"""

import os
import re
import ipaddress
import urllib.request
from pathlib import Path
from urllib.parse import urlparse
from contextlib import asynccontextmanager

import gdown
import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field




app = FastAPI(
    title="Phishing URL Detection API",
    description="Random Forest-based phishing URL classifier.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve relative to this file, not the process's working directory —
# Render (and most PaaS platforms) don't guarantee cwd == repo root.
MODEL_PATH = Path(__file__).parent / "phishing_url_rf_pipeline.pkl"

# Set ONE of these as a Render env var, depending on where the model lives:
#   GDRIVE_FILE_ID — the file ID from a Google Drive share link
#                     (https://drive.google.com/file/d/<THIS_PART>/view)
#   MODEL_URL       — any other direct-download URL (S3, Hugging Face, etc.)
GDRIVE_FILE_ID = os.environ.get("GDRIVE_FILE_ID")
MODEL_URL = os.environ.get("MODEL_URL")

NUMERIC_FEATURES = [
    'IsDomainIP',
    'URLLength',
    'NoOfQMarkInURL',
    'NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL',
    'SpacialCharRatioInURL',
    'HasObfuscation',
    'NoOfObfuscatedChar',
    'ObfuscationRatio',
    'TLDLength',
    'NoOfSelfRedirect',
    'NoOfURLRedirect'
]
CATEGORICAL_FEATURES = ['TLD']

PERCENT_ENCODING_RE = re.compile(r'%[0-9A-Fa-f]{2}')

# ----------------------------------------------------------------------
# Feature extraction (same logic as test_url.py, kept self-contained here
# so this file can be deployed on its own)
# ----------------------------------------------------------------------

def _strip_scheme_and_www(url: str) -> str:
    stripped = re.sub(r'^https?://', '', url, flags=re.IGNORECASE)
    stripped = re.sub(r'^www\.', '', stripped, flags=re.IGNORECASE)
    return stripped


def _get_tld(netloc: str) -> str:
    host = netloc.split(':')[0]
    parts = host.split('.')
    return parts[-1].lower() if len(parts) > 1 else host.lower()


def _is_ip(host: str) -> bool:
    host = host.split(':')[0]
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _get_redirect_features(url: str, timeout: int = 6):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; URLChecker/1.0)"}
        resp = requests.get(url, allow_redirects=True, timeout=timeout, headers=headers)
        n_redirects = len(resp.history)

        original_host = urlparse(url).netloc.split(':')[0].lower().lstrip('www.')
        self_redirects = 0
        for hop in resp.history:
            hop_host = urlparse(hop.url).netloc.split(':')[0].lower().lstrip('www.')
            if hop_host == original_host:
                self_redirects += 1

        return self_redirects, n_redirects
    except requests.RequestException:
        return 0, 0


def extract_features(url: str, fetch_live: bool = True) -> pd.DataFrame:
    parsed = urlparse(url if re.match(r'^https?://', url) else f"http://{url}")
    netloc = parsed.netloc

    stripped = _strip_scheme_and_www(url)
    special_chars = sum(1 for c in stripped if not c.isalnum())
    url_length = len(url)

    obf_matches = PERCENT_ENCODING_RE.findall(url)
    n_obfuscated = len(obf_matches)

    tld = _get_tld(netloc)

    if fetch_live:
        self_redirects, n_redirects = _get_redirect_features(url)
    else:
        self_redirects, n_redirects = 0, 0

    row = {
        'IsDomainIP': int(_is_ip(netloc)),
        'URLLength': url_length,
        'NoOfQMarkInURL': url.count('?'),
        'NoOfAmpersandInURL': url.count('&'),
        'NoOfOtherSpecialCharsInURL': special_chars,
        'SpacialCharRatioInURL': special_chars / url_length if url_length else 0.0,
        'HasObfuscation': int(n_obfuscated > 0),
        'NoOfObfuscatedChar': n_obfuscated,
        'ObfuscationRatio': n_obfuscated / url_length if url_length else 0.0,
        'TLD': tld,
        'TLDLength': len(tld),
        'NoOfSelfRedirect': self_redirects,
        'NoOfURLRedirect': n_redirects,
    }

    return pd.DataFrame([row], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)


# ----------------------------------------------------------------------
# API schema
# ----------------------------------------------------------------------

class PredictRequest(BaseModel):
    url: str = Field(..., example="https://example.com/login")
    fetch_live: bool = Field(
        True,
        description="If true, actually visits the URL to compute redirect "
                    "features. Set false for a faster, string-only check."
    )
    threshold: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Probability threshold above which a URL is labeled phishing."
    )


class PredictResponse(BaseModel):
    url: str
    prediction: str
    phishing_probability: float
    features: dict


class BatchPredictRequest(BaseModel):
    urls: list[str]
    fetch_live: bool = True
    threshold: float = 0.5


# ----------------------------------------------------------------------
# App + model lifecycle
# ----------------------------------------------------------------------

model_store = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup instead of per-request — this is the expensive part.
    if not MODEL_PATH.exists():
        if GDRIVE_FILE_ID:
            print(f"Downloading model from Google Drive (id={GDRIVE_FILE_ID}) ...")
            gdown.download(id=GDRIVE_FILE_ID, output=str(MODEL_PATH), quiet=False)
        elif MODEL_URL:
            print(f"Downloading model from {MODEL_URL} ...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Either commit "
            f"phishing_url_rf_pipeline.pkl to the repo, or set GDRIVE_FILE_ID "
            f"(for a Google Drive file) or MODEL_URL (for any other direct "
            f"download link) as an env var."
        )

    model_store["model"] = joblib.load(MODEL_PATH)
    yield
    model_store.clear()


app = FastAPI(
    title="Phishing URL Detection API",
    description="Random Forest-based phishing URL classifier.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in model_store}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = model_store.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features = extract_features(req.url, fetch_live=req.fetch_live)
        proba = float(model.predict_proba(features)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not score URL: {e}")

    is_phishing = proba >= req.threshold
    return PredictResponse(
        url=req.url,
        prediction="PHISHING" if is_phishing else "LEGITIMATE",
        phishing_probability=round(proba, 4),
        features=features.to_dict(orient="records")[0],
    )


@app.post("/predict-batch")
def predict_batch(req: BatchPredictRequest):
    model = model_store.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for url in req.urls:
        try:
            features = extract_features(url, fetch_live=req.fetch_live)
            proba = float(model.predict_proba(features)[0, 1])
            is_phishing = proba >= req.threshold
            results.append({
                "url": url,
                "prediction": "PHISHING" if is_phishing else "LEGITIMATE",
                "phishing_probability": round(proba, 4),
            })
        except Exception as e:
            results.append({"url": url, "error": str(e)})

    return {"results": results}
