"""
Beyond Words demo server: JSON API (FastAPI) + the Gradio page.

Routes
  GET  /api/health                 -> model names, labels, whether the ASR is loaded
  POST /api/text   {"text": ...}   -> dialect from the words alone (TF-IDF classifier,
                                      regional word matching, per-word evidence)
  POST /api/audio  multipart       -> dialect from the sound; optional fields:
         file       audio in any container ffmpeg reads (wav, webm/opus, m4a, mp3, ogg)
         text       a typed Bangla sentence to fuse with the acoustic result
         transcribe "1" to transcribe the recording with the Tugstugi (Ben-10)
                    recogniser and analyse its words (fused when no text is given)
  /                                 the Gradio page (same models, same code)

Environment
  MODEL_ID       acoustic classifier (HF id or local export)
  ASR_MODEL_ID   Whisper seq2seq checkpoint for transcription
  ASR_GEN_FROM   base model whose generation config supplies the language table
  LEXICAL_MODEL  joblib written by lexical_did.py --save-model
  LEXICON_DIR    folder with formal_vocab.txt, external.json, indomain_*.json
  BW_ROOT        repo root (for text_utils / lexical_did imports)
  ASR_ENABLED    "0" disables transcription (smaller container)

Run locally:
  MODEL_ID=checkpoints/hf_export/beyond-words-whisper-medium-did \
  .venv/bin/uvicorn app.server:api --port 7861
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("BW_ROOT", HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import inference  # noqa: E402
from inference import DISPLAY, SR, AcousticModel  # noqa: E402

VERSION = "2.0"
ALPHA = 0.25                      # fusion weight on the acoustic log-probabilities (thesis, strict split)
MAX_SECONDS = 60
MAX_BYTES = 25 * 1024 * 1024

ASR_ID = os.environ.get("ASR_MODEL_ID", "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium")
ASR_GEN_FROM = os.environ.get("ASR_GEN_FROM", "openai/whisper-medium")
ASR_ENABLED = os.environ.get("ASR_ENABLED", "1") != "0"
LEXICAL_MODEL = Path(os.environ.get("LEXICAL_MODEL", HERE / "models" / "lexical_tugstugi_ben10_strict.joblib"))
LEXICON_DIR = Path(os.environ.get("LEXICON_DIR", ROOT / "lexicons"))


def disp(label: str) -> str:
    return DISPLAY.get(label, label)


# ----------------------------------------------------------------- lexical route
class Lexical:
    """TF-IDF classifier + regional word matching + per-word evidence."""

    def __init__(self):
        import joblib
        from text_utils import load_vocab

        bundle = joblib.load(LEXICAL_MODEL)
        self.pipeline = bundle["pipeline"]
        self.classes = [disp(c) for c in bundle["classes"]]
        self.tau = float(bundle.get("tau") or 0.06)
        self.standard = load_vocab(LEXICON_DIR / "formal_vocab.txt", int(bundle.get("formal_min_count", 3)))
        self.lexicons = []                       # (name, {region: {word: weight}})
        for name in ("external.json", "indomain_tugstugi_ben10_strict.json"):
            path = LEXICON_DIR / name
            if path.exists():
                lex = json.load(open(path, encoding="utf-8"))["lexicon"]
                self.lexicons.append((name.split(".")[0].replace("indomain_tugstugi_ben10_strict", "in-domain"), lex))
        merged: dict[str, dict[str, float]] = {}
        for _, lex in reversed(self.lexicons):   # external wins where both list a word
            for region, words in lex.items():
                merged.setdefault(region, {}).update(words)
        self.merged = merged

    def analyse(self, text: str) -> dict:
        from lexical_did import match_scores
        from text_utils import tokenize

        tokens = tokenize(text or "")
        if not tokens:
            return {"tokens": [], "n_tokens": 0, "error": "no Bangla words found in the text"}
        probs = self.pipeline.predict_proba([" ".join(tokens)])[0]
        tfidf = {c: float(p) for c, p in zip(self.classes, probs)}
        nonstandard, scores = match_scores(tokens, self.standard, self.merged)
        scores = {disp(r): float(v) for r, v in scores.items()}
        best = max(scores, key=scores.get) if scores else None
        match_top = "Standard" if (nonstandard < self.tau or not best or scores[best] == 0) else best
        evidence = []
        for w in dict.fromkeys(t for t in tokens if t not in self.standard):
            hits = []
            for src, lex in self.lexicons:
                for region, words in lex.items():
                    if w in words:
                        hits.append({"region": disp(region), "weight": round(float(words[w]), 3), "source": src})
            hits.sort(key=lambda h: -h["weight"])
            evidence.append({"word": w, "regions": hits[:4]})
        tfidf_top = max(tfidf, key=tfidf.get)
        # Headline for the words route: no word outside standard Bangla means Standard
        # (the thesis's word-matching rule); regional evidence wins over the TF-IDF guess,
        # which was trained on 5 s ASR transcripts and is unreliable on short typed text.
        if nonstandard == 0:
            words_top = "Standard"
        elif match_top != "Standard":
            words_top = match_top
        else:
            words_top = tfidf_top
        return {
            "tokens": tokens, "n_tokens": len(tokens),
            "nonstandard_share": round(float(nonstandard), 3),
            "tfidf": tfidf, "tfidf_top": tfidf_top, "words_top": words_top,
            "match": {"scores": scores, "top": match_top, "tau": self.tau},
            "evidence": evidence,
        }


# ----------------------------------------------------------------- transcription
class ASR:
    def __init__(self):
        import torch
        from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.processor = WhisperProcessor.from_pretrained(ASR_ID)
        self.model = WhisperForConditionalGeneration.from_pretrained(ASR_ID)
        if not getattr(self.model.generation_config, "lang_to_id", None):
            self.model.generation_config = GenerationConfig.from_pretrained(ASR_GEN_FROM)
        self.model.config.forced_decoder_ids = None
        self.model.generation_config.forced_decoder_ids = None
        self.model = self.model.eval()

    def transcribe(self, wav: np.ndarray) -> str:
        feats = self.processor(wav[:30 * SR], sampling_rate=SR, return_tensors="pt").input_features
        with self.torch.no_grad():
            ids = self.model.generate(feats, language="bn", task="transcribe", num_beams=1, max_new_tokens=200)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


# ----------------------------------------------------------------- models (loaded once)
t0 = time.time()
acoustic = AcousticModel()
lexical = Lexical() if LEXICAL_MODEL.exists() else None
asr = ASR() if ASR_ENABLED else None
LOAD_SECONDS = round(time.time() - t0, 1)


def fuse(p_a: dict, p_l: dict, alpha: float = ALPHA) -> dict:
    labels = list(p_a)
    la = np.log(np.array([p_a[l] for l in labels]) + 1e-9)
    ll = np.log(np.array([p_l.get(l, 1e-9) for l in labels]) + 1e-9)
    z = alpha * la + (1 - alpha) * ll
    z = np.exp(z - z.max()); z /= z.sum()
    return {l: float(v) for l, v in zip(labels, z)}


def analyse_audio(wav: np.ndarray, text: str = "", transcribe: bool = False) -> dict:
    if len(wav) < SR:
        raise ValueError("the clip is shorter than one second; please record a little more")
    wav = wav[:MAX_SECONDS * SR]
    pred = acoustic.score(wav)
    out = {"acoustic": {"probs": pred.probs, "top": pred.top, "duration": round(pred.duration, 2),
                        "windows": pred.n_windows, "model": acoustic.model_id},
           "transcript": None, "transcript_words": None, "words": None, "fused": None,
           "alpha": ALPHA, "decided_by": "sound"}
    if transcribe:
        if asr is None:
            out["transcript_error"] = "transcription is switched off on this server"
        else:
            out["transcript"] = asr.transcribe(inference.prepare(wav))
            if lexical is not None and out["transcript"]:
                out["transcript_words"] = lexical.analyse(out["transcript"])
    source = None
    if text and text.strip() and lexical is not None:
        out["words"] = lexical.analyse(text); source = "typed text"
    elif out["transcript_words"] and "tfidf" in out["transcript_words"]:
        out["words"] = out["transcript_words"]; source = "transcript"
    if out["words"] and "tfidf" in out["words"]:
        out["fused"] = fuse(pred.probs, out["words"]["tfidf"])
        out["decided_by"] = f"sound + words ({source})"
    return out


# ----------------------------------------------------------------- API
api = FastAPI(title="Beyond Words API", version=VERSION)
api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"],
                   allow_headers=["*"], max_age=3600)


@api.get("/api/health")
def health():
    return {"ok": True, "version": VERSION, "model": acoustic.model_id, "labels": acoustic.labels,
            "asr": ASR_ID if asr is not None else None, "lexical": lexical is not None,
            "load_seconds": LOAD_SECONDS}


@api.post("/api/text")
async def text_route(payload: dict):
    if lexical is None:
        raise HTTPException(503, "the text models are not loaded on this server")
    text = (payload or {}).get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(400, "send {\"text\": \"a Bangla sentence\"}")
    res = lexical.analyse(text[:2000])
    if "error" in res:
        raise HTTPException(422, res["error"])
    return res


@api.post("/api/audio")
async def audio_route(file: UploadFile = File(...), text: str = Form(""), transcribe: str = Form("0")):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "the file is larger than 25 MB")
    try:
        wav = inference.decode_audio(data)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"could not decode the audio ({file.content_type or 'unknown type'}): {e}")
    try:
        t = time.time()
        res = analyse_audio(wav, text=text, transcribe=transcribe in ("1", "true", "yes", "on"))
        res["seconds"] = round(time.time() - t, 2)
        return res
    except ValueError as e:
        raise HTTPException(422, str(e))


def mount_gradio(app_: FastAPI):
    import gradio as gr
    from app import build_demo
    return gr.mount_gradio_app(app_, build_demo(), path="/")
