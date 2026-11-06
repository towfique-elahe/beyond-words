---
title: Beyond Words — Bangla Dialect ID
emoji: 🗣️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
short_description: Which Bangla dialect is this? 9-class ID from speech.
---

# Beyond Words — Bangla dialect identification

Record or upload Bangla speech and the app predicts one of nine classes —
Barishal, Chattogram, Dhaka, Khulna, Mymensingh, Noakhali, Rajshahi, Sylhet,
or the standard register (Standard) from the sound alone, using a Whisper-medium
encoder fine-tuned for dialect identification
([model](https://huggingface.co/towfique-elahe/beyond-words-whisper-medium-did)).

Test macro-F1: 0.984 on a conventional split, 0.919 on a speaker-disjoint split.
Code, data pipeline and thesis: <https://github.com/towfique-elahe/beyond-words>.

This folder is the demo. `server.py` loads the three models once (acoustic classifier,
Tugstugi Ben-10 recogniser, TF-IDF + word lists) and exposes `GET /api/health`,
`POST /api/text` and `POST /api/audio`; `app.py` is the Gradio page mounted at `/`;
`inference.py` holds the shared audio preprocessing (decode any container, trim
silence, normalise level to the training RMS, 5 s windows with 2.5 s hop);
`modal_app.py` builds the container and deploys to https://towfiqueelahe6--beyond-words.modal.run
(`modal deploy app/modal_app.py`). `models/` holds the saved lexical classifier
(`lexical_did.py --save-model`). The static page in `web/` talks to the same API.
The front matter above also lets the folder run as a Hugging Face Space (PRO plan needed for Gradio on free hardware).
