---
title: Beyond Words Bangla Dialect ID
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
`deploy_space.py` in the repo root publishes all of this to the free ZeroGPU Space at https://huggingface.co/spaces/towfique-elahe/beyond-words-dialect-id, which the public page calls. `modal_app.py` is the earlier host and still works if a Modal account has credit. `models/` holds the saved lexical classifier
(`lexical_did.py --save-model`) and the fitted weights that combine it with the word lists and the sound (`tune_words_route.py`). The React front end in the separate [beyond-words-web](https://github.com/towfique-elahe/beyond-words-web) repository talks to the same API. The public page is https://beyondwords.towfiqueelahe.com.
A free account is refused plain CPU Gradio Spaces (402 Payment Required) but may host ZeroGPU ones, so create the Space with `space_hardware="zero-a10g"`. ZeroGPU visitors have a daily allowance: about 40 GPU runs per connection when measured, and each call reserves `spaces.GPU(duration=...)` against it, so keep that value small.
