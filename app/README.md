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
or the standard register (Formal) — from the sound alone, using a Whisper-medium
encoder fine-tuned for dialect identification
([model](https://huggingface.co/towfiqueelahe/beyond-words-whisper-medium-did)).

Test macro-F1: 0.984 on a conventional split, 0.919 on a speaker-disjoint split.
Code, data pipeline and thesis: <https://github.com/towfique-elahe/beyond-words>.
