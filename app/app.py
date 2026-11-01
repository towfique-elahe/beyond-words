"""
Beyond Words — live demo: which Bangla dialect is this?

Gradio app for the fine-tuned Whisper-medium dialect classifier (strict-split
checkpoint, 0.92 macro-F1 on unseen speakers). Record or upload speech; the
audio is resampled to 16 kHz, cut into 5 s windows, each window scored by the
encoder-only classifier, and the softmax averaged.
"""

import os

import gradio as gr
import librosa
import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperForAudioClassification

MODEL_ID = os.environ.get("MODEL_ID", "towfiqueelahe/beyond-words-whisper-medium-did")
SR, WIN = 16000, 5 * 16000
DESCRIPTION = """
**Beyond Words** identifies which of nine Bangla dialect classes a speaker is using, from the
sound alone: Barishal, Chattogram, Dhaka, Khulna, Mymensingh, Noakhali, Rajshahi, Sylhet, or the
standard register (*Formal*). The model is a Whisper-medium encoder fine-tuned on 13 hours of
Bangladeshi television, film and online speech. Speak for at least 5 seconds; longer recordings
are scored in 5-second windows and averaged.
"""
NOTES = """
*Honest numbers:* 0.984 macro-F1 on a conventional test split, 0.919 on a split where no speaker
from training appears (the number that matters for a new voice). The model has seen only broadcast
and online media; phone-quality or noisy recordings, and speakers from regions outside the nine
classes, will be mapped to the closest class it knows. Part of the undergraduate thesis
*A Neural Network Approach to Classifying Bangla Accentual Diversity* (East Delta University, 2026).
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
fe = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
model = WhisperForAudioClassification.from_pretrained(MODEL_ID).to(device).eval()
LABELS = [model.config.id2label[i] for i in range(model.config.num_labels)]


def windows(wav):
    n = max(1, int(np.ceil(len(wav) / WIN)))
    for i in range(n):
        w = wav[i * WIN:(i + 1) * WIN]
        yield np.pad(w, (0, WIN - len(w))) if len(w) < WIN else w


@torch.no_grad()
def predict(audio):
    if audio is None:
        return {}, "Record or upload a clip first."
    sr, wav = audio
    wav = wav.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(1)
    if np.abs(wav).max() > 1.0:          # Gradio hands over int16-scaled arrays
        wav = wav / 32768.0
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    if len(wav) < SR:
        return {}, "That clip is shorter than one second; please record a little more."
    probs = []
    for w in windows(wav):
        x = fe(w, sampling_rate=SR, padding="max_length", max_length=WIN, return_tensors="pt").input_features
        probs.append(torch.softmax(model(input_features=x.to(device)).logits, -1)[0].cpu().numpy())
    p = np.mean(probs, 0)
    top = int(p.argmax())
    n_win = len(probs)
    msg = (f"**{LABELS[top]}** ({p[top] * 100:.1f} %), from {len(wav) / SR:.1f} s of audio "
           f"scored in {n_win} window{'s' if n_win > 1 else ''}.")
    return {l: float(v) for l, v in zip(LABELS, p)}, msg


with gr.Blocks(title="Beyond Words: Bangla dialect identification") as demo:
    gr.Markdown("# Beyond Words — Bangla dialect identification")
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Speak or upload (any format)")
            btn = gr.Button("Identify the dialect", variant="primary")
        with gr.Column():
            out_label = gr.Label(num_top_classes=5, label="Dialect probabilities")
            out_text = gr.Markdown()
    btn.click(predict, inputs=audio, outputs=[out_label, out_text])
    audio.stop_recording(predict, inputs=audio, outputs=[out_label, out_text])
    gr.Markdown(NOTES)

if __name__ == "__main__":
    demo.launch()
