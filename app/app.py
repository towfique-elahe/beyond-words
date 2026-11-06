"""
Beyond Words: live demo page (Gradio), which Bangla dialect is this?

Sound: record or upload speech; the fine-tuned Whisper-medium classifier
(strict-split checkpoint) scores it in 5 s windows and averages.
Words: type a Bangla sentence, or let the Tugstugi (Ben-10) recogniser
transcribe the recording; the TF-IDF classifier and the regional word lists
of the thesis give a second opinion, and the two are fused as in the thesis.

The models live in server.py so that the JSON API and this page share them.
Run standalone:  MODEL_ID=<export> python app/app.py
"""

import os
import sys

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server  # noqa: E402  (loads the models once)
from inference import decode_audio  # noqa: E402

DESCRIPTION = """
**Beyond Words** identifies which of nine Bangla dialect classes a speaker is using:
Barishal, Chattogram, Dhaka, Khulna, Mymensingh, Noakhali, Rajshahi, Sylhet, or the standard
register (*Standard*). The **sound** route is a Whisper-medium encoder fine-tuned on 13 hours of
Bangladeshi television, film and online speech. The **words** route classifies a typed sentence,
or the transcript of your recording, from its vocabulary. Speak for at least 5 seconds.
"""
NOTES = """
*Honest numbers:* the thesis model scores 0.984 macro-F1 on a conventional test split and 0.919 on a
split where no training speaker appears. This page serves a microphone-robust variant trained with
channel augmentation (0.896 on that strict split, but 0.84 instead of 0.39 on clips passed through a
simulated laptop microphone). Recordings are level-normalised before scoring, and read speech in the
standard register will, correctly, come out as Standard. Speakers from regions outside
the nine classes are mapped to the closest class the model knows. Part of the undergraduate thesis
*A Neural Network Approach to Classifying Bangla Accentual Diversity* (East Delta University, 2026).
"""


def _fmt_words(res):
    if not res or "probs" not in res:
        return {}, ""
    lines = [f"**Words say:** {res['words_top']} (TF-IDF: {res['tfidf_top']}; word matching: {res['match']['top']})  "
             f"({res['nonstandard_share'] * 100:.0f} % of {res['n_tokens']} words are outside standard Bangla)"]
    hits = [e for e in res["evidence"] if e["regions"]]
    if hits:
        lines.append("Regional words found: " + ", ".join(
            f"{e['word']} ({e['regions'][0]['region']} {e['regions'][0]['weight']:.2f})" for e in hits[:12]))
    return res["probs"], "\n\n".join(lines)


def run(audio, text, transcribe):
    text = (text or "").strip()
    if audio is None and not text:
        return {}, {}, {}, "Record or upload a clip, or type a sentence."
    if audio is None:
        res = server.lexical.analyse(text) if server.lexical else {"error": "text models not loaded"}
        if "error" in res:
            return {}, {}, {}, res["error"]
        probs, msg = _fmt_words(res)
        return {}, probs, {}, msg
    try:
        out = server.analyse_audio(decode_audio(audio), text=text, transcribe=bool(transcribe))
    except ValueError as e:
        return {}, {}, {}, str(e)
    a = out["acoustic"]
    msg = (f"**Sound says:** {a['top']} ({a['probs'][a['top']] * 100:.1f} %), from {a['duration']:.1f} s "
           f"of audio scored in {a['windows']} window{'s' if a['windows'] > 1 else ''}.")
    if out["transcript"] is not None:
        msg += f"\n\n**Transcript (Tugstugi Ben-10):** {out['transcript'] or '(nothing recognised)'}"
    words_probs, words_msg = _fmt_words(out["words"])
    if words_msg:
        msg += "\n\n" + words_msg
    fused = out["fused"] or {}
    if fused:
        top = max(fused, key=fused.get)
        msg += f"\n\n**Fused ({out['decided_by']}):** {top} ({fused[top] * 100:.1f} %)"
    return a["probs"], words_probs, fused, msg


def build_demo():
    with gr.Blocks(title="Beyond Words: Bangla dialect identification") as demo:
        gr.Markdown("# Beyond Words: Bangla dialect identification")
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Speak or upload (any format)")
                text = gr.Textbox(label="Or type a Bangla sentence (optional)", lines=2,
                                  placeholder="মুই বাড়ি যামু")
                transcribe = gr.Checkbox(label="Transcribe my recording and use its words too",
                                         value=False, visible=server.asr is not None)
                btn = gr.Button("Identify the dialect", variant="primary")
            with gr.Column():
                out_sound = gr.Label(num_top_classes=5, label="From the sound")
                out_words = gr.Label(num_top_classes=5, label="From the words")
                out_fused = gr.Label(num_top_classes=3, label="Fused")
                out_text = gr.Markdown()
        btn.click(run, inputs=[audio, text, transcribe], outputs=[out_sound, out_words, out_fused, out_text])
        gr.Markdown(NOTES)
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
