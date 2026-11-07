"""
Beyond Words: live demo page (Gradio), which Bangla dialect is this?

Sound: record or upload speech; the fine-tuned Whisper-medium classifier
(strict-split checkpoint) scores it in 5 s windows and averages.
Words: type a Bangla sentence, or let the Tugstugi (Ben-10) recogniser
transcribe the recording; the TF-IDF classifier and the regional word lists
of the thesis give a second opinion, and the two are fused as in the thesis.

The models live in server.py so that the JSON API and this page share them.
On a Hugging Face Space (ZeroGPU) the GPU is held only inside analyse(); the
api_* functions are what the web front end calls (/audio, /text, /health).
Run standalone:  MODEL_ID=<export> python app/app.py
"""

import os
import sys
import time

try:                       # must come before torch on ZeroGPU; absent everywhere else
    import spaces
    gpu = spaces.GPU(duration=30)   # reserved against the visitor's quota; a run needs 1 to 3 s, the first about 17 s
except ImportError:
    def gpu(fn):
        return fn

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server  # noqa: E402  (loads the models once)
from inference import SR, decode_audio  # noqa: E402

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


@gpu
def analyse(wav, text, transcribe):
    return server.analyse_audio(wav, text=text, transcribe=transcribe)


def api_audio(file, text, transcribe):
    if not file:
        raise gr.Error("No audio was received.")
    with open(file, "rb") as f:
        data = f.read()
    if len(data) > server.MAX_BYTES:
        raise gr.Error("The file is larger than 25 MB.")
    try:
        wav = decode_audio(data)
    except Exception as e:  # noqa: BLE001
        raise gr.Error(f"Could not decode the audio: {e}")
    if len(wav) < SR:
        raise gr.Error("The clip is shorter than one second. Please record a little more.")
    t = time.time()
    res = analyse(wav, (text or "").strip(), bool(transcribe))
    res["seconds"] = round(time.time() - t, 2)
    return res


def api_text(text):
    text = (text or "").strip()[:2000]
    if not text:
        raise gr.Error("Please type a Bangla sentence.")
    res = server.lexical.analyse(text, source="typed")
    if "error" in res:
        raise gr.Error(res["error"])
    return res


def api_health():
    return {"ok": True, "version": server.VERSION, "model": server.acoustic.model_id,
            "labels": server.acoustic.labels, "asr": server.ASR_ID if server.asr is not None else None,
            "lexical": server.lexical is not None}


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
        wav = decode_audio(audio)
        if len(wav) < SR:
            raise ValueError("The clip is shorter than one second. Please record a little more.")
        out = analyse(wav, text, bool(transcribe))
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
                                         value=server.asr is not None, visible=server.asr is not None)
                btn = gr.Button("Identify the dialect", variant="primary")
            with gr.Column():
                out_sound = gr.Label(num_top_classes=5, label="From the sound")
                out_words = gr.Label(num_top_classes=5, label="From the words")
                out_fused = gr.Label(num_top_classes=3, label="Fused")
                out_text = gr.Markdown()
        btn.click(run, inputs=[audio, text, transcribe], outputs=[out_sound, out_words, out_fused, out_text])
        gr.Markdown(NOTES)
        with gr.Row(visible=False):      # API for the web front end, not part of this page
            f_in = gr.File(type="filepath")
            t_in = gr.Textbox()
            x_in = gr.Checkbox()
            j_out = gr.JSON()
            b_audio, b_text, b_health = gr.Button(), gr.Button(), gr.Button()
        b_audio.click(api_audio, inputs=[f_in, t_in, x_in], outputs=j_out, api_name="audio")
        b_text.click(api_text, inputs=t_in, outputs=j_out, api_name="text")
        b_health.click(api_health, outputs=j_out, api_name="health")
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
