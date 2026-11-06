"""Serve the Beyond Words demo on Modal (serverless, scales to zero, persistent URL).

server.py holds the JSON API and loads the models; app.py is the Gradio page
mounted at "/". This file only builds the container image and wraps them.

One-time setup on your machine (the token stays in ~/.modal.toml):
    pip install modal
    modal setup

Deploy or update (the first build bakes the two Whisper checkpoints, about
2.7 GB, into the image so cold starts only load weights):
    modal deploy app/modal_app.py

URL (stable):  https://<workspace>--beyond-words.modal.run
    GET  /api/health      POST /api/text      POST /api/audio      /  (Gradio)
"""

import modal

MODEL_ID = "towfique-elahe/beyond-words-whisper-medium-did-robust"   # deployment variant; the thesis model is the same id without -robust
ASR_ID = "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"
ASR_GEN_FROM = "openai/whisper-medium"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install("torch==2.6.0", extra_index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "gradio>=5.0",
        "transformers>=4.40",
        "librosa",
        "numpy",
        "scipy",
        "scikit-learn==1.9.0",        # must match the version that wrote the joblib
        "joblib",
        "bnunicodenormalizer",
        "fastapi[standard]",
        "python-multipart",
        "huggingface_hub",
    )
    .env({"HF_HOME": "/model-cache", "HF_HUB_DISABLE_TELEMETRY": "1", "OMP_NUM_THREADS": "4",
          "MODEL_ID": MODEL_ID, "ASR_MODEL_ID": ASR_ID, "ASR_GEN_FROM": ASR_GEN_FROM,
          "BW_ROOT": "/root/bw", "LEXICON_DIR": "/root/bw/lexicons",
          "LEXICAL_MODEL": "/root/bw/app/models/lexical_tugstugi_ben10_strict.joblib"})
    .run_commands(
        f"hf download {MODEL_ID}",
        f"hf download {ASR_ID}",
        f"hf download {ASR_GEN_FROM} generation_config.json config.json",
    )
    .add_local_file("text_utils.py", "/root/bw/text_utils.py")
    .add_local_file("lexical_did.py", "/root/bw/lexical_did.py")
    .add_local_file("lexicons/formal_vocab.txt", "/root/bw/lexicons/formal_vocab.txt")
    .add_local_file("lexicons/external.json", "/root/bw/lexicons/external.json")
    .add_local_file("lexicons/indomain_tugstugi_ben10_strict.json",
                    "/root/bw/lexicons/indomain_tugstugi_ben10_strict.json")
    .add_local_dir("app", remote_path="/root/bw/app", ignore=["__pycache__", "*.pyc", "modal_app.py", "embed.html"])
)

app = modal.App("beyond-words-dialect-id", image=image)


@app.function(
    cpu=4.0,
    memory=8192,
    scaledown_window=300,   # keep a warm container 5 min after the last request
    max_containers=1,
    timeout=600,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app(label="beyond-words")
def web():
    import sys

    sys.path.insert(0, "/root/bw/app")
    sys.path.insert(0, "/root/bw")
    import server                      # loads the three models once per container

    return server.mount_gradio(server.api)
