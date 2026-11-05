"""Serve the Beyond Words demo on Modal (serverless, scales to zero, persistent URL).

The Gradio app itself is app.py; this file only wraps it for Modal.

One-time setup on your machine (the token stays in ~/.modal.toml):
    pip install modal
    modal setup                      # opens the browser to log in / create the account

Deploy or update (about 5 minutes the first time: builds the image and bakes the
1.2 GB model into it so cold starts only pay for loading, not downloading):
    modal deploy app/modal_app.py

The URL printed at the end is stable:  https://<workspace>--beyond-words.modal.run
"""

import modal

MODEL_ID = "towfique-elahe/beyond-words-whisper-medium-did"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "torch==2.6.0",
        extra_index_url="https://download.pytorch.org/whl/cpu",  # CPU wheel, ~200 MB instead of ~2.5 GB
    )
    .pip_install(
        "gradio>=5.0",
        "transformers>=4.40",
        "librosa",
        "numpy",
        "fastapi[standard]",
        "huggingface_hub",
    )
    .env({"HF_HOME": "/model-cache", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .run_commands(f"hf download {MODEL_ID}")  # public model, no token needed
    .add_local_dir("app", remote_path="/root/demo", ignore=["__pycache__", "*.pyc", "modal_app.py"])
)

app = modal.App("beyond-words-dialect-id", image=image)


@app.function(
    cpu=2.0,
    memory=4096,
    scaledown_window=300,   # keep a warm container 5 min after the last request
    max_containers=1,
    timeout=600,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app(label="beyond-words")
def web():
    import os
    import sys

    os.environ.setdefault("MODEL_ID", MODEL_ID)
    sys.path.insert(0, "/root/demo")

    import gradio as gr
    from fastapi import FastAPI

    import app as demo_module  # app.py: loads the model at import time

    return gr.mount_gradio_app(FastAPI(), demo_module.demo, path="/")
