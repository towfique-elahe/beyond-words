"""Publish the demo to the Hugging Face Space (free ZeroGPU hardware).

    .venv/bin/python deploy_space.py

Stages app/ with the modules and word lists it needs into one flat folder and
uploads it. Create the Space once with space_hardware="zero-a10g": a free account
is refused plain CPU Gradio Spaces (402) but may host ZeroGPU ones.
"""
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

SPACE = "towfique-elahe/beyond-words-dialect-id"
ROOT = Path(__file__).parent

README = """---
title: Beyond Words Bangla Dialect ID
emoji: 🗣️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Which Bangla dialect is this? Speak or type.
preload_from_hub:
  - towfique-elahe/beyond-words-whisper-medium-did-robust
  - bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium
  - openai/whisper-medium generation_config.json,config.json
---

Live demo of the undergraduate thesis *A Neural Network Approach to Classifying Bangla Accentual
Diversity* (East Delta University, 2026). Public page: https://beyondwords.towfiqueelahe.com.
Code: https://github.com/towfique-elahe/beyond-words.
"""
REQUIREMENTS = """torch==2.9.1
transformers==5.15.1
librosa==0.11.0
soundfile
numpy
scipy
pandas
scikit-learn==1.9.0
joblib
bnunicodenormalizer
fastapi
python-multipart
"""

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp)
    for name in ["app.py", "server.py", "inference.py"]:
        shutil.copy(ROOT / "app" / name, out)
    for name in ["text_utils.py", "lexical_did.py"]:
        shutil.copy(ROOT / name, out)
    shutil.copytree(ROOT / "app" / "models", out / "models", ignore=shutil.ignore_patterns("__pycache__"))
    (out / "lexicons").mkdir()
    for name in ["formal_vocab.txt", "external.json", "indomain_tugstugi_ben10_strict.json"]:
        shutil.copy(ROOT / "lexicons" / name, out / "lexicons")
    (out / "README.md").write_text(README)
    (out / "requirements.txt").write_text(REQUIREMENTS)
    (out / "packages.txt").write_text("ffmpeg\nlibsndfile1\n")
    print(HfApi().upload_folder(repo_id=SPACE, repo_type="space", folder_path=tmp,
                                commit_message="Deploy the demo"))
