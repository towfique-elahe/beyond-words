"""
Export a fine-tuned Whisper dialect classifier to Hugging Face format.

Rebuilds the WhisperClassifier of finetune_xlsr.py from a {tag}_best.pt
checkpoint and saves the inner WhisperForAudioClassification (config carries
max_source_positions=250 for 5 s inputs, id2label for the nine classes) plus
the feature extractor, ready for `hf upload` and for the Gradio demo.

Usage:
    python export_hf_model.py --checkpoint whisper_medium_best.pt --out hf_export/beyond-words-whisper-medium-did
"""

import argparse
import json
from pathlib import Path

import torch

from predict import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model, labels = load_model(args.checkpoint, "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    inner = model.model
    inner.config.id2label = dict(enumerate(labels))
    inner.config.label2id = {l: i for i, l in enumerate(labels)}
    inner.config.max_source_positions = inner.encoder.embed_positions.num_embeddings
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inner.save_pretrained(out, safe_serialization=True)
    model.fe.save_pretrained(out)
    json.dump({"source_checkpoint": Path(args.checkpoint).name, "base_model": ckpt.get("model"),
               "family": ckpt.get("family"), "best_epoch": ckpt.get("epoch"),
               "val_macroF1": ckpt.get("val_macroF1"), "labels": labels,
               "input": "first 5 s of audio at 16 kHz; 250 positional rows (500 log-mel frames)"},
              open(out / "training_summary.json", "w"), indent=2)
    print(f"saved {sum(p.numel() for p in inner.parameters()) / 1e6:.1f}M params -> {out}")


if __name__ == "__main__":
    main()
