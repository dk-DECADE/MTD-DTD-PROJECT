"""Recompute held-out results from saved weights without retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_six.data import clean_splits, evaluation_batch
from diffusion_six.baseline import LinearNoisePredictor
from diffusion_six.model import load_model
from train import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, metadata = load_model(ROOT / "model.pt")
    splits = clean_splits(metadata["seed"])
    test_images = splits["test"]["images"]
    baseline = LinearNoisePredictor.load(ROOT / "baseline.npz")
    metrics, _ = evaluate(model, baseline, evaluation_batch(test_images))
    by_alpha = []
    for alpha in (0.15, 0.35, 0.55, 0.75, 0.95):
        result, _ = evaluate(model, baseline, evaluation_batch(test_images, alpha=alpha))
        by_alpha.append({"alpha": alpha, **result})
    print(json.dumps({"test": metrics, "by_alpha": by_alpha}, indent=2))
    if args.verify:
        saved = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
        for filename, field in (("model.pt", "model_sha256"), ("baseline.npz", "baseline_sha256")):
            if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != saved[field]:
                raise SystemExit(f"{filename} hash does not match recorded evaluation.")
        if metadata["split_ids"] != {key: value["ids"] for key, value in splits.items()}:
            raise SystemExit("Dataset splits do not match saved metadata.")
        for actual, expected in [(metrics, saved["test"]), *zip(by_alpha, saved["by_alpha"])]:
            for key, value in actual.items():
                if not np.isclose(value, expected[key], rtol=1e-4, atol=1e-6):
                    raise SystemExit(f"Evaluation mismatch for {key}: {value} vs {expected[key]}")
        print("Verified: artifact hashes, image splits and all recorded metrics.")


if __name__ == "__main__":
    main()
