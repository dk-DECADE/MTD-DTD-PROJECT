"""Record variation across examples, noise levels and seeds, including latency."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_six.core import add_noise, load_image, mse, reconstruct
from diffusion_six.model import load_model, predict_baseline, predict_noise


def main():
    torch.set_num_threads(2)
    model, _ = load_model(ROOT / "model.pt")
    rows = []
    for path in sorted((ROOT / "examples").glob("*.png")):
        clean = load_image(path)
        for alpha in (0.15, 0.55, 0.95):
            for seed in (0, 42, 123):
                started = time.perf_counter()
                noisy, _ = add_noise(clean, alpha, seed)
                prediction = reconstruct(noisy, predict_noise(model, noisy, alpha), alpha)
                linear = reconstruct(noisy, predict_baseline(ROOT / "baseline.npz", noisy, alpha), alpha)
                rows.append({"image": path.name, "alpha": alpha, "seed": seed,
                             "cnn_mse": mse(clean, prediction), "linear_mse": mse(clean, linear),
                             "latency_ms": 1000 * (time.perf_counter() - started),
                             "finite": bool(np.isfinite(prediction).all())})
    result = {"case_count": len(rows), "all_finite": all(row["finite"] for row in rows),
              "cnn_mse_min": min(row["cnn_mse"] for row in rows),
              "cnn_mse_max": max(row["cnn_mse"] for row in rows),
              "latency_ms_p95": float(np.percentile([row["latency_ms"] for row in rows], 95)),
              "cases": rows}
    (ROOT / "results" / "observations.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
