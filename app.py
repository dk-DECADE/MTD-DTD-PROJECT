"""Streamlit interface: upload -> add Gaussian noise -> predict noise -> restore."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from diffusion_six.core import add_noise, load_image, mse, png_bytes, reconstruct
from diffusion_six.model import load_model, predict_baseline, predict_noise

ROOT = Path(__file__).resolve().parent
NIST_ROOT = ROOT / "models" / "nist"
NIST_LABEL = "NIST digits 0–9"
LOGGER = logging.getLogger("diffusion_six")


@st.cache_resource
def trained_model(checkpoint: Path):
    torch.set_num_threads(2)
    return load_model(checkpoint)


def show_image(column, title: str, image: np.ndarray, caption: str):
    with column:
        st.markdown(f"#### {title}")
        st.image(np.clip(image, 0, 1), width=220, clamp=True)
        st.caption(caption)


def main():
    st.set_page_config(page_title="Diffusion Six · Denoising Lab", page_icon="🧪", layout="wide")
    st.markdown("""<style>
        .block-container { max-width:1180px; padding-top:2.5rem; }
        h1 { letter-spacing:-1.5px; }
        [data-testid="stMetric"] { background:white; border:1px solid #e2e8f0;
          border-radius:12px; padding:18px; }
        .eyebrow { color:#2563eb; font-size:12px; font-weight:700; letter-spacing:2px; }
        </style>""", unsafe_allow_html=True)
    st.markdown('<p class="eyebrow">DATA, TECHNOLOGY & DESIGN / MODEL LAB</p>', unsafe_allow_html=True)
    st.title("A little noise. A clearer six.")
    st.markdown("Compare how linear regression and a CNN reconstruct **64 × 64 handwritten digits** from Gaussian noise.")
    with st.sidebar:
        st.title("Diffusion Six")
        st.caption("01  INPUT  →  02  NOISE  →  03  RESTORE")
        st.divider()
        model_choice = st.selectbox("Validation model", ["Digit 6 · course experiment", NIST_LABEL], key="model_choice")
        nist = model_choice == NIST_LABEL
        model_folder = NIST_ROOT if nist else ROOT
        digit_description = "a digit from 0 to 9" if nist else "a digit 6"
        source = st.radio("Image source", ["Try an example", "Upload your own"])
        clean = None
        example_label = None
        if source == "Try an example":
            if nist:
                example_label = st.selectbox("Held-out digit", range(10), key="nist_example")
                filename = f"digit-{example_label}.png"
            else:
                example = st.selectbox("Held-out example", ["Six 1", "Six 2", "Six 3"])
                example_label = 6
                filename = f"six_{example[-1]}.png"
            example_path = model_folder / "examples" / filename
            if not example_path.exists():
                st.error("The selected model's examples are missing. Check the files included in the deployment.")
                st.stop()
            clean = load_image(example_path)
        else:
            uploaded = st.file_uploader(f"Upload {digit_description}", type=["png", "jpg", "jpeg"],
                                        help="Use a tightly cropped digit, up to 5 MB. It is resized to 64 × 64.")
            dark_strokes = st.checkbox("Dark digit on a light background", value=True)
            if uploaded is not None:
                try:
                    clean = load_image(uploaded.getvalue(), dark_strokes=dark_strokes)
                except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                    st.error("This image could not be read. Try a smaller, valid PNG or JPEG.")
                    st.stop()
        st.divider()
        alpha = st.slider("Signal retention α", 0.15, 0.95, 0.65, 0.05,
                          help="Lower alpha adds more Gaussian noise. This is the model's training range.")
        seed = int(st.number_input("Noise seed", min_value=0, max_value=2**31 - 1, value=42, step=1))
        run = st.button("Reconstruct image", type="primary", use_container_width=True)
        st.caption("Same image + alpha + seed = same noisy observation.")
    if clean is None:
        st.info(f"Upload {digit_description} from the sidebar to begin, or switch to an example.")
        return
    if np.ptp(clean) < 0.05:
        st.warning("This image has very little contrast. Try a tightly cropped handwritten digit.")
    noisy, _ = add_noise(clean, alpha, seed)
    st.divider()
    if run:
        started = time.perf_counter()
        try:
            model, metadata = trained_model(model_folder / "model.pt")
            prediction = reconstruct(noisy, predict_noise(model, noisy, alpha), alpha)
            linear = reconstruct(noisy, predict_baseline(model_folder / "baseline.npz", noisy, alpha), alpha)
        except (OSError, ValueError, RuntimeError, KeyError):
            LOGGER.exception("Model loading or inference failed")
            st.error("The selected model could not be loaded. Check that its model.pt and baseline.npz are included.")
            return
        elapsed = (time.perf_counter() - started) * 1000
        columns = st.columns(4)
        show_image(columns[0], "01 · Original", clean, "Clean reference, resized to 64 × 64")
        show_image(columns[1], "02 · Noisy", noisy, f"α = {alpha:.2f} · seed {seed}")
        show_image(columns[2], "03 · Linear", linear, "Linear noise prediction baseline")
        show_image(columns[3], "04 · CNN", prediction, "Alpha-conditioned convolutional network")
        st.caption("The noisy preview is clipped for display. Inference uses the full, unclipped noisy values.")
        metrics = st.columns(3)
        linear_mse, cnn_mse = mse(clean, linear), mse(clean, prediction)
        metrics[0].metric("Linear reconstruction MSE", f"{linear_mse:.5f}")
        metrics[1].metric("CNN reconstruction MSE", f"{cnn_mse:.5f}")
        improvement = 100 * (1 - cnn_mse / linear_mse) if linear_mse > 0 else None
        metrics[2].metric("CNN improvement vs linear", f"{improvement:+.1f}%" if improvement is not None else "N/A")
        st.caption(f"Prediction completed in {elapsed:.0f} ms (includes first-load overhead). Lower MSE is better; improvement can vary by input.")
        png_name = "reconstructed-digit.png" if nist else "reconstructed-six.png"
        st.download_button("Download CNN reconstruction", png_bytes(prediction), png_name, "image/png")
        report = {"alpha": alpha, "seed": seed, "linear_mse": linear_mse, "cnn_mse": cnn_mse,
                  "dataset": metadata["dataset"], "example_digit": example_label,
                  "model_seed": metadata["seed"], "selected_epoch": metadata["best_epoch"]}
        st.download_button("Download this experiment (JSON)", json.dumps(report, indent=2), "experiment.json", "application/json")
        LOGGER.info("inference alpha=%.2f seed=%d cnn_mse=%.6f elapsed_ms=%.1f", alpha, seed, cnn_mse, elapsed)
    else:
        columns = st.columns(2)
        show_image(columns[0], "01 · Your reference", clean, "Light strokes on a black background")
        show_image(columns[1], "02 · Noisy observation", noisy, f"Signal retention {alpha:.0%} · seed {seed}")
        st.info("Choose a noise level, then click **Reconstruct image** in the sidebar.")
    with st.expander("How the experiment works"):
        st.latex(r"z_\alpha = \sqrt{\alpha}\,x + \sqrt{1-\alpha}\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)")
        st.latex(r"\hat{x} = \frac{z_\alpha - \sqrt{1-\alpha}\,\hat{\epsilon}_\theta(z_\alpha,\alpha)}{\sqrt{\alpha}}")
        st.write("The network estimates the added noise. Subtracting that estimate reconstructs the image; final pixel values are clipped to [0, 1]. MSE compares the reconstruction with your clean, resized reference.")
        if nist:
            st.caption("The NIST model was trained on digits 0–9 cropped and centered on 64 × 64 canvases. Use a tightly cropped clean digit as the reference for this single-step denoising demo.")
        else:
            st.caption("The course model was trained only on digit 6, using source images resized from 8 × 8 to 64 × 64. Select the NIST model to validate other digits.")
    metrics_path = model_folder / "results" / "metrics.json"
    if metrics_path.exists():
        with st.expander("Held-out evaluation"):
            saved = json.loads(metrics_path.read_text(encoding="utf-8"))
            st.write(f"{saved['metadata']['split_counts']['test']} unseen clean images, each evaluated with {saved['metadata']['evaluation_repeats']} noise draws. Checkpoints are selected using validation images only.")
            if nist:
                st.markdown("**Mean reconstruction MSE by digit**")
                st.dataframe(saved["by_digit"], use_container_width=True, hide_index=True)
            st.markdown("**Mean reconstruction MSE by noise level**")
            st.dataframe(saved["by_alpha"], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

    team_photo = ROOT / "teammates.jpeg"
    if team_photo.is_file():
        st.divider()
        st.header("Project team")
        st.image(str(team_photo), width=680)
