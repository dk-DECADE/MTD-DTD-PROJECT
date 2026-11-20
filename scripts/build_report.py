"""Generate an eight-page English PDF and editable Markdown from real evidence."""
from __future__ import annotations

import json
import textwrap
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def main():
    saved = read_json(ROOT / "results" / "metrics.json")
    metadata, result = saved["metadata"], saved["test"]
    observations = read_json(ROOT / "results" / "observations.json", {})
    container = read_json(ROOT / "results" / "container-check.json", {})
    junit = ROOT / "results" / "tests.xml"
    if junit.exists():
        suites = ET.parse(junit).getroot().findall("testsuite")
        count = sum(int(s.attrib.get("tests", 0)) for s in suites)
        failures = sum(int(s.attrib.get("failures", 0)) + int(s.attrib.get("errors", 0)) for s in suites)
        test_text = f"Local pytest evidence: {count} cases, {failures} failures/errors (results/tests.xml)."
    else:
        test_text = "Tests are implemented. Run pytest to generate local execution evidence."
    docker_text = (f"Local Docker evidence: HTTP health response '{container.get('http_health')}', "
                   f"Docker health status '{container.get('docker_health')}'. "
                   f"Image identifier: {container.get('image_id', 'see container-check.json')}."
                   if container else "Docker configuration is supplied; execution evidence has not yet been recorded.")
    sections = [
        ("1. Problem and Objective", [
            "Diffusion Six: from proof of concept to a tested, packaged denoising application. Data, Technology and Design, Term 7, 2026. Team names and student IDs: complete before submission.",
            "Objective: reconstruct a handwritten digit 6 after adding known Gaussian noise; compare a linear noise predictor with a small CNN, then make the application reproducible, tested, containerized and ready for deployment.",
            "Scope change: the assignment examples use 8 x 8 pixels. At the user's request, all model inputs, noise samples, reconstructions and downloaded PNGs in this implementation are 64 x 64. Source images are still 8 x 8; bilinear resizing occurs before independent 64 x 64 noise is added. Interpolation does not supply new high-resolution information.",
            "Forward process: z = sqrt(alpha) x + sqrt(1-alpha) epsilon, with epsilon drawn from N(0, I). Reconstruction: x_hat = (z - sqrt(1-alpha) epsilon_hat) / sqrt(alpha), clipped to [0,1]. Noisy observations remain unclipped during inference. MSE is calculated against the preprocessed clean reference.",
            "This demonstrates single-step diffusion-inspired denoising, rather than a full multi-step DDPM. The user uploads a clean reference and the application synthesizes the noisy observation. The model is trained only on digit 6 and alpha in [0.15, 0.95].",
            f"Evidence is generated from committed weights and evaluation artifacts. Training seed: {metadata['seed']}; image split: {metadata['split_counts']['train']} train, {metadata['split_counts']['validation']} validation, {metadata['split_counts']['test']} test. Public deployment and hosted CI evidence must be completed using the team's actual repository/account.",
        ], None),
        ("2. Baseline and CNN Improvement", [
            "The baseline independently fits epsilon_j = w_j z_j + v_j alpha + b_j at each pixel using regularized least squares (L2 penalty 1 on signal and alpha terms; intercept unpenalized). It has no neighboring-pixel information or nonlinear features. This deliberately simple baseline avoids a dense 4096 x 4096 regression matrix.",
            "The CNN uses noisy image and broadcast alpha channels. Four 3 x 3 convolutions have channels 2->16->16->16->1, dilations 1,2,4,1, SiLU activations between layers, and a 17 x 17 receptive field. Adam (learning rate 0.001) minimizes noise-prediction MSE; validation reconstruction MSE selects the saved checkpoint.",
            f"Training: {metadata['epochs']} epochs, {metadata['samples_per_epoch']} fresh noisy examples per epoch, batch size 32; baseline uses {metadata['baseline_samples']} draws. Both use the same clean training partition and noise law, but training budgets differ. Best checkpoint: epoch {metadata['best_epoch']}. On 296 held-out noisy observations: linear MSE {result['linear_mse']:.6f}; CNN MSE {result['cnn_mse']:.6f}; relative reduction {result['cnn_improvement_percent']:.2f}%.",
            "Local spatial structure generally helps the CNN remove independent noise. This is an empirical comparison for one seed and split, not a proof of universal superiority. At alpha=0.55 the CNN is slightly worse than the baseline; the complete results are retained rather than selecting only favorable conditions.",
        ], "results/comparison.png"),
        ("3. Project Structure and Reproducibility", [
            "train.py creates model.pt, baseline.npz, metrics.json, history.json, example PNGs and comparison plots. app.py loads the saved CNN and linear baseline. diffusion_six/ contains shared preprocessing, noise/reconstruction operations, model architecture, regression and data generation. tests/ holds numerical and application tests; scripts/ holds evaluation, observation and report generation.",
            "Data: sklearn's bundled optical-recognition digits, label 6 only; pixel values divided by 16 then bilinearly resized. There are 181 source images. Original image IDs are split before generating any noise, preventing variants of a test image from entering training. Validation selects the checkpoint; test images never choose model parameters. No runtime dataset download is required.",
            "Python, NumPy and PyTorch seeds are fixed and deterministic algorithms enabled. Metadata stores exact image IDs, hyperparameters, versions and device. The delivered checkpoint was trained using CUDA; inference runs on CPU. Results may differ slightly when retraining on different hardware, software or device backends.",
            "Setup: create a Python 3.10 virtual environment; python -m pip install -r requirements-dev.txt. Run: python -m streamlit run app.py. Retrain: python train.py (CPU) or python train.py --device cuda. Recompute and verify all recorded metrics: python scripts/evaluate.py --verify.",
            "Direct package versions are pinned, with a CPU PyTorch index for deployment. Transitive dependencies and the Python base-image tag may evolve; fully locked dependency hashes and a pinned image digest are future improvements. Artifact hashes ensure that stored evaluation results refer to the delivered weights.",
            "CNN SHA256: " + saved["model_sha256"],
            "Linear SHA256: " + saved["baseline_sha256"],
        ], None),
        ("4. Automated Testing", [
            test_text,
            "Run: python -m pytest -q --junitxml=results/tests.xml. Numerical tests verify native 64 x 64 shapes, deterministic seeds, unclipped noisy values, known MSE, alpha bounds, finite inputs, and exact recovery when the true synthetic noise is supplied. This inverse-property test detects broken forward/reverse equations rather than merely repeating implementation details.",
            "Input tests cover resizing, aspect-ratio padding, black-on-white polarity, transparent backgrounds, corrupt bytes and the 64 x 64 PNG download. Image-level split tests detect leakage. Checkpoint tests require explicit 64 x 64 metadata and reject stale 8 x 8 checkpoints. Saved CNN and linear predictors are exercised at multiple noise levels for finite, bounded reconstructions.",
            "Streamlit AppTest simulates loading the application, clicking reconstruction, changing alpha/seed and switching to upload mode without a file. It checks that the interface returns metric values and runs without exceptions. Core upload decoding is tested separately; actual browser file uploads are not covered by these simulated UI tests.",
            "Local passing tests are evidence of implemented behavior, not evidence that GitHub-hosted CI has run. Container startup/health checks are also separate from model correctness. Evaluation verification checks both weights' SHA256, original image IDs and aggregate/per-alpha metrics with a small numerical tolerance.",
            "Remaining tests to add: true browser uploads and downloads, reverse-proxy/WebSocket behavior, resource limits under concurrent sessions, and deliberate failures such as unavailable weights on an already running service.",
        ], None),
        ("5. Docker Packaging", [
            "The Dockerfile uses Python 3.10 slim, installs pinned direct dependencies and CPU PyTorch, copies the trained checkpoint/baseline and examples, then starts Streamlit on port 8501. The runtime user is non-root. A .dockerignore excludes local environments, editor state, logs, credentials and repository metadata.",
            "Build: docker build -t diffusion-six .\nRun: docker run --rm -p 8501:8501 --name diffusion-six diffusion-six\nOpen: http://localhost:8501",
            "The application binds to 0.0.0.0 inside the container. A HEALTHCHECK periodically calls http://localhost:8501/_stcore/health. Inspect health with docker inspect --format='{{.State.Health.Status}}' diffusion-six; inspect startup errors with docker logs diffusion-six.",
            docker_text,
            "The container packages inference, not training, and contains no user uploads or secrets. Host-specific CUDA is unnecessary. Model tensors load on CPU using a state_dict-compatible checkpoint and weights_only=True. Retraining requires regenerating the checkpoint/evidence, rebuilding the image and restarting the application.",
            "A successful local image build and health check demonstrate packaging and local startup. They do not establish production uptime, security of a future hosting environment, or functional browser uploads. Those require separate deployment evidence.",
        ], None),
        ("6. Automated GitHub Checks", [
            "The workflow .github/workflows/test.yml triggers on pushes, pull requests and workflow_dispatch. It grants read-only repository-content permissions. The tests job checks out the project, configures Python 3.10, installs requirements-dev.txt, runs pytest, verifies saved evaluation results and uploads the JUnit XML artifact even when tests fail.",
            "An independent Docker job builds the image, starts a container and polls the HTTP health endpoint with a bounded retry loop. Failures print container logs. Job timeouts prevent checks from hanging indefinitely. Tests and packaging checks run independently on clean Ubuntu runners.",
            "Lifecycle: commit -> push -> clean dependency installation -> numerical/application tests -> artifact verification -> PASS/FAIL. The Docker job adds build/startup feedback. These checks make common regressions visible before new code is accepted.",
            "Hosted execution status: NOT YET VERIFIED. A local Git repository has been initialized, but no actual GitHub remote or account is configured for this project. Local pytest and Docker output cannot be described as a successful GitHub Actions run.",
            "Before submission: create/connect the real repository; commit project files including weights and examples; push main; wait for both jobs to succeed; save the Actions run URL and a genuine success screenshot. Add this evidence and the repository URL to the report and submission checklist.",
            "Reproducibility improvement: pin third-party Actions to commit SHAs and use fully locked Python dependencies. These are future hardening steps, separate from the implemented educational workflow.",
        ], None),
        ("7. Deployment", [
            "The Streamlit application supports: choose an example or upload a PNG/JPEG -> preserve aspect ratio and normalize polarity -> set alpha and seed -> synthesize Gaussian noise -> predict noise with both models -> view reconstructed images, MSE and relative change -> download a native 64 x 64 PNG or experiment JSON.",
            "The UI explains the training range, input assumptions, reference-based MSE and the difference between clipped previews and unclipped inference. Invalid/corrupt uploads show a readable error; missing/incompatible weights show retraining instructions. Uploaded files remain in session memory rather than being saved to disk.",
            docker_text,
            "Public deployment status: PENDING. Localhost is a local access address, not a public URL. To complete deployment, push to the team's GitHub repository and select that repository, main branch, app.py and Python 3.10 in Streamlit Community Cloud. Alternatively deploy the Docker image to a server and configure external routing, WebSockets and HTTPS.",
            "Before submission, record the actual public URL and test from another device/browser. Upload a cropped handwritten 6, reconstruct at several noise levels and download the result. Preserve a real screenshot of the deployed application and logs/results showing it worked. No cloud success or external availability is claimed by this report.",
            "Operational checks should cover cold starts, repeated sessions, input decoding failures, memory use, low-alpha quality and server restarts. Synthetic local observations are useful diagnostics but do not substitute for monitoring on the chosen hosting platform.",
        ], None),
        ("8. Results and Reflection", [
            f"Held-out mean reconstruction MSE improved from {result['linear_mse']:.6f} (linear) to {result['cnn_mse']:.6f} (CNN), a {result['cnn_improvement_percent']:.2f}% reduction. Fixed-noise evaluation uses 37 unseen source images with 8 draws per image. The table in metrics.json includes all tested alpha levels; at alpha=0.55 the CNN is slightly worse, and low alpha remains difficult.",
            f"Local observation sweep: {observations.get('case_count', 'not yet recorded')} example/alpha/seed combinations; finite results: {observations.get('all_finite', 'not recorded')}. CNN MSE spans {observations.get('cnn_mse_min', 0):.5f} to {observations.get('cnn_mse_max', 0):.5f}. Local p95 operation time is {observations.get('latency_ms_p95', 0):.2f} ms; it is machine-specific and excludes web latency and model startup.",
            "Limitations: low-information upsampled sources, only digit 6, only one train/test split and training seed, different linear/CNN training budgets, and synthetic Gaussian noise. The 64 x 64 benchmark is easier in spatial smoothness than genuinely detailed 64 x 64 handwriting. Results do not establish robustness to other digits, photographs or real noise.",
            "Next version: collect genuine high-resolution handwriting, repeat training across seeds with uncertainty estimates, tune using validation only, improve low-alpha calibration, add a stronger spatial linear baseline, and run real-browser upload/download tests. Complete hosted CI and public deployment evidence before claiming the full lifecycle is finished.",
            "References: PyTorch serialization (docs.pytorch.org/docs/stable/notes/serialization); Streamlit testing (docs.streamlit.io/develop/concepts/app-testing); Docker deployment (docs.streamlit.io/deploy/tutorials/docker); Community Cloud deployment (docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy); dataset documentation via sklearn.datasets.load_digits.",
        ], "results/evaluation.png"),
    ]
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    markdown = ["# Diffusion Six - 64 x 64 Project Report", "", "Team names / student IDs: complete before submission.", ""]
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    with PdfPages(reports / "report.pdf") as pdf:
        for page, (title, paragraphs, image_path) in enumerate(sections, 1):
            fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
            fig.text(0.08, 0.955, "DIFFUSION SIX / DATA, TECHNOLOGY & DESIGN", fontsize=9, color="#2563eb", weight="bold")
            fig.text(0.08, 0.91, title, fontsize=20, color="#172033", weight="bold")
            y = 0.865
            font_size = 9.7 if image_path else 11
            line_height = 0.015 if image_path else 0.019
            width = 100 if image_path else 88
            for paragraph in paragraphs:
                lines = []
                for source_line in paragraph.splitlines():
                    lines.extend(textwrap.wrap(source_line, width=width, break_long_words=False) or [""])
                fig.text(0.08, y, "\n".join(lines), fontsize=font_size, color="#263449", va="top", linespacing=1.35)
                y -= len(lines) * line_height + 0.02
            if image_path:
                bottom = 0.09
                height = max(0.15, y - bottom - 0.015)
                ax = fig.add_axes([0.08, bottom, 0.84, height])
                ax.imshow(plt.imread(ROOT / image_path))
                ax.axis("off")
            elif y < 0.08:
                raise RuntimeError(f"Report text overflow on page {page}: {y}")
            fig.text(0.08, 0.035, "64 x 64 implementation | generated from local evidence | hosted CI / public deployment pending", fontsize=7, color="#64748b")
            fig.text(0.90, 0.035, f"{page} / 8", fontsize=9, color="#64748b")
            pdf.savefig(fig)
            plt.close(fig)
            markdown.extend([f"## {title}", "", *[p + "\n" for p in paragraphs]])
            if image_path:
                markdown.extend([f"![{title}](../{image_path})", ""])
    (reports / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    print("Generated reports/report.pdf (8 pages) and reports/report.md")


if __name__ == "__main__":
    main()
