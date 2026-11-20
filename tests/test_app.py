from pathlib import Path

from streamlit.testing.v1 import AppTest

from diffusion_six.core import add_noise, load_image, mse, reconstruct
from diffusion_six.model import load_model, predict_baseline, predict_noise

ROOT = Path(__file__).resolve().parents[1]


def test_example_reconstruction_flow():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert "six" in app.title[0].value.lower()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 3
    assert float(app.metric[0].value) >= 0
    assert float(app.metric[1].value) >= 0
    app.slider[0].set_value(0.15).run()
    app.number_input[0].set_value(123).run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 3


def test_upload_mode_waits_for_image():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    app.radio[0].set_value("Upload your own").run()
    assert not app.exception
    assert any("Upload a digit" in info.value for info in app.info)


def test_nist_validation_uses_the_selected_digit_and_model():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    app.selectbox(key="model_choice").select("NIST digits 0–9").run()
    digits = app.selectbox(key="nist_example")
    assert digits.options == [str(digit) for digit in range(10)]
    digits.select(9).run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 3

    nist = ROOT / "models" / "nist"
    clean = load_image(nist / "examples" / "digit-9.png")
    noisy, _ = add_noise(clean, 0.65, seed=42)
    model, _ = load_model(nist / "model.pt")
    cnn = reconstruct(noisy, predict_noise(model, noisy, 0.65), 0.65)
    linear = reconstruct(noisy, predict_baseline(nist / "baseline.npz", noisy, 0.65), 0.65)
    assert app.metric[0].value == f"{mse(clean, linear):.5f}"
    assert app.metric[1].value == f"{mse(clean, cnn):.5f}"
    assert app.dataframe[0].value["digit"].tolist() == list(range(10))

    app.radio[0].set_value("Upload your own").run()
    assert not app.exception
    assert any("Upload a digit from 0 to 9" in info.value for info in app.info)
