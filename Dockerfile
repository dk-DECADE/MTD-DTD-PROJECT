FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/appuser

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser diffusion_six ./diffusion_six
COPY --chown=appuser:appuser app.py model.pt baseline.npz teammates.jpeg ./
COPY --chown=appuser:appuser examples ./examples
COPY --chown=appuser:appuser models/nist ./models/nist
COPY --chown=appuser:appuser results/metrics.json ./results/metrics.json
COPY --chown=appuser:appuser .streamlit/config.toml ./.streamlit/config.toml

USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
