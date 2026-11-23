FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /usr/local/bin/uv

ENV UV_NO_CACHE=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --only-group servicio --no-install-project

COPY src ./src
COPY models ./models

RUN useradd --system --create-home apr
USER apr

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["bash", "-c", "uvicorn src.aguaapr.main:app --host 0.0.0.0 --port 8000 & streamlit run src/aguaapr/ui.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false & wait -n"]
