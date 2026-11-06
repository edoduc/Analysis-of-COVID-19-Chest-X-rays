FROM ghcr.io/astral-sh/uv:0.13.7 AS uv-bin

FROM python:3.12-slim

COPY --from=uv-bin /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY api ./api
COPY src ./src
COPY streamlit_app ./streamlit_app
COPY config.json ./config.json
COPY data/demo_samples ./data/demo_samples
COPY models/xgboost/xgb_best.pkl ./models/xgboost/xgb_best.pkl
COPY reports/figures ./reports/figures

RUN uv sync --frozen --no-dev

EXPOSE 8000 8501

CMD ["uv", "run", "--no-sync", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]