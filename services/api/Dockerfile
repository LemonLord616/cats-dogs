FROM python:3.11-slim

WORKDIR /srv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install uv

COPY pyproject.toml uv.lock README.md ./

RUN mkdir -p src/cats_dogs && touch src/cats_dogs/__init__.py

RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --frozen --no-dev --extra api -o requirements.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0"

COPY src/ ./src/
COPY services/api/ ./services/api/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps ".[api]"

WORKDIR /srv/services/api

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
