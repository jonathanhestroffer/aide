FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock  README.md ./

RUN uv sync --frozen --no-install-project

COPY src ./src
COPY tests ./tests

RUN uv sync --frozen --no-editable

ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home appuser

USER appuser

CMD ["/bin/bash"]