FROM python:3.13.3-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable
ENV PATH="/app/.venv/bin:$PATH" GAEJA_DATA_DIR=/data
RUN useradd --create-home app && mkdir /data && chown app /data
USER app
ENTRYPOINT ["gaeja"]
CMD ["--help"]
