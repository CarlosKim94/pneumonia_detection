FROM python:3.13.5-slim-bookworm

RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./

RUN uv sync --locked

COPY app.py ./
COPY model/pneumonia_mobilenet_v2.onnx ./model/

EXPOSE 8080

ENTRYPOINT ["/app/.venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]