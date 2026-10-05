# ---- Stage 1: Build dependencies ----
FROM python:3.13-slim AS builder
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir --user -r requirements-api.txt

# ---- Stage 2: Final runtime image ----
FROM python:3.13-slim
WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY src/ ./src/
COPY models/ ./models/

ENV MODEL_PATH=models/xgb_fraud_model_final.pkl
ENV FEATURE_COLUMNS_PATH=models/feature_columns_final.pkl
ENV DECISION_THRESHOLD=0.8
ENV MODEL_VERSION=v1-tuned

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]