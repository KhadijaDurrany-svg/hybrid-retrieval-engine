FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# Bundled evaluation corpus so the service answers immediately on boot.
COPY data ./data
ENV HRE_DATA_DIR=/app/data

EXPOSE 8000
CMD ["uvicorn", "hre.service:app", "--host", "0.0.0.0", "--port", "8000"]
