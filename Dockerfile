FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
# hnswlib is compiled from source, so the image needs a C++ compiler while installing
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
COPY ragkit ragkit
COPY data data
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "ragkit.api:app", "--host", "0.0.0.0", "--port", "8000"]
