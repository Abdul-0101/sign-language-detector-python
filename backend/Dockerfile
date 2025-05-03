FROM python:3.11-slim

# ── Install system libs that OpenCV needs ──
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Copy your Python code + model/data ──
COPY requirements.txt inference_classifier.py model.p dictionary.txt data.pickle app.py ./

# ── Install Python deps ──
RUN pip install -r requirements.txt

# ── Start FastAPI ──
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
