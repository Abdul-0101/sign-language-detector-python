FROM python:3.11-slim
WORKDIR /app

# copy everything needed
COPY requirements.txt inference_classifier.py model.p dictionary.txt data.pickle app.py ./

RUN pip install -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
