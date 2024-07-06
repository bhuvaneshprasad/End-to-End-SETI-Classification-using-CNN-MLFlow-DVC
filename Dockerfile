FROM python:3.12-slim

ENV MODEL_URI=artifacts/model_training/model.keras

RUN apt update -y
WORKDIR /app


COPY . /app
RUN mkdir -p /app/logs && chmod -R 777 /app
RUN pip install -r requirements.txt

EXPOSE 7384

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7384"]