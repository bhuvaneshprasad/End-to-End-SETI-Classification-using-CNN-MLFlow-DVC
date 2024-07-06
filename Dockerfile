FROM python:3.12-slim

ENV MODEL_URI=artifacts/model_training/model.keras

RUN apt update -y
WORKDIR /app


COPY . /app
RUN mkdir -p /app/logs && chmod -R 777 /app
RUN pip install -r requirements.txt

EXPOSE 8501
EXPOSE 7384

ENV PYTHONIOENCODING=UTF-8

CMD streamlit run streamlit_app/app.py --server.port 8501 & uvicorn app:app --port 7384