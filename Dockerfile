FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

RUN echo "Downloading model for Run ID: ${RUN_ID}"

ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=""

CMD echo "Serving model for Run ID: ${RUN_ID}" && python src/train.py