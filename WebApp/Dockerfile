# syntax=docker/dockerfile:1
FROM python:3.9-bookworm

LABEL maintanier="sudarshan.ng7@gmail.com"

RUN mkdir -p /monreader-webapp

WORKDIR /monreader-webapp

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["flask", "run","--host","0.0.0.0","--port","5000"]
