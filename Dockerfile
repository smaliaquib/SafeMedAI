FROM python:3.11-slim-bullseye

WORKDIR /app

COPY src/api/ .

COPY models/trained/*.pkl models/trained/

RUN pip install -r requirements.txt

EXPOSE 8000

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]