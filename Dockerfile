FROM python:3.11-slim-bullseye

WORKDIR /app

RUN mkdir -p models/trained
RUN mkdir -p mapper

COPY requirements.txt .
COPY models/trained/*.pkl models/trained/
COPY mapper/category_mappings.json mapper/
COPY mapper/reverse_category_mappings.json mapper/
COPY app.py .


RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]