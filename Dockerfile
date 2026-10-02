FROM python:3.12-slim

# pdftotext (poppler) para leer los PDF de las declaraciones
RUN apt-get update \
 && apt-get install -y --no-install-recommends poppler-utils \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py config.json ./

# Estado y bitácora en el volumen de Railway (montarlo en /data)
ENV DATA_DIR=/data PYTHONUNBUFFERED=1
CMD ["python", "scheduler.py"]
