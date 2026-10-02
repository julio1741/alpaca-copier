FROM python:3.12-slim

# pdftotext (poppler) para leer los PDF de las declaraciones (copiador)
RUN apt-get update \
 && apt-get install -y --no-install-recommends poppler-utils \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py config.json ./

# Una imagen para los dos bots. Cada servicio de Railway elige el suyo con BOT_CMD:
#   copier -> (por defecto) scheduler.py
#   income -> BOT_CMD="income_bot.py serve"
# Estado y bitácora en el volumen de Railway (montarlo en /data)
ENV DATA_DIR=/data PYTHONUNBUFFERED=1
CMD ["sh", "-c", "exec python ${BOT_CMD:-scheduler.py}"]
