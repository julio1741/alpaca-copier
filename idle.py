#!/usr/bin/env python3
"""Servicio en pausa: se usa como BOT_CMD para apagar un bot sin borrar su servicio ni su volumen.
Para reactivarlo, devolver BOT_CMD a su comando original (p. ej. scheduler.py para el copiador)."""
import time

print("Servicio en pausa (BOT_CMD=idle.py). No opera.", flush=True)
while True:
    time.sleep(86400)
