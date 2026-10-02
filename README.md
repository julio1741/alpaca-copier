# Copiador de operaciones de congresistas → Alpaca (paper)

Lee las declaraciones de transacciones (PTR) del Clerk de la Cámara de Representantes y las replica en
la cuenta **paper** de Alpaca según las reglas de `config.json`.

## A quién copia y con qué reglas

Pelosi (CA11), Cleo Fields (LA06) y Robert Latta (OH05), elegidos con `research/backtest.py`
(selección con 2024-25, validación con 2026). Reglas clave en `config.json`:

- monto por compra según el rango declarado (`size_tiers_pct`), tope 20% por ticker;
- calls → se compra la acción; venta declarada → se cierra la posición;
- efectivo ocioso estacionado en SPY (`park_cash_in`, colchón `cash_buffer_pct`);
- sin plata para una compra → se vende la posición copiada más antigua (`rotate_when_short`).

## Horario (scheduler.py)

| Fase  | Cuándo (hora NY)        | Qué hace                                         |
|-------|-------------------------|--------------------------------------------------|
| open  | apertura + 1 min        | busca declaraciones nuevas y ejecuta **compras** |
| close | cierre − 15 min         | vuelve a buscar y ejecuta **ventas**             |

Usa el calendario de Alpaca: respeta feriados, cierres tempranos y el cambio de horario.

## Local

```bash
.venv/bin/python copier.py status
.venv/bin/python copier.py run --phase all --dry-run
.venv/bin/python copier.py simulate 20035143 2026
```

## Railway

1. Crear un servicio desde este repo (usa `Dockerfile` y `railway.json`).
2. Variables del servicio: `APCA_API_BASE_URL`, `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` (las de `.env`).
3. Agregar un **Volume** montado en `/data` (estado y bitácora; sin él se pierden en cada deploy y el bot
   volvería a fijar la línea base).
4. Una sola réplica. Revisar logs: debe decir `Próxima fase: open ...`.

`.env` no se sube (está en `.gitignore` y `.dockerignore`).
