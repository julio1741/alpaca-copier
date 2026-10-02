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

## Bot 2: ingreso diario (`income_bot.py`, servicio `income`)

Objetivo: ~US$10+/día en promedio con el menor riesgo posible. Regla validada 2016-2023 → 2024-2026
(`research/daily_income_backtest.py`): comprar SPY en la subasta de cierre solo si el día cerró en la
mitad baja de su rango (IBS < 0,5) y vender en la subasta de apertura siguiente. Tamaño ajustado para
que la variación diaria típica sea ~0,3% del capital; freno a la mitad si los últimos 60 resultados
son negativos. Cada mañana concilia fills, anota el P&L real y compara con lo esperado.

Corre en su **propia cuenta paper** (Alpaca permite 3): variables `APCA_*` del servicio `income`,
`RAILWAY_DOCKERFILE_PATH=Dockerfile.income`, volumen en `/data`.

```bash
.venv/bin/python income_bot.py signal                 # señal de hoy
.venv/bin/python income_bot.py run --phase entry --dry-run
.venv/bin/python income_bot.py status
```
