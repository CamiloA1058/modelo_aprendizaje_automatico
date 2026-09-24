"""
Utilidades compartidas entre los tests del pipeline de producción
(train_kmeans_rf_prod.SalesForecastModel).
"""

import numpy as np
import pandas as pd


def synthetic_history(n_products=6, n_months=24, drop_months=None):
    """Historial mensual con tendencia creciente: los meses de prueba
    (finales de cada producto) tienen valores mayores que los de
    entrenamiento (iniciales), así la mediana del conjunto completo
    difiere de la mediana del entrenamiento.

    drop_months : dict opcional {producto: iterable de posiciones 0-based
        (dentro de la secuencia de fechas de ESE producto) a omitir}. Sirve
        para simular productos con meses sin ninguna venta (sin fila en el
        CSV crudo), como ocurre en los datos reales, y así poder probar el
        completado de calendario (tests/test_monthly_calendar.py). Por
        defecto no se omite ningún mes, así que el comportamiento existente
        de esta función no cambia."""
    rows = []
    fechas = pd.date_range("2024-01-01", periods=n_months, freq="MS")
    rng = np.random.default_rng(0)
    drop_months = drop_months or {}
    for p in range(n_products):
        meses_omitidos = set(drop_months.get(p, ()))
        for i, fecha in enumerate(fechas):
            if i in meses_omitidos:
                continue
            total = 1000.0 * (i + 1) * (p + 1) + rng.integers(0, 500)
            rows.append({
                "CODIGO": p,
                "DESCRIPCION": f"PRODUCTO {p}",
                "fecha": fecha,
                "VENTAS": float(i + 1 + p),
                "TOTAL_VENDIDO": total,
                "PRECIO_PROMEDIO": total / (i + 1 + p),
                "FRECUENCIA": i + 1,
                "ANIO": fecha.year,
                "MES": fecha.month,
            })
    return pd.DataFrame(rows)
