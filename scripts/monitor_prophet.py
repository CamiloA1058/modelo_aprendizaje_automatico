#!/usr/bin/env python
import time
import pandas as pd
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ventas_forecast.paths import PREDICTIONS

def check_progress():
    temp_file = PREDICTIONS / 'PREDICCIONES_PROPHET_temp.csv'
    final_file = PREDICTIONS / 'PREDICCIONES_PROPHET.csv'

    if temp_file.exists():
        mtime = datetime.fromtimestamp(temp_file.stat().st_mtime)
        temp_results = pd.read_csv(temp_file, encoding='utf-8-sig')
        print(f'[TEMP] {len(temp_results)} códigos procesados hasta {mtime}')
        print(temp_results.tail(3).to_string(index=False))
        return len(temp_results)

    elif final_file.exists():
        mtime = datetime.fromtimestamp(final_file.stat().st_mtime)
        # Solo considerar final si no hay temp y si es reciente (últimas 2 horas)
        if datetime.now() - mtime < pd.Timedelta(hours=2):
            final_results = pd.read_csv(final_file, encoding='utf-8-sig')
            print(f'[FINAL] {len(final_results)} códigos procesados - TERMINADO en {mtime}')
            return len(final_results)
        else:
            print(f'[OLD] Archivo final antiguo de {mtime}, esperando nuevos resultados...')
            return 0

    else:
        print('[INFO] No hay archivos de resultados aún')
        return 0

if __name__ == '__main__':
    while True:
        processed = check_progress()
        if processed >= 1800:
            print('[DONE] Procesamiento completo')
            break
        time.sleep(60)  # Revisar cada minuto