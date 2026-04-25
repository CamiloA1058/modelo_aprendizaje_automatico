#!/usr/bin/env python
import sys
import subprocess
import os

# Cambiar al directorio correcto
os.chdir(r"c:\Users\el pepe\OneDrive\Escritorio\modelo_aprendizaje_automatico")

# Ejecutar el modelo principal
result = subprocess.run([sys.executable, "model_k-means_random_forest.py"], 
                       capture_output=False, text=True)

sys.exit(result.returncode)
