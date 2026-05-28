@echo off
title Prediccion de Ventas — Dashboard
color 0A

echo.
echo  ================================================
echo   Sistema de Prediccion de Ventas
echo   KMeans + Random Forest
echo  ================================================
echo.

:: ── Verificar Python ─────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python no encontrado.
    echo  Por favor instala Python 3.9 o superior desde:
    echo  https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

:: ── Carpeta del script ───────────────────────────────────────────────────────
cd /d "%~dp0"

:: ── Instalar dependencias si hace falta ─────────────────────────────────────
if not exist ".deps_ok" (
    echo  Instalando dependencias por primera vez...
    echo  Esto puede tardar unos minutos.
    echo.
    pip install flask flask-cors pandas numpy scikit-learn matplotlib >nul 2>&1
    if errorlevel 1 (
        echo  [ERROR] No se pudieron instalar las dependencias.
        echo  Intenta ejecutar manualmente:
        echo  pip install flask flask-cors pandas numpy scikit-learn matplotlib
        pause
        exit /b 1
    )
    echo instalado > .deps_ok
    echo  [OK] Dependencias instaladas correctamente.
    echo.
)

:: ── Abrir navegador después de 3 segundos ────────────────────────────────────
echo  Iniciando servidor...
start "" /b cmd /c "timeout /t 3 >nul && start http://localhost:5000"

:: ── Arrancar Flask ───────────────────────────────────────────────────────────
echo  El dashboard se abrira automaticamente en tu navegador.
echo  Para cerrar el programa cierra esta ventana.
echo.
python app.py

pause
