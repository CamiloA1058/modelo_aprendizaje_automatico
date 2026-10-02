@echo off
setlocal
title Prediccion de Ventas - Dashboard
color 0A

echo.
echo  ================================================
echo   Sistema de Prediccion de Ventas
echo   KMeans + Random Forest
echo  ================================================
echo.

:: -- Carpeta del script ---------------------------------------------------
cd /d "%~dp0"

:: -- Elegir Python: entorno virtual del proyecto si existe ---------------
set "PY=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PY=%~dp0..\.venv\Scripts\python.exe"

"%PY%" --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python no encontrado.
    echo  Por favor instala Python 3.9 o superior desde:
    echo  https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

:: -- Verificar dependencias importandolas en el mismo Python ------------
"%PY%" -c "import flask, flask_cors, pandas, numpy, sklearn, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo  Instalando dependencias, esto puede tardar unos minutos...
    echo.
    "%PY%" -m pip install -r requirements_dashboard.txt
    if errorlevel 1 (
        echo.
        echo  [ERROR] No se pudieron instalar las dependencias.
        echo  Intenta ejecutar manualmente:
        echo  "%PY%" -m pip install -r requirements_dashboard.txt
        pause
        exit /b 1
    )
    echo.
    echo  [OK] Dependencias instaladas correctamente.
    echo.
)

:: -- Abrir navegador despues de 3 segundos -----------------------------
echo  Iniciando servidor...
start "" /b cmd /c "timeout /t 3 >nul && start http://localhost:5000"

:: -- Arrancar Flask -----------------------------------------------------
echo  El dashboard se abrira automaticamente en tu navegador.
echo  Para cerrar el programa cierra esta ventana.
echo.
"%PY%" app.py

pause
