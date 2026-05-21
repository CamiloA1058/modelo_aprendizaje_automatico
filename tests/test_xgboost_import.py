try:
    from xgboost import XGBRegressor
    print("XGBoost importado correctamente")
    print("Versión:", XGBRegressor.__version__ if hasattr(XGBRegressor, '__version__') else "No version")
except ImportError as e:
    print("Error importando XGBoost:", e)
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost'])
    try:
        from xgboost import XGBRegressor
        print("XGBoost instalado e importado correctamente")
    except ImportError as e2:
        print("Error después de instalación:", e2)