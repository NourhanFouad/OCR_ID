@echo off
setlocal
cd /d "%~dp0"
set "PADDLE_PDX_CACHE_HOME=%CD%\models\pretrained\paddlex_cache"
set "PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=False"
set "PYTHONUTF8=1"
if exist "%~dp0..\.venv\Scripts\python.exe" (
  "%~dp0..\.venv\Scripts\python.exe" -m streamlit run app.py
) else (
  python -m streamlit run app.py
)
