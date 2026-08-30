@echo off
rem Launch the merged SHAARP.si + SHAARP.ml GUI (Jupyter notebook).
rem Interpreter is auto-detected (no hardcoded path): SHAARP_PY_PYTHON env var, else the
rem py launcher, else python on PATH. Override by setting SHAARP_PY_PYTHON.
setlocal
cd /d "%~dp0"
set "PYEXE=%SHAARP_PY_PYTHON%"
if not defined PYEXE (
  where py >nul 2>nul && set "PYEXE=py -3"
)
if not defined PYEXE (
  where python >nul 2>nul && set "PYEXE=python"
)
if not defined PYEXE (
  echo Could not find a Python interpreter. Install Python, or set SHAARP_PY_PYTHON to your python.exe.
  pause
  exit /b 1
)
rem --ServerApp.root_dir overrides any machine-wide Jupyter config so this notebook
rem (wherever the project lives) opens correctly.
%PYEXE% -m notebook --ServerApp.root_dir="%~dp0." "notebooks\SHAARP_py_interactive_session.ipynb"
pause
