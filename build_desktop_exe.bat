@echo off
rem Build the standalone SHAARP.py desktop app (dist\SHAARP_py\SHAARP_py.exe).
rem One-FOLDER build (starts much faster than one-file; just zip the folder to share).
rem Interpreter auto-detected (no hardcoded path): SHAARP_PY_PYTHON env var, else py, else python.
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
rem ONE cross-platform build definition shared with the release gate and the CI workflows:
rem scripts\build_gui_bundle.py stages the PUBLIC (git-shippable) benchmarks tree, runs PyInstaller,
rem copies README/LICENSE, then bundle-hygiene-checks and smoke-tests the frozen app.
%PYEXE% scripts\build_gui_bundle.py --no-package
if errorlevel 1 (
  echo Build failed.
  pause
  exit /b 1
)
echo.
echo Build done. The app is dist\SHAARP_py\SHAARP_py.exe
pause
