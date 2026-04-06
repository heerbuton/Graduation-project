@echo off
setlocal EnableExtensions

REM One-click launcher for backend (Flask) + frontend (Vite)
REM Usage:
REM   - Double click this file in repo root
REM   - Or run: start_all.bat
REM   - Optional dry run: start_all.bat --dry-run

cd /d "%~dp0"
set "ROOT=%CD%"
set "PYTHON_EXE=F:\anaconda\envs\pytorch\python.exe"
set "DRY_RUN="

if /I "%~1"=="--dry-run" set "DRY_RUN=1"

if not exist "%ROOT%\backend\app.py" (
  echo [ERROR] Backend entry not found: "%ROOT%\backend\app.py"
  pause
  exit /b 1
)

if not exist "%ROOT%\frontend\package.json" (
  echo [ERROR] Frontend package.json not found: "%ROOT%\frontend\package.json"
  pause
  exit /b 1
)

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: "%PYTHON_EXE%"
  echo Please update PYTHON_EXE in start_all.bat
  pause
  exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
  echo [ERROR] npm is not available in PATH.
  echo Please install Node.js or fix PATH, then try again.
  pause
  exit /b 1
)

set "BACKEND_CMD=cd /d ""%ROOT%\backend"" && ""%PYTHON_EXE%"" app.py"
set "FRONTEND_CMD=cd /d ""%ROOT%\frontend"" && npm run dev"

if defined DRY_RUN (
  echo [DRY-RUN] start "Backend Flask" cmd /k "%BACKEND_CMD%"
  echo [DRY-RUN] start "Frontend Vite" cmd /k "%FRONTEND_CMD%"
  exit /b 0
)

start "Backend Flask" cmd /k "%BACKEND_CMD%"
start "Frontend Vite" cmd /k "%FRONTEND_CMD%"

echo.
echo [OK] Services are starting...
echo Backend:  http://127.0.0.1:5000
echo Frontend: http://localhost:5173  (or the port shown in the frontend window)
echo.
echo You can close each service by closing its command window.
exit /b 0

