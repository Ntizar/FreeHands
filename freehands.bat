@echo off
REM ──────────────────────────────────────────────────────────────────────
REM  FreeHands · launcher
REM  Uso:  freehands.bat [calibrate|run|doctor] [usuario]
REM  Por defecto:        run  luis
REM ──────────────────────────────────────────────────────────────────────
setlocal ENABLEDELAYEDEXPANSION

cd /d "%~dp0"

set "CMD=%~1"
set "USER=%~2"
if "%CMD%"==""  set "CMD=run"
if "%USER%"=="" set "USER=luis"

REM ── Crear venv si no existe ───────────────────────────────────────────
if not exist ".venv\Scripts\python.exe" (
    echo [FreeHands] Creando entorno virtual ^(.venv^)...
    python -m venv .venv
    if errorlevel 1 (
        echo [FreeHands] ERROR: no se pudo crear el venv. ^¿Tienes Python 3.11+ en el PATH?
        pause & exit /b 1
    )
)

REM ── Activar venv ──────────────────────────────────────────────────────
call ".venv\Scripts\activate.bat"

REM ── Instalar deps si faltan ───────────────────────────────────────────
python -c "import freehands" 2>nul
if errorlevel 1 (
    echo [FreeHands] Instalando dependencias ^(primera vez^)...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e .
    if errorlevel 1 (
        echo [FreeHands] ERROR instalando dependencias.
        pause & exit /b 1
    )
)

REM ── Ejecutar ──────────────────────────────────────────────────────────
if /I "%CMD%"=="doctor" (
    python -m freehands doctor
) else if /I "%CMD%"=="calibrate" (
    echo [FreeHands] Iniciando calibracion para usuario "%USER%"...
    python -m freehands calibrate --user "%USER%"
) else if /I "%CMD%"=="run" (
    echo [FreeHands] Iniciando sistema para usuario "%USER%"...
    python -m freehands run --user "%USER%"
) else (
    echo Uso: %~nx0 [calibrate^|run^|doctor] [usuario]
    exit /b 2
)

set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" pause
exit /b %EXITCODE%
