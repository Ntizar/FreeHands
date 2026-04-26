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

REM ── Comprobar Python ──────────────────────────────────────────
where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo [FreeHands] ERROR: Python no esta en el PATH.
    echo            Instala Python 3.11+ desde https://www.python.org/downloads/
    echo            y marca "Add Python to PATH" durante la instalacion.
    echo.
    pause & exit /b 1
)

REM ── Crear venv si no existe ─────────────────────────────────
if not exist ".venv\Scripts\python.exe" (
    echo [FreeHands] Creando entorno virtual ^(.venv^)...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo [FreeHands] ERROR: no se pudo crear el venv.
        pause & exit /b 1
    )
)

REM ── Activar venv ──────────────────────────────────────────────────────
call ".venv\Scripts\activate.bat"

REM ── Instalar deps si faltan ───────────────────────────────────────────
python -c "import freehands" 2>nul
if errorlevel 1 (
    echo [FreeHands] Instalando dependencias ^(primera vez, puede tardar varios minutos^)...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [FreeHands] ERROR instalando dependencias.
        pause & exit /b 1
    )
    python -m pip install -e .
    if errorlevel 1 (
        echo.
        echo [FreeHands] ERROR instalando el paquete freehands en modo editable.
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
    echo [FreeHands] Iniciando sistema para usuario "%USER%"...    echo            ^(Si no tienes perfil, se abrira la calibracion primero.^)    python -m freehands run --user "%USER%"
) else (
    echo Uso: %~nx0 [calibrate^|run^|doctor] [usuario]
    exit /b 2
)

set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo [FreeHands] El proceso termino con codigo %EXITCODE%.
    pause
)
exit /b %EXITCODE%
