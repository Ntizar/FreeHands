@echo off
REM ----------------------------------------------------------------------
REM  FreeHands - launcher unico (doble-click).
REM
REM  Sin argumentos              -> menu interactivo.
REM  FreeHands.bat run           -> arranca con usuario "Ntizar".
REM  FreeHands.bat calibrate     -> mirada + gestos para "Ntizar".
REM  FreeHands.bat gestures      -> solo gestos para "Ntizar".
REM  FreeHands.bat <cmd> <user>  -> usa el usuario indicado.
REM
REM  En la primera ejecucion crea el venv e instala dependencias.
REM  Si el usuario no tiene perfil, abre la calibracion automaticamente.
REM ----------------------------------------------------------------------
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "DEFAULT_USER=Ntizar"
set "CMD=%~1"
set "USER=%~2"
if "%USER%"=="" set "USER=%DEFAULT_USER%"
set "LOGDIR=%~dp0logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%" >nul 2>nul
set "LOGFILE=%LOGDIR%\FreeHands-last.log"
set "MENU_MODE=0"
if "%CMD%"=="" set "MENU_MODE=1"

REM -- Comprobar Python --------------------------------------------------
where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo [FreeHands] ERROR: Python no esta en el PATH.
    echo            Instala Python 3.11+ desde https://www.python.org/downloads/
    echo            y marca "Add Python to PATH" durante la instalacion.
    echo.
    pause & exit /b 1
)

REM -- Crear venv si no existe ------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [FreeHands] Creando entorno virtual ^(.venv^)...
    python -m venv .venv
    if errorlevel 1 (
        echo [FreeHands] ERROR: no se pudo crear el venv.
        pause & exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

REM -- Instalar dependencias si freehands no esta importable -------------
python -c "import freehands" 2>nul
if errorlevel 1 (
    echo [FreeHands] Instalando dependencias ^(primera vez, puede tardar varios minutos^)...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [FreeHands] ERROR instalando dependencias de requirements.txt
        pause & exit /b 1
    )
    python -m pip install -e .
    if errorlevel 1 (
        echo.
        echo [FreeHands] ERROR instalando el paquete freehands en modo editable.
        pause & exit /b 1
    )
)

REM -- Menu interactivo si no hay comando -------------------------------
if "%CMD%"=="" (
    echo.
    echo  ------------------------------------------
    echo   FreeHands - usuario: %USER%
    echo  ------------------------------------------
    echo   1^) Ejecutar sistema
    echo   2^) Calibrar todo (mirada + gestos^)
    echo   3^) Recalibrar solo mirada
    echo   4^) Recalibrar solo gestos
    echo   5^) Doctor (camara, micro, dependencias^)
    echo   6^) Cambiar usuario
    echo   7^) Salir
    echo.
    set /p CHOICE="  Elige una opcion [1]: "
    if "!CHOICE!"=="" set "CHOICE=1"
    if "!CHOICE!"=="1" set "CMD=run"
    if "!CHOICE!"=="2" set "CMD=calibrate"
    if "!CHOICE!"=="3" set "CMD=gaze"
    if "!CHOICE!"=="4" set "CMD=gestures"
    if "!CHOICE!"=="5" set "CMD=doctor"
    if "!CHOICE!"=="6" (
        set /p USER="  Nuevo usuario: "
        if "!USER!"=="" set "USER=%DEFAULT_USER%"
        set "CMD=run"
    )
    if "!CHOICE!"=="7" exit /b 0
)

REM -- Ejecucion ---------------------------------------------------------
echo [FreeHands] Log: "%LOGFILE%"
echo FreeHands %DATE% %TIME% > "%LOGFILE%"
echo CMD=%CMD% USER=%USER%>> "%LOGFILE%"

if /I "%CMD%"=="doctor" (
    echo [FreeHands] Ejecutando doctor...
    python -m freehands doctor >> "%LOGFILE%" 2>&1
) else if /I "%CMD%"=="calibrate" (
    echo [FreeHands] Calibrando mirada + gestos para usuario "%USER%"...
    python -m freehands calibrate --user "%USER%" >> "%LOGFILE%" 2>&1
) else if /I "%CMD%"=="gaze" (
    echo [FreeHands] Recalibrando solo mirada para usuario "%USER%"...
    python -m freehands calibrate-gaze --user "%USER%" >> "%LOGFILE%" 2>&1
) else if /I "%CMD%"=="gestures" (
    echo [FreeHands] Recalibrando solo gestos para usuario "%USER%"...
    python -m freehands calibrate-gestures --user "%USER%" >> "%LOGFILE%" 2>&1
) else if /I "%CMD%"=="run" (
    echo [FreeHands] Iniciando sistema para usuario "%USER%"...
    echo            ^(Si no tienes perfil, se abrira la calibracion primero.^)
    python -m freehands run --user "%USER%" >> "%LOGFILE%" 2>&1
) else (
    echo Uso:
    echo   FreeHands.bat                 ^(menu interactivo^)
    echo   FreeHands.bat run [usuario]
    echo   FreeHands.bat calibrate [usuario]
    echo   FreeHands.bat gaze [usuario]
    echo   FreeHands.bat gestures [usuario]
    echo   FreeHands.bat doctor
    pause & exit /b 1
)

set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo [FreeHands] El proceso termino con codigo %EXITCODE%.
    echo [FreeHands] Ultimas lineas del log:
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -Path '%LOGFILE%' -Tail 40"
    pause
    exit /b %EXITCODE%
)
if "%MENU_MODE%"=="1" (
    echo.
    echo [FreeHands] Proceso finalizado. Log guardado en:
    echo %LOGFILE%
    echo.
    echo [FreeHands] Ultimas lineas del log:
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -Path '%LOGFILE%' -Tail 30"
    pause
)
exit /b %EXITCODE%
