@echo off
REM ----------------------------------------------------------------------
REM  FreeHands - single launcher (double-click).
REM
REM  No arguments                -> interactive menu.
REM  FreeHands.bat run           -> starts FreeHands for user "Ntizar".
REM  FreeHands.bat calibrate     -> gaze + gestures for "Ntizar".
REM  FreeHands.bat gestures      -> gestures only for "Ntizar".
REM  FreeHands.bat camera        -> selects camera for "Ntizar".
REM  FreeHands.bat repair        -> reinstalls runtime dependencies.
REM  FreeHands.bat <cmd> <user>  -> uses the selected user.
REM
REM  On first run it creates the venv and installs dependencies.
REM  If the user has no profile, calibration opens automatically.
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
set "LOGFILE=%LOGDIR%\FreeHands-%RANDOM%-%RANDOM%.log"
set "LASTLOG=%LOGDIR%\FreeHands-last.log"
set "MENU_MODE=0"
if "%CMD%"=="" set "MENU_MODE=1"

REM -- Check Python ------------------------------------------------------
where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo [FreeHands] ERROR: Python is not on PATH.
    echo            Install Python 3.11+ from https://www.python.org/downloads/
    echo            and check "Add Python to PATH" during installation.
    echo.
    pause & exit /b 1
)

REM -- Create venv if missing -------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [FreeHands] Creating virtual environment ^(.venv^)...
    python -m venv .venv
    if errorlevel 1 (
        echo [FreeHands] ERROR: could not create the venv.
        pause & exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

REM -- Install dependencies if any core module is missing ----------------
python -c "import freehands, cv2, mediapipe, numpy, sklearn, PyQt6, pydantic, platformdirs" 2>nul
if errorlevel 1 (
    echo [FreeHands] Installing or repairing dependencies ^(this can take several minutes^)...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [FreeHands] ERROR installing dependencies from requirements.txt
        pause & exit /b 1
    )
    python -m pip install -e .
    if errorlevel 1 (
        echo.
        echo [FreeHands] ERROR installing the freehands package in editable mode.
        pause & exit /b 1
    )
)

REM -- Final core dependency check ---------------------------------------
python -c "import cv2, numpy, sklearn, PyQt6, pydantic, platformdirs; from freehands.gaze import GazeTracker; from freehands.gestures import HandTracker; g=GazeTracker(); g.close(); h=HandTracker(); h.close()" 2>nul
if errorlevel 1 (
    echo.
    echo [FreeHands] GazeTracker/HandTracker are not installed correctly. Repairing...
    python -m freehands repair
    if errorlevel 1 (
        echo.
        echo [FreeHands] ERROR: dependencies could not be repaired.
        echo            If mediapipe fails, install Python 3.11 and delete the .venv folder.
        pause & exit /b 1
    )
)

REM -- Interactive menu if no command was provided -----------------------
if "%CMD%"=="" (
    echo.
    echo  ------------------------------------------
    echo   FreeHands - user: %USER%
    echo  ------------------------------------------
    echo   1^) Start FreeHands ^(PC control^)
    echo   2^) Calibrate everything ^(gaze + gestures^)
    echo   3^) Recalibrate gaze only
    echo   4^) Recalibrate gestures only
    echo   5^) Select camera
    echo   6^) Doctor ^(camera, mic, dependencies^)
    echo   7^) Repair dependencies
    echo   8^) Change user
    echo   9^) Exit
    echo.
    set /p CHOICE="  Choose an option [1]: "
    if "!CHOICE!"=="" set "CHOICE=1"
    if "!CHOICE!"=="1" set "CMD=run"
    if "!CHOICE!"=="2" set "CMD=calibrate"
    if "!CHOICE!"=="3" set "CMD=gaze"
    if "!CHOICE!"=="4" set "CMD=gestures"
    if "!CHOICE!"=="5" set "CMD=camera"
    if "!CHOICE!"=="6" set "CMD=doctor"
    if "!CHOICE!"=="7" set "CMD=repair"
    if "!CHOICE!"=="8" (
        set /p USER="  New user: "
        if "!USER!"=="" set "USER=%DEFAULT_USER%"
        set "CMD=run"
    )
    if "!CHOICE!"=="9" exit /b 0
)

REM -- Execution ---------------------------------------------------------
echo [FreeHands] Log: "%LOGFILE%"
echo FreeHands %DATE% %TIME% > "%LOGFILE%"
echo CMD=%CMD% USER=%USER%>> "%LOGFILE%"

if /I "%CMD%"=="doctor" (
    echo [FreeHands] Running doctor...
    python -m freehands doctor >> "%LOGFILE%" 2>&1
) else if /I "%CMD%"=="repair" (
    echo [FreeHands] Repairing dependencies...
    python -m freehands repair >> "%LOGFILE%" 2>&1
) else if /I "%CMD%"=="camera" (
    echo [FreeHands] Selecting camera for user "%USER%"...
    python -m freehands camera --user "%USER%" >> "%LOGFILE%" 2>&1
) else if /I "%CMD%"=="calibrate" (
    echo [FreeHands] Calibrating gaze + gestures for user "%USER%"...
    python -m freehands calibrate --user "%USER%" >> "%LOGFILE%" 2>&1
) else if /I "%CMD%"=="gaze" (
    echo [FreeHands] Recalibrating gaze only for user "%USER%"...
    python -m freehands calibrate-gaze --user "%USER%" >> "%LOGFILE%" 2>&1
) else if /I "%CMD%"=="gestures" (
    echo [FreeHands] Recalibrating gestures only for user "%USER%"...
    python -m freehands calibrate-gestures --user "%USER%" >> "%LOGFILE%" 2>&1
) else if /I "%CMD%"=="run" (
    echo [FreeHands] Starting FreeHands for user "%USER%"...
    echo            ^(If you have no profile, calibration opens first.^)
    python -m freehands run --user "%USER%" >> "%LOGFILE%" 2>&1
) else (
    echo Usage:
    echo   FreeHands.bat                 ^(interactive menu^)
    echo   FreeHands.bat run [user]
    echo   FreeHands.bat calibrate [user]
    echo   FreeHands.bat gaze [user]
    echo   FreeHands.bat gestures [user]
    echo   FreeHands.bat camera [user]
    echo   FreeHands.bat doctor
    echo   FreeHands.bat repair
    pause & exit /b 1
)

set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo [FreeHands] Process finished with exit code %EXITCODE%.
    echo [FreeHands] Last log lines:
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -Path '%LOGFILE%' -Tail 40"
    pause
    exit /b %EXITCODE%
)
if "%MENU_MODE%"=="1" (
    copy /Y "%LOGFILE%" "%LASTLOG%" >nul 2>nul
    echo.
    echo [FreeHands] Process complete. Log saved at:
    echo %LOGFILE%
    echo.
    echo [FreeHands] Last log lines:
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -Path '%LOGFILE%' -Tail 30"
    pause
)
exit /b %EXITCODE%
