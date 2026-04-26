@echo off
REM ──────────────────────────────────────────────────────────────────────
REM  FreeHands · Doble-click aqui para empezar.
REM  Te pregunta tu nombre de usuario y arranca el sistema.
REM  Si es la primera vez, se abrira la calibracion automaticamente.
REM ──────────────────────────────────────────────────────────────────────
@echo on
@echo off
set /p USER="Nombre de usuario (Enter para 'luis'): "
if "%USER%"=="" set "USER=luis"
call "%~dp0freehands.bat" run %USER%
