@echo off
setlocal
pushd "%~dp0" || exit /b 1
if exist "resources\runtime\python.exe" (
    "resources\runtime\python.exe" main.py
) else (
    py -3.10 main.py
)
set "exit_code=%errorlevel%"
if not "%exit_code%"=="0" pause
popd
exit /b %exit_code%
