@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

python -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "python_cmd=python"
) else (
    py -3 -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo Python was not found. Please install Python 3 and try again.
        pause
        exit /b 1
    )
    set "python_cmd=py -3"
)

%python_cmd% "%~dp0obfuscate.py"
if errorlevel 1 goto :failed
%python_cmd% "%~dp0combine.py"
if errorlevel 1 goto :failed

echo.
echo Dataset generation finished.
pause
exit /b 0

:failed
set "exit_code=%errorlevel%"
echo.
echo Dataset generation finished with errors.
pause
exit /b %exit_code%
