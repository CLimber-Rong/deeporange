@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

where py >nul 2>&1
if not errorlevel 1 (
    py -3 "%~dp0combine.py"
) else (
    where python >nul 2>&1
    if errorlevel 1 (
        echo Python was not found. Please install Python 3 and try again.
        pause
        exit /b 1
    )
    python "%~dp0combine.py"
)

set "exit_code=%errorlevel%"
echo.
if "%exit_code%"=="0" (
    echo Dataset generation finished.
) else (
    echo Dataset generation finished with errors.
)
pause
exit /b %exit_code%
