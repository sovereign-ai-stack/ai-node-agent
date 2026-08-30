@echo off
setlocal enabledelayedexpansion

set GATEWAY_URL=%1
if "%GATEWAY_URL%"=="" set GATEWAY_URL=http://localhost:8200

echo ======================================================================
echo   ENTERPRISE AI NODE BOOTSTRAPPER (Windows Zero-Touch Provisioning)
echo   Central Gateway: %GATEWAY_URL%
echo ======================================================================

:: 1. Verify Python Installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python is not installed or not in PATH.
    echo [*] Attempting automated Python 3.11 installation via Windows Package Manager (winget)...
    winget --version >nul 2>&1
    if %errorlevel% equ 0 (
        winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
        echo [*] Refreshing environment variables...
        call refreshenv >nul 2>&1
    ) else (
        echo [x] Error: Python 3.8+ is required. Please install Python from https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

:: 2. Auto-Install Node Agent Python Dependencies
echo [*] Checking and installing Python dependencies (requests, psutil, pynvml)...
python -m pip install --upgrade pip -q >nul 2>&1
python -m pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo [!] Warning: Some dependencies failed to install silently. Retrying with verbose output...
    python -m pip install -r requirements.txt
)

:: 3. Launch the Smart Node Agent
echo [*] Launching Node Agent...
echo.
python node_agent.py --gateway-url %GATEWAY_URL% %2 %3 %4 %5

if %errorlevel% neq 0 (
    echo.
    echo [x] Node Agent terminated with exit code %errorlevel%.
)

pause
