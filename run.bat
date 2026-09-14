@echo off
setlocal

echo ======================================================================
echo   ENTERPRISE AI NODE BOOTSTRAPPER - Windows Zero-Touch
echo ======================================================================

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python not found in PATH.
    echo [*] Checking winget for automated install...
    winget --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo [*] Installing Python 3.11 automatically...
        winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
    ) else (
        echo [x] Error: Python 3.8+ is required. Please install Python from https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

:: 2. Fast check if dependencies are already installed (0 internet delay)
python -c "import requests, psutil" >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Installing required Python dependencies...
    python -m pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
)

:: 3. Optional Automated Tailscale Auto-Install & Auto-Connect
if exist "C:\Program Files\Tailscale" (
    set "PATH=C:\Program Files\Tailscale;%PATH%"
)
if exist "%LOCALAPPDATA%\Programs\Tailscale" (
    set "PATH=%LOCALAPPDATA%\Programs\Tailscale;%PATH%"
)
for /f "usebackq tokens=1,* delims==" %%A in (`findstr /b "TAILSCALE_AUTHKEY=" .env 2^>nul`) do (
    set TAIL_KEY=%%B
)
if defined TAIL_KEY if not "%TAIL_KEY%"=="" (
    tailscale status >nul 2>&1
    if %errorlevel% neq 0 (
        echo [*] Tailscale AuthKey detected. Checking Tailscale installation...
        where tailscale >nul 2>&1
        if %errorlevel% neq 0 (
            echo [*] Installing Tailscale via winget...
            winget install -e --id Tailscale.Tailscale --accept-package-agreements --accept-source-agreements
            if exist "C:\Program Files\Tailscale" (
                set "PATH=C:\Program Files\Tailscale;%PATH%"
            )
        )
        echo [*] Connecting Tailscale silently with AuthKey...
        tailscale up --authkey %TAIL_KEY% --unattended >nul 2>&1
    )
)

:: 4. Determine Gateway URL and Arguments
set GATEWAY_ARG=
if "%~1"=="" (
    set GATEWAY_ARG=
) else (
    echo %~1 | findstr /b /c:"--" >nul
    if %errorlevel% neq 0 (
        set GATEWAY_ARG=--gateway-url %1
        shift
    )
)

:: 5. Launch Node Agent
echo [*] Starting Smart AI Node Agent...
echo.
python node_agent.py %GATEWAY_ARG% %1 %2 %3 %4 %5

if %errorlevel% neq 0 (
    echo.
    echo [x] Node Agent stopped with exit code %errorlevel%.
)

pause
