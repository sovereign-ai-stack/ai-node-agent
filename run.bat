@echo off
setlocal

set GATEWAY_URL=%1
if "%GATEWAY_URL%"=="" set GATEWAY_URL=http://localhost:8200

echo ==================================================
echo  Starting Enterprise AI Node Agent (Windows)...
echo  Central Gateway: %GATEWAY_URL%
echo ==================================================

python node_agent.py --gateway-url %GATEWAY_URL%
pause
