@echo off
rem Share the site publicly for free, straight from this PC.
rem   1. rebuilds the public package (YouTube off, 60 s cap, 15 MB upload limit)
rem   2. starts the server on 127.0.0.1:7860
rem   3. opens a free tunnel and prints the public address (look for  https://....lhr.life )
rem Anyone with that address can use the site for as long as BOTH windows stay open.
rem The address changes every time you run this. Close the windows to stop sharing.

setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\ser_api\.venv\Scripts\python.exe"
set "PKG=%ROOT%\deploy\out\space"

echo Building the public package...
python "%ROOT%\deploy\build.py" --api-url https://unused.example >nul
if errorlevel 1 ( echo Build failed. & pause & exit /b 1 )

echo Starting the server in a new window (it takes ~10 seconds to load the model)...
start "SER server - keep open" /D "%PKG%" cmd /k "set SER_API_CONFIG=%PKG%\config.yaml&& set SER_API_MAX_CONCURRENT=1&& "%PY%" -m uvicorn ser_api.main:app --host 127.0.0.1 --port 7860"

timeout /t 15 /nobreak >nul
echo.
echo Opening the public tunnel. Your public address will appear below:
echo.
ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -R 80:127.0.0.1:7860 nokey@localhost.run
pause
