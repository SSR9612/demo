@echo off
chcp 65001 >nul
echo ========================================
echo  llama.cpp SmolVLM-500M vision server
echo ========================================
echo.
d:
cd /d D:\llama

llama-server.exe --jinja ^
  -m ./models/SmolVLM-500M-Instruct-f16.gguf ^
  --mmproj ./models/mmproj-SmolVLM-500M-Instruct-f16.gguf ^
  --host 127.0.0.1 --port 8080 ^
  -c 8192 -ngl 99 -t 14

set "RC=%ERRORLEVEL%"
echo.
echo Done. Exit code: %RC%. Press any key to close...
pause >nul
exit /b %RC%