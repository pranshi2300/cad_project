@echo off
setlocal

echo.
echo  =====================================================
echo   LLM Router Demo  --  Starting up...
echo  =====================================================
echo.

REM -- Use the project venv if it exists, otherwise fall back to whatever python is on PATH
set PYTHON=python
if exist ".venv\Scripts\python.exe" set PYTHON=.venv\Scripts\python.exe

echo  Using Python: %PYTHON%
echo.

echo  Step 1/2: Installing / verifying demo dependencies...
%PYTHON% -m pip install -q flask flask-cors requests tiktoken
echo  Done.
echo.

echo  Step 2/2: Launching server...
echo  Open http://localhost:5000 in your browser once you see "LLM Router demo ready!"
echo.
%PYTHON% app.py
pause
