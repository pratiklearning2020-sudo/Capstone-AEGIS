@echo off
title AEGIS-HEALTH - Agentic Healthcare Assistant
color 0B

echo ==============================================================================
echo                 AGENTIC HEALTHCARE MULTI-MODAL ASSISTANT
echo       Autonomous Clinical Triage, EHR Integration, RAG and Scheduling
echo ==============================================================================
echo.

cd /d "%~dp0"

:: 1. Check for Python virtual environments
if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment: venv
    call venv\Scripts\activate.bat
)
if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment: .venv
    call .venv\Scripts\activate.bat
)

:: 2. Check for Python executable
python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo [ERROR] Python is not recognized in your system PATH.
    echo Please install Python 3.10+ from https://www.python.org/
    echo Make sure to check "Add python.exe to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [*] Python detected:
python --version

:: 3. Launch Streamlit Application
echo.
echo [*] Launching Streamlit Clinical Command Center on http://localhost:8501 ...
echo [*] To stop the server, press Ctrl+C in this window.
echo ==============================================================================
echo.

python -m streamlit run app.py --server.port 8501

echo.
echo ==============================================================================
echo [*] Application has stopped.
pause
