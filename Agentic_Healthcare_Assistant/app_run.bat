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

:: 3. Self-provision .streamlit configuration from visible streamlit_config.toml
if not exist ".streamlit" mkdir ".streamlit" >nul 2>&1
if exist "streamlit_config.toml" (
    if not exist ".streamlit\config.toml" (
        copy /y "streamlit_config.toml" ".streamlit\config.toml" >nul 2>&1
        echo [*] Initialized .streamlit/config.toml from visible streamlit_config.toml
    )
)

:: 4. Launch Streamlit Application with embedded theme & server flags
echo.
echo [*] Launching Streamlit Clinical Command Center on http://localhost:8501 ...
echo [*] To stop the server, press Ctrl+C in this window.
echo ==============================================================================
echo.

python -m streamlit run app.py --server.port 8501 --theme.base dark --theme.primaryColor "#0284c7" --theme.backgroundColor "#090d16" --theme.secondaryBackgroundColor "#0f172a" --theme.textColor "#f1f5f9" --client.toolbarMode minimal --server.headless true

echo.
echo ==============================================================================
echo [*] Application has stopped.
pause
