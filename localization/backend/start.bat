@echo off
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting backend on http://localhost:8000
uvicorn main:app --reload --host 0.0.0.0 --port 8000
