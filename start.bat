@echo off
echo Starting Study Tools API...

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install/update dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Start the service
echo Starting service...
python main.py

REM Keep the window open if there's an error
pause 