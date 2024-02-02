@echo off

:: Activate virtual environment
venv\Scripts\activate

:: Install dependencies
pip install -r requirements.txt

:: Run Flask app
python app.py
