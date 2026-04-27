@echo off
echo ============================================
echo HaptiGuide3D FINAL HRI APP
echo ============================================

py -3.11 -m venv hri_env
.\hri_env\Scripts\python.exe -m pip install --upgrade pip
.\hri_env\Scripts\python.exe -m pip uninstall -y mediapipe numpy
.\hri_env\Scripts\python.exe -m pip install -r requirements.txt
.\hri_env\Scripts\python.exe -c "import mediapipe as mp; print('MediaPipe', mp.__version__, 'solutions=', hasattr(mp, 'solutions'))"
.\hri_env\Scripts\python.exe main.py

pause
