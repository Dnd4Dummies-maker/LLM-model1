@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe scripts\chat.py chat_model.pt --device cpu
pause
