@echo off
setlocal
py -m pip install -e .
if errorlevel 1 exit /b %errorlevel%
py -m stupidify.gui
