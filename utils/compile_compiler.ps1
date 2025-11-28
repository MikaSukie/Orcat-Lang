cd C:\Users\USER\Documents\Projects\OrcatTest\
python -m nuitka --onefile --windows-icon-from-ico=orcc.ico ORCC.py
. .\.venv\Scripts\Activate.ps1
python -m nuitka --onefile --windows-console-mode=disable --windows-icon-from-ico=orcc.ico ORCC.py
