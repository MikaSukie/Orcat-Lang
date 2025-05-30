cls
python3 ORCC.py main.orcat -o out.ll
clang out.ll -o main.exe
.\main.exe
echo %ERRORLEVEL%
