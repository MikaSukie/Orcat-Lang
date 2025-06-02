cls
.\ORCC.exe main.orcat -o out.ll
clang out.ll orcatRT.c -o main.exe
.\main.exe
echo exit code: %ERRORLEVEL%
