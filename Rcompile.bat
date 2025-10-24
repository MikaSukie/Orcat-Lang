@echo off
cls

set start=%TIME%

.\ORCC.exe main.orcat -o out.ll
clang -Wno-override-module out.ll -o main.exe
.\main.exe

echo exit code: %ERRORLEVEL%

set end=%TIME%

call :CalcElapsedTime %start% %end%
goto :eof

:CalcElapsedTime
setlocal

set start=%1
set end=%2

for /F "tokens=1-4 delims=:.," %%a in ("%start%") do (
  set /A start_h=%%a, start_m=%%b, start_s=%%c, start_cs=%%d
)

for /F "tokens=1-4 delims=:.," %%a in ("%end%") do (
  set /A end_h=%%a, end_m=%%b, end_s=%%c, end_cs=%%d
)

set /A start_total=((start_h*3600)+ (start_m*60) + start_s)*100 + start_cs
set /A end_total=((end_h*3600)+ (end_m*60) + end_s)*100 + end_cs

set /A elapsed=end_total - start_total

if %elapsed% LSS 0 (
  set /A elapsed=8640000 + elapsed
)
set /A elapsed_s=elapsed / 100
set /A elapsed_cs=elapsed %% 100

echo Elapsed time: %elapsed_s%.%elapsed_cs% seconds

endlocal
goto :eof
