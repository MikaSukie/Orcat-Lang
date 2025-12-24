@echo off
cls

echo ================================
echo Building with ORCC...
echo ================================
echo.

set start=%TIME%

REM Run ORCC.exe to generate LLVM IR
ORCC.exe main.orcat -o out.ll
if errorlevel 1 (
    echo ORCC failed with exit code %ERRORLEVEL%.
    goto end
)

echo.
echo ================================
echo Optimizing LLVM IR...
echo ================================
echo.

REM Optimize IR and produce readable text IR
opt -O2 -S out.ll -o out.opt.ll
if errorlevel 1 (
    echo LLVM optimization failed with exit code %ERRORLEVEL%.
    goto end
)

echo.
echo ================================
echo Compiling with Clang...
echo ================================
echo.

REM Compile optimized IR with Clang
clang -O2 -Wno-override-module out.opt.ll stdlib.c -o main.exe
if errorlevel 1 (
    echo Clang failed with exit code %ERRORLEVEL%.
    goto end
)

echo.
echo ================================
echo Running main.exe...
echo ================================
echo.

.\main.exe
set exe_exit=%ERRORLEVEL%

echo.
echo ================================
echo exit code: %exe_exit%
echo ================================

set end=%TIME%

REM Calculate elapsed time in seconds.centiseconds
call :CalcElapsedTime %start% %end%
goto end

:CalcElapsedTime
setlocal
set start=%1
set end=%2

for /F "tokens=1-4 delims=:.," %%a in ("%start%") do (
    set /A sh=%%a, sm=%%b, ss=%%c, sc=%%d
)
for /F "tokens=1-4 delims=:.," %%a in ("%end%") do (
    set /A eh=%%a, em=%%b, es=%%c, ec=%%d
)

set /A start_total=((sh*3600 + sm*60 + ss)*100 + sc)
set /A end_total=((eh*3600 + em*60 + es)*100 + ec)
set /A elapsed=end_total-start_total

if %elapsed% LSS 0 set /A elapsed=8640000+elapsed

set /A elapsed_s=elapsed/100
set /A elapsed_cs=elapsed%%100

echo Elapsed time: %elapsed_s%.%elapsed_cs% seconds
endlocal
goto :eof

:end
pause
