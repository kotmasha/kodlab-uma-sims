@ECHO OFF
SETLOCAL

del batchA.dat
del batchB.dat
for /L %%r in (1,1,50) do CALL :single "Running pair number %%r"

EXIT /B %ERRORLEVEL%

:single
ECHO %~1
CALL single_sniffy_timing.bat
EXIT /B 0