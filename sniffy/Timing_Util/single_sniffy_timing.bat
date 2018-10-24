start UMA.exe
python sniffy_timing.py batchA 50 1000 7
taskkill /im UMA.exe
start UMA.exe
python sniffy_timing.py batchB 100 1000 7
taskkill /im UMA.exe
