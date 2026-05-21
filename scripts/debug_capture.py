import subprocess
import sys
from pathlib import Path

script = Path(__file__).resolve().parent / 'train_kmeans_rf.py'
cmd = [sys.executable, str(script)]
print('CMD:', cmd)
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
out, err = proc.communicate(timeout=120)
print('RETURN', proc.returncode)
print('STDOUT')
print(out)
print('STDERR')
print(err)
