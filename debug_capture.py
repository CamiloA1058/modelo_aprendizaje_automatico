import subprocess
import sys

cmd = [sys.executable, 'model_k-means_random_forest.py']
print('CMD:', cmd)
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
out, err = proc.communicate(timeout=120)
print('RETURN', proc.returncode)
print('STDOUT')
print(out)
print('STDERR')
print(err)
