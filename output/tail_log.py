import sys
import os

filepath = 'output/got_swift_train.log'
if not os.path.exists(filepath):
    print("Log file does not exist yet.")
    sys.exit()
    
with open(filepath, 'r') as f:
    f.seek(0, 2)
    eof = f.tell()
    f.seek(max(0, eof - 2000))
    lines = f.readlines()
    print(''.join(lines))
