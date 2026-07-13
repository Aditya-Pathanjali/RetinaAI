import subprocess
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
exe_path = PROJECT_ROOT / "dist" / "RetinaAI.exe"

print(f"Diagnostic tool starting...")
print(f"Target executable: {exe_path}")

if not exe_path.exists():
    print(f"CRITICAL: Executable does not exist at {exe_path}")
    sys.exit(1)

try:
    print("Launching executable...")
    p = subprocess.Popen(
        [str(exe_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(PROJECT_ROOT)
    )
    
    print("Waiting 10 seconds to monitor startup...")
    for i in range(10):
        time.sleep(1)
        poll = p.poll()
        if poll is not None:
            break
            
    poll = p.poll()
    if poll is None:
        print("SUCCESS: The process is still running after 10 seconds. It started successfully without crashing.")
        p.terminate()
        print("Process terminated successfully.")
    else:
        print(f"CRASH: Process exited with return code: {poll}")
        stdout, stderr = p.communicate()
        print(stdout)
        print(stderr)
        
except Exception as e:
    print(f"Failed to run executable: {e}")
