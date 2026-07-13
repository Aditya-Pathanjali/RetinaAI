import os
import sys

# A simple wrapper to map `python evaluate.py` to `python main.py --mode eval`

if __name__ == "__main__":
    # pass all args to main.py
    args = " ".join(sys.argv[1:])
    os.system(f"{sys.executable} main.py --mode eval {args}")
