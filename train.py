import os
import sys

# A simple wrapper to map `python train.py` to `python main.py --mode train`
if __name__ == "__main__":
    # pass all args to main.py
    args = " ".join(sys.argv[1:])
    os.system(f"{sys.executable} main.py --mode train {args}")
