"""Execute the Ola-style demo stages sequentially."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    for name in ("calibrate.py", "part1_demo.py", "part2_demo.py", "part3_demo.py", "part4_demo.py"):
        print("\nRunning",name,flush=True)
        subprocess.run([sys.executable,str(Path(__file__).parent/name)],check=True)
