"""Run laboratory 1 from its current directory."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "src" / "finish_lab1.py"), run_name="__main__")
