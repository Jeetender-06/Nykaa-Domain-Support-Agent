"""Prepare both Chroma indexes, calibrate thresholds, compare retrieval strategies."""
import sys
import bootstrap
from cli import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "prepare"]
    main()
