"""Run the 15-case evaluation and save the 4 rubric scores."""
import sys
import bootstrap
from cli import main

if __name__ == "__main__":
    sys.argv=[sys.argv[0], "evaluate"]
    main()
