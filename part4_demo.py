"""Part 4: AutoGen review, governance and cache evidence (full integration demo)."""
import json
import bootstrap
from demo import generate_evidence

if __name__ == "__main__":
    report=generate_evidence()
    print(json.dumps(report,indent=2))
