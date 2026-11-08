"""Part 3: FastAPI + WebSocket demos and 15-query evaluation."""
import asyncio
import json
import bootstrap
from config import Settings
from demo import api_demonstrations
from evaluation import evaluate, write_json, write_evaluation_markdown
from service import SupportService

if __name__ == "__main__":
    svc = SupportService()
    checks = api_demonstrations(svc)
    results = asyncio.run(evaluate(svc))
    write_json(svc.settings.root / "reports/evaluation.json", results)
    write_evaluation_markdown(svc.settings.root / "reports/evaluation.md", results)
    print(json.dumps({"api_checks":checks, "evaluation_averages":results["averages"]}, indent=2))
