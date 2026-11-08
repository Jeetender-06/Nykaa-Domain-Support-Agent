"""Part 1: dataset, policy KB, chunking, RAG calibration and quality comparison."""
import json
import bootstrap
from config import Settings
from dataset import ORDERS, validate_orders
from evaluation import compare_strategies, load_cases, write_json, write_retrieval_markdown
from service import SupportService

if __name__ == "__main__":
    settings = Settings()
    print(json.dumps(validate_orders(ORDERS), indent=2))
    svc = SupportService(settings)
    comparison = compare_strategies(svc.rag, load_cases(settings.root))
    write_json(settings.root / "reports" / "calibration.json", svc.rag.calibration)
    write_json(settings.root / "reports" / "retrieval_comparison.json", comparison)
    write_retrieval_markdown(settings.root / "reports" / "retrieval_comparison.md", comparison)
    print(json.dumps({"collections": {k:v.count() for k,v in svc.rag.collections.items()}, "comparison": {k: comparison.get(k) for k in ["recommended_strategy"]}}, indent=2))
