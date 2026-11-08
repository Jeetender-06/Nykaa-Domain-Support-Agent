"""Sanity check the local model, data files and key dependency versions."""
import importlib.metadata
import json
import bootstrap
from config import Settings

if __name__ == "__main__":
    cfg=Settings()
    report={"python_requirement":"3.11", "model_available":(cfg.model_path / "config.json").exists(), "policy_documents":len(list(cfg.kb_path.glob("*.md"))), "saved_orders":(cfg.root / "data/orders.json").exists(), "packages":{}}
    for pkg in ["crewai","autogen-agentchat","chromadb","sentence-transformers","fastapi"]:
        try:report["packages"][pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:report["packages"][pkg]="NOT INSTALLED"
    print(json.dumps(report,indent=2))
