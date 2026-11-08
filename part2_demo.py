"""Part 2: real CrewAI tool invocation, session memory, guardrails, structured output."""
import asyncio
import json
import bootstrap
from demo import demonstrations
from service import SupportService

if __name__ == "__main__":
    checks = asyncio.run(demonstrations(SupportService()))
    print(json.dumps(checks, indent=2))
