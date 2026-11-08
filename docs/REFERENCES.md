# Implementation references

Only framework documentation/source was consulted for integration details. The dataset, policy text, examples, design decisions and analysis were authored for this capstone. The fictional KB is not a statement of actual Nykaa policy.

- [CrewAI custom LLM extension](https://docs.crewai.com/en/learn/custom-llm): `BaseLLM` subclass contract. The installed 1.15.37 implementation and native tool executor were also inspected locally to verify call schemas and structured output behavior.
- [CrewAI Flows](https://docs.crewai.com/en/concepts/flows): typed flows and lifecycle decorators; implementation is `SupportFlow`.
- [AutoGen AssistantAgent structured output](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html): final editor `output_content_type` and structured message behavior.
- [AutoGen team reference](https://microsoft.github.io/autogen/stable/reference/python/autogen_agentchat.teams.html): `RoundRobinGroupChat`, `max_turns`, and custom message registration.
- [AutoGen replay model client](https://microsoft.github.io/autogen/stable/reference/python/autogen_ext.models.replay.html): the offline dynamic review client extends the replay client interface and computes responses from evidence rather than replaying fixed verdicts.
- [AutoGen model client contract](https://microsoft.github.io/autogen/stable/reference/python/autogen_core.models.html): `CreateResult`, usage and model capabilities.
- [SentenceTransformers model interface](https://sbert.net/docs/package_reference/sentence_transformer/model.html): local-only loading, CPU model execution, and normalized embeddings.
- [MiniLM model repository](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): public embedding weights; the exact revision is pinned in the provisioning script.
- [Chroma Python collection reference](https://docs.trychroma.com/reference/python/collection): collection upserts, explicit embeddings, metadata and queries.
- [LangChain message history source](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/runnables/history.py): `RunnableWithMessageHistory`; the installed 1.7.6 contract was verified locally. Its deprecation warnings are expected for the interface requested by the brief.
- [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/): per-client chat handlers and `WebSocketDisconnect` handling.

Framework APIs are version-sensitive; use `requirements.txt` and `reports/environment.json` when reproducing this submission.
