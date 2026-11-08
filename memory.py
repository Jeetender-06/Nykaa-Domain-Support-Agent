from __future__ import annotations

import bootstrap  # offline/telemetry settings before framework imports

from collections import OrderedDict

from langchain_core.chat_history import InMemoryChatMessageHistory


class SessionMemory:
    def __init__(self, max_sessions: int = 128, history_turns: int = 6):
        self.histories: OrderedDict[str, InMemoryChatMessageHistory] = OrderedDict()
        self.max_sessions, self.history_turns = max_sessions, history_turns

    def get(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.histories:
            self.histories[session_id] = InMemoryChatMessageHistory()
        self.histories.move_to_end(session_id)
        if len(self.histories) > self.max_sessions:
            self.histories.popitem(last=False)
        history = self.histories[session_id]
        # Keep the latest complete exchanges before the next invocation.
        history.messages = history.messages[-2 * self.history_turns :]
        return history

    def clear(self, session_id: str) -> None:
        self.histories.pop(session_id, None)
