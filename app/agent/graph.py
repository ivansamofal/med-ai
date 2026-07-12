"""The chat/appointment agent: a LangGraph tool-calling loop over
`app.agent.tools.CHAT_TOOLS`, checkpointed per patient chat session
(`MongoDBSaver`, same pattern as the approval graph) so conversation memory
survives a process restart.

LangGraph owns this because it's stateful/branching, same reasoning as the
approval graph (Phase 4): a chain only ever goes one direction, but this
agent loops (model -> tools -> model...) and has to make a safety-critical
routing decision — red-flag lab value -> escalate, don't answer — that can't
be left to the model's own judgment alone.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.tools import CHAT_TOOLS
from app.config import settings
from app.db.sync_mongo import get_sync_client
from app.llm.interface import get_chat_model
from app.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are MedAI's patient/clinician assistant. You can check doctor "
    "availability and book, reschedule, or cancel appointments, and look up "
    "a patient's lab history. Guardrails: never state or imply a diagnosis; "
    "whenever you reference guideline or reference-range content, cite the "
    "source title you were given; if a tool result flags a red-flag lab "
    "value, escalate to the on-call clinician instead of answering."
)


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    patient_id: str


def build_agent_node(chat_model: BaseChatModel):
    model_with_tools = chat_model.bind_tools(CHAT_TOOLS)

    def agent_node(state: ChatState) -> ChatState:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    return agent_node


def escalate_node(state: ChatState) -> ChatState:
    """Deterministic escalation exit: reached only when a tool result flagged
    a red-flag lab value — bypasses further model output so escalation
    doesn't depend on the model choosing to call `escalate_to_oncall` itself.
    """
    logger.warning("chat_forced_escalation", patient_id=state["patient_id"])
    return {
        "messages": [
            AIMessage(
                content=(
                    "I've flagged this lab result to the on-call clinician for "
                    "immediate review rather than answering directly."
                )
            )
        ]
    }


def _tool_results_flagged(state: ChatState) -> bool:
    """Reads the `ToolMessage`s appended by the most recent `tools` node run
    and returns True if any of them signals a red flag."""
    for message in reversed(state["messages"]):
        if not isinstance(message, ToolMessage):
            break
        try:
            payload = json.loads(message.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and payload.get("red_flag"):
            return True
    return False


def get_checkpointer() -> MongoDBSaver:
    return MongoDBSaver(get_sync_client(), db_name=settings.mongo_db_name)


def build_chat_graph(checkpointer=None, chat_model: BaseChatModel | None = None) -> CompiledStateGraph:
    """TODO (Phase 5, yours to implement): the chat agent's conditional routing.

    WHAT: Wire `agent -> [tools | END]` (standard ReAct loop) and, once tools
    run, `tools -> [agent | escalate]` depending on whether the tool results
    that just came back flagged a red-flag lab value. Compile with
    `checkpointer` (falls back to the real Mongo-backed `get_checkpointer()`
    when `None`) and `chat_model` (falls back to `get_chat_model()` — the
    parameter exists so tests can inject a scripted fake chat model, same
    reason `build_approval_graph` takes an injectable `checkpointer`).

    WHY: The standard "does the model want to call a tool" routing
    (`agent -> tools` vs `agent -> END`) is exactly what LangGraph's prebuilt
    `tools_condition` already does — no reason to hand-roll it. What's
    genuinely this project's own decision is what happens *after* tools run:
    normally you'd loop straight back to the model so it can use the tool
    result. But the architecture's core guardrail is "red-flag lab value ->
    escalate, don't answer" — and that can't be a suggestion to the model,
    because a smaller (or the offline fake) model might just answer with the
    number anyway. So the graph itself has to intercept: if
    `get_patient_lab_history`'s result flagged a red flag, route to
    `escalate` deterministically instead of back to `agent`.

    Hints:
    - `StateGraph(ChatState)`, `.add_node(name, fn)`, `.set_entry_point(name)`,
      `.add_conditional_edges(source, routing_fn, path_map)`, `END` — same
      toolkit as the approval graph (Phase 4).
    - `build_agent_node(chat_model)` (provided above) closes over the chat
      model and returns the `agent` node function; `ToolNode(CHAT_TOOLS)` is
      the `tools` node; `escalate_node` (provided above) is the escalation
      exit.
    - `tools_condition` (from `langgraph.prebuilt`) is the standard
      `agent -> tools`/`agent -> END` routing function — pass it straight to
      `add_conditional_edges`, don't reimplement it.
    - `_tool_results_flagged(state)` (provided above) is your `tools ->
      [agent | escalate]` routing function's job to call.
    - `escalate` should be a terminal node (`add_edge("escalate", END)`) —
      the forced escalation message is the end of that turn, not a springboard
      back into free-form model chat.
    """
    resolved_chat_model = chat_model if chat_model is not None else get_chat_model()

    graph = StateGraph(ChatState)
    graph.add_node("agent", build_agent_node(resolved_chat_model))
    graph.add_node("tools", ToolNode(CHAT_TOOLS))
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})

    def _route_after_tools(state: ChatState) -> Literal["agent", "escalate"]:
        return "escalate" if _tool_results_flagged(state) else "agent"

    graph.add_conditional_edges("tools", _route_after_tools, {"agent": "agent", "escalate": "escalate"})
    graph.add_edge("escalate", END)

    return graph.compile(checkpointer=checkpointer or get_checkpointer())
