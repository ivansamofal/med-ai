"""Entry point the async FastAPI route calls into the chat agent through.
Sync (the graph and its `MongoDBSaver` checkpointer are sync-first, same as
the approval graph) — the caller wraps this in `asyncio.to_thread`.
"""

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_chat_graph

_compiled_graph = None


def _graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_chat_graph()
    return _compiled_graph


def send_chat_message(session_id: str, patient_id: str, message: str) -> str:
    config = {"configurable": {"thread_id": session_id}}
    result = _graph().invoke(
        {"messages": [HumanMessage(content=message)], "patient_id": patient_id},
        config=config,
    )
    reply = result["messages"][-1]
    return reply.content if isinstance(reply, AIMessage) else str(reply.content)
