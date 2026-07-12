from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent import tools as tools_module
from app.agent.graph import build_chat_graph


class ScriptedToolCallingChatModel(FakeMessagesListChatModel):
    """Cycles through scripted `AIMessage`s (including tool calls);
    `bind_tools` is a no-op since the responses are pre-scripted."""

    def bind_tools(self, tools, **kwargs):
        return self


def _tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _run(responses: list[AIMessage], patient_id: str = "p-1", session_id: str = "s-1") -> dict:
    compiled = build_chat_graph(
        checkpointer=MemorySaver(),
        chat_model=ScriptedToolCallingChatModel(responses=responses),
    )
    config = {"configurable": {"thread_id": session_id}}
    return compiled.invoke(
        {"messages": [HumanMessage(content="hi")], "patient_id": patient_id}, config=config
    )


def test_no_tool_call_ends_immediately_with_model_reply():
    result = _run([AIMessage(content="Hello, how can I help?")])

    assert result["messages"][-1].content == "Hello, how can I help?"


def test_non_red_flag_tool_result_loops_back_to_agent(monkeypatch):
    monkeypatch.setattr(
        tools_module,
        "build_lab_history_context",
        lambda patient_id, index=None: {"patient_id": patient_id, "results": [], "red_flag": False},
    )

    result = _run(
        [
            AIMessage(
                content="",
                tool_calls=[_tool_call("get_patient_lab_history", {"patient_id": "p-1"})],
            ),
            AIMessage(content="Your results look normal."),
        ]
    )

    assert result["messages"][-1].content == "Your results look normal."


def test_red_flag_tool_result_forces_escalation_without_a_second_agent_turn(monkeypatch):
    monkeypatch.setattr(
        tools_module,
        "build_lab_history_context",
        lambda patient_id, index=None: {
            "patient_id": patient_id,
            "results": [{"test_code": "K", "red_flag": "critical potassium"}],
            "red_flag": True,
        },
    )

    result = _run(
        [
            AIMessage(
                content="",
                tool_calls=[_tool_call("get_patient_lab_history", {"patient_id": "p-1"})],
            ),
            AIMessage(content="THIS SHOULD NOT APPEAR - escalation should short-circuit before this"),
        ]
    )

    final_content = result["messages"][-1].content
    assert "on-call clinician" in final_content
    assert "SHOULD NOT APPEAR" not in final_content


def test_book_appointment_tool_call_persists_and_agent_confirms():
    result = _run(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "book_appointment",
                        {
                            "patient_id": "p-1",
                            "doctor": "dr.jones",
                            "start_time": "2026-08-10T10:00:00+00:00",
                        },
                    )
                ],
            ),
            AIMessage(content="You're booked with dr.jones."),
        ]
    )

    assert result["messages"][-1].content == "You're booked with dr.jones."
