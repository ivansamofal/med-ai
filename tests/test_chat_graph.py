from datetime import datetime, timezone

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent import tools as tools_module
from app.agent.graph import build_chat_graph
from app.db.sync_mongo import get_sync_database
from app.domain.appointment import Appointment


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


def _insert_appointment(**overrides) -> str:
    defaults = dict(
        patient_id="p-owner",
        doctor="dr.jones",
        scheduled_at=datetime(2026, 8, 10, 10, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    result = get_sync_database()["appointments"].insert_one(Appointment(**defaults).model_dump())
    return str(result.inserted_id)


def test_reschedule_appointment_rejects_a_different_patients_appointment():
    appointment_id = _insert_appointment()

    result = _run(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "reschedule_appointment",
                        {"appointment_id": appointment_id, "new_start_time": "2026-08-11T10:00:00+00:00"},
                    )
                ],
            ),
            AIMessage(content="done"),
        ],
        patient_id="p-attacker",
    )

    tool_message = result["messages"][-2]
    assert "not_found" in tool_message.content

    doc = get_sync_database()["appointments"].find_one({"patient_id": "p-owner"})
    assert doc["scheduled_at"] == datetime(2026, 8, 10, 10, 0, tzinfo=timezone.utc)


def test_cancel_appointment_rejects_a_different_patients_appointment():
    appointment_id = _insert_appointment()

    result = _run(
        [
            AIMessage(
                content="",
                tool_calls=[_tool_call("cancel_appointment", {"appointment_id": appointment_id})],
            ),
            AIMessage(content="done"),
        ],
        patient_id="p-attacker",
    )

    tool_message = result["messages"][-2]
    assert "not_found" in tool_message.content

    doc = get_sync_database()["appointments"].find_one({"patient_id": "p-owner"})
    assert doc["status"] == "scheduled"
