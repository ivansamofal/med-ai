"""The single-shot retrieve -> prompt -> generate chain, built as a real
LangChain LCEL pipeline: `ChatPromptTemplate | chat_model | PydanticOutputParser`.
No branching, no state — Phase 4 is what adds that on top of this draft.
"""

from langchain_core.prompts import ChatPromptTemplate
from llama_index.core import VectorStoreIndex

from app.db.mongo import LabResultRepository, RecommendationRepository, get_database
from app.domain.recommendation import Recommendation
from app.llm.interface import get_chat_model
from app.recommendations.parser import get_output_parser
from app.recommendations.prompt import build_recommendation_prompt
from app.recommendations.retrieval import build_context_passages

_PROMPT = ChatPromptTemplate.from_messages([("human", "{prompt_body}\n\n{format_instructions}")])


async def generate_recommendation(lab_result_id: str, index: VectorStoreIndex | None = None) -> str:
    """Run the chain for one lab result and return the new recommendation's id."""
    db = get_database()
    lab_result = await LabResultRepository(db).get_by_id(lab_result_id)

    passages = build_context_passages(lab_result, index=index)
    prompt_body = build_recommendation_prompt(lab_result, passages)

    output_parser = get_output_parser()
    chain = _PROMPT | get_chat_model() | output_parser
    parsed = await chain.ainvoke(
        {"prompt_body": prompt_body, "format_instructions": output_parser.get_format_instructions()}
    )

    recommendation = Recommendation(
        lab_result_id=lab_result_id,
        patient_id=lab_result.patient_id,
        recommendation_text=parsed.text,
        citations=parsed.citations,
        raw_llm_response=parsed.model_dump_json(),
    )
    return await RecommendationRepository(db).insert(recommendation)
