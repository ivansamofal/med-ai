"""LLM behind an interface, built on real LangChain chat models: AWS Bedrock
(Claude) via `ChatBedrockConverse` is the real implementation, LangChain's own
`FakeListChatModel` is the default so `make test` — and a fresh clone with no
AWS setup — run with zero API keys. Mirrors the Embedding/VectorStore
interface split from Phase 2. Switch to Bedrock with `LLM_BACKEND=bedrock`
once you have your own AWS access configured.
"""

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import settings

# Deterministic canned response, valid JSON for `ParsedRecommendation`
# (see app.recommendations.parser) so the real output parser — not a stand-in —
# stays exercised by the test suite without a real model call.
FAKE_RESPONSE = json.dumps(
    {
        "text": (
            "This is a deterministic fake recommendation for offline testing; "
            "no real model was called."
        ),
        "citations": ["Fake Source A", "Fake Source B"],
    }
)


class _FakeChatModel(FakeListChatModel):
    """`FakeListChatModel` plus a no-op `bind_tools`, so the chat agent
    (Phase 5) can compile and run offline the same way the Phase 3
    recommendation chain does. It never actually emits a tool call —
    deciding *when* to call a tool needs real model intelligence, which the
    offline fake deliberately doesn't have; switch to `LLM_BACKEND=bedrock`
    to exercise real tool-calling.
    """

    def bind_tools(self, tools, **kwargs):
        return self


def get_chat_model(fake_response: str | None = None) -> BaseChatModel:
    """`fake_response` lets other structured-output call sites (e.g. OCR
    entity extraction) supply canned JSON matching *their* output schema
    instead of the recommendation chain's; defaults to `FAKE_RESPONSE` so
    existing callers are unaffected."""
    if settings.llm_backend == "bedrock":
        from langchain_aws import ChatBedrockConverse

        # Bedrock has no LocalStack equivalent, so this always calls real AWS —
        # unlike the SQS client, it does not read settings.aws_endpoint_url.
        return ChatBedrockConverse(
            model=settings.bedrock_model_id,
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            max_tokens=settings.bedrock_max_tokens,
        )

    return _FakeChatModel(responses=[fake_response or FAKE_RESPONSE])
