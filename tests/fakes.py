"""A fake chat model so the tests never need a real OpenAI key or internet."""

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

GOOD_JSON = {
    "summary": "A runny nose and sore throat are common and often caused by viral infections.",
    "possible_conditions": [
        {"name": "Common cold", "reason": "Runny nose and sore throat together are typical."},
        {"name": "Seasonal allergies", "reason": "Can also cause a runny nose."},
    ],
    "urgency_level": "LOW",
    "recommended_next_steps": ["Rest and drink fluids.", "Monitor symptoms.", "See a doctor if they worsen."],
    "questions_for_doctor": ["How long should I expect this to last?", "When should I come back?"],
    "warning_signs": ["Trouble breathing", "High fever lasting several days", "Severe throat swelling"],
}

NARRATIVE = (
    "I'm sorry you're feeling unwell. Your symptoms look mild, but only a healthcare "
    "professional can confirm the cause. This is educational information, not medical advice."
)


class FakeMediGuideLLM(BaseChatModel):
    """Returns JSON when JSON mode is requested, otherwise a short narrative."""

    json_text: str = json.dumps(GOOD_JSON)
    narrative_text: str = NARRATIVE
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-mediguide"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        text = self.json_text if "response_format" in kwargs else self.narrative_text
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        text = self.narrative_text
        for i in range(0, len(text), 10):
            yield ChatGenerationChunk(message=AIMessageChunk(content=text[i : i + 10]))
