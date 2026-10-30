"""
chains.py
=========
The LangChain heart of the app.

  build_llm()                 -> ChatOpenAI (the OpenAI chat model)
  build_assessment_chain()    -> LLMChain  (prompt + model, reusable)
  run_assessment()            -> runs the chain, returns the raw JSON text
  stream_narrative()          -> yields the readable guidance chunk by chunk
  demo_raw_messages()         -> shows SystemMessage / HumanMessage / AIMessage
  build_conversation_messages / stream_followup -> follow-up chat using AIMessage
"""

import warnings
from typing import Iterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src import config, prompts
from src.cache_manager import cached_stream

# LLMChain is the classic LangChain chain (prompt -> model). In LangChain 1.x it
# moved to the "langchain-classic" package, so we support both locations.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from langchain.chains import LLMChain
    except ImportError:  # pragma: no cover
        from langchain_classic.chains import LLMChain

# Asks OpenAI to return a syntactically valid JSON object ("JSON mode").
JSON_MODE_KWARGS = {"response_format": {"type": "json_object"}}


# --------------------------------------------------------------------------
# 1. ChatOpenAI
# --------------------------------------------------------------------------
def build_llm(
    api_key: str,
    model: str = config.DEFAULT_MODEL,
    temperature: float = config.DEFAULT_TEMPERATURE,
    use_cache: bool = True,
) -> ChatOpenAI:
    """
    Create the OpenAI chat model.

    use_cache=False tells this ONE model object to ignore the global cache
    (used for the automatic retry when a cached answer was not valid JSON).
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,  # comes from .env or the sidebar, never from the source code
        timeout=60,
        max_retries=2,
        cache=None if use_cache else False,
    )


# --------------------------------------------------------------------------
# 2. LLMChain (prompt + model) for the structured JSON assessment
# --------------------------------------------------------------------------
def build_assessment_chain(llm) -> "LLMChain":
    """A reusable chain: ChatPromptTemplate -> ChatOpenAI (in JSON mode)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LLMChain(
            llm=llm,
            prompt=prompts.ASSESSMENT_CHAT_TEMPLATE,
            llm_kwargs=JSON_MODE_KWARGS,
        )


def run_assessment(chain, inputs: dict) -> str:
    """Run the chain. Returns the model's raw text (it should be JSON)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = chain.invoke({k: inputs[k] for k in prompts.PATIENT_VARIABLES})
    return result["text"]


# --------------------------------------------------------------------------
# 3. Streaming the human-readable narrative
# --------------------------------------------------------------------------
def narrative_messages(inputs: dict) -> list:
    """Fill the narrative ChatPromptTemplate -> [SystemMessage, HumanMessage]."""
    return prompts.NARRATIVE_CHAT_TEMPLATE.format_messages(**inputs)


def stream_narrative(llm, inputs: dict) -> Iterator[str]:
    """
    Yield the narrative piece by piece using llm.stream().
    st.write_stream() prints each piece as it arrives (typing effect).
    """
    messages = narrative_messages(inputs)
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content  # st.write_stream prints each piece


def _cache_key(messages: list) -> str:
    return "\n".join(f"{m.type}: {m.content}" for m in messages)


def _llm_string(llm, label: str) -> str:
    model = getattr(llm, "model_name", None) or type(llm).__name__
    return f"{label}|{model}|{getattr(llm, 'temperature', '')}"


def stream_narrative_cached(llm, inputs: dict) -> Iterator[str]:
    """stream_narrative() + the global LangChain cache (see cache_manager.cached_stream)."""
    messages = narrative_messages(inputs)
    return cached_stream(
        _cache_key(messages),
        _llm_string(llm, "narrative"),
        lambda: stream_narrative(llm, inputs),
    )


# --------------------------------------------------------------------------
# 4. Raw System / Human / AI messages
# --------------------------------------------------------------------------
def demo_messages(question: str) -> list:
    """
    A tiny hand-made conversation that shows what each message type is for:

      SystemMessage -> the rules / role for the model
      HumanMessage  -> what the user says
      AIMessage     -> what the model said earlier (gives it memory of the chat)
    """
    return [
        SystemMessage(content=prompts.SYSTEM_PROMPT),
        HumanMessage(content="I have had a mild headache for two days. Is that serious?"),
        AIMessage(
            content=(
                "I'm sorry you're not feeling well. A mild headache is often related to "
                "tiredness, dehydration or stress, but I can't diagnose it. If it gets "
                "worse or comes with other symptoms, please see a healthcare professional."
            )
        ),
        HumanMessage(content=question),
    ]


def demo_raw_messages(llm, question: str):
    """Send the demo conversation to the model. Returns (messages, answer_text)."""
    messages = demo_messages(question)
    return messages, llm.invoke(messages).content


def build_conversation_messages(inputs: dict, assessment_json: str, history: list, question: str) -> list:
    """
    Build a follow-up conversation:
      System    -> safety rules (+ follow-up mode)
      Human     -> the original patient data
      AI        -> the guidance we already produced
      (earlier follow-up Q&A as Human/AI pairs)
      Human     -> the new question
    """
    patient_text = prompts.ASSESSMENT_PROMPT.format(**{k: inputs[k] for k in prompts.PATIENT_VARIABLES})
    messages = [
        SystemMessage(content=prompts.SYSTEM_PROMPT + prompts.FOLLOWUP_SYSTEM_SUFFIX),
        HumanMessage(content=patient_text),
        AIMessage(content=assessment_json),
    ]
    for question_text, answer_text in history:
        messages.append(HumanMessage(content=question_text))
        messages.append(AIMessage(content=answer_text))
    messages.append(HumanMessage(content=f"<patient_data>{question}</patient_data>"))
    return messages


def stream_followup(llm, inputs: dict, assessment_json: str, history: list, question: str) -> Iterator[str]:
    """Stream the answer to a follow-up question (with caching)."""
    messages = build_conversation_messages(inputs, assessment_json, history, question)

    def producer():
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content

    return cached_stream(_cache_key(messages), _llm_string(llm, "followup"), producer)
