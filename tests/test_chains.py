"""Tests for prompts, the LLMChain, streaming and both caches (fake LLM, no API key)."""

import pytest
from langchain_core.globals import set_llm_cache

from src import cache_manager, chains, config, prompts, utils
from tests.fakes import FakeMediGuideLLM


@pytest.fixture(autouse=True)
def isolated_caches(tmp_path, monkeypatch):
    """Every test gets fresh caches and its own SQLite file."""
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_cache.db")
    cache_manager._CACHES.clear()
    yield
    set_llm_cache(None)
    cache_manager._CACHES.clear()


@pytest.fixture
def inputs():
    return utils.build_inputs("25", "Female", ["Runny nose", "Sore throat"], "", "1-3 days", 2, "", "", "", "English")


def narrative_inputs(inputs):
    return {**inputs, "urgency_level": "LOW", "assessment_summary": "Mild cold-like symptoms."}


# ------------------------------------------------------------------ prompts
def test_prompt_template_is_filled(inputs):
    text = prompts.ASSESSMENT_PROMPT.format(**{k: inputs[k] for k in prompts.PATIENT_VARIABLES})
    assert "Age: 25" in text and "Runny nose, Sore throat" in text and '"urgency_level"' in text


def test_chat_prompt_template_has_system_and_human(inputs):
    messages = prompts.ASSESSMENT_CHAT_TEMPLATE.format_messages(**{k: inputs[k] for k in prompts.PATIENT_VARIABLES})
    assert [m.type for m in messages] == ["system", "human"]
    assert "NEVER state or imply a confirmed diagnosis" in messages[0].content
    assert "Female" in messages[1].content


def test_system_prompt_encodes_safety_rules():
    for phrase in ("NOT A DOCTOR", "confirmed diagnosis", "healthcare professional", "EMERGENCY"):
        assert phrase in prompts.SYSTEM_PROMPT


def test_language_reaches_the_prompt():  # scenario 6
    urdu = utils.build_inputs("25", "Male", ["Fever"], "", "1-3 days", 3, "", "", "", "Urdu")
    text = prompts.ASSESSMENT_PROMPT.format(**{k: urdu[k] for k in prompts.PATIENT_VARIABLES})
    assert "in: Urdu" in text


# -------------------------------------------------------------------- chain
def test_llmchain_returns_parsable_json(inputs):
    llm = FakeMediGuideLLM()
    raw = chains.run_assessment(chains.build_assessment_chain(llm), inputs)
    assert utils.parse_assessment(raw).ok
    assert llm.calls == 1


def test_streaming_yields_chunks(inputs):
    chunks = list(chains.stream_narrative(FakeMediGuideLLM(), narrative_inputs(inputs)))
    assert len(chunks) > 1
    assert "".join(chunks).startswith("I'm sorry")


def test_demo_messages_use_all_three_roles():
    types = [m.type for m in chains.demo_messages("Is it serious?")]
    assert types == ["system", "human", "ai", "human"]


def test_followup_messages_contain_previous_guidance(inputs):
    messages = chains.build_conversation_messages(inputs, '{"summary": "x"}', [("q1", "a1")], "q2")
    assert [m.type for m in messages] == ["system", "human", "ai", "human", "ai", "human"]
    assert messages[2].content == '{"summary": "x"}'


# ------------------------------------------------------------------- caching
@pytest.mark.parametrize("mode", [config.CACHE_MEMORY, config.CACHE_SQLITE])
def test_second_identical_run_comes_from_cache(mode, inputs):  # scenario 4
    cache = cache_manager.configure_cache(mode)
    llm = FakeMediGuideLLM()
    chain = chains.build_assessment_chain(llm)

    first = chains.run_assessment(chain, inputs)
    second = chains.run_assessment(chain, inputs)

    assert first == second
    assert llm.calls == 1, "the second request must not reach the model"
    assert (cache.hits, cache.misses) == (1, 1)


@pytest.mark.parametrize("mode", [config.CACHE_MEMORY, config.CACHE_SQLITE])
def test_streamed_narrative_is_cached_too(mode, inputs):
    cache_manager.configure_cache(mode)
    llm = FakeMediGuideLLM()

    first = "".join(chains.stream_narrative_cached(llm, narrative_inputs(inputs)))
    calls_after_first = llm.calls
    second = "".join(chains.stream_narrative_cached(llm, narrative_inputs(inputs)))

    assert first == second
    assert llm.calls == calls_after_first


def test_different_inputs_miss_the_cache(inputs):
    cache_manager.configure_cache(config.CACHE_MEMORY)
    llm = FakeMediGuideLLM()
    chain = chains.build_assessment_chain(llm)
    chains.run_assessment(chain, inputs)
    chains.run_assessment(chain, {**inputs, "severity": 9})
    assert llm.calls == 2


def test_cache_off_always_calls_the_model(inputs):
    assert cache_manager.configure_cache(config.CACHE_OFF) is None
    llm = FakeMediGuideLLM()
    chain = chains.build_assessment_chain(llm)
    chains.run_assessment(chain, inputs)
    chains.run_assessment(chain, inputs)
    assert llm.calls == 2


def test_sqlite_cache_survives_a_restart(inputs):
    cache_manager.configure_cache(config.CACHE_SQLITE)
    llm = FakeMediGuideLLM()
    chains.run_assessment(chains.build_assessment_chain(llm), inputs)
    assert config.SQLITE_PATH.exists()

    cache_manager._CACHES.clear()  # simulate closing and re-opening the app
    cache_manager.configure_cache(config.CACHE_SQLITE)
    new_llm = FakeMediGuideLLM()
    chains.run_assessment(chains.build_assessment_chain(new_llm), inputs)
    assert new_llm.calls == 0, "SQLiteCache should still know the answer"


def test_in_memory_cache_is_lost_on_restart(inputs):
    cache_manager.configure_cache(config.CACHE_MEMORY)
    chains.run_assessment(chains.build_assessment_chain(FakeMediGuideLLM()), inputs)

    cache_manager._CACHES.clear()  # simulate restart
    cache_manager.configure_cache(config.CACHE_MEMORY)
    new_llm = FakeMediGuideLLM()
    chains.run_assessment(chains.build_assessment_chain(new_llm), inputs)
    assert new_llm.calls == 1


def test_clear_cache_forgets_answers(inputs):
    cache_manager.configure_cache(config.CACHE_MEMORY)
    llm = FakeMediGuideLLM()
    chain = chains.build_assessment_chain(llm)
    chains.run_assessment(chain, inputs)
    cache_manager.clear_cache(config.CACHE_MEMORY)
    chains.run_assessment(chain, inputs)
    assert llm.calls == 2


def test_use_cache_false_bypasses_the_cache(inputs):
    cache_manager.configure_cache(config.CACHE_MEMORY)
    cached_llm = FakeMediGuideLLM()
    chains.run_assessment(chains.build_assessment_chain(cached_llm), inputs)

    bypass_llm = FakeMediGuideLLM(cache=False)
    chains.run_assessment(chains.build_assessment_chain(bypass_llm), inputs)
    assert bypass_llm.calls == 1


def test_build_llm_reads_key_from_argument_only():
    llm = chains.build_llm("sk-test-key", "gpt-4o-mini", 0.2)
    assert llm.model_name == "gpt-4o-mini"
    assert llm.openai_api_key.get_secret_value() == "sk-test-key"
