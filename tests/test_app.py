"""
End-to-end tests of the Streamlit UI using Streamlit's AppTest framework.
The OpenAI model is replaced by a fake, so no API key or internet is needed.
The scenario numbers match the "Testing Scenarios" table in the assignment (section 18).
"""

from pathlib import Path

import pytest
from langchain_core.globals import set_llm_cache
from streamlit.testing.v1 import AppTest

from src import cache_manager, chains, config
from tests.fakes import FakeMediGuideLLM

APP_FILE = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def llm(monkeypatch, tmp_path):
    """Fake model + fresh caches + a fake API key for every test."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "ui_cache.db")
    cache_manager._CACHES.clear()
    fake = FakeMediGuideLLM()
    monkeypatch.setattr(chains, "build_llm", lambda *args, **kwargs: fake)
    yield fake
    set_llm_cache(None)
    cache_manager._CACHES.clear()


def new_app() -> AppTest:
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def fill(at, age="25", symptoms=("Runny nose", "Sore throat"), duration="1-3 days", severity=2, gender="Female"):
    at.text_input(key="age_text").set_value(age)
    at.selectbox(key="gender").set_value(gender)
    at.multiselect(key="symptoms").set_value(list(symptoms))
    at.selectbox(key="duration").set_value(duration)
    at.slider(key="severity").set_value(severity)


def analyse(at):
    at.button(key="analyse").click().run()
    assert not at.exception
    return at


def texts(elements):
    return " ".join(str(e.value) for e in elements)


# --------------------------------------------------------------- first screen
def test_first_screen_has_disclaimers_and_required_widgets(llm):
    at = new_app()
    assert "NOT a doctor" in texts(at.warning)           # main area disclaimer
    assert "MEDICAL DISCLAIMER" in texts(at.sidebar.error)  # sidebar disclaimer
    assert at.text_input(key="age_text") and at.multiselect(key="symptoms")
    assert at.slider(key="severity") and at.button(key="analyse")
    assert len(at.text_area) == 3
    assert at.sidebar.selectbox(key="language") and at.sidebar.selectbox(key="model")
    assert llm.calls == 0


# --------------------------------------------------------------- scenarios
def test_scenario_1_mild_cold_is_low(llm):
    at = new_app()
    fill(at)
    analyse(at)
    result = at.session_state["result"]
    assert result["status"] == "ok"
    assert result["data"]["urgency_level"] == "LOW"
    assert "Urgency level" in [m.label for m in at.metric]
    assert result["narrative"]                           # narrative was streamed
    assert "educational" in texts(at.warning).lower()    # disclaimer inside the results


def test_scenario_2_fever_and_cough_medium_or_high(llm):
    llm.json_text = llm.json_text.replace('"LOW"', '"MEDIUM"')
    at = new_app()
    fill(at, age="40", symptoms=("Fever", "Cough"), duration="4-7 days", severity=6, gender="Male")
    analyse(at)
    assert at.session_state["result"]["data"]["urgency_level"] in ("MEDIUM", "HIGH")


def test_scenario_3_chest_pain_is_raised_to_emergency_even_if_ai_says_low(llm):
    at = new_app()  # the fake AI wrongly answers LOW
    fill(at, age="55", symptoms=("Chest pain", "Shortness of breath"), duration="Less than 1 day", severity=9)
    analyse(at)
    result = at.session_state["result"]
    assert result["ai_level"] == "LOW"
    assert result["data"]["urgency_level"] == "EMERGENCY"
    assert "immediately" in texts(at.error).lower()
    assert "Safety rules applied" in texts(at.warning)


def test_scenario_4_same_form_twice_second_run_is_cached(llm):
    at = new_app()
    fill(at)
    analyse(at)
    calls_after_first = llm.calls
    first = at.session_state["result"]
    assert first["assessment_cached"] is False

    analyse(at)
    second = at.session_state["result"]
    assert llm.calls == calls_after_first, "no new model call on the 2nd run"
    assert second["assessment_cached"] is True and second["narrative_cached"] is True
    assert second["data"] == first["data"] and second["narrative"] == first["narrative"]


def test_scenario_4b_cache_off_calls_model_every_time(llm):
    at = new_app()
    at.sidebar.radio(key="cache_mode").set_value(config.CACHE_OFF)
    fill(at)
    analyse(at)
    first_calls = llm.calls
    analyse(at)
    assert llm.calls > first_calls


def test_scenario_4c_sqlite_cache_via_ui(llm):
    at = new_app()
    at.sidebar.radio(key="cache_mode").set_value(config.CACHE_SQLITE)
    fill(at)
    analyse(at)
    calls = llm.calls
    analyse(at)
    assert llm.calls == calls and config.SQLITE_PATH.exists()


def test_scenario_5_empty_symptoms_warns_and_never_calls_api(llm):
    at = new_app()
    fill(at, symptoms=())
    analyse(at)
    assert "no request was sent" in texts(at.warning)
    assert llm.calls == 0
    assert at.session_state["result"] is None


def test_scenario_6_urdu_language_is_passed_through(llm):
    at = new_app()
    at.sidebar.selectbox(key="language").set_value("Urdu")
    fill(at)
    analyse(at)
    result = at.session_state["result"]
    assert result["language"] == "Urdu" and result["inputs"]["language"] == "Urdu"


# --------------------------------------------------------------- reliability
@pytest.mark.parametrize("bad", ["this is not json", "{ oops", "", "[1,2]"])
def test_invalid_json_shows_friendly_error_not_a_crash(llm, bad):
    llm.json_text = bad
    at = new_app()
    fill(at)
    analyse(at)
    assert at.session_state["result"]["status"] == "parse_error"
    assert "could not read" in texts(at.error)
    assert not at.exception


def test_json_inside_code_fences_still_works(llm):
    llm.json_text = "```json\n" + llm.json_text + "\n```"
    at = new_app()
    fill(at)
    analyse(at)
    assert at.session_state["result"]["status"] == "ok"


def test_api_error_is_friendly(llm, monkeypatch):
    class AuthenticationError(Exception):
        pass

    def boom(*args, **kwargs):
        raise AuthenticationError("401 invalid_api_key")

    monkeypatch.setattr(chains, "run_assessment", boom)
    at = new_app()
    fill(at)
    analyse(at)
    assert "API key" in texts(at.error)
    assert not at.exception


def test_missing_api_key_message(llm, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    at = new_app()
    fill(at)
    analyse(at)
    assert "No OpenAI API key" in texts(at.error)
    assert llm.calls == 0


def test_invalid_age_is_rejected(llm):
    at = new_app()
    fill(at, age="abc")
    analyse(at)
    assert "valid age" in texts(at.warning)
    assert llm.calls == 0


# --------------------------------------------------------------- extras
def test_demo_scenario_selector_fills_the_form(llm):
    at = new_app()
    at.selectbox(key="scenario").select("3) Chest pain + breathlessness").run()
    assert at.multiselect(key="symptoms").value == ["Chest pain", "Shortness of breath"]
    assert at.text_input(key="age_text").value == "55"


def test_history_collects_each_run(llm):
    at = new_app()
    fill(at)
    analyse(at)
    analyse(at)
    assert [h["urgency"] for h in at.session_state["history"]] == ["LOW", "LOW"]


def test_followup_question_uses_ai_message_history(llm):
    at = new_app()
    fill(at)
    analyse(at)
    at.text_input(key="followup_question").set_value("What can I do at home?")
    at.button(key="ask_followup").click().run()
    assert not at.exception
    assert len(at.session_state["followups"]) == 1


def test_lab_tab_runs_without_api_call(llm):
    at = new_app()
    assert not at.exception
    assert llm.calls == 0
