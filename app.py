"""
app.py  -  MediGuide AI (Streamlit user interface)
==================================================
Run with:   streamlit run app.py

This file contains ONLY the user interface. All AI / LangChain logic lives in
the `src/` folder:

    src/config.py         settings + form options
    src/prompts.py        PromptTemplate + ChatPromptTemplate + JSON schema
    src/chains.py         ChatOpenAI, LLMChain, streaming, raw messages
    src/cache_manager.py  InMemoryCache + SQLiteCache
    src/utils.py          validation, safe JSON parsing, safety rules

Data flow:
  form input -> build prompt inputs -> select cache -> build LLM + LLMChain
  -> run chain -> parse JSON -> stream narrative -> render dashboard
"""

import json
import time
from datetime import datetime

import streamlit as st

from src import cache_manager, chains, config, prompts, utils

# ==========================================================================
# Page setup + styling
# ==========================================================================
st.set_page_config(
    page_title=f"{config.APP_NAME} — Symptom Guidance",
    page_icon=config.APP_ICON,
    layout="wide",
)

CSS = """
<style>
.mg-hero {background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
          color: #fff; padding: 1.2rem 1.6rem; border-radius: 16px; margin-bottom: .8rem;}
.mg-hero h1 {margin: 0; font-size: 2rem; color: #fff;}
.mg-hero p {margin: .25rem 0 0; opacity: .93; font-size: 1.02rem;}
.mg-urgency {border-radius: 14px; padding: 1rem 1.3rem; color: #fff; margin: .6rem 0 1rem;}
.mg-urgency .lvl {font-size: 1.7rem; font-weight: 800; letter-spacing: .05em;}
.mg-urgency .txt {opacity: .96; margin-top: .15rem;}
</style>
"""

# Right-to-left languages (Urdu, Arabic...) look better aligned to the right.
RTL_CSS = """
<style>
.st-key-results [data-testid="stMarkdownContainer"],
.st-key-results [data-testid="stAlert"] {direction: rtl; text-align: right;}
</style>
"""

FORM_KEYS = [
    "age_text", "gender", "symptoms", "free_symptoms", "duration",
    "severity", "conditions", "medications", "notes",
]


# ==========================================================================
# Session state (Streamlit re-runs this whole file on every click)
# ==========================================================================
def init_state():
    """Give every widget a starting value (done once per browser session)."""
    defaults = {
        "age_text": "",
        "gender": config.GENDER_OPTIONS[0],
        "symptoms": [],
        "free_symptoms": "",
        "duration": config.DURATION_OPTIONS[1],
        "severity": 3,
        "conditions": "",
        "medications": "",
        "notes": "",
        "language": "English",
        "model": config.DEFAULT_MODEL,
        "temperature": config.DEFAULT_TEMPERATURE,
        "cache_mode": config.CACHE_MEMORY,
        "scenario": list(config.DEMO_SCENARIOS)[0],
        "result": None,      # the latest assessment (survives reruns)
        "history": [],       # all assessments of this session
        "followups": [],     # (question, answer) pairs for the latest assessment
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def load_scenario():
    """Callback: fill the form with one of the demo scenarios."""
    scenario = config.DEMO_SCENARIOS.get(st.session_state["scenario"])
    if scenario:
        for key, value in scenario.items():
            st.session_state[key] = value


# ==========================================================================
# Sidebar
# ==========================================================================
def render_sidebar() -> dict:
    """Sidebar: name, description, disclaimer, model config, language."""
    with st.sidebar:
        st.title(f"{config.APP_ICON} {config.APP_NAME}")
        st.caption(config.APP_TAGLINE)
        st.markdown(config.APP_DESCRIPTION)

        st.error(config.LONG_DISCLAIMER)  # medical disclaimer (impossible to miss)
        st.caption(config.EMERGENCY_NUMBERS_HINT)
        st.divider()

        st.subheader("🌐 Language")
        st.selectbox(
            "Answer language",
            config.LANGUAGE_OPTIONS,
            format_func=lambda name: config.LANGUAGES[name],
            key="language",
            help="The guidance text will be written in this language.",
        )

        st.subheader("⚙️ Model configuration")
        st.selectbox("OpenAI model", config.MODEL_OPTIONS, key="model")
        st.slider(
            "Temperature",
            0.0, 1.0, step=0.05, key="temperature",
            help="Lower = more careful and consistent. Higher = more varied wording.",
        )

        st.radio(
            "Caching",
            config.CACHE_MODES,
            key="cache_mode",
            help="InMemory = RAM (fast, lost on restart). SQLite = file on disk (survives restarts).",
        )
        stats_slot = st.empty()
        if st.button("🧹 Clear cached answers"):
            cache_manager.clear_cache(st.session_state["cache_mode"])
            st.toast("Cache cleared.")

        st.subheader("🔑 OpenAI API key")
        env_key = config.get_api_key()
        if env_key:
            st.success("Key loaded from your .env file.")
        else:
            st.warning("No key found in .env. Add OPENAI_API_KEY there, or paste a key below.")
        with st.expander("Use a different key (this session only)"):
            st.text_input("API key", type="password", key="api_key_input", placeholder="sk-...")
            st.caption("Never saved to disk and never written into the code.")

    api_key = (st.session_state.get("api_key_input") or "").strip() or env_key
    return {
        "api_key": api_key,
        "model": st.session_state["model"],
        "temperature": st.session_state["temperature"],
        "cache_mode": st.session_state["cache_mode"],
        "language": st.session_state["language"],
        "stats_slot": stats_slot,
    }


def update_cache_stats(settings: dict):
    """Refresh the cache counters in the sidebar (called at the end of every run)."""
    stats = cache_manager.get_stats()
    if settings["cache_mode"] == config.CACHE_OFF:
        settings["stats_slot"].caption("Caching is off.")
    else:
        settings["stats_slot"].caption(
            f"📦 {settings['cache_mode']} cache • ⚡ hits: {stats['hits']} • misses: {stats['misses']}"
        )


# ==========================================================================
# Main form
# ==========================================================================
def render_form(settings: dict) -> bool:
    """Draw the input form. Returns True when the Analyse button was pressed."""
    st.subheader("📝 Tell us about your symptoms")

    st.selectbox(
        "Quick demo scenarios (optional)",
        list(config.DEMO_SCENARIOS),
        key="scenario",
        on_change=load_scenario,
        help="Fills the form with an example so you can test the app quickly.",
    )

    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            st.text_input("Patient age (years)", key="age_text", placeholder="e.g. 32", max_chars=6)
            st.selectbox("Gender", config.GENDER_OPTIONS, key="gender")
            st.selectbox("How long have you had the symptoms?", config.DURATION_OPTIONS, key="duration")
            st.slider("Severity (1 = very mild, 10 = worst)", 1, 10, key="severity")
        with right:
            st.multiselect(
                "Symptoms (choose one or more)",
                config.SYMPTOM_OPTIONS,
                key="symptoms",
                placeholder="Select symptoms…",
            )
            st.text_input(
                "Other symptoms (optional free text)",
                key="free_symptoms",
                placeholder="e.g. sensitivity to light",
                max_chars=300,
            )
            st.info(f"Answer language: **{config.LANGUAGES[settings['language']]}** (change it in the sidebar)")

        c1, c2, c3 = st.columns(3)
        c1.text_area("Existing medical conditions", key="conditions", height=110,
                     max_chars=config.MAX_FREE_TEXT_CHARS, placeholder="e.g. asthma, diabetes")
        c2.text_area("Current medications", key="medications", height=110,
                     max_chars=config.MAX_FREE_TEXT_CHARS, placeholder="e.g. none")
        c3.text_area("Additional notes", key="notes", height=110,
                     max_chars=config.MAX_FREE_TEXT_CHARS, placeholder="Anything else that may be important")

    return st.button("🔍 Analyse symptoms", type="primary", key="analyse")


def current_inputs(language: str, fallback_age: str = "30") -> dict:
    """Build the prompt inputs from the form values currently on screen."""
    s = st.session_state
    age_text = s["age_text"] if utils.parse_age(s["age_text"]) is not None else fallback_age
    return utils.build_inputs(
        age_text, s["gender"], s["symptoms"], s["free_symptoms"], s["duration"],
        s["severity"], s["conditions"], s["medications"], s["notes"], language,
    )


# ==========================================================================
# Running the analysis
# ==========================================================================
def sqlite_location() -> str:
    """Where the SQLite cache file lives (shown relative to the project when possible)."""
    try:
        return str(config.SQLITE_PATH.relative_to(config.BASE_DIR))
    except ValueError:
        return str(config.SQLITE_PATH)


def was_cached(before: dict, after: dict) -> bool:
    """True when every cache lookup between the two snapshots was a HIT."""
    return after["misses"] == before["misses"] and after["hits"] > before["hits"]


def run_analysis(settings: dict):
    """The full pipeline, triggered by the Analyse button."""
    s = st.session_state

    # 1. Validate (empty symptoms => warn and do NOT call the API)
    errors = utils.validate_inputs(s["age_text"], s["symptoms"], s["free_symptoms"])
    if errors:
        for message in errors:
            st.warning(message)
        return "stopped"
    if not settings["api_key"]:
        st.error("No OpenAI API key found. Add OPENAI_API_KEY to your .env file or paste one in the sidebar.")
        return "stopped"

    # 2. Build prompt inputs + choose the cache
    inputs = utils.build_inputs(
        s["age_text"], s["gender"], s["symptoms"], s["free_symptoms"], s["duration"],
        s["severity"], s["conditions"], s["medications"], s["notes"], settings["language"],
    )
    cache_manager.configure_cache(settings["cache_mode"])
    stats_before = cache_manager.get_stats()

    # 3. Build LLM + LLMChain and run it (structured JSON)
    started = time.perf_counter()
    try:
        with st.spinner("🔎 Analysing your symptoms…"):
            llm = chains.build_llm(settings["api_key"], settings["model"], settings["temperature"])
            chain = chains.build_assessment_chain(llm)
            raw = chains.run_assessment(chain, inputs)
            parsed = utils.parse_assessment(raw)
            if not parsed.ok:
                # One automatic retry that skips the cache (a bad answer could have been cached).
                retry_llm = chains.build_llm(
                    settings["api_key"], settings["model"], settings["temperature"], use_cache=False
                )
                raw = chains.run_assessment(chains.build_assessment_chain(retry_llm), inputs)
                parsed = utils.parse_assessment(raw)
    except Exception as exc:  # network, key, quota... never crash the page
        st.error("❌ " + utils.friendly_api_error(exc))
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")
        return "stopped"
    assessment_seconds = time.perf_counter() - started
    stats_after = cache_manager.get_stats()

    # 4. Invalid JSON => friendly message + raw output, no crash
    if not parsed.ok:
        st.session_state["result"] = {"status": "parse_error", "error": parsed.error, "raw": raw}
        return "parse_error"

    # 5. Safety net: the AI can never go below the rule-based urgency floor
    ai_level = parsed.data["urgency_level"]
    data, floor_reasons = utils.apply_safety_floor(
        parsed.data, s["symptoms"],
        f"{s['free_symptoms']} {s['notes']} {s['conditions']}", s["severity"],
    )

    result = {
        "status": "ok",
        "inputs": inputs,
        "data": data,
        "ai_level": ai_level,
        "floor_reasons": floor_reasons,
        "notes": parsed.notes,
        "language": settings["language"],
        "model": settings["model"],
        "assessment_seconds": assessment_seconds,
        "assessment_cached": was_cached(stats_before, stats_after),
        "assessment_json": json.dumps(data, ensure_ascii=False),
        "narrative": None,
    }
    st.session_state["result"] = result
    st.session_state["followups"] = []

    # 6. Stream the narrative live while the dashboard is drawn
    narrative_inputs = {
        **inputs,
        "urgency_level": data["urgency_level"],
        "assessment_summary": data["summary"] or "(no summary)",
    }
    stream_llm = chains.build_llm(settings["api_key"], settings["model"], settings["temperature"])
    render_results(
        result,
        narrative_source=lambda: chains.stream_narrative_cached(stream_llm, narrative_inputs),
    )

    # 7. Save to the session history (used by the analytics tab)
    st.session_state["history"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "age": inputs["age"],
        "symptoms": inputs["symptoms"],
        "severity": inputs["severity"],
        "urgency": data["urgency_level"],
        "language": settings["language"],
        "cached": result["assessment_cached"],
    })
    return "drawn"


# ==========================================================================
# Results dashboard
# ==========================================================================
def urgency_banner(level: str):
    """Big coloured box: green / amber / orange / red."""
    st.markdown(
        f"""<div class="mg-urgency" style="background:{config.URGENCY_COLORS[level]}">
              <div class="lvl">{config.URGENCY_ICONS[level]} URGENCY: {level}</div>
              <div class="txt">{config.URGENCY_MEANING[level]}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def render_results(result: dict, narrative_source=None):
    """
    Draw the dashboard. If `narrative_source` is given, the narrative is
    streamed live with st.write_stream(); otherwise the saved text is shown.
    """
    data, inputs = result["data"], result["inputs"]
    level = data["urgency_level"]

    if result["language"] in config.RTL_LANGUAGES:
        st.markdown(RTL_CSS, unsafe_allow_html=True)

    with st.container(key="results"):
        st.header("📋 Results dashboard")

        for note in result.get("notes", []):
            st.info(note)
        if result.get("floor_reasons"):
            st.warning(
                f"🛡️ **Safety rules applied.** The AI suggested {result['ai_level']}, but urgency was "
                f"raised to **{level}** because " + "; ".join(result["floor_reasons"]) + "."
            )

        # (4) urgency + key numbers
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Urgency level", f"{config.URGENCY_ICONS[level]} {level}")
        c2.metric("Severity", f"{inputs['severity']}/10")
        c3.metric("Duration", inputs["duration"])
        with c4:
            time_slot = st.empty()
        time_slot.metric(
            "AI response time",
            f"{result['assessment_seconds']:.2f}s",
            delta="⚡ from cache" if result["assessment_cached"] else None,
            delta_color="off",
        )

        urgency_banner(level)
        if level == "EMERGENCY":
            st.error(config.EMERGENCY_MESSAGE)
        elif level == "HIGH":
            st.warning("🟠 Please seek in-person medical care **today**. This is not a diagnosis.")

        # Live narrative box (filled after the tabs exist, but displayed above them)
        narrative_box = st.container(border=True)

        # (1) – (7) structured details
        tabs = st.tabs([
            "👤 Patient summary", "ℹ️ General information", "🧬 Possible conditions",
            "🚦 Next steps", "❓ Questions for your doctor", "🚨 Warning signs",
        ])

        with tabs[0]:  # (1) patient symptom summary
            a, b = st.columns(2)
            a.markdown(
                f"**Age:** {inputs['age']}  \n**Gender:** {inputs['gender']}  \n"
                f"**Duration:** {inputs['duration']}  \n**Severity:** {inputs['severity']}/10"
            )
            b.markdown(
                f"**Symptoms:** {inputs['symptoms']}  \n**Existing conditions:** {inputs['conditions']}  \n"
                f"**Medications:** {inputs['medications']}  \n**Notes:** {inputs['notes']}"
            )

        with tabs[1]:  # (2) AI-generated general information
            st.info(data["summary"] or "No general information was returned.")

        with tabs[2]:  # (3) possible conditions (education only)
            st.warning("These are **possible** conditions for **education only**. They are NOT a diagnosis.")
            if data["possible_conditions"]:
                for i, item in enumerate(data["possible_conditions"], 1):
                    with st.expander(f"{i}. {item['name']}", expanded=(i == 1)):
                        st.write(item["reason"] or "No explanation provided.")
            else:
                st.write("No possible conditions were returned.")

        with tabs[3]:  # (5) recommended next steps
            if level in ("HIGH", "EMERGENCY"):
                st.error("Seek urgent medical care first — the steps below come after that.")
            for i, step in enumerate(data["recommended_next_steps"], 1):
                st.success(f"**{i}.** {step}")

        with tabs[4]:  # (6) questions for a healthcare professional
            for question in data["questions_for_doctor"]:
                st.markdown(f"- ❓ {question}")

        with tabs[5]:  # (7) warning signs
            st.error("Get immediate medical attention if you notice any of these:")
            for sign in data["warning_signs"]:
                st.markdown(f"- 🚨 {sign}")

        # Live narrative (streamed with st.write_stream)
        with narrative_box:
            st.subheader("💬 Guidance note")
            if narrative_source is not None:
                narrative_started = time.perf_counter()
                before = cache_manager.get_stats()
                try:
                    result["narrative"] = st.write_stream(narrative_source())
                except Exception as exc:
                    result["narrative"] = ""
                    st.error("❌ " + utils.friendly_api_error(exc))
                after = cache_manager.get_stats()
                result["narrative_seconds"] = time.perf_counter() - narrative_started
                result["narrative_cached"] = was_cached(before, after)
            elif result.get("narrative"):
                st.markdown(result["narrative"])

            if result.get("narrative_seconds") is not None:
                source = "⚡ replayed from cache" if result.get("narrative_cached") else "streamed live from OpenAI"
                st.caption(f"{source} in {result['narrative_seconds']:.2f}s")
                total = result["assessment_seconds"] + result["narrative_seconds"]
                time_slot.metric(
                    "Total time", f"{total:.2f}s",
                    delta="⚡ from cache" if (result["assessment_cached"] and result.get("narrative_cached")) else None,
                    delta_color="off",
                )

        st.warning(config.SHORT_DISCLAIMER)  # disclaimer inside the results

        st.download_button(
            "⬇️ Download this report (.md)",
            data=utils.build_markdown_report(result),
            file_name=f"mediguide_report_{datetime.now():%Y%m%d_%H%M}.md",
            mime="text/markdown",
        )


def render_parse_error(result: dict):
    """Friendly message + raw output when the JSON could not be parsed."""
    st.error(
        "😕 The AI answered in a format we could not read, so no dashboard can be shown. "
        "Please press **Analyse symptoms** again (or switch caching off / clear the cache)."
    )
    st.caption(f"Reason: {result['error']}")
    with st.expander("Raw model output (for debugging)"):
        st.code(result["raw"] or "(empty)", language="text")


# ==========================================================================
# Follow-up questions (raw System / Human / AI messages)
# ==========================================================================
def render_followup(settings: dict, result: dict):
    """Ask a follow-up question. The earlier guidance is sent as an AIMessage."""
    st.divider()
    st.subheader("💭 Ask a follow-up question")
    st.caption("Educational answers only. The earlier guidance is sent to the model as an AIMessage, "
               "so it remembers the conversation.")

    for question, answer in st.session_state["followups"]:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)

    question = st.text_input("Your question", key="followup_question",
                             placeholder="e.g. What can I do at home to feel better?", max_chars=300)
    if st.button("Ask", key="ask_followup"):
        if not question.strip():
            st.warning("Please type a question first.")
            return
        if not settings["api_key"]:
            st.error("No OpenAI API key found.")
            return
        cache_manager.configure_cache(settings["cache_mode"])
        llm = chains.build_llm(settings["api_key"], settings["model"], settings["temperature"])
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            try:
                answer = st.write_stream(chains.stream_followup(
                    llm, result["inputs"], result["assessment_json"],
                    st.session_state["followups"], utils.clean_text(question, 300),
                ))
                st.session_state["followups"].append((question, answer))
            except Exception as exc:
                st.error("❌ " + utils.friendly_api_error(exc))
        st.caption("Educational information only — consult a healthcare professional.")


# ==========================================================================
# Other tabs
# ==========================================================================
def render_assessment_tab(settings: dict):
    analyse_clicked = render_form(settings)
    st.write("")

    result = st.session_state["result"]
    show_followup = False

    if analyse_clicked:
        outcome = run_analysis(settings)  # "drawn" | "parse_error" | "stopped"
        result = st.session_state["result"]
        if outcome == "parse_error":
            render_parse_error(result)
        show_followup = outcome == "drawn"
        # "drawn": the dashboard was already drawn (and streamed) by run_analysis().
        # "stopped": a warning / error is already on screen, so old results are hidden.
    elif result:
        if result.get("status") == "parse_error":
            render_parse_error(result)
        else:
            render_results(result)
            show_followup = True
    else:
        st.info("👆 Fill in the form and press **Analyse symptoms** to see your results dashboard.")

    if show_followup:
        render_followup(settings, result)

    st.write("")
    st.warning(config.SHORT_DISCLAIMER)


def render_lab_tab(settings: dict):
    """A teaching tab that shows the LangChain building blocks in action."""
    st.subheader("🧪 LangChain Lab")
    st.caption("See exactly what the app sends to the model. Uses the values currently in the form.")
    inputs = current_inputs(settings["language"])

    st.markdown("#### 1️⃣ PromptTemplate — one string with variables")
    with st.expander("Template (with {placeholders})"):
        st.code(prompts.ASSESSMENT_PROMPT.template, language="text")
    with st.expander("Filled-in prompt", expanded=True):
        st.code(prompts.ASSESSMENT_PROMPT.format(**{k: inputs[k] for k in prompts.PATIENT_VARIABLES}), language="text")

    st.markdown("#### 2️⃣ ChatPromptTemplate — System + Human messages")
    messages = prompts.ASSESSMENT_CHAT_TEMPLATE.format_messages(**{k: inputs[k] for k in prompts.PATIENT_VARIABLES})
    for message in messages:
        with st.expander(f"{message.type.upper()} message ({len(message.content)} characters)"):
            st.code(message.content, language="text")

    st.markdown("#### 3️⃣ Raw SystemMessage / HumanMessage / AIMessage")
    question = st.text_input("Follow-up question for the demo conversation", key="lab_question",
                             value="Should I be worried if it continues for a week?", max_chars=200)
    for message in chains.demo_messages(question):
        role = {"system": "System", "human": "Human", "ai": "AI"}[message.type]
        if message.type == "system":
            st.caption("🛠️ SystemMessage → sets the role and the safety rules (hidden from the chat below)")
        else:
            with st.chat_message("assistant" if message.type == "ai" else "user"):
                st.markdown(f"**{role}Message:** {message.content}")
    if st.button("Send this conversation to the model", key="lab_run"):
        if not settings["api_key"]:
            st.error("No OpenAI API key found.")
        else:
            cache_manager.configure_cache(settings["cache_mode"])
            try:
                with st.spinner("Calling the model…"):
                    llm = chains.build_llm(settings["api_key"], settings["model"], settings["temperature"])
                    _, answer = chains.demo_raw_messages(llm, question)
                with st.chat_message("assistant"):
                    st.write(answer)
            except Exception as exc:
                st.error("❌ " + utils.friendly_api_error(exc))

    st.markdown("#### 4️⃣ Caching — InMemoryCache vs SQLiteCache")
    st.table({
        "Feature": [row[0] for row in cache_manager.CACHE_COMPARISON],
        "InMemoryCache": [row[1] for row in cache_manager.CACHE_COMPARISON],
        "SQLiteCache": [row[2] for row in cache_manager.CACHE_COMPARISON],
    })
    st.info(
        "Try it: choose a cache in the sidebar, press **Analyse symptoms**, then press it again with the "
        "same form. The second run is answered from the cache (look for **⚡ from cache**). "
        f"SQLite answers are stored in `{sqlite_location()}` and survive a restart."
    )

    st.markdown("#### 5️⃣ Pipeline")
    st.code(
        "form input → build prompt inputs → select cache → build LLM + LLMChain\n"
        "→ run chain → parse JSON → safety rules → stream narrative → render dashboard",
        language="text",
    )


def render_history_tab():
    st.subheader("🕘 Session history & analytics")
    history = st.session_state["history"]
    if not history:
        st.info("No assessments yet in this session. Run one from the Assessment tab.")
        st.warning(config.SHORT_DISCLAIMER)
        return

    distribution = utils.urgency_distribution(history)
    cols = st.columns(4)
    for col, level in zip(cols, config.URGENCY_LEVELS):
        col.metric(f"{config.URGENCY_ICONS[level]} {level}", distribution[level])

    st.markdown("**Urgency distribution**")
    # Numbers in front of the labels keep the bars in LOW → EMERGENCY order.
    st.bar_chart({f"{i} {level}": distribution[level] for i, level in enumerate(config.URGENCY_LEVELS, 1)})

    st.markdown("**Past assessments (newest first)**")
    st.dataframe(list(reversed(history)), hide_index=True)
    if st.button("Clear history"):
        st.session_state["history"] = []
        st.rerun()
    st.warning(config.SHORT_DISCLAIMER)


def render_about_tab():
    st.subheader("ℹ️ About this project")
    st.markdown(
        """
**MediGuide AI** is a LangChain + Streamlit teaching prototype.

| LangChain concept | Where it is used |
|---|---|
| `ChatOpenAI` | `src/chains.py → build_llm()` |
| `PromptTemplate` | `src/prompts.py → ASSESSMENT_PROMPT` |
| `ChatPromptTemplate` | `src/prompts.py → ASSESSMENT_CHAT_TEMPLATE` |
| System / Human / AI messages | `src/chains.py → demo_messages()`, follow-up chat |
| Structured JSON | JSON mode + `src/utils.py → parse_assessment()` |
| `LLMChain` | `src/chains.py → build_assessment_chain()` |
| Streaming | `llm.stream()` + `st.write_stream()` |
| Caching | `InMemoryCache` + `SQLiteCache` in `src/cache_manager.py` |

**Safety design**
- The system prompt forbids confirmed diagnoses and drug dosages.
- A rule-based safety net raises the urgency for red-flag symptoms (for example chest pain + shortness of breath), whatever the AI says.
- Invalid JSON never crashes the app — you get a friendly message and the raw output.
- Disclaimers appear in the sidebar, the main area and the results.
        """
    )
    st.error(config.LONG_DISCLAIMER)


# ==========================================================================
# Main
# ==========================================================================
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    settings = render_sidebar()

    st.markdown(
        f"""<div class="mg-hero"><h1>{config.APP_ICON} {config.APP_NAME}</h1>
            <p>{config.APP_TAGLINE} — educational prototype</p></div>""",
        unsafe_allow_html=True,
    )
    st.warning(config.SHORT_DISCLAIMER)

    tab_assess, tab_lab, tab_history, tab_about = st.tabs(
        ["🩺 Assessment", "🧪 LangChain Lab", "🕘 History & analytics", "ℹ️ About"]
    )
    with tab_assess:
        render_assessment_tab(settings)
    with tab_lab:
        render_lab_tab(settings)
    with tab_history:
        render_history_tab()
    with tab_about:
        render_about_tab()

    update_cache_stats(settings)


main()
