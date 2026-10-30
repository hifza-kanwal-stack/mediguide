# 🩺 MediGuide AI — AI-Powered Medical Symptom Assessment & Patient Guidance Assistant

A **LangChain + Streamlit** programming assignment (Course module: *Building LLM Applications with LangChain*).

> ## ⚠️ Medical & Safety Notice
> MediGuide AI is an **educational AI prototype only**. It is **not** a doctor, **not** a replacement for
> professional diagnosis, treatment or emergency services, and it **never presents a confirmed diagnosis**.
> Always consult a qualified healthcare professional, and **seek emergency help immediately** in urgent situations.
> This project is not a medical device and must not be used for real diagnosis or treatment.

The user enters age, gender, symptoms, duration, severity, existing conditions, medications and notes.
A LangChain pipeline sends this to an OpenAI model and renders a safety-focused dashboard:
summary, possible conditions (education only), **urgency level** (LOW / MEDIUM / HIGH / EMERGENCY),
next steps, questions for a doctor and warning signs — plus a **live-streamed** guidance note.

---

## ✨ Features

| Area | What you get |
|---|---|
| **Inputs** | `text_input` (age), `selectbox` (gender, duration, language, model), `multiselect` + free text (symptoms), `slider` (severity 1–10), 3× `text_area`, `button` |
| **Dashboard** | `st.metric`, `st.warning`, `st.info`, `st.error`, `st.success`, `st.expander`, tabs, columns, coloured urgency banner |
| **Streaming** | `llm.stream()` → `st.write_stream()` (typing effect) |
| **Structured JSON** | JSON mode + fence stripping + safe `json.loads` + schema normalisation — invalid JSON never crashes the app |
| **Caching** | `InMemoryCache` **and** `SQLiteCache`, switchable in the sidebar, with live hit/miss counters |
| **Safety** | Disclaimers in sidebar, main page and results · safety rules in the system prompt · **rule-based safety net** that can raise (never lower) the urgency · emergency message |
| **Languages** | 11 answer languages (English, Urdu, Hindi, Arabic, …) with right-to-left layout for Urdu/Arabic |
| **Bonus** | Session history + urgency analytics chart · Markdown report download · follow-up chat (uses `AIMessage`) · LangChain Lab tab · demo scenarios · Dockerfile · automated tests |

---

## 🗂️ Project structure

```
medical_ai_assistant/
├── app.py                  # Streamlit UI  (run this)
├── requirements.txt
├── .env.example            # copy to .env and add your key
├── README.md
├── Dockerfile              # bonus
├── pytest.ini
├── .streamlit/config.toml  # theme
├── src/
│   ├── __init__.py
│   ├── config.py           # settings + form options + safety constants
│   ├── prompts.py          # PromptTemplate + ChatPromptTemplate + JSON schema
│   ├── chains.py           # ChatOpenAI, LLMChain, streaming, raw messages
│   ├── cache_manager.py    # InMemoryCache + SQLiteCache
│   └── utils.py            # validation, safe JSON parsing, safety net
├── tests/                  # 61 automated tests (fake model — no API key needed)
└── docs/
    └── Medical_AI_Assignment.pdf
```

**Data flow:** form input → build prompt inputs → select cache → build LLM + `LLMChain` → run chain →
parse JSON → safety rules → stream narrative → render dashboard.

---

## 🚀 Setup & run

**Requirements:** Python 3.10 or newer and an OpenAI API key.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows:  .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
cp .env.example .env               # Windows:  copy .env.example .env
#    then open .env and paste:  OPENAI_API_KEY=sk-...

# 4. Run the app
streamlit run app.py
```

### 🔑 API key
1. Create an account at <https://platform.openai.com> and generate a key under **API keys**.
2. Paste it into `.env` as `OPENAI_API_KEY=sk-...`.
3. `.env` is listed in `.gitignore` — **never commit it**. The key is read with `python-dotenv` and is never hard-coded.
   (You can also paste a key in the sidebar for the current session only; it is never saved.)

### 🐳 Docker (bonus)
```bash
docker build -t mediguide-ai .
docker run -p 8501:8501 --env-file .env mediguide-ai
```

---

## 🧠 LangChain concepts — where to find them

| Concept | File / function |
|---|---|
| `ChatOpenAI` | `src/chains.py` → `build_llm()` |
| `PromptTemplate` (one string + variables) | `src/prompts.py` → `ASSESSMENT_PROMPT`, `NARRATIVE_PROMPT` |
| `ChatPromptTemplate` (System + Human) | `src/prompts.py` → `ASSESSMENT_CHAT_TEMPLATE`, `NARRATIVE_CHAT_TEMPLATE` |
| `SystemMessage` / `HumanMessage` / `AIMessage` | `src/chains.py` → `demo_messages()`, `build_conversation_messages()` (follow-up chat sends the earlier guidance as an `AIMessage`) |
| Structured JSON | `prompts.py` (schema) + `chains.py` (JSON mode) + `utils.py` → `parse_assessment()` |
| `LLMChain` | `src/chains.py` → `build_assessment_chain()` / `run_assessment()` |
| Streaming | `src/chains.py` → `stream_narrative()` + `st.write_stream()` in `app.py` |
| Caching | `src/cache_manager.py` → `configure_cache()` (`set_llm_cache`) |

The **🧪 LangChain Lab** tab in the app shows the filled-in `PromptTemplate`, the `ChatPromptTemplate` messages and a raw System/Human/AI conversation live.

---

## 📦 Caching: InMemoryCache vs SQLiteCache

Register a cache **once** with `set_llm_cache(...)`; LangChain then checks it automatically before every model call.
If the same prompt + model settings were seen before, the saved answer is returned instantly — no API call, no cost.

| | **InMemoryCache** | **SQLiteCache** |
|---|---|---|
| Stored in | RAM (memory) | A file on disk (`.cache/langchain_cache.db`) |
| Speed | Fastest | Fast, slightly slower |
| Survives restart? | **No** | **Yes** |
| Best for | One running session | Re-using answers across sessions |

How this project uses them:
* `configure_cache(mode)` in `src/cache_manager.py` registers the chosen cache (or `None` for **Off**).
* The cache objects are created once and re-used, because Streamlit re-runs the script on every click.
* A small `CountingCache` wrapper counts hits and misses, so the UI can show **⚡ from cache** and the tests can prove no second API call was made.
* LangChain's `.stream()` does not read the cache by itself, so `cached_stream()` uses the same cache object manually for the streamed narrative (cache hit → replay instantly, miss → stream live, then save).

**Try it:** press *Analyse symptoms*, then press it again with the same form → the second run shows **⚡ from cache** and is much faster.

---

## 🛡️ Medical safety design

1. **System prompt** forbids confirmed diagnoses and drug dosages, requires recommending a healthcare professional, defines the four urgency levels, tells the model to pick the *higher* level when unsure, and tells it to treat patient text as data (prompt-injection hardening).
2. **Rule-based safety net** (`utils.apply_safety_floor`): after the AI answers, red-flag input can only **raise** the urgency, e.g. chest pain + shortness of breath → `EMERGENCY`, stroke signs, severe bleeding, emergency keywords in free text, severity ≥ 9 → at least `HIGH`. The UI tells the user when this happened.
3. **EMERGENCY** output shows a red banner and "seek emergency help immediately".
4. **Disclaimers** appear in the sidebar, the main page, the results and the downloaded report.
5. **Reliability:** invalid JSON → one automatic retry that bypasses the cache → friendly message + raw output; API errors (key, quota, network) → plain-language messages.
6. **Privacy:** nothing is stored on disk except the optional SQLite cache file. Delete `.cache/` or use *Clear cached answers* to remove it.

---

## ✅ Testing

### Automated (no API key needed — a fake model is used)
```bash
pytest -q
```
The tests cover validation, safe JSON parsing, the safety net, prompts, `LLMChain`, streaming, both caches
(including "SQLite survives a restart, InMemory does not") and the full Streamlit UI via `AppTest`.

### Assignment scenarios (section 18)
Use the **Quick demo scenarios** dropdown in the app, or enter the values by hand.

| # | Input | Expected | Automated test |
|---|---|---|---|
| 1 | Age 25, runny nose + sore throat, 1–3 days, severity 2 | Urgency LOW; calm monitoring advice | `test_scenario_1_mild_cold_is_low` |
| 2 | Age 40, fever + cough, 4–7 days, severity 6 | MEDIUM/HIGH; advises seeing a professional | `test_scenario_2_…` |
| 3 | Severe chest pain + shortness of breath | HIGH/EMERGENCY; urges immediate help | `test_scenario_3_…` (even if the AI wrongly says LOW) |
| 4 | Submit the same form twice (cache on) | Second run faster; identical result | `test_scenario_4_…` (InMemory + SQLite) |
| 5 | Empty symptoms | App warns, **no API call** | `test_scenario_5_…` |
| 6 | Language = Urdu | Guidance returned in Urdu | `test_scenario_6_…` (language reaches the prompt) + live check below |

> Scenario 6 and the real model's wording must be checked live with your own API key — the automated tests
> use a fake model, so they prove the app's logic but not OpenAI's output.

---

## 📤 Submission checklist

- [ ] Public/shared GitHub repo with every file above (`git init`, `git add .`, `git commit`, push)
- [ ] `requirements.txt` and `.env.example` included — **no real key** committed
- [ ] This README (setup, run instructions, caching explanation)
- [ ] Short screen recording / live demo: streaming output + dashboard + cache speed-up
- [ ] Runs with `streamlit run app.py`

---

*Reminder: this project is for education only. It is not a medical device and must not be used for real diagnosis or treatment.*
