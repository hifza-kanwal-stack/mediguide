"""
config.py
=========
Central place for settings, form options and safety constants.

Everything the rest of the app needs to "know" lives here, so the other
files stay clean. Nothing in this file calls the OpenAI API.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Paths + environment
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# Read the .env file (if it exists) so OPENAI_API_KEY becomes available via
# os.getenv(). The key is NEVER written in the source code.
load_dotenv(BASE_DIR / ".env")

PLACEHOLDER_KEYS = {"", "sk-your-key-here", "your-key-here", "sk-..."}


def get_api_key() -> str:
    """Return the OpenAI API key from the environment ('' if missing)."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return "" if key in PLACEHOLDER_KEYS else key


# --------------------------------------------------------------------------
# App identity
# --------------------------------------------------------------------------
APP_NAME = "MediGuide AI"
APP_ICON = "🩺"
APP_TAGLINE = "AI-powered medical symptom assessment & patient guidance"

APP_DESCRIPTION = (
    "Enter your age, symptoms, how long you have had them and how severe they "
    "feel. MediGuide AI uses LangChain + an OpenAI model to produce structured, "
    "safety-focused **preliminary guidance**: an urgency level, next steps, "
    "questions for your doctor and warning signs."
)

# --------------------------------------------------------------------------
# Medical safety text (shown in the sidebar, main area and results)
# --------------------------------------------------------------------------
SHORT_DISCLAIMER = (
    "⚠️ **Educational AI prototype only.** MediGuide AI is NOT a doctor and is "
    "NOT a replacement for professional diagnosis, treatment or emergency "
    "services. It never gives a confirmed diagnosis. Always consult a qualified "
    "healthcare professional — and seek emergency help immediately in urgent "
    "situations."
)

LONG_DISCLAIMER = (
    "**⚠️ MEDICAL DISCLAIMER**\n\n"
    "This application is an **educational AI prototype**. It is **not** a "
    "replacement for a licensed doctor, professional diagnosis, emergency "
    "service or medical treatment.\n\n"
    "- It **never** presents a confirmed diagnosis.\n"
    "- Always consult a qualified healthcare professional.\n"
    "- If you think this is an emergency, **call your local emergency number "
    "right now** — do not wait for this app."
)

EMERGENCY_MESSAGE = (
    "🚨 **Possible EMERGENCY.** Please seek emergency medical help **immediately**: "
    "call your local emergency number or go to the nearest emergency department. "
    "Do not wait for more information from this app, and do not drive yourself "
    "if you feel unwell."
)

EMERGENCY_NUMBERS_HINT = (
    "Examples of emergency numbers: 911 (USA/Canada), 999 (UK), 112 (EU/Europe), "
    "000 (Australia), 1122 or 115 (Pakistan)."
)

# --------------------------------------------------------------------------
# Model + cache options
# --------------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
if DEFAULT_MODEL not in MODEL_OPTIONS:
    MODEL_OPTIONS.insert(0, DEFAULT_MODEL)

DEFAULT_TEMPERATURE = 0.3

CACHE_OFF = "Off"
CACHE_MEMORY = "In-Memory"
CACHE_SQLITE = "SQLite"
CACHE_MODES = [CACHE_OFF, CACHE_MEMORY, CACHE_SQLITE]

# The SQLite cache is a single file on disk, so it survives app restarts.
SQLITE_PATH = BASE_DIR / ".cache" / "langchain_cache.db"

# --------------------------------------------------------------------------
# Form options
# --------------------------------------------------------------------------
GENDER_OPTIONS = ["Female", "Male", "Non-binary", "Prefer not to say"]

DURATION_OPTIONS = [
    "Less than 1 day",
    "1-3 days",
    "4-7 days",
    "1-2 weeks",
    "2-4 weeks",
    "More than 1 month",
]

# Symptoms the user can pick from (they can also type their own).
SYMPTOM_OPTIONS = [
    "Fever",
    "Cough",
    "Sore throat",
    "Runny nose",
    "Nasal congestion",
    "Headache",
    "Dizziness",
    "Fatigue / weakness",
    "Body aches",
    "Chills",
    "Nausea",
    "Vomiting",
    "Diarrhea",
    "Constipation",
    "Abdominal pain",
    "Chest pain",
    "Shortness of breath",
    "Palpitations",
    "Rash",
    "Joint pain",
    "Back pain",
    "Ear pain",
    "Loss of taste or smell",
    "Painful urination",
    "Swollen glands",
    "Blurred vision",
    "Confusion",
    "Loss of consciousness / fainting",
    "Seizure",
    "Severe bleeding",
    "Slurred speech",
    "Sudden weakness or numbness",
    "Stiff neck",
]

# Answer languages. The key is the English name sent to the model; the
# value is the native name shown in the UI.
LANGUAGES = {
    "English": "English",
    "Urdu": "اردو (Urdu)",
    "Hindi": "हिन्दी (Hindi)",
    "Arabic": "العربية (Arabic)",
    "Punjabi (Shahmukhi)": "پنجابی (Punjabi)",
    "Bengali": "বাংলা (Bengali)",
    "Spanish": "Español (Spanish)",
    "French": "Français (French)",
    "German": "Deutsch (German)",
    "Turkish": "Türkçe (Turkish)",
    "Chinese (Simplified)": "简体中文 (Chinese)",
}
LANGUAGE_OPTIONS = list(LANGUAGES.keys())
RTL_LANGUAGES = {"Urdu", "Arabic", "Punjabi (Shahmukhi)"}

# --------------------------------------------------------------------------
# Urgency levels (lowest → highest) + colours used in the dashboard
# --------------------------------------------------------------------------
URGENCY_LEVELS = ["LOW", "MEDIUM", "HIGH", "EMERGENCY"]

URGENCY_COLORS = {
    "LOW": "#1b8a5a",        # green
    "MEDIUM": "#c98a00",     # amber
    "HIGH": "#e8590c",       # orange
    "EMERGENCY": "#c92a2a",  # red
}

URGENCY_ICONS = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "EMERGENCY": "🔴"}

URGENCY_MEANING = {
    "LOW": "Mild — home monitoring is usually reasonable; see a doctor if it persists or worsens.",
    "MEDIUM": "Moderate — arrange a visit with a healthcare professional in the coming days.",
    "HIGH": "Serious — seek in-person medical care today (urgent care / same-day doctor).",
    "EMERGENCY": "Possible emergency — get emergency medical help immediately.",
}

# --------------------------------------------------------------------------
# Rule-based safety net (applied AFTER the AI answers; the AI can never
# lower the urgency below these floors).
# --------------------------------------------------------------------------
# Any ONE of these symptoms → at least HIGH
HIGH_RISK_SYMPTOMS = {
    "Chest pain",
    "Shortness of breath",
    "Confusion",
    "Loss of consciousness / fainting",
    "Seizure",
    "Severe bleeding",
    "Slurred speech",
    "Sudden weakness or numbness",
}

# Any ONE of these symptoms → EMERGENCY (stroke signs, severe bleeding, ...)
EMERGENCY_SYMPTOMS = {
    "Slurred speech",
    "Sudden weakness or numbness",
    "Severe bleeding",
    "Loss of consciousness / fainting",
}

# These symptom COMBINATIONS → EMERGENCY
EMERGENCY_COMBOS = [
    {"Chest pain", "Shortness of breath"},
    {"Fever", "Stiff neck", "Confusion"},
]

# Words in the free-text fields that always trigger EMERGENCY.
EMERGENCY_KEYWORDS = [
    "chest pain",
    "can't breathe",
    "cant breathe",
    "cannot breathe",
    "difficulty breathing",
    "unconscious",
    "unresponsive",
    "overdose",
    "stroke",
    "heart attack",
    "suicide",
    "suicidal",
    "kill myself",
    "self-harm",
    "self harm",
    "heavy bleeding",
    "coughing blood",
    "vomiting blood",
]

# A very high severity score alone raises the floor to HIGH.
HIGH_SEVERITY_THRESHOLD = 9

# --------------------------------------------------------------------------
# Input limits (keeps prompts small and the app predictable)
# --------------------------------------------------------------------------
MAX_FREE_TEXT_CHARS = 1000
MIN_AGE = 0
MAX_AGE = 120

# --------------------------------------------------------------------------
# Demo scenarios (match the "Testing Scenarios" table in the assignment)
# --------------------------------------------------------------------------
DEMO_SCENARIOS = {
    "— choose a demo scenario —": None,
    "1) Mild cold (age 25)": {
        "age_text": "25",
        "gender": "Female",
        "symptoms": ["Runny nose", "Sore throat"],
        "free_symptoms": "",
        "duration": "1-3 days",
        "severity": 2,
        "conditions": "",
        "medications": "",
        "notes": "",
    },
    "2) Fever + cough (age 40)": {
        "age_text": "40",
        "gender": "Male",
        "symptoms": ["Fever", "Cough"],
        "free_symptoms": "",
        "duration": "4-7 days",
        "severity": 6,
        "conditions": "",
        "medications": "",
        "notes": "Feeling very tired in the evenings.",
    },
    "3) Chest pain + breathlessness": {
        "age_text": "55",
        "gender": "Male",
        "symptoms": ["Chest pain", "Shortness of breath"],
        "free_symptoms": "",
        "duration": "Less than 1 day",
        "severity": 9,
        "conditions": "High blood pressure",
        "medications": "",
        "notes": "Pain started suddenly about an hour ago.",
    },
}
