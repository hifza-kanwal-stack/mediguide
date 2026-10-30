"""
utils.py
========
Helper functions that do NOT talk to the OpenAI API:

  * validating and cleaning the form input
  * safe JSON parsing (invalid JSON must never crash the app)
  * a rule-based safety net for urgent symptoms
  * friendly error messages and a downloadable report
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime

from src import config

# --------------------------------------------------------------------------
# 1. Input validation + preparation
# --------------------------------------------------------------------------


def parse_age(age_text: str):
    """Turn the age text into a number, or return None if it is not valid."""
    try:
        age = float(str(age_text).strip())
    except ValueError:
        return None
    if age < config.MIN_AGE or age > config.MAX_AGE:
        return None
    return int(age) if age == int(age) else round(age, 1)


def validate_inputs(age_text: str, symptoms: list, free_symptoms: str):
    """
    Check the form BEFORE calling the API (saves money, avoids bad prompts).
    Returns a list of friendly error messages (empty list = everything is OK).
    """
    errors = []
    if parse_age(age_text) is None:
        errors.append(
            f"Please enter a valid age as a number between {config.MIN_AGE} and {config.MAX_AGE}."
        )
    if not symptoms and not (free_symptoms or "").strip():
        errors.append(
            "Please select at least one symptom (or describe it in the free-text box) — "
            "no request was sent to the AI."
        )
    return errors


def clean_text(text: str, max_chars: int = config.MAX_FREE_TEXT_CHARS) -> str:
    """Trim whitespace, drop control characters, limit length, neutralise our tags."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text or "").strip()
    # The prompt wraps patient text in <patient_data> tags; stop users from closing them.
    text = re.sub(r"</?\s*patient_data\s*>", "", text, flags=re.IGNORECASE)
    return text[:max_chars]


def build_inputs(
    age_text,
    gender,
    symptoms,
    free_symptoms,
    duration,
    severity,
    conditions,
    medications,
    notes,
    language,
) -> dict:
    """Collect the form values into the dictionary that the prompt templates expect."""
    all_symptoms = list(symptoms or [])
    free = clean_text(free_symptoms, 300)
    if free:
        all_symptoms.append(free)

    return {
        "age": parse_age(age_text),
        "gender": gender,
        "symptoms": ", ".join(all_symptoms),
        "duration": duration,
        "severity": int(severity),
        "conditions": clean_text(conditions) or "None reported",
        "medications": clean_text(medications) or "None reported",
        "notes": clean_text(notes) or "None",
        "language": language,
    }


# --------------------------------------------------------------------------
# 2. Safe JSON parsing
# --------------------------------------------------------------------------

_URGENCY_ALIASES = {
    "MODERATE": "MEDIUM",
    "MID": "MEDIUM",
    "MILD": "LOW",
    "URGENT": "HIGH",
    "SEVERE": "HIGH",
    "CRITICAL": "EMERGENCY",
    "EMERGENT": "EMERGENCY",
}


@dataclass
class ParseResult:
    """What parse_assessment() returns. Check `ok` first."""

    ok: bool
    data: dict = None
    error: str = ""
    notes: list = field(default_factory=list)


def clean_json_text(raw: str) -> str:
    """
    Models sometimes wrap JSON in ```json fences or add a sentence around it.
    This removes the fences and keeps only the part between the first '{'
    and the last '}'.
    """
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    return text.strip()


def _as_text(value) -> str:
    if value is None:
        return ""
    return value.strip() if isinstance(value, str) else str(value).strip()


def _as_text_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [_as_text(v) for v in value if _as_text(v)]
    return [_as_text(value)]


def _as_conditions(value) -> list:
    """Always return a list of {"name": ..., "reason": ...} dictionaries."""
    result = []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return result
    for item in value:
        if isinstance(item, dict):
            name = _as_text(item.get("name") or item.get("condition"))
            reason = _as_text(item.get("reason") or item.get("explanation"))
        else:
            name, reason = _as_text(item), ""
        if name:
            result.append({"name": name, "reason": reason})
    return result


def normalize_urgency(value) -> str:
    """Return one of LOW / MEDIUM / HIGH / EMERGENCY, or '' if not recognisable."""
    level = _as_text(value).upper()
    level = _URGENCY_ALIASES.get(level, level)
    return level if level in config.URGENCY_LEVELS else ""


def parse_assessment(raw: str) -> ParseResult:
    """
    Safely turn the model's raw text into a clean dictionary.

    This function NEVER raises an exception. If something is wrong it returns
    ParseResult(ok=False, error="...") so the UI can show a friendly message.
    """
    text = clean_json_text(raw)
    if not text:
        return ParseResult(False, error="The AI returned an empty answer.")

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        return ParseResult(False, error=f"The AI answer was not valid JSON ({exc}).")

    if not isinstance(parsed, dict):
        return ParseResult(False, error="The AI answer was JSON, but not a JSON object.")

    expected = {
        "summary",
        "possible_conditions",
        "urgency_level",
        "recommended_next_steps",
        "questions_for_doctor",
        "warning_signs",
    }
    if not expected & set(parsed):
        return ParseResult(False, error="The JSON did not contain any of the expected fields.")

    notes = []
    data = {
        "summary": _as_text(parsed.get("summary")),
        "possible_conditions": _as_conditions(parsed.get("possible_conditions")),
        "urgency_level": normalize_urgency(parsed.get("urgency_level")),
        "recommended_next_steps": _as_text_list(parsed.get("recommended_next_steps")),
        "questions_for_doctor": _as_text_list(parsed.get("questions_for_doctor")),
        "warning_signs": _as_text_list(parsed.get("warning_signs")),
    }

    missing = sorted(expected - set(parsed))
    if missing:
        notes.append("Some fields were missing in the AI answer: " + ", ".join(missing) + ".")

    if not data["urgency_level"]:
        # Unknown urgency → choose the cautious middle level, never LOW.
        data["urgency_level"] = "MEDIUM"
        notes.append("The AI did not give a valid urgency level, so MEDIUM was used to stay safe.")

    return ParseResult(True, data=data, notes=notes)


# --------------------------------------------------------------------------
# 3. Rule-based safety net
# --------------------------------------------------------------------------


def urgency_rank(level: str) -> int:
    return config.URGENCY_LEVELS.index(level) if level in config.URGENCY_LEVELS else 1


def highest_urgency(*levels: str) -> str:
    return max(levels, key=urgency_rank)


def safety_floor(symptoms: list, free_text: str, severity: int):
    """
    Decide the MINIMUM urgency allowed for this patient, using simple rules.
    Returns (level, reasons). The AI can raise the level but never go below it.
    """
    selected = set(symptoms or [])
    text = (free_text or "").lower()
    level, reasons = "LOW", []

    def raise_to(new_level, reason):
        nonlocal level
        if urgency_rank(new_level) > urgency_rank(level):
            level = new_level
        reasons.append(reason)

    for combo in config.EMERGENCY_COMBOS:
        if combo <= selected:
            raise_to("EMERGENCY", "the combination " + " + ".join(sorted(combo)).lower())
    for symptom in sorted(selected & config.EMERGENCY_SYMPTOMS):
        raise_to("EMERGENCY", f"'{symptom}' can be a medical emergency")
    for symptom in sorted(selected & config.HIGH_RISK_SYMPTOMS):
        raise_to("HIGH", f"'{symptom}' is a high-risk symptom")
    for keyword in config.EMERGENCY_KEYWORDS:
        if keyword in text:
            raise_to("EMERGENCY", f"the text mentions '{keyword}'")
            break
    if severity >= config.HIGH_SEVERITY_THRESHOLD:
        raise_to("HIGH", f"the reported severity is {severity}/10")

    return level, reasons


def apply_safety_floor(data: dict, symptoms: list, free_text: str, severity: int):
    """
    Return (new_data, reasons). If the AI's urgency was below the safety floor,
    it is raised and an extra first step is added. `reasons` is empty when the
    AI's answer already respected the floor.
    """
    data = {**data}
    floor, floor_reasons = safety_floor(symptoms, free_text, severity)
    original = data["urgency_level"]
    reasons = []

    if urgency_rank(floor) > urgency_rank(original):
        data["urgency_level"] = floor
        reasons = floor_reasons
        if floor == "EMERGENCY":
            step = "Call your local emergency number or go to the nearest emergency department immediately."
        else:
            step = "Seek in-person medical care today (urgent care or a same-day doctor visit)."
        data["recommended_next_steps"] = [step] + [
            s for s in data["recommended_next_steps"] if s != step
        ]

    if data["urgency_level"] == "EMERGENCY" and not reasons:
        # The AI itself chose EMERGENCY: make sure the first step says so.
        first = (data["recommended_next_steps"] or [""])[0].lower()
        if "emergency" not in first and "immediately" not in first:
            data["recommended_next_steps"] = [
                "Call your local emergency number or go to the nearest emergency department immediately."
            ] + data["recommended_next_steps"]

    return data, reasons


# --------------------------------------------------------------------------
# 4. Friendly errors
# --------------------------------------------------------------------------


def friendly_api_error(exc: Exception) -> str:
    """Translate raw API exceptions into a message a normal user can understand."""
    name = type(exc).__name__
    text = str(exc).lower()
    if "authentication" in name.lower() or "invalid_api_key" in text or "incorrect api key" in text or "401" in text:
        return "OpenAI rejected the API key. Please check OPENAI_API_KEY in your .env file (or the sidebar)."
    if "insufficient_quota" in text or "quota" in text or "billing" in text:
        return "Your OpenAI account has no remaining quota. Please check your billing at platform.openai.com."
    if "ratelimit" in name.lower() or "rate limit" in text or "429" in text:
        return "OpenAI is rate-limiting requests right now. Please wait a few seconds and try again."
    if "notfound" in name.lower() or "model_not_found" in text or "does not exist" in text:
        return "The selected model is not available for your account. Please choose another model in the sidebar."
    if "timeout" in name.lower() or "connection" in name.lower() or "connect" in text:
        return "Could not reach OpenAI. Please check your internet connection and try again."
    return "Something went wrong while contacting the AI model. Please try again."


# --------------------------------------------------------------------------
# 5. Report download + analytics helpers
# --------------------------------------------------------------------------


def urgency_distribution(history: list) -> dict:
    """Count how many past assessments had each urgency level (for the chart)."""
    counts = {level: 0 for level in config.URGENCY_LEVELS}
    for entry in history:
        if entry.get("urgency") in counts:
            counts[entry["urgency"]] += 1
    return counts


def build_markdown_report(result: dict) -> str:
    """Create a plain-markdown report the user can download and share with a doctor."""
    inputs, data = result["inputs"], result["data"]
    lines = [
        f"# {config.APP_NAME} — Preliminary Guidance Report",
        f"_Generated: {datetime.now():%Y-%m-%d %H:%M}_",
        "",
        "> **Educational AI prototype — NOT a diagnosis.** Please consult a qualified healthcare "
        "professional, and seek emergency help immediately in urgent situations.",
        "",
        "## Patient summary",
        f"- Age: {inputs['age']}",
        f"- Gender: {inputs['gender']}",
        f"- Symptoms: {inputs['symptoms']}",
        f"- Duration: {inputs['duration']}",
        f"- Severity: {inputs['severity']}/10",
        f"- Existing conditions: {inputs['conditions']}",
        f"- Medications: {inputs['medications']}",
        f"- Notes: {inputs['notes']}",
        "",
        f"## Urgency level: {data['urgency_level']}",
        config.URGENCY_MEANING[data["urgency_level"]],
        "",
        "## General information",
        data["summary"] or "-",
        "",
        "## Possible conditions (education only)",
    ]
    for item in data["possible_conditions"] or []:
        lines.append(f"- **{item['name']}** — {item['reason']}")
    lines += ["", "## Recommended next steps"]
    lines += [f"{i}. {s}" for i, s in enumerate(data["recommended_next_steps"], 1)]
    lines += ["", "## Questions to ask a healthcare professional"]
    lines += [f"- {q}" for q in data["questions_for_doctor"]]
    lines += ["", "## Warning signs — get immediate help if you notice"]
    lines += [f"- {w}" for w in data["warning_signs"]]
    if result.get("narrative"):
        lines += ["", "## AI guidance note", result["narrative"]]
    lines += ["", "---", config.SHORT_DISCLAIMER]
    return "\n".join(lines)
