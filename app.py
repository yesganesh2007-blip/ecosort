"""
EcoSort AI — Flask Backend
==========================
Provides a Gemini-powered API for e-waste disposal guidance.

Setup
-----
1. Install dependencies:
       pip install -r requirements.txt

2. Copy the environment template and fill in your key:
       cp .env.example .env
   Then edit .env and set GEMINI_API_KEY to your real key from
   https://aistudio.google.com/app/apikey

3. Run the server:
       python app.py
   or:
       flask run --port 5000

4. Point the frontend at this server:
   In index.html, set:
       const BACKEND_URL = "http://localhost:5000";

The server will listen on http://localhost:5000 by default.
CORS is restricted to FRONTEND_ORIGIN (see .env.example).
"""

import json
import logging
import os
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# CORS — restrict to the configured frontend origin only
frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5500")
CORS(app, resources={r"/api/*": {"origins": [
    frontend_origin,
    "http://localhost:5000",
    "http://localhost:5500",
    "http://localhost:5501",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:5501",
    "null",  # file:// origin
]}})

# ---------------------------------------------------------------------------
# Gemini setup
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CONFIGURED = False

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_CONFIGURED = True
        logger.info("Gemini API configured successfully.")
    except Exception as exc:
        logger.error("Failed to configure Gemini API: %s", exc)
else:
    logger.warning("GEMINI_API_KEY not set — AI features will be unavailable.")

# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

KB_PATH = Path(__file__).parent / "knowledge_base.json"

try:
    with open(KB_PATH, "r", encoding="utf-8") as fh:
        KNOWLEDGE_BASE = json.load(fh)
    logger.info("Knowledge base loaded: %d categories.", len(KNOWLEDGE_BASE.get("categories", [])))
except FileNotFoundError:
    logger.error("knowledge_base.json not found — classification will be limited.")
    KNOWLEDGE_BASE = {"categories": [], "disposal_channels": [], "co2_comparisons": {}}
except json.JSONDecodeError as exc:
    logger.error("Failed to parse knowledge_base.json: %s", exc)
    KNOWLEDGE_BASE = {"categories": [], "disposal_channels": [], "co2_comparisons": {}}

CATEGORIES = KNOWLEDGE_BASE.get("categories", [])

# ---------------------------------------------------------------------------
# Rate limiter (simple in-memory, per-IP)
# ---------------------------------------------------------------------------

_rate_store: dict = defaultdict(list)
RATE_LIMIT_REQUESTS = 30   # max requests
RATE_LIMIT_WINDOW_S = 60   # per N seconds


def is_rate_limited(ip: str) -> bool:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_S
    # Prune old timestamps
    _rate_store[ip] = [t for t in _rate_store[ip] if t > window_start]
    if len(_rate_store[ip]) >= RATE_LIMIT_REQUESTS:
        return True
    _rate_store[ip].append(now)
    return False


def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.remote_addr or "unknown"
        if is_rate_limited(ip):
            logger.warning("Rate limit exceeded for IP: %s", ip)
            return jsonify({"error": "Too many requests. Please wait a moment and try again.", "code": "RATE_LIMITED"}), 429
        return f(*args, **kwargs)
    return decorated

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_MESSAGE_LEN = 2000
MAX_HISTORY_TURNS = 20


def local_classify(description: str) -> dict | None:
    """Keyword-based local classification. Returns category dict or None."""
    desc_lower = description.lower()
    best_match = None
    best_score = 0

    for cat in CATEGORIES:
        if cat.get("id") == "non_ewaste":
            continue
        score = sum(1 for kw in cat.get("keywords", []) if kw.lower() in desc_lower)
        if score > best_score:
            best_score = score
            best_match = cat

    if best_score == 0:
        # Check non-e-waste keywords
        for cat in CATEGORIES:
            if cat.get("id") == "non_ewaste":
                score = sum(1 for kw in cat.get("keywords", []) if kw.lower() in desc_lower)
                if score > 0:
                    return cat

    return best_match


def build_classification_prompt(item_description: str, follow_up: dict, matched_category: dict | None) -> str:
    """Build a structured prompt for Gemini classification."""
    category_context = ""
    if matched_category:
        category_context = f"""
Preliminary local classification matched this item to the '{matched_category.get('name')}' category
(Hazard: {matched_category.get('hazard_level')}, Recommended action: {matched_category.get('recommended_action')}).
Use this as a guide but refine it based on the description.
"""

    fu_text = ""
    if follow_up:
        fu_text = "Follow-up answers from the user:\n"
        if follow_up.get("working_status"):
            fu_text += f"  - Working status: {follow_up['working_status']}\n"
        if follow_up.get("visible_damage"):
            fu_text += f"  - Visible damage: {follow_up['visible_damage']}\n"
        if follow_up.get("age_years"):
            fu_text += f"  - Approximate age: {follow_up['age_years']} years\n"

    return f"""You are an expert e-waste classification system for EcoSort AI.
Classify the following item and respond ONLY with a valid JSON object (no markdown, no explanation outside JSON).

Item description: "{item_description}"
{fu_text}
{category_context}

Respond with exactly this JSON structure:
{{
  "is_ewaste": true/false,
  "category": "category name or 'Non-E-Waste'",
  "category_id": "matching id from: battery, charger, cable, phone, earphones, keyboard_mouse, small_appliance, laptop, tv_monitor, printer, non_ewaste, or 'unknown'",
  "hazard_level": "Low" | "Medium" | "High" | "None",
  "hazard_code": "short code or N/A",
  "recommended_action": "Repair | Reuse | Donate | Recycle | Safe Discard | Standard Recycling",
  "disposal_text": "2-3 sentence disposal guidance",
  "safety_notes": ["note1", "note2"],
  "environmental_note": "1-2 sentence environmental impact note",
  "recoverable_materials": ["material1", "material2"],
  "estimated_co2_saved_kg": <number>,
  "confidence": "High" | "Medium" | "Low"
}}"""


SYSTEM_PROMPT = """You are EcoSort Assistant, an AI-powered e-waste disposal guidance assistant embedded in the EcoSort AI platform, created for an AI for Sustainability project aligned with SDG 12 (Responsible Consumption and Production) and SDG 11 (Sustainable Cities and Communities).

Your role is to help users make responsible decisions about their electronic waste: safe handling, repair vs. replace decisions, recycling guidance, hazard awareness, and environmental impact.

RULES YOU MUST ALWAYS FOLLOW:
1. Stay focused on e-waste, electronics disposal, sustainability, and environmental topics. If asked about unrelated topics, politely redirect: "That's outside my focus — I'm here to help with e-waste and electronics disposal. Is there something along those lines I can help with?"
2. URGENT SAFETY: If a user describes a smoking, leaking, swollen, or burning battery or device, immediately tell them: STOP handling the item. Place it on a non-flammable surface away from flammables. Do NOT charge it. Contact local emergency services or a hazmat disposal line. Do NOT attempt to fix or dispose of it yourself.
3. NEVER recommend burning electronics, illegal dumping, pouring chemicals down drains, or unsafe dismantling without proper PPE.
4. Always clarify you are guidance-only, not a certified hazardous-waste professional, for ambiguous or high-risk cases.
5. Keep responses concise. Use short paragraphs or bullet points. Avoid walls of text.
6. Be warm, helpful, and encouraging — sustainability choices matter and users should feel empowered, not lectured.

KNOWLEDGE AREAS:
- E-waste categories: batteries, phones, laptops, TVs/monitors, chargers, cables, earphones, keyboards/mice, small appliances, printers
- Hazard levels and why certain materials (lithium, lead, mercury, cadmium, BFRs) are dangerous
- Repair vs. replace decisions and the right-to-repair movement
- Certified recycling channels: authorized recyclers, manufacturer take-back, retailer drop-boxes, municipal drives
- Environmental impact: CO₂ savings, resource recovery, circular economy principles
- SDG 12 and SDG 11 context"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/api/health", methods=["GET"])
def health():
    """Health check — frontend uses this for the status indicator."""
    return jsonify({
        "status": "ok",
        "gemini_configured": GEMINI_CONFIGURED,
        "categories_loaded": len(CATEGORIES),
    })


@app.route("/api/chat", methods=["POST"])
@rate_limit
def chat():
    """
    Chat endpoint.
    Body: { "message": "...", "history": [{"role": "user"|"model", "parts": ["..."]}] }
    Returns: { "reply": "..." }
    """
    if not GEMINI_CONFIGURED:
        return jsonify({"error": "AI assistant is not configured. Please set GEMINI_API_KEY.", "code": "NOT_CONFIGURED"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body.", "code": "BAD_REQUEST"}), 400

    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Message cannot be empty.", "code": "EMPTY_MESSAGE"}), 400
    if len(message) > MAX_MESSAGE_LEN:
        return jsonify({"error": f"Message too long (max {MAX_MESSAGE_LEN} characters).", "code": "MESSAGE_TOO_LONG"}), 400

    raw_history = data.get("history", [])
    if not isinstance(raw_history, list):
        raw_history = []

    # Sanitise and cap history
    history = []
    for turn in raw_history[-MAX_HISTORY_TURNS:]:
        role = turn.get("role", "")
        parts = turn.get("parts", [])
        if role in ("user", "model") and parts and isinstance(parts, list):
            history.append({"role": role, "parts": [str(p)[:MAX_MESSAGE_LEN] for p in parts]})

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=SYSTEM_PROMPT,
        )
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(message)
        reply = response.text
        return jsonify({"reply": reply})

    except Exception as exc:
        exc_str = str(exc).lower()
        logger.error("Gemini chat error: %s", exc)

        if "api_key" in exc_str or "invalid" in exc_str or "401" in exc_str or "permission" in exc_str:
            return jsonify({"error": "Invalid or missing Gemini API key. Please check server configuration.", "code": "AUTH_ERROR"}), 401
        if "quota" in exc_str or "429" in exc_str or "resource_exhausted" in exc_str:
            return jsonify({"error": "AI service is temporarily busy (rate limit). Please try again in a moment.", "code": "RATE_LIMITED"}), 429
        if "timeout" in exc_str or "deadline" in exc_str:
            return jsonify({"error": "Request timed out. Please try again.", "code": "TIMEOUT"}), 504
        return jsonify({"error": "AI assistant is temporarily unavailable. Please try again shortly.", "code": "SERVER_ERROR"}), 500


@app.route("/api/classify", methods=["POST"])
@rate_limit
def classify():
    """
    Classification endpoint.
    Body: { "item_description": "...", "follow_up_answers": { ... } }
    Returns structured classification JSON.
    Falls back to local keyword matching if Gemini is unavailable.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body.", "code": "BAD_REQUEST"}), 400

    item_description = (data.get("item_description") or "").strip()
    if not item_description:
        return jsonify({"error": "item_description cannot be empty.", "code": "EMPTY_INPUT"}), 400
    if len(item_description) > MAX_MESSAGE_LEN:
        return jsonify({"error": f"Description too long (max {MAX_MESSAGE_LEN} characters).", "code": "INPUT_TOO_LONG"}), 400

    follow_up = data.get("follow_up_answers", {})
    if not isinstance(follow_up, dict):
        follow_up = {}

    # Always run local classification first (used as fallback and context)
    local_match = local_classify(item_description)

    if GEMINI_CONFIGURED:
        try:
            prompt = build_classification_prompt(item_description, follow_up, local_match)
            model = genai.GenerativeModel(model_name="gemini-2.0-flash")
            response = model.generate_content(prompt)
            raw = response.text.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            result = json.loads(raw)
            result["source"] = "gemini"
            return jsonify(result)

        except json.JSONDecodeError:
            logger.warning("Gemini returned non-JSON classification response — falling back to local.")
        except Exception as exc:
            logger.error("Gemini classify error: %s", exc)
            # Fall through to local

    # Local fallback
    if local_match:
        return jsonify({
            "is_ewaste": local_match.get("id") != "non_ewaste",
            "category": local_match.get("name", "Unknown"),
            "category_id": local_match.get("id", "unknown"),
            "hazard_level": local_match.get("hazard_level", "Unknown"),
            "hazard_code": local_match.get("hazard_code", "N/A"),
            "recommended_action": local_match.get("recommended_action", "Recycle"),
            "disposal_text": local_match.get("disposal_text", "Please take this item to a certified e-waste recycler."),
            "safety_notes": local_match.get("safety_notes", []),
            "environmental_note": local_match.get("environmental_note", ""),
            "recoverable_materials": local_match.get("recoverable_materials", []),
            "estimated_co2_saved_kg": local_match.get("estimated_co2_saved_kg", 0),
            "confidence": "Medium",
            "source": "local",
        })

    return jsonify({
        "is_ewaste": None,
        "category": "Unknown",
        "category_id": "unknown",
        "hazard_level": "Unknown",
        "hazard_code": "N/A",
        "recommended_action": "Consult a certified recycler",
        "disposal_text": "Could not classify this item. Please consult your local certified e-waste recycler.",
        "safety_notes": ["Handle with care until classification is confirmed."],
        "environmental_note": "When in doubt, always choose a certified recycler over landfill disposal.",
        "recoverable_materials": [],
        "estimated_co2_saved_kg": 0,
        "confidence": "Low",
        "source": "local",
    })


@app.route("/api/knowledge", methods=["GET"])
def knowledge():
    """Return the full knowledge base (categories + channels) to the frontend."""
    return jsonify(KNOWLEDGE_BASE)


@app.route("/api/repair-advice", methods=["POST"])
@rate_limit
def repair_advice():
    """
    Gemini-powered repair vs. replace advice with cost auto-prediction.
    Body: { "category_id": "...", "repair_cost": 60, "replace_cost": 250,
            "condition": "...", "age_years": 3 }
    Returns: { "recommendation": "Repair|Replace|Borderline",
               "reasoning": "...", "predicted_repair_cost": N,
               "predicted_replace_cost": N, "source": "gemini|local" }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body.", "code": "BAD_REQUEST"}), 400

    category_id   = (data.get("category_id") or "").strip()
    repair_cost   = data.get("repair_cost")    # may be None → ask Gemini to predict
    replace_cost  = data.get("replace_cost")   # may be None → ask Gemini to predict
    condition     = (data.get("condition") or "not specified").strip()
    age_years     = data.get("age_years", "unknown")

    cat = next((c for c in CATEGORIES if c.get("id") == category_id), None)
    if not cat:
        return jsonify({"error": "Unknown category_id.", "code": "BAD_INPUT"}), 400

    # Local fallback values from KB
    kb_repair   = cat.get("avg_repair_cost_usd", 0)
    kb_replace  = cat.get("avg_replacement_cost_usd", 0)
    feasibility = cat.get("repair_feasibility", "Medium")

    if GEMINI_CONFIGURED:
        try:
            costs_note = ""
            if repair_cost is None and replace_cost is None:
                costs_note = "The user has NOT provided cost estimates — predict realistic average costs for this category."
            elif repair_cost is None:
                costs_note = f"The user provided a replacement cost of ${replace_cost}. Predict a realistic repair cost."
            elif replace_cost is None:
                costs_note = f"The user provided a repair cost of ${repair_cost}. Predict a realistic replacement cost."
            else:
                costs_note = f"The user provided: repair=${repair_cost}, replacement=${replace_cost}."

            prompt = f"""You are an e-waste repair advisor for EcoSort AI.

Item category: {cat.get('name')}
Repair feasibility (from knowledge base): {feasibility}
Knowledge base average repair cost: ${kb_repair}
Knowledge base average replacement cost: ${kb_replace}
Item condition: {condition}
Approximate age: {age_years} years
{costs_note}

Respond ONLY with a valid JSON object (no markdown, no text outside JSON):
{{
  "recommendation": "Repair" | "Replace" | "Borderline",
  "predicted_repair_cost": <number in USD>,
  "predicted_replace_cost": <number in USD>,
  "reasoning": "2-4 sentence explanation covering cost ratio, sustainability angle, and any condition/age factors",
  "tip": "One practical tip (e.g. where to find a repair service or what to look for in a replacement)"
}}"""

            model = genai.GenerativeModel(model_name="gemini-2.0-flash")
            response = model.generate_content(prompt)
            raw = response.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            result = json.loads(raw)
            result["source"] = "gemini"
            return jsonify(result)

        except (json.JSONDecodeError, Exception) as exc:
            logger.error("Gemini repair-advice error: %s", exc)
            # fall through to local

    # Local fallback
    r_cost  = repair_cost  if repair_cost  is not None else kb_repair
    rc_cost = replace_cost if replace_cost is not None else kb_replace
    ratio   = (r_cost / rc_cost) if rc_cost > 0 else 1

    if feasibility == "Low":
        rec = "Replace"
        reasoning = f"Repair feasibility for {cat.get('name')} is generally Low — parts and technicians can be hard to find. Recycling the old unit and purchasing a refurbished replacement is usually the better choice."
    elif ratio <= 0.5:
        rec = "Repair"
        reasoning = f"At ${r_cost:.0f}, the repair cost is {ratio*100:.0f}% of replacement (${rc_cost:.0f}) — well under the 50% threshold. Repairing extends the product's life and keeps it out of the waste stream."
    elif ratio <= 0.75:
        rec = "Borderline"
        reasoning = f"Repair cost is {ratio*100:.0f}% of replacement. Weigh the item's remaining lifespan against the cost. If it's under 3 years old and in otherwise good condition, repair is usually worth it."
    else:
        rec = "Replace"
        reasoning = f"At {ratio*100:.0f}% of replacement cost, repair is not economical. Consider a certified refurbished replacement and recycle the old device through an authorized e-waste program."

    return jsonify({
        "recommendation": rec,
        "predicted_repair_cost":   r_cost,
        "predicted_replace_cost":  rc_cost,
        "reasoning": reasoning,
        "tip": "Search for local repair cafes or manufacturer service centres before buying new.",
        "source": "local",
    })


@app.route("/api/dropoff", methods=["POST"])
@rate_limit
def dropoff():
    """
    Gemini-powered drop-off guidance for a given city/area.
    Body: { "city": "Mumbai", "category_id": "battery" (optional) }
    Returns: { "guidance": "...", "channels": [...] }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body.", "code": "BAD_REQUEST"}), 400

    city = (data.get("city") or "").strip()
    if not city:
        return jsonify({"error": "city cannot be empty.", "code": "BAD_INPUT"}), 400
    if len(city) > 100:
        return jsonify({"error": "City name too long.", "code": "BAD_INPUT"}), 400

    category_id = (data.get("category_id") or "").strip()
    cat = next((c for c in CATEGORIES if c.get("id") == category_id), None)
    item_context = f"The user wants to dispose of: {cat.get('name')} (Hazard: {cat.get('hazard_level')})." if cat else "The user has not specified a particular item type."

    if not GEMINI_CONFIGURED:
        return jsonify({
            "guidance": f"To find e-waste drop-off points in {city}, search '{city} e-waste recycling' on Google Maps, or visit your city's official municipal waste website.",
            "channels": [],
            "source": "local",
        })

    try:
        prompt = f"""You are an e-waste disposal location advisor for EcoSort AI.

User location: {city}
{item_context}

Provide practical, actionable drop-off guidance for this location. Include:
- Any well-known national or regional e-waste programs likely available in that country/region
- Types of drop-off points they should look for (retailer drop-boxes, municipal centres, etc.)
- Any country-specific e-waste regulations or free take-back schemes the user should know about
- 1-2 search tips to find the nearest certified recycler

Keep the response concise (4-6 sentences or short bullet points). Do NOT make up specific addresses or phone numbers.
Respond in plain text (no JSON needed)."""

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction="You are EcoSort Assistant, an e-waste disposal guidance AI. Give accurate, location-aware recycling guidance. Never invent specific business addresses or contact numbers.",
        )
        response = model.generate_content(prompt)
        return jsonify({
            "guidance": response.text.strip(),
            "source": "gemini",
        })

    except Exception as exc:
        logger.error("Gemini dropoff error: %s", exc)
        return jsonify({
            "guidance": f"To find e-waste drop-off points in {city}, search \"{city} e-waste recycling\" on Google Maps, check your municipality's official waste page, or visit earth911.com.",
            "source": "local",
        })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    logger.info("Starting EcoSort AI backend on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
