import os
import re
import json
from io import BytesIO
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image
import streamlit as st

# --- Azure Doc Intelligence SDK (OCR) ---
try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
except Exception:
    DocumentIntelligenceClient = None
    AzureKeyCredential = None

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Suvichaar Literature Insight", page_icon="📚", layout="centered")
st.title("📚 Suvichaar — Literature Insight (Text & Poetry)")
st.caption("Upload a quote/poem image or paste text → OCR → Literary analysis: literal meaning, figurative sense, devices, line-by-line notes, glossary.")

# =========================
# SECRETS / CONFIG
# =========================
def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default

AZURE_API_KEY     = get_secret("AZURE_API_KEY")
AZURE_ENDPOINT    = get_secret("AZURE_ENDPOINT")
AZURE_DEPLOYMENT  = get_secret("AZURE_DEPLOYMENT", "gpt-4o")
AZURE_API_VERSION = get_secret("AZURE_API_VERSION", "2024-08-01-preview")

AZURE_DI_ENDPOINT = get_secret("AZURE_DI_ENDPOINT")
AZURE_DI_KEY      = get_secret("AZURE_DI_KEY")

if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT):
    st.warning("Add Azure OpenAI secrets in `.streamlit/secrets.toml` → AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT.")

# =========================
# SAFE SANITIZATION WRAPPER
# =========================
def make_classroom_safe(text: str):
    """
    Replace risky words with classroom-friendly alternatives to avoid Azure GPT-4o content filtering.
    """
    replacements = {
        r"\bkill(ed|ing)?\b": "harm",
        r"\bmurder(ed|ing)?\b": "serious harm",
        r"\bdeath\b": "loss",
        r"\bdie(s|d)?\b": "pass away",
        r"\bblood\b": "red liquid",
        r"\bsuicide\b": "personal struggle",
        r"\bviolence\b": "conflict",
        r"\bhate(d|s)?\b": "dislike",
        r"\babuse(d|s|ive)?\b": "mistreatment",
        r"\bdrugs?\b": "substances",
        r"\balcohol\b": "drinks",
        r"\bsex(ual|ually)?\b": "personal topic"
    }
    for pattern, safe_word in replacements.items():
        text = re.sub(pattern, safe_word, text, flags=re.IGNORECASE)
    return text

# =========================
# AZURE GPT CALL
# =========================
def call_azure_chat(messages, *, temperature=0.1, max_tokens=2200):
    """Call Azure GPT-4o with JSON-only mode and safe retry fallback."""
    headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions"
    params = {"api-version": AZURE_API_VERSION}
    body = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"}  # force JSON output
    }

    try:
        r = requests.post(url, headers=headers, params=params, json=body, timeout=120)
        if r.status_code == 200:
            return True, r.json()["choices"][0]["message"]["content"]

        # If blocked due to Azure content moderation → retry safely
        if r.status_code == 400 and "filtered" in r.text.lower():
            return False, "FILTERED"

        return False, f"Azure error {r.status_code}: {r.text[:300]}"
    except Exception as e:
        return False, f"Azure request failed: {e}"

# =========================
# OCR (IMAGES / PDFs)
# =========================
def ocr_read_any(bytes_blob: bytes) -> str:
    """Use Azure DI 'prebuilt-read' to extract text from images or PDFs."""
    if DocumentIntelligenceClient is None or AzureKeyCredential is None:
        return ""
    if not (AZURE_DI_ENDPOINT and AZURE_DI_KEY):
        return ""
    try:
        client = DocumentIntelligenceClient(endpoint=AZURE_DI_ENDPOINT.rstrip("/"), credential=AzureKeyCredential(AZURE_DI_KEY))
        poller = client.begin_analyze_document("prebuilt-read", body=bytes_blob)
        doc = poller.result()
        parts = []
        if getattr(doc, "pages", None):
            for p in doc.pages:
                lines = [ln.content for ln in getattr(p, "lines", []) or [] if getattr(ln, "content", None)]
                page_txt = "\n".join(lines).strip()
                if page_txt:
                    parts.append(page_txt)
        elif getattr(doc, "paragraphs", None):
            parts.append("\n".join(pp.content for pp in doc.paragraphs if getattr(pp, "content", None)))
        else:
            raw = (getattr(doc, "content", "") or "").strip()
            if raw:
                parts.append(raw)
        return "\n".join(parts).strip()
    except Exception:
        return ""

# =========================
# LITERARY ANALYSIS PROMPT
# =========================
LIT_SYSTEM = (
    "You are a veteran literature teacher and critic. "
    "Analyze this text for students in a CLASSROOM-SAFE manner. "
    "Avoid explicit, violent, or inappropriate language. "
    "Focus on themes, devices, meanings, and explanations suitable for children."
)

LIT_JSON_SCHEMA = {
    "language": "en|hi",
    "text_type": "quote|prose|poetry",
    "literal_meaning": "short plain-language paraphrase",
    "figurative_meaning": "themes, symbolism, deeper sense",
    "speaker_or_voice": "who is speaking (if clear) or 'narrator'",
    "tone_mood": "tone words (e.g., admiring, melancholic) and mood",
    "devices": [
        {"name": "Simile|Metaphor|Personification|Alliteration|Hyperbole|Imagery|Symbolism|Rhyme",
         "evidence": "exact words from text",
         "explanation": "why this is that device"}
    ],
    "word_by_word_defs": [
        {"word": "face", "meaning": "literal + connotation"},
        {"word": "moon", "meaning": "literal + figurative meaning"}
    ],
    "line_by_line": [
        {"line": "original line text", "explanation": "simple meaning"}
    ],
    "cultural_context": "notes on idioms, proverbs, or symbolism",
    "vocabulary_glossary": [
        {"term": "...", "meaning": "..."}
    ],
    "misconceptions": ["common misunderstandings to avoid"],
    "one_sentence_takeaway": "student-friendly summary"
}

PROMPT_FMT = f"""
Return ONLY a valid JSON object with these keys:
{json.dumps(LIT_JSON_SCHEMA, ensure_ascii=False, indent=2)}
Be concise, classroom-safe, and neutral.
"""

# =========================
# UI INPUTS
# =========================
st.markdown("### 📥 Input")
text_input = st.text_area("Paste a quote or poem (optional)", height=120, placeholder="e.g., Your face is like Moon")
files = st.file_uploader("Or upload an image/PDF containing the text", type=["jpg","jpeg","png","webp","tiff","pdf"], accept_multiple_files=False)
lang_choice = st.selectbox("Target explanation language", ["Auto-detect","English","Hindi"], index=0)

show_devices_table = st.toggle("Show literary devices table", value=True)
show_line_by_line = st.toggle("Show line-by-line explanation (if poetry)", value=True)

run = st.button("🔎 Analyze")

# =========================
# MAIN EXECUTION
# =========================
if run:
    # Build source text
    source_text = (text_input or "").strip()
    if files and not source_text:
        with st.spinner("Running OCR on uploaded file…"):
            blob = files.read()
            ocr_text = ocr_read_any(blob)
            if ocr_text:
                source_text = ocr_text
                st.success("OCR text extracted:")
                with st.expander("Show OCR text"):
                    st.write(ocr_text[:20000])
            else:
                st.error("OCR returned no text. Try a clearer image or paste the text manually.")
                st.stop()

    if not source_text:
        st.error("Please paste text or upload a file.")
        st.stop()

    # Language detection
    def detect_hi_or_en(text: str) -> str:
        devanagari = sum(0x0900 <= ord(c) <= 0x097F for c in text)
        latin = sum(('A' <= c <= 'Z') or ('a' <= c <= 'z') for c in text)
        total = devanagari + latin
        if total == 0:
            return "en"
        return "hi" if (devanagari / total) >= 0.25 else "en"

    detected = detect_hi_or_en(source_text)
    explain_lang = "en" if lang_choice == "English" else "hi" if lang_choice == "Hindi" else detected
    st.info(f"Explanation language: **{explain_lang}** (detected: {detected})")

    # Sanitize before sending to GPT-4o
    safe_text = make_classroom_safe(source_text)

    # Build messages
    system_msg = LIT_SYSTEM + (" Respond in Hindi." if explain_lang.startswith("hi") else " Respond in English.")
    user_msg = f"TEXT TO ANALYZE:\n{safe_text}\n\n{PROMPT_FMT}\nOutput strictly as JSON."

    with st.spinner("Calling Azure GPT-4o for literary analysis…"):
        ok, content = call_azure_chat([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ])

    # Retry in student-safe mode if blocked
    if not ok and content == "FILTERED":
        st.warning("⚠️ Sensitive content detected. Retrying in student-safe mode…")
        safe_prompt = f"Provide a neutral, student-safe analysis of:\n{safe_text}"
        ok, content = call_azure_chat([
            {"role": "system", "content": "You are a literature teacher for school students. Avoid explicit terms."},
            {"role": "user", "content": safe_prompt},
        ])

    if not ok:
        st.error(content)
        st.stop()

    # Parse JSON
    try:
        data = json.loads(content)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", content)
        data = json.loads(match.group(0)) if match else {}

    if not data:
        st.error("Model did not return valid JSON.")
        st.stop()

    # Display results
    st.success("✅ Analysis ready")

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Literal meaning**")
        st.write(data.get("literal_meaning", "—"))
        st.markdown("**Figurative meaning / themes**")
        st.write(data.get("figurative_meaning", "—"))
        st.markdown("**Speaker/Voice**")
        st.write(data.get("speaker_or_voice", "—"))
    with cols[1]:
        st.markdown("**Tone & Mood**")
        st.write(data.get("tone_mood", "—"))
        st.markdown("**Cultural context**")
        st.write(data.get("cultural_context", "—"))
        st.markdown("**One-sentence takeaway**")
        st.write(data.get("one_sentence_takeaway", "—"))

    # Word-by-word definitions
    wbw = data.get("word_by_word_defs") or []
    if wbw:
        st.markdown("### 🧩 Word-by-word meanings & connotations")
        st.table([{"word": w.get("word", ""), "meaning": w.get("meaning", "")} for w in wbw])

    # Devices table
    if show_devices_table:
        devices = data.get("devices") or []
        st.markdown("### 🎭 Literary devices")
        if devices:
            st.table([{
                "device": d.get("name", ""),
                "evidence": d.get("evidence", ""),
                "why": d.get("explanation", "")
            } for d in devices])
        else:
            st.info("No clear devices detected.")

    # Line-by-line explanations (poetry)
    if show_line_by_line and (data.get("text_type") == "poetry" or data.get("line_by_line")):
        st.markdown("### 📖 Line-by-line explanation")
        for i, item in enumerate(data.get("line_by_line", []), start=1):
            st.markdown(f"**Line {i}:** {item.get('line', '')}")
            st.write(item.get("explanation", ""))
            st.divider()

    # Glossary & misconceptions
    gl = data.get("vocabulary_glossary") or []
    if gl:
        st.markdown("### 📒 Glossary")
        st.table(gl)

    mc = data.get("misconceptions") or []
    if mc:
        st.markdown("### ⚠️ Misconceptions to avoid")
        st.write("\n".join(f"• {m}" for m in mc))

    # Raw JSON download
    with st.expander("🔧 Debug / Raw JSON"):
        st.json(data, expanded=False)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "⬇️ Download analysis JSON",
            data=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"literature_analysis_{ts}.json",
            mime="application/json",
        )
