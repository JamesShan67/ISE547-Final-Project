# faq_core.py
"""
Core logic for the Smart FAQ Generator (final pipeline).

Final configuration:
- Provider: Gemini
- Model:   gemini-2.0-flash
- Prompt:  structured (EXACTLY 3 FAQ entries per chunk, JSON output)

This module is used by:
- main.py  (CLI backend)
- app.py   (Streamlit frontend demo)
"""

import os
import json
import csv
import re
from typing import List, Dict, Optional

import PyPDF2
import requests

# ---------- Final pipeline constants ----------

GEMINI_MODEL = "gemini-2.0-flash"
MAX_CHARS_PER_CHUNK = 2000  # for chunking


# ---------- Gemini call ----------

def call_gemini(model_name: str, prompt: str) -> str:
    """
    Call Gemini (Google) text model via REST API.

    Uses the Gemini API (ai.google.dev) and GEMINI_API_KEY in env.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    params = {"key": api_key}

    r = requests.post(url, headers=headers, params=params, json=payload)
    r.raise_for_status()
    data = r.json()

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(
            f"Unexpected Gemini response format:\n{json.dumps(data, indent=2)}"
        )

    return text.strip()


# ---------- Text extraction ----------

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file using PyPDF2.
    """
    text = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\n".join(text)


def extract_text_from_txt(file_path: str) -> str:
    """
    Extract text from a plain text file.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(file_path: str) -> str:
    """
    Dispatch to the appropriate text extraction function based on extension.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".txt", ".md"]:
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ---------- Chunking ----------

def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """
    Simple chunker: split into paragraphs, then pack into ~max_chars chunks.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for p in paragraphs:
        if len(p) > max_chars:
            # flush current, then hard-split this long paragraph
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(p), max_chars):
                chunks.append(p[i:i + max_chars])
            continue

        if len(current) + len(p) + 2 <= max_chars:
            current += ("\n\n" + p) if current else p
        else:
            if current:
                chunks.append(current.strip())
            current = p

    if current:
        chunks.append(current.strip())

    return chunks


# ---------- Structured prompt ----------

STRUCTURED_PROMPT = """
You are an assistant that turns long documents into FAQ entries.

Read the text below and generate EXACTLY 3 FAQ entries in JSON format:
[
  {"question": "...", "answer": "..."},
  {"question": "...", "answer": "..."},
  {"question": "...", "answer": "..."}
]

Requirements:
- Each question should be specific and helpful, as if it were on a real FAQ page.
- Each answer should be 1–3 sentences, concise and clear.
- Avoid overlapping or redundant questions.
- Do not include any text before or after the JSON array.

Text:
\"\"\"{chunk}\"\"\"
""".strip()


# ---------- JSON extraction helper ----------

def extract_json_array_from_text(text: str) -> Optional[str]:
    """
    Try to extract a JSON array substring from a larger text blob.
    Handles cases like code fences and extra explanation around JSON.
    Returns the JSON array string or None if not found.
    """
    text = text.strip()

    # Remove fenced code block markers like ```json ... ```
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return None


# ---------- FAQ generation for one chunk ----------

def generate_faq_for_chunk(chunk: str) -> List[Dict[str, str]]:
    """
    Generate FAQ entries for a single chunk of text using Gemini + structured prompt.
    Returns a list of {"question": ..., "answer": ...}.
    """
    prompt = STRUCTURED_PROMPT.replace("{chunk}", chunk)
    content = call_gemini(GEMINI_MODEL, prompt).strip()

    # 1) try parse whole content
    parsed = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # 2) try to extract JSON array substring
        json_snippet = extract_json_array_from_text(content)
        if json_snippet:
            try:
                parsed = json.loads(json_snippet)
            except json.JSONDecodeError:
                print("Warning: failed to parse JSON even after extraction.")
                # print(content)
                return []
        else:
            print("Warning: could not find JSON array in Gemini output.")
            # print(content)
            return []

    # normalize
    if isinstance(parsed, dict):
        faqs = [parsed]
    elif isinstance(parsed, list):
        faqs = parsed
    else:
        print(f"Warning: parsed JSON is not list/dict (type={type(parsed)}).")
        return []

    clean: List[Dict[str, str]] = []
    for item in faqs:
        if not isinstance(item, dict):
            continue
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if q and a:
            clean.append({"question": q, "answer": a})

    return clean


# ---------- Aggregation & deduplication ----------

def deduplicate_faqs(faqs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Deduplicate FAQ entries by lowercased question text.
    """
    seen = set()
    unique_faqs: List[Dict[str, str]] = []
    for item in faqs:
        q_lower = item["question"].strip().lower()
        if q_lower not in seen:
            seen.add(q_lower)
            unique_faqs.append(item)
    return unique_faqs


def generate_faq_for_document(
    file_path: str,
    max_chars_per_chunk: int = MAX_CHARS_PER_CHUNK,
) -> List[Dict[str, str]]:
    """
    End-to-end pipeline for a single document using the final pipeline:
    - extract text
    - chunk
    - Gemini (structured prompt) on each chunk
    - aggregate + deduplicate FAQs
    """
    print(f"[*] Extracting text from: {file_path}")
    text = extract_text(file_path)

    print("[*] Splitting into chunks...")
    chunks = chunk_text(text, max_chars=max_chars_per_chunk)
    print(f"[*] Number of chunks: {len(chunks)}")

    all_faqs: List[Dict[str, str]] = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"[*] Processing chunk {i}/{len(chunks)} with Gemini model={GEMINI_MODEL} ...")
        faqs = generate_faq_for_chunk(chunk)
        all_faqs.extend(faqs)

    print(f"[*] Total FAQs before deduplication: {len(all_faqs)}")
    unique_faqs = deduplicate_faqs(all_faqs)
    print(f"[*] Total FAQs after deduplication: {len(unique_faqs)}")

    return unique_faqs


# ---------- Save results as text ----------

def save_faqs_as_text(faqs: List[Dict[str, str]], output_path: str) -> None:
    """
    Save FAQ entries as a simple text file with numbered Q and A.
    """
    lines: List[str] = []
    for i, item in enumerate(faqs, start=1):
        lines.append(f"Q{i}. {item['question']}")
        lines.append(f"A{i}. {item['answer']}")
        lines.append("")  # blank line

    text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[*] Saved FAQs to: {output_path}")


# ---------- CSV logging for evaluation (optional) ----------

def log_faqs_to_csv(
    faqs: List[Dict[str, str]],
    csv_path: str,
    document_name: str,
) -> None:
    """
    Append FAQ entries to a CSV file for later evaluation.
    Columns: document, question, answer
    (You can extend with model, run_id, etc., if you like.)
    """
    if not faqs:
        print("[*] No FAQs to log; skipping CSV logging.")
        return

    csv_dir = os.path.dirname(csv_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "document",
        "model",
        "prompt",
        "question",
        "answer",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for item in faqs:
            writer.writerow(
                {
                    "document": document_name,
                    "model": GEMINI_MODEL,
                    "prompt": "structured",
                    "question": item["question"],
                    "answer": item["answer"],
                }
            )

    print(f"[*] Logged {len(faqs)} FAQs to CSV: {csv_path}")
