# faq_core.py
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
from pypdf import PdfReader

# --- Gemini client (AI Studio / google-genai) ---
# pip install google-genai
from google import genai

GEMINI_MODEL = "gemini-2.0-flash"

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

MAX_FAQS_PER_CHUNK = 8  # you can tune
MAX_CHARS_PER_CHUNK = 3500


@dataclass
class FAQItem:
    document_name: str
    chunk_id: int
    question: str
    answer: str
    provider: str = "gemini"
    model: str = GEMINI_MODEL
    prompt_type: str = "structured"


# ---------- STRUCTURED PROMPT ----------

STRUCTURED_FAQ_PROMPT = """You are an expert FAQ generator.

You are given a chunk of a document between <DOCUMENT_CHUNK> tags.
Your task is to write high–quality FAQs that would help a user quickly
understand and navigate this document section.

<DOCUMENT_CHUNK>
{chunk_text}
</DOCUMENT_CHUNK>

Instructions:
1. Write clear, concise questions that a real user would ask.
2. Each question must have a helpful, accurate answer based ONLY on the text.
3. Avoid duplicate or trivially similar questions.
4. Focus on useful information: rules, procedures, steps, limitations, key concepts.
5. Do NOT hallucinate or add information that is not clearly supported.

Return your result as a JSON array. Each element must be an object:
{{
  "question": "...",
  "answer": "..."
}}

Generate at most {max_faqs} FAQs.
"""


# ---------- PDF → TEXT CHUNKS ----------

def load_pdf_to_chunks(pdf_path: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    reader = PdfReader(pdf_path)
    chunks: List[str] = []
    buffer = ""

    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.replace("\u00a0", " ").strip()
        if not text:
            continue

        if len(buffer) + len(text) <= max_chars:
            buffer += "\n" + text
        else:
            if buffer.strip():
                chunks.append(buffer.strip())
            buffer = text

    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks


# ---------- GEMINI CALL ----------

def call_gemini_faq(chunk_text: str,
                    max_faqs: int = MAX_FAQS_PER_CHUNK,
                    model: str = GEMINI_MODEL) -> List[Dict[str, Any]]:
    prompt = STRUCTURED_FAQ_PROMPT.format(
        chunk_text=chunk_text,
        max_faqs=max_faqs,
    )

    response = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        config={
            "temperature": 0.3,
            "max_output_tokens": 1024,
        },
    )

    content = (response.text or "").strip()

    try:
        # try to carve out a JSON array
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1:
            json_str = content[start:end + 1]
        else:
            json_str = content

        faqs = json.loads(json_str)
        assert isinstance(faqs, list)
    except Exception:
        faqs = []

    return faqs


# ---------- FULL PIPELINE ----------

def generate_faqs_for_pdf(
    pdf_path: str,
    prompt_type: str = "structured",
    provider: str = "gemini",
    model: str = GEMINI_MODEL,
) -> pd.DataFrame:
    document_name = os.path.basename(pdf_path)
    chunks = load_pdf_to_chunks(pdf_path)

    all_items: List[FAQItem] = []

    for idx, chunk_text in enumerate(chunks):
        faqs = call_gemini_faq(chunk_text, model=model)

        for faq in faqs:
            q = faq.get("question", "").strip()
            a = faq.get("answer", "").strip()
            if not q or not a:
                continue

            all_items.append(
                FAQItem(
                    document_name=document_name,
                    chunk_id=idx,
                    question=q,
                    answer=a,
                    provider=provider,
                    model=model,
                    prompt_type=prompt_type,
                )
            )

    rows = [
        {
            "document_name": x.document_name,
            "chunk_id": x.chunk_id,
            "provider": x.provider,
            "model": x.model,
            "prompt_type": x.prompt_type,
            "question": x.question,
            "answer": x.answer,
        }
        for x in all_items
    ]

    return pd.DataFrame(rows)


def save_faqs_to_csv(df: pd.DataFrame, out_path: str) -> None:
    df.to_csv(out_path, index=False)
