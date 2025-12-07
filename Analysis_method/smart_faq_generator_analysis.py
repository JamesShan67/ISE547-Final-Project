import os
import sys
import json
import csv
import re
import argparse
import datetime
from typing import List, Dict, Optional

from openai import OpenAI
import PyPDF2
import requests
# from bs4 import BeautifulSoup   # Uncomment later if you add HTML support


# ---------- OpenAI client & call ----------

def get_openai_client() -> OpenAI:
    """
    Create an OpenAI client using the OPENAI_API_KEY environment variable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key)


def call_openai_chat(model_name: str, prompt: str) -> str:
    """
    Call OpenAI chat completion API and return the text content.
    """
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ---------- Gemini call (Google) ----------

def call_gemini(model_name: str, prompt: str) -> str:
    """
    Call Gemini (Google) text model via REST API.

    Assumes you are using the Gemini API (ai.google.dev) and have set:
    GEMINI_API_KEY in your environment.

    Typical model names:
      - gemini-1.5-flash
      - gemini-1.5-pro
      - gemini-2.0-flash
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    # Endpoint for the Gemini API (text / multimodal)
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

    # Very simple extraction – adjust if your response shape differs
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(
            f"Unexpected Gemini response format: {json.dumps(data, indent=2)}"
        )

    return text.strip()


# ---------- Unified LLM entry point ----------

def call_llm(provider: str, model_name: str, prompt: str) -> str:
    """
    Call the specified LLM provider with a prompt and return the raw text output.
    """
    provider = provider.lower()
    if provider == "openai":
        return call_openai_chat(model_name, prompt)
    elif provider == "gemini":
        return call_gemini(model_name, prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: openai, gemini")


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


# (Optional) HTML support for later
# def extract_text_from_html(file_path: str) -> str:
#     with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#         html = f.read()
#     soup = BeautifulSoup(html, "html.parser")
#     return soup.get_text(separator="\n")


def extract_text(file_path: str) -> str:
    """
    Dispatch to the appropriate text extraction function based on file extension.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".txt", ".md"]:
        return extract_text_from_txt(file_path)
    # elif ext in [".html", ".htm"]:
    #     return extract_text_from_html(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ---------- Chunking ----------

def chunk_text(text: str, max_chars: int = 2000) -> List[str]:
    """
    Simple chunker: split into paragraphs, then pack into ~max_chars chunks.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for p in paragraphs:
        # If a single paragraph is too long, split it directly
        if len(p) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(p), max_chars):
                chunks.append(p[i:i + max_chars])
            continue

        # Try to add paragraph to current chunk
        if len(current) + len(p) + 2 <= max_chars:
            current += ("\n\n" + p) if current else p
        else:
            if current:
                chunks.append(current.strip())
            current = p

    if current:
        chunks.append(current.strip())

    return chunks


# ---------- Prompt templates ----------

PROMPTS: Dict[str, str] = {
    "simple": """
You are an assistant that generates FAQ-style questions and answers.

Read the following text and create 2–3 useful FAQ entries.
Each entry should be a JSON object with the fields "question" and "answer".

Return your output as a JSON array, like:
[
  {"question": "...", "answer": "..."},
  {"question": "...", "answer": "..."}
]

Requirements:
- Questions should be practical and helpful to someone who does not want to read the full document.
- Answers should be short and clear (1–3 sentences).
- Avoid duplicate or very similar questions.
- Do not include any text before or after the JSON array.

Text:
\"\"\"{chunk}\"\"\"
""".strip(),

    "user_centered": """
You are generating FAQ-style questions and answers for a user who does not want to read a long document.

Based on the text below, write 2–3 FAQ entries that focus on what a typical reader would care about:
- rules, policies, deadlines, procedures, or "how do I..." style questions.
Each entry should be a JSON object with "question" and "answer".

Return your output as a JSON array, like:
[
  {"question": "...", "answer": "..."},
  {"question": "...", "answer": "..."}
]

Requirements:
- Make questions practical and concrete, not trivial rephrasings.
- Answers should be short and clear (1–3 sentences).
- Avoid duplicate or very similar questions.
- Do not include any text before or after the JSON array.

Text:
\"\"\"{chunk}\"\"\"
""".strip(),

    "structured": """
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
""".strip(),
}


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

    # Find the JSON array inside the text
    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return None


# ---------- FAQ generation using provider+model ----------

def generate_faq_for_chunk(
    chunk: str,
    model_name: str,
    provider: str,
    prompt_template: str,
) -> List[Dict[str, str]]:
    """
    Call chosen LLM provider to generate FAQ entries for a single chunk of text.
    Returns a list of {"question": ..., "answer": ...}.
    """
    prompt = prompt_template.replace("{chunk}", chunk)

    content = call_llm(provider=provider, model_name=model_name, prompt=prompt)
    content = content.strip()

    # Step 1: try to parse the whole content as JSON
    parsed = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Step 2: try to extract a JSON array substring (for Gemini etc.)
        json_snippet = extract_json_array_from_text(content)
        if json_snippet:
            try:
                parsed = json.loads(json_snippet)
            except json.JSONDecodeError:
                print(
                    f"Warning: still failed to parse JSON after extracting array (provider={provider})."
                )
                # Uncomment for debugging:
                # print("Raw model output:\n", content)
                return []
        else:
            print(
                f"Warning: could not find JSON array in model output (provider={provider})."
            )
            # Uncomment for debugging:
            # print("Raw model output:\n", content)
            return []

    # Normalize to list
    if isinstance(parsed, dict):
        faqs = [parsed]
    elif isinstance(parsed, list):
        faqs = parsed
    else:
        print(
            f"Warning: parsed JSON is not a list or dict (type={type(parsed)}), skipping chunk."
        )
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
    model_name: str,
    provider: str,
    max_chars_per_chunk: int = 2000,
    prompt_name: str = "simple",
) -> List[Dict[str, str]]:
    """
    End-to-end pipeline:
    - extract text
    - chunk
    - call provider+model for each chunk
    - aggregate + deduplicate FAQs
    """
    if prompt_name not in PROMPTS:
        raise ValueError(
            f"Unknown prompt_name: {prompt_name}. "
            f"Available: {list(PROMPTS.keys())}"
        )

    prompt_template = PROMPTS[prompt_name]

    print(f"[*] Extracting text from: {file_path}")
    text = extract_text(file_path)

    print("[*] Splitting into chunks...")
    chunks = chunk_text(text, max_chars=max_chars_per_chunk)
    print(f"[*] Number of chunks: {len(chunks)}")

    all_faqs: List[Dict[str, str]] = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"[*] Processing chunk {i}/{len(chunks)} with provider={provider}, model={model_name}...")
        faqs = generate_faq_for_chunk(
            chunk=chunk,
            model_name=model_name,
            provider=provider,
            prompt_template=prompt_template,
        )
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


# ---------- CSV logging for evaluation ----------

def log_faqs_to_csv(
    faqs: List[Dict[str, str]],
    csv_path: str,
    document_name: str,
    model_name: str,
    prompt_name: str,
    provider: str,
    run_id: str,
) -> None:
    """
    Append FAQ entries to a CSV file for later evaluation.
    Columns: run_id, document, provider, model, prompt, question, answer
    """
    if not faqs:
        print("[*] No FAQs to log; skipping CSV logging.")
        return

    # Ensure parent directory exists (if csv_path has a folder)
    csv_dir = os.path.dirname(csv_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "run_id",
        "document",
        "provider",
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
                    "run_id": run_id,
                    "document": document_name,
                    "provider": provider,
                    "model": model_name,
                    "prompt": prompt_name,
                    "question": item["question"],
                    "answer": item["answer"],
                }
            )

    print(f"[*] Logged {len(faqs)} FAQs to CSV: {csv_path}")


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Smart FAQ Generator – turn long documents into FAQs."
    )
    parser.add_argument("input_file", help="Path to input document (PDF or TXT)")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output FAQ text file (default: auto-named with model/prompt/provider/timestamp)",
        default=None,
    )
    parser.add_argument(
        "--provider",
        help="LLM provider: openai or gemini (default: openai)",
        default="openai",
    )
    parser.add_argument(
        "--model",
        help="Model name. For openai: e.g., gpt-4o-mini. For gemini: e.g., gemini-2.0-flash",
        default="gpt-4o-mini",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Max characters per chunk (default: 2000)",
    )
    parser.add_argument(
        "--prompt",
        help="Prompt style to use: simple, user_centered, structured (default: simple)",
        default="simple",
    )
    parser.add_argument(
        "--csv-log",
        help="Path to CSV file to append FAQ entries for evaluation (default: faq_log.csv)",
        default="faq_log.csv",
    )

    args = parser.parse_args()

    input_file = args.input_file
    if not os.path.exists(input_file):
        print(f"Error: input file does not exist: {input_file}")
        sys.exit(1)

    provider = args.provider.lower()
    if provider not in ["openai", "gemini"]:
        print(f"Error: unsupported provider '{provider}'. Use 'openai' or 'gemini'.")
        sys.exit(1)

    # Create a run_id and timestamp for filenames & logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = timestamp  # simple, unique enough

    # Decide output text file path
    if args.output:
        output_file = args.output
    else:
        base = os.path.splitext(os.path.basename(input_file))[0]
        safe_model = args.model.replace("/", "_")
        output_file = f"{base}_{provider}_{args.prompt}_{safe_model}_{timestamp}.txt"

    # Generate FAQs
    faqs = generate_faq_for_document(
        file_path=input_file,
        model_name=args.model,
        provider=provider,
        max_chars_per_chunk=args.max_chars,
        prompt_name=args.prompt,
    )

    # Save as text
    save_faqs_as_text(faqs, output_file)

    # CSV logging
    if args.csv_log:
        document_name = os.path.basename(input_file)
        log_faqs_to_csv(
            faqs=faqs,
            csv_path=args.csv_log,
            document_name=document_name,
            model_name=args.model,
            prompt_name=args.prompt,
            provider=provider,
            run_id=run_id,
        )


if __name__ == "__main__":
    main()
