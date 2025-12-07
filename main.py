# main.py
"""
CLI entry point for the Smart FAQ Generator (final pipeline).

Usage example:
    export GEMINI_API_KEY="..."
    python main.py docs/Nikon_user_guide.pdf
"""

import os
import sys
import argparse
import datetime

from faq_core import (
    generate_faq_for_document,
    save_faqs_as_text,
    log_faqs_to_csv,
    GEMINI_MODEL,
    MAX_CHARS_PER_CHUNK,
)


def main():
    parser = argparse.ArgumentParser(
        description="Smart FAQ Generator – Gemini 2.0 Flash + structured prompt."
    )
    parser.add_argument("input_file", help="Path to input document (PDF or TXT)")
    parser.add_argument(
        "-o",
        "--output",
        help="Output FAQ text file (default: auto-named with timestamp).",
        default=None,
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=MAX_CHARS_PER_CHUNK,
        help="Max characters per chunk (default: 2000)",
    )
    parser.add_argument(
        "--csv-log",
        help="CSV file to append FAQ entries for evaluation (default: faq_log.csv).",
        default="faq_log.csv",
    )

    args = parser.parse_args()

    input_file = args.input_file
    if not os.path.exists(input_file):
        print(f"Error: input file does not exist: {input_file}")
        sys.exit(1)

    # Timestamp for filenames & logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Decide output text file path
    if args.output:
        output_file = args.output
    else:
        base = os.path.splitext(os.path.basename(input_file))[0]
        output_file = f"{base}_gemini_structured_{timestamp}.txt"

    # Generate FAQs with final pipeline
    faqs = generate_faq_for_document(
        file_path=input_file,
        max_chars_per_chunk=args.max_chars,
    )

    # Save as text for easy viewing
    save_faqs_as_text(faqs, output_file)

    # CSV logging (optional but nice for analysis section in report)
    if args.csv_log:
        document_name = os.path.basename(input_file)
        log_faqs_to_csv(
            faqs=faqs,
            csv_path=args.csv_log,
            document_name=document_name,
        )


if __name__ == "__main__":
    main()
