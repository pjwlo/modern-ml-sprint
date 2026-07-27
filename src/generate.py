"""
src/generate.py

Calls Gemini to generate a summary of a source text.
This is the "generation" step of the pipeline: source -> Gemini -> summary.

Note: this uses the current `google-genai` SDK. The older `google-generativeai`
package is fully deprecated (as of 2025) and no longer receives model updates --
using it will fail against current models like the gemini-3.x family.
"""

import os
from google import genai


def get_client(api_key: str = None) -> genai.Client:
    """
    Create a Gemini client. Reads from GEMINI_API_KEY env var by default
    (the new SDK also auto-detects GOOGLE_GENAI_API_KEY / GOOGLE_API_KEY).
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "No Gemini API key found. Set the GEMINI_API_KEY environment variable."
        )
    return genai.Client(api_key=key)


def generate_summary(
    source_text: str,
    client: genai.Client = None,
    model_name: str = "gemini-3.6-flash",
) -> str:
    """
    Generate a summary of source_text using Gemini.

    Args:
        source_text: The text to summarize.
        client: An existing genai.Client instance. If not provided, one is
            created from the GEMINI_API_KEY environment variable.
        model_name: Which Gemini model to use.

    Returns:
        The generated summary as a string.
    """
    client = client or get_client()

    prompt = (
        "Summarize the following text in 2-3 sentences. "
        "Only include information that is explicitly stated in the text. "
        "Do not add outside knowledge or assumptions.\n\n"
        f"TEXT:\n{source_text}"
    )

    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text.strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # A file path was provided on the command line, e.g.:
        #   python src/generate.py path/to/source.txt
        file_path = sys.argv[1]
        with open(file_path, "r", encoding="utf-8") as f:
            source_text = f.read()
        print(f"Read {len(source_text)} characters from {file_path}\n")
    else:
        # No file provided -- fall back to the built-in sample text.
        source_text = (
            "Bridgewater Associates is an investment management firm founded in 1975 "
            "by Ray Dalio. The firm is known for its systems-driven approach to "
            "understanding markets and economies."
        )
        print("No file provided, using built-in sample text.\n")

    summary = generate_summary(source_text)
    print(f"Summary: {summary}")
