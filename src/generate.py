"""
src/generate.py

Calls Gemini to generate a summary of a source text.
This is the "generation" step of the pipeline: source -> Gemini -> summary.
"""

import os
import google.generativeai as genai


def configure_gemini(api_key: str = None) -> None:
    """Configure the Gemini client. Reads from GEMINI_API_KEY env var by default."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "No Gemini API key found. Set the GEMINI_API_KEY environment variable."
        )
    genai.configure(api_key=key)


def generate_summary(source_text: str, model_name: str = "gemini-1.5-flash") -> str:
    """
    Generate a summary of source_text using Gemini.

    Args:
        source_text: The text to summarize.
        model_name: Which Gemini model to use.

    Returns:
        The generated summary as a string.
    """
    model = genai.GenerativeModel(model_name)

    prompt = (
        "Summarize the following text in 2-3 sentences. "
        "Only include information that is explicitly stated in the text. "
        "Do not add outside knowledge or assumptions.\n\n"
        f"TEXT:\n{source_text}"
    )

    response = model.generate_content(prompt)
    return response.text.strip()


if __name__ == "__main__":
    configure_gemini()
    sample_text = (
        "Bridgewater Associates is an investment management firm founded in 1975 "
        "by Ray Dalio. The firm is known for its systems-driven approach to "
        "understanding markets and economies."
    )
    summary = generate_summary(sample_text)
    print(f"Summary: {summary}")
