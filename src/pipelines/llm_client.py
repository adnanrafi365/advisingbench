"""
LLM client for AdvisingBench.

This file safely loads the Gemini API key from .env and sends prompts to Gemini.

Important:
- Do not hard-code API keys.
- Keep .env ignored by Git.
- This client is used by the LLM-only, Basic RAG, and Citation RAG pipelines.
"""

import os
from dotenv import load_dotenv
from google import genai


DEFAULT_MODEL = "gemini-2.5-flash"


def get_gemini_client():
    """Create a Gemini client using GEMINI_API_KEY from .env."""
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise ValueError(
            "Missing GEMINI_API_KEY. Add your Gemini API key to the .env file first."
        )

    return genai.Client(api_key=api_key)


def generate_text(prompt: str, model_name: str = DEFAULT_MODEL) -> str:
    """Generate text from Gemini."""
    client = get_gemini_client()

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    return response.text or ""


def main():
    """Small test call."""
    test_prompt = (
        "In one sentence, explain why citation grounding matters for academic advising."
    )

    answer = generate_text(test_prompt)

    print("Model response:")
    print(answer)


if __name__ == "__main__":
    main()
