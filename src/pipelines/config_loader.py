"""
Configuration utilities for AdvisingBench pipelines.
"""

from pathlib import Path
import os

import yaml
from dotenv import load_dotenv


CONFIG_FILE = Path("configs/config.yaml")


def load_config() -> dict:
    """Load project config from configs/config.yaml."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_FILE}")

    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_api_keys() -> dict:
    """Load API keys from .env."""
    load_dotenv()

    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
    }
