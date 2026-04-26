from typing import Any
from dotenv import load_dotenv
import os


def _build_llm(model_str: str) -> Any:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or .env file."
        )

    try:
        from langchain_openai import ChatOpenAI as RuntimeChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "langchain-openai is not installed in the active Python environment."
        ) from exc

    if "gpt-5" in model_str:
        return RuntimeChatOpenAI(
            model=model_str, reasoning_effort="medium", api_key=api_key
        )
    elif "gpt-4" in model_str:
        return RuntimeChatOpenAI(model=model_str, temperature=0.7, api_key=api_key)
    else:
        return RuntimeChatOpenAI(model=model_str, api_key=api_key)
