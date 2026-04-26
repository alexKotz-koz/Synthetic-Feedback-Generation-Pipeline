from typing import Any
from dataclasses import asdict, is_dataclass
import os, re, json, random
from ..persona_generation_1 import ClinicianPersona
import base64
import mimetypes
from pathlib import Path
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity


def _persona_value(persona: ClinicianPersona | dict[str, Any], key: str) -> Any:
    if is_dataclass(persona):
        return getattr(persona, key)
    if isinstance(persona, dict):
        return persona[key]
    return getattr(persona, key)


def _persona_to_json(persona: ClinicianPersona) -> str:
    if is_dataclass(persona):
        payload = asdict(persona)
    elif isinstance(persona, dict):
        payload = persona
    elif hasattr(persona, "__dict__"):
        payload = vars(persona)
    else:
        raise TypeError("persona must be a dataclass, dict, or object with attributes")

    return json.dumps(payload, indent=2, ensure_ascii=False)


def _normalize_opening_phrase(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _extract_opening_phrase(text: str, max_words: int) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    first_clause = re.split(r"[,:;.!?\n]", stripped, maxsplit=1)[0].strip()
    words = first_clause.split()
    if not words:
        return ""

    return " ".join(words[:max_words])


def _get_banned_opening_map(
    history: list[dict[str, Any]], max_words: int
) -> dict[str, str]:
    banned: dict[str, str] = {}

    for item in history:
        opening_phrase = str(item.get("opening_phrase", "")).strip()
        if not opening_phrase:
            continue

        shortened = _extract_opening_phrase(opening_phrase, max_words=max_words)
        normalized = _normalize_opening_phrase(shortened)
        if normalized and normalized not in banned:
            banned[normalized] = shortened

    return banned


def _format_banned_openings_text(banned_opening_map: dict[str, str]) -> str:
    if not banned_opening_map:
        return "- none yet"
    return "\n".join(f"- {opening}" for opening in banned_opening_map.values())


def _build_feedback_controls(
    persona: ClinicianPersona | dict[str, Any], attempt: int = 0
) -> dict:
    rng = random.Random(
        f"{_persona_value(persona, 'id')}:{attempt}"
    )  # create a rng with a consistent 'id' seed
    detail_level = _persona_value(persona, "va_modulator")
    contextual_relevance = _persona_value(persona, "cr_modulator")

    OPENING_MODES = [
        "start with a positive first impression",
        "start with a concern or hesitation",
        "start with patient usability",
        "start with clinic workflow impact",
    ]

    if detail_level == "vague/ambiguous":
        length_buckets: dict[str, tuple[int, int]] = {
            "short": (20, 45),
            "medium": (35, 60),
        }
        structure_modes = [
            "one compact paragraph",
            "two very short paragraphs",
        ]
    else:
        length_buckets = {
            "medium": (55, 90),
            "long": (90, 160),
        }
        structure_modes = [
            "two short paragraphs",
            "three short paragraphs",
        ]

    q1_sentiment = rng.choice(["positive", "mixed", "negative"])
    # q2_sentiment = rng.choice(["positive", "mixed", "negative"])
    # q3_sentiment = rng.choice(["positive", "mixed", "negative"])

    q1_length = rng.choice(list(length_buckets.values()))
    opening_mode = rng.choice(OPENING_MODES)
    structure_mode = rng.choice(structure_modes)
    # q2_length = rng.choice(list(LENGTH_BUCKETS.values()))
    # q3_length = rng.choice(list(LENGTH_BUCKETS.values()))

    print(
        "[feedback] Controls: "
        f"persona_id={_persona_value(persona, 'id')} attempt={attempt} "
        f"detail_level={detail_level} contextual_relevance={contextual_relevance} "
        f"sentiment={q1_sentiment} length={q1_length} "
        f"opening_mode={opening_mode!r} structure_mode={structure_mode!r}"
    )

    return {
        "question_1": {"sentiment": q1_sentiment, "length": q1_length},
        "opening_mode": opening_mode,
        "structure_mode": structure_mode,
        "detail_level": detail_level,
        "contextual_relevance": contextual_relevance,
        # "question_2": {"sentiment": q2_sentiment, "length": q2_length},
        # "question_3": {"sentiment": q3_sentiment, "length": q3_length},
    }


def _controls_to_prompt_text(
    controls: dict[str, Any],
) -> str:
    lines = [
        "Private writing controls. Follow these, but do not mention them explicitly.",
    ]

    sentiment = controls["question_1"]["sentiment"]
    min_words, max_words = controls["question_1"]["length"]
    lines.append(
        f"- question_1: sentiment={sentiment}; target length={min_words}-{max_words} words"
    )
    lines.append(f"- opening mode: {controls['opening_mode']}")
    lines.append(f"- structure mode: {controls['structure_mode']}")
    if controls["detail_level"] == "vague/ambiguous":
        lines.append(
            "- detail rule: keep the answer impressionistic, brief, and somewhat noncommittal"
        )
    else:
        lines.append(
            "- detail rule: be concrete, specific, and explicit about what works or does not work"
        )
    if controls["contextual_relevance"] == "contextually_relevant":
        lines.append(
            "- contextual relevance rule: at least partially answer the question's informational intent and connect your observations directly to what was asked"
        )
    else:
        lines.append(
            "- contextual relevance rule: stay plausible and on-theme, but do not squarely answer the question's informational intent; drift toward adjacent impressions or loosely related concerns"
        )
    return "\n".join(lines)
    # for key in ("question_1", "question_2", "question_3"):
    #     sentiment = controls[key]["sentiment"]
    #     min_words, max_words = controls[key]["length"]
    #     lines.append(
    #         f"- {key}: sentiment={sentiment}; target length={min_words}-{max_words} words"
    #     )
    # return "\n".join(lines)


def _image_file_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = "image/png"

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _response_to_text(response: Any) -> str:
    content = response.content
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


def _parse_feedback_response(raw_text: str) -> dict[str, str]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {raw_text}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Model output must be a JSON object")

    # required_keys = [
    #     "question_1_answer",
    #     "question_2_answer",
    #     "question_3_answer",
    # ]
    required_keys = ["question_1_answer", "persona_use"]

    missing_keys = [key for key in required_keys if key not in parsed]
    extra_keys = [key for key in parsed if key not in required_keys]

    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    if extra_keys:
        raise ValueError(f"Unexpected keys in response: {extra_keys}")

    normalized: dict[str, str] = {}
    for key in required_keys:
        value = parsed[key]
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string, got {type(value).__name__}")
        normalized[key] = value.strip()

    return normalized


@lru_cache(maxsize=1)
def _get_embedding_model(EMBEDDING_MODEL_NAME) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed in the active Python environment."
        ) from exc

    print(f"[feedback] Loading embedding model: {EMBEDDING_MODEL_NAME}")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _embed_text(
    text: str, EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> list[float]:
    model = _get_embedding_model(EMBEDDING_MODEL_NAME)
    vector = model.encode(text, normalize_embeddings=True)
    return [float(value) for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    return float(cosine_similarity([left], [right])[0][0])


def _find_max_semantic_similarity(
    candidate_embedding: list[float], history: list[dict[str, Any]]
) -> tuple[float, str | None]:
    best_score = 0.0
    best_answer: str | None = None

    for item in history:
        previous_embedding = item.get("embedding")
        previous_answer = item.get("answer")
        if not isinstance(previous_embedding, list) or not isinstance(
            previous_answer, str
        ):
            continue

        score = _cosine_similarity(candidate_embedding, previous_embedding)
        if score > best_score:
            best_score = score
            best_answer = previous_answer

    return best_score, best_answer


# Use: main.py
# Purpose: Build a record of all previous feedback responses
def _build_feedback_history_entry(
    answer: str, opening_phrase_words: int = 3
) -> dict[str, Any]:
    opening_phrase = _extract_opening_phrase(answer, max_words=opening_phrase_words)
    entry = {
        "answer": answer,
        "embedding": _embed_text(answer),
        "opening_phrase": opening_phrase,
        "opening_phrase_normalized": _normalize_opening_phrase(opening_phrase),
    }
    print(
        f"[feedback] Created history entry with opening phrase: {entry['opening_phrase']!r}"
    )
    return entry
