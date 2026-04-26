import os, json, random, re
import base64
import mimetypes
from pathlib import Path
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from sklearn.metrics.pairwise import cosine_similarity

from ..utils import _build_llm
from modules.persona_generation_1 import ClinicianPersona

from .generate_feedback_helpers import (
    _get_banned_opening_map,
    _extract_opening_phrase,
    _normalize_opening_phrase,
    _persona_value,
    _build_feedback_controls,
    _format_banned_openings_text,
    _persona_to_json,
    _controls_to_prompt_text,
    _image_file_to_data_url,
    _parse_feedback_response,
    _response_to_text,
    _embed_text,
    _find_max_semantic_similarity,
)

PI_DESCRIPTION_CONTEXT = """
PI, the patient intake application is a web-based platform for automating and enhancing the patient intake process. PI transforms standard paper demographic questionnaires, patient reported outcome surveys, and other medical, dental, and behavioral health surveys into clean, user friendly forms for patients to fill out on their personal laptops or mobile devices. We work with clinics and hospitals to ensure all of their paper or electronic based patient intake forms, questionnaires, and surveys can be converted to a digital version. Medical receptionists and administrative staff set up templates for different visit types, to enable an automatic distribution of the correct forms to the patients upon the scheduling of a visit. We directly integrate our solution with popular electronic health records such as Epic, athenaHealth, Oracle health, and others and pass discrete datapoints to EHR native forms, minimizing the number of PDF documents in your EHR. PI is accessible for patients and medical providers 24 hours a day, 7 days per week.
"""

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def generate_feedback(
    persona: ClinicianPersona | dict,
    screenshot_paths: list[str] | None,
    history: list[dict[str, Any]] | None,
    llm: ChatOpenAI | None = None,
    model_str: str = "",
    max_similarity: float = 0.65,
    max_attempts: int = 3,
    banned_opening_threshold: int = 3,
) -> dict[str, str]:

    llm = llm or _build_llm(model_str=model_str)
    screenshot_paths = screenshot_paths or []
    history = history or []

    print(
        "\n[feedback] Starting generation: "
        f"persona_id={_persona_value(persona, 'id')} history_size={len(history)} "
        f"similarity_threshold={max_similarity}"
    )

    common_system_messages = [
        SystemMessage(
            content="You are simulating a clinician giving realistic first-person feedback."
        ),
        SystemMessage(
            content="Use the provided persona and occupation title to shape tone, level of detail, and features you would use/pay attention to based on the occupation title. Write in first person."
        ),
        SystemMessage(
            content="Explicitly use the va_modulator charactersitic in the persona to shape the level of detail in the feedback response. If vague/ambiguous, keep the response relatively short and use vague/ambiguous language. If clear/detailed, feel free to vary the length of the feedback response, but ensure all topics are throughoughly discussed, not using any vague or ambiguous language."
        ),
        SystemMessage(
            content="Explicitly use the cr_modulator characteristic in the persona. If contextually_relevant, address the question's informational intent directly. If contextually_irrelevant, miss or sidestep that intent and speak instead about adjacent healthcare technology concerns in a plausible first-person way."
        ),
        SystemMessage(
            content="After you answer the feedback question, write a short description of how you used the charactersitics of the persona to shape your response. Provide this description in the 'persona_use' key."
        ),
        SystemMessage(
            content='Return exactly one valid JSON object with key "question_1_answer", "persona_use". Do not include markdown, code fences, or extra keys. Each value must be a string.'
        ),
        SystemMessage(
            content="Do not restate the persona explicitly. Instead, let it show through in the wording, concerns, priorities, and examples."
        ),
        SystemMessage(
            content="Keep the feedback realistic and natural. It should sound like a clinician speaking candidly, not a polished product review. It is acceptable to be mixed, critical, uncertain, or blunt when appropriate."
        ),
    ]

    local_banned_opening_map = _get_banned_opening_map(
        history, max_words=banned_opening_threshold
    )
    last_opening_phrase = ""
    last_similarity_score: float | None = None
    last_matched_answer: str | None = None

    # attempt to build feedback response using the persona object, controls, and banned opening phrases. repeat until feedback response average pairwise cosine similarity is less than the threshold (i.e., is diverse)
    for attempt in range(max_attempts):
        controls = _build_feedback_controls(persona=persona, attempt=attempt)
        retry_instruction = ""
        banned_openings_text = _format_banned_openings_text(local_banned_opening_map)
        if attempt > 0:
            retry_reasons: list[str] = []

            if last_opening_phrase:
                retry_reasons.append(
                    f"Do not begin with this opening or anything close to it: '{last_opening_phrase}'."
                )

            if last_similarity_score is not None:
                retry_reasons.append(
                    "Your previous draft was too semantically close to an earlier response "
                    f"(cosine similarity {last_similarity_score:.3f})."
                )

            if last_matched_answer:
                retry_reasons.append(
                    "Do not mirror the framing of the earlier response that discussed similar workflow concerns."
                )

            retry_instruction = (
                "Retry instruction: "
                + " ".join(retry_reasons)
                + " Keep the same persona and overall meaning, but change the opening phrase, wording, and sentence structure so it reads clearly differently."
            )

        if controls["contextual_relevance"] == "contextually_irrelevant":
            question_text = (
                "Give a first-person opinion about healthcare technology, EHR burden, "
                "or clinic change fatigue. Do not mention PI, the application, the "
                "wireframes, screenshots, screens, layout, or interface details."
            )
            system_messages = common_system_messages + [
                SystemMessage(
                    content="For contextually_irrelevant responses, do not mention PI, the application, the screenshots, wireframes, screens, layout, navigation, forms, buttons, tabs, colors, or interface details. Speak only in broad terms about healthcare technology, EHR burden, digital workflow frustration, or clinic change fatigue."
                ),
                SystemMessage(
                    content="Base your answer only on the persona object and the writing controls. Do not use the PI description or screenshot content."
                ),
            ]
            human_content = [
                {
                    "type": "text",
                    "text": (
                        f"Persona JSON:\n{_persona_to_json(persona)}\n\n"
                        f"{_controls_to_prompt_text(controls)}\n\n"
                        "Do not begin the response with any of these opening phrases or obvious close variants:\n"
                        f"{banned_openings_text}\n\n"
                        f"{retry_instruction}\n\n"
                        "Please answer this question in first person:\n"
                        f"{question_text}"
                    ),
                }
            ]
        else:
            question_text = (
                "What are your initial thoughts and opinions on the wireframe "
                "screenshots of the PI application interface?"
            )
            system_messages = common_system_messages + [
                SystemMessage(
                    content="You are simulating a clinician providing initial impressions and thoughts on wire-frames for a patient intake application your clinic is going to implement."
                ),
                SystemMessage(
                    content="Base your answers only on the provided PI description, screenshots, and persona object."
                ),
                SystemMessage(
                    content="It should sound like a clinician discussing the wire-frame in a think-aloud session."
                ),
            ]
            human_content = [
                {
                    "type": "text",
                    "text": (
                        f"Persona JSON:\n{_persona_to_json(persona)}\n\n"
                        f"PI description:\n{PI_DESCRIPTION_CONTEXT}\n\n"
                        f"{_controls_to_prompt_text(controls)}\n\n"
                        "Do not begin the response with any of these opening phrases or obvious close variants:\n"
                        f"{banned_openings_text}\n\n"
                        f"{retry_instruction}\n\n"
                        "The following wireframe screenshots show parts of the PI interface.\n"
                        "Use them together with the PI description when answering.\n\n"
                        "Please answer this question in first person:\n"
                        f"{question_text}"
                    ),
                }
            ]

        if controls["contextual_relevance"] == "contextually_relevant":
            for screenshot_path in screenshot_paths:
                human_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_file_to_data_url(screenshot_path)},
                    }
                )

        print(
            f"[feedback] Attempt {attempt + 1}/{max_attempts} with {len(local_banned_opening_map)} banned openings"
        )
        response = llm.invoke(system_messages + [HumanMessage(content=human_content)])
        parsed_response = _parse_feedback_response(_response_to_text(response))
        parsed_response["controls"] = {
            "sentiment": controls["question_1"]["sentiment"],
            "word_range": controls["question_1"]["length"],
            "opening_mode": controls["opening_mode"],
            "structure_mode": controls["structure_mode"],
            "feedback_detail": controls["detail_level"],
            "contextual_relevance": controls["contextual_relevance"],
        }

        answer = parsed_response["question_1_answer"]
        opening_phrase = _extract_opening_phrase(
            answer, max_words=banned_opening_threshold
        )
        normalized_opening = _normalize_opening_phrase(opening_phrase)
        opening_phrase_is_banned = normalized_opening in local_banned_opening_map
        candidate_embedding = _embed_text(
            text=answer, EMBEDDING_MODEL_NAME=EMBEDDING_MODEL_NAME
        )
        similarity_score, matched_answer = _find_max_semantic_similarity(
            candidate_embedding, history
        )

        print(
            "[feedback] Attempt result: "
            f"opening_phrase={opening_phrase!r} banned={opening_phrase_is_banned} "
            f"semantic_similarity={similarity_score:.3f}"
        )

        if not opening_phrase_is_banned and similarity_score <= max_similarity:
            print("[feedback] Accepted response")
            return parsed_response

        if normalized_opening:
            local_banned_opening_map[normalized_opening] = opening_phrase
            print(
                f"[feedback] Added rejected opening phrase to local ban list: {opening_phrase!r}"
            )

        last_opening_phrase = opening_phrase
        last_similarity_score = similarity_score
        last_matched_answer = matched_answer

        if matched_answer:
            print(
                "[feedback] Closest prior answer snippet: " f"{matched_answer[:120]!r}"
            )

    print("[feedback] Returning final attempt after exhausting retries")
    return parsed_response


if __name__ == "__main__":
    generate_feedback()
