import os, json
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from IPython.display import Image, display

from pathlib import Path
from typing import Any, TypedDict

from modules.persona_generation_1.static import (
    MIN_PRACTICE_AGE_BY_OCCUPATION,
    INCOME_BY_OCCUPATION,
    PERSONALITY_BY_OCCUPATION,
)


class ValidationState(TypedDict):
    personas: list[dict[str, Any]]
    output_path: str | None
    llm: Any
    issues: list[str]


def build_llm(model: str = "gpt-5.4-mini") -> ChatOpenAI:
    load_dotenv()
    return ChatOpenAI(model=model, reasoning_effort="medium")


def regenerate_name(
    persona: dict[str, Any],
    used_names: set[str],
    llm: Any,
) -> str:
    print(f"Duplicate name found: {persona['full_name']}")
    excluded_names = "\n".join(sorted(used_names)) if used_names else "None"

    prompt = f"""
Generate a realistic full name for a fictional clinician persona.

Persona characteristics:
- age: {persona.get("age")}
- sex: {persona.get("sex")}
- race_ethnicity: {persona.get("race_ethnicity")}
- place_of_birth: {persona.get("place_of_birth")}
- current_occupation_title: {persona.get("current_occupation_title")}
- organization_location: {persona.get("organization_location")}
- personality_traits: {persona.get("personality_traits")}

Already used names:
{excluded_names}

Requirements:
- Return only the full name
- Do not repeat any excluded name
- Do not include titles
- Output format: Firstname Lastname
""".strip()

    response = llm.invoke(prompt)
    content = response.content
    return content.strip() if isinstance(content, str) else str(content).strip()


def dedupe_full_names(state: ValidationState) -> ValidationState:
    print("Running full name de-duplication tool...\n")
    personas = []
    seen_names: set[str] = set()
    llm = state["llm"]

    for persona in state["personas"]:
        current = dict(persona)
        full_name = str(current.get("full_name", "")).strip()
        normalized = full_name.casefold()

        if not full_name or normalized in seen_names:
            current["full_name"] = regenerate_name(current, seen_names, llm)
            normalized = current["full_name"].casefold()

        seen_names.add(normalized)
        personas.append(current)

    return {
        **state,
        "personas": personas,
    }


def normalize_organization_locations(state: ValidationState) -> ValidationState:
    print("Running organization location and name validation...")
    personas = []
    first_location_by_affiliation: dict[str, str] = {}

    for persona in state["personas"]:
        current = dict(persona)
        affiliation = str(current.get("organization_affiliation", "")).strip()
        location = str(current.get("organization_location", "")).strip()

        if affiliation:
            if affiliation not in first_location_by_affiliation:
                first_location_by_affiliation[affiliation] = location
            else:
                current["organization_location"] = first_location_by_affiliation[
                    affiliation
                ]

        personas.append(current)

    return {
        **state,
        "personas": personas,
    }


def review_persona_with_llm(persona: dict[str, Any], llm: Any) -> dict[str, Any]:
    print(f"Running final review with {llm.model_name or llm.model} on {persona['id']}")
    prompt = f"""
Review the clinician persona JSON below for internal consistency and plausibility.

Checks to perform:
- age, occupation, and years_of_experience should make sense together
- technological_skill_level should match age, experience, and occupation
- organization_affiliation and organization_location should be consistent
- support_system and lifestyle_characteristics_hobbies should remain plausible

Instructions:
- Return exactly one valid JSON object with the same keys as the input
- Keep the id unchanged
- Make the smallest possible number of edits needed
- If no changes are needed, return the persona unchanged

Persona JSON:
{json.dumps(persona, indent=2, ensure_ascii=False)}
""".strip()

    response = llm.invoke(prompt)
    content = response.content
    text = content.strip() if isinstance(content, str) else str(content).strip()

    try:
        reviewed = json.loads(text)
    except json.JSONDecodeError:
        return persona

    if not isinstance(reviewed, dict):
        return persona

    merged = dict(persona)
    for key in merged:
        if key in reviewed:
            merged[key] = reviewed[key]
    merged["id"] = persona["id"]
    return merged


def llm_reasoning_review(state: ValidationState) -> ValidationState:
    llm = state["llm"]
    reviewed = [review_persona_with_llm(persona, llm) for persona in state["personas"]]
    return {
        **state,
        "personas": reviewed,
    }


def write_output(state: ValidationState) -> ValidationState:
    output_path = state.get("output_path")
    if output_path:
        Path(output_path).write_text(
            json.dumps(state["personas"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return state


def build_graph():
    print("Building Agent Graph...")
    graph = StateGraph(ValidationState)

    graph.add_node("dedupe_full_names", dedupe_full_names)
    graph.add_node("normalize_organization_locations", normalize_organization_locations)
    graph.add_node("llm_reasoning_review", llm_reasoning_review)
    graph.add_node("write_output", write_output)

    graph.add_edge(START, "dedupe_full_names")
    graph.add_edge("dedupe_full_names", "normalize_organization_locations")
    graph.add_edge("normalize_organization_locations", "llm_reasoning_review")
    graph.add_edge("llm_reasoning_review", "write_output")
    graph.add_edge("write_output", END)
    agent_graph = graph.compile()
    display(Image(agent_graph.get_graph().draw_mermaid_png()))

    return agent_graph


# In-memory validation (run in persona generation pipeline)
def validate_personas(
    personas: list[dict[str, Any]], llm: Any | None = None
) -> list[dict[str, Any]]:
    llm = llm or build_llm()
    app = build_graph()
    result = app.invoke(
        {
            "personas": personas,
            "output_path": None,
            "llm": llm,
            "issues": [],
        }
    )
    return result["personas"]


# File-based validation (run with existing personas file)
def validate_personas_file(
    input_path: str, output_path: str | None = None
) -> list[dict[str, Any]]:
    source = Path(input_path)
    destination = (
        Path(output_path)
        if output_path is not None
        else source.with_name(f"{source.stem}_validated{source.suffix}")
    )

    personas = json.loads(source.read_text(encoding="utf-8"))
    llm = build_llm()
    app = build_graph()

    result = app.invoke(
        {
            "personas": personas,
            "output_path": str(destination),
            "llm": llm,
            "issues": [],
        }
    )
    return result["personas"]


if __name__ == "__main__":
    validate_personas()
