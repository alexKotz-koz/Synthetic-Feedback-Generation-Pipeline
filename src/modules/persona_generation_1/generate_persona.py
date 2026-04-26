import os, uuid, random, json
from typing import Any
from dotenv import load_dotenv

from .ClinicianPersona import ClinicianPersona
from .static import US_STATES
from . import generate_genai_helpers as _dynamic
from . import generate_static_helpers as _static
from ..utils import _build_llm


def _invoke_llm_text(llm: Any, prompt: str) -> str:
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(str(part) for part in content).strip()
    return str(content).strip()


def _parse_list_response(text: str, expected_count: int) -> list[str]:
    items = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*0123456789. ").strip()
        if cleaned:
            items.append(cleaned)
    if not items:
        raise ValueError("LLM returned an empty list response")
    return items[:expected_count]


def _normalize_single_line(text: str) -> str:
    return text.splitlines()[0].strip().strip('"').rstrip(".")


def generate_personas(total_personas: int = 10, model_str: str = ""):

    llm = _build_llm(model_str=model_str)
    personas: list[ClinicianPersona] = []
    ###SDG Controllers
    sdg_controllers = _static._build_feedback_label_pairs(total_personas)

    for i in range(total_personas):
        print(f"Generating persona {i+1}")

        # 1) Sociodemographic A
        race = _static._find_race(seed=random.randint(0, 9))
        print(f"\tRace: {race}")
        age = _static._find_age(seed=random.randint(0, 3))
        print(f"\tAge: {age}")
        sex = _static._find_sex(seed=random.randint(0, 1))
        print(f"\tSex: {sex}")
        support_system = _static._find_support_system(n=random.randint(2, 8))
        print(f"\tSupport System: {support_system}")
        # 2) SDG
        contextual_relevance, vagueness_ambiguity = sdg_controllers[i]
        print(f"\tSDG CR: {contextual_relevance} | VA: {vagueness_ambiguity}")
        # 3) Occupation A
        current_occupation_title = _static._find_current_occupation_title(age=age)
        print(f"\tCurrent Occupation: {current_occupation_title}")
        annual_income = _static._find_annual_income(
            current_occupation_title=current_occupation_title,
            income_seed=random.randint(0, 100_000),
        )
        print(f"\tIncome: {annual_income}")
        years_of_experience = _static._find_years_of_experience(
            age=age, current_occupation_title=current_occupation_title
        )
        print(f"\tYears of Experience: {years_of_experience}")
        # 4) Clinical
        clinical_priorities = _static._find_clinical_priorities(n=5)
        print(f"\tClinical Priorities: {clinical_priorities}")
        personality_traits = _static._find_personality(current_occupation_title)
        print(f"\tPersonality: {personality_traits}")
        # 6) Sociodemographic B
        place_of_birth = _normalize_single_line(
            _invoke_llm_text(
                llm,
                _dynamic._generate_location_prompt(
                    seed=random.randint(0, len(US_STATES) - 1)
                ),
            )
        )
        print(f"\tPlace of Birth: {place_of_birth}")
        # 7) Occupation B
        organization_location = _normalize_single_line(
            _invoke_llm_text(
                llm,
                _dynamic._generate_location_prompt(
                    seed=random.randint(0, len(US_STATES) - 1)
                ),
            )
        )
        print(f"\tOrganization Location: {organization_location}")
        organization_affiliation = _normalize_single_line(
            _invoke_llm_text(
                llm, _dynamic._generate_organizational_affiliation_prompt()
            )
        )
        print(f"\tOrganization Name: {organization_affiliation}")
        # 8) Sociodemographic C
        hobby_count = random.randint(3, 5)
        hobbies = _parse_list_response(
            _invoke_llm_text(
                llm,
                _dynamic._generate_lifestyle_characteristics_hobbies_prompt(
                    age=age,
                    current_occupation_title=current_occupation_title,
                    years_of_experience=years_of_experience,
                    personality_traits=personality_traits,
                    support_system=support_system,
                    place_of_birth=place_of_birth,
                    organization_location=organization_location,
                    annual_income=annual_income,
                    n=hobby_count,
                ),
            ),
            expected_count=hobby_count,
        )
        hobbies = ", ".join(hobbies)
        print(f"\tHobbies Generated with {len(hobbies)} characters.")

        # 9) Technology
        technological_skill_level = _invoke_llm_text(
            llm,
            _dynamic._generate_technological_skill_level_prompt(
                age=age,
                annual_income=annual_income,
                current_occupation_title=current_occupation_title,
                years_of_experience=years_of_experience,
                organization_affiliation=organization_affiliation,
                clinical_priorities=clinical_priorities,
                lifestyle_characteristics_hobbies=hobbies,
            ),
        )
        print(
            f"\tTech Skill Generated with {len(technological_skill_level)} characters."
        )

        # 10) Build full persona before full_name
        persona = ClinicianPersona(
            id=str(uuid.uuid4()),
            full_name="",
            age=age,
            sex=sex,
            race_ethnicity=race,
            place_of_birth=place_of_birth,
            support_system=support_system,
            hobbies_narrative=hobbies,
            annual_income=annual_income,
            technological_skill_level=technological_skill_level,
            current_occupation_title=current_occupation_title,
            years_of_experience=years_of_experience,
            current_organization_name=organization_affiliation,
            current_organization_location=organization_location,
            clinical_priorities=clinical_priorities,
            personality_traits=personality_traits,
            va_modulator=vagueness_ambiguity,
            cr_modulator=contextual_relevance,
        )
        # 11) Last: full_name
        persona.full_name = _normalize_single_line(
            _invoke_llm_text(llm, _dynamic._generate_full_name_prompt(persona))
        )
        print(f"\tFull Name: {persona.full_name}")
        personas.append(persona)
    return personas


if __name__ == "__main__":
    generate_personas()
