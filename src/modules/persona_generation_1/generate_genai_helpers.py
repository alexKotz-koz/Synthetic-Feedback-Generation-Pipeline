from .ClinicianPersona import ClinicianPersona
from .static import US_STATES


def _generate_location_prompt(seed: int) -> str:
    state = US_STATES[seed % len(US_STATES)]
    return (
        f"Generate a realistic U.S. city in {state}. Return only one line in the format City, State. "
        f"Do not include explanation, bullet points, or extra text."
    )


def _generate_organizational_affiliation_prompt() -> str:
    return """
Generate a realistic name for a healthcare organization in the family medicine or primary care specialties.

Requirements:
- Return only the organization name
- Make it sound like a real clinic, practice, medical group, health center, or outpatient organization
- Keep it to one line
- Do not include quotation marks
- Do not include explanation
- Do not include a street address
- Do not include obviously fake or playful wording

Good examples:
Riverview Family Medicine Clinic
Westside Community Health Center
Lakeshore Medical Group
Harbor Point Outpatient Care
Northgate Primary Care Associates

Return only the organization name.
""".strip()


def _generate_lifestyle_characteristics_hobbies_prompt(
    place_of_birth: str,
    organization_location: str,
    annual_income: int,
    age: int,
    current_occupation_title: str,
    years_of_experience: int,
    personality_traits: str,
    support_system: list[str],
    n: int,
) -> str:
    if n < 1:
        raise ValueError("n must be at least 1")

    support_summary = ", ".join(support_system)

    return f"""
Generate exactly {n} realistic hobbies or lifestyle characteristics for a fictional clinician persona.

Persona context:
- Age: {age}
- Current occupation title: {current_occupation_title}
- Years of experience: {years_of_experience}
- Personality traits: {personality_traits}
- Place of birth / where they grew up: {place_of_birth}
- Current organization location / where they likely live now: {organization_location}
- Approximate annual income: ${annual_income:,}
- Support system: {support_summary}

Instructions:
- Generate hobbies and lifestyle patterns that plausibly fit this specific person, not a generic clinician.
- Use age, career stage, workload, likely schedule, location, climate, income, and personality to shape the choices.
- Include a mix of hobby types rather than making all items outdoorsy, fitness-based, or food-related.
- Across the full list, include variety from different categories when possible:
  one practical routine,
  one social or family-oriented habit,
  one leisure or entertainment habit,
  one location-sensitive activity,
  and one low-energy or home-based activity.
- Some items can be true hobbies and some can be recurring lifestyle patterns.
- Favor concrete, slightly specific activities over generic phrases.

Avoid overused outputs such as:
- farmers market meal prep
- weekend hikes
- trail running
- coffee at home
- kayaking
- generic meal prep
unless they are unusually well justified by the persona.

Constraints:
- Return exactly {n} items
- One item per line
- No numbering
- No bullet points
- No explanation
- No full sentences
- Each item should be short, about 2 to 7 words
- Do not repeat items
- Do not make the whole list sound affluent, outdoorsy, or highly curated
- At least one item should feel ordinary rather than aspirational

Return only the {n} items.
""".strip()


def _generate_technological_skill_level_prompt(
    age: int,
    annual_income: int,
    current_occupation_title: str,
    years_of_experience: int,
    organization_affiliation: str,
    clinical_priorities: list[str],
    lifestyle_characteristics_hobbies: str,
) -> str:
    priorities = ", ".join(clinical_priorities)
    return f"""
Generate a realistic 3 to 4 sentence narrative describing how a fictional clinician actually interacts with technology in daily work and personal life.

Persona context:
- Age: {age}
- Approximate annual income: ${annual_income:,}
- Current occupation title: {current_occupation_title}
- Years of experience: {years_of_experience}
- Organization affiliation: {organization_affiliation}
- Clinical priorities: {priorities}
- Lifestyle characteristics and hobbies: {lifestyle_characteristics_hobbies}

Instructions:
- Write like an observer describing this clinician's real-world technology habits, not like a résumé or performance review.
- Make the description specific to this role. A pharmacist, behavioral health counselor, medical assistant, and orthopedic surgeon should not sound interchangeable.
- Treat technology skill as a spectrum, not a binary. The clinician may be efficient in some areas and hesitant, slower, or more dependent in others.
- Describe concrete workplace behavior:
  how they use or avoid the EHR, templates, inboxes, scheduling tools, telehealth, device setup, task routing, order entry, chart review, or troubleshooting.
- Include one realistic friction point or limitation:
  for example, resistance to new interfaces, reliance on familiar workflows, difficulty recovering from unexpected errors, preference for paper notes first, or dependence on coworkers for rare tasks.
- Include one realistic strength:
  for example, fast navigation in routine workflows, strong use of templates, comfort troubleshooting minor issues, good telehealth flow, or careful digital documentation.
- Briefly connect their personal technology habits to the same general style without making the personal examples dominate the paragraph.
- Let age, experience, and clinical priorities influence the balance between speed, caution, efficiency, and adaptability.

Constraints:
- Return only the narrative
- Write exactly 3 or 4 sentences
- Do not use bullet points
- Do not mention race, ethnicity, or sex
- Do not use the phrases "high technology literacy" or "low technology literacy"
- Avoid generic phrases like "good with technology" or "comfortable with digital tools" unless you immediately support them with specific behavior
- Avoid making every clinician sound highly optimized, fast, and self-sufficient
""".strip()


def _generate_full_name_prompt(existing_persona: ClinicianPersona) -> str:
    return f"""
Generate a realistic full name for a fictional clinician persona.

Persona characteristics:
- age: {existing_persona.age}
- sex: {existing_persona.sex}
- race_ethnicity: {existing_persona.race_ethnicity}
- place_of_birth: {existing_persona.place_of_birth}
- current_occupation_title: {existing_persona.current_occupation_title}
- organization_location: {existing_persona.current_organization_location}
- personality_traits: {existing_persona.personality_traits}

Requirements:
- Return only the full name
- The name must be plausible for the persona's background
- Do not include titles like Dr., RN, NP, PA-C, DDS
- Do not include explanation or extra text
- Output format: Firstname Lastname
""".strip()
