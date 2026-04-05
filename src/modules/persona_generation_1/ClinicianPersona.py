from pydantic import BaseModel, UUID4
from .static import (
    RACE_ETHNICITY_CATEGORIES,
    SEX_CATEGORIES,
    SUPPORT_SYSTEM_OPTIONS,
    US_STATES,
    OCCUPATION_TITLES,
    MIN_PRACTICE_AGE_BY_OCCUPATION,
    INCOME_BY_OCCUPATION,
    CLINICAL_PRIORITIES,
    PERSONALITY_BY_OCCUPATION,
)


class ClinicianPersona(BaseModel):
    id: UUID4
    ### Sociodemographics
    full_name: str
    age: int  # add validation for 18-75
    sex: str  # add validation for m/f
    race_ethnicity: str  # add validation for NIH categories
    place_of_birth: str
    support_system: list[str]
    lifestyle_narrative: str
    hobbies_narrative: str
    annual_income: int

    ### Technology Literacy
    technological_skill_level: str

    ### Occupation
    current_occupation_title: str
    years_of_experience: int
    current_organization_name: str
    current_organization_location: str

    ### Clinical
    clinical_priorities: list[str]
    personality_traits: str

    ### SDG
    va_modulator = str
    cr_modulator = str
