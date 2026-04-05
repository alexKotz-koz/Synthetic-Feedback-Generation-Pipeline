import random
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


def _find_race(seed: int) -> str:
    print("...Race")
    return RACE_ETHNICITY_CATEGORIES[seed % len(RACE_ETHNICITY_CATEGORIES)]


def _find_age(seed: int) -> int:
    print("...Age")
    age_categories = [(18, 29), (30, 49), (50, 64), (65, 75)]
    return random.randint(*age_categories[seed % len(age_categories)])


def _find_sex(seed: int) -> str:
    print("...Sex")
    return SEX_CATEGORIES[seed % len(SEX_CATEGORIES)]


def _find_support_system(n: int) -> list[str]:
    print("...Support System")
    selected_support = []
    counts: dict[str, int] = {}
    parent_total = 0
    grandparent_total = 0
    married_total = 0
    parents = {"mother", "father"}
    grandparents = {"grandmother", "grandfather"}

    for _ in range(n):
        allowed_choices = []
        for item in SUPPORT_SYSTEM_OPTIONS:
            if item in parents and parent_total >= 2:
                continue
            if item in grandparents and grandparent_total >= 4:
                continue
            allowed_choices.append(item)

        if not allowed_choices:
            break

        chosen = random.choice(allowed_choices)
        counts[chosen] = counts.get(chosen, 0) + 1
        selected_support.append(f"{chosen}_{counts[chosen]}")
        if chosen in parents:
            parent_total += 1
        elif chosen in grandparents:
            grandparent_total += 1

    return selected_support


def _find_current_occupation_title(age: int) -> str:
    print("...Occupation Title")
    if age < 18:
        raise ValueError("age must be at least 18")

    if age <= 23:
        occupations = [
            "Certified Nursing Assistant",
            "Medical Assistant",
            "Behavioral Health Counselor",
            "Dental Hygienist",
            "Registered Nurse",
        ]
    elif age <= 27:
        occupations = [
            "Certified Nursing Assistant",
            "Medical Assistant",
            "Behavioral Health Counselor",
            "Dental Hygienist",
            "Registered Nurse",
            "Dietitian",
            "Physician Assistant",
            "Nurse Practitioner",
        ]
    elif age <= 29:
        occupations = [
            "Certified Nursing Assistant",
            "Medical Assistant",
            "Behavioral Health Counselor",
            "Dental Hygienist",
            "Registered Nurse",
            "Dietitian",
            "Physician Assistant",
            "Nurse Practitioner",
            "Dentist",
            "Pharmacist",
            "Physician (M.D. or D.O.)",
        ]
    else:
        occupations = OCCUPATION_TITLES

    valid_occupations = [
        occupation
        for occupation in occupations
        if MIN_PRACTICE_AGE_BY_OCCUPATION[occupation] <= age
    ]

    if not valid_occupations:
        raise ValueError(f"No valid occupations available for age {age}")

    return random.choice(valid_occupations)


def _find_annual_income(
    current_occupation_title: str, income_seed: int | None = None
) -> int:
    print("...Income")
    income_category = INCOME_BY_OCCUPATION[current_occupation_title]
    generator = random.Random(income_seed) if income_seed is not None else random
    return generator.randint(*income_category["range"])


def _find_years_of_experience(age: int, current_occupation_title: str) -> int:
    print("...Years of Experience")
    if age < 18:
        raise ValueError("age must be at least 18")

    min_practice_age = MIN_PRACTICE_AGE_BY_OCCUPATION[current_occupation_title]
    max_possible_experience = max(0, age - min_practice_age)

    if max_possible_experience == 0:
        return 0

    if max_possible_experience <= 3:
        return random.randint(0, max_possible_experience)

    if max_possible_experience <= 8:
        experience_ranges = [
            (0, 2),
            (2, 4),
            (4, max_possible_experience),
        ]
    elif max_possible_experience <= 15:
        experience_ranges = [
            (0, 3),
            (3, 7),
            (7, max_possible_experience),
        ]
    else:
        experience_ranges = [
            (0, 5),
            (5, 10),
            (10, 15),
            (15, max_possible_experience),
        ]

    valid_ranges = [
        (low, high)
        for low, high in experience_ranges
        if low <= max_possible_experience and low <= high
    ]

    chosen_low, chosen_high = random.choice(valid_ranges)
    return random.randint(chosen_low, chosen_high)


def _find_clinical_priorities(n: int = 5) -> list[str]:
    print("...Clinicial Priorities")
    return random.sample(CLINICAL_PRIORITIES, k=min(n, len(CLINICAL_PRIORITIES)))


def _find_personality(occupation_title: str) -> str:
    return PERSONALITY_BY_OCCUPATION[occupation_title]


def _build_feedback_label_pairs(total_personas: int) -> list[tuple[str, str]]:
    if total_personas % 2 != 0:
        raise ValueError(
            "total_personas must be even to enforce exact 50/50 balance for both contextual relevance and feedback detail"
        )

    base_count = total_personas // 4
    remainder = total_personas % 4

    pair_counts: list[tuple[tuple[str, str], int]] = [
        (("contextually_relevant", "clear/detailed"), base_count),
        (("contextually_irrelevant", "clear/detailed"), base_count),
        (("contextually_relevant", "vague/ambiguous"), base_count),
        (("contextually_irrelevant", "vague/ambiguous"), base_count),
    ]

    if remainder == 2:
        pair_counts[0] = (pair_counts[0][0], pair_counts[0][1] + 1)
        pair_counts[3] = (pair_counts[3][0], pair_counts[3][1] + 1)

    labels: list[tuple[str, str]] = []
    for pair, count in pair_counts:
        labels.extend([pair] * count)

    random.shuffle(labels)
    return labels
