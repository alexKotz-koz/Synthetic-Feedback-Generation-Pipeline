import os, json


from modules.persona_generation_1 import generate_personas
from modules.validation_agent_2 import validate_personas_file, validate_personas
from modules.feedback_generation_3 import (
    generate_feedback,
    _build_feedback_history_entry,
)


def load_personas(filename: str):
    with open(filename, "r") as f:
        personas = json.load(f)
    return personas


def main():

    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "data")
    personas_dir = os.path.join(data_dir, "personas")
    feedback_dir = os.path.join(data_dir, "feedback")
    model_str = "gpt-4o"
    ##### Generate Personas #####
    personas = generate_personas(total_personas=2, model_str=model_str)
    personas_file = os.path.join(personas_dir, "test_4o_personas.json")
    # updated_personas = validate_personas(personas=[p.model_dump(mode="json") for p in personas])
    with open(personas_file, "w") as file:
        json.dump([p.model_dump(mode="json") for p in personas], file, indent=2)

    ##### Validate Personas #####
    personas_file_name = "test_4o_personas.json"
    personas_file = os.path.join(personas_dir, personas_file_name)
    validate_personas_file(
        input_path=personas_file,
        output_path=os.path.join(personas_dir, f"validated_{personas_file_name}"),
    )
    validated_persona_file = os.path.join(
        personas_dir, f"validated_{personas_file_name}"
    )

    ##### Generate Feedback #####
    pi_wireframe_dir = os.path.join(cwd, "pi_wireframe")
    pi_clinician_view = os.path.join(pi_wireframe_dir, "PI_Clinician_View.png")
    pi_patient_view = os.path.join(pi_wireframe_dir, "PI_Patient_View.png")
    pi_find_form_view = os.path.join(pi_wireframe_dir, "PI_Find_Form_View.png")
    pi_complete_form_view = os.path.join(pi_wireframe_dir, "PI_Complete_Form_View.png")

    responses = {}

    personas = load_personas(filename=validated_persona_file)
    feedback_history = []
    for persona in personas:
        id = persona["id"]
        print(f"\n[main] Generating feedback for persona_id={id}")
        response = generate_feedback(
            persona=persona,
            model_str=model_str,
            screenshot_paths=[
                pi_clinician_view,
                pi_patient_view,
                pi_complete_form_view,
                pi_find_form_view,
            ],
            history=feedback_history,
            max_similarity=0.75,
        )
        feedback_history.append(
            _build_feedback_history_entry(answer=response["question_1_answer"])
        )
        responses[id] = response
        print(
            f"[main] Stored feedback for persona_id={id}; history_size={len(feedback_history)}"
        )
    with open(os.path.join(feedback_dir, "responses.json"), "w") as f:
        json.dump(responses, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
