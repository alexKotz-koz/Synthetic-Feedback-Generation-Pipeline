import os, json


from modules.persona_generation_1 import generate_personas
from modules.validation_agent_2 import validate_personas_file, validate_personas


def main():

    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "data")
    personas_dir = os.path.join(data_dir, "personas")
    feedback_dir = os.path.join(data_dir, "feedback")

    ##### Generate Personas #####
    # personas = generate_personas(total_personas=2, model_str="gpt-4o")
    # personas_file = os.path.join(personas_dir, "test_4o_personas.json")
    # with open(personas_file, "w") as file:
    #     json.dump([p.model_dump(mode="json") for p in personas], file, indent=2)

    ##### Validate Personas #####
    # personas_file_name = "test_4o_personas.json"
    # personas_file = os.path.join(personas_dir, personas_file_name)
    # validate_personas_file(
    #     input_path=personas_file,
    #     output_path=os.path.join(personas_dir, f"validated_{personas_file_name}"),
    # )


if __name__ == "__main__":
    main()
