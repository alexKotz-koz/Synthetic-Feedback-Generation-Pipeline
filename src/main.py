import os, json


from modules.persona_generation_1 import generate_personas


def main():

    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "data")
    personas_dir = os.path.join(data_dir, "personas")
    feedback_dir = os.path.join(data_dir, "feedback")

    personas = generate_personas(total_personas=2, model_str="gpt-4o")
    personas_file = os.path.join(personas_dir, "test_4o_personas.json")
    with open(personas_file, "w") as file:
        json.dump([p.model_dump(mode="json") for p in personas], file, indent=2)


if __name__ == "__main__":
    main()
