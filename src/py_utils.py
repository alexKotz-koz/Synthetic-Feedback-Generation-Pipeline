import os
import json
import pandas as pd

def dictKeys(dict: dict, filename: str, fileLocation=None):
    """
    Description: Prints the structure of a dictionary

    Args:
        <python dict>
        <filename.extension>
        <fileLocation> with trailing '/' (optional)
    """
    cwd = os.getcwd()
    if fileLocation:
        os.path.join(cwd, fileLocation)
    else:
        fileLocation = cwd
    with open(os.path.join(fileLocation, filename), "w") as file:
        json.dump(dict, file)

def import_data(filename):
    script_dir = os.getcwd()
    data_dir = os.path.join(script_dir, "data")
    file = os.path.join(data_dir, filename)
    df = pd.read_excel(file)
    return df

def pretty_print_json(obj):
    print(json.dumps(obj, indent=2), ensure_ascii=False)

def write_to_json(file, data):
    with open(file, "w") as f:
        json.dump(data, f)

