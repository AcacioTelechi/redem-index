import json


def save_json(folder_name, file_name, obj):
    with open(folder_name + "/" + file_name + ".json", "w", encoding="utf8") as f:
        json.dump(obj, f)


def open_json(folder_name, file_name):
    with open(folder_name + "/" + file_name, "r", encoding="utf8") as f:
        return json.load(f)

def min_max_normalize(column):
    return (column - column.min()) / (column.max() - column.min())
