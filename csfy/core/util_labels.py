import os
from cornsnake import util_json

from csfy import config

def get_label_mapping(label_encoder):
    return dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

def _get_label_mapping_file_path(path_to_dir = None):
    path_to_dir = path_to_dir if path_to_dir else config.OUTPUT_DIR

    path_to_mapping = os.path.join(path_to_dir, "label_mapping.json")
    return path_to_mapping

def save_label_mapping(mapping):
    "Save to the current configured output location (to be with the output model)"
    path_to_mapping = _get_label_mapping_file_path()
    print(f"Saving label mapping to {path_to_mapping}")

    values = []
    for key in mapping:
        values.append(key)

    mapping_model = { "labels": values }
    util_json.write_to_json_file(mapping_model, path_to_mapping)

def load_label_mapping(path_to_dir):
    "Load from the same dir as the model (not necessarily the same as current output location)"
    path_to_mapping = _get_label_mapping_file_path(path_to_dir)
    print(f"Loading label mapping from {path_to_mapping}")
    mapping_model = util_json.read_from_json_file(path_to_mapping)
    return mapping_model['labels']
