import json


base_dir = "/scratch/ace14856qn/openi/"


def read_raw_json(filename):
    with open(f"{base_dir}{filename}") as f:
        return [json.loads(val) for val in f.readlines()]


def preprocess_json(data):
    data = [{"impression": " ".join(val["impression"]), "findings": " ".join(val["findings"])} for val in data]
    return data


def write_json(dataset_name, data):
    with open(f"{base_dir}{dataset_name}_openi.json", "w") as f:
        f.writelines("\n".join([json.dumps(val) for val in data]))


for dataset in ["train", "valid", "test"]:
    write_json(dataset, preprocess_json(read_raw_json(f"{dataset}_openi_text.json")))
