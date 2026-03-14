import sys
sys.path.insert(0, '..')

import json
import pandas as pd
from collections import Counter  
import matplotlib.pyplot as plt  

import os
os.chdir(os.path.dirname(__file__))

GROUPS = {"LGBTQ", "Region", "Sexism", "Racism", "others", "non-hate"}
HATEFUL_LABELS = {"hate", "non-hate"}
SEP_TOKEN = "[SEP]"
END_TOKEN = "[END]"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_quadruples(sample):
    quadruples = []
    i = 1
    while True:
        key_target = f"Q{i} Target"
        if key_target not in sample:
            break
        quadruples.append(
            {
                "target": sample[f"Q{i} Target"],
                "argument": sample[f"Q{i} Argument"],
                "group": sample[f"Q{i} Group"],
                "hateful": sample[f"Q{i} hateful"],
            }
        )
        i += 1
    return quadruples

def quadruples_to_str(quadruples):
    parts = [
        f"{q['target']}|{q['argument']}|{q['group']}|{q['hateful']}"
        for q in quadruples
    ]
    return f" {SEP_TOKEN} ".join(parts) + f" {END_TOKEN}"


def str_to_quadruples(text):
    text = text.strip()
    if END_TOKEN in text:
        text = text[: text.index(END_TOKEN)].strip()

    quadruples = []
    for part in text.split(SEP_TOKEN):
        part = part.strip()
        if not part:
            continue
        fields = [f.strip() for f in part.split("|")]
        if len(fields) == 4:
            quadruples.append(
                {
                    "target": fields[0],
                    "argument": fields[1],
                    "group": fields[2],
                    "hateful": fields[3],
                }
            )
    return quadruples


def prepare_samples(data):
    samples = []
    for record in data:
        quads = extract_quadruples(record)
        samples.append(
            {
                "id": record["id"],
                "content": record["content"],
                "topic": record.get("topic", ""),
                "sen_hate": int(record.get("sen_hate", 0)),
                "quadruples": quads,
                "output": quadruples_to_str(quads),
            }
        )
    return samples


def load_dataset(path):
    result = {}
    if path is not None:
        result = prepare_samples(load_json(path))
    return result



result = load_dataset("../data/train.json")
with open("../data/train_output.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
print("Dataset saved to train_output.json")
result = load_dataset("../data/test.json")
with open("../data/test_output.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
print("Dataset saved to test_output.json")