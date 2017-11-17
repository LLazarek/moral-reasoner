# Parser loading data

import json

def load(path):
    with open(path, 'r') as f:
        return json.loads(f.read())

def normalize(data):
    normalizers = {
        "mental_state":{
            "neither": 0,
            "negligent": 1,
            "reckless":2,
            "intend":3
        },
        "foreseeability":{
            "low":0,
            "high":1
        }
    }
    yes_no = {"y": 1, "n": 0}
    return [{factor: normalizers.get(factor, yes_no).get(value, value)
             for (factor, value) in case.items()}
            for case in data]

def load_training():
    return normalize(load("../data/train.json"))

def load_test():
    return normalize(load("../data/test.json"))
