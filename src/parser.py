# Parser loading data

import json
import numpy as np
from random import shuffle

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
            "n":0,
            "low":1,
            "high":2
        },
        "verdict":{
            "not guilty":0,
            "guilty":1
        }
    }
    yes_no = {"y": 1, "n": 0}
    return [{factor: normalizers.get(factor, yes_no).get(value, value)
             for (factor, value) in case.items()}
            for case in data]

def load_training():
    return to_matrix(normalize(load("../data/train.json")))

def load_test():
    return to_matrix(normalize(load("../data/test.json")))

def to_matrix(json_data):
    X = []
    y = []
    for case in json_data:
        # Map dict into list of tuples: (factor, value)
        items = case.items()
        # filter out verdict and caseId
        factors = filter(lambda factor:\
                         factor[0] != "verdict" and factor[0] != "caseId",
                         items)
        # insert verdict into y
        if not "verdict" in case:
            print("WARNING: verdict not found in case: {}. Aborting..."\
                  .format(case["caseId"]))
            exit(1)
        y.append([case["verdict"]])
        # Sort list by factor so that all Xs are consistent
        sorted_factors = sorted(factors, key=lambda item: item[0])
        # Map list into list of value, insert into X
        values = map(lambda item: item[1], sorted_factors)
        X.append(values)

    shuffle(X)
    shuffle(y)
    return (np.array(X), np.array(y))
