import os
import json
import warnings
import argparse
from pprint import pprint


def read_jsonl_file(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            content = json.loads(line)
            if content.get("name", None) is not None:
                data.append(json.loads(line))
    return data


def check_filenames(file_path):
    data = read_jsonl_file(file_path)
    memo = {}
    for i in range(len(data)):
        name = data[i].get("name", None)
        memo[name] = memo.get(name, 0) + 1

    result = {}
    for key, value in memo.items():
        if value > 1:
            result[key] = value

    if len(result) > 0:
        for key, value in result.items():
            content = "filename: " + str(key) + " occurs: " + str(value) + " times"
            warnings.warn(content)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--file_path", type=str, required=True)

    args = argparser.parse_args()
    file_path = args.file_path
    check_filenames(file_path)
