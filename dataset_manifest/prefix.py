import json
import argparse


class Prefix:
    def __init__(self, input_dir, output_dir, prefix) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.prefix = prefix

    def read_jsonl_file(self, file_path):
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def add_prefix(self, data, prefix):
        for i in range(len(data)):
            name = data[i].get("name", None)
            if name is not None:
                data[i]["name"] = str(prefix) + "/" + str(name)
        return data

    def write_jsonl_file(self, data, file_path):
        with open(file_path, "w") as f:
            for line in data:
                f.write(json.dumps(line) + "\n")

    def process(self):
        data = self.read_jsonl_file(self.input_dir)
        data = self.add_prefix(data, self.prefix)
        self.write_jsonl_file(data, self.output_dir)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", type=str, required=True)
    argparser.add_argument("--output_dir", type=str, required=True)
    argparser.add_argument("--prefix", type=str, required=True)
    args = argparser.parse_args()

    # input_dir = "./output/manifest.jsonl"
    # output_dir = "./output/NB_FullStack_0727_manifest.jsonl"
    # prefix = "NB_FullStack_0727"
    prefix = Prefix(args.input_dir, args.output_dir, args.prefix)
    prefix.process()
