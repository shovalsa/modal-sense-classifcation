import jsonlines
import os
from random import shuffle

def build_jsonlines_file(datapath, dataset, binarize=False, convert_to_int=False):
    if binarize:
        output = f"{datapath}/{dataset}_binary.jsonl"
    elif convert_to_int:
        output = f"{datapath}/{dataset}_int.jsonl"
    else:
        output = f"{datapath}/{dataset}.jsonl"
    with jsonlines.open(output, "w") as output:
        for root, dirs, files in os.walk(f"{datapath}/{dataset}"):
            for file in files:
                modal_verb = file.split("_")[0]
                with open(os.path.join(root, file)) as f:
                    try:
                        for line in f.readlines():
                            line = line.split("\t")
                            label = line[1].strip()
                            if label not in ["ep", "de", "dy"]:
                                label = line[-1].split(",")[-1].strip()
                            if binarize:
                                label = "priority" if label == "de" else "non-priority"
                            elif convert_to_int:
                                label = "0" if label == "de" else "1" if label == "ep" else "2"
                            line = {"sentence": line[0], "label": label, "modal_verb": modal_verb}
                            output.write(line)
                    except UnicodeDecodeError:
                        print(file)


def create_validation_set(datapath, binary=False, convert_to_int=False):
    train = "{}/train_binary.jsonl".format(datapath) if binary else "{}/train_int.jsonl".format(datapath) if convert_to_int else "{}/train.jsonl".format(datapath)
    validation = "{}/validation_binary.jsonl".format(datapath) if binary else "{}/validation_int.jsonl".format(datapath) if convert_to_int else "{}/validation.jsonl".format(datapath)
    new_train = "{}/dtrain_binary.jsonl".format(datapath) if binary else "{}/dtrain_int.jsonl".format(datapath) if convert_to_int else "{}/dtrain.jsonl".format(datapath)
    with jsonlines.open(train, "r") as tr:
        with jsonlines.open(validation, mode="w") as trainf:
            with jsonlines.open(new_train, mode="w") as valf:
                enum = 0
                for line in tr:
                    if enum % 8 != 0:
                        valf.write(line)
                    else:
                        trainf.write(line)
                    enum += 1

def validate_output(fp):
    labels = set()
    with jsonlines.open(fp, "r") as f:
        for line in f:
            if line["label"] not in ["ep", "de", "dy"]:
                print(line)
        print(labels)

if __name__ == "__main__":
    build_jsonlines_file(datapath="../../data/EPOS_E", dataset="train", binarize=False, convert_to_int=False)
    build_jsonlines_file(datapath="../../data/EPOS_E", dataset="test", binarize=False, convert_to_int=False)
    create_validation_set("../../data/EPOS_E", binary=False, convert_to_int=False)
    # validate_output("../../data/EPOS_E/dtrain.jsonl")

