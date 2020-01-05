import jsonlines
import os
from argparse import ArgumentParser

def build_jsonlines_file(datapath, dataset, binarize=False, convert_to_int=False, balanced=".txt", remove_duplicates=False):
    addition = "_binary" if binarize else "_int" if convert_to_int else "_duplicates_removed" if remove_duplicates else \
        "{}".format(balanced) if balanced != ".txt" else ""
    output = f"{datapath}/{dataset}{addition}.jsonl"
    in_ds = set()
    with jsonlines.open(output, "w") as output:
        for root, dirs, files in os.walk(f"{datapath}/{dataset}"):
            for file in files:
                modal_verb = file.split("_")[0]
                if balanced in file:
                    with open(os.path.join(root, file)) as f:
                        try:
                            for line in f.readlines():
                                line = line.split("\t")
                                if remove_duplicates:
                                    if line[0].strip() in in_ds:
                                        continue
                                    else:
                                        in_ds.add(line[0].strip())
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
                            print("skipping file {}".format(file))
    if dataset == "train":
        is_balanced = False if balanced == ".txt" else balanced
        create_validation_set(datapath, binary=binarize, convert_to_int=convert_to_int, balanced=is_balanced, duplicates=remove_duplicates)


def create_validation_set(datapath, binary=False, convert_to_int=False, balanced=None, duplicates=False):
    addition = "_binary" if binary else "_int" if convert_to_int else "_duplicates_removed" if duplicates else "{}".format(balanced) if balanced else ""
    train = "{}/train{}.jsonl".format(datapath, addition)
    validation = "{}/validation{}.jsonl".format(datapath, addition)
    new_train = "{}/dtrain{}.jsonl".format(datapath, addition)
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
    arg_parser = ArgumentParser()

    arg_parser.add_argument("--datapath", default="../../data/EPOS_E")
    arg_parser.add_argument("--binarize", default="False")
    arg_parser.add_argument("--convert_to_int", default="False")
    arg_parser.add_argument("--balanced", default=".txt")
    arg_parser.add_argument("--remove_duplicates", default="False")

    args = arg_parser.parse_args()

    # datapath, dataset, binarize = False, convert_to_int = False, balanced = ".txt", remove_duplicates = False

    build_jsonlines_file(datapath=args.datapath, dataset="train", remove_duplicates=args.remove_duplicates,
                         binarize=args.binarize, convert_to_int=args.convert_to_int, balanced=args.balanced)
    build_jsonlines_file(datapath=args.datapath, dataset="test", remove_duplicates=args.remove_duplicates,
                         binarize=args.binarize, convert_to_int=args.convert_to_int, balanced=args.balanced)
    # create_validation_set("../../data/EPOS_E", binary=False, convert_to_int=False)
    validate_output("../../data/EPOS_E/dtrain_balance_.jsonl")

