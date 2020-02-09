import jsonlines
from argparse import ArgumentParser
import os


def build_jsonl_by_datasource(datapath, dataset, datatype, balanced):
    balance = "" if balanced == ".txt" else balanced
    output = f"{datapath}/{dataset}_{datatypes[datatype]}_{balance}.jsonl"
    with jsonlines.open(output, "w") as outfile:
        for root, dirs, files in os.walk(f"{datapath}/{dataset}"):
            for file in files:
                if datatype in file:
                    write_dict_to_file(root, file, balanced, outfile)
    if dataset == "train":
        is_balanced = False if balanced == ".txt" else balanced
        create_validation_set(datapath, balanced=is_balanced, trainfile=output)

def write_dict_to_file(root, file, balanced, outfile):
    modal_verb = file.split("_")[0]
    if balanced in file:
        with open(os.path.join(root, file)) as f:
            try:
                for line in f.readlines():
                    line = line.split("\t")
                    label = line[1].strip()
                    if label not in ["ep", "de", "dy"]:
                        label = line[-1].split(",")[-1].strip()
                    line = {"sentence": line[0], "label": label, "modal_verb": modal_verb}
                    outfile.write(line)
            except UnicodeDecodeError:
                print("skipping file {}".format(file))


def build_jsonlines_file(datapath, dataset, balanced=".txt"):
    addition = "{}".format(balanced) if balanced != ".txt" else ""
    output = f"{datapath}/{dataset}{addition}.jsonl"
    with jsonlines.open(output, "w") as outfile:
        for root, dirs, files in os.walk(f"{datapath}/{dataset}"):
            for file in files:
                write_dict_to_file(root, file, balanced, outfile)
    if dataset == "train":
        is_balanced = False if balanced == ".txt" else balanced
        create_validation_set(datapath, output, balanced=is_balanced)


def create_validation_set(datapath, trainfile, balanced=None):
    addition = "{}".format(balanced) if balanced else ""
    # train = "{}/train{}.jsonl".format(datapath, addition)
    validation = trainfile.replace("train", "validation")
    # validation = "{}/validation{}.jsonl".format(datapath, addition)
    new_train = trainfile.replace("train", "dtrain")
    # new_train = "{}/dtrain{}.jsonl".format(datapath, addition)
    with jsonlines.open(trainfile, "r") as tr:
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

def single_dataset_to_jsonl(datapath, outfile):
    with jsonlines.open(os.path.join(datapath, outfile), "w") as jlf:
        for root, dirs, files in os.walk(datapath):
            for file in files:
                if file.endswith(".txt"):
                    modal_verb = file.split("__")[-1].split(".")[0]
                    with open(os.path.join(root, file), "r") as f:
                        for line in f.readlines():
                            try:
                                line = line.split("\t")
                                jlf.write({"sentence": line[0].strip(),
                                           "label": line[1].strip(),
                                           "modal_verb": modal_verb})
                            except:
                                print(line)

def from_pound_separated_csv_to_jsonl(datapath, infile, all_modal_targets, old_labels):
    dataset_size = 0
    if all_modal_targets:
        outfile = infile.replace(".txt", ".jsonl")
    else:
        outfile = infile.replace(".txt", "_only_modal_verbs.jsonl")
    with jsonlines.open(os.path.join(datapath, outfile), "w") as jlout:
        with open(os.path.join(datapath, infile), "r") as src:
            for line in src.readlines():
                tokens = [t.split("###") for t in line.split()]
                for t, label in tokens:
                    if label.startswith("S") or label.startswith("B"):
                        if "circumstantial" in label:
                            continue
                        else:
                            if all_modal_targets or t.lower() in ["can", "could", "may", "must", "shall", "should"]:
                                jlout.write({"sentence": " ".join([token[0] for token in tokens]),
                                             "label": old_labels[label.split("-")[1]], "modal_verb": t})
                                dataset_size += 1
    enum = 0
    train_size, val_size = int(dataset_size*0.8), int(dataset_size*0.9)
    train = os.path.join(datapath, "dtrain_" + outfile)
    validation = os.path.join(datapath, "validation_" + outfile)
    test = os.path.join(datapath, "test_" + outfile)
    with jsonlines.open(os.path.join(datapath, outfile), "r") as jlin:
        with jsonlines.open(train, "w") as jltrain:
            with jsonlines.open(validation, "w") as jlval:
                with jsonlines.open(test, "w") as jltest:
                    for line in jlin:
                        if enum < train_size:
                            jltrain.write(line)
                        elif enum < val_size:
                            jlval.write(line)
                        else:
                            jltest.write(line)
                        enum += 1
    print(dataset_size, train_size, val_size)
    
if __name__ == "__main__":
    arg_parser = ArgumentParser()

    arg_parser.add_argument("--datapath", default="../../data/EPOS_E")
    arg_parser.add_argument("--balanced", default=".txt")
    arg_parser.add_argument("--split_to_sources", default="False")
    arg_parser.add_argument("--infile", default="modal-BIOSE-coarse.txt")
    arg_parser.add_argument("--only_mv", default="False")

    "../../data/GME/coarse_only_modal_verbs.jsonl"
    args = arg_parser.parse_args()

    split_to_sources = False if args.split_to_sources in ["False", "0", "f", "F", "no", "No", "false"] else True
    all_modal_targets = True if args.only_mv in ["False", "0", "f", "F", "no", "No", "false"] else False
    # datapath, dataset, binarize = False, convert_to_int = False, balanced = ".txt", remove_duplicates = False
    datatypes = {"classifier1": "MPQA", "classifier2": "EPOS+MPQA", "classifier3": "EPOS"}

    if "EPOS_E" in args.datapath:
        if split_to_sources:
            for dtype in datatypes:
                build_jsonl_by_datasource(datapath=args.datapath, dataset="train", datatype=dtype, balanced=args.balanced)
                build_jsonl_by_datasource(datapath=args.datapath, dataset="test", datatype=dtype, balanced=args.balanced)
        else:
            build_jsonlines_file(datapath=args.datapath, dataset="train", balanced=args.balanced)
            build_jsonlines_file(datapath=args.datapath, dataset="test", balanced=args.balanced)

    elif "MASC" in args.datapath:
        single_dataset_to_jsonl(args.datapath, "tagged_masc.jsonl")


    elif "GME" in args.datapath:
        old_labels = {
                        'ability': "dy",
                        'ability_circumstantial': "dy",
                        'buletic': "de",
                        'buletic_teleological': "de",
                        'deontic': "de",
                        'epistemic': "ep",
                        'epistemic_circumstantial': "ep",
                        'priority': "de",
                        'teleological': "de",
                        'circumstantial': "circ"
                      }
        from_pound_separated_csv_to_jsonl(datapath=args.datapath, infile=args.infile,
                                          all_modal_targets=all_modal_targets, old_labels=old_labels)

