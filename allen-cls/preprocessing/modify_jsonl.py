import jsonlines
from argparse import ArgumentParser
import os

def mask_modal_target(jsonl_input_path):
    with jsonlines.open(jsonl_input_path, "r") as jlin:
        with jsonlines.open(jsonl_input_path.replace(".jsonl", "_masked.jsonl"), "w") as jlout:
            for line in jlin:
                tokens = line["sentence"].split()
                masked_tokens= " ".join([t for t in tokens if t != line["modal_verb"]])
                jlout.write({"sentence": masked_tokens, "label": line["label"], "modal_verb": line["modal_verb"]})
                
                
if __name__ == "__main__":
    arg_parser = ArgumentParser()

    arg_parser.add_argument("--datapath", default="../../data", help="path to data directory")
    arg_parser.add_argument("--dataset", default="EPOS_E", help="which dataset to modify (EPOS_E, GME etc.)")
    arg_parser.add_argument("--file", default="dtrain_EPOS+MPQA_unbalance.jsonl", help="file to modify")
    arg_parser.add_argument("--modification", default="mask", help="")
    

    args = arg_parser.parse_args()
    
    filepath = os.path.join(args.datapath, args.dataset, args.file)
    if args.modification == "mask":
        mask_modal_target(filepath) 