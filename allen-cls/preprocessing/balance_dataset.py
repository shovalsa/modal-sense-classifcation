import jsonlines
import pandas as pd
from random import shuffle
import os


def jsonl_to_pd(path, sent_set):
    with jsonlines.open(path, "r") as trainfile:
        for line in trainfile:
            sent_set.add((line['sentence'], line['label'], line['modal_verb']))
    return sent_set

def add_sentences_to_balancing_list(mv, label, start, end):
    balancing_list = []
    while len(balancing_list) < (end-start):
        for i, row in tqdm(df_train[(df_train['label'] == label) & (df_train['modal_verb'] == mv)].iterrows()):
            if start < end:
                balancing_list.append({"sentence": row["sentence"], "label": row["label"], "modal_verb": row["modal_verb"]})
                start += 1
            else:
                break
    return balancing_list

def remove_items_from_df(mv, label, start, end, df_train):
    df = df_train[(df_train['label'] == label) & (df_train['modal_verb'] == mv)]
    for i, row in tqdm(df.iterrows()):
        if start > end:
            df_train = df_train.drop(i)
            start -= 1
        else:
            break
    return df_train


if __name__ == "__main__":
    arg_parser = ArgumentParser()

    arg_parser.add_argument("--datapath", default="../../data", help="path to data directory")
    arg_parser.add_argument("--dataset", default="EPOS_E", help="which dataset to modify (EPOS_E, GME etc.)")
    arg_parser.add_argument("--trainfile", default="train_EPOS+MPQA_balance.jsonl", help="file to modify")
    arg_parser.add_argument("--testfile", default="test_EPOS+MPQA_balance.jsonl", help="file to modify")
  

    args = arg_parser.parse_args()
    
    
    epos_sentences = set()
    epos_sentences = jsonl_to_pd(os.path.join(args.datapath, args.dataset, args.trainfile), epos_sentences)
    epos_sentences = jsonl_to_pd(os.path.join(args.datapath, args.dataset, args.testfile), epos_sentences)
    
    
    epos_sentences = list(epos_sentences)
    shuffle(epos_sentences)
    
    train_portion = int(len(epos_sentences)*0.9)
    
    df_train = pd.DataFrame(epos_sentences[:train_portion], columns=["sentence", "label", "modal_verb"])

    df_test = pd.DataFrame(epos_sentences[train_portion:], columns=["sentence", "label", "modal_verb"])
    
    total_balancing_list = []

    """ the following numbers are correct for the mv distribution as was captured in a specific train/test division.
        since the initial list is shuffled before division, these numbers might need modification. It is recommended to 
        print the distribution and call this method separately. 
        the original distribution:
            must    ep 586,  dy 0,    de 531
            may     ep 886,  dy 0,    de 127
            can     ep 16,   dy 351,  de 106
            should  ep 29,   dy 0,    de 345
            could   ep 160,  dy 88,   de 36
            shall   ep 0,    dy 5,    de 13
    """
    total_balancing_list += add_sentences_to_balancing_list("must", "de", 531, 586)
    total_balancing_list += add_sentences_to_balancing_list("may", "de", 127, 886)
    total_balancing_list += add_sentences_to_balancing_list("can", "ep", 16, 106)
    total_balancing_list += add_sentences_to_balancing_list("should", "ep", 29, 345)
    total_balancing_list += add_sentences_to_balancing_list("could", "dy", 88, 160)
    total_balancing_list += add_sentences_to_balancing_list("could", "de", 36, 160)
    total_balancing_list += add_sentences_to_balancing_list("shall", "dy", 5, 13)

    df_train = remove_items_from_df("can", "dy", 351, 106, df_train)
    
    b_df = pd.DataFrame(total_balancing_list)

    balanced_train = pd.concat([df_train, b_df])

    test_path = args.testfile.replace("balance", "re-balance")
    train_path = args.trainfile.replace("balance", "re-balance")
    dtrain_path = train_path.replace("train", "dtrain")
    val_path = train_path.replace("train", "validation")
    
    df_test.to_json(path_or_buf=os.path.join(args.datapath, args.dataset, test_path), orient='records', lines=True)
    balanced_train[385:].to_json(path_or_buf=os.path.join(args.datapath, args.dataset, dtrain_path), orient='records', lines=True)
    balanced_train[0:385].to_json(path_or_buf=val_path, orient='records', lines=True)
   

    
    filepath = os.path.join(args.datapath, args.dataset, args.file)
    if args.modification == "mask":
        mask_modal_target(filepath) 