import xml.etree.ElementTree as ET
import os
from pathlib import Path
import jsonlines


def annotate_sentence(sentence) -> list:
    modals = dict()
    modal_types = {"Epistemic": "ep", "Deontic": "de", "Dynamic": "dy",
                   "Optative": "op", "Concessive": "cs", "Conditional": "cn"}
    frames = sentence[2][1] if "frames" in [child.tag for child in sentence[2]] else None
    if frames:
        for frame in frames:
            modal_type = frame.attrib["name"]
            if modal_type in modal_types:
                fenode = frame[0][0]
                modal_verb_id = fenode.attrib["idref"]
                modals[modal_verb_id] = {"label": modal_types[modal_type]}
    terminals = sentence[0][0]
    text = ""
    for token in terminals.iter("t"):
        token_id = token.attrib["id"]
        word = token.attrib["word"]
        text += f"{word} "
        if token_id in modals:
            modals[token_id]["modal_verb"] = word

    instances = [{"sentence": text,
                  "label": labelXverb["label"],
                  "modal_verb": labelXverb["modal_verb"]} for modal_id, labelXverb in modals.items() if
                 "modal_verb" in labelXverb]
    return instances


def main(root, jlpath):
    with jsonlines.open(jlpath, "w") as f:
        body = root[1]
        for sentence in body.iter("s"):
            try:
                instances = annotate_sentence(sentence)
                for instance in instances:
                    f.write(instance)
            except:
                print(sentence.attrib["id"])

if __name__ == "__main__":
    path_to_mpqa = os.path.join(Path().resolve().parent.parent, "data", "R_R_MPQA")
    modalia = os.path.join(path_to_mpqa, "mpqa_modalia_release1.0.xml")
    modalia_tree = ET.parse(modalia)
    modalia_root = modalia_tree.getroot()
    jlpath = os.path.join(path_to_mpqa, "tagged_mpqa1.jsonl")
    main(modalia_root, jlpath)
