import json
import logging
from typing import Iterable, Dict, List

from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer
from allennlp.data.fields import MetadataField, TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer


logger = logging.getLogger(__name__)


@DatasetReader.register("modalitylite")
class ModalityDatasetReader(DatasetReader):
    def __init__(self,
                 wiki_tokenizer: Tokenizer = None,
                 sentence_tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__()
        self._wiki_tokenizer = wiki_tokenizer or WordTokenizer()
        self._sentence_tokenizer = sentence_tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterable[Instance]:
        logger.info("Reading Modal Sense instances from {}".format(file_path))
        with open(file_path,"r") as file:
            for line in file:
                json_line = json.loads(line)
                json_line.pop("modal_verb", None)
                yield self.sentence_to_instance(**json_line)

    def sentence_to_instance(self, sentence:str, label:str=None) -> Instance:
        sentence_tokens = self._sentence_tokenizer.tokenize(sentence)

        instance_meta = {"sentence_tokens": sentence_tokens}

        instance_dict = {"tokens": TextField(sentence_tokens, self._token_indexers)#,
                         #"metadata": MetadataField(instance_meta)
                         }

        if label is not None:
            instance_dict["label"] = LabelField(label)

        return Instance(instance_dict)

